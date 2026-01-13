// TODO: Review allow attributes at a later point
#![allow(clippy::useless_conversion, clippy::too_many_arguments)]

use crate::processing::ProcessedStructure;
use crate::spec::{CoordFormat, OutputSpec};
use crate::{forcefield, formats, formatters, geometry, physics, processing, spec};
use numpy::PyArray1;
use numpy::PyArrayMethods;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Parse a PDB file and return raw atom data (low-level)
/// Returns first model only for backward compatibility.
#[pyfunction]
pub fn parse_pdb(path: String) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let (raw_data, model_ids) = formats::pdb::parse_pdb_file(&path).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("PDB parsing failed: {}", e))
        })?;

        // Filter to first model only (legacy behavior)
        let first_model = model_ids.first().copied().unwrap_or(1);
        let filtered = processing::filter_models(&raw_data, &model_ids, &[first_model]);

        filtered.to_py_dict(py).map(|dict| dict.into_py(py))
    })
}

/// Parse an mmCIF file and return raw atom data (low-level)
#[pyfunction]
pub fn parse_mmcif(path: String) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let (raw_data, _model_ids) = formats::mmcif::parse_mmcif_file(&path).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("mmCIF parsing failed: {}", e))
        })?;

        raw_data.to_py_dict(py).map(|dict| dict.into_py(py))
    })
}

/// Parse a PQR file and return raw atom data with charges and radii
#[pyfunction]
pub fn parse_pqr(path: String) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let raw_data = formats::pqr::parse_pqr_file(&path).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("PQR parsing failed: {}", e))
        })?;

        raw_data.to_py_dict(py).map(|dict| dict.into_py(py))
    })
}

/// Parse a FoldComp file and return AtomicSystem
#[pyfunction]
pub fn parse_foldcomp(path: String) -> Result<crate::structure::systems::AtomicSystem, PyErr> {
    formats::foldcomp::read_foldcomp(&path).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("FoldComp parsing failed: {}", e))
    })
}

/// Parse a structure file (PDB/mmCIF) into a format suitable for the Protein class.
#[pyfunction]
#[pyo3(signature = (path, spec=None))]
pub fn parse_structure(path: String, spec: Option<OutputSpec>) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let spec = spec.unwrap_or_default();
        let target = spec::OutputFormatTarget::from_str(&spec.output_format_target)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;

        // Check cache (only if features are simple)
        let should_cache = spec.enable_caching
            && !spec.compute_rbf
            && !spec.compute_electrostatics
            && !spec.compute_vdw
            && !spec.parameterize_md
            && !spec.infer_bonds
            && !spec.add_hydrogens;

        if should_cache {
            let key = formatters::CacheKey::new(
                &path,
                spec.coord_format,
                spec.remove_solvent,
                spec.include_hetatm,
            );
            if let Some(cached) = formatters::get_cached(&key) {
                log::debug!("Cache hit for {}", path);
                return cached.to_py_dict(py);
            }
        }

        // 1. Parse PDB or mmCIF
        let (raw_data_all, model_ids) = if path.ends_with(".cif") || path.ends_with(".mmcif") {
            formats::mmcif::parse_mmcif_file(&path).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("mmCIF parsing failed: {}", e))
            })?
        } else {
            formats::pdb::parse_pdb_file(&path).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("PDB parsing failed: {}", e))
            })?
        };

        // 2. Identify and Filter Models
        let split_data = processing::models::split_by_model(&raw_data_all, &model_ids);

        let mut unique_models = Vec::new();
        for &m in &model_ids {
            if !unique_models.contains(&m) {
                unique_models.push(m);
            }
        }

        let mut models_to_process = Vec::new();
        if let Some(ref req) = spec.models {
            for (i, &m_id) in unique_models.iter().enumerate() {
                if req.contains(&m_id) && i < split_data.len() {
                    models_to_process.push(&split_data[i]);
                }
            }
        } else {
            // Default: ALL models
            for data in &split_data {
                models_to_process.push(data);
            }
        }

        if models_to_process.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "No models found matching request",
            ));
        }

        // 3. Process Reference Model (First)
        let ref_raw = models_to_process[0];
        let mut processed = ProcessedStructure::from_raw(ref_raw.clone()).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Structure processing failed: {}", e))
        })?;

        // 4. Add Hydrogens / Infer Bonds on Reference
        if spec.add_hydrogens {
            log::debug!("Adding hydrogens to reference model...");
            let all_coords = processed.extract_all_coords();
            let all_elements = &processed.raw_atoms.elements;

            // Infer bonds if not provided
            let mut bonds_for_h = geometry::topology::infer_bonds(&all_coords, all_elements, 1.3);

            eprintln!(
                "[OXIDIZE DEBUG] Calling add_hydrogens_with_relax. relax={}, max_iter={:?}",
                spec.relax_hydrogens, spec.relax_max_iterations
            );
            let num_h_added = geometry::hydrogens::add_hydrogens_with_relax(
                &mut processed,
                &mut bonds_for_h,
                spec.relax_hydrogens,
                spec.relax_max_iterations,
            )
            .map(|(n, _, _)| n)
            .map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("Hydrogen addition failed: {}", e))
            })?;

            log::debug!("Added {} hydrogens to reference model", num_h_added);

            // Note: ProcessedStructure was updated in-place, no need to reassign
            let _ = processed; // Keep processed in scope (no-op, updated in place)
        } else if spec.infer_bonds {
            // Bonds will be inferred later in section "Topology" if needed
        }

        // Store parameterized structure for later output
        let mut md_params = None;

        // --- MD Parameterization (Moved Up) ---
        // Parameterize *before* formatting/features so we can use the parameters
        if spec.parameterize_md {
            if let Some(ref ff_path) = spec.force_field {
                log::debug!("Parameterizing structure with {}", ff_path);

                let ff = forcefield::parse_forcefield_xml(ff_path).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "Force field parsing failed: {}",
                        e
                    ))
                })?;

                let param_options = physics::md_params::ParamOptions {
                    auto_terminal_caps: spec.auto_terminal_caps,
                    missing_mode: match spec.missing_residue_mode {
                        spec::MissingResidueMode::SkipWarn => {
                            physics::md_params::MissingResidueMode::SkipWarn
                        }
                        spec::MissingResidueMode::Fail => {
                            physics::md_params::MissingResidueMode::Fail
                        }
                        spec::MissingResidueMode::GaffFallback => {
                            physics::md_params::MissingResidueMode::GaffFallback
                        }
                        spec::MissingResidueMode::ClosestMatch => {
                            physics::md_params::MissingResidueMode::ClosestMatch
                        }
                    },
                };

                let params =
                    physics::md_params::parameterize_structure(&processed, &ff, &param_options)
                        .map_err(|e| {
                            pyo3::exceptions::PyValueError::new_err(format!(
                                "Parameterization failed: {}",
                                e
                            ))
                        })?;

                processed.raw_atoms.charges = Some(params.charges.clone());
                processed.raw_atoms.sigmas = Some(params.sigmas.clone());
                processed.raw_atoms.epsilons = Some(params.epsilons.clone());

                log::info!(
                    "Parameterized {}/{} atoms",
                    params.num_parameterized,
                    processed.raw_atoms.num_atoms
                );

                md_params = Some(params);
            } else {
                log::warn!("parameterize_md=true but no force_field path provided");
            }
        }

        // 5. Format and Stack
        let (dict, cached_structure) = match spec.coord_format {
            CoordFormat::Atom37 => {
                let _formatter = formatters::Atom37Formatter;
                let ref_formatted = formatters::Atom37Formatter::format(&processed, &spec)
                    .map_err(|e| {
                        pyo3::exceptions::PyValueError::new_err(format!("Formatting failed: {}", e))
                    })?;

                let dict = ref_formatted.to_py_dict(py)?;

                // --- Multi-Model Stacking for Atom37 ---
                if models_to_process.len() > 1 {
                    let n_res = ref_formatted.coordinates.len() / (37 * 3);
                    let n_models = models_to_process.len();

                    let mut all_coords = Vec::with_capacity(n_models * n_res * 37 * 3);
                    all_coords.extend_from_slice(&ref_formatted.coordinates);

                    for (i, m_raw) in models_to_process.iter().enumerate().skip(1) {
                        // We must process each model to map atoms correctly
                        let mut m_processed = ProcessedStructure::from_raw((*m_raw).clone())
                            .map_err(|e| {
                                pyo3::exceptions::PyValueError::new_err(format!(
                                    "Processing model {} failed: {}",
                                    i + 1,
                                    e
                                ))
                            })?;

                        if spec.add_hydrogens {
                            // Re-infer for consistent H addition
                            let all_coords = m_processed.extract_all_coords();
                            let all_elements = &m_processed.raw_atoms.elements;
                            let mut m_bonds =
                                geometry::topology::infer_bonds(&all_coords, all_elements, 1.3);
                            geometry::hydrogens::add_hydrogens_with_relax(
                                &mut m_processed,
                                &mut m_bonds,
                                spec.relax_hydrogens,
                                spec.relax_max_iterations,
                            )
                            .map_err(|e| {
                                pyo3::exceptions::PyValueError::new_err(format!(
                                    "H-add model {}: {}",
                                    i + 1,
                                    e
                                ))
                            })?;
                            // m_processed is updated in-place, no reassignment needed
                        }

                        let m_formatted = formatters::Atom37Formatter::format(&m_processed, &spec)
                            .map_err(|e| {
                                pyo3::exceptions::PyValueError::new_err(format!(
                                    "Format model {}: {}",
                                    i + 1,
                                    e
                                ))
                            })?;

                        // Strict check on size
                        if m_formatted.coordinates.len() != ref_formatted.coordinates.len() {
                            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                                "Model {} size mismatch ({} vs {})",
                                i + 1,
                                m_formatted.coordinates.len(),
                                ref_formatted.coordinates.len()
                            )));
                        }
                        all_coords.extend_from_slice(&m_formatted.coordinates);
                    }

                    // Reshape to (N_models, N_res, 37, 3)
                    let flat_array = PyArray1::from_slice_bound(py, &all_coords);
                    let shaped = flat_array.reshape((n_models, n_res, 37, 3)).map_err(|e| {
                        pyo3::exceptions::PyValueError::new_err(format!("Reshape failed: {}", e))
                    })?;
                    let dict_bound = dict.downcast_bound::<PyDict>(py).unwrap();
                    dict_bound.set_item("coordinates", shaped)?;
                }

                let cached = if should_cache {
                    Some(formatters::CachedStructure {
                        coordinates: ref_formatted.coordinates.clone(),
                        atom_mask: ref_formatted.atom_mask.clone(),
                        aatype: ref_formatted.aatype.clone(),
                        residue_index: ref_formatted.residue_index.clone(),
                        chain_index: ref_formatted.chain_index.clone(),
                        _num_residues: ref_formatted.aatype.len(),
                        atom_names: None,
                        coord_shape: None,
                    })
                } else {
                    None
                };

                (dict, cached)
            }
            CoordFormat::Atom14 => {
                let formatted =
                    formatters::Atom14Formatter::format(&processed, &spec).map_err(|e| {
                        pyo3::exceptions::PyValueError::new_err(format!("Formatting failed: {}", e))
                    })?;
                let dict = formatted.to_py_dict(py)?;
                // No multi-model stacking implemented for Atom14 yet
                (dict, None)
            }
            CoordFormat::BackboneOnly => {
                let formatted =
                    formatters::BackboneFormatter::format(&processed, &spec).map_err(|e| {
                        pyo3::exceptions::PyValueError::new_err(format!("Formatting failed: {}", e))
                    })?;
                let dict = formatted.to_py_dict(py)?;
                (dict, None)
            }
            CoordFormat::Full => {
                let formatted =
                    formatters::FullFormatter::format(&processed, &spec).map_err(|e| {
                        pyo3::exceptions::PyValueError::new_err(format!("Formatting failed: {}", e))
                    })?;

                let cached = if should_cache {
                    Some(formatters::CachedStructure {
                        coordinates: formatted.coordinates.clone(),
                        atom_mask: formatted.atom_mask.clone(),
                        aatype: formatted.aatype.clone(),
                        residue_index: formatted.residue_index.clone(),
                        chain_index: formatted.chain_index.clone(),
                        _num_residues: formatted.aatype.len(),
                        atom_names: Some(formatted.atom_names.clone()),
                        coord_shape: Some(formatted.coord_shape),
                    })
                } else {
                    None
                };

                (formatted.to_py_dict(py)?, cached)
            }
        };

        // Insert into cache if needed
        if let Some(cached) = cached_structure {
            let key = formatters::CacheKey::new(
                &path,
                spec.coord_format,
                spec.remove_solvent,
                spec.include_hetatm,
            );
            formatters::insert_cached(key, cached);
        }

        // Downcast to PyDict to add more fields
        let dict_bound = dict.downcast_bound::<PyDict>(py).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Failed to downcast to dict: {}", e))
        })?;

        // --- Chain Information ---
        // Expose unique chain IDs corresponding to chain_index
        // chain_index (from formatters) maps to index in this list
        // Note: processed.raw_atoms.chain_ids contains per-atom chain IDs.
        // We need to extract unique ones preserving order of appearance/index.

        let mut unique_chains: Vec<String> = Vec::new();
        let mut seen_chains = std::collections::HashSet::new();

        // We iterate over atoms to find unique chains in order
        for cid in &processed.raw_atoms.chain_ids {
            if !seen_chains.contains(cid) {
                seen_chains.insert(cid.clone());
                unique_chains.push(cid.clone());
            }
        }
        // However, if we filtered models/atoms, we should check `formatted.chain_index` (if accessible)
        // But `formatters` don't return chain_index map easily, they just output indices.
        // The indices 0..K usually correspond to the unique chains found in the structure.
        // Let's assume the formatter follows standard unique-order.

        // But wait, Atom37 formatter groups by residue.
        // `processed` structure has raw atoms.
        // If we grouped by residue, the chain index is per residue.
        // The mapping 0->ChainA, 1->ChainB depends on how formatter assigned indices.
        // Atom37Formatter usually assigns based on appearance.
        // So `unique_chains` calculated above (in order of appearance) should match.

        let unique_chains_list: Vec<&str> = unique_chains.iter().map(|s| s.as_str()).collect();
        dict_bound.set_item("unique_chain_ids", unique_chains_list)?;

        // Also provide per-atom chain_ids (list of str) for completeness if needed?
        // Or "chain_ids" key usually means unique or per-atom?
        // In RawAtomData it's per-atom.
        // In Atom37 output, we have chain_index (int) per residue.
        // Providing `unique_chain_ids` allows reconstruction.
        dict_bound.set_item("chain_ids", processed.raw_atoms.chain_ids.clone())?; // List[str] per atom

        // --- Geometry Features ---
        if spec.compute_rbf {
            // Extract coordinates
            let ca_coords = processed.extract_ca_coords();
            let backbone_coords = processed.extract_backbone_coords(target);

            // 1. Find neighbors
            let neighbor_indices =
                geometry::neighbors::find_k_nearest_neighbors(&ca_coords, spec.rbf_num_neighbors);

            // Convert neighbors to (N, K) array
            let n_res = neighbor_indices.len();
            let k_neighbors = spec.rbf_num_neighbors;
            let mut flat_neighbors = vec![-1i32; n_res * k_neighbors];

            for (i, neighbors) in neighbor_indices.iter().enumerate() {
                for (j, &neighbor_idx) in neighbors.iter().enumerate() {
                    if j < k_neighbors {
                        flat_neighbors[i * k_neighbors + j] = neighbor_idx as i32;
                    }
                }
            }

            let neighbors_array = PyArray1::from_slice_bound(py, &flat_neighbors);
            let neighbors_reshaped =
                neighbors_array.reshape((n_res, k_neighbors)).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "Failed to reshape neighbor indices: {}",
                        e
                    ))
                })?;
            dict_bound.set_item("neighbor_indices", neighbors_reshaped)?;

            // 2. Compute RBF
            let rbf_result = geometry::radial_basis::compute_radial_basis_with_shape(
                &backbone_coords,
                &neighbor_indices,
            );

            // 3. Convert to NumPy (N, K, 400)
            let shape = (rbf_result.shape.0, rbf_result.shape.1, rbf_result.shape.2);
            let flat_array = PyArray1::from_slice_bound(py, &rbf_result.features);
            let rbf_array = flat_array.reshape(shape).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Failed to reshape RBF array: {}",
                    e
                ))
            })?;

            dict_bound.set_item("rbf_features", rbf_array)?;
        }

        // --- Topology (Bonds, Angles, Dihedrals, Impropers) ---
        if spec.infer_bonds {
            let all_coords = processed.extract_all_coords();
            let all_elements = &processed.raw_atoms.elements;

            // Generate full topology with default tolerance 1.3
            let topology =
                forcefield::topology::Topology::from_coords(&all_coords, all_elements, 1.3);

            // Bonds (N_bonds, 2)
            if !topology.bonds.is_empty() {
                let mut bonds_flat = Vec::with_capacity(topology.bonds.len() * 2);
                for bond in &topology.bonds {
                    bonds_flat.push(bond.i);
                    bonds_flat.push(bond.j);
                }
                let bonds_array = PyArray1::from_slice_bound(py, &bonds_flat);
                let bonds_reshaped =
                    bonds_array
                        .reshape((topology.bonds.len(), 2))
                        .map_err(|e| {
                            pyo3::exceptions::PyValueError::new_err(format!(
                                "Failed to reshape bonds: {}",
                                e
                            ))
                        })?;
                dict_bound.set_item("bonds", bonds_reshaped)?;
            } else {
                let empty: &[usize] = &[];
                let arr = PyArray1::from_slice_bound(py, empty);
                dict_bound.set_item("bonds", arr.reshape((0, 2)).unwrap())?;
            }

            // Angles (N_angles, 3)
            if !topology.angles.is_empty() {
                let mut angles_flat = Vec::with_capacity(topology.angles.len() * 3);
                for angle in &topology.angles {
                    angles_flat.push(angle.i);
                    angles_flat.push(angle.j);
                    angles_flat.push(angle.k);
                }
                let angles_array = PyArray1::from_slice_bound(py, &angles_flat);
                let angles_reshaped =
                    angles_array
                        .reshape((topology.angles.len(), 3))
                        .map_err(|e| {
                            pyo3::exceptions::PyValueError::new_err(format!(
                                "Failed to reshape angles: {}",
                                e
                            ))
                        })?;
                dict_bound.set_item("angles", angles_reshaped)?;
            } else {
                let empty: &[usize] = &[];
                let arr = PyArray1::from_slice_bound(py, empty);
                dict_bound.set_item("angles", arr.reshape((0, 3)).unwrap())?;
            }

            // Proper Dihedrals (N_dihedrals, 4)
            if !topology.proper_dihedrals.is_empty() {
                let mut dihedrals_flat = Vec::with_capacity(topology.proper_dihedrals.len() * 4);
                for dih in &topology.proper_dihedrals {
                    dihedrals_flat.push(dih.i);
                    dihedrals_flat.push(dih.j);
                    dihedrals_flat.push(dih.k);
                    dihedrals_flat.push(dih.l);
                }
                let dih_array = PyArray1::from_slice_bound(py, &dihedrals_flat);
                let dih_reshaped = dih_array
                    .reshape((topology.proper_dihedrals.len(), 4))
                    .map_err(|e| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "Failed to reshape dihedrals: {}",
                            e
                        ))
                    })?;
                dict_bound.set_item("dihedrals", dih_reshaped)?;
            } else {
                let empty: &[usize] = &[];
                let arr = PyArray1::from_slice_bound(py, empty);
                dict_bound.set_item("dihedrals", arr.reshape((0, 4)).unwrap())?;
            }

            // Improper Dihedrals (N_impropers, 4)
            if !topology.improper_dihedrals.is_empty() {
                let mut impropers_flat = Vec::with_capacity(topology.improper_dihedrals.len() * 4);
                for imp in &topology.improper_dihedrals {
                    impropers_flat.push(imp.i);
                    impropers_flat.push(imp.j);
                    impropers_flat.push(imp.k);
                    impropers_flat.push(imp.l);
                }
                let imp_array = PyArray1::from_slice_bound(py, &impropers_flat);
                let imp_reshaped = imp_array
                    .reshape((topology.improper_dihedrals.len(), 4))
                    .map_err(|e| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "Failed to reshape impropers: {}",
                            e
                        ))
                    })?;
                dict_bound.set_item("impropers", imp_reshaped)?;
            } else {
                let empty: &[usize] = &[];
                let arr = PyArray1::from_slice_bound(py, empty);
                dict_bound.set_item("impropers", arr.reshape((0, 4)).unwrap())?;
            }
        }

        // --- Molecule Type ---
        // 0=Protein, 1=Ligand, 2=Solvent, 3=Ion
        let mol_type_array = PyArray1::from_slice_bound(py, &processed.molecule_type);
        dict_bound.set_item("molecule_type", mol_type_array)?;
        if spec.compute_electrostatics {
            if let Some(ref charges) = processed.raw_atoms.charges {
                // Ensure charges match atoms
                if charges.len() == processed.raw_atoms.num_atoms {
                    let mut backbone_coords = processed.extract_backbone_coords(target);
                    let all_coords = processed.extract_all_coords();
                    let backbone_charges = processed.extract_backbone_charges(target);

                    // Infer missing CB positions (e.g. for Glycine or snippets)
                    for res_bb in backbone_coords.iter_mut() {
                        if res_bb[3][0].is_nan() {
                            // 3 is CB_INDEX
                            res_bb[3] =
                                physics::frame::compute_c_beta(res_bb[0], res_bb[1], res_bb[2]);
                        }
                    }

                    // Compute 3D forces at backbone atoms
                    let forces = physics::electrostatics::compute_coulomb_forces_at_backbone(
                        &backbone_coords,
                        &all_coords,
                        &backbone_charges,
                        charges,
                    );

                    // Project forces onto local backbone frames
                    // Result is flat vector (N_res * 5)
                    let features =
                        physics::frame::project_backbone_forces(&forces, &backbone_coords);

                    // Shape (N_res, 5)
                    let shape = (backbone_coords.len(), 5);
                    let flat_array = PyArray1::from_slice_bound(py, &features);
                    let array = flat_array.reshape(shape).map_err(|e| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "Failed to reshape electrostatics array: {}",
                            e
                        ))
                    })?;

                    dict_bound.set_item("electrostatic_features", array)?;
                } else {
                    log::warn!("Charge count mismatch, skipping electrostatics");
                }
            } else {
                log::warn!("No charges found in structure, skipping electrostatics");
            }
        }

        if spec.compute_vdw {
            let mut backbone_coords = processed.extract_backbone_coords(target);
            let all_coords = processed.extract_all_coords();

            // Infer missing CB positions
            for res_bb in backbone_coords.iter_mut() {
                if res_bb[3][0].is_nan() {
                    // 3 is CB_INDEX
                    res_bb[3] = physics::frame::compute_c_beta(res_bb[0], res_bb[1], res_bb[2]);
                }
            }

            // Get LJ parameters - either from structure (parameterize_md) or defaults
            let (all_sigmas, all_epsilons): (Vec<f32>, Vec<f32>) =
                if let (Some(ref sigmas), Some(ref epsilons)) =
                    (&processed.raw_atoms.sigmas, &processed.raw_atoms.epsilons)
                {
                    (sigmas.clone(), epsilons.clone())
                } else {
                    // Use default values for all atoms
                    let n_atoms = all_coords.len();
                    let default_sigmas = vec![physics::constants::DEFAULT_SIGMA; n_atoms];
                    let default_epsilons = vec![physics::constants::DEFAULT_EPSILON; n_atoms];
                    (default_sigmas, default_epsilons)
                };

            let backbone_sigmas = processed.extract_backbone_sigmas(target);
            let backbone_epsilons = processed.extract_backbone_epsilons(target);

            // Compute 3D forces
            let forces = physics::vdw::compute_lj_forces_at_backbone(
                &backbone_coords,
                &all_coords,
                &backbone_sigmas,
                &backbone_epsilons,
                &all_sigmas,
                &all_epsilons,
            );

            // Project forces onto local backbone frames
            let features = physics::frame::project_backbone_forces(&forces, &backbone_coords);

            // Shape (N_res, 5)
            let shape = (backbone_coords.len(), 5);
            let flat_array = PyArray1::from_slice_bound(py, &features);
            let array = flat_array.reshape(shape).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Failed to reshape VdW array: {}",
                    e
                ))
            })?;

            dict_bound.set_item("vdw_features", array)?;
        }

        // --- Output MD Parameters ---
        if let Some(params) = md_params {
            dict_bound.set_item("charges", PyArray1::from_slice_bound(py, &params.charges))?;
            dict_bound.set_item("sigmas", PyArray1::from_slice_bound(py, &params.sigmas))?;
            dict_bound.set_item("epsilons", PyArray1::from_slice_bound(py, &params.epsilons))?;

            if let Some(ref radii) = params.radii {
                dict_bound.set_item(
                    "gbsa_radii",
                    PyArray1::from_slice_bound(py, radii.as_slice()),
                )?;
            }
            if let Some(ref scales) = params.scales {
                dict_bound.set_item(
                    "gbsa_scales",
                    PyArray1::from_slice_bound(py, scales.as_slice()),
                )?;
            }

            if !params.bonds.is_empty() {
                let mut flat = Vec::with_capacity(params.bonds.len() * 2);
                for b in &params.bonds {
                    flat.extend_from_slice(b);
                }
                let arr = PyArray1::from_slice_bound(py, &flat);
                dict_bound.set_item("bonds", arr.reshape((params.bonds.len(), 2)).unwrap())?;
            }
            if !params.bond_params.is_empty() {
                let mut flat = Vec::with_capacity(params.bond_params.len() * 2);
                for p in &params.bond_params {
                    flat.extend_from_slice(p);
                }
                let arr = PyArray1::from_slice_bound(py, &flat);
                dict_bound.set_item(
                    "bond_params",
                    arr.reshape((params.bond_params.len(), 2)).unwrap(),
                )?;
            }
            if !params.angles.is_empty() {
                let mut flat = Vec::with_capacity(params.angles.len() * 3);
                for x in &params.angles {
                    flat.extend_from_slice(x);
                }
                let arr = PyArray1::from_slice_bound(py, &flat);
                dict_bound.set_item("angles", arr.reshape((params.angles.len(), 3)).unwrap())?;
            }
            if !params.angle_params.is_empty() {
                let mut flat = Vec::with_capacity(params.angle_params.len() * 2);
                for p in &params.angle_params {
                    flat.extend_from_slice(p);
                }
                let arr = PyArray1::from_slice_bound(py, &flat);
                dict_bound.set_item(
                    "angle_params",
                    arr.reshape((params.angle_params.len(), 2)).unwrap(),
                )?;
            }
            if !params.dihedrals.is_empty() {
                let mut flat = Vec::with_capacity(params.dihedrals.len() * 4);
                for x in &params.dihedrals {
                    flat.extend_from_slice(x);
                }
                let arr = PyArray1::from_slice_bound(py, &flat);
                dict_bound.set_item(
                    "dihedrals",
                    arr.reshape((params.dihedrals.len(), 4)).unwrap(),
                )?;
            }
            if !params.dihedral_params.is_empty() {
                let mut flat = Vec::with_capacity(params.dihedral_params.len() * 3);
                for p in &params.dihedral_params {
                    flat.extend_from_slice(p);
                }
                let arr = PyArray1::from_slice_bound(py, &flat);
                dict_bound.set_item(
                    "dihedral_params",
                    arr.reshape((params.dihedral_params.len(), 3)).unwrap(),
                )?;
            }
            if !params.impropers.is_empty() {
                let mut flat = Vec::with_capacity(params.impropers.len() * 4);
                for x in &params.impropers {
                    flat.extend_from_slice(x);
                }
                let arr = PyArray1::from_slice_bound(py, &flat);
                dict_bound.set_item(
                    "impropers",
                    arr.reshape((params.impropers.len(), 4)).unwrap(),
                )?;
            }
            if !params.improper_params.is_empty() {
                let mut flat = Vec::with_capacity(params.improper_params.len() * 3);
                for p in &params.improper_params {
                    flat.extend_from_slice(p);
                }
                let arr = PyArray1::from_slice_bound(py, &flat);
                dict_bound.set_item(
                    "improper_params",
                    arr.reshape((params.improper_params.len(), 3)).unwrap(),
                )?;
            }

            if !params.pairs_14.is_empty() {
                let mut flat = Vec::with_capacity(params.pairs_14.len() * 2);
                for x in &params.pairs_14 {
                    flat.extend_from_slice(x);
                }
                let arr = PyArray1::from_slice_bound(py, &flat);
                dict_bound
                    .set_item("pairs_14", arr.reshape((params.pairs_14.len(), 2)).unwrap())?;
            }

            let atom_types: Vec<&str> = params.atom_types.iter().map(|s| s.as_str()).collect();
            dict_bound.set_item("atom_types", atom_types)?;
            dict_bound.set_item("num_parameterized", params.num_parameterized)?;
            dict_bound.set_item("num_skipped", params.num_skipped)?;
        }

        // --- GAFF Typing (if requested) ---
        if let Some(ref ff) = spec.force_field {
            if ff.to_lowercase() == "gaff" {
                log::info!("Assigning GAFF atom types...");
                let all_coords = processed.extract_all_coords();
                let all_elements = &processed.raw_atoms.elements;
                // Infer topology for typing
                let topology =
                    forcefield::topology::Topology::from_coords(&all_coords, all_elements, 1.3);
                let gaff = forcefield::gaff::GaffParameters::new();
                let types = forcefield::gaff::assign_gaff_types(all_elements, &topology, &gaff);
                dict_bound.set_item("atom_types", types)?;
            }
        }

        Ok(dict.into_py(py))
    })
}

/// Project a parsed structure to MPNNBatch format for training
///
/// This function:
/// 1. Parses the structure file (PDB/mmCIF/FoldComp)
/// 2. Optionally applies Gaussian noising to coordinates
/// 3. Computes RBF features from backbone geometry
/// 4. Optionally computes physics features from force field parameters
///
/// Returns a dict with keys: aatype, residue_index, chain_index, mask,
/// rbf_features, neighbor_indices, physics_features (optional)
#[pyfunction]
#[pyo3(signature = (path, num_neighbors=30, noise_std=None, noise_seed=0, compute_physics=false))]
pub fn project_to_mpnn_batch(
    path: String,
    num_neighbors: usize,
    noise_std: Option<f32>,
    noise_seed: u64,
    compute_physics: bool,
) -> PyResult<PyObject> {
    use numpy::PyArray1;
    use numpy::PyArrayMethods;

    Python::with_gil(|py| {
        // 1. Parse the structure
        let (raw_data_all, model_ids) = if path.ends_with(".cif") || path.ends_with(".mmcif") {
            formats::mmcif::parse_mmcif_file(&path).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("mmCIF parsing failed: {}", e))
            })?
        } else if path.ends_with(".fcz") {
            // FoldComp
            let _system = formats::foldcomp::read_foldcomp(&path).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("FoldComp parsing failed: {}", e))
            })?;
            // Convert AtomicSystem to RawAtomData
            return Err(pyo3::exceptions::PyValueError::new_err(
                "FoldComp projection not yet implemented - use parse_structure instead",
            ));
        } else {
            formats::pdb::parse_pdb_file(&path).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("PDB parsing failed: {}", e))
            })?
        };

        // 2. Get first model
        let first_model = model_ids.first().copied().unwrap_or(1);
        let filtered = processing::filter_models(&raw_data_all, &model_ids, &[first_model]);

        // 3. Process structure
        let structure = ProcessedStructure::from_raw(filtered).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Structure processing failed: {}", e))
        })?;

        // 4. Project to MPNNBatch
        let result = processing::project_to_mpnn_batch(
            &structure,
            num_neighbors,
            noise_std,
            noise_seed,
            compute_physics,
        )
        .map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Projection failed: {}", e))
        })?;

        // 5. Build Python dict
        let dict = pyo3::types::PyDict::new_bound(py);

        dict.set_item("aatype", PyArray1::from_slice_bound(py, &result.aatype))?;
        dict.set_item(
            "residue_index",
            PyArray1::from_slice_bound(py, &result.residue_index),
        )?;
        dict.set_item(
            "chain_index",
            PyArray1::from_slice_bound(py, &result.chain_index),
        )?;
        dict.set_item("mask", PyArray1::from_slice_bound(py, &result.mask))?;

        // RBF features: reshape to (N_res, K, 400)
        let rbf_array = PyArray1::from_slice_bound(py, &result.rbf_features);
        let rbf_shaped = rbf_array
            .reshape((result.n_residues, result.n_neighbors, 400))
            .map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("RBF reshape failed: {}", e))
            })?;
        dict.set_item("rbf_features", rbf_shaped)?;

        // Neighbor indices: reshape to (N_res, K)
        let neighbors_array = PyArray1::from_slice_bound(py, &result.neighbor_indices);
        let neighbors_shaped = neighbors_array
            .reshape((result.n_residues, result.n_neighbors))
            .map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("Neighbors reshape failed: {}", e))
            })?;
        dict.set_item("neighbor_indices", neighbors_shaped)?;

        // Physics features if present: reshape to (N_res, 15)
        if let Some(ref feats) = result.physics_features {
            let phys_array = PyArray1::from_slice_bound(py, feats);
            let phys_shaped = phys_array.reshape((result.n_residues, 15)).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("Physics reshape failed: {}", e))
            })?;
            dict.set_item("physics_features", phys_shaped)?;
        }

        dict.set_item("n_residues", result.n_residues)?;
        dict.set_item("n_neighbors", result.n_neighbors)?;

        Ok(dict.into_py(py))
    })
}

/// FoldComp Database accessor
#[pyclass]
pub struct FoldCompDatabase {
    inner: formats::foldcomp::db::FoldCompDb,
}

#[pymethods]
impl FoldCompDatabase {
    #[new]
    pub fn new(path: String) -> PyResult<Self> {
        let db = formats::foldcomp::db::FoldCompDb::open(&path).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to open FoldComp database: {}", e))
        })?;
        Ok(FoldCompDatabase { inner: db })
    }

    pub fn get(&self, id: u32) -> Result<crate::structure::systems::AtomicSystem, PyErr> {
        self.inner.get(id).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Failed to retrieve ID {}: {}", id, e))
        })
    }

    pub fn get_by_name(
        &self,
        name: String,
    ) -> Result<crate::structure::systems::AtomicSystem, PyErr> {
        self.inner.get_by_name(&name).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Failed to retrieve entry {}: {}",
                name, e
            ))
        })
    }

    pub fn __len__(&self) -> usize {
        self.inner.keys.len()
    }

    pub fn __contains__(&self, name: String) -> bool {
        self.inner.lookup.contains_key(&name)
    }
}
