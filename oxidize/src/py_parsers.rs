use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::PyArray1;
use numpy::PyArrayMethods;
use crate::{formats, formatters, geometry, physics, processing, spec, forcefield};
use crate::processing::ProcessedStructure;
use crate::spec::{CoordFormat, OutputSpec};

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

/// Parse a structure file (PDB/mmCIF) into a format suitable for the Protein class.
#[pyfunction]
#[pyo3(signature = (path, spec=None))]
pub fn parse_structure(path: String, spec: Option<OutputSpec>) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let spec = spec.unwrap_or_default();

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
                spec.coord_format.clone(),
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
        let mut bonds_for_h = Vec::new();
        if spec.add_hydrogens {
            log::debug!("Adding hydrogens to reference model...");
            let all_coords = processed.extract_all_coords();
            let all_elements = &processed.raw_atoms.elements;

            // Infer bonds if not provided
            bonds_for_h = geometry::topology::infer_bonds(&all_coords, all_elements, 1.3);

            let num_h_added = geometry::hydrogens::add_hydrogens(&mut processed, &mut bonds_for_h)
                .map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "Hydrogen addition failed: {}",
                        e
                    ))
                })?;

            log::debug!("Added {} hydrogens to reference model", num_h_added);

            // Note: ProcessedStructure was updated in-place, no need to reassign
            let _ = processed; // Keep processed in scope (no-op, updated in place)
        } else if spec.infer_bonds {
            let all_coords = processed.extract_all_coords();
            let all_elements = &processed.raw_atoms.elements;
            bonds_for_h = geometry::topology::infer_bonds(&all_coords, all_elements, 1.3);
        }

        // 5. Format and Stack
        let (mut dict, cached_structure) = match spec.coord_format {
            CoordFormat::Atom37 => {
                let formatter = formatters::Atom37Formatter;
                let ref_formatted = formatters::Atom37Formatter::format(&processed, &spec)
                    .map_err(|e| {
                        pyo3::exceptions::PyValueError::new_err(format!("Formatting failed: {}", e))
                    })?;

                let mut dict = ref_formatted.to_py_dict(py)?;

                // --- Multi-Model Stacking for Atom37 ---
                if models_to_process.len() > 1 {
                    let n_res = ref_formatted.coordinates.len() / (37 * 3);
                    let n_models = models_to_process.len();

                    let mut all_coords = Vec::with_capacity(n_models * n_res * 37 * 3);
                    all_coords.extend_from_slice(&ref_formatted.coordinates);

                    for i in 1..n_models {
                        let m_raw = models_to_process[i];
                        // We must process each model to map atoms correctly
                        let mut m_processed =
                            ProcessedStructure::from_raw(m_raw.clone()).map_err(|e| {
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
                            geometry::hydrogens::add_hydrogens(&mut m_processed, &mut m_bonds)
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
                        num_residues: ref_formatted.aatype.len(),
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
                        num_residues: formatted.aatype.len(),
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

        // --- Geometry Features ---
        if spec.compute_rbf {
            // Extract coordinates
            let ca_coords = processed.extract_ca_coords();
            let backbone_coords = processed.extract_backbone_coords();

            // 1. Find neighbors
            let neighbor_indices =
                geometry::neighbors::find_k_nearest_neighbors(&ca_coords, spec.rbf_num_neighbors);

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
                    let backbone_coords = processed.extract_backbone_coords();
                    let all_coords = processed.extract_all_coords();

                    let features = physics::electrostatics::compute_electrostatic_features(
                        &backbone_coords,
                        &all_coords,
                        charges,
                    );

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
            let backbone_coords = processed.extract_backbone_coords();
            let all_coords = processed.extract_all_coords();

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

            // Compute VdW features (N_res * 5 values)
            let features = physics::vdw::compute_vdw_features(
                &backbone_coords,
                &all_coords,
                &all_sigmas,
                &all_epsilons,
            );

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

        // --- MD Parameterization ---
        if spec.parameterize_md {
            if let Some(ref ff_path) = spec.force_field {
                // Parse force field
                let ff = forcefield::parse_forcefield_xml(ff_path).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "Force field parsing failed: {}",
                        e
                    ))
                })?;

                // Convert spec mode to physics mode
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

                // Parameterize structure
                let params =
                    physics::md_params::parameterize_structure(&processed, &ff, &param_options)
                        .map_err(|e| {
                            pyo3::exceptions::PyValueError::new_err(format!(
                                "Parameterization failed: {}",
                                e
                            ))
                        })?;

                // Add to output
                dict_bound.set_item("charges", PyArray1::from_slice_bound(py, &params.charges))?;
                dict_bound.set_item("sigmas", PyArray1::from_slice_bound(py, &params.sigmas))?;
                dict_bound
                    .set_item("epsilons", PyArray1::from_slice_bound(py, &params.epsilons))?;

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

                // Bonds (N, 2)
                if !params.bonds.is_empty() {
                    let mut bonds_flat = Vec::with_capacity(params.bonds.len() * 2);
                    for b in &params.bonds {
                        bonds_flat.extend_from_slice(b);
                    }
                    let arr = PyArray1::from_slice_bound(py, &bonds_flat);
                    dict_bound.set_item("bonds", arr.reshape((params.bonds.len(), 2)).unwrap())?;
                }

                // Bond Params (N, 2)
                if !params.bond_params.is_empty() {
                    let mut params_flat = Vec::with_capacity(params.bond_params.len() * 2);
                    for p in &params.bond_params {
                        params_flat.extend_from_slice(p);
                    }
                    let arr = PyArray1::from_slice_bound(py, &params_flat);
                    dict_bound.set_item(
                        "bond_params",
                        arr.reshape((params.bond_params.len(), 2)).unwrap(),
                    )?;
                }

                // Angles (N, 3)
                if !params.angles.is_empty() {
                    let mut flat = Vec::with_capacity(params.angles.len() * 3);
                    for a in &params.angles {
                        flat.extend_from_slice(a);
                    }
                    let arr = PyArray1::from_slice_bound(py, &flat);
                    dict_bound
                        .set_item("angles", arr.reshape((params.angles.len(), 3)).unwrap())?;
                }

                // Angle Params (N, 2)
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

                // Proper Dihedrals (N, 4)
                if !params.dihedrals.is_empty() {
                    let mut flat = Vec::with_capacity(params.dihedrals.len() * 4);
                    for d in &params.dihedrals {
                        flat.extend_from_slice(d);
                    }
                    let arr = PyArray1::from_slice_bound(py, &flat);
                    dict_bound.set_item(
                        "dihedrals",
                        arr.reshape((params.dihedrals.len(), 4)).unwrap(),
                    )?;
                }

                // Proper Dihedral Params (N, 3)
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

                // Improper Dihedrals (N, 4)
                if !params.impropers.is_empty() {
                    let mut flat = Vec::with_capacity(params.impropers.len() * 4);
                    for i in &params.impropers {
                        flat.extend_from_slice(i);
                    }
                    let arr = PyArray1::from_slice_bound(py, &flat);
                    dict_bound.set_item(
                        "impropers",
                        arr.reshape((params.impropers.len(), 4)).unwrap(),
                    )?;
                }

                // Improper Params (N, 3)
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

                // 1-4 Pairs (N, 2)
                if !params.pairs_14.is_empty() {
                    let mut flat = Vec::with_capacity(params.pairs_14.len() * 2);
                    for p in &params.pairs_14 {
                        flat.extend_from_slice(p);
                    }
                    let arr = PyArray1::from_slice_bound(py, &flat);
                    dict_bound
                        .set_item("pairs_14", arr.reshape((params.pairs_14.len(), 2)).unwrap())?;
                }

                // Atom types as list
                let atom_types: Vec<&str> = params.atom_types.iter().map(|s| s.as_str()).collect();
                dict_bound.set_item("atom_types", atom_types)?;

                dict_bound.set_item("num_parameterized", params.num_parameterized)?;
                dict_bound.set_item("num_skipped", params.num_skipped)?;

                log::info!(
                    "Parameterized {}/{} atoms",
                    params.num_parameterized,
                    processed.raw_atoms.num_atoms
                );
            } else {
                log::warn!("parameterize_md=true but no force_field path provided");
            }
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
