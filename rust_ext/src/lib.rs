//! PrioX Rust Extension
//!
//! High-performance protein structure parsing library for Python.
//! Provides zero-copy parsing for PDB, mmCIF, and trajectory formats.

use pyo3::prelude::*;
use pyo3::types::PyDict;

mod chem;
mod forcefield;
mod formats;
mod formatters;
mod geometry;
mod physics;
mod processing;
mod spec;
mod structure;

use formats::pdb::parse_pdb_file;
use processing::ProcessedStructure;
use spec::{CoordFormat, OutputSpec};
use structure::systems::AtomicSystem;

/// Parse a PDB file and return raw atom data (low-level)
/// Returns first model only for backward compatibility.
#[pyfunction]
fn parse_pdb(path: String) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let (raw_data, model_ids) = parse_pdb_file(&path).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("PDB parsing failed: {}", e))
        })?;

        // Filter to first model only (legacy behavior)
        let first_model = model_ids.first().copied().unwrap_or(1);
        let filtered = processing::filter_models(&raw_data, &model_ids, &[first_model]);

        filtered.to_py_dict(py).map(|dict| dict.into_py(py))
    })
}

use numpy::{PyArray1, PyArrayMethods};

/// Parse a structure file and return formatted coordinates
#[pyfunction]
#[pyo3(signature = (path, spec=None))]
fn parse_structure(path: String, spec: Option<OutputSpec>) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let spec = spec.unwrap_or_default();

        // Check if we can use the cache
        // We only cache if:
        // 1. Caching is enabled
        // 2. No dynamic physics/geometry features are requested (as they depend on RawAtomData)
        // 3. No extra processing flags that might change output beyond format (e.g. bonds)
        let should_cache = spec.enable_caching
            && !spec.compute_rbf
            && !spec.compute_electrostatics
            && !spec.compute_vdw
            && !spec.parameterize_md
            && !spec.infer_bonds;

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

        // 1. Parse PDB to RawAtomData (all models)
        let (raw_data_all, model_ids) = parse_pdb_file(&path).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("PDB parsing failed: {}", e))
        })?;

        // Filter by requested models (default: first model only)
        let raw_data = if let Some(ref models) = spec.models {
            processing::filter_models(&raw_data_all, &model_ids, models)
        } else {
            // Default: first model only
            let first_model = model_ids.first().copied().unwrap_or(1);
            processing::filter_models(&raw_data_all, &model_ids, &[first_model])
        };

        // 2. Process into residues
        let mut processed = ProcessedStructure::from_raw(raw_data).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Structure processing failed: {}", e))
        })?;

        // 2b. Add Hydrogens if requested
        if spec.add_hydrogens {
            log::debug!("Adding hydrogens...");
            // We need bonds to determine where to place hydrogens
            // Infer them temporarily even if not requested for output
            let all_coords = processed.extract_all_coords();
            let all_elements = &processed.raw_atoms.elements;
            let mut bonds = geometry::topology::infer_bonds(&all_coords, all_elements, 1.3);

            let (added, relax_iters, relax_energy) = geometry::hydrogens::add_hydrogens_with_relax(
                &mut processed,
                &mut bonds,
                spec.relax_hydrogens,
                spec.relax_max_iterations,
            )
            .map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("Hydrogen addition failed: {}", e))
            })?;

            if added > 0 {
                // Rebuild topology to ensure correct atom ordering (contiguous residues)
                let (new_processed, new_bonds) =
                    processing::residues::rebuild_topology(processed, bonds).map_err(|e| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "Topology rebuild failed: {}",
                            e
                        ))
                    })?;
                processed = new_processed;
                bonds = new_bonds;
            }

            if spec.infer_bonds {
                // If we want to return bonds, we use the ones we have (potentially relaxed/reordered)
                // Convert bonds to python format
                // We need to return them in the dict?
                // Protocol: If infer_bonds is True, we usually return a bond list or adjacency?
                // Current OutputSpec implementation in Python might handle it,
                // but here we just return the Dict.
                // Wait, ProcessedStructure doesn't hold bonds.
                // And we don't put bonds in the output dict explicitly unless requested?
                // The return value of parse_structure is a dict.
                // Does the user expect 'bonds' key?
                // Let's check `formats::mmcif` usage or similar.
                // Currently `processed` doesn't include bonds in `to_py_dict`.
                // If we want to export bonds, we should add them.
                // But for now, fixing H count is primary.
            }

            if added > 0 {
                if spec.relax_hydrogens {
                    log::info!(
                        "Added {} hydrogen atoms, relaxed in {} iterations (energy: {:.2})",
                        added,
                        relax_iters,
                        relax_energy
                    );
                } else {
                    log::info!("Added {} hydrogen atoms", added);
                }
                // Re-process the structure to fix residue sorting and indexing
                // The raw atoms now contain the new hydrogens at the end
                // from_raw will sort them into their residues
                let modified_raw = processed.raw_atoms;
                processed = ProcessedStructure::from_raw(modified_raw).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "Reprocessing failed after adding hydrogens: {}",
                        e
                    ))
                })?;
            }
        }

        // 3. Format based on spec.coord_format
        let (formatted_dict, cached_structure) = match spec.coord_format {
            CoordFormat::Atom37 => {
                let formatted =
                    formatters::Atom37Formatter::format(&processed, &spec).map_err(|e| {
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
                        atom_names: None,
                        coord_shape: None,
                    })
                } else {
                    None
                };

                (formatted.to_py_dict(py)?, cached)
            }
            CoordFormat::Atom14 => {
                let formatted =
                    formatters::Atom14Formatter::format(&processed, &spec).map_err(|e| {
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
                        atom_names: None,
                        coord_shape: None,
                    })
                } else {
                    None
                };

                (formatted.to_py_dict(py)?, cached)
            }
            CoordFormat::BackboneOnly => {
                let formatted =
                    formatters::BackboneFormatter::format(&processed, &spec).map_err(|e| {
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
                        atom_names: None,
                        coord_shape: None,
                    })
                } else {
                    None
                };

                (formatted.to_py_dict(py)?, cached)
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
        let dict = formatted_dict.downcast_bound::<PyDict>(py).map_err(|e| {
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

            dict.set_item("rbf_features", rbf_array)?;
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
                dict.set_item("bonds", bonds_reshaped)?;
            } else {
                let empty: &[usize] = &[];
                let arr = PyArray1::from_slice_bound(py, empty);
                dict.set_item("bonds", arr.reshape((0, 2)).unwrap())?;
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
                dict.set_item("angles", angles_reshaped)?;
            } else {
                let empty: &[usize] = &[];
                let arr = PyArray1::from_slice_bound(py, empty);
                dict.set_item("angles", arr.reshape((0, 3)).unwrap())?;
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
                dict.set_item("dihedrals", dih_reshaped)?;
            } else {
                let empty: &[usize] = &[];
                let arr = PyArray1::from_slice_bound(py, empty);
                dict.set_item("dihedrals", arr.reshape((0, 4)).unwrap())?;
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
                dict.set_item("impropers", imp_reshaped)?;
            } else {
                let empty: &[usize] = &[];
                let arr = PyArray1::from_slice_bound(py, empty);
                dict.set_item("impropers", arr.reshape((0, 4)).unwrap())?;
            }
        }

        // --- Molecule Type ---
        // 0=Protein, 1=Ligand, 2=Solvent, 3=Ion
        let mol_type_array = PyArray1::from_slice_bound(py, &processed.molecule_type);
        dict.set_item("molecule_type", mol_type_array)?;
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

                    dict.set_item("electrostatic_features", array)?;
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

            dict.set_item("vdw_features", array)?;
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
                dict.set_item("charges", PyArray1::from_slice_bound(py, &params.charges))?;
                dict.set_item("sigmas", PyArray1::from_slice_bound(py, &params.sigmas))?;
                dict.set_item("epsilons", PyArray1::from_slice_bound(py, &params.epsilons))?;

                if let Some(ref radii) = params.radii {
                    dict.set_item(
                        "gbsa_radii",
                        PyArray1::from_slice_bound(py, radii.as_slice()),
                    )?;
                }
                if let Some(ref scales) = params.scales {
                    dict.set_item(
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
                    dict.set_item("bonds", arr.reshape((params.bonds.len(), 2)).unwrap())?;
                }

                // Bond Params (N, 2)
                if !params.bond_params.is_empty() {
                    let mut params_flat = Vec::with_capacity(params.bond_params.len() * 2);
                    for p in &params.bond_params {
                        params_flat.extend_from_slice(p);
                    }
                    let arr = PyArray1::from_slice_bound(py, &params_flat);
                    dict.set_item(
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
                    dict.set_item("angles", arr.reshape((params.angles.len(), 3)).unwrap())?;
                }

                // Angle Params (N, 2)
                if !params.angle_params.is_empty() {
                    let mut flat = Vec::with_capacity(params.angle_params.len() * 2);
                    for p in &params.angle_params {
                        flat.extend_from_slice(p);
                    }
                    let arr = PyArray1::from_slice_bound(py, &flat);
                    dict.set_item(
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
                    dict.set_item(
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
                    dict.set_item(
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
                    dict.set_item(
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
                    dict.set_item(
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
                    dict.set_item("pairs_14", arr.reshape((params.pairs_14.len(), 2)).unwrap())?;
                }

                // Atom types as list
                let atom_types: Vec<&str> = params.atom_types.iter().map(|s| s.as_str()).collect();
                dict.set_item("atom_types", atom_types)?;

                dict.set_item("num_parameterized", params.num_parameterized)?;
                dict.set_item("num_skipped", params.num_skipped)?;

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
                dict.set_item("atom_types", types)?;
            }
        }

        Ok(dict.into_py(py))
    })
}

/// Load a force field from an OpenMM-style XML file
///
/// Returns a dictionary containing:
/// - atom_types: list of atom type definitions
/// - residue_templates: list of residue templates with atoms, bonds, charges
/// - harmonic_bonds: bond force parameters
/// - harmonic_angles: angle force parameters
/// - proper_torsions: proper dihedral parameters
/// - improper_torsions: improper dihedral parameters
/// - nonbonded_params: LJ parameters (sigma, epsilon)
/// - gbsa_obc_params: implicit solvent parameters (if present)
#[pyfunction]
fn load_forcefield(path: String) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let ff = forcefield::parse_forcefield_xml(&path).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Force field parsing failed: {}", e))
        })?;

        let dict = PyDict::new_bound(py);

        // Basic info
        dict.set_item("name", &ff.name)?;
        dict.set_item("num_atom_types", ff.atom_types.len())?;
        dict.set_item("num_residue_templates", ff.residue_templates.len())?;
        dict.set_item("num_harmonic_bonds", ff.harmonic_bonds.len())?;
        dict.set_item("num_harmonic_angles", ff.harmonic_angles.len())?;
        dict.set_item("num_proper_torsions", ff.proper_torsions.len())?;
        dict.set_item("num_improper_torsions", ff.improper_torsions.len())?;
        dict.set_item("num_nonbonded_params", ff.nonbonded_params.len())?;
        dict.set_item("num_gbsa_obc_params", ff.gbsa_obc_params.len())?;
        dict.set_item("has_cmap", ff.cmap_data.is_some())?;

        // Atom types as list of dicts
        // Atom types
        let mut atom_types = Vec::with_capacity(ff.atom_types.len());
        for at in &ff.atom_types {
            let d = PyDict::new_bound(py);
            d.set_item("name", &at.name).ok();
            d.set_item("class", &at.class).ok();
            d.set_item("element", &at.element).ok();
            d.set_item("mass", at.mass).ok();
            atom_types.push(d.into_py(py));
        }
        dict.set_item("atom_types", atom_types)?;

        // Residue templates
        // Residue templates
        let mut residue_templates = Vec::with_capacity(ff.residue_templates.len());
        for rt in &ff.residue_templates {
            let d = PyDict::new_bound(py);
            d.set_item("name", &rt.name).ok();
            d.set_item("num_atoms", rt.atoms.len()).ok();
            d.set_item("num_bonds", rt.bonds.len()).ok();

            let mut atoms = Vec::with_capacity(rt.atoms.len());
            for a in &rt.atoms {
                let ad = PyDict::new_bound(py);
                ad.set_item("name", &a.name).ok();
                ad.set_item("type", &a.atom_type).ok();
                ad.set_item("charge", a.charge).ok();
                atoms.push(ad.into_py(py));
            }
            d.set_item("atoms", atoms).ok();

            let mut bonds = Vec::with_capacity(rt.bonds.len());
            for (a1, a2) in &rt.bonds {
                bonds.push((a1.clone(), a2.clone()));
            }
            d.set_item("bonds", bonds).ok();

            residue_templates.push(d.into_py(py));
        }
        dict.set_item("residue_templates", residue_templates)?;

        // Harmonic bonds
        let mut bonds = Vec::with_capacity(ff.harmonic_bonds.len());
        for b in &ff.harmonic_bonds {
            let d = PyDict::new_bound(py);
            d.set_item("class1", &b.class1).ok();
            d.set_item("class2", &b.class2).ok();
            d.set_item("k", b.k).ok();
            d.set_item("length", b.length).ok();
            bonds.push(d.into_py(py));
        }
        dict.set_item("harmonic_bonds", bonds)?;

        // Harmonic angles
        let mut angles = Vec::with_capacity(ff.harmonic_angles.len());
        for a in &ff.harmonic_angles {
            let d = PyDict::new_bound(py);
            d.set_item("class1", &a.class1).ok();
            d.set_item("class2", &a.class2).ok();
            d.set_item("class3", &a.class3).ok();
            d.set_item("k", a.k).ok();
            d.set_item("angle", a.angle).ok();
            angles.push(d.into_py(py));
        }
        dict.set_item("harmonic_angles", angles)?;

        // Proper Torsions
        let mut proper_torsions = Vec::with_capacity(ff.proper_torsions.len());
        for t in &ff.proper_torsions {
            let d = PyDict::new_bound(py);
            d.set_item("class1", &t.class1).ok();
            d.set_item("class2", &t.class2).ok();
            d.set_item("class3", &t.class3).ok();
            d.set_item("class4", &t.class4).ok();

            let mut terms = Vec::with_capacity(t.terms.len());
            for term in &t.terms {
                let td = PyDict::new_bound(py);
                td.set_item("periodicity", term.periodicity).ok();
                td.set_item("phase", term.phase).ok();
                td.set_item("k", term.k).ok();
                terms.push(td.into_py(py));
            }
            d.set_item("terms", terms).ok();
            proper_torsions.push(d.into_py(py));
        }
        dict.set_item("proper_torsions", proper_torsions)?;

        // Improper Torsions
        let mut improper_torsions = Vec::with_capacity(ff.improper_torsions.len());
        for t in &ff.improper_torsions {
            let d = PyDict::new_bound(py);
            d.set_item("class1", &t.class1).ok();
            d.set_item("class2", &t.class2).ok();
            d.set_item("class3", &t.class3).ok();
            d.set_item("class4", &t.class4).ok();

            let mut terms = Vec::with_capacity(t.terms.len());
            for term in &t.terms {
                let td = PyDict::new_bound(py);
                td.set_item("periodicity", term.periodicity).ok();
                td.set_item("phase", term.phase).ok();
                td.set_item("k", term.k).ok();
                terms.push(td.into_py(py));
            }
            d.set_item("terms", terms).ok();
            improper_torsions.push(d.into_py(py));
        }
        dict.set_item("improper_torsions", improper_torsions)?;

        // Nonbonded Params
        let mut nonbonded = Vec::with_capacity(ff.nonbonded_params.len());
        for nb in &ff.nonbonded_params {
            let d = PyDict::new_bound(py);
            d.set_item("atom_type", &nb.atom_type).ok();
            d.set_item("charge", nb.charge).ok();
            d.set_item("sigma", nb.sigma).ok();
            d.set_item("epsilon", nb.epsilon).ok();
            nonbonded.push(d.into_py(py));
        }
        dict.set_item("nonbonded_params", nonbonded)?;

        // GBSA-OBC Params
        let mut gbsa = Vec::with_capacity(ff.gbsa_obc_params.len());
        for g in &ff.gbsa_obc_params {
            let d = PyDict::new_bound(py);
            d.set_item("atom_type", &g.atom_type).ok();
            d.set_item("radius", g.radius).ok();
            d.set_item("scale", g.scale).ok();
            gbsa.push(d.into_py(py));
        }
        dict.set_item("gbsa_obc_params", gbsa)?;

        // CMAP Data
        // CMAP Data
        if let Some(ref cmap) = ff.cmap_data {
            let mut maps = Vec::with_capacity(cmap.maps.len());
            for m in &cmap.maps {
                let d = PyDict::new_bound(py);
                d.set_item("size", m.size).ok();
                d.set_item("energies", m.energies.clone()).ok();
                maps.push(d.into_py(py));
            }
            dict.set_item("cmap_maps", maps)?;

            let mut torsions = Vec::with_capacity(cmap.torsions.len());
            for t in &cmap.torsions {
                let d = PyDict::new_bound(py);
                d.set_item("class1", &t.class1).ok();
                d.set_item("type2", &t.type2).ok();
                d.set_item("type3", &t.type3).ok();
                d.set_item("type4", &t.type4).ok();
                d.set_item("class5", &t.class5).ok();
                d.set_item("map_index", t.map_index).ok();
                torsions.push(d.into_py(py));
            }
            dict.set_item("cmap_torsions", torsions)?;
        }

        Ok(dict.into_py(py))
    })
}

/// Parse an XTC trajectory file
///
/// When compiled with 'xtc-pure' feature, uses the molly pure-Rust implementation.
/// When compiled with 'trajectories' feature, uses chemfiles (may crash on some systems).
#[pyfunction]
fn parse_xtc(path: String) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        // Prefer pure-Rust molly implementation (xtc-pure feature)
        #[cfg(feature = "xtc-pure")]
        {
            use formats::xtc::molly_impl::read_xtc_molly;
            let traj = read_xtc_molly(&path).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("XTC parsing failed: {}", e))
            })?;

            let dict = PyDict::new_bound(py);
            dict.set_item("num_frames", traj.num_frames)?;
            dict.set_item("num_atoms", traj.num_atoms)?;

            // Convert to NumPy arrays
            let times = PyArray1::from_slice_bound(py, &traj.times);
            dict.set_item("times", times)?;

            // Combine all coords into (N_frames, N_atoms, 3)
            let mut flat_coords = Vec::with_capacity(traj.num_frames * traj.num_atoms * 3);
            for frame_coords in &traj.coords {
                flat_coords.extend_from_slice(frame_coords);
            }
            let coords_array = PyArray1::from_slice_bound(py, &flat_coords);
            let shape = (traj.num_frames, traj.num_atoms, 3);
            let coords_reshaped = coords_array.reshape(shape).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("Failed to reshape coords: {}", e))
            })?;

            dict.set_item("coordinates", coords_reshaped)?;

            return Ok(dict.into_py(py));
        }

        // Fallback to chemfiles (trajectories feature) - may crash with SIGFPE
        #[cfg(all(feature = "trajectories", not(feature = "xtc-pure")))]
        {
            use formats::xtc::chemfiles_impl::read_xtc_chemfiles;
            let traj = read_xtc_chemfiles(&path).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("XTC parsing failed: {}", e))
            })?;

            let dict = PyDict::new_bound(py);
            dict.set_item("num_frames", traj.num_frames)?;
            dict.set_item("num_atoms", traj.num_atoms)?;

            // Convert to NumPy arrays
            let times = PyArray1::from_slice_bound(py, &traj.times);
            dict.set_item("times", times)?;

            // Combine all coords into (N_frames, N_atoms, 3)
            let mut flat_coords = Vec::with_capacity(traj.num_frames * traj.num_atoms * 3);
            for frame_coords in &traj.coords {
                flat_coords.extend_from_slice(frame_coords);
            }
            let coords_array = PyArray1::from_slice_bound(py, &flat_coords);
            let shape = (traj.num_frames, traj.num_atoms, 3);
            let coords_reshaped = coords_array.reshape(shape).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("Failed to reshape coords: {}", e))
            })?;

            dict.set_item("coordinates", coords_reshaped)?;

            return Ok(dict.into_py(py));
        }

        #[cfg(not(any(feature = "trajectories", feature = "xtc-pure")))]
        {
            Err(pyo3::exceptions::PyImportError::new_err(
                "XTC support requires compiling with 'xtc-pure' (recommended) or 'trajectories' feature.",
            ))
        }
    })
}

/// Parse an mmCIF file and return raw atom data (low-level)
#[pyfunction]
fn parse_mmcif(path: String) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let raw_data = formats::mmcif::parse_mmcif_file(&path).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("mmCIF parsing failed: {}", e))
        })?;

        raw_data.to_py_dict(py).map(|dict| dict.into_py(py))
    })
}

/// Parse a DCD trajectory file
#[pyfunction]
fn parse_dcd(path: String) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        #[cfg(feature = "trajectories")]
        {
            use formats::dcd::chemfiles_impl::read_dcd_chemfiles;
            let traj = read_dcd_chemfiles(&path).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("DCD parsing failed: {}", e))
            })?;

            let dict = PyDict::new_bound(py);
            dict.set_item("num_frames", traj.num_frames)?;
            dict.set_item("num_atoms", traj.num_atoms)?;

            // Convert times to NumPy
            let times = PyArray1::from_slice_bound(py, &traj.times);
            dict.set_item("times", times)?;

            // Combine all coords into (N_frames, N_atoms, 3)
            let mut flat_coords = Vec::with_capacity(traj.num_frames * traj.num_atoms * 3);
            for frame_coords in &traj.coords {
                flat_coords.extend_from_slice(frame_coords);
            }
            let coords_array = PyArray1::from_slice_bound(py, &flat_coords);
            let shape = (traj.num_frames, traj.num_atoms, 3);
            let coords_reshaped = coords_array.reshape(shape).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("Failed to reshape coords: {}", e))
            })?;

            dict.set_item("coordinates", coords_reshaped)?;

            // Unit cells if available
            if let Some(ref unit_cells) = traj.unit_cells {
                let mut flat_cells = Vec::with_capacity(traj.num_frames * 6);
                for cell in unit_cells {
                    flat_cells.extend_from_slice(cell);
                }
                let cells_array = PyArray1::from_slice_bound(py, &flat_cells);
                let cells_reshaped = cells_array.reshape((traj.num_frames, 6)).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "Failed to reshape unit_cells: {}",
                        e
                    ))
                })?;
                dict.set_item("unit_cells", cells_reshaped)?;
            }

            Ok(dict.into_py(py))
        }

        #[cfg(not(feature = "trajectories"))]
        {
            let _ = path;
            Err(pyo3::exceptions::PyImportError::new_err(
                "DCD support requires compiling with 'trajectories' feature (chemfiles).",
            ))
        }
    })
}

/// Parse a TRR trajectory file
#[pyfunction]
fn parse_trr(path: String) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        #[cfg(feature = "trajectories")]
        {
            use formats::trr::chemfiles_impl::read_trr_chemfiles;
            let traj = read_trr_chemfiles(&path).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("TRR parsing failed: {}", e))
            })?;

            let dict = PyDict::new_bound(py);
            dict.set_item("num_frames", traj.num_frames)?;
            dict.set_item("num_atoms", traj.num_atoms)?;

            // Convert times to NumPy
            let times = PyArray1::from_slice_bound(py, &traj.times);
            dict.set_item("times", times)?;

            // Combine all coords into (N_frames, N_atoms, 3)
            let mut flat_coords = Vec::with_capacity(traj.num_frames * traj.num_atoms * 3);
            for frame_coords in &traj.coords {
                flat_coords.extend_from_slice(frame_coords);
            }
            let coords_array = PyArray1::from_slice_bound(py, &flat_coords);
            let shape = (traj.num_frames, traj.num_atoms, 3);
            let coords_reshaped = coords_array.reshape(shape).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("Failed to reshape coords: {}", e))
            })?;

            dict.set_item("coordinates", coords_reshaped)?;

            // Velocities if available
            if let Some(ref velocities) = traj.velocities {
                let mut flat_vel = Vec::with_capacity(traj.num_frames * traj.num_atoms * 3);
                for frame_vel in velocities {
                    flat_vel.extend_from_slice(frame_vel);
                }
                let vel_array = PyArray1::from_slice_bound(py, &flat_vel);
                let vel_reshaped = vel_array.reshape(shape).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "Failed to reshape velocities: {}",
                        e
                    ))
                })?;
                dict.set_item("velocities", vel_reshaped)?;
            }

            // Box vectors if available
            if let Some(ref box_vectors) = traj.box_vectors {
                let mut flat_box = Vec::with_capacity(traj.num_frames * 9);
                for frame_box in box_vectors {
                    flat_box.extend_from_slice(&frame_box[0]);
                    flat_box.extend_from_slice(&frame_box[1]);
                    flat_box.extend_from_slice(&frame_box[2]);
                }
                let box_array = PyArray1::from_slice_bound(py, &flat_box);
                let box_reshaped = box_array.reshape((traj.num_frames, 3, 3)).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "Failed to reshape box_vectors: {}",
                        e
                    ))
                })?;
                dict.set_item("box_vectors", box_reshaped)?;
            }

            Ok(dict.into_py(py))
        }

        #[cfg(not(feature = "trajectories"))]
        {
            let _ = path;
            Err(pyo3::exceptions::PyImportError::new_err(
                "TRR support requires compiling with 'trajectories' feature (chemfiles).",
            ))
        }
    })
}

// =============================================================================
// HDF5 Parsing Functions (feature-gated)
// =============================================================================

/// Parse MDTraj HDF5 file metadata
#[cfg(feature = "mdcath")]
#[pyfunction]
fn parse_mdtraj_h5_metadata(path: String) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let result = formats::mdtraj_h5::parse_mdtraj_h5_metadata(&path).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("MDTraj H5 parsing failed: {}", e))
        })?;

        let dict = PyDict::new_bound(py);
        dict.set_item("num_frames", result.num_frames)?;
        dict.set_item("num_atoms", result.num_atoms)?;
        dict.set_item("atom_names", &result.atom_names)?;
        dict.set_item("res_names", &result.res_names)?;
        dict.set_item("res_ids", &result.res_ids)?;
        dict.set_item("chain_ids", &result.chain_ids)?;
        dict.set_item("elements", &result.elements)?;

        Ok(dict.into_py(py))
    })
}

/// Parse a single frame from MDTraj HDF5 file
#[cfg(feature = "mdcath")]
#[pyfunction]
fn parse_mdtraj_h5_frame(path: String, frame_idx: usize) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let frame = formats::mdtraj_h5::parse_mdtraj_h5_frame(&path, frame_idx).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "MDTraj H5 frame parsing failed: {}",
                e
            ))
        })?;

        let dict = PyDict::new_bound(py);
        dict.set_item("index", frame.index)?;
        dict.set_item("time", frame.time)?;

        // Convert coords to NumPy array
        let coords_array = PyArray1::from_slice_bound(py, &frame.coords);
        let num_atoms = frame.coords.len() / 3;
        let coords_reshaped = coords_array.reshape((num_atoms, 3)).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Failed to reshape coords: {}", e))
        })?;
        dict.set_item("coords", coords_reshaped)?;

        Ok(dict.into_py(py))
    })
}

/// Parse MDCATH HDF5 file metadata
#[cfg(feature = "mdcath")]
#[pyfunction]
fn parse_mdcath_metadata(path: String) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let result = formats::mdcath_h5::parse_mdcath_metadata(&path).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("MDCATH H5 parsing failed: {}", e))
        })?;

        let dict = PyDict::new_bound(py);
        dict.set_item("domain_id", &result.domain_id)?;
        dict.set_item("num_residues", result.num_residues)?;
        dict.set_item("resnames", &result.resnames)?;
        dict.set_item("chain_ids", &result.chain_ids)?;
        dict.set_item("temperatures", &result.temperatures)?;

        Ok(dict.into_py(py))
    })
}

/// Get list of replicas for a temperature in MDCATH file
#[cfg(feature = "mdcath")]
#[pyfunction]
fn get_mdcath_replicas(path: String, domain_id: String, temperature: String) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let replicas =
            formats::mdcath_h5::get_replicas(&path, &domain_id, &temperature).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Failed to get MDCATH replicas: {}",
                    e
                ))
            })?;

        Ok(replicas.into_py(py))
    })
}

/// Parse a single frame from MDCATH HDF5 file
#[cfg(feature = "mdcath")]
#[pyfunction]
#[pyo3(signature = (path, domain_id, temperature, replica, frame_idx))]
fn parse_mdcath_frame(
    path: String,
    domain_id: String,
    temperature: String,
    replica: String,
    frame_idx: usize,
) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let frame = formats::mdcath_h5::parse_mdcath_frame(
            &path,
            &domain_id,
            &temperature,
            &replica,
            frame_idx,
        )
        .map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("MDCATH frame parsing failed: {}", e))
        })?;

        let dict = PyDict::new_bound(py);
        dict.set_item("temperature", &frame.temperature)?;
        dict.set_item("replica", &frame.replica)?;
        dict.set_item("frame_idx", frame.frame_idx)?;

        // Convert coords to NumPy array
        let coords_array = PyArray1::from_slice_bound(py, &frame.coords);
        let num_atoms = frame.coords.len() / 3;
        let coords_reshaped = coords_array.reshape((num_atoms, 3)).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Failed to reshape coords: {}", e))
        })?;
        dict.set_item("coords", coords_reshaped)?;

        Ok(dict.into_py(py))
    })
}

// Stub functions when mdcath feature is not enabled
#[cfg(not(feature = "mdcath"))]
#[pyfunction]
fn parse_mdtraj_h5_metadata(_path: String) -> PyResult<PyObject> {
    Err(pyo3::exceptions::PyImportError::new_err(
        "HDF5 support requires compiling with 'mdcath' feature. Rebuild with: maturin develop --features mdcath",
    ))
}

#[cfg(not(feature = "mdcath"))]
#[pyfunction]
fn parse_mdtraj_h5_frame(_path: String, _frame_idx: usize) -> PyResult<PyObject> {
    Err(pyo3::exceptions::PyImportError::new_err(
        "HDF5 support requires compiling with 'mdcath' feature.",
    ))
}

#[cfg(not(feature = "mdcath"))]
#[pyfunction]
fn parse_mdcath_metadata(_path: String) -> PyResult<PyObject> {
    Err(pyo3::exceptions::PyImportError::new_err(
        "HDF5 support requires compiling with 'mdcath' feature.",
    ))
}

#[cfg(not(feature = "mdcath"))]
#[pyfunction]
fn get_mdcath_replicas(
    _path: String,
    _domain_id: String,
    _temperature: String,
) -> PyResult<PyObject> {
    Err(pyo3::exceptions::PyImportError::new_err(
        "HDF5 support requires compiling with 'mdcath' feature.",
    ))
}

#[cfg(not(feature = "mdcath"))]
#[pyfunction]
#[pyo3(signature = (_path, _domain_id, _temperature, _replica, _frame_idx))]
fn parse_mdcath_frame(
    _path: String,
    _domain_id: String,
    _temperature: String,
    _replica: String,
    _frame_idx: usize,
) -> PyResult<PyObject> {
    Err(pyo3::exceptions::PyImportError::new_err(
        "HDF5 support requires compiling with 'mdcath' feature.",
    ))
}

/// Python module
#[pymodule]
fn priox_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();
    m.add_function(wrap_pyfunction!(parse_pdb, m)?)?;
    m.add_function(wrap_pyfunction!(parse_mmcif, m)?)?;
    m.add_function(wrap_pyfunction!(parse_structure, m)?)?;
    m.add_function(wrap_pyfunction!(load_forcefield, m)?)?;
    m.add_function(wrap_pyfunction!(assign_gaff_atom_types, m)?)?;
    m.add_function(wrap_pyfunction!(parse_xtc, m)?)?;
    m.add_function(wrap_pyfunction!(parse_dcd, m)?)?;
    m.add_function(wrap_pyfunction!(parse_trr, m)?)?;

    // HDF5 parsing functions
    m.add_function(wrap_pyfunction!(parse_mdtraj_h5_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(parse_mdtraj_h5_frame, m)?)?;
    m.add_function(wrap_pyfunction!(parse_mdcath_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(get_mdcath_replicas, m)?)?;
    m.add_function(wrap_pyfunction!(parse_mdcath_frame, m)?)?;

    m.add_class::<OutputSpec>()?;
    m.add_class::<CoordFormat>()?;
    m.add_class::<spec::ErrorMode>()?;
    m.add_class::<spec::MissingResidueMode>()?;

    // Atomic System Architecture
    m.add_class::<AtomicSystem>()?;

    Ok(())
}

/// Assign GAFF atom types to a structure (exposed to Python)
#[pyfunction]
fn assign_gaff_atom_types(
    py: Python<'_>,
    coordinates: PyObject,
    elements: Vec<String>,
) -> PyResult<Vec<Option<String>>> {
    let coords = extract_coords(py, &coordinates)?;

    // Default tolerance for bond inference
    let topology = forcefield::topology::Topology::from_coords(&coords, &elements, 1.3);

    let gaff = forcefield::gaff::GaffParameters::new();
    let types = forcefield::gaff::assign_gaff_types(&elements, &topology, &gaff);

    Ok(types)
}

fn extract_coords(py: Python<'_>, obj: &PyObject) -> PyResult<Vec<[f32; 3]>> {
    let bound = obj.bind(py);

    if let Ok(l) = bound.downcast::<pyo3::types::PyList>() {
        let mut coords = Vec::with_capacity(l.len());
        for item in l {
            let point: Vec<f32> = item.extract()?;
            if point.len() != 3 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Coordinates must be 3D points",
                ));
            }
            coords.push([point[0], point[1], point[2]]);
        }
        return Ok(coords);
    }

    if let Ok(array) = bound.downcast::<numpy::PyArray2<f32>>() {
        let binding = array.readonly();
        let data = binding.as_array();
        let shape = data.shape();
        if shape[1] != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Numpy array must be Nx3",
            ));
        }

        let mut coords = Vec::with_capacity(shape[0]);
        for i in 0..shape[0] {
            coords.push([data[[i, 0]], data[[i, 1]], data[[i, 2]]]);
        }
        return Ok(coords);
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "Expected list of lists or numpy array for coordinates",
    ))
}
