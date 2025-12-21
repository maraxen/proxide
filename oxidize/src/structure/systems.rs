//! Atomic System Architecture
//! Matches Python's atomic_system.py structure

use numpy::PyArrayMethods;
use pyo3::prelude::*;
use rand::prelude::*;
use rand_distr::{Distribution, Normal};

use crate::{geometry, physics};

/// Base class for any atomic system
#[pyclass]
#[derive(Debug, Clone)]
pub struct AtomicSystem {
    /// Flattened coordinates (N_atoms * 3)
    #[pyo3(get, set)]
    pub coordinates: Vec<f32>,
    /// Atom mask (N_atoms)
    #[pyo3(get, set)]
    pub atom_mask: Vec<f32>,
    /// Atom names
    #[pyo3(get, set)]
    pub atom_names: Vec<String>,
    /// Element symbols
    #[pyo3(get, set)]
    pub elements: Vec<String>,

    /// Topology
    #[pyo3(get, set)]
    pub bonds: Option<Vec<[usize; 2]>>,
    #[pyo3(get, set)]
    pub angles: Option<Vec<[usize; 3]>>,
    #[pyo3(get, set)]
    pub proper_dihedrals: Option<Vec<[usize; 4]>>,
    #[pyo3(get, set)]
    pub impropers: Option<Vec<[usize; 4]>>,

    /// Optional MD parameters
    #[pyo3(get, set)]
    pub charges: Option<Vec<f32>>,
    #[pyo3(get, set)]
    pub sigmas: Option<Vec<f32>>,
    #[pyo3(get, set)]
    pub epsilons: Option<Vec<f32>>,
    #[pyo3(get, set)]
    pub radii: Option<Vec<f32>>,

    #[pyo3(get, set)]
    pub residue_index: Option<Vec<i32>>,
    #[pyo3(get, set)]
    pub chain_index: Option<Vec<i32>>,
    #[pyo3(get, set)]
    pub unique_chain_ids: Option<Vec<String>>,

    /// Features
    #[pyo3(get, set)]
    pub neighbor_indices: Option<Vec<i32>>, // Flattened (N_res, K)
    #[pyo3(get, set)]
    pub rbf_features: Option<Vec<f32>>, // Flattened (N_res, K, D)
    #[pyo3(get, set)]
    pub rbf_num_neighbors: Option<usize>,
    #[pyo3(get, set)]
    pub vdw_features: Option<Vec<f32>>,
    #[pyo3(get, set)]
    pub electrostatic_features: Option<Vec<f32>>,

    #[pyo3(get, set)]
    pub num_atoms: usize,
}

#[pymethods]
impl AtomicSystem {
    #[new]
    #[pyo3(signature = (coordinates, atom_mask, atom_names=None, elements=None))]
    pub fn new(
        coordinates: Vec<f32>,
        atom_mask: Vec<f32>,
        atom_names: Option<Vec<String>>,
        elements: Option<Vec<String>>,
    ) -> Self {
        let num_atoms = atom_mask.len();
        Self {
            coordinates,
            atom_mask,
            atom_names: atom_names.unwrap_or_default(),
            elements: elements.unwrap_or_default(),
            bonds: None,
            angles: None,
            proper_dihedrals: None,
            impropers: None,
            charges: None,
            sigmas: None,
            epsilons: None,
            radii: None,
            residue_index: None,
            chain_index: None,
            unique_chain_ids: None,
            neighbor_indices: None,
            rbf_features: None,
            rbf_num_neighbors: None,
            vdw_features: None,
            electrostatic_features: None,
            num_atoms,
        }
    }

    /// Update coordinates with Gaussian noise and recompute features
    /// Returns a new AtomicSystem with updated coordinates and features.
    #[pyo3(signature = (sigma, seed=0))]
    pub fn update_with_noise(&self, sigma: f32, seed: u64) -> PyResult<Self> {
        let mut new_system = self.clone();

        // 1. Apply Noise
        let mut rng = StdRng::seed_from_u64(seed);
        let normal = Normal::new(0.0, sigma).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Invalid sigma: {}", e))
        })?;

        for x in new_system.coordinates.iter_mut() {
            *x += normal.sample(&mut rng);
        }

        // 2. Recompute Features
        new_system.recompute_features()?;

        Ok(new_system)
    }

    /// Update coordinates from explicit array and recompute features
    #[pyo3(signature = (new_coords))]
    pub fn update_coordinates(&self, new_coords: Vec<f32>) -> PyResult<Self> {
        let mut new_system = self.clone();
        if new_coords.len() != self.coordinates.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Coordinate shape mismatch",
            ));
        }
        new_system.coordinates = new_coords;

        // Reuse the recompute logic?
        // Since I implemented it in update_with_noise, I should refactor.
        // But for now, I'll call a private helper or just duplicate the call logic (calling update_with_noise with 0 sigma is wasteful of RNG).
        // Better: Refactor recompute_features into a private method.

        new_system.recompute_features()?;
        Ok(new_system)
    }

    /// Convert to a Python dictionary (compatible with Protein.from_rust_dict)
    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let dict = pyo3::types::PyDict::new_bound(py);

        dict.set_item(
            "coordinates",
            numpy::PyArray1::from_slice_bound(py, &self.coordinates),
        )?;
        dict.set_item(
            "atom_mask",
            numpy::PyArray1::from_slice_bound(py, &self.atom_mask),
        )?;
        dict.set_item("atom_names", &self.atom_names)?;
        dict.set_item("elements", &self.elements)?;

        if let Some(ref bonds) = self.bonds {
            let flat: Vec<usize> = bonds.iter().flatten().copied().collect();
            let arr = numpy::PyArray1::from_slice_bound(py, &flat);
            dict.set_item("bonds", arr.reshape((bonds.len(), 2))?)?;
        }

        if let Some(ref charges) = self.charges {
            dict.set_item("charges", numpy::PyArray1::from_slice_bound(py, charges))?;
        }

        // ... more fields could be added here if needed

        if let Some(ref res_idx) = self.residue_index {
            dict.set_item(
                "residue_index",
                numpy::PyArray1::from_slice_bound(py, res_idx),
            )?;
        }
        if let Some(ref chain_idx) = self.chain_index {
            dict.set_item(
                "chain_index",
                numpy::PyArray1::from_slice_bound(py, chain_idx),
            )?;
        }
        // unique_chain_ids is list of string, so PyList
        if let Some(ref u_chains) = self.unique_chain_ids {
            dict.set_item("unique_chain_ids", u_chains)?;
        }

        if let Some(ref idx) = self.neighbor_indices {
            let k = self.rbf_num_neighbors.unwrap_or(30);
            let n = idx.len() / k;
            let arr = numpy::PyArray1::from_slice_bound(py, idx);
            dict.set_item("neighbor_indices", arr.reshape((n, k))?)?;
        }
        if let Some(ref rbf) = self.rbf_features {
            // RBF is (N, K, D)
            // D is 400 normally (radial_basis checks this)
            // But we flat stored it.
            let k = self.rbf_num_neighbors.unwrap_or(30);

            // Usually D=400 (from `spec.rs` or `radial_basis.rs`).
            // Let's assume D=400? Or extract from size?
            // Size = N * K * D.
            // We know neighbors.len() which is N*K.
            let n_k = if let Some(ref nidx) = self.neighbor_indices {
                nidx.len()
            } else {
                0
            };
            if n_k > 0 {
                let d = rbf.len() / n_k;
                let n = n_k / k;
                let arr = numpy::PyArray1::from_slice_bound(py, rbf);
                dict.set_item("rbf_features", arr.reshape((n, k, d))?)?;
            }
        }
        if let Some(ref vdw) = self.vdw_features {
            let n_res = vdw.len() / 5;
            let arr = numpy::PyArray1::from_slice_bound(py, vdw);
            dict.set_item("vdw_features", arr.reshape((n_res, 5))?)?;
        }
        if let Some(ref elec) = self.electrostatic_features {
            let n_res = elec.len() / 5;
            let arr = numpy::PyArray1::from_slice_bound(py, elec);
            dict.set_item("electrostatic_features", arr.reshape((n_res, 5))?)?;
        }

        Ok(dict)
    }
}

impl AtomicSystem {
    // Helper to map property to backbone
    fn extract_backbone_map(&self, map: &[[Option<usize>; 5]], data: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0; map.len() * 5];
        for (r, atoms) in map.iter().enumerate() {
            for (i, atom_idx) in atoms.iter().enumerate() {
                if let Some(idx) = atom_idx {
                    if *idx < data.len() {
                        out[r * 5 + i] = data[*idx];
                    }
                }
            }
        }
        out
    }

    fn recompute_features(&mut self) -> PyResult<()> {
        // Shared logic refactor from update_with_noise
        // ... (Omitting full implementation here to save tokens, assuming update_with_noise covers it or I should refactor properly)
        // Actually, I can't easily refactor into a method I haven't written yet.
        // I will copy-paste or structure it so update_with_noise calls this.

        if (self.rbf_features.is_some()
            || self.vdw_features.is_some()
            || self.electrostatic_features.is_some())
            && self.residue_index.is_some()
        {
            // ... Same logic as above ...
            // Since I can't put the logic twice in one replacement block easily without bloating,
            // I will implement recompute_features fully and have update_with_noise call it.
            let residue_index = self.residue_index.as_ref().unwrap();
            let num_residues = if residue_index.is_empty() {
                0
            } else {
                (*residue_index.iter().max().unwrap_or(&-1) + 1) as usize
            };

            if num_residues > 0 {
                let mut backbone_map = vec![[None; 5]; num_residues];
                for (i, &res_idx) in residue_index.iter().enumerate() {
                    let res_idx = res_idx as usize;
                    if res_idx < num_residues {
                        let name = &self.atom_names[i];
                        match name.as_str() {
                            "N" => backbone_map[res_idx][0] = Some(i),
                            "CA" => backbone_map[res_idx][1] = Some(i),
                            "C" => backbone_map[res_idx][2] = Some(i),
                            "CB" => backbone_map[res_idx][3] = Some(i),
                            "O" => backbone_map[res_idx][4] = Some(i),
                            _ => {}
                        }
                    }
                }

                let mut backbone_coords = vec![[[f32::NAN; 3]; 5]; num_residues];
                let mut ca_coords = vec![[f32::NAN; 3]; num_residues];
                for r in 0..num_residues {
                    for atom_type in 0..5 {
                        if let Some(idx) = backbone_map[r][atom_type] {
                            let range = idx * 3..idx * 3 + 3;
                            let c = &self.coordinates[range];
                            backbone_coords[r][atom_type] = [c[0], c[1], c[2]];
                            if atom_type == 1 {
                                ca_coords[r] = [c[0], c[1], c[2]];
                            }
                        }
                    }
                    if backbone_map[r][3].is_none()
                        && !backbone_coords[r][0][0].is_nan()
                        && !backbone_coords[r][1][0].is_nan()
                        && !backbone_coords[r][2][0].is_nan()
                    {
                        backbone_coords[r][3] = physics::frame::compute_c_beta(
                            backbone_coords[r][0],
                            backbone_coords[r][1],
                            backbone_coords[r][2],
                        );
                    }
                }

                if self.rbf_features.is_some() {
                    let k = self.rbf_num_neighbors.unwrap_or(30);
                    let neighbors = geometry::neighbors::find_k_nearest_neighbors(&ca_coords, k);
                    let mut flat_neighbors = vec![-1i32; num_residues * k];
                    for (i, nlist) in neighbors.iter().enumerate() {
                        for (j, &nidx) in nlist.iter().enumerate() {
                            if j < k {
                                flat_neighbors[i * k + j] = nidx as i32;
                            }
                        }
                    }
                    self.neighbor_indices = Some(flat_neighbors);
                    let rbf = geometry::radial_basis::compute_radial_basis_with_shape(
                        &backbone_coords,
                        &neighbors,
                    );
                    self.rbf_features = Some(rbf.features);
                }

                let all_coords: Vec<[f32; 3]> = self
                    .coordinates
                    .chunks(3)
                    .map(|c| [c[0], c[1], c[2]])
                    .collect();

                if self.electrostatic_features.is_some() && self.charges.is_some() {
                    let backbone_charges =
                        self.extract_backbone_map(&backbone_map, self.charges.as_ref().unwrap());
                    if let Some(charges) = self.charges.as_ref() {
                        let forces = physics::electrostatics::compute_coulomb_forces_at_backbone(
                            &backbone_coords,
                            &all_coords,
                            &backbone_charges,
                            charges,
                        );
                        self.electrostatic_features = Some(
                            physics::frame::project_backbone_forces(&forces, &backbone_coords),
                        );
                    }
                }

                if self.vdw_features.is_some() {
                    let n = self.num_atoms;
                    let (all_sigmas, all_epsilons) =
                        if let (Some(s), Some(e)) = (&self.sigmas, &self.epsilons) {
                            (s.clone(), e.clone())
                        } else {
                            (
                                vec![physics::constants::DEFAULT_SIGMA; n],
                                vec![physics::constants::DEFAULT_EPSILON; n],
                            )
                        };
                    let backbone_sigmas = self.extract_backbone_map(&backbone_map, &all_sigmas);
                    let backbone_epsilons = self.extract_backbone_map(&backbone_map, &all_epsilons);
                    let forces = physics::vdw::compute_lj_forces_at_backbone(
                        &backbone_coords,
                        &all_coords,
                        &backbone_sigmas,
                        &backbone_epsilons,
                        &all_sigmas,
                        &all_epsilons,
                    );
                    self.vdw_features = Some(physics::frame::project_backbone_forces(
                        &forces,
                        &backbone_coords,
                    ));
                }
            }
        }
        Ok(())
    }
}
