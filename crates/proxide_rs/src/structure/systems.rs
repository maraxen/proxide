// TODO: Review allow attributes at a later point
#![allow(clippy::useless_conversion)]

//! Atomic System Architecture

use rand::prelude::*;
use rand_distr::{Distribution, Normal};

use crate::{geometry, physics};

#[derive(Debug, Clone)]
pub struct AtomicSystem {
    pub coordinates: Vec<f32>,
    pub atom_mask: Vec<f32>,
    pub atom_names: Vec<String>,
    pub elements: Vec<String>,

    pub bonds: Option<Vec<[usize; 2]>>,
    pub angles: Option<Vec<[usize; 3]>>,
    pub proper_dihedrals: Option<Vec<[usize; 4]>>,
    pub impropers: Option<Vec<[usize; 4]>>,

    pub charges: Option<Vec<f32>>,
    pub sigmas: Option<Vec<f32>>,
    pub epsilons: Option<Vec<f32>>,
    pub radii: Option<Vec<f32>>,

    pub residue_index: Option<Vec<i32>>,
    pub chain_index: Option<Vec<i32>>,
    pub unique_chain_ids: Option<Vec<String>>,

    pub neighbor_indices: Option<Vec<i32>>,
    pub rbf_features: Option<Vec<f32>>,
    pub rbf_num_neighbors: Option<usize>,
    pub vdw_features: Option<Vec<f32>>,
    pub electrostatic_features: Option<Vec<f32>>,

    pub num_atoms: usize,
}

impl AtomicSystem {
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

    pub fn update_with_noise(&mut self, sigma: f32, seed: u64) -> Result<(), String> {
        let mut rng = StdRng::seed_from_u64(seed);
        let normal = Normal::new(0.0, sigma).map_err(|e| e.to_string())?;

        for x in self.coordinates.iter_mut() {
            *x += normal.sample(&mut rng);
        }

        self.recompute_features().map_err(|e| e.to_string())
    }

    pub fn update_coordinates(&mut self, new_coords: Vec<f32>) -> Result<(), String> {
        if new_coords.len() != self.coordinates.len() {
            return Err("Coordinate shape mismatch".to_string());
        }
        self.coordinates = new_coords;
        self.recompute_features().map_err(|e| e.to_string())
    }

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

    fn recompute_features(&mut self) -> Result<(), std::io::Error> {
        if (self.rbf_features.is_some()
            || self.vdw_features.is_some()
            || self.electrostatic_features.is_some())
            && self.residue_index.is_some()
        {
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
