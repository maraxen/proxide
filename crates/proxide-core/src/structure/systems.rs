// TODO: Review allow attributes at a later point
#![allow(clippy::useless_conversion)]

//! Atomic System Architecture

use rand::prelude::*;
use rand_distr::{Distribution, Normal};

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

    /// Update coordinates with noise (internal state only)
    pub fn update_with_noise(&mut self, sigma: f32, seed: u64) -> Result<(), String> {
        let mut rng = StdRng::seed_from_u64(seed);
        let normal = Normal::new(0.0, sigma).map_err(|e| e.to_string())?;

        for x in self.coordinates.iter_mut() {
            *x += normal.sample(&mut rng);
        }
        Ok(())
    }

    /// Update coordinates (internal state only)
    pub fn update_coordinates(&mut self, new_coords: Vec<f32>) -> Result<(), String> {
        if new_coords.len() != self.coordinates.len() {
            return Err("Coordinate shape mismatch".to_string());
        }
        self.coordinates = new_coords;
        Ok(())
    }
}
