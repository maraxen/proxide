//! Atomic System Architecture
//! Matches Python's atomic_system.py structure

use pyo3::prelude::*;

/// Base class for any atomic system
#[pyclass]
#[derive(Debug, Clone)]
pub struct AtomicSystem {
    /// Flattened coordinates (N_atoms * 3)
    pub coordinates: Vec<f32>,
    /// Atom mask (N_atoms)
    pub atom_mask: Vec<f32>,
    /// Atom names
    pub atom_names: Vec<String>,
    /// Element symbols
    pub elements: Vec<String>,

    /// Optional MD parameters
    pub charges: Option<Vec<f32>>,
    pub sigmas: Option<Vec<f32>>,
    pub epsilons: Option<Vec<f32>>,
    pub radii: Option<Vec<f32>>,

    pub num_atoms: usize,
}

#[pymethods]
impl AtomicSystem {
    #[new]
    pub fn new(
        coordinates: Vec<f32>,
        atom_mask: Vec<f32>,
        atom_names: Vec<String>,
        elements: Vec<String>,
    ) -> Self {
        let num_atoms = atom_mask.len();
        Self {
            coordinates,
            atom_mask,
            atom_names,
            elements,
            charges: None,
            sigmas: None,
            epsilons: None,
            radii: None,
            num_atoms,
        }
    }
}
