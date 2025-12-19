//! Atomic System Architecture
//! Matches Python's atomic_system.py structure

use numpy::PyArrayMethods;
use pyo3::prelude::*;

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
            num_atoms,
        }
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

        Ok(dict)
    }
}
