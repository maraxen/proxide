//! Core data structures for protein parsing
//!
//! This module defines Rust structures for raw atom data
//! matching biotite's AtomArray format.
//!
//! Note: Some utilities like with_capacity are for optional optimization.

#![allow(dead_code)]

use numpy::PyArray1;
use pyo3::prelude::*;

pub mod systems;
// Note: systems module is available but not re-exported to avoid unused import warnings
// Use structure::systems::* directly if needed

/// Raw atom data from PDB/mmCIF parsing
/// Matches biotite's AtomArray - variable-length, ALL atoms
#[derive(Clone, Debug)]
pub struct RawAtomData {
    /// Coordinates (N_atoms, 3) flattened
    pub coords: Vec<f32>,

    /// Atom names for each atom
    pub atom_names: Vec<String>,

    /// Element symbols for each atom
    pub elements: Vec<String>,

    /// Serial numbers from PDB
    pub serial_numbers: Vec<i32>,

    /// Alternate location indicators
    pub alt_locs: Vec<char>,

    /// Residue names (repeated per atom in that residue)
    pub res_names: Vec<String>,

    /// Residue sequence numbers (repeated per atom)
    pub res_ids: Vec<i32>,

    /// Insertion codes
    pub insertion_codes: Vec<char>,

    /// Chain IDs (as strings, repeated per atom)
    pub chain_ids: Vec<String>,

    /// Temperature factors (B-factors)
    pub b_factors: Vec<f32>,

    /// Occupancy values
    pub occupancy: Vec<f32>,

    /// Optional: charges (for PQR format or MD parameterization)
    pub charges: Option<Vec<f32>>,

    /// Optional: radii (for PQR format or GBSA)
    pub radii: Option<Vec<f32>>,

    /// Optional: LJ sigma parameters (from MD parameterization)
    pub sigmas: Option<Vec<f32>>,

    /// Optional: LJ epsilon parameters (from MD parameterization)
    pub epsilons: Option<Vec<f32>>,

    /// Total number of atoms
    pub num_atoms: usize,

    /// HETATM flag
    pub is_hetatm: Vec<bool>,
}

impl RawAtomData {
    /// Create a new empty RawAtomData
    pub fn new() -> Self {
        Self {
            coords: Vec::new(),
            atom_names: Vec::new(),
            elements: Vec::new(),
            serial_numbers: Vec::new(),
            alt_locs: Vec::new(),
            res_names: Vec::new(),
            res_ids: Vec::new(),
            insertion_codes: Vec::new(),
            chain_ids: Vec::new(),
            b_factors: Vec::new(),
            occupancy: Vec::new(),
            charges: None,
            radii: None,
            sigmas: None,
            epsilons: None,
            num_atoms: 0,
            is_hetatm: Vec::new(),
        }
    }

    /// Create with known capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            coords: Vec::with_capacity(capacity * 3),
            atom_names: Vec::with_capacity(capacity),
            elements: Vec::with_capacity(capacity),
            serial_numbers: Vec::with_capacity(capacity),
            alt_locs: Vec::with_capacity(capacity),
            res_names: Vec::with_capacity(capacity),
            res_ids: Vec::with_capacity(capacity),
            insertion_codes: Vec::with_capacity(capacity),
            chain_ids: Vec::with_capacity(capacity),
            b_factors: Vec::with_capacity(capacity),
            occupancy: Vec::with_capacity(capacity),
            charges: None,
            radii: None,
            sigmas: None,
            epsilons: None,
            num_atoms: 0,
            is_hetatm: Vec::with_capacity(capacity),
        }
    }

    /// Add an atom to the structure
    pub fn add_atom(&mut self, atom: AtomRecord) {
        self.coords.push(atom.x);
        self.coords.push(atom.y);
        self.coords.push(atom.z);
        self.atom_names.push(atom.atom_name);
        self.elements.push(atom.element);
        self.serial_numbers.push(atom.serial);
        self.alt_locs.push(atom.alt_loc);
        self.res_names.push(atom.res_name);
        self.res_ids.push(atom.res_seq);
        self.insertion_codes.push(atom.i_code);
        self.chain_ids.push(atom.chain_id);
        self.b_factors.push(atom.temp_factor);
        self.occupancy.push(atom.occupancy);
        self.is_hetatm.push(atom.is_hetatm);

        // Handle optional fields
        if let Some(charge) = atom.charge {
            self.charges
                .get_or_insert_with(|| Vec::with_capacity(self.num_atoms + 1))
                .push(charge);
        }

        if let Some(radius) = atom.radius {
            self.radii
                .get_or_insert_with(|| Vec::with_capacity(self.num_atoms + 1))
                .push(radius);
        }

        self.num_atoms += 1;
    }

    /// Convert to Python dictionary with NumPy arrays
    pub fn to_py_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let dict = pyo3::types::PyDict::new_bound(py);

        // Coordinates as flat array
        dict.set_item("coords", PyArray1::from_slice_bound(py, &self.coords))?;
        dict.set_item("num_atoms", self.num_atoms)?;

        // String fields as Python lists
        let atom_names_list: Vec<&str> = self.atom_names.iter().map(|s| s.as_str()).collect();
        dict.set_item("atom_names", atom_names_list)?;

        let elements_list: Vec<&str> = self.elements.iter().map(|s| s.as_str()).collect();
        dict.set_item("elements", elements_list)?;

        let res_names_list: Vec<&str> = self.res_names.iter().map(|s| s.as_str()).collect();
        dict.set_item("res_names", res_names_list)?;

        let chain_ids_list: Vec<&str> = self.chain_ids.iter().map(|s| s.as_str()).collect();
        dict.set_item("chain_ids", chain_ids_list)?;

        // Numeric arrays
        dict.set_item(
            "serial_numbers",
            PyArray1::from_slice_bound(py, &self.serial_numbers),
        )?;
        dict.set_item("res_ids", PyArray1::from_slice_bound(py, &self.res_ids))?;
        dict.set_item("b_factors", PyArray1::from_slice_bound(py, &self.b_factors))?;
        dict.set_item("occupancy", PyArray1::from_slice_bound(py, &self.occupancy))?;

        // Character fields as strings
        let alt_locs_list: Vec<String> = self.alt_locs.iter().map(|c| c.to_string()).collect();
        dict.set_item("alt_locs", alt_locs_list)?;

        let insertion_codes_list: Vec<String> =
            self.insertion_codes.iter().map(|c| c.to_string()).collect();
        dict.set_item("insertion_codes", insertion_codes_list)?;

        // Optional fields
        if let Some(ref charges) = self.charges {
            dict.set_item("charges", PyArray1::from_slice_bound(py, charges))?;
        }

        if let Some(ref radii) = self.radii {
            dict.set_item("radii", PyArray1::from_slice_bound(py, radii))?;
        }

        // Return as AtomicSystem compatible dict
        Ok(dict.into_any())
    }
}

impl Default for RawAtomData {
    fn default() -> Self {
        Self::new()
    }
}

/// Atom record from PDB/mmCIF parsing
#[derive(Clone, Debug)]
pub struct AtomRecord {
    pub serial: i32,
    pub atom_name: String,
    pub alt_loc: char,
    pub res_name: String,
    pub chain_id: String,
    pub res_seq: i32,
    pub i_code: char,
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub occupancy: f32,
    pub temp_factor: f32,
    pub element: String,
    pub charge: Option<f32>,
    pub radius: Option<f32>,
    pub is_hetatm: bool,
}

impl Default for AtomRecord {
    fn default() -> Self {
        Self {
            serial: 0,
            atom_name: String::new(),
            alt_loc: ' ',
            res_name: String::new(),
            chain_id: String::new(),
            res_seq: 0,
            i_code: ' ',
            x: 0.0,
            y: 0.0,
            z: 0.0,
            occupancy: 1.0,
            temp_factor: 0.0,
            element: String::new(),
            charge: None,
            radius: None,
            is_hetatm: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_raw_atom_data_creation() {
        let data = RawAtomData::new();
        assert_eq!(data.num_atoms, 0);
        assert!(data.coords.is_empty());
        assert!(data.atom_names.is_empty());
    }

    #[test]
    fn test_add_atom() {
        let mut data = RawAtomData::new();
        let atom = AtomRecord {
            serial: 1,
            atom_name: "CA".to_string(),
            alt_loc: ' ',
            res_name: "ALA".to_string(),
            chain_id: "A".to_string(),
            res_seq: 1,
            i_code: ' ',
            x: 1.0,
            y: 2.0,
            z: 3.0,
            occupancy: 1.0,
            temp_factor: 20.0,
            element: "C".to_string(),
            charge: None,
            radius: None,
            is_hetatm: false,
        };

        data.add_atom(atom);

        assert_eq!(data.num_atoms, 1);
        assert_eq!(data.coords.len(), 3);
        assert_eq!(data.atom_names[0], "CA");
        assert_eq!(data.res_names[0], "ALA");
    }
}
