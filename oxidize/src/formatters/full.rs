//! Full coordinate formatter
//!
//! Converts ProcessedStructure into padded (N_res, max_atoms, 3) arrays
//! that include ALL atoms, not just the standard 37.
//!
//! This format is useful for workflows that need access to all atoms
//! (e.g., ligands, non-standard residues, hydrogens).

use crate::processing::ProcessedStructure;
use crate::spec::OutputSpec;

/// Maximum atoms per residue for padding
/// Set to 27 which covers most amino acids with hydrogens
const MAX_ATOMS_PER_RESIDUE: usize = 27;

/// Formatted structure with all atoms (padded representation)
#[derive(Debug)]
pub struct FormattedFull {
    /// Coordinates: (N_res * max_atoms * 3) flattened
    pub coordinates: Vec<f32>,
    /// Atom mask: (N_res * max_atoms) - 1.0 for present atoms
    pub atom_mask: Vec<f32>,
    /// Atom names per residue: (N_res * max_atoms) - empty string for absent
    pub atom_names: Vec<String>,
    /// Residue type indices: (N_res)
    pub aatype: Vec<i8>,
    /// Residue indices: (N_res) - PDB numbering
    pub residue_index: Vec<i32>,
    /// Chain indices: (N_res)
    pub chain_index: Vec<i32>,
    /// Shape info: (num_residues, max_atoms_in_any_residue, 3)
    pub coord_shape: (usize, usize, usize),
}

/// Full formatter - returns all atoms as flat array
///
/// Unlike Atom37/Atom14, this outputs ALL atoms including hydrogens
/// as a flat (N_atoms, 3) array.
pub struct FullFormatter;

impl FullFormatter {
    /// Format a ProcessedStructure into full atom representation
    ///
    /// Returns a flat (N_atoms, 3) coordinate array containing all atoms
    /// including any hydrogens added by add_hydrogens().
    pub fn format(
        processed: &ProcessedStructure,
        _spec: &OutputSpec,
    ) -> Result<FormattedFull, String> {
        let num_atoms = processed.raw_atoms.num_atoms;
        let num_residues = processed.num_residues;

        log::debug!(
            "Formatting Full for {} atoms ({} residues)",
            num_atoms,
            num_residues
        );

        // Output is flat (N_atoms, 3) - directly copy all coords
        let coordinates = processed.raw_atoms.coords.clone();
        let atom_mask = vec![1.0f32; num_atoms]; // All atoms are valid
        let atom_names = processed.raw_atoms.atom_names.clone();

        // Build aatype from residue info (need to map atoms to residues)
        let mut aatype = vec![0i8; num_residues];
        for (res_idx, res_info) in processed.residue_info.iter().enumerate() {
            aatype[res_idx] = res_info.res_type as i8;
        }

        // Residue indices
        let residue_index: Vec<i32> = processed.residue_info.iter().map(|r| r.res_id).collect();

        // Chain indices
        let chain_index: Vec<i32> = processed
            .residue_info
            .iter()
            .map(|r| *processed.chain_indices.get(&r.chain_id).unwrap_or(&0) as i32)
            .collect();

        Ok(FormattedFull {
            coordinates,
            atom_mask,
            atom_names,
            aatype,
            residue_index,
            chain_index,
            coord_shape: (num_atoms, 3, 1), // Flat shape indicator
        })
    }
}

impl FormattedFull {
    /// Convert to Python dictionary
    pub fn to_py_dict(&self, py: pyo3::Python) -> pyo3::PyResult<pyo3::PyObject> {
        use numpy::PyArray1;
        use pyo3::prelude::*;
        use pyo3::types::PyDict;

        let dict = PyDict::new_bound(py);

        // Convert to NumPy arrays
        dict.set_item(
            "coordinates",
            PyArray1::from_vec_bound(py, self.coordinates.clone()),
        )?;
        dict.set_item(
            "atom_mask",
            PyArray1::from_vec_bound(py, self.atom_mask.clone()),
        )?;
        dict.set_item("aatype", PyArray1::from_vec_bound(py, self.aatype.clone()))?;
        dict.set_item(
            "residue_index",
            PyArray1::from_vec_bound(py, self.residue_index.clone()),
        )?;
        dict.set_item(
            "chain_index",
            PyArray1::from_vec_bound(py, self.chain_index.clone()),
        )?;

        // Atom names as Python list
        let atom_names_list: Vec<&str> = self.atom_names.iter().map(|s| s.as_str()).collect();
        dict.set_item("atom_names", atom_names_list)?;

        // Shape info
        dict.set_item(
            "coord_shape",
            (self.coord_shape.0, self.coord_shape.1, self.coord_shape.2),
        )?;

        Ok(dict.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::OutputSpec;
    use crate::structure::{AtomRecord, RawAtomData};

    fn create_test_spec() -> OutputSpec {
        OutputSpec::default()
    }

    #[test]
    fn test_full_single_residue() {
        // Create ALA with 5 atoms
        let mut raw = RawAtomData::with_capacity(5);

        let atoms = vec![
            ("N", 0.0, 0.0, 0.0),
            ("CA", 1.0, 0.0, 0.0),
            ("C", 2.0, 0.0, 0.0),
            ("CB", 1.0, 1.0, 0.0),
            ("O", 3.0, 0.0, 0.0),
        ];

        for (i, (name, x, y, z)) in atoms.iter().enumerate() {
            raw.add_atom(AtomRecord {
                serial: i as i32 + 1,
                atom_name: name.to_string(),
                alt_loc: ' ',
                res_name: "ALA".to_string(),
                chain_id: "A".to_string(),
                res_seq: 1,
                i_code: ' ',
                x: *x,
                y: *y,
                z: *z,
                occupancy: 1.0,
                temp_factor: 20.0,
                element: "C".to_string(),
                charge: None,
                radius: None,
                is_hetatm: false,
            });
        }

        let processed = ProcessedStructure::from_raw(raw).unwrap();
        let spec = create_test_spec();
        let formatted = FullFormatter::format(&processed, &spec).unwrap();

        // Verify dimensions
        assert_eq!(formatted.coord_shape.0, 1); // 1 residue
        assert!(formatted.coord_shape.1 >= 5); // at least 5 atoms
        assert_eq!(formatted.coord_shape.2, 3); // 3D

        // Check aatype (ALA = 0)
        assert_eq!(formatted.aatype[0], 0);

        // Check that first 5 atoms have mask = 1
        for i in 0..5 {
            assert_eq!(formatted.atom_mask[i], 1.0, "Atom {} should be present", i);
        }

        // Check padding has mask = 0
        for i in 5..formatted.coord_shape.1 {
            assert_eq!(formatted.atom_mask[i], 0.0, "Atom {} should be padding", i);
        }

        // Check atom names
        assert_eq!(formatted.atom_names[0], "N");
        assert_eq!(formatted.atom_names[1], "CA");
    }

    #[test]
    fn test_full_multi_residue() {
        let mut raw = RawAtomData::with_capacity(9);

        // Residue 1: 5 atoms
        for (i, name) in ["N", "CA", "C", "CB", "O"].iter().enumerate() {
            raw.add_atom(AtomRecord {
                serial: i as i32 + 1,
                atom_name: name.to_string(),
                alt_loc: ' ',
                res_name: "ALA".to_string(),
                chain_id: "A".to_string(),
                res_seq: 1,
                i_code: ' ',
                x: i as f32,
                y: 0.0,
                z: 0.0,
                occupancy: 1.0,
                temp_factor: 20.0,
                element: "C".to_string(),
                charge: None,
                radius: None,
                is_hetatm: false,
            });
        }

        // Residue 2: 4 atoms (GLY, no CB)
        for (i, name) in ["N", "CA", "C", "O"].iter().enumerate() {
            raw.add_atom(AtomRecord {
                serial: (i + 5) as i32 + 1,
                atom_name: name.to_string(),
                alt_loc: ' ',
                res_name: "GLY".to_string(),
                chain_id: "A".to_string(),
                res_seq: 2,
                i_code: ' ',
                x: (i + 10) as f32,
                y: 0.0,
                z: 0.0,
                occupancy: 1.0,
                temp_factor: 20.0,
                element: "C".to_string(),
                charge: None,
                radius: None,
                is_hetatm: false,
            });
        }

        let processed = ProcessedStructure::from_raw(raw).unwrap();
        let spec = create_test_spec();
        let formatted = FullFormatter::format(&processed, &spec).unwrap();

        // Should have 2 residues
        assert_eq!(formatted.coord_shape.0, 2);
        assert_eq!(formatted.aatype.len(), 2);

        // Check residue types
        assert_eq!(formatted.aatype[0], 0); // ALA
        assert_eq!(formatted.aatype[1], 7); // GLY

        // First residue: 5 atoms present
        let max_atoms = formatted.coord_shape.1;
        for i in 0..5 {
            assert_eq!(formatted.atom_mask[i], 1.0);
        }

        // Second residue: 4 atoms present
        for i in 0..4 {
            assert_eq!(formatted.atom_mask[max_atoms + i], 1.0);
        }
        // 5th position should be padding
        assert_eq!(formatted.atom_mask[max_atoms + 4], 0.0);
    }
}
