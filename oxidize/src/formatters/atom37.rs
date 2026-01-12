//! Atom37 coordinate formatter
//!
//! Converts ProcessedStructure into standardized (N_res, 37, 3) coordinate arrays
//! compatible with AlphaFold and ProteinTuple conventions.

use crate::chem::{build_atom_order, build_standard_atom_mask};
use crate::processing::ProcessedStructure;
use crate::spec::OutputSpec;
use std::collections::HashMap;

/// Formatted structure in Atom37 representation
#[derive(Debug)]
pub struct FormattedAtom37 {
    pub coordinates: Vec<f32>,   // Flat (N_res * 37 * 3) - reshaped in Python
    pub atom_mask: Vec<f32>,     // Flat (N_res * 37)
    pub aatype: Vec<i8>,         // (N_res,) residue type indices (0-20)
    pub residue_index: Vec<i32>, // (N_res,) PDB residue numbers
    pub chain_index: Vec<i32>,   // (N_res,) chain indices
}

/// Atom37 formatter - converts to standardized (N_res, 37, 3) representation
pub struct Atom37Formatter;

impl Atom37Formatter {
    /// Format a ProcessedStructure into Atom37 representation
    pub fn format(
        processed: &ProcessedStructure,
        _spec: &OutputSpec,
    ) -> Result<FormattedAtom37, String> {
        let num_residues = processed.num_residues;
        let atom_order = build_atom_order();
        let standard_mask = build_standard_atom_mask();

        log::debug!("Formatting Atom37 for {} residues", num_residues);

        // Pre-allocate output arrays
        let mut coordinates = vec![0.0f32; num_residues * 37 * 3];
        let mut atom_mask = vec![0.0f32; num_residues * 37];
        let mut aatype = vec![0i8; num_residues];
        let mut residue_index = vec![0i32; num_residues];
        let mut chain_index = vec![0i32; num_residues];

        // Process each residue
        for (res_idx, res_info) in processed.residue_info.iter().enumerate() {
            // Set residue metadata
            aatype[res_idx] = res_info.res_type as i8;
            residue_index[res_idx] = res_info.res_id;

            // Map chain to index
            chain_index[res_idx] = *processed
                .chain_indices
                .get(&res_info.chain_id)
                .unwrap_or(&0) as i32;

            // Build atom name -> local index mapping for this residue
            let mut residue_atoms: HashMap<String, usize> = HashMap::new();
            let start = res_info.start_atom;
            let end = start + res_info.num_atoms;

            for local_idx in 0..(end - start) {
                let global_idx = start + local_idx;
                let atom_name = &processed.raw_atoms.atom_names[global_idx];
                residue_atoms.insert(atom_name.clone(), global_idx);
            }

            // Fill in atom37 positions
            for (atom_name, &atom_type_idx) in atom_order.iter() {
                let coord_base = (res_idx * 37 + atom_type_idx) * 3;
                let mask_idx = res_idx * 37 + atom_type_idx;

                // Check if this atom type is expected for this residue type
                let is_expected = if res_info.res_type < standard_mask.len() {
                    standard_mask[res_info.res_type][atom_type_idx] == 1
                } else {
                    false // Unknown residue type
                };

                if let Some(&global_idx) = residue_atoms.get(atom_name) {
                    // Atom is present - copy coordinates
                    coordinates[coord_base] = processed.raw_atoms.coords[global_idx * 3];
                    coordinates[coord_base + 1] = processed.raw_atoms.coords[global_idx * 3 + 1];
                    coordinates[coord_base + 2] = processed.raw_atoms.coords[global_idx * 3 + 2];
                    atom_mask[mask_idx] = 1.0;
                } else if is_expected {
                    // Expected atom is missing - log warning
                    log::warn!(
                        "Missing expected atom {} in residue {} {} (chain {})",
                        atom_name,
                        res_info.res_name,
                        res_info.res_id,
                        res_info.chain_id
                    );
                    // Coordinates already initialized to 0.0, mask already 0.0
                } else {
                    // Atom not expected for this residue type - leave as zeros
                    // No warning needed
                }
            }
        }

        Ok(FormattedAtom37 {
            coordinates,
            atom_mask,
            aatype,
            residue_index,
            chain_index,
        })
    }
}

impl FormattedAtom37 {
    /// Convert to Python dictionary
    pub fn to_py_dict(&self, py: pyo3::Python) -> pyo3::PyResult<pyo3::PyObject> {
        use numpy::PyArray1;
        use pyo3::prelude::*;
        use pyo3::types::PyDict;

        let dict = PyDict::new_bound(py);

        // Convert to NumPy arrays (zero-copy when possible)
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

        Ok(dict.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chem::UNK_RESTYPE_INDEX;
    use crate::spec::OutputSpec;
    use crate::structure::{AtomRecord, RawAtomData};

    fn create_test_spec() -> OutputSpec {
        OutputSpec::default()
    }

    #[test]
    fn test_atom37_single_residue_ala() {
        // Create minimal ALA residue with N, CA, C, CB, O
        let mut raw = RawAtomData::with_capacity(5);

        let atoms = [
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
        let formatted = Atom37Formatter::format(&processed, &spec).unwrap();

        // Verify dimensions
        assert_eq!(formatted.coordinates.len(), 37 * 3);
        assert_eq!(formatted.atom_mask.len(), 37);
        assert_eq!(formatted.aatype.len(), 1);

        // Check aatype (ALA = 0)
        assert_eq!(formatted.aatype[0], 0);

        // Check that N, CA, C, CB, O are present
        let atom_order = build_atom_order();
        for atom_name in &["N", "CA", "C", "CB", "O"] {
            let idx = atom_order[*atom_name];
            assert_eq!(
                formatted.atom_mask[idx], 1.0,
                "Atom {} should be present",
                atom_name
            );
        }

        // Check coordinates for CA (index 1)
        let ca_idx = atom_order["CA"];
        assert_eq!(formatted.coordinates[ca_idx * 3], 1.0);
        assert_eq!(formatted.coordinates[ca_idx * 3 + 1], 0.0);
        assert_eq!(formatted.coordinates[ca_idx * 3 + 2], 0.0);
    }

    #[test]
    fn test_atom37_missing_atoms() {
        // GLY with only backbone (no CB)
        let mut raw = RawAtomData::with_capacity(4);

        let atoms = [
            ("N", 0.0, 0.0, 0.0),
            ("CA", 1.0, 0.0, 0.0),
            ("C", 2.0, 0.0, 0.0),
            ("O", 3.0, 0.0, 0.0),
        ];

        for (i, (name, x, y, z)) in atoms.iter().enumerate() {
            raw.add_atom(AtomRecord {
                serial: i as i32 + 1,
                atom_name: name.to_string(),
                alt_loc: ' ',
                res_name: "GLY".to_string(),
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
        let formatted = Atom37Formatter::format(&processed, &spec).unwrap();

        // GLY should not have CB - check mask
        let atom_order = build_atom_order();
        let cb_idx = atom_order["CB"];
        assert_eq!(formatted.atom_mask[cb_idx], 0.0, "GLY should not have CB");

        // Coordinates for missing atoms should be zero
        assert_eq!(formatted.coordinates[cb_idx * 3], 0.0);
        assert_eq!(formatted.coordinates[cb_idx * 3 + 1], 0.0);
        assert_eq!(formatted.coordinates[cb_idx * 3 + 2], 0.0);
    }

    #[test]
    fn test_atom37_multi_chain() {
        let mut raw = RawAtomData::with_capacity(2);

        // Chain A, residue 1
        raw.add_atom(AtomRecord {
            serial: 1,
            atom_name: "CA".to_string(),
            alt_loc: ' ',
            res_name: "ALA".to_string(),
            chain_id: "A".to_string(),
            res_seq: 1,
            i_code: ' ',
            x: 0.0,
            y: 0.0,
            z: 0.0,
            occupancy: 1.0,
            temp_factor: 20.0,
            element: "C".to_string(),
            charge: None,
            radius: None,
            is_hetatm: false,
        });

        // Chain B, residue 1
        raw.add_atom(AtomRecord {
            serial: 2,
            atom_name: "CA".to_string(),
            alt_loc: ' ',
            res_name: "GLY".to_string(),
            chain_id: "B".to_string(),
            res_seq: 1,
            i_code: ' ',
            x: 10.0,
            y: 0.0,
            z: 0.0,
            occupancy: 1.0,
            temp_factor: 20.0,
            element: "C".to_string(),
            charge: None,
            radius: None,
            is_hetatm: false,
        });

        let processed = ProcessedStructure::from_raw(raw).unwrap();
        let spec = create_test_spec();
        let formatted = Atom37Formatter::format(&processed, &spec).unwrap();

        // Should have 2 residues
        assert_eq!(formatted.aatype.len(), 2);

        // Check chain indices are different
        assert_ne!(formatted.chain_index[0], formatted.chain_index[1]);

        // Check residue types
        assert_eq!(formatted.aatype[0], 0); // ALA
        assert_eq!(formatted.aatype[1], 7); // GLY
    }

    #[test]
    fn test_atom37_unknown_residue() {
        let mut raw = RawAtomData::with_capacity(1);

        raw.add_atom(AtomRecord {
            serial: 1,
            atom_name: "CA".to_string(),
            alt_loc: ' ',
            res_name: "UNK".to_string(),
            chain_id: "A".to_string(),
            res_seq: 1,
            i_code: ' ',
            x: 0.0,
            y: 0.0,
            z: 0.0,
            occupancy: 1.0,
            temp_factor: 20.0,
            element: "C".to_string(),
            charge: None,
            radius: None,
            is_hetatm: false,
        });

        let processed = ProcessedStructure::from_raw(raw).unwrap();
        let spec = create_test_spec();
        let formatted = Atom37Formatter::format(&processed, &spec).unwrap();

        // Unknown residue should have type 20
        assert_eq!(formatted.aatype[0], UNK_RESTYPE_INDEX as i8);

        // CA should still be present
        let atom_order = build_atom_order();
        let ca_idx = atom_order["CA"];
        assert_eq!(formatted.atom_mask[ca_idx], 1.0);
    }
}
