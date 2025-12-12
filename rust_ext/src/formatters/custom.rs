//! Custom coordinate formatter
//!
//! Allows users to specify exactly which atoms they want extracted per residue.
//! Useful for non-standard representations or specific analysis needs.

#![allow(dead_code)]

use crate::processing::ProcessedStructure;
use crate::spec::OutputSpec;
use std::collections::HashMap;

/// Formatted structure with custom atom selection
#[derive(Debug)]
pub struct FormattedCustom {
    /// Coordinates: (N_res, num_atoms, 3) flattened
    pub coordinates: Vec<f32>,
    /// Atom mask: (N_res, num_atoms) flattened - 1.0 if present, 0.0 if missing
    pub atom_mask: Vec<f32>,
    /// Residue types (0-20)
    pub aatype: Vec<i8>,
    /// Residue indices (PDB numbering)
    pub residue_index: Vec<i32>,
    /// Chain indices
    pub chain_index: Vec<i32>,
    /// The atom names requested (for reference)
    pub requested_atoms: Vec<String>,
    /// Number of atoms per residue in output
    pub atoms_per_residue: usize,
}

/// Custom formatter with user-specified atoms
pub struct CustomFormatter;

impl CustomFormatter {
    /// Format a ProcessedStructure with custom atom selection
    ///
    /// # Arguments
    /// * `processed` - The processed structure to format
    /// * `_spec` - Output specification (unused for direct format call)
    /// * `atom_names` - List of atom names to extract (e.g., ["CA", "CB", "N", "C", "O"])
    pub fn format(
        processed: &ProcessedStructure,
        _spec: &OutputSpec,
        atom_names: &[String],
    ) -> Result<FormattedCustom, String> {
        if atom_names.is_empty() {
            return Err("No atom names specified for custom formatter".to_string());
        }

        let n_res = processed.num_residues;
        let n_atoms = atom_names.len();

        // Build atom name to index mapping
        let atom_to_idx: HashMap<&str, usize> = atom_names
            .iter()
            .enumerate()
            .map(|(i, name)| (name.as_str(), i))
            .collect();

        // Initialize output arrays
        let mut coordinates = vec![f32::NAN; n_res * n_atoms * 3];
        let mut atom_mask = vec![0.0f32; n_res * n_atoms];
        let mut aatype = Vec::with_capacity(n_res);
        let mut residue_index = Vec::with_capacity(n_res);
        let mut chain_index = Vec::with_capacity(n_res);

        // Process each residue
        for (res_idx, res_info) in processed.residue_info.iter().enumerate() {
            aatype.push(res_info.res_type as i8);
            residue_index.push(res_info.res_id);

            // Get chain index
            let c_idx = *processed
                .chain_indices
                .get(&res_info.chain_id)
                .unwrap_or(&0);
            chain_index.push(c_idx as i32);

            // Process atoms in this residue
            for atom_idx in res_info.start_atom..(res_info.start_atom + res_info.num_atoms) {
                let atom_name = &processed.raw_atoms.atom_names[atom_idx];

                if let Some(&slot) = atom_to_idx.get(atom_name.as_str()) {
                    // Calculate output index
                    let out_idx = (res_idx * n_atoms + slot) * 3;

                    // Copy coordinates
                    coordinates[out_idx] = processed.raw_atoms.coords[atom_idx * 3];
                    coordinates[out_idx + 1] = processed.raw_atoms.coords[atom_idx * 3 + 1];
                    coordinates[out_idx + 2] = processed.raw_atoms.coords[atom_idx * 3 + 2];

                    // Set mask
                    atom_mask[res_idx * n_atoms + slot] = 1.0;
                }
            }
        }

        Ok(FormattedCustom {
            coordinates,
            atom_mask,
            aatype,
            residue_index,
            chain_index,
            requested_atoms: atom_names.to_vec(),
            atoms_per_residue: n_atoms,
        })
    }
}

impl FormattedCustom {
    /// Convert to Python dictionary
    pub fn to_py_dict(&self, py: pyo3::Python) -> pyo3::PyResult<pyo3::PyObject> {
        use numpy::PyArray1;
        use pyo3::prelude::*;
        use pyo3::types::PyDict;

        let dict = PyDict::new_bound(py);

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

        let atom_names: Vec<&str> = self.requested_atoms.iter().map(|s| s.as_str()).collect();
        dict.set_item("requested_atoms", atom_names)?;
        dict.set_item("atoms_per_residue", self.atoms_per_residue)?;

        Ok(dict.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::processing::ProcessedStructure;
    use crate::spec::OutputSpec;
    use crate::structure::{AtomRecord, RawAtomData};

    fn create_test_structure() -> ProcessedStructure {
        let mut raw = RawAtomData::new();

        // Alanine with N, CA, C, O, CB
        let atoms = [
            ("N", 0.0, 0.0, 0.0),
            ("CA", 1.5, 0.0, 0.0),
            ("C", 2.5, 0.0, 0.0),
            ("O", 3.0, 1.0, 0.0),
            ("CB", 1.5, 1.5, 0.0),
        ];

        for (i, (name, x, y, z)) in atoms.iter().enumerate() {
            raw.add_atom(AtomRecord {
                serial: (i + 1) as i32,
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

        ProcessedStructure::from_raw(raw).unwrap()
    }

    #[test]
    fn test_custom_format_backbone() {
        let processed = create_test_structure();
        let spec = OutputSpec::default();

        let atom_names: Vec<String> = vec!["N", "CA", "C", "O"]
            .into_iter()
            .map(|s| s.to_string())
            .collect();

        let formatted = CustomFormatter::format(&processed, &spec, &atom_names).unwrap();

        assert_eq!(formatted.aatype.len(), 1);
        assert_eq!(formatted.atoms_per_residue, 4);
        assert_eq!(formatted.coordinates.len(), 1 * 4 * 3);
        assert_eq!(formatted.atom_mask.len(), 1 * 4);

        // All backbone atoms should be present
        for i in 0..4 {
            assert_eq!(formatted.atom_mask[i], 1.0);
        }
    }

    #[test]
    fn test_custom_format_partial() {
        let processed = create_test_structure();
        let spec = OutputSpec::default();

        // Request atoms that partially exist
        let atom_names: Vec<String> = vec!["CA", "CB", "CG"] // CG doesn't exist in ALA
            .into_iter()
            .map(|s| s.to_string())
            .collect();

        let formatted = CustomFormatter::format(&processed, &spec, &atom_names).unwrap();

        // CA and CB should be present, CG should be missing
        assert_eq!(formatted.atom_mask[0], 1.0); // CA
        assert_eq!(formatted.atom_mask[1], 1.0); // CB
        assert_eq!(formatted.atom_mask[2], 0.0); // CG (missing)
    }
}
