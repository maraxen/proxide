//! Atom14 coordinate formatter
//!
//! Converts ProcessedStructure into reduced (N_res, 14, 3) coordinate arrays.
//! Uses residue-specific atom ordering from restype_name_to_atom14_names.

use crate::chem::build_restype_atom14_names;
use crate::processing::ProcessedStructure;
use crate::spec::OutputSpec;
use std::collections::HashMap;

/// Formatted structure in Atom14 representation
#[derive(Debug)]
pub struct FormattedAtom14 {
    pub coordinates: Vec<f32>,   // Flat (N_res * 14 * 3)
    pub atom_mask: Vec<f32>,     // Flat (N_res * 14)
    pub aatype: Vec<i8>,         // (N_res,) residue type indices
    pub residue_index: Vec<i32>, // (N_res,) PDB residue numbers
    pub chain_index: Vec<i32>,   // (N_res,) chain indices
}

/// Atom14 formatter - converts to reduced (N_res, 14, 3) representation
pub struct Atom14Formatter;

impl Atom14Formatter {
    /// Format a ProcessedStructure into Atom14 representation
    pub fn format(processed: &ProcessedStructure, _spec: &OutputSpec) -> Result<FormattedAtom14, String> {
        let num_residues = processed.num_residues;
        let atom14_names = build_restype_atom14_names();
        
        log::debug!("Formatting Atom14 for {} residues", num_residues);
        
        // Pre-allocate output arrays
        let mut coordinates = vec![0.0f32; num_residues * 14 * 3];
        let mut atom_mask = vec![0.0f32; num_residues * 14];
        let mut aatype = vec![0i8; num_residues];
        let mut residue_index = vec![0i32; num_residues];
        let mut chain_index = vec![0i32; num_residues];
        
        // Process each residue
        for (res_idx, res_info) in processed.residue_info.iter().enumerate() {
            // Set residue metadata
            aatype[res_idx] = res_info.res_type as i8;
            residue_index[res_idx] = res_info.res_id;
            
            // Map chain to index
            chain_index[res_idx] = *processed.chain_indices
                .get(&res_info.chain_id)
                .unwrap_or(&0) as i32;
            
            // Get atom14 names for this residue type
            let res_name = res_info.res_name.as_str();
            let atom14_list = atom14_names.get(res_name)
                .or_else(|| atom14_names.get("UNK"))
                .unwrap();
            
            // Build atom name -> coordinates mapping for this residue
            let mut residue_atoms: HashMap<String, usize> = HashMap::new();
            let start = res_info.start_atom;
            let end = start + res_info.num_atoms;
            
            for local_idx in 0..(end - start) {
                let global_idx = start + local_idx;
                let atom_name = &processed.raw_atoms.atom_names[global_idx];
                residue_atoms.insert(atom_name.clone(), global_idx);
            }
            
            // Fill in atom14 positions
            for (atom14_idx, atom_name) in atom14_list.iter().enumerate() {
                if atom_name.is_empty() {
                    continue;
                }
                
                let coord_base = (res_idx * 14 + atom14_idx) * 3;
                let mask_idx = res_idx * 14 + atom14_idx;
                
                if let Some(&global_idx) = residue_atoms.get(*atom_name) {
                    // Atom is present - copy coordinates
                    coordinates[coord_base] = processed.raw_atoms.coords[global_idx * 3];
                    coordinates[coord_base + 1] = processed.raw_atoms.coords[global_idx * 3 + 1];
                    coordinates[coord_base + 2] = processed.raw_atoms.coords[global_idx * 3 + 2];
                    atom_mask[mask_idx] = 1.0;
                }
                // Missing atoms stay as zeros
            }
        }
        
        Ok(FormattedAtom14 {
            coordinates,
            atom_mask,
            aatype,
            residue_index,
            chain_index,
        })
    }
}

impl FormattedAtom14 {
    /// Convert to Python dictionary
    pub fn to_py_dict(&self, py: pyo3::Python) -> pyo3::PyResult<pyo3::PyObject> {
        use pyo3::prelude::*;
        use pyo3::types::PyDict;
        use numpy::PyArray1;
        
        let dict = PyDict::new_bound(py);
        
        dict.set_item("coordinates", PyArray1::from_vec_bound(py, self.coordinates.clone()))?;
        dict.set_item("atom_mask", PyArray1::from_vec_bound(py, self.atom_mask.clone()))?;
        dict.set_item("aatype", PyArray1::from_vec_bound(py, self.aatype.clone()))?;
        dict.set_item("residue_index", PyArray1::from_vec_bound(py, self.residue_index.clone()))?;
        dict.set_item("chain_index", PyArray1::from_vec_bound(py, self.chain_index.clone()))?;
        
        Ok(dict.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structure::{AtomRecord, RawAtomData};
    use crate::spec::OutputSpec;
    
    #[test]
    fn test_atom14_single_residue() {
        let mut raw = RawAtomData::with_capacity(5);
        
        let atoms = vec![
            ("N", 0.0, 0.0, 0.0),
            ("CA", 1.0, 0.0, 0.0),
            ("C", 2.0, 0.0, 0.0),
            ("O", 3.0, 0.0, 0.0),
            ("CB", 1.0, 1.0, 0.0),
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
        
        let processed = crate::processing::ProcessedStructure::from_raw(raw).unwrap();
        let spec = OutputSpec::default();
        let formatted = Atom14Formatter::format(&processed, &spec).unwrap();
        
        // Verify dimensions
        assert_eq!(formatted.coordinates.len(), 1 * 14 * 3);
        assert_eq!(formatted.atom_mask.len(), 1 * 14);
        
        // N, CA, C, O, CB should be at indices 0, 1, 2, 3, 4 respectively
        assert_eq!(formatted.atom_mask[0], 1.0); // N
        assert_eq!(formatted.atom_mask[1], 1.0); // CA
        assert_eq!(formatted.atom_mask[2], 1.0); // C
        assert_eq!(formatted.atom_mask[3], 1.0); // O
        assert_eq!(formatted.atom_mask[4], 1.0); // CB
        
        // Check CA coordinates (index 1)
        assert_eq!(formatted.coordinates[1 * 3], 1.0);
        assert_eq!(formatted.coordinates[1 * 3 + 1], 0.0);
        assert_eq!(formatted.coordinates[1 * 3 + 2], 0.0);
    }
}
