//! Backbone-only coordinate formatter
//!
//! Converts ProcessedStructure into minimal (N_res, 4, 3) backbone coordinates.
//! Extracts only N, CA, C, O atoms.

use crate::processing::ProcessedStructure;
use crate::spec::OutputSpec;
use std::collections::HashMap;

/// Backbone atom ordering
const BACKBONE_ATOMS: [&str; 4] = ["N", "CA", "C", "O"];

/// Formatted structure in backbone-only representation
#[derive(Debug)]
pub struct FormattedBackbone {
    pub coordinates: Vec<f32>,   // Flat (N_res * 4 * 3)
    pub atom_mask: Vec<f32>,     // Flat (N_res * 4)
    pub aatype: Vec<i8>,         // (N_res,) residue type indices
    pub residue_index: Vec<i32>, // (N_res,) PDB residue numbers
    pub chain_index: Vec<i32>,   // (N_res,) chain indices
}

/// Backbone formatter - extracts only N, CA, C, O atoms
pub struct BackboneFormatter;

impl BackboneFormatter {
    /// Format a ProcessedStructure into backbone-only representation
    pub fn format(processed: &ProcessedStructure, _spec: &OutputSpec) -> Result<FormattedBackbone, String> {
        let num_residues = processed.num_residues;
        
        log::debug!("Formatting Backbone for {} residues", num_residues);
        
        // Pre-allocate output arrays
        let mut coordinates = vec![0.0f32; num_residues * 4 * 3];
        let mut atom_mask = vec![0.0f32; num_residues * 4];
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
            
            // Build atom name -> coordinates mapping for this residue
            let mut residue_atoms: HashMap<String, usize> = HashMap::new();
            let start = res_info.start_atom;
            let end = start + res_info.num_atoms;
            
            for local_idx in 0..(end - start) {
                let global_idx = start + local_idx;
                let atom_name = &processed.raw_atoms.atom_names[global_idx];
                residue_atoms.insert(atom_name.clone(), global_idx);
            }
            
            // Fill in backbone positions
            for (bb_idx, atom_name) in BACKBONE_ATOMS.iter().enumerate() {
                let coord_base = (res_idx * 4 + bb_idx) * 3;
                let mask_idx = res_idx * 4 + bb_idx;
                
                if let Some(&global_idx) = residue_atoms.get(*atom_name) {
                    // Atom is present - copy coordinates
                    coordinates[coord_base] = processed.raw_atoms.coords[global_idx * 3];
                    coordinates[coord_base + 1] = processed.raw_atoms.coords[global_idx * 3 + 1];
                    coordinates[coord_base + 2] = processed.raw_atoms.coords[global_idx * 3 + 2];
                    atom_mask[mask_idx] = 1.0;
                } else {
                    log::warn!(
                        "Missing backbone atom {} in residue {} {} (chain {})",
                        atom_name,
                        res_info.res_name,
                        res_info.res_id,
                        res_info.chain_id
                    );
                }
            }
        }
        
        Ok(FormattedBackbone {
            coordinates,
            atom_mask,
            aatype,
            residue_index,
            chain_index,
        })
    }
}

impl FormattedBackbone {
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
    fn test_backbone_single_residue() {
        let mut raw = RawAtomData::with_capacity(5);
        
        let atoms = vec![
            ("N", 0.0, 0.0, 0.0),
            ("CA", 1.0, 0.0, 0.0),
            ("C", 2.0, 0.0, 0.0),
            ("O", 3.0, 0.0, 0.0),
            ("CB", 1.0, 1.0, 0.0),  // Should be ignored
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
        let formatted = BackboneFormatter::format(&processed, &spec).unwrap();
        
        // Verify dimensions (4 backbone atoms per residue)
        assert_eq!(formatted.coordinates.len(), 1 * 4 * 3);
        assert_eq!(formatted.atom_mask.len(), 1 * 4);
        
        // All backbone atoms should be present
        assert_eq!(formatted.atom_mask[0], 1.0); // N
        assert_eq!(formatted.atom_mask[1], 1.0); // CA
        assert_eq!(formatted.atom_mask[2], 1.0); // C
        assert_eq!(formatted.atom_mask[3], 1.0); // O
        
        // Check O coordinates (index 3)
        assert_eq!(formatted.coordinates[3 * 3], 3.0);
        assert_eq!(formatted.coordinates[3 * 3 + 1], 0.0);
        assert_eq!(formatted.coordinates[3 * 3 + 2], 0.0);
    }
}
