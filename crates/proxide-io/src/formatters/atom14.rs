//! Atom14 coordinate formatter
//!
//! Converts ProcessedStructure into reduced (N_res, 14, 3) coordinate arrays.
//! Uses residue-specific atom ordering from restype_name_to_atom14_names.

use proxide_core::chem::{build_restype_atom14_names, RESTYPE_1TO3};
use proxide_core::processing::ProcessedStructure;
use proxide_core::spec::OutputSpec;
use std::collections::HashMap;

/// Formatted structure in Atom14 representation
#[derive(Debug)]
pub struct FormattedAtom14 {
    pub coordinates: Vec<f32>,   // Flat (N_res * 14 * 3)
    pub atom_mask: Vec<f32>,     // Flat (N_res * 14)
    pub aatype: Vec<i8>,         // (N_res,) residue type indices
    pub residue_index: Vec<i32>, // (N_res,) PDB residue numbers
    pub chain_index: Vec<i32>,   // (N_res,) chain indices
    pub unplaced_residues: Vec<(usize, String)>, // (residue_index, res_name) for residues with atoms but no atom14 layout
}

/// Atom14 formatter - converts to reduced (N_res, 14, 3) representation
pub struct Atom14Formatter;

impl Atom14Formatter {
    /// Format a ProcessedStructure into Atom14 representation
    pub fn format(
        processed: &ProcessedStructure,
        _spec: &OutputSpec,
    ) -> Result<FormattedAtom14, String> {
        let num_residues = processed.num_residues;
        let atom14_names = build_restype_atom14_names();

        log::debug!("Formatting Atom14 for {} residues", num_residues);

        // Pre-allocate output arrays
        let mut coordinates = vec![0.0f32; num_residues * 14 * 3];
        let mut atom_mask = vec![0.0f32; num_residues * 14];
        let mut aatype = vec![0i8; num_residues];
        let mut residue_index = vec![0i32; num_residues];
        let mut chain_index = vec![0i32; num_residues];
        let mut unplaced_residues: Vec<(usize, String)> = Vec::new();

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

            // Get the canonical 3-letter residue name from res_type
            let canonical_res_name = if res_info.res_type < RESTYPE_1TO3.len() {
                RESTYPE_1TO3[res_info.res_type].1
            } else {
                "UNK"
            };

            // Get atom14 names for this residue type
            let atom14_list = atom14_names
                .get(canonical_res_name)
                .unwrap_or_else(|| atom14_names.get("UNK").unwrap());

            // Build atom name -> coordinates mapping for this residue
            let mut residue_atoms: HashMap<String, usize> = HashMap::new();
            let start = res_info.start_atom;
            let end = start + res_info.num_atoms;

            for local_idx in 0..(end - start) {
                let global_idx = start + local_idx;
                let atom_name = &processed.raw_atoms.atom_names[global_idx];
                residue_atoms.insert(atom_name.clone(), global_idx);
            }

            // Check if this is an unplaced residue (atoms but no atom14 layout)
            if res_info.num_atoms > 0 && canonical_res_name == "UNK" {
                unplaced_residues.push((res_idx, res_info.res_name.clone()));
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

        // Warn if there are unplaced residues
        if !unplaced_residues.is_empty() {
            let count = unplaced_residues.len();
            let names: Vec<String> = unplaced_residues
                .iter()
                .take(10)
                .map(|(idx, name)| format!("RES#{}{}", idx, name))
                .collect();
            let suffix = if count > 10 {
                format!(" (+{} more)", count - 10)
            } else {
                String::new()
            };
            log::warn!(
                "Atom14: {} residues with atoms but no atom14 layout: {}{}",
                count,
                names.join(", "),
                suffix
            );
        }

        Ok(FormattedAtom14 {
            coordinates,
            atom_mask,
            aatype,
            residue_index,
            chain_index,
            unplaced_residues,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proxide_core::spec::OutputSpec;
    use proxide_core::structure::{AtomRecord, RawAtomData};

    #[test]
    fn test_atom14_single_residue() {
        let mut raw = RawAtomData::with_capacity(5);

        let atoms = [
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

        let processed = proxide_core::processing::ProcessedStructure::from_raw(raw).unwrap();
        let spec = OutputSpec::default();
        let formatted = Atom14Formatter::format(&processed, &spec).unwrap();

        // Verify dimensions
        assert_eq!(formatted.coordinates.len(), 14 * 3);
        assert_eq!(formatted.atom_mask.len(), 14);

        // N, CA, C, O, CB should be at indices 0, 1, 2, 3, 4 respectively
        assert_eq!(formatted.atom_mask[0], 1.0); // N
        assert_eq!(formatted.atom_mask[1], 1.0); // CA
        assert_eq!(formatted.atom_mask[2], 1.0); // C
        assert_eq!(formatted.atom_mask[3], 1.0); // O
        assert_eq!(formatted.atom_mask[4], 1.0); // CB

        // Check CA coordinates (index 1)
        assert_eq!(formatted.coordinates[3], 1.0);
        assert_eq!(formatted.coordinates[3 + 1], 0.0);
        assert_eq!(formatted.coordinates[3 + 2], 0.0);
    }

    #[test]
    fn test_atom14_his_alias_variant() {
        // Test that HSD (alias for HIS) with HIS heavy atoms gets placed correctly
        // by looking up the atom14 layout via res_type, not res_name
        let mut raw = RawAtomData::with_capacity(10);

        let his_atoms = [
            ("N", 0.0, 0.0, 0.0),
            ("CA", 1.0, 0.0, 0.0),
            ("C", 2.0, 0.0, 0.0),
            ("O", 3.0, 0.0, 0.0),
            ("CB", 1.0, 1.0, 0.0),
            ("CG", 1.0, 2.0, 0.0),
            ("ND1", 1.0, 3.0, 0.0),
            ("CD2", 2.0, 3.0, 0.0),
            ("CE1", 2.0, 2.0, 0.0),
            ("NE2", 2.0, 1.0, 0.0),
        ];

        for (i, (name, x, y, z)) in his_atoms.iter().enumerate() {
            raw.add_atom(AtomRecord {
                serial: i as i32 + 1,
                atom_name: name.to_string(),
                alt_loc: ' ',
                res_name: "HSD".to_string(), // HIS alias variant
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

        let processed = proxide_core::processing::ProcessedStructure::from_raw(raw).unwrap();
        let spec = OutputSpec::default();
        let formatted = Atom14Formatter::format(&processed, &spec).unwrap();

        // HIS atom14 layout: N, CA, C, O, CB, CG, ND1, CD2, CE1, NE2 (10 atoms)
        // All atoms should be placed
        assert_eq!(formatted.atom_mask[0], 1.0); // N
        assert_eq!(formatted.atom_mask[1], 1.0); // CA
        assert_eq!(formatted.atom_mask[2], 1.0); // C
        assert_eq!(formatted.atom_mask[3], 1.0); // O
        assert_eq!(formatted.atom_mask[4], 1.0); // CB
        assert_eq!(formatted.atom_mask[5], 1.0); // CG
        assert_eq!(formatted.atom_mask[6], 1.0); // ND1
        assert_eq!(formatted.atom_mask[7], 1.0); // CD2
        assert_eq!(formatted.atom_mask[8], 1.0); // CE1
        assert_eq!(formatted.atom_mask[9], 1.0); // NE2

        // The sum of the mask should be 10
        let mask_sum: f32 = formatted.atom_mask.iter().sum();
        assert_eq!(mask_sum, 10.0);

        // No unplaced residues (HSD has a HIS type)
        assert_eq!(formatted.unplaced_residues.len(), 0);
    }

    #[test]
    fn test_atom14_mse_unplaced() {
        // Test that MSE (selenomethionine, ligand/unknown type) with atoms
        // gets recorded in unplaced_residues
        let mut raw = RawAtomData::with_capacity(10);

        let mse_atoms = [
            ("N", 0.0, 0.0, 0.0),
            ("CA", 1.0, 0.0, 0.0),
            ("C", 2.0, 0.0, 0.0),
            ("O", 3.0, 0.0, 0.0),
            ("CB", 1.0, 1.0, 0.0),
            ("CG", 1.0, 2.0, 0.0),
            ("SE", 1.0, 3.0, 0.0),
            ("CE", 2.0, 3.0, 0.0),
        ];

        for (i, (name, x, y, z)) in mse_atoms.iter().enumerate() {
            raw.add_atom(AtomRecord {
                serial: i as i32 + 1,
                atom_name: name.to_string(),
                alt_loc: ' ',
                res_name: "MSE".to_string(),
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

        let processed = proxide_core::processing::ProcessedStructure::from_raw(raw).unwrap();
        let spec = OutputSpec::default();
        let formatted = Atom14Formatter::format(&processed, &spec).unwrap();

        // MSE has res_type = UNK (index 20), so it has no atom14 layout
        // All atom14 positions should be zero
        let mask_sum: f32 = formatted.atom_mask.iter().sum();
        assert_eq!(mask_sum, 0.0);

        // The residue should be recorded in unplaced_residues
        assert_eq!(formatted.unplaced_residues.len(), 1);
        assert_eq!(formatted.unplaced_residues[0].0, 0); // residue index 0
        assert_eq!(formatted.unplaced_residues[0].1, "MSE");
    }
}
