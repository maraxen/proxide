#[cfg(test)]
mod tests {
    use super::*;
    use proxide_core::structure::AtomRecords;
    use crate::processing::residues::{ProcessedStructure, ResidueInfo};

    fn create_mock_structure() -> ProcessedStructure {
        let mut atoms = AtomRecords::new();
        // Add a simple C-N bond (e.g. part of a backbone)
        atoms.add_atom(crate::geometry::AtomRecord {
            serial: 1,
            atom_name: "C".to_string(),
            alt_loc: ' ',
            res_name: "ALA".to_string(),
            chain_id: "A".to_string(),
            res_seq: 1,
            i_code: ' ',
            x: 0.0,
            y: 0.0,
            z: 0.0,
            occupancy: 1.0,
            temp_factor: 0.0,
            element: "C".to_string(),
            charge: None,
            radius: None,
            is_hetatm: false,
        });
        atoms.add_atom(crate::geometry::AtomRecord {
            serial: 2,
            atom_name: "N".to_string(),
            alt_loc: ' ',
            res_name: "ALA".to_string(),
            chain_id: "A".to_string(),
            res_seq: 1,
            i_code: ' ',
            x: 1.5,
            y: 0.0,
            z: 0.0,
            occupancy: 1.0,
            temp_factor: 0.0,
            element: "N".to_string(),
            charge: None,
            radius: None,
            is_hetatm: false,
        });

        ProcessedStructure {
            raw_atoms: atoms,
            molecule_type: vec![0, 0],
            residue_info: vec![ResidueInfo {
                res_id: 1,
                res_name: "ALA".to_string(),
                num_atoms: 2,
                first_atom_index: 0,
            }],
        }
    }

    #[test]
    fn test_add_hydrogens_basic() {
        init_fragment_library();
        let mut structure = create_mock_structure();
        let mut bonds = vec![[0, 1]];

        let added = add_hydrogens(&mut structure, &mut bonds).expect("Failed to add hydrogens");
        assert!(added > 0, "Should have added at least one hydrogen");
        assert_eq!(structure.raw_atoms.num_atoms, 2 + added);
    }
}
