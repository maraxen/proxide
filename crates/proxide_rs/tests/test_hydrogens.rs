use proxide_rs::geometry::hydrogens::{add_hydrogens, init_fragment_library};
use proxide_rs::processing::residues::{ProcessedStructure, ResidueInfo};
use proxide_core::structure::{AtomRecord, Atoms};

#[test]
fn test_add_hydrogens_basic() {
    init_fragment_library();
    
    // Construct a minimal AtomRecord for an Oxygen (Water model)
    let mut raw_atoms = Atoms::new();
    raw_atoms.add_atom(AtomRecord {
        serial: 1,
        atom_name: "O".to_string(),
        alt_loc: ' ',
        res_name: "HOH".to_string(),
        chain_id: "A".to_string(),
        res_seq: 1,
        i_code: ' ',
        x: 0.0,
        y: 0.0,
        z: 0.0,
        occupancy: 1.0,
        temp_factor: 0.0,
        element: "O".to_string(),
        charge: None,
        radius: None,
        is_hetatm: false,
    });
    
    let mut structure = ProcessedStructure {
        raw_atoms,
        molecule_type: vec![0], // Water
        residue_info: vec![ResidueInfo {
            res_id: 1,
            res_name: "HOH".to_string(),
            num_atoms: 1,
            is_protein: false,
        }],
    };
    
    let mut bonds = Vec::new(); // No bonds for a lone oxygen, won't add H

    let result = add_hydrogens(&mut structure, &mut bonds);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 0);
}
