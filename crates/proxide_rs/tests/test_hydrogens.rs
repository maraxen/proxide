use proxide_rs::geometry::hydrogens::{add_hydrogens, init_fragment_library};
use proxide_rs::processing::residues::{ProcessedStructure};
use proxide_core::structure::{AtomRecord, RawAtomData};

#[test]
fn test_add_hydrogens_basic() {
    init_fragment_library();
    
    // Construct a minimal RawAtomData
    let mut raw_atoms = RawAtomData::new();
    raw_atoms.add_atom(AtomRecord {
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
    
    let mut structure = ProcessedStructure::from_raw(raw_atoms).unwrap();
    
    let mut bonds = Vec::new();

    let result = add_hydrogens(&mut structure, &mut bonds);
    assert!(result.is_ok());
    // Should add hydrogens if fragment exists
}
