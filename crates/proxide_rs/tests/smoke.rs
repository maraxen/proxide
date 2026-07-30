use proxide_core::structure::systems::AtomicSystem;
use proxide_geometry::generate_topology;
use proxide_io::formats::pdb::parse_pdb_file;
use std::path::PathBuf;

#[test]
fn test_smoke_parsing_topology_physics() {
    // 1. Parsing
    let pdb_path = PathBuf::from("tests/data/1crn.pdb");
    if !pdb_path.exists() {
        // Skip if test data is missing in this environment
        return;
    }
    let (raw_data, _model_ids) = parse_pdb_file(&pdb_path).expect("Failed to parse PDB");

    let res_ids = raw_data.res_ids.clone();
    let coords = raw_data.coords.clone();
    let elements = raw_data.elements.clone();

    // Convert flattened coords to &[[f32; 3]]
    let coords_slice: &[[f32; 3]] = bytemuck::cast_slice(&coords);

    // Create AtomicSystem from raw_data
    use proxide_core::structure::systems::AtomicSystemArgs;
    let system = AtomicSystem::new(AtomicSystemArgs {
        coordinates: raw_data.coords,
        atom_mask: vec![1.0; raw_data.num_atoms],
        atom_names: Some(raw_data.atom_names),
        elements: Some(raw_data.elements),
        bonds: None,
        charges: raw_data.charges,
        sigmas: raw_data.sigmas,
        epsilons: raw_data.epsilons,
        radii: raw_data.radii,
        residue_index: Some(raw_data.res_ids),
        chain_index: Some(res_ids),
    });

    assert!(system.num_atoms > 0, "System should have atoms");

    // 2. Topology
    let topo = generate_topology(coords_slice, &elements, 1.3);
    assert!(topo.bonds.len() > 0, "Should have inferred some bonds");

    // 3. Physics
    if let Some(charges) = &system.charges {
        assert_eq!(charges.len(), system.num_atoms, "Charges count mismatch");
    }
}
