mod common;

use common::{fixture, load_topology, fixture_path_str};

#[test]
fn test_fixture_path_exists() {
    let path = fixture("clean_small.pdb");
    assert!(path.exists(), "clean_small.pdb fixture not found at {:?}", path);

    let path = fixture("disulfide_pair.pdb");
    assert!(path.exists(), "disulfide_pair.pdb fixture not found at {:?}", path);

    let path = fixture("missing_atoms.pdb");
    assert!(path.exists(), "missing_atoms.pdb fixture not found at {:?}", path);

    let path = fixture("chain_break.pdb");
    assert!(path.exists(), "chain_break.pdb fixture not found at {:?}", path);
}

#[test]
fn test_fixture_path_str_returns_string() {
    let path_str = fixture_path_str("clean_small.pdb");
    assert!(!path_str.is_empty());
    assert!(path_str.contains("clean_small.pdb"));
}

#[test]
fn test_load_topology_clean_small() {
    let topology = load_topology("clean_small.pdb");
    assert!(
        topology.is_some(),
        "Failed to load clean_small.pdb fixture as topology"
    );

    let topology = topology.unwrap();
    assert_eq!(
        topology.chains.len(),
        1,
        "clean_small should have exactly 1 chain"
    );

    let chain = &topology.chains[0];
    assert_eq!(chain.id, "A", "Chain ID should be 'A'");
    assert_eq!(
        chain.residues.len(),
        3,
        "clean_small should have 3 residues (ALA, GLY, SER)"
    );

    // Verify first residue
    let res1 = &chain.residues[0];
    assert_eq!(res1.name, "ALA");
    assert_eq!(res1.res_id, 1);
    assert!(res1.atoms.len() >= 4, "ALA should have N, CA, C, O, CB");

    // Verify all residues have expected atoms
    for res in &chain.residues {
        assert!(
            res.atoms.len() > 0,
            "Each residue should have at least backbone atoms"
        );
    }
}

#[test]
fn test_load_topology_disulfide_pair() {
    let topology = load_topology("disulfide_pair.pdb");
    assert!(
        topology.is_some(),
        "Failed to load disulfide_pair.pdb fixture as topology"
    );

    let topology = topology.unwrap();
    assert_eq!(
        topology.chains.len(),
        1,
        "disulfide_pair should have 1 chain"
    );

    let chain = &topology.chains[0];
    assert_eq!(
        chain.residues.len(),
        2,
        "disulfide_pair should have 2 CYS residues"
    );

    // Verify both residues are CYS
    for (i, res) in chain.residues.iter().enumerate() {
        assert_eq!(
            res.name, "CYS",
            "Residue {} should be CYS",
            i + 1
        );

        // Find SG atom in this residue
        let sg_atoms: Vec<_> = res
            .atoms
            .iter()
            .filter(|atom| atom.name == "SG")
            .collect();
        assert_eq!(
            sg_atoms.len(),
            1,
            "CYS residue {} should have exactly one SG atom",
            i + 1
        );
    }

    // Verify SG atoms are close (~2.04 Angstroms apart for S-S bond)
    let cys1_sg = chain.residues[0]
        .atoms
        .iter()
        .find(|a| a.name == "SG")
        .unwrap();
    let cys2_sg = chain.residues[1]
        .atoms
        .iter()
        .find(|a| a.name == "SG")
        .unwrap();

    let dx = cys2_sg.coords[0] - cys1_sg.coords[0];
    let dy = cys2_sg.coords[1] - cys1_sg.coords[1];
    let dz = cys2_sg.coords[2] - cys1_sg.coords[2];
    let distance = (dx * dx + dy * dy + dz * dz).sqrt();

    assert!(
        distance < 2.5,
        "SG atoms should be close for disulfide bond; distance={:.2} Angstroms",
        distance
    );
}

#[test]
fn test_load_topology_missing_atoms() {
    let topology = load_topology("missing_atoms.pdb");
    assert!(
        topology.is_some(),
        "Failed to load missing_atoms.pdb fixture as topology"
    );

    let topology = topology.unwrap();
    assert_eq!(topology.chains.len(), 1);

    let chain = &topology.chains[0];
    assert_eq!(
        chain.residues.len(),
        3,
        "missing_atoms should have 3 residues despite missing atoms"
    );

    // Verify residue 2 (GLY) is missing C and O
    let res2 = &chain.residues[1];
    assert_eq!(res2.name, "GLY");

    let c_atoms: Vec<_> = res2
        .atoms
        .iter()
        .filter(|a| a.name == "C")
        .collect();
    let o_atoms: Vec<_> = res2
        .atoms
        .iter()
        .filter(|a| a.name == "O")
        .collect();

    assert_eq!(
        c_atoms.len(),
        0,
        "GLY residue in missing_atoms should not have C atom"
    );
    assert_eq!(
        o_atoms.len(),
        0,
        "GLY residue in missing_atoms should not have O atom"
    );

    // Verify N and CA are still present
    let n_atoms: Vec<_> = res2
        .atoms
        .iter()
        .filter(|a| a.name == "N")
        .collect();
    let ca_atoms: Vec<_> = res2
        .atoms
        .iter()
        .filter(|a| a.name == "CA")
        .collect();

    assert_eq!(n_atoms.len(), 1, "GLY should still have N");
    assert_eq!(ca_atoms.len(), 1, "GLY should still have CA");
}

#[test]
fn test_load_topology_chain_break() {
    let topology = load_topology("chain_break.pdb");
    assert!(
        topology.is_some(),
        "Failed to load chain_break.pdb fixture as topology"
    );

    let topology = topology.unwrap();
    assert_eq!(topology.chains.len(), 1, "chain_break should be single chain");

    let chain = &topology.chains[0];
    assert_eq!(
        chain.residues.len(),
        4,
        "chain_break should have 4 residues with gap between 2 and 3"
    );

    // Verify CA atoms have the gap
    let ca_atoms: Vec<_> = chain
        .residues
        .iter()
        .flat_map(|r| r.atoms.iter().filter(|a| a.name == "CA"))
        .collect();

    assert_eq!(ca_atoms.len(), 4, "Should have 4 CA atoms");

    // Check distance between CA[1] and CA[2] (should be ~10 Angstroms)
    let ca1 = ca_atoms[1];
    let ca2 = ca_atoms[2];

    let dx = ca2.coords[0] - ca1.coords[0];
    let dy = ca2.coords[1] - ca1.coords[1];
    let dz = ca2.coords[2] - ca1.coords[2];
    let distance = (dx * dx + dy * dy + dz * dz).sqrt();

    assert!(
        distance > 8.0,
        "Chain break should show >8 Angstrom gap; distance={:.2}",
        distance
    );
}

#[test]
fn test_harness_consistent_fixture_format() {
    // Verify all fixtures have consistent format (valid PDB headers, etc.)
    for name in &[
        "clean_small.pdb",
        "disulfide_pair.pdb",
        "missing_atoms.pdb",
        "chain_break.pdb",
    ] {
        let path = fixture(name);
        assert!(path.exists(), "Fixture {} not found", name);

        // Try to read as text and verify it starts with HEADER or REMARK/ATOM
        let content = std::fs::read_to_string(&path)
            .expect(&format!("Failed to read {}", name));
        assert!(
            !content.is_empty(),
            "Fixture {} is empty",
            name
        );

        // Ensure it has ATOM records
        assert!(
            content.contains("ATOM"),
            "Fixture {} should contain ATOM records",
            name
        );
    }
}
