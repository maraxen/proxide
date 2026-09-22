#![cfg(feature = "disulfide")]

// Test C2 (disulfide sanitizer gate) conformance for fixtures.
// These tests run with --features disulfide and cross-check the exact
// disulfide detection and renaming for each fixture.
mod common;

use common::load_topology;
use proxide_fixer::sanitizers::disulfide::DisulfideSanitizer;

#[test]
fn test_c2_disulfide_pair() {
    // disulfide_pair.pdb: 2 CYS residues with SG atoms ~2.04 Å apart.
    // Expected: exactly one Disulfide detected, both residues renamed to CYX.
    let mut topology = load_topology("disulfide_pair.pdb");

    // Verify initial state: both residues named CYS
    assert_eq!(topology.chains[0].residues[0].name, "CYS");
    assert_eq!(topology.chains[0].residues[1].name, "CYS");

    let mut sanitizer = DisulfideSanitizer::new(&mut topology);
    let disulfides = sanitizer.run().expect("run should succeed");

    // Verify exactly one disulfide was detected
    assert_eq!(
        disulfides.len(),
        1,
        "Should detect exactly one disulfide bond"
    );

    let disulfide = &disulfides[0];
    assert_eq!(disulfide.chain_a, "A", "First chain should be A");
    assert_eq!(disulfide.res_a, 1, "First residue should be 1");
    assert_eq!(disulfide.chain_b, "A", "Second chain should be A");
    assert_eq!(disulfide.res_b, 2, "Second residue should be 2");

    // Verify distance is ~2.04 Å (SG atoms)
    // disulfide_pair.pdb: SG1 at (3.740, -0.760, -1.207), SG2 at (5.780, -0.760, -1.207)
    // Distance = sqrt((5.780 - 3.740)^2 + 0 + 0) = 2.04
    assert!(
        (disulfide.distance - 2.04).abs() < 1e-3,
        "SG-SG distance should be ~2.04 Å, got {:.4}",
        disulfide.distance
    );

    // Verify residue names were updated to CYX
    assert_eq!(
        topology.chains[0].residues[0].name, "CYX",
        "First CYS should be renamed to CYX"
    );
    assert_eq!(
        topology.chains[0].residues[1].name, "CYX",
        "Second CYS should be renamed to CYX"
    );
}

#[test]
fn test_c2_clean_small() {
    // clean_small.pdb: ALA1, GLY2, SER3 — no CYS residues.
    // Expected: no disulfides, all residues unchanged.
    let mut topology = load_topology("clean_small.pdb");

    // Verify initial state: no CYS residues
    for residue in &topology.chains[0].residues {
        assert!(
            residue.name != "CYS" && residue.name != "CYH" && residue.name != "CYM",
            "clean_small should have no cysteine residues"
        );
    }

    let mut sanitizer = DisulfideSanitizer::new(&mut topology);
    let disulfides = sanitizer.run().expect("run should succeed");

    assert_eq!(disulfides.len(), 0, "Should detect no disulfides");

    // Verify residues are unchanged
    assert_eq!(topology.chains[0].residues[0].name, "ALA");
    assert_eq!(topology.chains[0].residues[1].name, "GLY");
    assert_eq!(topology.chains[0].residues[2].name, "SER");
}

#[test]
fn test_c2_missing_atoms() {
    // missing_atoms.pdb: ALA1, GLY2, SER3 — no CYS residues.
    // Expected: no disulfides, all residues unchanged.
    let mut topology = load_topology("missing_atoms.pdb");

    // Verify initial state: no CYS residues
    for residue in &topology.chains[0].residues {
        assert!(
            residue.name != "CYS" && residue.name != "CYH" && residue.name != "CYM",
            "missing_atoms should have no cysteine residues"
        );
    }

    let mut sanitizer = DisulfideSanitizer::new(&mut topology);
    let disulfides = sanitizer.run().expect("run should succeed");

    assert_eq!(disulfides.len(), 0, "Should detect no disulfides");

    // Verify residues are unchanged
    assert_eq!(topology.chains[0].residues[0].name, "ALA");
    assert_eq!(topology.chains[0].residues[1].name, "GLY");
    assert_eq!(topology.chains[0].residues[2].name, "SER");
}

#[test]
fn test_c2_chain_break() {
    // chain_break.pdb: ALA1, GLY2, SER3, ALA4 — no CYS residues.
    // Expected: no disulfides, all residues unchanged.
    let mut topology = load_topology("chain_break.pdb");

    // Verify initial state: no CYS residues
    for residue in &topology.chains[0].residues {
        assert!(
            residue.name != "CYS" && residue.name != "CYH" && residue.name != "CYM",
            "chain_break should have no cysteine residues"
        );
    }

    let mut sanitizer = DisulfideSanitizer::new(&mut topology);
    let disulfides = sanitizer.run().expect("run should succeed");

    assert_eq!(disulfides.len(), 0, "Should detect no disulfides");

    // Verify residues are unchanged
    assert_eq!(topology.chains[0].residues[0].name, "ALA");
    assert_eq!(topology.chains[0].residues[1].name, "GLY");
    assert_eq!(topology.chains[0].residues[2].name, "SER");
    assert_eq!(topology.chains[0].residues[3].name, "ALA");
}
