// Test C1 (ConFind precondition gate) conformance for all fixtures.
// These tests run under --no-default-features and cross-check the exact
// violations reported by check_preconditions for each fixture.
mod common;

use common::{load_backbone, load_topology};
use proxide_confind::precondition::{check_preconditions, require_preconditions};

#[test]
fn test_c1_clean_small() {
    // clean_small.pdb: 3 residues (ALA1, GLY2, SER3), all complete, no violations.
    let topology = load_topology("clean_small.pdb");
    assert_eq!(
        (topology.chains.len(), topology.chains[0].residues.len()),
        (1, 3),
        "clean_small should have 1 chain with 3 residues"
    );

    // Verify (chain, res_id, res_name) sequence
    let seq: Vec<(String, i32, String)> = topology.chains[0]
        .residues
        .iter()
        .map(|r| ("A".to_string(), r.res_id, r.name.clone()))
        .collect();
    assert_eq!(
        seq,
        vec![
            ("A".to_string(), 1, "ALA".to_string()),
            ("A".to_string(), 2, "GLY".to_string()),
            ("A".to_string(), 3, "SER".to_string()),
        ]
    );

    let backbone = load_backbone("clean_small.pdb");
    let report = check_preconditions(&backbone);

    assert!(
        report.is_clean(),
        "clean_small should have no violations, got: {:?}",
        report.violations
    );
    assert!(
        require_preconditions(&backbone).is_ok(),
        "require_preconditions should pass"
    );
}

#[test]
fn test_c1_disulfide_pair() {
    // disulfide_pair.pdb: 2 residues (CYS1, CYS2), all complete, no violations.
    let topology = load_topology("disulfide_pair.pdb");
    assert_eq!(
        (topology.chains.len(), topology.chains[0].residues.len()),
        (1, 2),
        "disulfide_pair should have 1 chain with 2 residues"
    );

    let seq: Vec<(String, i32, String)> = topology.chains[0]
        .residues
        .iter()
        .map(|r| ("A".to_string(), r.res_id, r.name.clone()))
        .collect();
    assert_eq!(
        seq,
        vec![
            ("A".to_string(), 1, "CYS".to_string()),
            ("A".to_string(), 2, "CYS".to_string()),
        ]
    );

    let backbone = load_backbone("disulfide_pair.pdb");
    let report = check_preconditions(&backbone);

    assert!(
        report.is_clean(),
        "disulfide_pair should have no violations, got: {:?}",
        report.violations
    );
    assert!(
        require_preconditions(&backbone).is_ok(),
        "require_preconditions should pass"
    );
}

#[test]
fn test_c1_missing_atoms() {
    // missing_atoms.pdb: 3 residues (ALA1, GLY2 with missing C and O, SER3).
    // Expected violations:
    // - (ResidueIndex 1, res_id 2, GLY, MissingBackboneAtom "C") — Error
    // - (ResidueIndex 1, res_id 2, GLY, UndefinedPhi) — Error
    // - (ResidueIndex 1, res_id 2, GLY, UndefinedPsi) — Error
    // Total: 3 errors, 0 warnings
    let topology = load_topology("missing_atoms.pdb");
    assert_eq!(
        (topology.chains.len(), topology.chains[0].residues.len()),
        (1, 3),
        "missing_atoms should have 1 chain with 3 residues"
    );

    let seq: Vec<(String, i32, String)> = topology.chains[0]
        .residues
        .iter()
        .map(|r| ("A".to_string(), r.res_id, r.name.clone()))
        .collect();
    assert_eq!(
        seq,
        vec![
            ("A".to_string(), 1, "ALA".to_string()),
            ("A".to_string(), 2, "GLY".to_string()),
            ("A".to_string(), 3, "SER".to_string()),
        ]
    );

    let backbone = load_backbone("missing_atoms.pdb");
    let report = check_preconditions(&backbone);

    // Collect errors and warnings
    let errors: Vec<_> = report.errors().collect();
    let warnings: Vec<_> = report.warnings().collect();

    assert_eq!(errors.len(), 3, "Should have 3 errors");
    assert_eq!(warnings.len(), 0, "Should have 0 warnings");

    // Verify error kinds match expected violations for residue index 1 (GLY, res_id 2)
    let error_kinds: Vec<_> = errors.iter().map(|e| &e.kind).collect();
    let has_missing_c = error_kinds.iter().any(|k| {
        matches!(
            k,
            proxide_confind::precondition::ViolationKind::MissingBackboneAtom { atom: "C" }
        )
    });
    let has_undef_phi = error_kinds.iter().any(|k| {
        matches!(
            k,
            proxide_confind::precondition::ViolationKind::UndefinedPhi
        )
    });
    let has_undef_psi = error_kinds.iter().any(|k| {
        matches!(
            k,
            proxide_confind::precondition::ViolationKind::UndefinedPsi
        )
    });

    assert!(
        has_missing_c,
        "Should have MissingBackboneAtom(C) violation"
    );
    assert!(has_undef_phi, "Should have UndefinedPhi violation");
    assert!(has_undef_psi, "Should have UndefinedPsi violation");

    assert!(
        require_preconditions(&backbone).is_err(),
        "require_preconditions should fail with 3 errors"
    );

    // CANARY for debt #1890: verify the gap-bridging behaviour.
    // bb[2] (SER3) should have phi != 9999.0 and is_cis_peptide == true.
    assert_ne!(
        backbone.bb[2].phi, 9999.0,
        "CANARY (debt #1890): bb[2].phi should not be 9999.0 (gap-bridging is active)"
    );
    assert!(
        backbone.bb[2].is_cis_peptide,
        "CANARY (debt #1890): bb[2].is_cis_peptide should be true (gap-bridging is active)"
    );
}

#[test]
fn test_c1_chain_break() {
    // chain_break.pdb: 4 residues (ALA1, GLY2, SER3, ALA4).
    // Gap between CA2 and CA3 should trigger one ChainBreak warning.
    // Expected: 0 errors, 1 warning (ChainBreak)
    let topology = load_topology("chain_break.pdb");
    assert_eq!(
        (topology.chains.len(), topology.chains[0].residues.len()),
        (1, 4),
        "chain_break should have 1 chain with 4 residues"
    );

    let seq: Vec<(String, i32, String)> = topology.chains[0]
        .residues
        .iter()
        .map(|r| ("A".to_string(), r.res_id, r.name.clone()))
        .collect();
    assert_eq!(
        seq,
        vec![
            ("A".to_string(), 1, "ALA".to_string()),
            ("A".to_string(), 2, "GLY".to_string()),
            ("A".to_string(), 3, "SER".to_string()),
            ("A".to_string(), 4, "ALA".to_string()),
        ]
    );

    let backbone = load_backbone("chain_break.pdb");
    let report = check_preconditions(&backbone);

    let errors: Vec<_> = report.errors().collect();
    let warnings: Vec<_> = report.warnings().collect();

    assert_eq!(errors.len(), 0, "Should have 0 errors");
    assert_eq!(warnings.len(), 1, "Should have 1 warning (ChainBreak)");

    // Verify the warning is a ChainBreak on residue index 2 (SER, res_id 3)
    let warning = &warnings[0];
    assert_eq!(
        warning.residue.0, 2,
        "ChainBreak should be for residue index 2"
    );
    assert_eq!(warning.id.res_id, 3, "ChainBreak should be for res_id 3");
    assert_eq!(warning.res_name, "SER", "ChainBreak should be for SER");

    // Verify the gap_to_next_ca distance is approximately correct.
    // chain_break.pdb: CA2 at [4.797, 1.566, 0.0], CA3 at [12.916, 3.964, 0.0]
    // Distance ≈ sqrt((12.916 - 4.797)^2 + (3.964 - 1.566)^2) ≈ sqrt(65.95 + 5.76) ≈ 8.466
    if let proxide_confind::precondition::ViolationKind::ChainBreak { gap_to_next_ca } =
        warning.kind
    {
        assert!(
            (gap_to_next_ca - 8.466).abs() < 0.01,
            "gap_to_next_ca should be ~8.466 Å, got {:.3}",
            gap_to_next_ca
        );
    } else {
        panic!("Expected ChainBreak violation");
    }

    assert!(
        require_preconditions(&backbone).is_ok(),
        "require_preconditions should pass (warnings don't fail)"
    );
}

#[test]
fn test_c1_truncated_sidechain() {
    // truncated_sidechain.pdb: 1 residue (LYS1), complete backbone, no violations.
    let topology = load_topology("truncated_sidechain.pdb");
    assert_eq!(
        (topology.chains.len(), topology.chains[0].residues.len()),
        (1, 1),
        "truncated_sidechain should have 1 chain with 1 residue"
    );

    let seq: Vec<(String, i32, String)> = topology.chains[0]
        .residues
        .iter()
        .map(|r| ("A".to_string(), r.res_id, r.name.clone()))
        .collect();
    assert_eq!(seq, vec![("A".to_string(), 1, "LYS".to_string())]);

    let backbone = load_backbone("truncated_sidechain.pdb");
    let report = check_preconditions(&backbone);

    assert!(
        report.is_clean(),
        "truncated_sidechain should have no violations (sidechain truncation is not a backbone violation), got: {:?}",
        report.violations
    );
    assert!(
        require_preconditions(&backbone).is_ok(),
        "require_preconditions should pass"
    );
}
