// Test C1 (ConFind precondition gate) conformance for all fixtures.
// These tests run under --no-default-features and cross-check the exact
// violations reported by check_preconditions for each fixture.
mod common;

use common::{load_backbone, load_processed, load_topology};
use proxide_confind::error::ConFindError;
use proxide_confind::precondition::{check_preconditions, require_preconditions, ViolationKind};
use proxide_confind::{
    extract_f64_backbone_with_options, BackboneOptions, ChainBreakPolicy, MissingAtomPolicy,
};

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
    // Default policy is now {Bridge, PerDihedral} (Mosaist semantics, debt
    // #1890): GLY2's own MissingBackboneAtom/UndefinedPhi/UndefinedPsi still
    // fire, and SER3 additionally gets its own UndefinedPhi because PerDihedral
    // no longer silently bridges phi(SER3) over GLY2's missing C using ALA1's C
    // (that bridging behaviour is now opt-in via MissingAtomPolicy::LegacyCompact
    // — see test_c1_missing_atoms_legacy_compact_matches_golden below).
    // Expected violations, in order:
    // - (ResidueIndex 1, res_id 2, GLY, MissingBackboneAtom "C") — Error
    // - (ResidueIndex 1, res_id 2, GLY, UndefinedPhi) — Error
    // - (ResidueIndex 1, res_id 2, GLY, UndefinedPsi) — Error
    // - (ResidueIndex 2, res_id 3, SER, UndefinedPhi) — Error
    // Total: 4 errors, 0 warnings
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

    // bb[1] (GLY, res_id 2) is the residue missing C and O in missing_atoms.pdb.
    assert!(
        backbone.bb[1].c.is_none(),
        "GLY residue (index 1) should be missing its C atom"
    );
    assert!(
        backbone.bb[1].o.is_none(),
        "GLY residue (index 1) should be missing its O atom"
    );

    let report = check_preconditions(&backbone);

    // Collect errors and warnings
    let errors: Vec<_> = report.errors().collect();
    let warnings: Vec<_> = report.warnings().collect();

    assert_eq!(errors.len(), 4, "Should have 4 errors");
    assert_eq!(warnings.len(), 0, "Should have 0 warnings");

    // Verify the EXACT ordered sequence of violations and their attribution.
    let expected: [(u32, i32, &str, ViolationKind); 4] = [
        (
            1,
            2,
            "GLY",
            ViolationKind::MissingBackboneAtom { atom: "C" },
        ),
        (1, 2, "GLY", ViolationKind::UndefinedPhi),
        (1, 2, "GLY", ViolationKind::UndefinedPsi),
        (2, 3, "SER", ViolationKind::UndefinedPhi),
    ];
    assert_eq!(
        errors.len(),
        expected.len(),
        "Expected exactly {} errors in order",
        expected.len()
    );
    for (i, (error, (exp_residue, exp_res_id, exp_res_name, exp_kind))) in
        errors.iter().zip(expected.iter()).enumerate()
    {
        assert_eq!(&error.kind, exp_kind, "Error {} kind mismatch", i);
        assert_eq!(
            error.residue.0, *exp_residue,
            "Error {} should be attributed to residue index {}",
            i, exp_residue
        );
        assert_eq!(
            error.id.res_id, *exp_res_id,
            "Error {} should be attributed to res_id {}",
            i, exp_res_id
        );
        assert_eq!(
            &error.res_name, exp_res_name,
            "Error {} should be attributed to res_name {}",
            i, exp_res_name
        );
    }

    let result = require_preconditions(&backbone);
    assert!(
        matches!(result, Err(ConFindError::PreconditionsFailed(4))),
        "Expected Err(ConFindError::PreconditionsFailed(4)), got {:?}",
        result
    );

    // debt #1890: under the new default {Bridge, PerDihedral} policy, SER3's phi
    // is undefined (GLY2's C, its true immediate predecessor, is missing) and its
    // omega is undefined too (needs GLY2's CA and C); GLY2's own omega remains
    // defined (it only needs ALA1's CA/C and GLY2's own N/CA, none of which are
    // missing).
    assert_eq!(
        backbone.bb[2].phi, 9999.0,
        "debt #1890: SER3 phi should be 9999.0 under PerDihedral (no bridging over GLY2's missing C)"
    );
    assert!(
        backbone.bb[2].omega.is_none(),
        "debt #1890: SER3 omega should be None (needs GLY2's CA/C, and GLY2's C is missing)"
    );
    assert!(
        !backbone.bb[2].is_cis_peptide,
        "debt #1890: SER3 is_cis_peptide should be false when omega is None"
    );
    let gly2_omega = backbone.bb[1]
        .omega
        .expect("debt #1890: GLY2 omega should be defined (doesn't need GLY2's own C)");
    assert!(
        gly2_omega.abs() > 150.0,
        "debt #1890: GLY2 omega should be a trans-like angle (|omega| > 150), got {gly2_omega}"
    );
}

#[test]
fn test_c1_missing_atoms_legacy_compact_matches_golden() {
    // debt #1890: MissingAtomPolicy::LegacyCompact reproduces base 316d1ab's
    // dense-bridging output bit-for-bit — see
    // crates/proxide-confind/tests/data/dihedral_golden_316d1ab.json,
    // fixtures."missing_atoms.pdb".extract_f64_backbone (phi_bits/psi_bits are
    // the IEEE-754 bit patterns of -180.0 and 9999.0; omega_bits 0 is 0.0).
    // This test replaces the old #1890 canary that asserted this same bridging
    // was ConFind's only (non-opt-in) behaviour.
    let processed = load_processed("missing_atoms.pdb");
    let opts = BackboneOptions::default().with_missing_atoms(MissingAtomPolicy::LegacyCompact);
    let backbone = extract_f64_backbone_with_options(&processed, &opts)
        .expect("LegacyCompact extract should not error");

    assert_eq!(
        backbone.bb[0].phi.to_bits(),
        9999.0f64.to_bits(),
        "ALA1 phi"
    );
    assert_eq!(
        backbone.bb[0].psi.to_bits(),
        (-180.0f64).to_bits(),
        "ALA1 psi (bridged)"
    );
    assert!(backbone.bb[0].omega.is_none(), "ALA1 omega (chain-first)");

    assert_eq!(
        backbone.bb[1].phi.to_bits(),
        9999.0f64.to_bits(),
        "GLY2 phi (untouched, excluded from dense array)"
    );
    assert_eq!(
        backbone.bb[1].psi.to_bits(),
        9999.0f64.to_bits(),
        "GLY2 psi (untouched, excluded from dense array)"
    );
    assert!(
        backbone.bb[1].omega.is_none(),
        "GLY2 omega (untouched, excluded from dense array)"
    );

    assert_eq!(
        backbone.bb[2].phi.to_bits(),
        (-180.0f64).to_bits(),
        "SER3 phi (bridged)"
    );
    assert_eq!(
        backbone.bb[2].psi.to_bits(),
        9999.0f64.to_bits(),
        "SER3 psi (chain-last)"
    );
    assert_eq!(
        backbone.bb[2].omega.map(f64::to_bits),
        Some(0.0f64.to_bits()),
        "SER3 omega (bridged)"
    );
    assert!(
        backbone.bb[2].is_cis_peptide,
        "SER3 is_cis_peptide (bridged omega near 0)"
    );

    let report = check_preconditions(&backbone);
    let errors: Vec<_> = report.errors().collect();
    let warnings: Vec<_> = report.warnings().collect();
    assert_eq!(warnings.len(), 0, "Should have 0 warnings");
    let expected_kinds = [
        ViolationKind::MissingBackboneAtom { atom: "C" },
        ViolationKind::UndefinedPhi,
        ViolationKind::UndefinedPsi,
    ];
    assert_eq!(
        errors.len(),
        expected_kinds.len(),
        "LegacyCompact should reproduce the old 3 violations (the #1890 canary)"
    );
    for (i, (error, expected_kind)) in errors.iter().zip(expected_kinds.iter()).enumerate() {
        assert_eq!(
            &error.kind, expected_kind,
            "LegacyCompact error {} kind mismatch",
            i
        );
        assert_eq!(
            error.residue.0, 1,
            "LegacyCompact error {} residue index",
            i
        );
        assert_eq!(error.id.res_id, 2, "LegacyCompact error {} res_id", i);
        assert_eq!(error.res_name, "GLY", "LegacyCompact error {} res_name", i);
    }

    let result = require_preconditions(&backbone);
    assert!(
        matches!(result, Err(ConFindError::PreconditionsFailed(3))),
        "Expected Err(ConFindError::PreconditionsFailed(3)), got {:?}",
        result
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
fn test_c1_chain_break_split() {
    // chain_break.pdb under ChainBreakPolicy::Split (debt #1890): the break sits
    // between GLY2 (index 1) and SER3 (index 2), so GLY2's psi and SER3's
    // phi/omega are severed even though all their atoms are physically present.
    // check_preconditions' step-5 exemption (spec D) means this does NOT show up
    // as UndefinedPhi/UndefinedPsi errors — only the usual ChainBreak warning.
    let processed = load_processed("chain_break.pdb");
    let opts = BackboneOptions::default().with_chain_breaks(ChainBreakPolicy::Split);
    let backbone = extract_f64_backbone_with_options(&processed, &opts)
        .expect("Split extract should not error");

    assert_eq!(
        backbone.bb[1].psi, 9999.0,
        "GLY2 psi should be severed by the Split policy"
    );
    assert_eq!(
        backbone.bb[2].phi, 9999.0,
        "SER3 phi should be severed by the Split policy"
    );
    assert!(
        backbone.bb[2].omega.is_none(),
        "SER3 omega should be severed by the Split policy"
    );

    let report = check_preconditions(&backbone);
    let errors: Vec<_> = report.errors().collect();
    let warnings: Vec<_> = report.warnings().collect();

    assert_eq!(
        errors.len(),
        0,
        "Split's severed dihedrals are exempted (their atoms are present): {:?}",
        errors
    );
    assert_eq!(
        warnings.len(),
        1,
        "Should still have exactly 1 ChainBreak warning"
    );
    assert!(
        matches!(warnings[0].kind, ViolationKind::ChainBreak { .. }),
        "the one warning should be ChainBreak"
    );

    assert!(
        require_preconditions(&backbone).is_ok(),
        "require_preconditions should pass under Split too"
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

#[test]
fn test_c1_topology_backbone_identity() {
    // Cross-API identity check: for every committed fixture, load_topology() and
    // load_backbone() must agree on the ordered sequence of (chain id, res_id, res_name).
    // These two loaders traverse independent code paths (Topology::from_raw_atom_data vs.
    // ProcessedStructure::from_raw + extract_f64_backbone); a silent divergence between
    // them would mean the precondition-check identity used elsewhere in this file
    // (residue index / res_id / res_name) doesn't actually match what load_topology reports.
    for name in &[
        "clean_small.pdb",
        "disulfide_pair.pdb",
        "missing_atoms.pdb",
        "chain_break.pdb",
        "truncated_sidechain.pdb",
        "no_hydrogens.pdb",
    ] {
        let topology = load_topology(name);
        let topo_seq: Vec<(String, i32, String)> = topology
            .chains
            .iter()
            .flat_map(|chain| {
                chain
                    .residues
                    .iter()
                    .map(move |r| (chain.id.clone(), r.res_id, r.name.clone()))
            })
            .collect();

        let backbone = load_backbone(name);
        let backbone_seq: Vec<(String, i32, String)> = backbone
            .ids
            .iter()
            .zip(backbone.bb.iter())
            .map(|(id, rb)| (id.chain_id.clone(), id.res_id, rb.res_name.clone()))
            .collect();

        assert_eq!(
            topo_seq, backbone_seq,
            "load_topology and load_backbone disagree on the (chain, res_id, res_name) \
             sequence for fixture {}: topology={:?} backbone={:?}",
            name, topo_seq, backbone_seq
        );
    }
}
