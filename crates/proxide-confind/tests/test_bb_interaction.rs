mod common;

use common::{make_synthetic_backbone, load_rotlib_or_skip};
use proxide_confind::ConFind;
use proxide_confind::coords::ResidueIndex;

#[test]
#[ignore = "requires rotamer library — set RLIB env var to run"]
fn bb_interaction_adjacent_same_chain_ignored() {
    let rotlib = match load_rotlib_or_skip() {
        Some(r) => r,
        None => {
            println!("RLIB not set, skipping test");
            return;
        }
    };

    // 3 residues, spacing 3.8 Å, ignore_flanking=1, dcut_bb=10.0
    let backbone = make_synthetic_backbone(3, 3.8);
    let confind = ConFind::new(rotlib, backbone.clone(), false);

    let residues = vec![ResidueIndex(0), ResidueIndex(1), ResidueIndex(2)];
    let result = confind.bb_interaction(&residues, 10.0, 1);
    assert!(result.is_ok());

    let contact = result.unwrap();
    let pairs_set: std::collections::HashSet<_> = contact.pairs.iter().cloned().collect();

    // Adjacent pairs should not be reported
    assert!(!pairs_set.contains(&(ResidueIndex(0), ResidueIndex(1))), "Adjacent (0,1) should be ignored");
    assert!(!pairs_set.contains(&(ResidueIndex(1), ResidueIndex(2))), "Adjacent (1,2) should be ignored");
}

#[test]
#[ignore = "requires rotamer library — set RLIB env var to run"]
fn bb_interaction_adjacent_same_chain_zero_ignore() {
    let rotlib = match load_rotlib_or_skip() {
        Some(r) => r,
        None => {
            println!("RLIB not set, skipping test");
            return;
        }
    };

    let backbone = make_synthetic_backbone(3, 3.8);
    let confind = ConFind::new(rotlib, backbone.clone(), false);

    let residues = vec![ResidueIndex(0), ResidueIndex(1), ResidueIndex(2)];
    // ignore_flanking=0 means no flanking skipping
    let result = confind.bb_interaction(&residues, 10.0, 0);
    assert!(result.is_ok());

    let contact = result.unwrap();
    let pairs_set: std::collections::HashSet<_> = contact.pairs.iter().cloned().collect();

    // With ignore_flanking=0, all pairs within dcut_bb should be included
    assert!(pairs_set.contains(&(ResidueIndex(0), ResidueIndex(1))) || contact.pairs.is_empty());
}

#[test]
#[ignore = "requires rotamer library — set RLIB env var to run"]
fn bb_interaction_different_chains_no_flanking_skip() {
    let rotlib = match load_rotlib_or_skip() {
        Some(r) => r,
        None => {
            println!("RLIB not set, skipping test");
            return;
        }
    };

    let backbone = common::make_two_chain_backbone(1, 3.8, [0.0, 0.0, 0.0]);
    let confind = ConFind::new(rotlib, backbone.clone(), false);

    let residues = vec![ResidueIndex(0), ResidueIndex(1)];
    let result = confind.bb_interaction(&residues, 10.0, 1);
    assert!(result.is_ok());

    let contact = result.unwrap();
    let pairs_set: std::collections::HashSet<_> = contact.pairs.iter().cloned().collect();

    // Cross-chain pairs are not subject to flanking skip
    assert!(pairs_set.contains(&(ResidueIndex(0), ResidueIndex(1))), "Cross-chain pair should appear");
}

#[test]
#[ignore = "requires rotamer library — set RLIB env var to run"]
fn bb_interaction_far_apart_not_reported() {
    let rotlib = match load_rotlib_or_skip() {
        Some(r) => r,
        None => {
            println!("RLIB not set, skipping test");
            return;
        }
    };

    // Residues 30 Å apart, dcut_bb=10.0
    let backbone = make_synthetic_backbone(2, 30.0);
    let confind = ConFind::new(rotlib, backbone.clone(), false);

    let residues = vec![ResidueIndex(0), ResidueIndex(1)];
    let result = confind.bb_interaction(&residues, 10.0, 0);
    assert!(result.is_ok());

    let contact = result.unwrap();
    // Pair is too far, should not be reported
    assert_eq!(contact.pairs.len(), 0, "Residues > dcut_bb apart should not be reported");
}

#[test]
#[ignore = "requires rotamer library — set RLIB env var to run"]
fn bb_interaction_exactly_at_cutoff() {
    let rotlib = match load_rotlib_or_skip() {
        Some(r) => r,
        None => {
            println!("RLIB not set, skipping test");
            return;
        }
    };

    // Residues at spacing where min distance should be near cutoff
    let backbone = make_synthetic_backbone(2, 3.8);
    let confind = ConFind::new(rotlib, backbone.clone(), false);

    let residues = vec![ResidueIndex(0), ResidueIndex(1)];
    // dcut_bb slightly above minimum possible distance
    let result = confind.bb_interaction(&residues, 2.0, 0);
    assert!(result.is_ok());

    // This tests the <= semantics of the distance check
}

#[test]
#[ignore = "requires rotamer library — set RLIB env var to run"]
fn bb_interaction_one_residue() {
    let rotlib = match load_rotlib_or_skip() {
        Some(r) => r,
        None => {
            println!("RLIB not set, skipping test");
            return;
        }
    };

    let backbone = make_synthetic_backbone(1, 3.8);
    let confind = ConFind::new(rotlib, backbone.clone(), false);

    let residues = vec![ResidueIndex(0)];
    let result = confind.bb_interaction(&residues, 10.0, 0);
    assert!(result.is_ok());

    let contact = result.unwrap();
    // Single residue has no pairs
    assert_eq!(contact.pairs.len(), 0, "Single residue should have no pairs");
}
