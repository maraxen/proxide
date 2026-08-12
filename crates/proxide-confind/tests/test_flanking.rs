mod common;

use common::{load_rotlib_or_skip, make_synthetic_backbone, make_two_chain_backbone};
use proxide_confind::coords::ResidueIndex;
use proxide_confind::ConFind;

#[test]
#[ignore = "requires rotamer library — set RLIB env var to run"]
fn flanking_same_chain_adjacent_skipped() {
    let rotlib = match load_rotlib_or_skip() {
        Some(r) => r,
        None => {
            println!("RLIB not set, skipping test");
            return;
        }
    };

    // 3-residue chain, spacing 3.8 Å
    let backbone = make_synthetic_backbone(3, 3.8);
    let confind = ConFind::new(rotlib, backbone.clone(), false);

    // ignore_flanking=1, dcut_bb=15.0
    let residues = vec![ResidueIndex(0), ResidueIndex(1), ResidueIndex(2)];
    let result = confind.bb_interaction(&residues, 15.0, 1);
    assert!(result.is_ok(), "bb_interaction should succeed");

    let contact = result.unwrap();

    // With ignore_flanking=1, pairs (0,1) and (1,2) should NOT be in results
    let pairs_set: std::collections::HashSet<_> = contact.pairs.iter().cloned().collect();
    assert!(
        !pairs_set.contains(&(ResidueIndex(0), ResidueIndex(1))),
        "Adjacent pair (0,1) should be skipped"
    );
    assert!(
        !pairs_set.contains(&(ResidueIndex(1), ResidueIndex(2))),
        "Adjacent pair (1,2) should be skipped"
    );
}

#[test]
#[ignore = "requires rotamer library — set RLIB env var to run"]
fn flanking_same_chain_nonadjacent_included() {
    let rotlib = match load_rotlib_or_skip() {
        Some(r) => r,
        None => {
            println!("RLIB not set, skipping test");
            return;
        }
    };

    // 3-residue chain, spacing 3.8 Å
    let backbone = make_synthetic_backbone(3, 3.8);
    let confind = ConFind::new(rotlib, backbone.clone(), false);

    let residues = vec![ResidueIndex(0), ResidueIndex(1), ResidueIndex(2)];
    let result = confind.bb_interaction(&residues, 15.0, 1);
    assert!(result.is_ok());

    let contact = result.unwrap();

    // With ignore_flanking=1, pair (0,2) IS in results (not adjacent)
    let pairs_set: std::collections::HashSet<_> = contact.pairs.iter().cloned().collect();
    assert!(
        pairs_set.contains(&(ResidueIndex(0), ResidueIndex(2))),
        "Non-adjacent pair (0,2) should be included"
    );
}

#[test]
#[ignore = "requires rotamer library — set RLIB env var to run"]
fn flanking_zero_same_chain_all_included() {
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
    let result = confind.bb_interaction(&residues, 15.0, 0);
    assert!(result.is_ok());

    let contact = result.unwrap();

    // With ignore_flanking=0, all pairs within dcut should be included
    let pairs_set: std::collections::HashSet<_> = contact.pairs.iter().cloned().collect();
    assert!(
        pairs_set.contains(&(ResidueIndex(0), ResidueIndex(1))),
        "Pair (0,1) should be included with ignore_flanking=0"
    );
    assert!(
        pairs_set.contains(&(ResidueIndex(0), ResidueIndex(2))),
        "Pair (0,2) should be included with ignore_flanking=0"
    );
    assert!(
        pairs_set.contains(&(ResidueIndex(1), ResidueIndex(2))),
        "Pair (1,2) should be included with ignore_flanking=0"
    );
}

#[test]
#[ignore = "requires rotamer library — set RLIB env var to run"]
fn flanking_large_ignore_value() {
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
    // ignore_flanking=100 should skip all pairs on same chain
    let result = confind.bb_interaction(&residues, 15.0, 100);
    assert!(result.is_ok());

    let contact = result.unwrap();

    // No pairs should be reported (all are too close via flanking rule)
    assert_eq!(
        contact.pairs.len(),
        0,
        "With large ignore_flanking, no same-chain pairs should be reported"
    );
}

#[test]
#[ignore = "requires rotamer library — set RLIB env var to run"]
fn flanking_cross_chain_always_included() {
    let rotlib = match load_rotlib_or_skip() {
        Some(r) => r,
        None => {
            println!("RLIB not set, skipping test");
            return;
        }
    };

    // Two chains, residue 0 from each placed 3.8 Å apart
    let backbone = make_two_chain_backbone(1, 3.8, [0.0, 0.0, 0.0]);
    let confind = ConFind::new(rotlib, backbone.clone(), false);

    let residues = vec![ResidueIndex(0), ResidueIndex(1)];
    let result = confind.bb_interaction(&residues, 15.0, 1);
    assert!(result.is_ok());

    let contact = result.unwrap();

    // Cross-chain pair should appear even with ignore_flanking=1
    let pairs_set: std::collections::HashSet<_> = contact.pairs.iter().cloned().collect();
    assert!(
        pairs_set.contains(&(ResidueIndex(0), ResidueIndex(1))),
        "Cross-chain pair should appear"
    );
}
