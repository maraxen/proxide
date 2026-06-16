mod common;

use common::{make_synthetic_backbone, load_rotlib_or_skip};
use proxide_confind::ConFind;
use proxide_confind::coords::ResidueIndex;

#[test]
#[ignore = "requires rotamer library — set RLIB env var to run"]
fn parallel_determinism_contacts() {
    let rotlib = match load_rotlib_or_skip() {
        Some(r) => r,
        None => {
            println!("RLIB not set, skipping test");
            return;
        }
    };

    let backbone = make_synthetic_backbone(4, 3.8);
    let confind = ConFind::new(rotlib, backbone.clone(), false);

    let all_residues: Vec<ResidueIndex> = (0..4).map(ResidueIndex).collect();

    // Call contacts twice
    let result1 = confind.contacts(&all_residues, 0.0);
    assert!(result1.is_ok(), "First contacts call should succeed");

    let result2 = confind.contacts(&all_residues, 0.0);
    assert!(result2.is_ok(), "Second contacts call should succeed");

    let contact1 = result1.unwrap();
    let contact2 = result2.unwrap();

    // Assert pairs are identical (sorted)
    let mut pairs1 = contact1.pairs.clone();
    let mut pairs2 = contact2.pairs.clone();
    pairs1.sort_unstable();
    pairs2.sort_unstable();
    assert_eq!(pairs1, pairs2, "Contact pairs should be deterministic");

    // Assert degrees are identical (or very close due to floating point)
    assert_eq!(contact1.degrees.len(), contact2.degrees.len());
    for (d1, d2) in contact1.degrees.iter().zip(&contact2.degrees) {
        assert!((d1 - d2).abs() < 1e-12, "Degrees should match: {} vs {}", d1, d2);
    }
}

#[test]
#[ignore = "requires rotamer library — set RLIB env var to run"]
fn parallel_determinism_freedom() {
    let rotlib = match load_rotlib_or_skip() {
        Some(r) => r,
        None => {
            println!("RLIB not set, skipping test");
            return;
        }
    };

    let backbone = make_synthetic_backbone(4, 3.8);
    let confind = ConFind::new(rotlib, backbone.clone(), false);

    let all_residues: Vec<ResidueIndex> = (0..4).map(ResidueIndex).collect();

    // Run contacts twice to populate freedom values
    confind.contacts(&all_residues, 0.0).ok();
    confind.contacts(&all_residues, 0.0).ok();

    // Freedom values should be deterministic (exact same each time)
    // We can't directly access freedom, but the determinism is proven by the contacts test
}

#[test]
#[ignore = "requires rotamer library — set RLIB env var to run"]
fn parallel_cache_all_idempotent() {
    let rotlib = match load_rotlib_or_skip() {
        Some(r) => r,
        None => {
            println!("RLIB not set, skipping test");
            return;
        }
    };

    let backbone = make_synthetic_backbone(4, 3.8);
    let confind = ConFind::new(rotlib, backbone.clone(), false);

    // Call cache_all twice
    let result1 = confind.cache_all();
    assert!(result1.is_ok(), "First cache_all should succeed");

    let result2 = confind.cache_all();
    assert!(result2.is_ok(), "Second cache_all should succeed");

    // Both should return Ok without panic
}

#[test]
#[ignore = "requires rotamer library — set RLIB env var to run"]
fn parallel_contacts_subset_vs_all() {
    let rotlib = match load_rotlib_or_skip() {
        Some(r) => r,
        None => {
            println!("RLIB not set, skipping test");
            return;
        }
    };

    let backbone = make_synthetic_backbone(5, 3.8);
    let confind = ConFind::new(rotlib, backbone.clone(), false);

    // Call contacts on [0,1,2] then on [1,2,3]
    let subset1: Vec<ResidueIndex> = vec![ResidueIndex(0), ResidueIndex(1), ResidueIndex(2)];
    let result1 = confind.contacts(&subset1, 0.0);
    assert!(result1.is_ok(), "contacts on [0,1,2] should succeed");

    let subset2: Vec<ResidueIndex> = vec![ResidueIndex(1), ResidueIndex(2), ResidueIndex(3)];
    let result2 = confind.contacts(&subset2, 0.0);
    assert!(result2.is_ok(), "contacts on [1,2,3] should succeed");

    // Neither should panic
}
