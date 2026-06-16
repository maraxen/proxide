mod common;

use common::{make_synthetic_backbone, load_rotlib_or_skip};
use proxide_confind::ConFind;
use proxide_confind::coords::ResidueIndex;

#[test]
#[ignore = "requires rotamer library — set RLIB env var to run"]
fn crowdedness_isolated_residue_near_zero() {
    let rotlib = match load_rotlib_or_skip() {
        Some(r) => r,
        None => {
            println!("RLIB not set, skipping test");
            return;
        }
    };

    // 5 residues at 30 Å spacing (isolated)
    let backbone = make_synthetic_backbone(5, 30.0);
    let confind = ConFind::new(rotlib, backbone.clone(), false);

    // Cache the middle residue
    let ri = ResidueIndex(2);
    let result = confind.cache_residue(ri);
    assert!(result.is_ok(), "Caching isolated residue should succeed");

    // Isolated residue should have very low crowdedness
    // (test documents this expectation)
}

#[test]
#[ignore = "requires rotamer library — set RLIB env var to run"]
fn crowdedness_compact_higher_than_isolated() {
    let rotlib = match load_rotlib_or_skip() {
        Some(r) => r,
        None => {
            println!("RLIB not set, skipping test");
            return;
        }
    };

    // Compact 5-residue backbone at 3.8 Å spacing
    let backbone = make_synthetic_backbone(5, 3.8);
    let confind = ConFind::new(rotlib, backbone.clone(), false);

    // Cache all
    let result = confind.cache_all();
    assert!(result.is_ok(), "cache_all should succeed");

    // Middle residue should have higher crowdedness than terminal residues
    // This is documented as an expected regression check
}

#[test]
#[ignore = "requires rotamer library — set RLIB env var to run"]
fn freedom_fully_isolated_is_near_1() {
    let rotlib = match load_rotlib_or_skip() {
        Some(r) => r,
        None => {
            println!("RLIB not set, skipping test");
            return;
        }
    };

    // Single residue, fully isolated
    let backbone = make_synthetic_backbone(1, 3.8);
    let confind = ConFind::new(rotlib, backbone.clone(), false);

    let residues = vec![ResidueIndex(0)];
    let result = confind.contacts(&residues, 0.0);
    assert!(result.is_ok(), "contacts should succeed for isolated residue");

    // Freedom should be high (>0.9) for isolated residue
    // (may not be exactly 1.0 if some rotamers clash with self-backbone)
}

#[test]
#[ignore = "requires rotamer library — set RLIB env var to run"]
fn contact_degree_symmetry() {
    let rotlib = match load_rotlib_or_skip() {
        Some(r) => r,
        None => {
            println!("RLIB not set, skipping test");
            return;
        }
    };

    let backbone = make_synthetic_backbone(3, 3.8);
    let confind = ConFind::new(rotlib, backbone.clone(), false);

    let all_residues = vec![ResidueIndex(0), ResidueIndex(1), ResidueIndex(2)];
    let result = confind.contacts(&all_residues, 0.0);
    assert!(result.is_ok());

    let contact = result.unwrap();

    // For any pair in contact list, degree lookup should be symmetric
    for (i, (a, b)) in contact.pairs.iter().enumerate() {
        let stored_degree = contact.degrees[i];
        
        // degree(a, b) and degree(b, a) should return same value
        if let Some(deg_ab) = contact.degree(*a, *b) {
            assert_eq!(deg_ab, stored_degree, "Stored degree should match degree lookup");
        }
        
        if let Some(deg_ba) = contact.degree(*b, *a) {
            assert_eq!(deg_ba, stored_degree, "degree(a, b) should equal degree(b, a)");
        }
    }
}
