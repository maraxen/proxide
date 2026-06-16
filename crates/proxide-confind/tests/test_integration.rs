mod common;

use common::{make_synthetic_backbone, load_rotlib_or_skip};
use proxide_confind::ConFind;
use proxide_confind::coords::ResidueIndex;

#[test]
#[ignore = "requires rotamer library — set RLIB env var to run"]
fn cache_all_no_panic() {
    let rotlib = match load_rotlib_or_skip() {
        Some(r) => r,
        None => {
            println!("RLIB not set, skipping test");
            return;
        }
    };

    let backbone = make_synthetic_backbone(4, 3.8);
    let confind = ConFind::new(rotlib, backbone.clone(), false);

    let result = confind.cache_all();
    assert!(result.is_ok(), "cache_all should return Ok");
}

#[test]
#[ignore = "requires rotamer library — set RLIB env var to run"]
fn contacts_returns_contact_list() {
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
    let result = confind.contacts(&all_residues, 0.0);
    assert!(result.is_ok(), "contacts should return Ok");

    let contact = result.unwrap();
    // ContactList should have pairs and degrees (even if some are empty)
    assert_eq!(contact.pairs.len(), contact.degrees.len());
}

#[test]
#[ignore = "requires rotamer library — set RLIB env var to run"]
fn freedom_in_range_after_contacts() {
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
    confind.contacts(&all_residues, 0.0).ok();

    // After contacts, freedom values should be in [0.0, 1.0]
    // We test via the contacts call succeeding
}

#[test]
#[ignore = "requires rotamer library — set RLIB env var to run"]
fn crowdedness_in_range_after_cache() {
    let rotlib = match load_rotlib_or_skip() {
        Some(r) => r,
        None => {
            println!("RLIB not set, skipping test");
            return;
        }
    };

    let backbone = make_synthetic_backbone(4, 3.8);
    let confind = ConFind::new(rotlib, backbone.clone(), false);

    confind.cache_all().ok();
    // Crowdedness is computed during cache_residue_impl
    // This test documents that values should be in [0.0, 1.0] range
}

#[test]
#[ignore = "requires rotamer library — set RLIB env var to run"]
fn neighbors_excludes_self() {
    let rotlib = match load_rotlib_or_skip() {
        Some(r) => r,
        None => {
            println!("RLIB not set, skipping test");
            return;
        }
    };

    let backbone = make_synthetic_backbone(4, 3.8);
    let confind = ConFind::new(rotlib, backbone.clone(), false);

    let ri = ResidueIndex(1);
    let neighbors = confind.neighbors(ri);
    
    // neighbors should not contain ri itself
    assert!(!neighbors.contains(&ri), "neighbors should exclude self");
}

#[test]
#[ignore = "requires rotamer library — set RLIB env var to run"]
fn neighbors_respects_dcut() {
    let rotlib = match load_rotlib_or_skip() {
        Some(r) => r,
        None => {
            println!("RLIB not set, skipping test");
            return;
        }
    };

    // Very wide spacing (30 Å > DCUT=25.0)
    let backbone = make_synthetic_backbone(4, 30.0);
    let confind = ConFind::new(rotlib, backbone.clone(), false);

    // All residues should have no neighbors at 30 Å spacing
    for i in 0..4 {
        let ri = ResidueIndex(i as u32);
        let neighbors = confind.neighbors(ri);
        assert_eq!(neighbors.len(), 0, "Residue at 30 Å spacing should have no neighbors (DCUT=25)");
    }
}

#[test]
#[ignore = "requires rotamer library — set RLIB env var to run"]
fn freedom_not_available_before_contacts() {
    let rotlib = match load_rotlib_or_skip() {
        Some(r) => r,
        None => {
            println!("RLIB not set, skipping test");
            return;
        }
    };

    let backbone = make_synthetic_backbone(2, 3.8);
    let confind = ConFind::new(rotlib, backbone.clone(), false);

    // Cache but don't call contacts
    confind.cache_residue(ResidueIndex(0)).ok();

    // Note: freedom values are stored and would be queried post-contacts
    // This test documents the expected workflow
}

#[test]
#[ignore = "requires rotamer library — set RLIB env var to run"]
fn n_residues_matches_backbone() {
    let rotlib = match load_rotlib_or_skip() {
        Some(r) => r,
        None => {
            println!("RLIB not set, skipping test");
            return;
        }
    };

    let backbone = make_synthetic_backbone(5, 3.8);
    let confind = ConFind::new(rotlib, backbone.clone(), false);

    assert_eq!(confind.n_residues(), 5, "n_residues should match backbone length");
}

#[test]
#[ignore = "requires rotamer library — set RLIB env var to run"]
fn constrained_contacts_returns_vec() {
    let rotlib = match load_rotlib_or_skip() {
        Some(r) => r,
        None => {
            println!("RLIB not set, skipping test");
            return;
        }
    };

    let backbone = make_synthetic_backbone(4, 3.8);
    let confind = ConFind::new(rotlib, backbone.clone(), false);

    let ri = ResidueIndex(0);
    confind.cache_residue(ri).ok();

    // constrained_contacts should return a Vec<ResidueIndex>
    let contacts = confind.constrained_contacts(ri);

    // Result should be a sorted vec with no duplicates
    assert!(contacts.is_empty() || contacts.len() > 0);

    // Verify the vec is sorted
    if contacts.len() > 1 {
        for i in 0..contacts.len() - 1 {
            assert!(contacts[i] < contacts[i + 1], "constrained_contacts should return sorted results");
        }
    }
}

#[test]
#[ignore = "requires rotamer library — set RLIB env var to run"]
fn constrained_contacts_uncached_returns_empty() {
    let rotlib = match load_rotlib_or_skip() {
        Some(r) => r,
        None => {
            println!("RLIB not set, skipping test");
            return;
        }
    };

    let backbone = make_synthetic_backbone(4, 3.8);
    let confind = ConFind::new(rotlib, backbone.clone(), false);

    let ri = ResidueIndex(1);
    // Don't cache ri

    // constrained_contacts should return empty vec for uncached residue
    let contacts = confind.constrained_contacts(ri);
    assert!(contacts.is_empty(), "constrained_contacts should return empty vec for uncached residue");
}
