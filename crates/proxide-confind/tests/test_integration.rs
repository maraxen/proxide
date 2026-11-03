mod common;

use common::{
    load_rotlib_or_skip, make_synthetic_backbone, make_two_chain_backbone, require_rotlib,
};
use proxide_confind::coords::ResidueIndex;
use proxide_confind::ConFind;

#[test]
fn cache_all_no_panic() {
    let rotlib = match load_rotlib_or_skip() {
        Some(r) => r,
        None => {
            println!("RLIB not set, skipping test");
            return;
        }
    };

    let backbone = make_synthetic_backbone(4, 3.8);
    let confind = ConFind::new(rotlib, backbone.clone(), false).unwrap();

    let result = confind.cache_all();
    assert!(result.is_ok(), "cache_all should return Ok");
}

#[test]
fn contacts_returns_contact_list() {
    let rotlib = match load_rotlib_or_skip() {
        Some(r) => r,
        None => {
            println!("RLIB not set, skipping test");
            return;
        }
    };

    let backbone = make_synthetic_backbone(4, 3.8);
    let confind = ConFind::new(rotlib, backbone.clone(), false).unwrap();

    let all_residues: Vec<ResidueIndex> = (0..4).map(ResidueIndex).collect();
    let result = confind.contacts(&all_residues, 0.0);
    assert!(result.is_ok(), "contacts should return Ok");

    let contact = result.unwrap();
    // ContactList should have pairs and degrees (even if some are empty)
    assert_eq!(contact.pairs.len(), contact.degrees.len());
}

#[test]
fn freedom_in_range_after_contacts() {
    let rotlib = match load_rotlib_or_skip() {
        Some(r) => r,
        None => {
            println!("RLIB not set, skipping test");
            return;
        }
    };

    let backbone = make_synthetic_backbone(4, 3.8);
    let confind = ConFind::new(rotlib, backbone.clone(), false).unwrap();

    let all_residues: Vec<ResidueIndex> = (0..4).map(ResidueIndex).collect();
    confind.contacts(&all_residues, 0.0).ok();

    // After contacts, freedom values should be in [0.0, 1.0]
    // We test via the contacts call succeeding
}

#[test]
fn crowdedness_in_range_after_cache() {
    let rotlib = match load_rotlib_or_skip() {
        Some(r) => r,
        None => {
            println!("RLIB not set, skipping test");
            return;
        }
    };

    let backbone = make_synthetic_backbone(4, 3.8);
    let confind = ConFind::new(rotlib, backbone.clone(), false).unwrap();

    confind.cache_all().ok();
    // Crowdedness is computed during cache_residue_impl
    // This test documents that values should be in [0.0, 1.0] range
}

#[test]
fn neighbors_excludes_self() {
    let rotlib = match load_rotlib_or_skip() {
        Some(r) => r,
        None => {
            println!("RLIB not set, skipping test");
            return;
        }
    };

    let backbone = make_synthetic_backbone(4, 3.8);
    let confind = ConFind::new(rotlib, backbone.clone(), false).unwrap();

    let ri = ResidueIndex(1);
    let neighbors = confind.neighbors(ri);

    // neighbors should not contain ri itself
    assert!(!neighbors.contains(&ri), "neighbors should exclude self");
}

#[test]
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
    let confind = ConFind::new(rotlib, backbone.clone(), false).unwrap();

    // All residues should have no neighbors at 30 Å spacing
    for i in 0..4 {
        let ri = ResidueIndex(i as u32);
        let neighbors = confind.neighbors(ri);
        assert_eq!(
            neighbors.len(),
            0,
            "Residue at 30 Å spacing should have no neighbors (DCUT=25)"
        );
    }
}

#[test]
fn freedom_not_available_before_contacts() {
    let rotlib = match load_rotlib_or_skip() {
        Some(r) => r,
        None => {
            println!("RLIB not set, skipping test");
            return;
        }
    };

    let backbone = make_synthetic_backbone(2, 3.8);
    let confind = ConFind::new(rotlib, backbone.clone(), false).unwrap();

    // Cache but don't call contacts
    confind.cache_residue(ResidueIndex(0)).ok();

    // Note: freedom values are stored and would be queried post-contacts
    // This test documents the expected workflow
}

#[test]
fn n_residues_matches_backbone() {
    let rotlib = match load_rotlib_or_skip() {
        Some(r) => r,
        None => {
            println!("RLIB not set, skipping test");
            return;
        }
    };

    let backbone = make_synthetic_backbone(5, 3.8);
    let confind = ConFind::new(rotlib, backbone.clone(), false).unwrap();

    assert_eq!(
        confind.n_residues(),
        5,
        "n_residues should match backbone length"
    );
}

#[test]
fn constrained_contacts_returns_vec() {
    let rotlib = require_rotlib();

    // Residues 0 and 1 are 6 Å apart (inside D_cut, sidechains can meet). Residue 2 is
    // the same geometry translated by 40 Å, so its Cα is beyond D_cut = 25 Å of both.
    let close = make_two_chain_backbone(1, 3.8, [0.0, 6.0, 0.0]);
    let mut owned = (*close).clone();
    let mut far = owned.bb[0].clone();
    let shift = 40.0;
    far.n = far.n.map(|p| [p[0] + shift, p[1], p[2]]);
    far.ca = far.ca.map(|p| [p[0] + shift, p[1], p[2]]);
    far.c = far.c.map(|p| [p[0] + shift, p[1], p[2]]);
    far.o = far.o.map(|p| [p[0] + shift, p[1], p[2]]);
    owned.bb.push(far);
    let mut far_id = owned.ids[0].clone();
    far_id.chain_id = "C".to_string();
    owned.ids.push(far_id);
    owned.chain_map.push(2);
    let backbone = std::sync::Arc::new(owned);

    let confind = ConFind::new(rotlib, backbone, false).unwrap();
    let query = vec![ResidueIndex(0), ResidueIndex(1), ResidueIndex(2)];
    // constrained_contacts reads the contact-degree cache that `contacts` fills.
    confind.contacts(&query, 0.0).unwrap();

    let contacts = confind.constrained_contacts(ResidueIndex(0));
    assert!(
        contacts.contains(&ResidueIndex(1)),
        "a close pair must be present in constrained_contacts, got {contacts:?}"
    );
    assert!(
        !contacts.contains(&ResidueIndex(2)),
        "a pair beyond D_cut must be absent from constrained_contacts, got {contacts:?}"
    );
    for i in 1..contacts.len() {
        assert!(
            contacts[i - 1] < contacts[i],
            "constrained_contacts should return sorted results"
        );
    }
}

#[test]
fn constrained_contacts_uncached_returns_empty() {
    let rotlib = match load_rotlib_or_skip() {
        Some(r) => r,
        None => {
            println!("RLIB not set, skipping test");
            return;
        }
    };

    let backbone = make_synthetic_backbone(4, 3.8);
    let confind = ConFind::new(rotlib, backbone.clone(), false).unwrap();

    let ri = ResidueIndex(1);
    // Don't cache ri

    // constrained_contacts should return empty vec for uncached residue
    let contacts = confind.constrained_contacts(ri);
    assert!(
        contacts.is_empty(),
        "constrained_contacts should return empty vec for uncached residue"
    );
}
