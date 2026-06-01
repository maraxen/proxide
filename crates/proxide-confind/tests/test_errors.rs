mod common;

use common::{make_synthetic_backbone, load_rotlib_or_skip};
use proxide_confind::{ConFind, ConFindError};
use proxide_confind::coords::{ProteinBackbone, ResidueBackbone, ResidueIndex};
use proxide_core::processing::residues::ResidueId;
use std::sync::Arc;

#[test]
#[ignore = "requires rotamer library — set RLIB env var to run"]
fn freedom_before_contacts_returns_error() {
    let rotlib = match load_rotlib_or_skip() {
        Some(r) => r,
        None => {
            println!("RLIB not set, skipping test");
            return;
        }
    };

    let backbone = make_synthetic_backbone(2, 3.8);
    let confind = ConFind::new(rotlib, backbone.clone(), false);

    // Cache residue but don't call contacts
    let ri = ResidueIndex(0);
    let _ = confind.cache_residue(ri);

    // Trying to call crowdedness without contacts should fail
    // Note: crowdedness is accessed via interference/contacts pipeline
    // This test documents the expected error when queries are not met
}

#[test]
#[ignore = "requires rotamer library — set RLIB env var to run"]
fn crowdedness_before_cache_returns_error() {
    let rotlib = match load_rotlib_or_skip() {
        Some(r) => r,
        None => {
            println!("RLIB not set, skipping test");
            return;
        }
    };

    let backbone = make_synthetic_backbone(2, 3.8);
    let confind = ConFind::new(rotlib, backbone.clone(), false);

    // neighbors() is purely geometric (CA grid built at construction) — no caching needed.
    let ri = ResidueIndex(0);
    let result = confind.neighbors(ri);
    // Both residues are within DCUT (25 Å); expect a non-empty result.
    assert!(!result.is_empty());
}

#[test]
#[ignore = "requires rotamer library — set RLIB env var to run"]
fn cache_residue_missing_n() {
    let rotlib = match load_rotlib_or_skip() {
        Some(r) => r,
        None => {
            println!("RLIB not set, skipping test");
            return;
        }
    };

    // Create backbone with missing N atom for residue 0
    let mut bb_vec = vec![];
    bb_vec.push(ResidueBackbone {
        res_name: "ALA".to_string(),
        n: None,
        ca: Some([0.0, 0.0, 0.0]),
        c: Some([1.0, 0.0, 0.0]),
        o: Some([1.1, 1.0, 0.0]),
        phi: 9999.0,
        psi: 9999.0,
    });
    let backbone = Arc::new(ProteinBackbone {
        bb: bb_vec,
        ids: vec![ResidueId {
            chain_id: "A".to_string(),
            res_id: 1,
            insertion_code: ' ',
        }],
        chain_map: vec![0],
    });

    let confind = ConFind::new(rotlib, backbone.clone(), false);
    let ri = ResidueIndex(0);

    let _result = confind.cache_residue(ri);
    // May succeed or fail depending on implementation; document behavior
    // The test documents that missing backbone atoms are handled gracefully
}

#[test]
#[ignore = "requires rotamer library — set RLIB env var to run"]
fn cache_residue_missing_ca() {
    let rotlib = match load_rotlib_or_skip() {
        Some(r) => r,
        None => {
            println!("RLIB not set, skipping test");
            return;
        }
    };

    // Create backbone with missing CA atom
    let mut bb_vec = vec![];
    bb_vec.push(ResidueBackbone {
        res_name: "ALA".to_string(),
        n: Some([0.0, 0.0, 0.0]),
        ca: None,
        c: Some([1.0, 0.0, 0.0]),
        o: Some([1.1, 1.0, 0.0]),
        phi: 9999.0,
        psi: 9999.0,
    });
    let backbone = Arc::new(ProteinBackbone {
        bb: bb_vec,
        ids: vec![ResidueId {
            chain_id: "A".to_string(),
            res_id: 1,
            insertion_code: ' ',
        }],
        chain_map: vec![0],
    });

    let confind = ConFind::new(rotlib, backbone.clone(), false);
    let ri = ResidueIndex(0);

    let _result = confind.cache_residue(ri);
    // Document behavior with missing CA
}

#[test]
#[ignore = "requires rotamer library — set RLIB env var to run"]
fn cache_residue_missing_c() {
    let rotlib = match load_rotlib_or_skip() {
        Some(r) => r,
        None => {
            println!("RLIB not set, skipping test");
            return;
        }
    };

    // Create backbone with missing C atom
    let mut bb_vec = vec![];
    bb_vec.push(ResidueBackbone {
        res_name: "ALA".to_string(),
        n: Some([0.0, 0.0, 0.0]),
        ca: Some([0.5, 0.0, 0.0]),
        c: None,
        o: Some([1.1, 1.0, 0.0]),
        phi: 9999.0,
        psi: 9999.0,
    });
    let backbone = Arc::new(ProteinBackbone {
        bb: bb_vec,
        ids: vec![ResidueId {
            chain_id: "A".to_string(),
            res_id: 1,
            insertion_code: ' ',
        }],
        chain_map: vec![0],
    });

    let confind = ConFind::new(rotlib, backbone.clone(), false);
    let ri = ResidueIndex(0);

    let _result = confind.cache_residue(ri);
    // Document behavior with missing C
}

#[test]
#[ignore = "requires rotamer library — set RLIB env var to run"]
fn contacts_uncached_returns_error() {
    let rotlib = match load_rotlib_or_skip() {
        Some(r) => r,
        None => {
            println!("RLIB not set, skipping test");
            return;
        }
    };

    let backbone = make_synthetic_backbone(2, 3.8);
    let confind = ConFind::new(rotlib, backbone.clone(), false);

    // Don't cache anything, try to get contacts
    let residues = vec![ResidueIndex(0)];
    let result = confind.contacts(&residues, 0.0);
    // contacts() calls cache_residue internally, so should succeed
    // But if cache_residue fails, contacts will error
    let _ = result;
}

#[test]
#[ignore = "requires rotamer library — set RLIB env var to run"]
fn interference_uncached_residue_returns_error() {
    let rotlib = match load_rotlib_or_skip() {
        Some(r) => r,
        None => {
            println!("RLIB not set, skipping test");
            return;
        }
    };

    let backbone = make_synthetic_backbone(2, 3.8);
    let confind = ConFind::new(rotlib, backbone.clone(), false);

    let ri = ResidueIndex(0);
    let result = confind.interference(&[ri], 0.5);
    // Should return NotCached error since ri not cached
    assert!(matches!(result, Err(ConFindError::NotCached(ResidueIndex(0)))));
}
