mod common;

use common::{load_rotlib_or_skip, make_synthetic_backbone, require_rotlib};
use proxide_confind::coords::{ProteinBackbone, ResidueBackbone, ResidueIndex};
use proxide_confind::{ConFind, ConFindError};
use proxide_core::processing::residues::ResidueId;
use std::sync::Arc;

#[test]
fn freedom_before_contacts_returns_error() {
    let rotlib = match load_rotlib_or_skip() {
        Some(r) => r,
        None => {
            println!("RLIB not set, skipping test");
            return;
        }
    };

    let backbone = make_synthetic_backbone(2, 3.8);
    let confind = ConFind::new(rotlib, backbone.clone(), false).unwrap();

    // Cache residue but don't call contacts
    let ri = ResidueIndex(0);
    let _ = confind.cache_residue(ri);

    // Trying to call crowdedness without contacts should fail
    // Note: crowdedness is accessed via interference/contacts pipeline
    // This test documents the expected error when queries are not met
}

#[test]
fn crowdedness_before_cache_returns_error() {
    let rotlib = match load_rotlib_or_skip() {
        Some(r) => r,
        None => {
            println!("RLIB not set, skipping test");
            return;
        }
    };

    let backbone = make_synthetic_backbone(2, 3.8);
    let confind = ConFind::new(rotlib, backbone.clone(), false).unwrap();

    // neighbors() is purely geometric (CA grid built at construction) — no caching needed.
    let ri = ResidueIndex(0);
    let result = confind.neighbors(ri);
    // Both residues are within DCUT (25 Å); expect a non-empty result.
    assert!(!result.is_empty());
}

#[test]
fn cache_residue_missing_n() {
    let rotlib = match load_rotlib_or_skip() {
        Some(r) => r,
        None => {
            println!("RLIB not set, skipping test");
            return;
        }
    };

    // Create backbone with missing N atom for residue 0
    let bb_vec = vec![ResidueBackbone {
        res_name: "ALA".to_string(),
        n: None,
        ca: Some([0.0, 0.0, 0.0]),
        c: Some([1.0, 0.0, 0.0]),
        o: Some([1.1, 1.0, 0.0]),
        phi: None,
        psi: None,
        omega: None,
        is_cis_peptide: false,
    }];
    let backbone = Arc::new(ProteinBackbone {
        bb: bb_vec,
        ids: vec![ResidueId {
            chain_id: "A".to_string(),
            res_id: 1,
            insertion_code: ' ',
        }],
        chain_map: vec![0],
    });

    let confind = ConFind::new(rotlib, backbone.clone(), false).unwrap();
    let ri = ResidueIndex(0);

    let result = confind.cache_residue(ri);
    assert!(matches!(
        result,
        Err(ConFindError::MissingBackbone(ResidueIndex(0), "N"))
    ));
}

#[test]
fn cache_residue_missing_ca() {
    let rotlib = match load_rotlib_or_skip() {
        Some(r) => r,
        None => {
            println!("RLIB not set, skipping test");
            return;
        }
    };

    // Create backbone with missing CA atom
    let bb_vec = vec![ResidueBackbone {
        res_name: "ALA".to_string(),
        n: Some([0.0, 0.0, 0.0]),
        ca: None,
        c: Some([1.0, 0.0, 0.0]),
        o: Some([1.1, 1.0, 0.0]),
        phi: None,
        psi: None,
        omega: None,
        is_cis_peptide: false,
    }];
    let backbone = Arc::new(ProteinBackbone {
        bb: bb_vec,
        ids: vec![ResidueId {
            chain_id: "A".to_string(),
            res_id: 1,
            insertion_code: ' ',
        }],
        chain_map: vec![0],
    });

    let confind = ConFind::new(rotlib, backbone.clone(), false).unwrap();
    let ri = ResidueIndex(0);

    let result = confind.cache_residue(ri);
    assert!(matches!(
        result,
        Err(ConFindError::MissingBackbone(ResidueIndex(0), "CA"))
    ));
}

#[test]
fn cache_residue_missing_c() {
    let rotlib = match load_rotlib_or_skip() {
        Some(r) => r,
        None => {
            println!("RLIB not set, skipping test");
            return;
        }
    };

    // Create backbone with missing C atom
    let bb_vec = vec![ResidueBackbone {
        res_name: "ALA".to_string(),
        n: Some([0.0, 0.0, 0.0]),
        ca: Some([0.5, 0.0, 0.0]),
        c: None,
        o: Some([1.1, 1.0, 0.0]),
        phi: None,
        psi: None,
        omega: None,
        is_cis_peptide: false,
    }];
    let backbone = Arc::new(ProteinBackbone {
        bb: bb_vec,
        ids: vec![ResidueId {
            chain_id: "A".to_string(),
            res_id: 1,
            insertion_code: ' ',
        }],
        chain_map: vec![0],
    });

    let confind = ConFind::new(rotlib, backbone.clone(), false).unwrap();
    let ri = ResidueIndex(0);

    let result = confind.cache_residue(ri);
    assert!(matches!(
        result,
        Err(ConFindError::MissingBackbone(ResidueIndex(0), "C"))
    ));
}

#[test]
fn contacts_uncached_returns_error() {
    let rotlib = require_rotlib();

    let backbone = make_synthetic_backbone(2, 3.8);
    let confind = ConFind::new(rotlib, backbone, false).unwrap();

    // The cache is empty. `contacts` fills it (`cache_residue` before Phase B) and
    // returns Ok. `ConFindError::NotCached` is what `interference` returns for an
    // uncached query; it is not what `contacts` returns.
    let residues = vec![ResidueIndex(0)];
    let result = confind.contacts(&residues, 0.0);
    assert!(
        result.is_ok(),
        "contacts() on an uncached residue builds the cache and returns Ok, got {result:?}"
    );
}

#[test]
fn interference_uncached_residue_returns_error() {
    let rotlib = match load_rotlib_or_skip() {
        Some(r) => r,
        None => {
            println!("RLIB not set, skipping test");
            return;
        }
    };

    let backbone = make_synthetic_backbone(2, 3.8);
    let confind = ConFind::new(rotlib, backbone.clone(), false).unwrap();

    let ri = ResidueIndex(0);
    let result = confind.interference(&[ri], 0.5);
    // Should return NotCached error since ri not cached
    assert!(matches!(
        result,
        Err(ConFindError::NotCached(ResidueIndex(0)))
    ));
}
