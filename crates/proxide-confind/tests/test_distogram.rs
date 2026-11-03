//! Mathematical invariant tests for ConFind contact degree, crowdedness, and freedom.
//!
//! Citations:
//! - Zheng, F., Zhang, J., & Grigoryan, G. (2015). "Tertiary Structural Propensities Reveal
//!   Fundamental Sequence/Structure Relationships in Proteins." *Structure*, 23(5), 961–971.
//! - Zheng, W., & Grigoryan, G. (2017). "Sequence statistics of tertiary structural motifs
//!   reflect protein stability." *PLoS ONE*, 12(5), e0178272.

mod common;

use common::{load_rotlib_or_skip, make_synthetic_backbone, make_two_chain_backbone};
use proxide_confind::coords::ResidueIndex;
use proxide_confind::ConFind;
use std::sync::Arc;

/// Invariant 1: Contact degree symmetry c(i, j) = c(j, i).
#[test]
fn test_invariant_symmetry() {
    let rotlib = match load_rotlib_or_skip() {
        Some(rl) => rl,
        None => return,
    };

    let backbone = make_two_chain_backbone(5, 3.8, [0.0, 6.0, 0.0]);
    let cf = ConFind::new(rotlib, backbone, false).unwrap();
    let residues: Vec<ResidueIndex> = (0..10).map(ResidueIndex).collect();
    let contacts = cf.contacts(&residues, 0.0).unwrap();

    for i in 0..10 {
        let ri = ResidueIndex(i as u32);
        for j in 0..10 {
            let rj = ResidueIndex(j as u32);
            let d_ij = contacts.degree(ri, rj).unwrap_or(0.0);
            let d_ji = contacts.degree(rj, ri).unwrap_or(0.0);
            assert_eq!(d_ij, d_ji, "Contact degree must be symmetric for ({i}, {j})");
        }
    }
}

/// Invariant 2: Spatial cutoff monotonicity ||CA_i - CA_j|| > 25.0 Å => c(i, j) = 0.0.
#[test]
fn test_invariant_spatial_cutoff() {
    let rotlib = match load_rotlib_or_skip() {
        Some(rl) => rl,
        None => return,
    };

    // Chain B placed at y = 30.0 Å (> DCUT = 25.0 Å)
    let backbone = make_two_chain_backbone(5, 3.8, [0.0, 30.0, 0.0]);
    let cf = ConFind::new(rotlib, backbone, false).unwrap();
    let residues: Vec<ResidueIndex> = (0..10).map(ResidueIndex).collect();
    let contacts = cf.contacts(&residues, 0.0).unwrap();

    // Cross-chain pairs must have contact degree == 0.0
    for i in 0..5 {
        let ri = ResidueIndex(i as u32);
        for j in 5..10 {
            let rj = ResidueIndex(j as u32);
            let cd = contacts.degree(ri, rj).unwrap_or(0.0);
            assert_eq!(cd, 0.0, "Residues separated by > 25 Å must have contact degree 0.0");
        }
    }
}

/// Invariant 3: Metric boundedness: c(i, j) in [0, 1], C(i) in [0, 1], F(i) in [0, 1].
#[test]
fn test_invariant_metric_bounds() {
    let rotlib = match load_rotlib_or_skip() {
        Some(rl) => rl,
        None => return,
    };

    let backbone = make_two_chain_backbone(5, 3.8, [0.0, 8.0, 0.0]);
    let cf = ConFind::new(rotlib, backbone, false).unwrap();
    let residues: Vec<ResidueIndex> = (0..10).map(ResidueIndex).collect();
    let contacts = cf.contacts(&residues, 0.0).unwrap();

    for &cd in &contacts.degrees {
        assert!((0.0..=1.0).contains(&cd), "Contact degree {cd} must be in [0, 1]");
    }

    for &ri in &residues {
        let cr = cf.crowdedness(ri).unwrap();
        assert!((0.0..=1.0).contains(&cr), "Crowdedness {cr} must be in [0, 1]");

        let fr = cf.freedom(ri).unwrap();
        assert!((0.0..=1.0).contains(&fr), "Freedom {fr} must be in [0, 1]");
    }
}

/// Invariant 4: Rigid-Body SE(3) Invariance: c(i, j; R * bb + t) = c(i, j; bb).
#[test]
fn test_invariant_se3_invariance() {
    let rotlib = match load_rotlib_or_skip() {
        Some(rl) => rl,
        None => return,
    };

    let bb_orig = make_two_chain_backbone(4, 3.8, [0.0, 7.0, 0.0]);

    // Apply rotation of 90 degrees around z-axis and translation by [10, 20, 30]
    let mut bb_trans = (*bb_orig).clone();
    let shift = [10.0, 20.0, 30.0];
    for rb in &mut bb_trans.bb {
        let rot_trans = |p: [f64; 3]| [-p[1] + shift[0], p[0] + shift[1], p[2] + shift[2]];
        rb.n = rb.n.map(rot_trans);
        rb.ca = rb.ca.map(rot_trans);
        rb.c = rb.c.map(rot_trans);
        rb.o = rb.o.map(rot_trans);
    }

    let cf_orig = ConFind::new(rotlib.clone(), bb_orig, false).unwrap();
    let cf_trans = ConFind::new(rotlib, Arc::new(bb_trans), false).unwrap();

    let residues: Vec<ResidueIndex> = (0..8).map(ResidueIndex).collect();
    let cd_orig = cf_orig.contacts(&residues, 0.0).unwrap();
    let cd_trans = cf_trans.contacts(&residues, 0.0).unwrap();

    for i in 0..8 {
        let ri = ResidueIndex(i as u32);
        for j in (i + 1)..8 {
            let rj = ResidueIndex(j as u32);
            let d_orig = cd_orig.degree(ri, rj).unwrap_or(0.0);
            let d_trans = cd_trans.degree(ri, rj).unwrap_or(0.0);
            assert!(
                (d_orig - d_trans).abs() < 1e-5,
                "SE(3) invariance violated for ({i}, {j}): orig={d_orig}, trans={d_trans}"
            );
        }
    }
}

/// Invariant 5: Crowdedness-freedom anti-correlation on isolated backbone.
#[test]
fn test_invariant_isolated_crowdedness_freedom() {
    let rotlib = match load_rotlib_or_skip() {
        Some(rl) => rl,
        None => return,
    };

    // Widely spaced residues (spacing 20.0 Å) have minimal crowding and high freedom
    let backbone = make_synthetic_backbone(3, 20.0);
    let cf = ConFind::new(rotlib, backbone, false).unwrap();
    let residues: Vec<ResidueIndex> = (0..3).map(ResidueIndex).collect();
    let _ = cf.contacts(&residues, 0.0).unwrap();

    for &ri in &residues {
        let cr = cf.crowdedness(ri).unwrap();
        let fr = cf.freedom(ri).unwrap();
        // With wide spacing, crowdedness is low and freedom is high
        assert!(cr < 0.5, "Isolated residue crowdedness should be low, got {cr}");
        assert!(fr > 0.5, "Isolated residue freedom should be high, got {fr}");
    }
}
