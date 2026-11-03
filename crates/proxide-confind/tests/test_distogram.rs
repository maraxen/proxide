//! Mathematical invariant tests for ConFind contact degree, crowdedness, and freedom.
//!
//! Citations:
//! - Zheng, F., Zhang, J., & Grigoryan, G. (2015). "Tertiary Structural Propensities Reveal
//!   Fundamental Sequence/Structure Relationships in Proteins." *Structure*, 23(5), 961–971.
//! - Zheng, W., & Grigoryan, G. (2017). "Sequence statistics of tertiary structural motifs
//!   reflect protein stability." *PLoS ONE*, 12(5), e0178272.

mod common;

use common::{
    load_rotlib_or_skip, make_synthetic_backbone, make_two_chain_backbone, require_rotlib,
};
use proxide_confind::coords::{ProteinBackbone, ResidueBackbone, ResidueIndex};
use proxide_confind::ConFind;
use proxide_core::processing::residues::ResidueId;
use proxide_rotlib::pb::rotlib_v1 as pb;
use proxide_rotlib::RotamerLibrary;
use std::sync::Arc;

/// Invariant 1: Contact degree symmetry c(i, j) = c(j, i).
#[test]
fn test_invariant_symmetry() {
    let rotlib = require_rotlib();

    let backbone = make_two_chain_backbone(5, 3.8, [0.0, 6.0, 0.0]);
    let cf = ConFind::new(rotlib, backbone, false).unwrap();
    let residues: Vec<ResidueIndex> = (0..10).map(ResidueIndex).collect();
    let contacts = cf.contacts(&residues, 0.0).unwrap();

    let mut saw_positive = false;
    for i in 0..10 {
        let ri = ResidueIndex(i as u32);
        for j in 0..10 {
            let rj = ResidueIndex(j as u32);
            let d_ij = contacts.degree(ri, rj).unwrap_or(0.0);
            let d_ji = contacts.degree(rj, ri).unwrap_or(0.0);
            assert_eq!(
                d_ij, d_ji,
                "Contact degree must be symmetric for ({i}, {j})"
            );
            if d_ij > 0.0 {
                saw_positive = true;
            }
        }
    }
    // 0 == 0 is symmetric. The invariant is only tested if some pair actually contacts.
    assert!(
        saw_positive,
        "a pair inside contact range must have c(i, j) > 0"
    );
}

/// Invariant 2: Spatial cutoff monotonicity ||CA_i - CA_j|| > 25.0 Å => c(i, j) = 0.0.
#[test]
fn test_invariant_spatial_cutoff() {
    let rotlib = require_rotlib();

    // Chain B placed at y = 30.0 Å (> DCUT = 25.0 Å)
    let backbone = make_two_chain_backbone(5, 3.8, [0.0, 30.0, 0.0]);
    let cf = ConFind::new(rotlib.clone(), backbone, false).unwrap();
    let residues: Vec<ResidueIndex> = (0..10).map(ResidueIndex).collect();
    let contacts = cf.contacts(&residues, 0.0).unwrap();

    // Cross-chain pairs must have contact degree == 0.0
    for i in 0..5 {
        let ri = ResidueIndex(i as u32);
        for j in 5..10 {
            let rj = ResidueIndex(j as u32);
            let cd = contacts.degree(ri, rj).unwrap_or(0.0);
            assert_eq!(
                cd, 0.0,
                "Residues separated by > 25 Å must have contact degree 0.0"
            );
        }
    }

    // The far-pair check passes if contacts() returns 0 for every pair. A close control
    // at the same chain length must produce a strictly positive cross-chain sum.
    let close = make_two_chain_backbone(5, 3.8, [0.0, 6.0, 0.0]);
    let cf_close = ConFind::new(rotlib, close, false).unwrap();
    let close_contacts = cf_close.contacts(&residues, 0.0).unwrap();
    let mut close_sum = 0.0;
    for i in 0..5 {
        for j in 5..10 {
            close_sum += close_contacts
                .degree(ResidueIndex(i), ResidueIndex(j))
                .unwrap_or(0.0);
        }
    }
    assert!(
        close_sum > 0.0,
        "cross-chain pairs at 6 Å must have positive contact degree, got {close_sum}"
    );
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
        assert!(
            (0.0..=1.0).contains(&cd),
            "Contact degree {cd} must be in [0, 1]"
        );
    }

    for &ri in &residues {
        let cr = cf.crowdedness(ri).unwrap();
        assert!(
            (0.0..=1.0).contains(&cr),
            "Crowdedness {cr} must be in [0, 1]"
        );

        let fr = cf.freedom(ri).unwrap();
        assert!((0.0..=1.0).contains(&fr), "Freedom {fr} must be in [0, 1]");
    }
}

/// Invariant 4: Rigid-Body SE(3) Invariance: c(i, j; R * bb + t) = c(i, j; bb).
#[test]
fn test_invariant_se3_invariance() {
    let rotlib = require_rotlib();

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

    let mut saw_positive = false;
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
            if d_orig > 0.0 {
                saw_positive = true;
            }
        }
    }
    // Translating an all-zero matrix preserves it. Require a real contact so the
    // equality is about a non-zero degree.
    assert!(
        saw_positive,
        "SE(3) fixture must contain a pair with c(i, j) > 0"
    );
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
        assert!(
            cr < 0.5,
            "Isolated residue crowdedness should be low, got {cr}"
        );
        assert!(
            fr > 0.5,
            "Isolated residue freedom should be high, got {fr}"
        );
    }
}

/// One contact atom per amino acid, placed at local (0, 5, 0) in the residue frame (spec §2.7).
/// Alanine's only sidechain atom is CB; every other amino acid uses CG so it counts in
/// `S_cont` (CB is excluded for non-alanine). Each amino acid has a single rotamer with
/// probability 1 at (φ, ψ) = (0, 0).
fn single_contact_atom_library() -> Arc<RotamerLibrary> {
    let residues = proxide_confind::AA_NAMES
        .iter()
        .map(|aa| {
            let atom_name = if *aa == "ALA" { "CB" } else { "CG" };
            pb::ResidueEntry {
                code: (*aa).to_string(),
                atom_names: vec![atom_name.to_string()],
                num_chi: 0,
                phi_centers: vec![0.0],
                psi_centers: vec![0.0],
                default_bin: 0,
                bins: vec![pb::Bin {
                    phi: 0.0,
                    psi: 0.0,
                    freq: 1.0,
                    rotamers: vec![pb::Rotamer {
                        prob: 1.0,
                        chi: vec![],
                        coords: vec![pb::Vec3 {
                            x: 0.0,
                            y: 5.0,
                            z: 0.0,
                        }],
                    }],
                }],
            }
        })
        .collect();
    let pb_lib = pb::RotamerLibrary {
        attribution: "synthetic-contact-fixture".to_string(),
        geometry_mode: pb::GeometryMode::Precomputed as i32,
        residues,
        ..Default::default()
    };
    Arc::new(RotamerLibrary::from_protobuf(&pb_lib).expect("synthetic rotlib"))
}

/// Identity residue frame (spec §2.7): N = CA − x̂, C = CA + ŷ, so a local (0, 5, 0)
/// atom is placed at CA + (0, 5, 0).
fn identity_frame_residue(
    chain: &str,
    res_id: i32,
    chain_idx: usize,
    ca: [f64; 3],
) -> (ResidueBackbone, ResidueId, usize) {
    (
        ResidueBackbone {
            res_name: "ALA".to_string(),
            n: Some([ca[0] - 1.0, ca[1], ca[2]]),
            ca: Some(ca),
            c: Some([ca[0], ca[1] + 1.0, ca[2]]),
            o: Some([ca[0], ca[1] + 2.0, ca[2]]),
            phi: Some(0.0),
            psi: Some(0.0),
            omega: Some(180.0),
            is_cis_peptide: false,
        },
        ResidueId {
            chain_id: chain.to_string(),
            res_id,
            insertion_code: ' ',
        },
        chain_idx,
    )
}

/// Pair of sidechain atoms 1.0 Å apart (well inside d_cont = 3.0 Å) has c(i, j) = 1.
///
/// Both residues share the library offset (0, 5, 0). With parallel frames, separating
/// the Cα atoms by (1, 0, 0) separates the contact atoms by 1.0 Å. That is less than
/// d_cont and greater than d_clash from every backbone atom of the other residue
/// (nearest approach is √10 ≈ 3.16 Å > 2.0 Å), so no rotamer is pruned and every
/// w(r_a, r_b) = 1. Spec §2.1 then gives
/// c = (Σ_a Σ_b P(a) P(b)) / (Σ_a P(a))² = 1.
/// A third residue 40 Å away is beyond D_cut and must contribute 0.
#[test]
fn contact_degree_inside_d_cont_matches_analytical_one() {
    let rotlib = single_contact_atom_library();
    let specs = [
        identity_frame_residue("A", 1, 0, [0.0, 0.0, 0.0]),
        identity_frame_residue("B", 1, 1, [1.0, 0.0, 0.0]),
        identity_frame_residue("C", 1, 2, [40.0, 0.0, 0.0]),
    ];
    let mut bb = Vec::new();
    let mut ids = Vec::new();
    let mut chain_map = Vec::new();
    for (residue, id, chain) in specs {
        bb.push(residue);
        ids.push(id);
        chain_map.push(chain);
    }
    let backbone = Arc::new(ProteinBackbone { bb, ids, chain_map });
    let cf = ConFind::new(rotlib, backbone, false).unwrap();
    let residues = vec![ResidueIndex(0), ResidueIndex(1), ResidueIndex(2)];
    let contacts = cf.contacts(&residues, 0.0).unwrap();

    let close = contacts
        .degree(ResidueIndex(0), ResidueIndex(1))
        .expect("pair whose sidechain atoms lie 1.0 Å apart must be reported");
    assert!(
        close > 0.0,
        "c(i, j) inside d_cont must be strictly positive"
    );
    assert!(
        (close - 1.0).abs() < 1e-9,
        "analytical contact degree is 1, got {close}"
    );
    let far = contacts
        .degree(ResidueIndex(0), ResidueIndex(2))
        .unwrap_or(0.0);
    assert_eq!(far, 0.0, "Cα separation of 40 Å is beyond D_cut");
}
