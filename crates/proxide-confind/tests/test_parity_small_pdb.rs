//! Independent tests for contact degree distance monotonicity, interference, and chain break propagation.
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

/// Contact degree monotonicity: as two parallel helices move further apart,
/// contact degree strictly decreases or stays zero.
#[test]
fn test_contact_degree_distance_monotonicity() {
    let rotlib = match load_rotlib_or_skip() {
        Some(rl) => rl,
        None => return,
    };

    let distances = [6.0, 7.5, 9.0, 11.0, 14.0, 26.0];
    let mut prev_total_contacts = f64::INFINITY;

    for &dist in &distances {
        let backbone = make_two_chain_backbone(3, 3.8, [0.0, dist, 0.0]);
        let cf = ConFind::new(rotlib.clone(), backbone, false);
        let residues: Vec<ResidueIndex> = (0..6).map(ResidueIndex).collect();
        let contacts = cf.contacts(&residues, 0.0).unwrap();

        // Sum cross-chain contact degrees
        let mut cross_sum = 0.0;
        for i in 0..3 {
            let ri = ResidueIndex(i as u32);
            for j in 3..6 {
                let rj = ResidueIndex(j as u32);
                cross_sum += contacts.degree(ri, rj).unwrap_or(0.0);
            }
        }

        assert!(
            cross_sum <= prev_total_contacts + 1e-6,
            "Contact degree sum must decrease with distance: dist={dist}, cross_sum={cross_sum}, prev={prev_total_contacts}"
        );
        prev_total_contacts = cross_sum;
    }

    // At 26 Å (> DCUT 25 Å), total contact sum must be exactly 0.0
    assert_eq!(prev_total_contacts, 0.0);
}

/// Mutual interference: clashing sidechains produce non-negative interference.
#[test]
fn test_interference_properties() {
    let rotlib = match load_rotlib_or_skip() {
        Some(rl) => rl,
        None => return,
    };

    // Close spacing (4.0 Å) induces steric interference
    let backbone = make_two_chain_backbone(3, 3.8, [0.0, 4.0, 0.0]);
    let cf = ConFind::new(rotlib, backbone, false);
    cf.cache_all().unwrap();
    let residues: Vec<ResidueIndex> = (0..6).map(ResidueIndex).collect();

    let interference = cf.interference(&residues, 0.0).unwrap();
    for &deg in &interference.degrees {
        assert!(deg >= 0.0, "Interference must be non-negative: got {deg}");
    }
}

/// Undefined dihedral propagation: terminal residues have Option dihedrals None.
#[test]
fn test_dihedral_none_propagation() {
    let backbone = make_synthetic_backbone(4, 3.8);
    // N-terminal residue 0: phi is None, omega is None
    assert_eq!(backbone.bb[0].phi, None);
    assert_eq!(backbone.bb[0].omega, None);
    assert!(backbone.bb[0].psi.is_some());

    // C-terminal residue 3: psi is None
    assert_eq!(backbone.bb[3].psi, None);
    assert!(backbone.bb[3].phi.is_some());
    assert!(backbone.bb[3].omega.is_some());
}
