//! Kernel-level acceptance tests for the configuration options (spec §3.8.3, §3.9.6):
//! injected rotamer sets are evaluated against a synthetic backbone, so every expected
//! value is derived from the specification's own worked examples.

mod common;

use common::make_synthetic_backbone;
use proxide_confind::cache::{
    build_backbone_grid, evaluate_candidates, ResidueCache, RotamerCandidate,
};
use proxide_confind::{CbClashPolicy, CrowdednessMode, ResidueIndex};

const I: ResidueIndex = ResidueIndex(1);
const J: ResidueIndex = ResidueIndex(0);
const K: ResidueIndex = ResidueIndex(2);

/// A point 1.9 Å from the Cα of residue 0 (x = 0) that is >2 Å from every other backbone atom.
const NEAR_J: [f64; 3] = [0.0, -1.9, 0.0];
/// Same for residue 2 (x = 100).
const NEAR_K: [f64; 3] = [100.0, -1.9, 0.0];
/// A point far from every backbone atom.
const FAR: [f64; 3] = [300.0, 300.0, 300.0];

/// (P(r), contact atoms, optional C-beta position)
type Cand = (f64, Vec<[f64; 3]>, Option<[f64; 3]>);

fn run(
    aa: &'static str,
    p_a: f64,
    cands: Vec<Cand>,
    policy: CbClashPolicy,
    d_clash: f64,
) -> ResidueCache {
    // Residues at x = 0, 50, 100: residue 1 is the one under test, 0 and 2 are j and k.
    let bb = make_synthetic_backbone(3, 50.0);
    let grid = build_backbone_grid(&bb);
    let candidates = cands
        .into_iter()
        .enumerate()
        .map(|(rot_idx, (p_r, cont_atoms, cb))| RotamerCandidate {
            aa,
            rot_idx,
            p_a,
            p_r,
            cont_atoms,
            cb,
        })
        .collect();
    evaluate_candidates(I, "ALA", false, candidates, &grid, d_clash, policy)
}

fn interference(rc: &ResidueCache, r: ResidueIndex) -> f64 {
    rc.interference.get(&r).copied().unwrap_or(0.0)
}

/// Fixture T1 (spec §3.9.6): non-alanine, P(a) = 10, P = {0.5, 0.3, 0.2}; r1 has a non-Cβ
/// atom 1.9 Å from B_j; r2 has only its Cβ 1.9 Å from B_k; r3 is clash-free.
fn t1(aa: &'static str, policy: CbClashPolicy) -> ResidueCache {
    let r2_cont = if aa == "ALA" {
        // Cβ is the contact atom of alanine.
        vec![NEAR_K]
    } else {
        vec![FAR]
    };
    run(
        aa,
        10.0,
        vec![
            (0.5, vec![NEAR_J], Some(FAR)),
            (0.3, r2_cont, Some(NEAR_K)),
            (0.2, vec![FAR], Some(FAR)),
        ],
        policy,
        2.0,
    )
}

#[test]
fn t1_all_residues_row() {
    let rc = t1("LEU", CbClashPolicy::AllResidues);
    assert_eq!((rc.n_surviving, rc.n_library), (1, 3));
    let cw = rc.crowdedness(CrowdednessMode::ProbabilityWeighted);
    let cn = rc.crowdedness(CrowdednessMode::PrunedCountFraction);
    assert!((cw - 0.8).abs() < 1e-12, "C_w = {cw}");
    assert!((cn - 2.0 / 3.0).abs() < 1e-12, "C_n = {cn}");
    assert!((interference(&rc, J) - 0.05).abs() < 1e-12);
    assert!((interference(&rc, K) - 0.03).abs() < 1e-12);
}

#[test]
fn t1_ala_only_row() {
    let rc = t1("LEU", CbClashPolicy::AlaOnly);
    assert_eq!((rc.n_surviving, rc.n_library), (2, 3));
    let cw = rc.crowdedness(CrowdednessMode::ProbabilityWeighted);
    let cn = rc.crowdedness(CrowdednessMode::PrunedCountFraction);
    assert!((cw - 0.5).abs() < 1e-12, "C_w = {cw}");
    assert!((cn - 1.0 / 3.0).abs() < 1e-12, "C_n = {cn}");
    assert!((interference(&rc, J) - 0.05).abs() < 1e-12);
    assert_eq!(interference(&rc, K), 0.0);
}

#[test]
fn t1_alanine_gives_all_residues_row_for_both_policies() {
    for policy in [CbClashPolicy::AlaOnly, CbClashPolicy::AllResidues] {
        let rc = t1("ALA", policy);
        assert_eq!(rc.n_surviving, 1);
        let cw = rc.crowdedness(CrowdednessMode::ProbabilityWeighted);
        assert!((cw - 0.8).abs() < 1e-12);
        assert!((interference(&rc, J) - 0.05).abs() < 1e-12);
        assert!((interference(&rc, K) - 0.03).abs() < 1e-12);
    }
}

#[test]
fn crowdedness_modes_agree_for_uniform_single_amino_acid_fixture() {
    // Equal rotamer probabilities, one amino acid, one of four rotamers clashing.
    let rc = run(
        "LEU",
        7.0,
        vec![
            (0.25, vec![NEAR_J], None),
            (0.25, vec![FAR], None),
            (0.25, vec![FAR], None),
            (0.25, vec![FAR], None),
        ],
        CbClashPolicy::AllResidues,
        2.0,
    );
    let cw = rc.crowdedness(CrowdednessMode::ProbabilityWeighted);
    let cn = rc.crowdedness(CrowdednessMode::PrunedCountFraction);
    assert!((cw - 0.25).abs() < 1e-12);
    assert!((cw - cn).abs() < 1e-12);
}

#[test]
fn defaults_reproduce_interference_worked_example() {
    // §3.8.3 test 1: r1 clashes (non-Cβ) with B_j only; r2 with B_j and B_k; r3 nothing.
    let rc = run(
        "LEU",
        10.0,
        vec![
            (0.5, vec![NEAR_J], None),
            (0.3, vec![NEAR_J, NEAR_K], None),
            (0.2, vec![FAR], None),
        ],
        CbClashPolicy::default(),
        2.0,
    );
    assert!((interference(&rc, J) - 0.08).abs() < 1e-9);
    assert!((interference(&rc, K) - 0.03).abs() < 1e-9);
    assert!(
        !rc.interference.contains_key(&I),
        "own-residue backbone is never evaluated"
    );
}

#[test]
fn interference_is_independent_of_atom_and_rotamer_order() {
    let a = vec![
        (0.5, vec![NEAR_J, FAR], None),
        (0.3, vec![NEAR_J, NEAR_K], None),
        (0.2, vec![FAR], None),
    ];
    let b = vec![
        (0.2, vec![FAR], None),
        (0.3, vec![NEAR_K, NEAR_J], None),
        (0.5, vec![FAR, NEAR_J], None),
    ];
    let ra = run("LEU", 10.0, a, CbClashPolicy::default(), 2.0);
    let rb = run("LEU", 10.0, b, CbClashPolicy::default(), 2.0);
    for r in [J, K] {
        assert!((interference(&ra, r) - interference(&rb, r)).abs() < 1e-12);
    }
}

#[test]
fn many_clashing_atoms_contribute_once() {
    let rc = run(
        "LEU",
        10.0,
        vec![(1.0, vec![NEAR_J; 25], Some(NEAR_J))],
        CbClashPolicy::AllResidues,
        2.0,
    );
    assert!((interference(&rc, J) - 0.1).abs() < 1e-12);
}

#[test]
fn interference_bounded_by_pruned_mass() {
    let rc = t1("LEU", CbClashPolicy::AllResidues);
    let z = rc.omega_lib / 100.0;
    let c_w = rc.crowdedness(CrowdednessMode::ProbabilityWeighted);
    for r in [J, K] {
        assert!(interference(&rc, r) <= z * c_w + 1e-12);
    }
}

#[test]
fn clash_distance_is_inclusive() {
    let at = |d: f64| {
        run(
            "LEU",
            10.0,
            vec![(1.0, vec![[0.0, -d, 0.0]], None)],
            CbClashPolicy::default(),
            2.0,
        )
    };
    assert_eq!(at(2.0).n_surviving, 0, "exactly 2.0 A prunes");
    assert_eq!(at(2.0 + 1e-9).n_surviving, 1, "2.0 + 1e-9 A does not");
    assert_eq!(at(2.0001).n_surviving, 1);
}

#[test]
fn configured_clash_distance_is_used() {
    let at = |d_clash: f64| {
        run(
            "LEU",
            10.0,
            vec![(1.0, vec![NEAR_J], None)],
            CbClashPolicy::default(),
            d_clash,
        )
    };
    assert_eq!(at(1.8).n_surviving, 1, "1.9 A is outside d_clash = 1.8");
    assert_eq!(
        at(1.9).n_surviving,
        0,
        "1.9 A is inside d_clash = 1.9 (inclusive)"
    );
    // A d_clash below every atom-backbone distance prunes nothing: C = 0, N_surv = N_lib.
    let rc = at(0.01);
    assert_eq!(rc.n_surviving, rc.n_library);
    assert_eq!(rc.crowdedness(CrowdednessMode::ProbabilityWeighted), 0.0);
    assert_eq!(rc.crowdedness(CrowdednessMode::PrunedCountFraction), 0.0);
}

#[test]
fn own_residue_backbone_is_never_tested() {
    // Atom 1.5 A from the Cα of the residue under test (x = 50): bonded-Cβ-like distance.
    let rc = run(
        "LEU",
        10.0,
        vec![(1.0, vec![[50.0, -1.5, 0.0]], Some([50.0, -1.5, 0.0]))],
        CbClashPolicy::AllResidues,
        2.0,
    );
    assert_eq!(rc.n_surviving, 1);
    assert!(rc.interference.is_empty());
}

#[test]
fn glycine_like_empty_library_has_zero_crowdedness() {
    let rc = run("LEU", 10.0, vec![], CbClashPolicy::default(), 2.0);
    assert_eq!(rc.n_library, 0);
    assert_eq!(rc.crowdedness(CrowdednessMode::ProbabilityWeighted), 0.0);
    assert_eq!(rc.crowdedness(CrowdednessMode::PrunedCountFraction), 0.0);
}
