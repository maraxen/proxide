#[path = "helpers.rs"]
mod helpers;

use helpers::{write_minimal_lib, BinSpec, RotSpec, real_rotlib_path};
use proxide_rotlib::{RotamerLibrary, RotamerId, RotlibError};

/// Create a 9-bin "TST" library for probability tests.
/// phi_centers = {-120, 0, 120}, psi_centers = {-120, 0, 120}
/// Default bin (index 4, phi=0, psi=0) has freq=999.9
/// All other bins have freq=1.0
/// Each bin has 2 rotamers with probs=[0.6, 0.3] (bin 4) or [0.5, 0.4] (others)
fn make_prob_lib() -> (tempfile::NamedTempFile, RotamerLibrary) {
    let phi_centers = [-120.0f32, 0.0, 120.0];
    let psi_centers = [-120.0f32, 0.0, 120.0];

    let mut bins = Vec::new();
    for (phi_idx, &phi) in phi_centers.iter().enumerate() {
        for (psi_idx, &psi) in psi_centers.iter().enumerate() {
            let is_default = phi_idx == 1 && psi_idx == 1;
            let freq = if is_default { 999.9 } else { 1.0 };
            bins.push(BinSpec { phi, psi, freq });
        }
    }

    // For simplicity, all bins use the same rotamer distribution.
    // write_minimal_lib applies rotamers_per_bin to all bins uniformly.
    // We'll verify bin 4's probabilities match expectations in the test.
    let rotamers = vec![
        RotSpec {
            prob: 0.6,
            coords: vec![[1.0, 0.0, 0.0]],
        },
        RotSpec {
            prob: 0.3,
            coords: vec![[0.0, 1.0, 0.0]],
        },
    ];

    let tmp = write_minimal_lib("TST", &["CB"], &bins, &rotamers);
    let lib = RotamerLibrary::load(tmp.path()).unwrap();
    (tmp, lib)
}

#[test]
fn test_prob_sum_le_one_per_bin() {
    let (_t, lib) = make_prob_lib();
    // 9 bins, each with 2 rotamers
    for bin_idx in 0u32..9 {
        let p0 = lib
            .rotamer_probability_by_id(&RotamerId {
                aa: "TST".to_string(),
                bin_index: bin_idx,
                rot_index: 0,
            })
            .unwrap();
        let p1 = lib
            .rotamer_probability_by_id(&RotamerId {
                aa: "TST".to_string(),
                bin_index: bin_idx,
                rot_index: 1,
            })
            .unwrap();
        let sum = p0 + p1;
        assert!(
            sum <= 1.0 + 1e-9,
            "bin {}: sum {} exceeds 1.0",
            bin_idx,
            sum
        );
    }
}

#[test]
fn test_prob_by_id_matches_direct() {
    let (_t, lib) = make_prob_lib();
    // default bin is 4 (phi=0, psi=0, located at phi_idx=1, psi_idx=1 -> 1*3+1=4)
    let direct = lib.rotamer_probability("TST", 0, 0.0, 0.0).unwrap();
    let by_id = lib
        .rotamer_probability_by_id(&RotamerId {
            aa: "TST".to_string(),
            bin_index: 4,
            rot_index: 0,
        })
        .unwrap();
    assert_eq!(direct, by_id);
}

#[test]
fn test_prob_sentinel_bin() {
    let (_t, lib) = make_prob_lib();
    // sentinel (9999.0) -> default_bin=4, second rotamer (rot=1)
    let p = lib.rotamer_probability("TST", 1, 9999.0, 9999.0).unwrap();
    assert!(
        (p - 0.3).abs() < 1e-6,
        "expected ~0.3 for sentinel bin, got {}",
        p
    );
}

#[test]
fn test_prob_oob_rot_index() {
    let (_t, lib) = make_prob_lib();
    // rot=99 is out of bounds (only 2 rotamers per bin)
    let result = lib.rotamer_probability("TST", 99, 0.0, 0.0);
    assert!(matches!(
        result,
        Err(RotlibError::RotIndexOob(ref aa, 99, 2)) if aa == "TST"
    ));
}

#[test]
fn test_prob_unknown_aa() {
    let (_t, lib) = make_prob_lib();
    let result = lib.rotamer_probability("ZZZ", 0, 0.0, 0.0);
    assert!(matches!(
        result,
        Err(RotlibError::UnknownAa(ref aa)) if aa == "ZZZ"
    ));
}

#[test]
fn test_prob_by_id_unknown_aa() {
    let (_t, lib) = make_prob_lib();
    let result = lib.rotamer_probability_by_id(&RotamerId {
        aa: "ZZZ".to_string(),
        bin_index: 0,
        rot_index: 0,
    });
    assert!(matches!(
        result,
        Err(RotlibError::UnknownAa(ref aa)) if aa == "ZZZ"
    ));
}

#[test]
fn test_prob_by_id_oob_rot_index() {
    let (_t, lib) = make_prob_lib();
    let result = lib.rotamer_probability_by_id(&RotamerId {
        aa: "TST".to_string(),
        bin_index: 4,
        rot_index: 99,
    });
    assert!(matches!(
        result,
        Err(RotlibError::RotIndexOob(ref aa, 99, 2)) if aa == "TST"
    ));
}

#[test]
#[ignore]
fn test_prob_sum_real_rotlib() {
    let lib = RotamerLibrary::load(&real_rotlib_path()).unwrap();
    let aa_names = [
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "HIS", "ILE", "LEU", "LYS", "MET",
        "PHE", "SER", "THR", "TRP", "TYR", "VAL",
    ];

    for aa in &aa_names {
        // Confirm AA is present
        assert!(lib.contains_aa(aa), "AA {} not in library", aa);

        // Check that default bin probabilities sum to <= 1.0
        let nr = lib.num_rotamers(aa, 9999.0, 9999.0).unwrap();
        let sum: f64 = (0..nr)
            .map(|i| lib.rotamer_probability(aa, i, 9999.0, 9999.0).unwrap())
            .sum();
        assert!(
            sum <= 1.0 + 1e-6,
            "{}: default bin prob sum = {}, exceeds 1.0",
            aa,
            sum
        );
    }
}
