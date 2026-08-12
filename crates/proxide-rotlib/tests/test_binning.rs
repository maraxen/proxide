#[path = "helpers.rs"]
mod helpers;
use helpers::{write_minimal_lib, BinSpec, RotSpec};
use proxide_rotlib::{RotamerLibrary, RotlibError};

fn make_3x3_lib() -> (tempfile::NamedTempFile, RotamerLibrary) {
    let phi_centers = [-120.0f32, 0.0, 120.0];
    let psi_centers = [-120.0f32, 0.0, 120.0];
    let mut bins = Vec::new();
    for &phi in &phi_centers {
        for &psi in &psi_centers {
            let freq = if phi == 0.0 && psi == 0.0 { 999.9 } else { 1.0 };
            bins.push(BinSpec { phi, psi, freq });
        }
    }
    let rotamers = vec![RotSpec {
        prob: 0.5,
        coords: vec![[1.0, 0.0, 0.0]],
    }];
    let tmp = write_minimal_lib("TST", &["CB"], &bins, &rotamers);
    let lib = RotamerLibrary::load(tmp.path()).unwrap();
    (tmp, lib)
}

// bin_index = phi_ind * 3 + psi_ind

#[test]
fn test_binning_sentinel_phi() {
    let (_t, lib) = make_3x3_lib();
    assert_eq!(lib.backbone_bin("TST", 9999.0, -120.0, false).unwrap(), 4);
}

#[test]
fn test_binning_sentinel_psi() {
    let (_t, lib) = make_3x3_lib();
    assert_eq!(lib.backbone_bin("TST", -120.0, 9999.0, false).unwrap(), 4);
}

#[test]
fn test_binning_sentinel_both() {
    let (_t, lib) = make_3x3_lib();
    assert_eq!(lib.backbone_bin("TST", 9999.0, 9999.0, false).unwrap(), 4);
}

#[test]
fn test_binning_exact_center() {
    let (_t, lib) = make_3x3_lib();
    // phi=0 → ind=1, psi=0 → ind=1 → 1*3+1=4
    assert_eq!(lib.backbone_bin("TST", 0.0, 0.0, false).unwrap(), 4);
}

#[test]
fn test_binning_known_corner() {
    let (_t, lib) = make_3x3_lib();
    // phi=-120 → ind=0, psi=-120 → ind=0 → 0
    assert_eq!(lib.backbone_bin("TST", -120.0, -120.0, false).unwrap(), 0);
}

#[test]
fn test_binning_known_edge() {
    let (_t, lib) = make_3x3_lib();
    // phi=120 → ind=2, psi=0 → ind=1 → 2*3+1=7
    assert_eq!(lib.backbone_bin("TST", 120.0, 0.0, false).unwrap(), 7);
}

#[test]
fn test_binning_near_center_offset() {
    let (_t, lib) = make_3x3_lib();
    // phi=40 → nearest 0 (dist=40 vs 80), psi=-40 → nearest 0 (dist=40 vs 80) → 4
    assert_eq!(lib.backbone_bin("TST", 40.0, -40.0, false).unwrap(), 4);
}

#[test]
fn test_binning_wrap_around_pos() {
    let (_t, lib) = make_3x3_lib();
    // phi=179.9 → nearest 120 (dist≈60) vs 0 (dist≈180) vs -120 (dist≈60.1) → ind=2; psi=0 → ind=1 → 7
    assert_eq!(lib.backbone_bin("TST", 179.9, 0.0, false).unwrap(), 7);
}

#[test]
fn test_binning_wrap_around_neg() {
    let (_t, lib) = make_3x3_lib();
    // phi=-180.0: angle_diff(-120,-180)=60, angle_diff(0,-180)=180, angle_diff(120,-180)=60
    // tie at ind=0(-120) and ind=2(120) → first wins → ind=0; psi=0 → ind=1 → 0*3+1=1
    assert_eq!(lib.backbone_bin("TST", -180.0, 0.0, false).unwrap(), 1);
}

#[test]
fn test_binning_tie_lower_bin_wins() {
    let (_t, lib) = make_3x3_lib();
    // phi=60.0: equidist from 0 (dist=60) and 120 (dist=60) → first=ind=1(0°); psi=0→ind=1 → 4
    assert_eq!(lib.backbone_bin("TST", 60.0, 0.0, false).unwrap(), 4);
}

#[test]
fn test_binning_unknown_aa() {
    let (_t, lib) = make_3x3_lib();
    assert!(matches!(
        lib.backbone_bin("ZZZ", 0.0, 0.0, false),
        Err(RotlibError::UnknownAa(_))
    ));
}

#[test]
fn test_default_bin_first_wins_on_tie() {
    // Two bins with equal frequency — default_bin must be 0 (first)
    let tmp = helpers::write_minimal_lib(
        "TST",
        &["CB"],
        &[
            helpers::BinSpec {
                phi: -60.0,
                psi: 0.0,
                freq: 5.0,
            },
            helpers::BinSpec {
                phi: 60.0,
                psi: 0.0,
                freq: 5.0,
            },
        ],
        &[helpers::RotSpec {
            prob: 0.5,
            coords: vec![[1.0, 0.0, 0.0]],
        }],
    );
    let lib = RotamerLibrary::load(tmp.path()).unwrap();
    assert_eq!(
        lib.backbone_bin("TST", 9999.0, 9999.0, false).unwrap(),
        0,
        "first-maximum wins on tie"
    );
}
