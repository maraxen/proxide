#[path = "helpers.rs"]
mod helpers;

use helpers::{write_minimal_lib, BinSpec, RotSpec, real_rotlib_path};
use proxide_rotlib::{RotamerLibrary, RotamerId, RotlibError};
use std::io::Write;

/// Create a test library with both ALA (1 atom: CB) and ARG (6 atoms: CB,CG,CD,NE,CZ,NH1).
/// Single bin at (phi=0, psi=0). 1 rotamer per bin.
/// CB stored at [1.0, 0.0, 0.0].
fn make_placement_lib() -> (tempfile::NamedTempFile, RotamerLibrary) {
    let bin = vec![BinSpec {
        phi: 0.0,
        psi: 0.0,
        freq: 1.0,
    }];

    let ala_tmp = write_minimal_lib(
        "ALA",
        &["CB"],
        &bin,
        &[RotSpec {
            prob: 0.9,
            coords: vec![[1.0f32, 0.0, 0.0]],
        }],
    );

    let arg_tmp = write_minimal_lib(
        "ARG",
        &["CB", "CG", "CD", "NE", "CZ", "NH1"],
        &bin,
        &[RotSpec {
            prob: 0.9,
            coords: vec![
                [1.0f32, 0.0, 0.0],
                [2.0f32, 0.0, 0.0],
                [3.0f32, 0.0, 0.0],
                [4.0f32, 0.0, 0.0],
                [5.0f32, 0.0, 0.0],
                [6.0f32, 0.0, 0.0],
            ],
        }],
    );

    // Combine into one library by concatenating the two files
    let mut combined = tempfile::NamedTempFile::new().unwrap();
    let mut ala_file = std::fs::File::open(ala_tmp.path()).unwrap();
    let mut arg_file = std::fs::File::open(arg_tmp.path()).unwrap();
    std::io::copy(&mut ala_file, &mut combined).unwrap();
    std::io::copy(&mut arg_file, &mut combined).unwrap();
    combined.flush().unwrap();

    let lib = RotamerLibrary::load(combined.path()).unwrap();
    (combined, lib)
}

// Backbone coordinates for tests
const N: [f64; 3] = [0.0, 0.0, 0.0];
const CA: [f64; 3] = [1.458, 0.0, 0.0];
const C: [f64; 3] = [1.980, 1.418, 0.0];

#[test]
fn test_place_ala_one_atom() {
    let (_lib_tmp, lib) = make_placement_lib();
    let placed = lib.place_rotamer("ALA", 0.0, 0.0, 0, N, CA, C).unwrap();
    assert_eq!(placed.atoms.len(), 1);
    assert_eq!(placed.atoms[0].name, "CB");
}

#[test]
fn test_place_arg_atom_count() {
    let (_lib_tmp, lib) = make_placement_lib();
    let placed = lib.place_rotamer("ARG", 0.0, 0.0, 0, N, CA, C).unwrap();
    assert!(
        placed.atoms.len() >= 5,
        "expected >=5 atoms, got {}",
        placed.atoms.len()
    );
}

#[test]
fn test_place_no_backbone_atoms() {
    let (_lib_tmp, lib) = make_placement_lib();
    let placed = lib.place_rotamer("ARG", 0.0, 0.0, 0, N, CA, C).unwrap();
    for atom in &placed.atoms {
        assert!(
            !["N", "CA", "C", "O"].contains(&atom.name.as_str()),
            "backbone atom '{}' found in sidechain output",
            atom.name
        );
    }
}

#[test]
fn test_place_correct_rotamer_id() {
    let (_lib_tmp, lib) = make_placement_lib();
    let placed = lib.place_rotamer("ALA", 0.0, 0.0, 0, N, CA, C).unwrap();
    assert_eq!(
        placed.id,
        RotamerId {
            aa: "ALA".to_string(),
            bin_index: 0,
            rot_index: 0
        }
    );
}

#[test]
fn test_place_unknown_aa() {
    let (_lib_tmp, lib) = make_placement_lib();
    let result = lib.place_rotamer("ZZZ", 0.0, 0.0, 0, N, CA, C);
    assert!(matches!(
        result,
        Err(RotlibError::UnknownAa(ref aa)) if aa == "ZZZ"
    ));
}

#[test]
fn test_place_oob_rot_index() {
    let (_lib_tmp, lib) = make_placement_lib();
    // ALA has only 1 rotamer (rot=0); rot=99 is out of bounds
    let result = lib.place_rotamer("ALA", 0.0, 0.0, 99, N, CA, C);
    assert!(matches!(
        result,
        Err(RotlibError::RotIndexOob(ref aa, 99, 1)) if aa == "ALA"
    ));
}

#[test]
fn test_place_sentinel_uses_default_bin() {
    let (_lib_tmp, lib) = make_placement_lib();
    // phi=9999, psi=9999 -> sentinel -> default_bin
    // With only 1 bin, default_bin=0
    let placed = lib
        .place_rotamer("ALA", 9999.0, 9999.0, 0, N, CA, C)
        .unwrap();
    assert_eq!(placed.id.bin_index, 0);
}

#[test]
fn test_place_ala_cb_exact_coords() {
    // Unit-level exact coordinate check: CB stored at [1.0, 0.0, 0.0] in canonical frame.
    // Backbone: N=[0,0,0], CA=[1.458,0,0], C=[1.980,1.418,0].
    // Expected lab-frame CB = [2.458, 0.0, 0.0] (derived analytically).
    let (_lib_tmp, lib) = make_placement_lib();
    let placed = lib.place_rotamer("ALA", 0.0, 0.0, 0, N, CA, C).unwrap();
    let cb_xyz = placed.atoms[0].xyz;
    let expected = [2.458, 0.0, 0.0];
    for i in 0..3 {
        assert!((cb_xyz[i] - expected[i]).abs() < 1e-9,
            "CB coord[{i}]: expected {:.6}, got {:.6}", expected[i], cb_xyz[i]);
    }
}

#[test]
fn test_place_parity_mosaist() {
    // Reference CB in lab frame derived from rotlib.bin canonical coords
    // (f32 → f64: [0.519, -0.670, -1.262]) placed onto the test backbone.
    // For N=[0,0,0], CA=[1.458,0,0], C=[1.980,1.418,0] the residue frame
    // axes align with lab axes, so lab_CB = CA + canonical_CB.
    // Verified independently by reading rotlib.bin directly in C++ using the
    // same parsing path as Mosaist (mstrotlib.cpp placeRotamer lines 182–194).
    const REF_CB: [f64; 3] = [1.976_999_993_801_117, -0.670_000_016_689_300_5, -1.261_999_964_714_050_3];

    let lib = RotamerLibrary::load(&real_rotlib_path()).unwrap();
    let placed = lib
        .place_rotamer("ALA", 9999.0, 9999.0, 0, N, CA, C)
        .unwrap();

    assert_eq!(placed.atoms.len(), 1, "ALA should have 1 atom (CB)");
    let xyz = placed.atoms[0].xyz;

    for i in 0..3 {
        assert!(
            (xyz[i] - REF_CB[i]).abs() < 1e-5,
            "CB coord[{i}]: expected {:.9}, got {:.9}, diff {:.2e}",
            REF_CB[i], xyz[i], (xyz[i] - REF_CB[i]).abs()
        );
    }
}
