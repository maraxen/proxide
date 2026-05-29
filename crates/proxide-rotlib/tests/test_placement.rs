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
#[ignore]
fn test_place_parity_mosaist() {
    // Integration test: verify ALA CB placement on real rotlib
    // Reference backbone: N=[0,0,0], CA=[1.458,0,0], C=[1.980,1.418,0]
    // phi=9999 (sentinel), psi=9999 (sentinel), rot=0
    let lib = RotamerLibrary::load(&real_rotlib_path()).unwrap();
    let placed = lib
        .place_rotamer("ALA", 9999.0, 9999.0, 0, N, CA, C)
        .unwrap();

    assert_eq!(placed.atoms.len(), 1, "ALA should have 1 atom (CB)");
    let xyz = placed.atoms[0].xyz;

    // Verify coordinates are finite (sanity check)
    for &v in &xyz {
        assert!(v.is_finite(), "non-finite coordinate: {}", v);
    }

    // Verify CB distance from CA is reasonable (~1.5 Å for C-C bond)
    let dist_from_ca =
        ((xyz[0] - CA[0]).powi(2) + (xyz[1] - CA[1]).powi(2) + (xyz[2] - CA[2]).powi(2)).sqrt();
    assert!(
        dist_from_ca > 0.5 && dist_from_ca < 3.0,
        "CB distance from CA = {:.3} Å, expected ~1.5",
        dist_from_ca
    );
}
