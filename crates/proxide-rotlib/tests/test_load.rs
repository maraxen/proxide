#[path = "helpers.rs"]
mod helpers;
use helpers::{real_rotlib_path, write_minimal_lib, BinSpec, RotSpec};
use proxide_rotlib::{RotamerLibrary, RotlibError};

const AA_NAMES: &[&str] = &[
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET",
    "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
];

#[test]
fn test_load_all_aa_names_present() {
    let lib = RotamerLibrary::load(&real_rotlib_path()).unwrap();
    for &aa in AA_NAMES {
        assert!(lib.contains_aa(aa), "missing AA: {aa}");
    }
}

#[test]
fn test_num_rotamers_sentinel_positive() {
    let lib = RotamerLibrary::load(&real_rotlib_path()).unwrap();
    for &aa in AA_NAMES {
        let n = lib.num_rotamers(aa, 9999.0, 9999.0, false).unwrap();
        assert!(n > 0, "{aa}: sentinel bin has 0 rotamers");
    }
}

#[test]
fn test_load_truncated_file() {
    // Write a file that starts reading a record but is truncated mid-i32
    // "TST\0" (4 bytes) + partial i32 (2 bytes instead of 4) = 6 bytes
    let tmp = write_minimal_lib(
        "TST",
        &["CB"],
        &[BinSpec {
            phi: 0.0,
            psi: 0.0,
            freq: 1.0,
        }],
        &[RotSpec {
            prob: 0.9,
            coords: vec![[1.0, 0.0, 0.0]],
        }],
    );
    let path = tmp.path();
    // Truncate after AA name + 2 bytes of nc field (need 4 bytes total)
    std::fs::write(path, b"TST\x00\x00\x00").unwrap();
    let result = RotamerLibrary::load(path);
    assert!(
        matches!(result, Err(RotlibError::Io(_))),
        "expected Io error, got: {:?}",
        result
    );
}

#[test]
fn test_load_non_rectangular_grid() {
    // 3 bins but unique_phi=2, unique_psi=2 → product=4 ≠ 3
    let tmp = write_minimal_lib(
        "TST",
        &["CB"],
        &[
            BinSpec {
                phi: -60.0,
                psi: -60.0,
                freq: 1.0,
            },
            BinSpec {
                phi: -60.0,
                psi: 60.0,
                freq: 1.0,
            },
            BinSpec {
                phi: 60.0,
                psi: -60.0,
                freq: 1.0,
            },
        ],
        &[RotSpec {
            prob: 0.9,
            coords: vec![[1.0, 0.0, 0.0]],
        }],
    );
    let result = RotamerLibrary::load(tmp.path());
    assert!(
        matches!(result, Err(RotlibError::InvalidFormat(_))),
        "expected InvalidFormat, got: {:?}",
        result
    );
}

#[test]
fn test_load_duplicate_phi_psi() {
    // 2 bins with identical (phi, psi)
    let tmp = write_minimal_lib(
        "TST",
        &["CB"],
        &[
            BinSpec {
                phi: -60.0,
                psi: -40.0,
                freq: 1.0,
            },
            BinSpec {
                phi: -60.0,
                psi: -40.0,
                freq: 2.0,
            },
        ],
        &[RotSpec {
            prob: 0.9,
            coords: vec![[1.0, 0.0, 0.0]],
        }],
    );
    let result = RotamerLibrary::load(tmp.path());
    assert!(
        matches!(result, Err(RotlibError::InvalidFormat(_))),
        "expected InvalidFormat, got: {:?}",
        result
    );
}
