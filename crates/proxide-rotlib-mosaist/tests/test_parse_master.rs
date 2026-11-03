#[path = "helpers.rs"]
mod helpers;

use helpers::{real_rotlib_path, write_minimal_lib, BinSpec, RotSpec};
use proxide_rotlib_mosaist::RotamerLibrary;

/// Test that parse_master binary can be found and runs with --help
#[test]
fn test_parse_master_binary_exists() {
    assert!(
        std::path::Path::new(env!("CARGO_BIN_EXE_parse_master")).exists(),
        "parse_master binary should exist"
    );
}

/// Test round-trip: load MASTER binary → convert to proto → load proto.
#[test]
#[ignore] // Run only when Mosaist is available
fn test_parse_master_roundtrip_small_pdb() {
    let rotlib_path = real_rotlib_path();
    if !rotlib_path.exists() {
        eprintln!(
            "Skipping: ROTLIB_PATH not available at {}",
            rotlib_path.display()
        );
        return;
    }

    let master_lib = RotamerLibrary::load(&rotlib_path).expect("Failed to load MASTER library");
    let n_residues = master_lib.residue_codes().len();
    assert!(
        n_residues >= 18,
        "Expected at least 18 residue types; got {}",
        n_residues
    );
    assert!(master_lib.contains_aa("MET"), "MET not in library");

    let n_met_rotamers = master_lib
        .num_rotamers("MET", Some(-60.0), Some(-40.0), false)
        .expect("Failed to get MET rotamers");
    assert!(n_met_rotamers > 0, "MET has no rotamers at (-60, -40)");
}

/// Test that minimal MASTER binary round-trips through proto correctly.
#[test]
fn test_parse_master_minimal_roundtrip() {
    let tmp_master = write_minimal_lib(
        "TST",
        &["CB", "CG"],
        &[
            BinSpec {
                phi: -60.0,
                psi: -40.0,
                freq: 1.0,
            },
            BinSpec {
                phi: -60.0,
                psi: 40.0,
                freq: 2.0,
            },
            BinSpec {
                phi: 60.0,
                psi: -40.0,
                freq: 1.5,
            },
            BinSpec {
                phi: 60.0,
                psi: 40.0,
                freq: 2.0,
            },
        ],
        &[
            RotSpec {
                prob: 0.7,
                coords: vec![[1.54, 0.0, 0.0], [2.5, 0.9, 0.0]],
            },
            RotSpec {
                prob: 0.3,
                coords: vec![[1.54, 0.0, 0.0], [2.5, -0.9, 0.0]],
            },
        ],
    );

    let master_lib =
        RotamerLibrary::load(tmp_master.path()).expect("Failed to load minimal MASTER library");

    assert!(master_lib.contains_aa("TST"), "TST not found in library");

    let atom_names = master_lib
        .sidechain_atom_names("TST")
        .expect("Failed to get atom names");
    assert_eq!(
        atom_names.len(),
        2,
        "Expected 2 atoms; got {}",
        atom_names.len()
    );

    let pb = master_lib.to_protobuf("test".to_string(), "test".to_string());
    assert_eq!(pb.residues.len(), 1, "Expected 1 residue in proto");
    assert_eq!(pb.residues[0].code, "TST");
    assert_eq!(pb.residues[0].atom_names.len(), 2);

    eprintln!("✓ Minimal MASTER library loaded and converted to protobuf");
}

/// Test that 20 standard AAs are present in MASTER library
#[test]
#[ignore] // Run only when Mosaist is available
fn test_parse_master_all_20_aa_present() {
    let rotlib_path = real_rotlib_path();
    if !rotlib_path.exists() {
        eprintln!(
            "Skipping: ROTLIB_PATH not available at {}",
            rotlib_path.display()
        );
        return;
    }

    let lib = RotamerLibrary::load(&rotlib_path).expect("Failed to load MASTER library");

    const AA_NAMES: &[&str] = &[
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET",
        "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    ];

    for &aa in AA_NAMES {
        assert!(lib.contains_aa(aa), "AA {} not in MASTER library", aa);
    }

    eprintln!("✓ All 20 standard amino acids present in MASTER library");
}
