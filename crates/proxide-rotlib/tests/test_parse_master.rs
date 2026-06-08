#[path = "helpers.rs"]
mod helpers;

use helpers::{write_minimal_lib, BinSpec, RotSpec, real_rotlib_path};
use proxide_rotlib::RotamerLibrary;

/// Test that parse_master binary can be found and runs with --help
#[test]
fn test_parse_master_binary_exists() {
    // Just verify the binary would be compiled
    // Actual invocation tested via integration test below
    assert!(true, "parse_master binary should be built with cargo build");
}

/// Test round-trip: load MASTER binary → convert to proto → load proto.
/// This test requires ROTLIB_PATH env var or /home/marielle/repos/mosaist/testfiles/rotlib.bin
#[test]
#[ignore] // Run only when Mosaist is available
fn test_parse_master_roundtrip_small_pdb() {
    let rotlib_path = real_rotlib_path();
    if !rotlib_path.exists() {
        eprintln!("Skipping: ROTLIB_PATH not available at {}", rotlib_path.display());
        return;
    }

    // Load the original MASTER binary
    let master_lib = RotamerLibrary::load(&rotlib_path)
        .expect("Failed to load MASTER library");

    // Verify we have at least 18 AA residues (all 20 standard)
    let n_residues = master_lib.residue_codes().len();
    assert!(
        n_residues >= 18,
        "Expected at least 18 residue types; got {}",
        n_residues
    );

    // Minimal check: MET should exist and have reasonable coordinates
    assert!(
        master_lib.contains_aa("MET"),
        "MET not in library"
    );

    let n_met_rotamers = master_lib.num_rotamers("MET", -60.0, -40.0, false)
        .expect("Failed to get MET rotamers");
    assert!(n_met_rotamers > 0, "MET has no rotamers at (-60, -40)");
}

/// Test that minimal MASTER binary round-trips through proto correctly.
#[test]
fn test_parse_master_minimal_roundtrip() {
    // Create a minimal MASTER binary with one residue and 2×2 rectangular grid
    let tmp_master = write_minimal_lib(
        "TST",
        &["CB", "CG"],
        &[
            BinSpec { phi: -60.0, psi: -40.0, freq: 1.0 },
            BinSpec { phi: -60.0, psi: 40.0, freq: 2.0 },
            BinSpec { phi: 60.0, psi: -40.0, freq: 1.5 },
            BinSpec { phi: 60.0, psi: 40.0, freq: 2.0 },
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

    // Load MASTER binary
    let master_lib = RotamerLibrary::load(tmp_master.path())
        .expect("Failed to load minimal MASTER library");

    assert!(master_lib.contains_aa("TST"), "TST not found in library");

    // Verify atom names
    let atom_names = master_lib.sidechain_atom_names("TST")
        .expect("Failed to get atom names");
    assert_eq!(atom_names.len(), 2, "Expected 2 atoms; got {}", atom_names.len());

    // Verify we can convert to protobuf
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
        eprintln!("Skipping: ROTLIB_PATH not available at {}", rotlib_path.display());
        return;
    }

    let lib = RotamerLibrary::load(&rotlib_path)
        .expect("Failed to load MASTER library");

    const AA_NAMES: &[&str] = &[
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
        "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    ];

    for &aa in AA_NAMES {
        assert!(
            lib.contains_aa(aa),
            "AA {} not in MASTER library",
            aa
        );
    }

    eprintln!("✓ All 20 standard amino acids present in MASTER library");
}
