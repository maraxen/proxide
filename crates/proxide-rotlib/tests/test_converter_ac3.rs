/// AC-3 acceptance test for the P4 Converter.
///
/// Tests that the converter produces a valid rotamer library with correct structure
/// and contains expected residues and rotamers.
use proxide_rotlib::pb::rotlib_v1;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::process::Command;

/// AC-3 test: After running the converter on the real input file, load the output
/// and assert key properties.
#[test]
#[ignore] // Requires large input file and some time to run
fn test_converter_ac3_full_library() {
    // Run the converter
    let input_path = Path::new(
        "/home/marielle/projects/proxide/data/rotlibs/SimpleOpt1-5/ALL.bbdep.rotamers.lib",
    );
    let output_path = Path::new("/tmp/ac3_test_rotlib.pb.zst");

    if !input_path.exists() {
        eprintln!(
            "Input file not found: {}. Skipping test.",
            input_path.display()
        );
        return;
    }

    // Build and run the converter
    let output = Command::new("cargo")
        .args(&[
            "run",
            "--release",
            "--",
            "--input",
            input_path.to_str().unwrap(),
            "--output",
            output_path.to_str().unwrap(),
        ])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("Failed to run converter");

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("Converter failed:\n{}", stderr);
    }

    // Load and verify the output
    verify_rotamer_library(output_path);

    // Clean up
    let _ = std::fs::remove_file(output_path);
}

/// Verify key properties of the generated rotamer library.
fn verify_rotamer_library(path: &Path) {
    // Read and decompress
    let mut file = File::open(path).expect("Failed to open output file");
    let mut compressed = Vec::new();
    file.read_to_end(&mut compressed)
        .expect("Failed to read file");

    let decompressed = zstd::decode_all(&compressed[..]).expect("Failed to decompress");

    // Decode protobuf
    let lib: rotlib_v1::RotamerLibrary =
        prost::Message::decode(&decompressed[..]).expect("Failed to decode protobuf");

    // AC-3 assertions

    // 1. Attribution field is non-empty
    assert!(!lib.attribution.is_empty(), "Attribution field is empty");
    println!(
        "✓ Attribution field is non-empty: {}",
        &lib.attribution[..100]
    );

    // 2. data_license == "ODC-BY-1.0"
    assert_eq!(
        lib.data_license, "ODC-BY-1.0",
        "data_license should be ODC-BY-1.0, got: {}",
        lib.data_license
    );
    println!("✓ data_license == ODC-BY-1.0");

    // 3. geometry_mode == PRECOMPUTED (value 1)
    assert_eq!(
        lib.geometry_mode, 1,
        "geometry_mode should be PRECOMPUTED (1), got: {}",
        lib.geometry_mode
    );
    println!("✓ geometry_mode == PRECOMPUTED");

    // 4. All 22 residue codes present in Dunbrack library
    // (Note: ALA and GLY are not rotameric, so not in BBDEP library)
    let expected_codes = vec![
        "ARG", "ASN", "ASP", "CPR", "CYD", "CYH", "CYS", "GLN", "GLU", "HIS", "ILE", "LEU", "LYS",
        "MET", "PHE", "PRO", "SER", "THR", "TPR", "TRP", "TYR", "VAL",
    ];
    let actual_codes: Vec<&str> = lib.residues.iter().map(|r| r.code.as_str()).collect();

    assert_eq!(
        actual_codes.len(),
        22,
        "Expected 22 residues, got {}",
        actual_codes.len()
    );

    for code in &expected_codes {
        assert!(
            actual_codes.contains(code),
            "Missing residue code: {}",
            code
        );
    }
    println!("✓ All 22 residue codes present: {:?}", actual_codes);

    // 5. For PRO at (phi=-60, psi=-40): 2+ rotamers, first rotamer prob > 0
    let pro_entry = lib
        .residues
        .iter()
        .find(|r| r.code == "PRO")
        .expect("PRO not found");

    let pro_bin = pro_entry
        .bins
        .iter()
        .find(|b| {
            (b.phi as i32 == -60 || b.phi as i32 == -59 || b.phi as i32 == -61)
                && (b.psi as i32 == -40 || b.psi as i32 == -39 || b.psi as i32 == -41)
        })
        .expect("PRO bin at phi=-60, psi=-40 not found");

    assert!(
        pro_bin.rotamers.len() >= 2,
        "Expected at least 2 rotamers for PRO at phi=-60, psi=-40, got {}",
        pro_bin.rotamers.len()
    );

    assert!(
        pro_bin.rotamers[0].prob > 0.0,
        "First rotamer probability should be > 0, got {}",
        pro_bin.rotamers[0].prob
    );
    println!(
        "✓ PRO at (phi=-60, psi=-40): {} rotamers, first prob = {}",
        pro_bin.rotamers.len(),
        pro_bin.rotamers[0].prob
    );

    // 6. For CYS at (phi=-60, psi=-40): rotamers present
    let cys_entry = lib
        .residues
        .iter()
        .find(|r| r.code == "CYS")
        .expect("CYS not found");

    let cys_bin = cys_entry
        .bins
        .iter()
        .find(|b| {
            (b.phi as i32 == -60 || b.phi as i32 == -59 || b.phi as i32 == -61)
                && (b.psi as i32 == -40 || b.psi as i32 == -39 || b.psi as i32 == -41)
        })
        .expect("CYS bin at phi=-60, psi=-40 not found");

    assert!(!cys_bin.rotamers.is_empty(), "CYS bin should have rotamers");
    println!(
        "✓ CYS at (phi=-60, psi=-40): {} rotamers",
        cys_bin.rotamers.len()
    );

    println!("✓✓✓ All AC-3 assertions passed!");
}
