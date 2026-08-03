use proxide_confind::extract_f64_backbone;
use proxide_core::processing::residues::ProcessedStructure;
use std::io::Write;
use tempfile::NamedTempFile;

#[test]
fn extract_f64_backbone_from_synthetic_pdb() {
    // Create a minimal synthetic PDB with 2 residues (ALA + GLY)
    let pdb_content = r#"ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       2.000   2.000   3.000  1.00  0.00           C
ATOM      3  C   ALA A   1       3.000   2.000   3.000  1.00  0.00           C
ATOM      4  O   ALA A   1       3.000   3.000   3.000  1.00  0.00           O
ATOM      5  N   GLY A   2       4.000   2.000   3.000  1.00  0.00           N
ATOM      6  CA  GLY A   2       5.000   2.000   3.000  1.00  0.00           C
ATOM      7  C   GLY A   2       6.000   2.000   3.000  1.00  0.00           C
ATOM      8  O   GLY A   2       6.000   3.000   3.000  1.00  0.00           O
END
"#;

    // Write to temp file
    let mut temp_file = NamedTempFile::new().expect("failed to create temp file");
    temp_file
        .write_all(pdb_content.as_bytes())
        .expect("failed to write PDB content");
    let temp_path = temp_file.path().to_path_buf();

    // Parse with proxide_io
    let (raw, _) =
        proxide_io::formats::pdb::parse_pdb_file(&temp_path).expect("failed to parse PDB");
    let processed = ProcessedStructure::from_raw(raw).expect("failed to create ProcessedStructure");

    // Extract backbone
    let backbone = extract_f64_backbone(&processed).expect("failed to extract backbone");

    // Assertions
    assert_eq!(backbone.bb.len(), 2, "should have 2 residues");

    // Check residue 0 (ALA)
    assert_eq!(backbone.bb[0].res_name, "ALA");
    assert!(backbone.bb[0].ca.is_some());
    assert_eq!(
        backbone.bb[0].phi, 9999.0,
        "N-terminus should have phi=9999"
    );

    // Check residue 1 (GLY)
    assert_eq!(backbone.bb[1].res_name, "GLY");
    assert!(backbone.bb[1].ca.is_some());
    assert_eq!(
        backbone.bb[1].psi, 9999.0,
        "C-terminus should have psi=9999"
    );

    // Check that CA coords are present
    let ca0 = backbone.bb[0].ca.unwrap();
    let ca1 = backbone.bb[1].ca.unwrap();
    assert!((ca0[0] - 2.0).abs() < 1e-6);
    assert!((ca1[0] - 5.0).abs() < 1e-6);
}

#[test]
fn extract_f64_backbone_single_residue_minimal() {
    // Create a minimal single-residue PDB with only required backbone atoms
    let pdb_content = r#"ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.000   0.000   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       2.000   1.000   0.000  1.00  0.00           O
END
"#;

    let mut temp_file = NamedTempFile::new().expect("failed to create temp file");
    temp_file
        .write_all(pdb_content.as_bytes())
        .expect("failed to write PDB content");
    let temp_path = temp_file.path().to_path_buf();

    let (raw, _) =
        proxide_io::formats::pdb::parse_pdb_file(&temp_path).expect("failed to parse PDB");
    let processed = ProcessedStructure::from_raw(raw).expect("failed to create ProcessedStructure");

    let backbone = extract_f64_backbone(&processed).expect("failed to extract backbone");

    assert_eq!(backbone.bb.len(), 1);
    assert_eq!(backbone.ids.len(), 1);
    assert_eq!(backbone.chain_map.len(), 1);
    assert_eq!(backbone.bb[0].res_name, "ALA");
    assert!(backbone.bb[0].ca.is_some());
}
