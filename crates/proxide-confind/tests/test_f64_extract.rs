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
    assert_eq!(backbone.bb[0].phi, None, "N-terminus should have phi=None");

    // Check residue 1 (GLY)
    assert_eq!(backbone.bb[1].res_name, "GLY");
    assert!(backbone.bb[1].ca.is_some());
    assert_eq!(backbone.bb[1].psi, None, "C-terminus should have psi=None");

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

#[test]
fn chain_terminal_undefined_angle_reaches_rotlib_and_records_default_bin() {
    use proxide_confind::{ConFind, ResidueIndex};
    use proxide_rotlib::pb::rotlib_v1::{
        Bin, GeometryMode, ResidueEntry, Rotamer, RotamerLibrary as PbRotamerLibrary, Vec3,
    };
    use proxide_rotlib::RotamerLibrary;
    use std::sync::Arc;

    // 3-residue synthetic peptide (ALA-ALA-ALA) with realistic non-collinear geometry
    let pdb_content = r#"ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.251   2.390   0.000  1.00  0.00           O
ATOM      5  N   ALA A   2       3.330   1.540   0.000  1.00  0.00           N
ATOM      6  CA  ALA A   2       3.980   2.845   0.000  1.00  0.00           C
ATOM      7  C   ALA A   2       5.490   2.710   0.000  1.00  0.00           C
ATOM      8  O   ALA A   2       6.030   1.600   0.000  1.00  0.00           O
ATOM      9  N   ALA A   3       6.160   3.860   0.000  1.00  0.00           N
ATOM     10  CA  ALA A   3       7.610   3.890   0.000  1.00  0.00           C
ATOM     11  C   ALA A   3       8.160   5.310   0.000  1.00  0.00           C
ATOM     12  O   ALA A   3       7.400   6.280   0.000  1.00  0.00           O
END
"#;

    let mut temp_file = NamedTempFile::new().expect("failed to create temp file");
    temp_file
        .write_all(pdb_content.as_bytes())
        .expect("failed to write PDB content");
    let (raw, _) =
        proxide_io::formats::pdb::parse_pdb_file(temp_file.path()).expect("failed to parse PDB");
    let processed = ProcessedStructure::from_raw(raw).expect("failed to create ProcessedStructure");
    let backbone = extract_f64_backbone(&processed).expect("failed to extract backbone");

    // Verify N-terminal phi is None, interior phi/psi are Some, C-terminal psi is None
    assert_eq!(backbone.bb[0].phi, None);
    assert!(backbone.bb[0].psi.is_some());
    assert!(backbone.bb[1].phi.is_some());
    assert!(backbone.bb[1].psi.is_some());
    assert!(backbone.bb[2].phi.is_some());
    assert_eq!(backbone.bb[2].psi, None);

    // Build minimal synthetic RotamerLibrary covering all 20 AAs so ConFind::cache_residue runs
    let aas = [
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET",
        "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    ];
    let residues = aas
        .iter()
        .map(|&code| ResidueEntry {
            code: code.to_string(),
            atom_names: if code == "GLY" {
                vec![]
            } else {
                vec!["CB".to_string()]
            },
            num_chi: 0,
            phi_centers: vec![-120.0, -60.0],
            psi_centers: vec![-45.0, 120.0],
            default_bin: 2,
            bins: [
                (-120.0, -45.0),
                (-120.0, 120.0),
                (-60.0, -45.0),
                (-60.0, 120.0),
            ]
            .iter()
            .map(|&(phi, psi)| Bin {
                phi,
                psi,
                freq: 0.25,
                rotamers: vec![Rotamer {
                    prob: 1.0,
                    coords: if code == "GLY" {
                        vec![]
                    } else {
                        vec![Vec3 {
                            x: 0.5,
                            y: 1.2,
                            z: 0.8,
                        }]
                    },
                    chi: vec![],
                }],
            })
            .collect(),
        })
        .collect();
    let pb_lib = PbRotamerLibrary {
        version: 1,
        provenance: "synthetic".to_string(),
        attribution: "synthetic".to_string(),
        data_license: "MIT".to_string(),
        geometry_mode: GeometryMode::Precomputed as i32,
        residues,
        geometry_source: String::new(),
        geometry_license: String::new(),
    };
    let rotlib = Arc::new(RotamerLibrary::from_protobuf(&pb_lib).unwrap());

    // Direct rotlib boundary verification: chain-terminal residue 0 (phi=None) triggers default_bin
    let (bin0, fb0) = rotlib
        .backbone_bin_with_fallback(
            &backbone.bb[0].res_name,
            backbone.bb[0].phi,
            backbone.bb[0].psi,
            backbone.bb[0].is_cis_peptide,
        )
        .unwrap();
    assert_eq!(
        bin0, 2,
        "chain-terminal residue must resolve to default_bin"
    );
    assert!(
        fb0,
        "chain-terminal residue must record used_default_bin=true"
    );

    // Interior residue 1 (phi=Some, psi=Some) does NOT use default_bin fallback
    let (_, fb1) = rotlib
        .backbone_bin_with_fallback(
            &backbone.bb[1].res_name,
            backbone.bb[1].phi,
            backbone.bb[1].psi,
            backbone.bb[1].is_cis_peptide,
        )
        .unwrap();
    assert!(!fb1, "interior residue must record used_default_bin=false");

    // End-to-end ConFind verification
    let cf = ConFind::new(rotlib, Arc::new(backbone), false).unwrap();
    assert!(
        cf.used_default_bin(ResidueIndex(0)).unwrap(),
        "N-terminal residue must record used_default_bin=true in ConFind"
    );
    assert!(
        !cf.used_default_bin(ResidueIndex(1)).unwrap(),
        "Interior residue must record used_default_bin=false in ConFind"
    );
    assert!(
        cf.used_default_bin(ResidueIndex(2)).unwrap(),
        "C-terminal residue must record used_default_bin=true in ConFind"
    );
}

/// Interior φ, ψ and ω, including sign, against a hand-computed IUPAC torsion.
///
/// Convention (spec §3.5): looking along B→C, the angle is positive when A→B must
/// rotate clockwise to eclipse C→D. The calibration quadruplet
/// A=(0,1,0), B=(0,0,0), C=(1,0,0), D=(1,0,1) is +90°. The same right-handed
/// formula, applied to the orthogonal coordinates below, fixes every angle here
/// to a multiple of 90° (or 0° / −180°). `is_cis_peptide` is |ω| < 30°.
///
/// Atoms (Å), residue order preserved, consecutive Cα–Cα ≤ 2.3 Å so no chain break:
///   res 1: N(0,1,−1) CA(0,1,0) C(0,0,0)
///   res 2: N(1,0,0)  CA(1,0,1) C(1,1,1)
///   res 3: N(0,1,1)  CA(0,0,1) C(0,0,0)
///   res 4: N(1,0,0)  CA(1,0,−1) C(1,1,−1)
#[test]
fn interior_dihedrals_match_iupac_sign() {
    let pdb_content = "\
ATOM      1  N   ALA A   1       0.000   1.000  -1.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       0.000   1.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       0.000   0.000   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1      -1.000   0.000   0.000  1.00  0.00           O
ATOM      5  N   ALA A   2       1.000   0.000   0.000  1.00  0.00           N
ATOM      6  CA  ALA A   2       1.000   0.000   1.000  1.00  0.00           C
ATOM      7  C   ALA A   2       1.000   1.000   1.000  1.00  0.00           C
ATOM      8  O   ALA A   2       2.000   1.000   1.000  1.00  0.00           O
ATOM      9  N   ALA A   3       0.000   1.000   1.000  1.00  0.00           N
ATOM     10  CA  ALA A   3       0.000   0.000   1.000  1.00  0.00           C
ATOM     11  C   ALA A   3       0.000   0.000   0.000  1.00  0.00           C
ATOM     12  O   ALA A   3       0.000  -1.000   0.000  1.00  0.00           O
ATOM     13  N   ALA A   4       1.000   0.000   0.000  1.00  0.00           N
ATOM     14  CA  ALA A   4       1.000   0.000  -1.000  1.00  0.00           C
ATOM     15  C   ALA A   4       1.000   1.000  -1.000  1.00  0.00           C
ATOM     16  O   ALA A   4       2.000   1.000  -1.000  1.00  0.00           O
END
";

    let mut temp_file = NamedTempFile::new().expect("failed to create temp file");
    temp_file
        .write_all(pdb_content.as_bytes())
        .expect("failed to write PDB content");
    let (raw, _) =
        proxide_io::formats::pdb::parse_pdb_file(temp_file.path()).expect("failed to parse PDB");
    let processed = ProcessedStructure::from_raw(raw).expect("failed to create ProcessedStructure");
    let backbone = extract_f64_backbone(&processed).expect("failed to extract backbone");
    assert_eq!(backbone.bb.len(), 4);

    let angle = |got: Option<f64>, expected: f64, label: &str| {
        let got = got.unwrap_or_else(|| panic!("{label} is undefined"));
        assert!(
            (got - expected).abs() < 1e-6,
            "{label}: got {got}, IUPAC expected {expected}"
        );
    };

    // N-terminus: φ and ω undefined. ψ(N, CA, C, N_next) = +90.
    assert_eq!(backbone.bb[0].phi, None);
    assert_eq!(backbone.bb[0].omega, None);
    assert!(!backbone.bb[0].is_cis_peptide);
    angle(backbone.bb[0].psi, 90.0, "psi0");

    // Interior residue 1: φ = −90, ψ = +90, ω = +90 (trans).
    angle(backbone.bb[1].phi, -90.0, "phi1");
    angle(backbone.bb[1].psi, 90.0, "psi1");
    angle(backbone.bb[1].omega, 90.0, "omega1");
    assert!(!backbone.bb[1].is_cis_peptide);

    // Interior residue 2: ω = 0, so |ω| < 30 and the peptide is cis.
    angle(backbone.bb[2].phi, -90.0, "phi2");
    angle(backbone.bb[2].psi, 90.0, "psi2");
    angle(backbone.bb[2].omega, 0.0, "omega2");
    assert!(backbone.bb[2].is_cis_peptide);

    // C-terminus: ψ undefined. ω = −180 (trans; the branch of atan2, not +180).
    assert_eq!(backbone.bb[3].psi, None);
    angle(backbone.bb[3].phi, 90.0, "phi3");
    angle(backbone.bb[3].omega, -180.0, "omega3");
    assert!(!backbone.bb[3].is_cis_peptide);
}
