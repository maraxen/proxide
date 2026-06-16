use prost::Message;
use proxide_rotlib::pb::rotlib_v1::{
    RotamerLibrary, ResidueEntry, Bin, Rotamer, ChiValue, Vec3, GeometryMode,
    ResidueGeometryTable, ResidueGeometry, IcRecord,
};

fn minimal_library() -> RotamerLibrary {
    RotamerLibrary {
        version: 1,
        provenance: "test".to_string(),
        attribution: "Contains information from the 2010 Backbone-Dependent Rotamer Library \
            (http://dunbrack.fccc.edu/bbdep2010), made available under the ODC Attribution \
            License.".to_string(),
        data_license: "ODC-BY-1.0".to_string(),
        geometry_mode: GeometryMode::Precomputed as i32,
        residues: vec![ResidueEntry {
            code: "CPR".to_string(),
            atom_names: vec!["CB".to_string(), "CG".to_string(), "CD".to_string()],
            num_chi: 3,
            phi_centers: vec![-180.0],
            psi_centers: vec![-180.0],
            default_bin: 0,
            bins: vec![Bin {
                phi: -180.0,
                psi: -180.0,
                freq: 1.0,
                rotamers: vec![
                    Rotamer {
                        prob: 0.6,
                        chi: vec![
                            ChiValue { val: 32.5, sigma: 6.0 },
                            ChiValue { val: -36.0, sigma: 8.0 },
                            ChiValue { val: 25.1, sigma: 8.0 },
                        ],
                        coords: vec![
                            Vec3 { x: 1.0, y: 0.5, z: 0.2 },
                            Vec3 { x: 2.1, y: 0.8, z: 0.1 },
                            Vec3 { x: 2.8, y: -0.3, z: 0.4 },
                        ],
                    },
                    Rotamer {
                        prob: 0.4,
                        chi: vec![
                            ChiValue { val: -20.3, sigma: 6.0 },
                            ChiValue { val: 34.0, sigma: 8.0 },
                            ChiValue { val: -33.8, sigma: 8.0 },
                        ],
                        coords: vec![
                            Vec3 { x: 1.0, y: -0.5, z: 0.2 },
                            Vec3 { x: 2.0, y: -0.9, z: 0.1 },
                            Vec3 { x: 2.7, y: 0.3, z: 0.3 },
                        ],
                    },
                ],
            }],
        }],
        geometry_source: String::new(),
        geometry_license: String::new(),
    }
}

#[test]
fn test_roundtrip_prost_only() {
    let lib = minimal_library();
    let bytes = lib.encode_to_vec();
    assert!(!bytes.is_empty(), "encoded bytes must be non-empty");
    let decoded = RotamerLibrary::decode(bytes.as_slice()).expect("prost decode failed");
    assert_eq!(decoded.version, 1);
    assert_eq!(decoded.attribution, lib.attribution);
    assert_eq!(decoded.data_license, "ODC-BY-1.0");
    assert_eq!(decoded.residues.len(), 1);
    assert_eq!(decoded.residues[0].code, "CPR");
    assert_eq!(decoded.residues[0].bins[0].rotamers.len(), 2);
    // Check chi values are losslessly preserved (f32)
    let r0 = &decoded.residues[0].bins[0].rotamers[0];
    assert!((r0.chi[0].val - 32.5).abs() < 1e-4, "chi1 round-trip: got {}", r0.chi[0].val);
    assert!((r0.chi[1].val - (-36.0)).abs() < 1e-4, "chi2 round-trip");
}

#[test]
fn test_roundtrip_with_zstd() {
    let lib = minimal_library();
    let prost_bytes = lib.encode_to_vec();
    // Compress
    let compressed = zstd::encode_all(prost_bytes.as_slice(), 3)
        .expect("zstd compress failed");
    assert!(compressed.len() < prost_bytes.len() + 100, "compressed should not be vastly larger");
    // Decompress
    let decompressed = zstd::decode_all(compressed.as_slice())
        .expect("zstd decompress failed");
    assert_eq!(decompressed, prost_bytes, "zstd round-trip must be byte-identical");
    // Decode protobuf
    let decoded = RotamerLibrary::decode(decompressed.as_slice()).expect("prost decode after zstd failed");
    assert_eq!(decoded.version, lib.version);
    assert_eq!(decoded.attribution, lib.attribution);
    assert_eq!(decoded.residues.len(), 1);
}

#[test]
fn test_attribution_field_present() {
    // The spec requires attribution to be non-empty — verify the schema carries it.
    let lib = minimal_library();
    let bytes = lib.encode_to_vec();
    let decoded = RotamerLibrary::decode(bytes.as_slice()).unwrap();
    assert!(!decoded.attribution.is_empty(), "attribution must survive round-trip non-empty");
}

fn minimal_geometry_table() -> ResidueGeometryTable {
    ResidueGeometryTable {
        source: "charmm36".to_string(),
        version: "charmm36_jul2024".to_string(),
        license: "NOT-OSI: MacKerell lab academic use only; see https://mackerell.umaryland.edu/charmm_ff.shtml".to_string(),
        citation: "doi:10.1021/jp973084f".to_string(),
        residues: vec![
            ResidueGeometry {
                name: "MET".to_string(),
                ic: vec![
                    IcRecord {
                        atom_i: "CB".to_string(),
                        atom_j: "CG".to_string(),
                        atom_k: "SD".to_string(),
                        atom_l: "CE".to_string(),
                        branch: false,
                        b_ij: 1.5460,
                        theta_ijk: 110.2800,
                        phi_ijkl: 180.0000,
                        theta_jkl: 98.9400,
                        b_kl: 1.8206,
                    },
                ],
            },
        ],
    }
}

#[test]
fn test_residue_geometry_table_roundtrip_prost() {
    let table = minimal_geometry_table();
    let bytes = table.encode_to_vec();
    assert!(!bytes.is_empty(), "encoded geometry table must be non-empty");

    let decoded = ResidueGeometryTable::decode(bytes.as_slice())
        .expect("prost decode of ResidueGeometryTable failed");

    assert_eq!(decoded.source, "charmm36");
    assert_eq!(decoded.version, "charmm36_jul2024");
    assert!(decoded.license.contains("academic"));
    assert!(decoded.citation.contains("doi:"));
    assert_eq!(decoded.residues.len(), 1);
    assert_eq!(decoded.residues[0].name, "MET");
    assert_eq!(decoded.residues[0].ic.len(), 1);

    let ic = &decoded.residues[0].ic[0];
    assert_eq!(ic.atom_i, "CB");
    assert_eq!(ic.atom_j, "CG");
    assert_eq!(ic.atom_k, "SD");
    assert_eq!(ic.atom_l, "CE");
    assert!(!ic.branch, "branch flag should be false");
    assert!((ic.b_ij - 1.5460).abs() < 1e-5, "bond length i-j round-trip failed");
    assert!((ic.theta_ijk - 110.2800).abs() < 1e-4, "angle i-j-k round-trip failed");
    assert!((ic.phi_ijkl - 180.0).abs() < 1e-4, "dihedral round-trip failed");
    assert!((ic.theta_jkl - 98.9400).abs() < 1e-4, "angle j-k-l round-trip failed");
    assert!((ic.b_kl - 1.8206).abs() < 1e-5, "bond length k-l round-trip failed");
}

#[test]
fn test_residue_geometry_table_roundtrip_with_zstd() {
    let table = minimal_geometry_table();
    let prost_bytes = table.encode_to_vec();

    // Compress with zstd
    let compressed = zstd::encode_all(prost_bytes.as_slice(), 3)
        .expect("zstd compress of geometry table failed");
    assert!(compressed.len() < prost_bytes.len() + 100, "compressed should not be vastly larger");

    // Decompress
    let decompressed = zstd::decode_all(compressed.as_slice())
        .expect("zstd decompress failed");
    assert_eq!(decompressed, prost_bytes, "zstd round-trip must be byte-identical");

    // Decode protobuf
    let decoded = ResidueGeometryTable::decode(decompressed.as_slice())
        .expect("prost decode after zstd failed");
    assert_eq!(decoded.source, "charmm36");
    assert_eq!(decoded.residues.len(), 1);
}

#[test]
fn test_rotamer_library_with_geometry_fields() {
    // Test that the new geometry_source and geometry_license fields are preserved
    let lib = RotamerLibrary {
        version: 1,
        provenance: "test".to_string(),
        attribution: "test attribution".to_string(),
        data_license: "ODC-BY-1.0".to_string(),
        geometry_mode: GeometryMode::Precomputed as i32,
        residues: vec![],
        geometry_source: "charmm36".to_string(),
        geometry_license: "NOT-OSI: MacKerell lab academic use only".to_string(),
    };

    let bytes = lib.encode_to_vec();
    let decoded = RotamerLibrary::decode(bytes.as_slice())
        .expect("prost decode of RotamerLibrary failed");

    assert_eq!(decoded.geometry_source, "charmm36");
    assert_eq!(decoded.geometry_license, "NOT-OSI: MacKerell lab academic use only");
    assert_eq!(decoded.data_license, "ODC-BY-1.0");
}
