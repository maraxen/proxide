use prost::Message;
use proxide_rotlib::pb::rotlib_v1::{
    RotamerLibrary, ResidueEntry, Bin, Rotamer, ChiValue, Vec3, GeometryMode,
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
