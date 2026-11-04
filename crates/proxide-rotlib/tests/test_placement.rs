#[path = "helpers.rs"]
mod helpers;

use helpers::committed_rotlib_path;
use proxide_geometry::geometry::angles::dihedral_angle_iupac_f64;
use proxide_rotlib::pb::rotlib_v1::{
    Bin, GeometryMode, ResidueEntry, Rotamer, RotamerLibrary as PbLib, Vec3,
};
use proxide_rotlib::{RotamerId, RotamerLibrary, RotlibError};

/// Create a test library with both ALA (1 atom: CB) and ARG (6 atoms: CB,CG,CD,NE,CZ,NH1).
/// Single bin at (phi=0, psi=0). 1 rotamer per bin.
/// CB stored at [1.0, 0.0, 0.0].
fn make_placement_lib() -> RotamerLibrary {
    let pb = PbLib {
        version: 1,
        provenance: "test".to_string(),
        attribution:
            "Contains information from the 2010 Backbone-Dependent Rotamer Library \
            (http://dunbrack.fccc.edu/bbdep2010), made available under the ODC Attribution License."
                .to_string(),
        data_license: "ODC-BY-1.0".to_string(),
        geometry_mode: GeometryMode::Precomputed as i32,
        residues: vec![
            ResidueEntry {
                code: "ALA".to_string(),
                atom_names: vec!["CB".to_string()],
                num_chi: 0,
                phi_centers: vec![0.0],
                psi_centers: vec![0.0],
                default_bin: 0,
                bins: vec![Bin {
                    phi: 0.0,
                    psi: 0.0,
                    freq: 1.0,
                    rotamers: vec![Rotamer {
                        prob: 0.9,
                        chi: vec![],
                        coords: vec![Vec3 {
                            x: 1.0,
                            y: 0.0,
                            z: 0.0,
                        }],
                    }],
                }],
            },
            ResidueEntry {
                code: "ARG".to_string(),
                atom_names: vec![
                    "CB".to_string(),
                    "CG".to_string(),
                    "CD".to_string(),
                    "NE".to_string(),
                    "CZ".to_string(),
                    "NH1".to_string(),
                ],
                num_chi: 0,
                phi_centers: vec![0.0],
                psi_centers: vec![0.0],
                default_bin: 0,
                bins: vec![Bin {
                    phi: 0.0,
                    psi: 0.0,
                    freq: 1.0,
                    rotamers: vec![Rotamer {
                        prob: 0.9,
                        chi: vec![],
                        coords: vec![
                            Vec3 {
                                x: 1.0,
                                y: 0.0,
                                z: 0.0,
                            },
                            Vec3 {
                                x: 2.0,
                                y: 0.0,
                                z: 0.0,
                            },
                            Vec3 {
                                x: 3.0,
                                y: 0.0,
                                z: 0.0,
                            },
                            Vec3 {
                                x: 4.0,
                                y: 0.0,
                                z: 0.0,
                            },
                            Vec3 {
                                x: 5.0,
                                y: 0.0,
                                z: 0.0,
                            },
                            Vec3 {
                                x: 6.0,
                                y: 0.0,
                                z: 0.0,
                            },
                        ],
                    }],
                }],
            },
        ],
        geometry_source: String::new(),
        geometry_license: String::new(),
    };
    RotamerLibrary::from_protobuf(&pb).unwrap()
}

// Backbone coordinates for tests
const N: [f64; 3] = [0.0, 0.0, 0.0];
const CA: [f64; 3] = [1.458, 0.0, 0.0];
const C: [f64; 3] = [1.980, 1.418, 0.0];

#[test]
fn test_place_ala_one_atom() {
    let lib = make_placement_lib();
    let placed = lib
        .place_rotamer("ALA", Some(0.0), Some(0.0), 0, false, N, CA, C)
        .unwrap();
    assert_eq!(placed.atoms.len(), 1);
    assert_eq!(placed.atoms[0].name, "CB");
}

#[test]
fn test_place_arg_atom_count() {
    let lib = make_placement_lib();
    let placed = lib
        .place_rotamer("ARG", Some(0.0), Some(0.0), 0, false, N, CA, C)
        .unwrap();
    assert!(
        placed.atoms.len() >= 5,
        "expected >=5 atoms, got {}",
        placed.atoms.len()
    );
}

#[test]
fn test_place_no_backbone_atoms() {
    let lib = make_placement_lib();
    let placed = lib
        .place_rotamer("ARG", Some(0.0), Some(0.0), 0, false, N, CA, C)
        .unwrap();
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
    let lib = make_placement_lib();
    let placed = lib
        .place_rotamer("ALA", Some(0.0), Some(0.0), 0, false, N, CA, C)
        .unwrap();
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
    let lib = make_placement_lib();
    let result = lib.place_rotamer("ZZZ", Some(0.0), Some(0.0), 0, false, N, CA, C);
    assert!(matches!(
        result,
        Err(RotlibError::UnknownAa(ref aa)) if aa == "ZZZ"
    ));
}

#[test]
fn test_place_oob_rot_index() {
    let lib = make_placement_lib();
    // ALA has only 1 rotamer (rot=0); rot=99 is out of bounds
    let result = lib.place_rotamer("ALA", Some(0.0), Some(0.0), 99, false, N, CA, C);
    assert!(matches!(
        result,
        Err(RotlibError::RotIndexOob(ref aa, 99, 1)) if aa == "ALA"
    ));
}

#[test]
fn test_place_sentinel_uses_default_bin() {
    let lib = make_placement_lib();
    // phi=None, psi=None -> default_bin
    // With only 1 bin, default_bin=0
    let placed = lib
        .place_rotamer("ALA", None, None, 0, false, N, CA, C)
        .unwrap();
    assert_eq!(placed.id.bin_index, 0);
}

#[test]
fn test_place_ala_cb_exact_coords() {
    // Unit-level exact coordinate check: CB stored at [1.0, 0.0, 0.0] in canonical frame.
    // Backbone: N=[0,0,0], CA=[1.458,0,0], C=[1.980,1.418,0].
    // Expected lab-frame CB = [2.458, 0.0, 0.0] (derived analytically).
    let lib = make_placement_lib();
    let placed = lib
        .place_rotamer("ALA", Some(0.0), Some(0.0), 0, false, N, CA, C)
        .unwrap();
    let cb_xyz = placed.atoms[0].xyz;
    let expected = [2.458, 0.0, 0.0];
    for i in 0..3 {
        assert!(
            (cb_xyz[i] - expected[i]).abs() < 1e-9,
            "CB coord[{i}]: expected {:.6}, got {:.6}",
            expected[i],
            cb_xyz[i]
        );
    }
}

/// All 20 standard amino acids: atom count, finite coords, CB distance from CA.
#[test]
fn test_place_all_aa_smoke() {
    const ALL_AA: &[&str] = &[
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET",
        "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    ];
    let ccd_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../data/rotlibs/proxide-rotlib-dunbrack2010-ccd.pb.zst");
    let rotlib_path = if ccd_path.exists() {
        ccd_path
    } else {
        committed_rotlib_path()
    };
    if !rotlib_path.exists() {
        eprintln!(
            "Skipping: rotlib not available at {}",
            rotlib_path.display()
        );
        return;
    }
    let lib = RotamerLibrary::load_pb_zst(&rotlib_path).unwrap();
    for &aa in ALL_AA {
        if !lib.contains_aa(aa) {
            continue;
        }
        let placed = lib
            .place_rotamer(aa, None, None, 0, false, N, CA, C)
            .unwrap_or_else(|e| panic!("{aa}: place_rotamer failed: {e}"));
        let expected_na = lib.sidechain_atom_names(aa).unwrap().len();
        if aa == "GLY" {
            assert_eq!(placed.atoms.len(), 0, "GLY should have 0 sidechain atoms");
        } else {
            assert_eq!(placed.atoms.len(), expected_na, "{aa}: atom count mismatch");
        }
        for atom in &placed.atoms {
            for &v in &atom.xyz {
                assert!(v.is_finite(), "{aa}/{}: non-finite coord", atom.name);
            }
        }
        // All non-GLY have a CB; check bond distance to CA is chemically plausible.
        if let Some(cb) = placed.atoms.iter().find(|a| a.name == "CB") {
            let d = ((cb.xyz[0] - CA[0]).powi(2)
                + (cb.xyz[1] - CA[1]).powi(2)
                + (cb.xyz[2] - CA[2]).powi(2))
            .sqrt();
            assert!(
                d > 1.2 && d < 1.8,
                "{aa}: CB-CA dist {d:.3} Å out of [1.2, 1.8]"
            );
        }
    }
}

/// Analytical verification of the Gram-Schmidt orthonormal coordinate frame transformation:
/// For arbitrary backbone geometry N, CA, C, verify that:
/// P_lab = CA + x * u_x + y * u_y + z * u_z
/// where:
/// u_x = (CA - N) / ||CA - N||
/// u_z = ((CA - N) x (C - CA)) / ||...||
/// u_y = u_z x u_x
#[test]
fn test_place_analytical_transformation() {
    // Arbitrary non-aligned backbone coordinates
    let n = [1.2, 3.4, 5.6];
    let ca = [2.5, 4.1, 6.8];
    let c = [3.1, 5.2, 6.0];

    // Compute Gram-Schmidt basis analytically from first principles:
    let sub = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let cross = |a: [f64; 3], b: [f64; 3]| -> [f64; 3] {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let norm = |a: [f64; 3]| dot(a, a).sqrt();
    let normalize = |a: [f64; 3]| {
        let l = norm(a);
        [a[0] / l, a[1] / l, a[2] / l]
    };

    let vx = sub(ca, n);
    let ux = normalize(vx);
    let vc = sub(c, ca);
    let vz = cross(vx, vc);
    let uz = normalize(vz);
    let uy = cross(uz, ux);

    let lib = make_placement_lib();
    let placed_arg = lib
        .place_rotamer("ARG", Some(0.0), Some(0.0), 0, false, n, ca, c)
        .unwrap();

    // Canonical coords in test ARG are [1..6, 0, 0]
    for (i, atom) in placed_arg.atoms.iter().enumerate() {
        let local_x = (i + 1) as f64;
        let local_y = 0.0;
        let local_z = 0.0;

        let expected_lab = [
            ca[0] + local_x * ux[0] + local_y * uy[0] + local_z * uz[0],
            ca[1] + local_x * ux[1] + local_y * uy[1] + local_z * uz[1],
            ca[2] + local_x * ux[2] + local_y * uy[2] + local_z * uz[2],
        ];

        for (k, (&placed_coord, &expected_coord)) in atom.xyz.iter().zip(&expected_lab).enumerate()
        {
            let diff = (placed_coord - expected_coord).abs();
            assert!(
                diff < 1e-9,
                "atom {} axis {k}: placed {:.9}, expected {:.9}, diff {:.2e}",
                atom.name,
                placed_coord,
                expected_coord,
                diff
            );
        }
    }
}

/// Place THR A:2 from 1CRN.pdb using real φ/ψ computed from adjacent backbone atoms.
/// Exercises the backbone_bin grid lookup path with non-sentinel angles.
#[test]
fn test_place_met_real_phi_psi_small_pdb() {
    let pdb = helpers::real_pdb_path();
    let rotlib = committed_rotlib_path();

    let residues = helpers::parse_pdb_backbone(&pdb);
    let find_res = |chain: char, seq: i32| -> &helpers::BackboneResidue {
        residues
            .iter()
            .find(|r| r.chain == chain && r.res_seq == seq)
            .unwrap_or_else(|| panic!("residue {chain}:{seq} not found in 1CRN.pdb"))
    };
    let res1 = find_res('A', 1);
    let res2 = find_res('A', 2);
    let res3 = find_res('A', 3);

    // place_rotamer bins by IUPAC phi/psi (debt #3230: this test's private helper
    // returned -IUPAC, the #3229 bug class; its assertions are sign-agnostic).
    let phi = dihedral_angle_iupac_f64(&res1.c, &res2.n, &res2.ca, &res2.c).to_degrees();
    let psi = dihedral_angle_iupac_f64(&res2.n, &res2.ca, &res2.c, &res3.n).to_degrees();

    // Sanity-check: angles are real backbone-derived values.
    assert!(phi.abs() < 500.0, "phi looks invalid: {phi}");
    assert!(psi.abs() < 500.0, "psi looks invalid: {psi}");

    let lib = RotamerLibrary::load_pb_zst(&rotlib).unwrap();

    let placed = lib
        .place_rotamer(
            &res2.aa,
            Some(phi),
            Some(psi),
            0,
            false,
            res2.n,
            res2.ca,
            res2.c,
        )
        .unwrap();

    // Real φ/ψ must route to a valid bin and record used_default_bin = false
    let sentinel = lib
        .place_rotamer(&res2.aa, None, None, 0, false, res2.n, res2.ca, res2.c)
        .unwrap();
    assert!(!placed.used_default_bin);
    assert!(sentinel.used_default_bin);
    assert_ne!(
        placed.id.bin_index, sentinel.id.bin_index,
        "real φ/ψ ({phi:.1}°, {psi:.1}°) must select a different bin than default — \
         backbone_bin grid lookup path is not being exercised"
    );

    // Chemical plausibility: CB within bonded distance from CA, all coords finite.
    let cb = placed
        .atoms
        .iter()
        .find(|a| a.name == "CB")
        .expect("THR should have a CB atom");
    let ca = res2.ca;
    let d =
        ((cb.xyz[0] - ca[0]).powi(2) + (cb.xyz[1] - ca[1]).powi(2) + (cb.xyz[2] - ca[2]).powi(2))
            .sqrt();
    assert!(
        d > 1.2 && d < 1.8,
        "CB-CA dist {d:.3} Å out of chemically plausible range [1.2, 1.8]"
    );
    for atom in &placed.atoms {
        for &v in &atom.xyz {
            assert!(
                v.is_finite(),
                "MET/{}: non-finite coord in real-angle placement",
                atom.name
            );
        }
    }
}
