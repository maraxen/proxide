#[path = "helpers.rs"]
mod helpers;

use helpers::{real_rotlib_path, write_minimal_lib, BinSpec, RotSpec};
use proxide_rotlib::{RotamerId, RotamerLibrary, RotlibError};
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

fn dihedral_angle(p1: [f64; 3], p2: [f64; 3], p3: [f64; 3], p4: [f64; 3]) -> f64 {
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
    let b1 = sub(p2, p1);
    let b2 = sub(p3, p2);
    let b3 = sub(p4, p3);
    let n1 = cross(b1, b2);
    let n2 = cross(b2, b3);
    let b2n = {
        let s = 1.0 / norm(b2);
        [b2[0] * s, b2[1] * s, b2[2] * s]
    };
    let m1 = cross(n1, b2n);
    f64::atan2(dot(m1, n2), dot(n1, n2))
}

#[test]
fn test_place_ala_one_atom() {
    let (_lib_tmp, lib) = make_placement_lib();
    let placed = lib
        .place_rotamer("ALA", 0.0, 0.0, 0, false, N, CA, C)
        .unwrap();
    assert_eq!(placed.atoms.len(), 1);
    assert_eq!(placed.atoms[0].name, "CB");
}

#[test]
fn test_place_arg_atom_count() {
    let (_lib_tmp, lib) = make_placement_lib();
    let placed = lib
        .place_rotamer("ARG", 0.0, 0.0, 0, false, N, CA, C)
        .unwrap();
    assert!(
        placed.atoms.len() >= 5,
        "expected >=5 atoms, got {}",
        placed.atoms.len()
    );
}

#[test]
fn test_place_no_backbone_atoms() {
    let (_lib_tmp, lib) = make_placement_lib();
    let placed = lib
        .place_rotamer("ARG", 0.0, 0.0, 0, false, N, CA, C)
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
    let (_lib_tmp, lib) = make_placement_lib();
    let placed = lib
        .place_rotamer("ALA", 0.0, 0.0, 0, false, N, CA, C)
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
    let (_lib_tmp, lib) = make_placement_lib();
    let result = lib.place_rotamer("ZZZ", 0.0, 0.0, 0, false, N, CA, C);
    assert!(matches!(
        result,
        Err(RotlibError::UnknownAa(ref aa)) if aa == "ZZZ"
    ));
}

#[test]
fn test_place_oob_rot_index() {
    let (_lib_tmp, lib) = make_placement_lib();
    // ALA has only 1 rotamer (rot=0); rot=99 is out of bounds
    let result = lib.place_rotamer("ALA", 0.0, 0.0, 99, false, N, CA, C);
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
        .place_rotamer("ALA", 9999.0, 9999.0, 0, false, N, CA, C)
        .unwrap();
    assert_eq!(placed.id.bin_index, 0);
}

#[test]
fn test_place_ala_cb_exact_coords() {
    // Unit-level exact coordinate check: CB stored at [1.0, 0.0, 0.0] in canonical frame.
    // Backbone: N=[0,0,0], CA=[1.458,0,0], C=[1.980,1.418,0].
    // Expected lab-frame CB = [2.458, 0.0, 0.0] (derived analytically).
    let (_lib_tmp, lib) = make_placement_lib();
    let placed = lib
        .place_rotamer("ALA", 0.0, 0.0, 0, false, N, CA, C)
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
#[ignore] // Run only when Mosaist is available
fn test_place_all_aa_smoke() {
    const ALL_AA: &[&str] = &[
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET",
        "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    ];
    let rotlib_path = real_rotlib_path();
    if !rotlib_path.exists() {
        eprintln!(
            "Skipping: ROTLIB_PATH not available at {}",
            rotlib_path.display()
        );
        return;
    }
    let lib = RotamerLibrary::load(&rotlib_path).unwrap();
    for &aa in ALL_AA {
        let placed = lib
            .place_rotamer(aa, 9999.0, 9999.0, 0, false, N, CA, C)
            .unwrap_or_else(|e| panic!("{aa}: place_rotamer failed: {e}"));
        let expected_na = lib.sidechain_atom_names(aa).unwrap().len();
        assert_eq!(placed.atoms.len(), expected_na, "{aa}: atom count mismatch");
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

#[test]
#[ignore] // Run only when Mosaist is available
fn test_place_parity_mosaist() {
    // Reference CB in lab frame derived from rotlib.bin canonical coords
    // (f32 → f64: [0.519, -0.670, -1.262]) placed onto the test backbone.
    // For N=[0,0,0], CA=[1.458,0,0], C=[1.980,1.418,0] the residue frame
    // axes align with lab axes, so lab_CB = CA + canonical_CB.
    // Verified independently by reading rotlib.bin directly in C++ using the
    // same parsing path as Mosaist (mstrotlib.cpp placeRotamer lines 182–194).
    const REF_CB: [f64; 3] = [
        1.976_999_993_801_117,
        -0.670_000_016_689_300_5,
        -1.261_999_964_714_050_3,
    ];

    let rotlib_path = real_rotlib_path();
    if !rotlib_path.exists() {
        eprintln!(
            "Skipping: ROTLIB_PATH not available at {}",
            rotlib_path.display()
        );
        return;
    }
    let lib = RotamerLibrary::load(&rotlib_path).unwrap();
    let placed = lib
        .place_rotamer("ALA", 9999.0, 9999.0, 0, false, N, CA, C)
        .unwrap();

    assert_eq!(placed.atoms.len(), 1, "ALA should have 1 atom (CB)");
    let xyz = placed.atoms[0].xyz;

    for i in 0..3 {
        assert!(
            (xyz[i] - REF_CB[i]).abs() < 1e-5,
            "CB coord[{i}]: expected {:.9}, got {:.9}, diff {:.2e}",
            REF_CB[i],
            xyz[i],
            (xyz[i] - REF_CB[i]).abs()
        );
    }
}

/// Place MET A:2 from small.pdb using real φ/ψ computed from adjacent backbone atoms.
/// Exercises the backbone_bin grid lookup path with non-sentinel angles.
/// φ/ψ are negated per Mosaist sign convention (coords.rs fill_dihedrals).
#[test]
#[ignore]
fn test_place_met_real_phi_psi_small_pdb() {
    let pdb = helpers::real_pdb_path();
    if !pdb.exists() {
        return;
    }
    let rotlib = real_rotlib_path();
    if !rotlib.exists() {
        return;
    }

    let residues = helpers::parse_pdb_backbone(&pdb);
    let find_res = |chain: char, seq: i32| -> &helpers::BackboneResidue {
        residues
            .iter()
            .find(|r| r.chain == chain && r.res_seq == seq)
            .unwrap_or_else(|| panic!("residue {chain}:{seq} not found in small.pdb"))
    };
    let arg1 = find_res('A', 1);
    let met2 = find_res('A', 2);
    let lys3 = find_res('A', 3);

    // Mosaist sign convention: negate the raw dihedral angle.
    let phi = -dihedral_angle(arg1.c, met2.n, met2.ca, met2.c).to_degrees();
    let psi = -dihedral_angle(met2.n, met2.ca, met2.c, lys3.n).to_degrees();

    // Sanity-check: angles are real (non-sentinel) backbone-derived values.
    assert!(phi.abs() < 500.0, "phi looks like a sentinel: {phi}");
    assert!(psi.abs() < 500.0, "psi looks like a sentinel: {psi}");

    let lib = RotamerLibrary::load(&rotlib).unwrap();

    let placed = lib
        .place_rotamer("MET", phi, psi, 0, false, met2.n, met2.ca, met2.c)
        .unwrap();

    // Real φ/ψ must route to a different bin than sentinel (exercises grid lookup).
    let sentinel = lib
        .place_rotamer("MET", 9999.0, 9999.0, 0, false, met2.n, met2.ca, met2.c)
        .unwrap();
    assert_ne!(
        placed.id.bin_index, sentinel.id.bin_index,
        "real φ/ψ ({phi:.1}°, {psi:.1}°) must select a different bin than sentinel — \
         backbone_bin grid lookup path is not being exercised"
    );

    // Chemical plausibility: CB within bonded distance from CA, all coords finite.
    let cb = placed
        .atoms
        .iter()
        .find(|a| a.name == "CB")
        .expect("MET should have a CB atom");
    let ca = met2.ca;
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
