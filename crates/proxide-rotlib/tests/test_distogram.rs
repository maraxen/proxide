#[path = "helpers.rs"]
mod helpers;

use helpers::{real_pdb_path, real_rotlib_path, parse_pdb_backbone};
use proxide_rotlib::RotamerLibrary;
use std::path::PathBuf;

/// Reference CB-CB distance matrix for chain A of small.pdb (7 residues: ARG MET LYS GLN LEU GLU ASP).
/// Computed by applying the same frame transform as proxide to backbone extracted from the PDB,
/// using rotlib.bin rot_index=0 default-bin canonical coords. Row-major, 7×7 = 49 values.
const REF_DISTOGRAM: [f64; 49] = [
    0.000000000, 5.350014872, 5.221347349, 4.695044671, 6.797364701, 9.073599913, 9.562848625,
    5.350014872, 0.000000000, 4.981813638, 7.287197023, 5.786180088, 6.069736770, 9.358732650,
    5.221347349, 4.981813638, 0.000000000, 5.142041501, 7.114062169, 5.751336019, 6.338944406,
    4.695044671, 7.287197023, 5.142041501, 0.000000000, 5.249585670, 7.296483494, 5.695136389,
    6.797364701, 5.786180088, 7.114062169, 5.249585670, 0.000000000, 5.354200760, 7.019106952,
    9.073599913, 6.069736770, 5.751336019, 7.296483494, 5.354200760, 0.000000000, 5.084358374,
    9.562848625, 9.358732650, 6.338944406, 5.695136389, 7.019106952, 5.084358374, 0.000000000,
];

/// CB position tolerance: reference is f64 arithmetic on f32-sourced coords; 1e-6 Å is safe.
const TOL: f64 = 1e-6;

fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0]-b[0]).powi(2) + (a[1]-b[1]).powi(2) + (a[2]-b[2]).powi(2)).sqrt()
}

/// Parity test: place rot_index=0 default-bin CB onto every chain-A residue backbone from
/// small.pdb, compute the 7×7 CB-CB distance matrix, and assert it matches the Mosaist-
/// derived reference within 1e-6 Å.
///
/// This catches regressions in the frame construction (backbone_frame), the switch_frames
/// transform, and the binary parser — all exercised on real non-trivial backbone geometry
/// with a diverse set of multi-chi residues (ARG, MET, LYS, GLN, LEU, GLU, ASP).
#[test]
fn test_distogram_chain_a_small_pdb() {
    let lib = RotamerLibrary::load(&real_rotlib_path()).unwrap();

    let all_residues = parse_pdb_backbone(&real_pdb_path());
    let chain_a: Vec<_> = all_residues.iter().filter(|r| r.chain == 'A').collect();
    assert_eq!(chain_a.len(), 7, "expected 7 chain-A residues in small.pdb");

    // Place CB for each residue (rot_index=0, sentinel phi/psi → default bin).
    let cb_positions: Vec<[f64; 3]> = chain_a.iter().map(|res| {
        let placed = lib
            .place_rotamer(&res.aa, 9999.0, 9999.0, 0, res.n, res.ca, res.c)
            .unwrap_or_else(|e| panic!("{}/{}: place_rotamer failed: {e}", res.chain, res.aa));
        placed.atoms.iter()
            .find(|a| a.name == "CB")
            .unwrap_or_else(|| panic!("{}/{}: no CB atom", res.chain, res.aa))
            .xyz
    }).collect();

    // Assert pairwise CB-CB distances match reference.
    let n = chain_a.len();
    for i in 0..n {
        for j in 0..n {
            let got = dist(cb_positions[i], cb_positions[j]);
            let ref_val = REF_DISTOGRAM[i * n + j];
            assert!(
                (got - ref_val).abs() < TOL,
                "distogram[{i},{j}] ({}/{}): expected {ref_val:.9}, got {got:.9}, diff {:.2e}",
                chain_a[i].aa, chain_a[j].aa,
                (got - ref_val).abs()
            );
        }
    }
}

/// Reference CB-CB distance matrix for chain B of small.pdb (7 residues: ARG MET LYS GLN LEU GLU ASP).
/// Same AA sequence as chain A but different backbone geometry. Row-major, 7×7 = 49 values.
const REF_DISTOGRAM_CHAIN_B: [f64; 49] = [
    0.000000000, 5.042123003, 5.810731058, 5.265353624, 6.950706050, 9.359338534, 9.983103353,
    5.042123003, 0.000000000, 4.930241183, 7.331023334, 5.975333623, 6.476195890, 9.643688508,
    5.810731058, 4.930241183, 0.000000000, 5.355330787, 7.183010214, 5.545144578, 6.454032520,
    5.265353624, 7.331023334, 5.355330787, 0.000000000, 5.287528097, 7.175713381, 5.433929545,
    6.950706050, 5.975333623, 7.183010214, 5.287528097, 0.000000000, 5.420716352, 7.025066702,
    9.359338534, 6.476195890, 5.545144578, 7.175713381, 5.420716352, 0.000000000, 5.137950184,
    9.983103353, 9.643688508, 6.454032520, 5.433929545, 7.025066702, 5.137950184, 0.000000000,
];

/// Parity test: place rot_index=0 default-bin CB onto every chain-B residue backbone from
/// small.pdb, compute the 7×7 CB-CB distance matrix, and assert it matches the Mosaist-
/// derived reference within 1e-6 Å.
///
/// Chain B has the same amino acid sequence as chain A but different backbone geometry,
/// so the distances should differ from chain A. This verifies that residue-specific backbone
/// geometry is correctly handled.
#[test]
fn test_distogram_chain_b_small_pdb() {
    let lib = RotamerLibrary::load(&real_rotlib_path()).unwrap();

    let all_residues = parse_pdb_backbone(&real_pdb_path());
    let chain_b: Vec<_> = all_residues.iter().filter(|r| r.chain == 'B').collect();
    assert_eq!(chain_b.len(), 7, "expected 7 chain-B residues in small.pdb");

    // Place CB for each residue (rot_index=0, sentinel phi/psi → default bin).
    let cb_positions: Vec<[f64; 3]> = chain_b.iter().map(|res| {
        let placed = lib
            .place_rotamer(&res.aa, 9999.0, 9999.0, 0, res.n, res.ca, res.c)
            .unwrap_or_else(|e| panic!("{}/{}: place_rotamer failed: {e}", res.chain, res.aa));
        placed.atoms.iter()
            .find(|a| a.name == "CB")
            .unwrap_or_else(|| panic!("{}/{}: no CB atom", res.chain, res.aa))
            .xyz
    }).collect();

    // Assert pairwise CB-CB distances match reference.
    let n = chain_b.len();
    for i in 0..n {
        for j in 0..n {
            let got = dist(cb_positions[i], cb_positions[j]);
            let ref_val = REF_DISTOGRAM_CHAIN_B[i * n + j];
            assert!(
                (got - ref_val).abs() < TOL,
                "distogram[{i},{j}] ({}/{}): expected {ref_val:.9}, got {got:.9}, diff {:.2e}",
                chain_b[i].aa, chain_b[j].aa,
                (got - ref_val).abs()
            );
        }
    }
}

/// Test GLY handling: GLY has na=0 (no sidechain atoms), so place_rotamer should return
/// an empty atom list. This test verifies that placing GLY and PRO (with distinct CD atom
/// in rotlib) are correctly handled on a real PDB.
#[test]
fn test_distogram_with_gly_and_pro() {
    let lib = RotamerLibrary::load(&real_rotlib_path()).unwrap();

    // Use 1DC7.pdb which contains both GLY and PRO in chain A.
    let pdb_path = PathBuf::from("/home/marielle/repos/mosaist/testfiles/1DC7.pdb");
    if !pdb_path.exists() {
        eprintln!("Skipping test: 1DC7.pdb not found at {:?}", pdb_path);
        return;
    }

    let all_residues = parse_pdb_backbone(&pdb_path);
    let chain_a_first_50: Vec<_> = all_residues
        .iter()
        .filter(|r| r.chain == 'A')
        .take(50)
        .collect();

    // Find GLY and PRO residues.
    let gly_residues: Vec<_> = chain_a_first_50
        .iter()
        .filter(|r| r.aa == "GLY")
        .collect();
    let pro_residues: Vec<_> = chain_a_first_50
        .iter()
        .filter(|r| r.aa == "PRO")
        .collect();

    assert!(!gly_residues.is_empty(), "expected GLY residues in 1DC7 chain A first 50");
    assert!(!pro_residues.is_empty(), "expected PRO residues in 1DC7 chain A first 50");

    // Test GLY: placing should return empty atoms.
    for res in &gly_residues {
        let placed = lib
            .place_rotamer(&res.aa, 9999.0, 9999.0, 0, res.n, res.ca, res.c)
            .unwrap_or_else(|e| panic!("GLY place_rotamer failed: {e}"));
        assert!(
            placed.atoms.is_empty(),
            "GLY should have no sidechain atoms; got {} atoms",
            placed.atoms.len()
        );
    }

    // Test PRO: first atom should be "CD" (not "CB").
    for res in &pro_residues {
        let placed = lib
            .place_rotamer(&res.aa, 9999.0, 9999.0, 0, res.n, res.ca, res.c)
            .unwrap_or_else(|e| panic!("PRO place_rotamer failed: {e}"));
        assert!(
            !placed.atoms.is_empty(),
            "PRO should have sidechain atoms"
        );
        let first_atom = &placed.atoms[0];
        assert_eq!(
            first_atom.name, "CD",
            "PRO first atom should be 'CD', got '{}'",
            first_atom.name
        );
    }
}
