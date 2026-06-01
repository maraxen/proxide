#[path = "helpers.rs"]
mod helpers;

use helpers::{real_pdb_path, real_rotlib_path, parse_pdb_backbone};
use proxide_rotlib::RotamerLibrary;

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
