#[path = "helpers.rs"]
mod helpers;

use helpers::{committed_rotlib_path, parse_pdb_backbone, real_pdb_path, real_pdb_path_2zta};
use proxide_rotlib::RotamerLibrary;

fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Verify mathematical and physical metric space invariants on a distance matrix:
/// 1. Self-distance zero: D[i, i] == 0
/// 2. Non-negativity: D[i, j] >= 0
/// 3. Strict positivity for distinct points: D[i, j] > 0 for i != j
/// 4. Symmetry: D[i, j] == D[j, i]
/// 5. Triangle inequality: D[i, k] <= D[i, j] + D[j, k]
/// 6. Physical peptide chain continuity: adjacent CB-CB distance in [3.0, 8.0] Å.
fn verify_distance_matrix_invariants(dists: &[Vec<f64>]) {
    let n = dists.len();
    for i in 0..n {
        assert_eq!(dists[i].len(), n, "matrix must be square");
        assert!(
            dists[i][i].abs() < 1e-9,
            "self-distance D[{i},{i}] must be 0, got {}",
            dists[i][i]
        );
        for j in 0..n {
            let dij = dists[i][j];
            let dji = dists[j][i];
            assert!(
                dij >= 0.0,
                "distance D[{i},{j}] must be non-negative, got {dij}"
            );
            assert!(
                (dij - dji).abs() < 1e-9,
                "symmetry failed: D[{i},{j}]={dij} != D[{j},{i}]={dji}"
            );
            if i != j {
                assert!(
                    dij > 0.1,
                    "distinct residues {i} and {j} must have non-zero distance, got {dij}"
                );
            }
            // Triangle inequality: D[i, k] <= D[i, j] + D[j, k]
            for (k, &dik) in dists[i].iter().enumerate().take(n) {
                let djk = dists[j][k];
                assert!(
                    dik <= dij + djk + 1e-6,
                    "triangle inequality failed: D[{i},{k}]={dik} > D[{i},{j}]={dij} + D[{j},{k}]={djk}"
                );
            }
        }
        // Consecutive adjacent residues (i, i+1) along peptide chain
        if i + 1 < n {
            let d_adj = dists[i][i + 1];
            assert!(
                d_adj > 3.0 && d_adj < 8.0,
                "adjacent residue CB-CB distance out of plausible range [3.0, 8.0]: {d_adj}"
            );
        }
    }
}

/// Helper to place CB atoms and compute distance matrix for a list of backbone residues.
fn compute_cb_distogram(
    lib: &RotamerLibrary,
    residues: &[&helpers::BackboneResidue],
) -> Vec<Vec<f64>> {
    let cb_positions: Vec<[f64; 3]> = residues
        .iter()
        .map(|res| {
            let placed = lib
                .place_rotamer(&res.aa, None, None, 0, false, res.n, res.ca, res.c)
                .unwrap_or_else(|e| panic!("{}/{}: place_rotamer failed: {e}", res.chain, res.aa));
            placed
                .atoms
                .iter()
                .find(|a| a.name == "CB")
                .unwrap_or_else(|| panic!("{}/{}: no CB atom", res.chain, res.aa))
                .xyz
        })
        .collect();

    let n = cb_positions.len();
    let mut matrix = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            matrix[i][j] = dist(cb_positions[i], cb_positions[j]);
        }
    }
    matrix
}

/// Invariant test on segment 1 of 1CRN.pdb (7 contiguous residues 1..=7).
#[test]
fn test_distogram_invariants_chain_a_small_pdb() {
    let rotlib_path = committed_rotlib_path();
    let pdb_path = real_pdb_path();
    let lib = RotamerLibrary::load_pb_zst(&rotlib_path).unwrap();

    let all_residues = parse_pdb_backbone(&pdb_path);
    let chain_a: Vec<_> = all_residues
        .iter()
        .filter(|r| r.chain == 'A' && r.aa != "GLY")
        .take(7)
        .collect();
    assert_eq!(
        chain_a.len(),
        7,
        "expected 7 chain-A residues in 1CRN segment 1"
    );

    let distogram = compute_cb_distogram(&lib, &chain_a);
    verify_distance_matrix_invariants(&distogram);
}

/// Invariant test on segment 2 of 1CRN.pdb (7 contiguous residues 10..=16).
#[test]
fn test_distogram_invariants_chain_b_small_pdb() {
    let rotlib_path = committed_rotlib_path();
    let pdb_path = real_pdb_path();
    let lib = RotamerLibrary::load_pb_zst(&rotlib_path).unwrap();

    let all_residues = parse_pdb_backbone(&pdb_path);
    let chain_b: Vec<_> = all_residues
        .iter()
        .filter(|r| r.chain == 'A' && (10..=16).contains(&r.res_seq))
        .collect();
    assert_eq!(
        chain_b.len(),
        7,
        "expected 7 residues in 1CRN segment 2 (10..=16)"
    );

    let distogram = compute_cb_distogram(&lib, &chain_b);
    verify_distance_matrix_invariants(&distogram);
}

/// Invariant test on ubiquitin (1UBQ chain A, contiguous non-GLY segment).
#[test]
fn test_distogram_invariants_2zta_chain_a() {
    let rotlib_path = committed_rotlib_path();
    let pdb_path = real_pdb_path_2zta();
    let lib = RotamerLibrary::load_pb_zst(&rotlib_path).unwrap();

    let all_residues = parse_pdb_backbone(&pdb_path);
    // Take first 9 contiguous non-GLY residues of 1UBQ (1..=9)
    let chain_a: Vec<_> = all_residues
        .iter()
        .filter(|r| r.chain == 'A' && r.aa != "GLY")
        .take(9)
        .collect();
    assert_eq!(
        chain_a.len(),
        9,
        "expected 9 contiguous non-GLY residues in 1UBQ"
    );

    let distogram = compute_cb_distogram(&lib, &chain_a);
    verify_distance_matrix_invariants(&distogram);
}

/// SE(3) Invariance: Rigid rotation and translation of the backbone must yield
/// an identical distance matrix within 1e-6 Å.
#[test]
fn test_distogram_se3_invariance() {
    let rotlib_path = committed_rotlib_path();
    let pdb_path = real_pdb_path();
    let lib = RotamerLibrary::load_pb_zst(&rotlib_path).unwrap();

    let all_residues = parse_pdb_backbone(&pdb_path);
    let chain_a: Vec<_> = all_residues
        .iter()
        .filter(|r| r.chain == 'A' && r.aa != "GLY")
        .take(7)
        .collect();

    // Baseline distance matrix
    let d_base = compute_cb_distogram(&lib, &chain_a);

    // Apply arbitrary 3D rotation (angle = 45 deg around axis [1, 2, 3]) and translation [10, -20, 30]
    let theta: f64 = std::f64::consts::FRAC_PI_4;
    let axis = [
        1.0 / (14.0_f64).sqrt(),
        2.0 / (14.0_f64).sqrt(),
        3.0 / (14.0_f64).sqrt(),
    ];
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    let omc = 1.0 - cos_t;

    let r = [
        [
            cos_t + axis[0] * axis[0] * omc,
            axis[0] * axis[1] * omc - axis[2] * sin_t,
            axis[0] * axis[2] * omc + axis[1] * sin_t,
        ],
        [
            axis[1] * axis[0] * omc + axis[2] * sin_t,
            cos_t + axis[1] * axis[1] * omc,
            axis[1] * axis[2] * omc - axis[0] * sin_t,
        ],
        [
            axis[2] * axis[0] * omc - axis[1] * sin_t,
            axis[2] * axis[1] * omc + axis[0] * sin_t,
            cos_t + axis[2] * axis[2] * omc,
        ],
    ];
    let t = [10.5, -20.2, 30.8];

    let xform = |p: [f64; 3]| {
        [
            r[0][0] * p[0] + r[0][1] * p[1] + r[0][2] * p[2] + t[0],
            r[1][0] * p[0] + r[1][1] * p[1] + r[1][2] * p[2] + t[1],
            r[2][0] * p[0] + r[2][1] * p[1] + r[2][2] * p[2] + t[2],
        ]
    };

    let transformed_chain_a: Vec<helpers::BackboneResidue> = chain_a
        .iter()
        .map(|res| helpers::BackboneResidue {
            chain: res.chain,
            res_seq: res.res_seq,
            i_code: res.i_code,
            aa: res.aa.clone(),
            n: xform(res.n),
            ca: xform(res.ca),
            c: xform(res.c),
        })
        .collect();

    let transformed_refs: Vec<&helpers::BackboneResidue> = transformed_chain_a.iter().collect();
    let d_xform = compute_cb_distogram(&lib, &transformed_refs);

    let n = chain_a.len();
    for i in 0..n {
        for j in 0..n {
            let diff = (d_base[i][j] - d_xform[i][j]).abs();
            assert!(
                diff < 1e-6,
                "SE(3) invariance violated at ({i}, {j}): base={:.6}, xform={:.6}, diff={:.2e}",
                d_base[i][j],
                d_xform[i][j],
                diff
            );
        }
    }
}
