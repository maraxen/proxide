//! Distance calculations for protein structures
//!
//! Note: These utilities will be exposed to Python in a future phase.

#![allow(dead_code)]

/// Compute the Euclidean distance between two 3D points
#[inline]
pub fn euclidean_distance(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Compute squared Euclidean distance (faster when only comparing distances)
#[inline]
pub fn euclidean_distance_squared(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

/// Compute pairwise distance matrix for a set of points
/// Returns a flattened upper triangular matrix
pub fn pairwise_distances(coords: &[[f32; 3]]) -> Vec<f32> {
    let n = coords.len();
    // `n.saturating_sub(1) * n`, not `n * (n - 1)`: the latter underflows
    // (usize) for `n == 0`, panicking on an empty input in debug builds.
    let mut distances = Vec::with_capacity(n.saturating_sub(1) * n / 2);

    for i in 0..n {
        for j in (i + 1)..n {
            distances.push(euclidean_distance(&coords[i], &coords[j]));
        }
    }

    distances
}

/// Compute CA-CA distance matrix for backbone analysis
pub fn ca_distance_matrix(ca_coords: &[[f32; 3]]) -> Vec<Vec<f32>> {
    let n = ca_coords.len();
    let mut matrix = vec![vec![0.0f32; n]; n];

    for i in 0..n {
        for j in (i + 1)..n {
            let dist = euclidean_distance(&ca_coords[i], &ca_coords[j]);
            matrix[i][j] = dist;
            matrix[j][i] = dist;
        }
    }

    matrix
}

/// Orthorhombic (axis-aligned) periodic box dimensions, in whatever length
/// unit the caller's positions use — this type carries no unit tag itself,
/// same convention as the rest of this module.
///
/// Only the three axis lengths are represented: this is deliberately not a
/// general triclinic cell. Every proxide trajectory format currently wired
/// up to [`pairwise_distances_mic`] (XTC) has confirmed-orthorhombic boxes,
/// and building only the axis-aligned case keeps the wrapping arithmetic
/// unambiguous. A triclinic box would need shear (off-diagonal) terms folded
/// into the minimum-image search, which this type does not attempt.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoxDims {
    /// Per-axis box lengths `[Lx, Ly, Lz]`.
    pub lengths: [f32; 3],
}

impl BoxDims {
    /// Construct directly from per-axis lengths.
    pub fn new(lengths: [f32; 3]) -> Self {
        Self { lengths }
    }

    /// Build from a full 3x3 box-vector matrix (row- or column-major —
    /// irrelevant here, since only the diagonal is read), taking the
    /// diagonal entries as the per-axis lengths and ignoring any
    /// off-diagonal (shear) terms. Correct for an orthorhombic box; silently
    /// drops shear information for a triclinic one, so only use this on a
    /// box already known to be orthorhombic (see the struct-level doc).
    pub fn from_diagonal_matrix(box_vectors: &[[f32; 3]; 3]) -> Self {
        Self {
            lengths: [box_vectors[0][0], box_vectors[1][1], box_vectors[2][2]],
        }
    }

    /// Whether this box represents a real periodic cell worth wrapping
    /// against — every axis length finite and strictly positive.
    ///
    /// A degenerate box (any axis zero, negative, `NaN`, or infinite) has no
    /// physical periodic image to wrap into. This is the exact case an
    /// implicit-solvent or in-vacuo trajectory (or any XTC frame with an
    /// absent/zero-filled box record) produces, and
    /// [`pairwise_distances_mic`] uses this to fall back to plain Euclidean
    /// distance automatically rather than dividing by zero.
    pub fn is_periodic(&self) -> bool {
        self.lengths.iter().all(|&l| l.is_finite() && l > 0.0)
    }
}

/// Compute the upper-triangle (`i < j`) pairwise distances between
/// `positions`, using the minimum-image convention (MIC) against `box_dims`
/// when it represents a real periodic cell ([`BoxDims::is_periodic`]), or
/// plain Euclidean distance otherwise — transparently, with no need for the
/// caller to know in advance which case applies. Passing `box_dims: None`
/// (no box at all) always takes the plain-Euclidean path.
///
/// This is a system-agnostic primitive: `positions` is just a flat ordered
/// list of points — Cα atoms, arbitrary markers, whatever the caller
/// selected — with no assumption about chain count, residue count, or what
/// the points represent. Pair order matches `numpy.triu_indices(n, k=1)`:
/// row-major iteration over the upper triangle (`i = 0..n`, and for each
/// `i`, `j = i+1..n`), which is exactly the nested-loop order below.
///
/// MIC wrapping is per-axis and independent (see [`BoxDims`]): for each
/// Cartesian component `k` of the raw difference vector `d`,
/// `d[k] -= box_dims.lengths[k] * round(d[k] / box_dims.lengths[k])`. This
/// is the standard axis-aligned minimum-image formula (as used throughout
/// MDAnalysis/GROMACS-style tooling for orthorhombic cells): it wraps each
/// component into `(-L/2, L/2]`, which is the correct minimum-image
/// separation for an axis-aligned box without needing any bond graph or
/// molecule-unwrapping step first.
pub fn pairwise_distances_mic(positions: &[[f32; 3]], box_dims: Option<&BoxDims>) -> Vec<f32> {
    let n = positions.len();
    let n_pairs = n.saturating_sub(1) * n / 2;
    let mut out = Vec::with_capacity(n_pairs);
    let periodic_box = box_dims.filter(|b| b.is_periodic());

    for i in 0..n {
        let pi = positions[i];
        for pj in positions.iter().take(n).skip(i + 1) {
            let mut d = [pj[0] - pi[0], pj[1] - pi[1], pj[2] - pi[2]];
            if let Some(b) = periodic_box {
                for (dk, &lk) in d.iter_mut().zip(b.lengths.iter()) {
                    *dk -= lk * (*dk / lk).round();
                }
            }
            out.push((d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt());
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euclidean_distance() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        assert!((euclidean_distance(&a, &b) - 1.0).abs() < 1e-6);

        let c = [1.0, 1.0, 1.0];
        let d = [2.0, 2.0, 2.0];
        let expected = 3.0f32.sqrt();
        assert!((euclidean_distance(&c, &d) - expected).abs() < 1e-6);
    }

    #[test]
    fn test_pairwise_distances() {
        let coords = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];

        let dists = pairwise_distances(&coords);
        assert_eq!(dists.len(), 3); // 3 choose 2 = 3
        assert!((dists[0] - 1.0).abs() < 1e-6); // (0,1)
        assert!((dists[1] - 1.0).abs() < 1e-6); // (0,2)
        assert!((dists[2] - 2.0f32.sqrt()).abs() < 1e-6); // (1,2)
    }

    #[test]
    fn test_pairwise_distances_mic_no_box_matches_plain_pairwise_distances() {
        // With no periodic box (None, or an all-zero/degenerate box), the
        // MIC primitive must degrade to exactly the existing non-periodic
        // `pairwise_distances` — this doubles as a generality check: it
        // holds for any atom count/shape, not just one specific topology.
        let systems: Vec<Vec<[f32; 3]>> = vec![
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            vec![[0.0, 0.0, 0.0]], // single atom, zero pairs
            vec![],                // zero atoms
            vec![
                [0.0, 0.0, 0.0],
                [2.5, 0.0, 0.0],
                [0.0, 3.5, 0.0],
                [1.0, 1.0, 1.0],
                [-2.0, 5.0, 0.5],
                [10.0, -3.0, 2.0],
                [0.1, 0.2, 0.3],
            ],
        ];
        for coords in systems {
            let expected = pairwise_distances(&coords);
            let got_no_box = pairwise_distances_mic(&coords, None);
            let zero_box = BoxDims::new([0.0, 0.0, 0.0]);
            let got_zero_box = pairwise_distances_mic(&coords, Some(&zero_box));
            assert_eq!(got_no_box.len(), expected.len());
            assert_eq!(got_zero_box.len(), expected.len());
            for (a, b) in got_no_box.iter().zip(expected.iter()) {
                assert!((a - b).abs() < 1e-5);
            }
            for (a, b) in got_zero_box.iter().zip(expected.iter()) {
                assert!((a - b).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn test_pairwise_distances_mic_degenerate_box_variants_fall_back() {
        // Zero, negative, NaN, and infinite axis lengths must all be treated
        // as "no real periodic cell" and fall back to plain Euclidean
        // distance — not division-by-zero, not NaN output, not a panic.
        let coords = [[0.0, 0.0, 0.0], [9.9, 0.0, 0.0]];
        let variants = [
            BoxDims::new([0.0, 10.0, 10.0]),
            BoxDims::new([-10.0, 10.0, 10.0]),
            BoxDims::new([f32::NAN, 10.0, 10.0]),
            BoxDims::new([f32::INFINITY, 10.0, 10.0]),
        ];
        let expected = euclidean_distance(&coords[0], &coords[1]);
        for box_dims in &variants {
            assert!(!box_dims.is_periodic());
            let got = pairwise_distances_mic(&coords, Some(box_dims));
            assert_eq!(got.len(), 1);
            assert!(got[0].is_finite());
            assert!(
                (got[0] - expected).abs() < 1e-5,
                "degenerate box {:?} should fall back to plain Euclidean distance, got {} expected {}",
                box_dims,
                got[0],
                expected
            );
        }
    }

    #[test]
    fn test_pairwise_distances_mic_wraps_across_periodic_boundary() {
        // Box: 10x10x10. Two atoms placed near opposite faces along x: 0.5
        // and 9.5. Raw separation is 9.0 — but the true minimum-image
        // separation is 1.0 (image the second atom at 9.5 - 10.0 = -0.5).
        // This is exactly the case a naive non-PBC-aware distance
        // calculation gets wrong (it would report 9.0), and the case this
        // test specifically exercises.
        let box_dims = BoxDims::new([10.0, 10.0, 10.0]);
        let coords = [[0.5, 5.0, 5.0], [9.5, 5.0, 5.0]];
        let got = pairwise_distances_mic(&coords, Some(&box_dims));
        assert_eq!(got.len(), 1);
        assert!(
            (got[0] - 1.0).abs() < 1e-5,
            "expected minimum-image distance 1.0, got {}",
            got[0]
        );

        // Sanity: the naive (unwrapped) distance really is 9.0, confirming
        // this fixture actually distinguishes MIC from plain Euclidean.
        let naive = euclidean_distance(&coords[0], &coords[1]);
        assert!((naive - 9.0).abs() < 1e-5);
    }

    /// Independent ground truth for MIC: brute-force check all 27 periodic
    /// images (each axis's image offset in {-1, 0, 1} box lengths) and take
    /// the minimum. A different, more obviously-correct algorithm than the
    /// per-axis wrap formula under test, so agreement is real signal rather
    /// than the same bug reproduced in two places.
    fn brute_force_min_image(a: &[f32; 3], b: &[f32; 3], lengths: &[f32; 3]) -> f32 {
        let mut best = f32::INFINITY;
        for ix in -1..=1 {
            for iy in -1..=1 {
                for iz in -1..=1 {
                    let shifted = [
                        b[0] + ix as f32 * lengths[0],
                        b[1] + iy as f32 * lengths[1],
                        b[2] + iz as f32 * lengths[2],
                    ];
                    let dx = shifted[0] - a[0];
                    let dy = shifted[1] - a[1];
                    let dz = shifted[2] - a[2];
                    best = best.min((dx * dx + dy * dy + dz * dz).sqrt());
                }
            }
        }
        best
    }

    #[test]
    fn test_pairwise_distances_mic_matches_brute_force_27_image_search() {
        let box_dims = BoxDims::new([12.0, 8.0, 15.0]);
        let coords = [
            [0.2, 7.9, 0.1],   // near a box corner
            [11.8, 0.1, 14.9], // near the opposite corner
            [6.0, 4.0, 7.5],   // interior, far from any face
            [0.0, 0.0, 0.0],
        ];

        let got = pairwise_distances_mic(&coords, Some(&box_dims));
        let mut idx = 0;
        for i in 0..coords.len() {
            for j in (i + 1)..coords.len() {
                let expected = brute_force_min_image(&coords[i], &coords[j], &box_dims.lengths);
                assert!(
                    (got[idx] - expected).abs() < 1e-4,
                    "pair ({}, {}): mic={} brute_force={}",
                    i,
                    j,
                    got[idx],
                    expected
                );
                idx += 1;
            }
        }
    }

    #[test]
    fn test_pairwise_distances_mic_pair_order_matches_numpy_triu_indices() {
        // 4 points on a line at x = 0, 1, 3, 6 — pairwise distances are all
        // distinct, so pair identity is unambiguous from the value alone.
        // Expected np.triu_indices(4, k=1) order:
        // (0,1) (0,2) (0,3) (1,2) (1,3) (2,3).
        let coords = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
        ];
        let got = pairwise_distances_mic(&coords, None);
        let expected = [1.0, 3.0, 6.0, 2.0, 5.0, 3.0];
        assert_eq!(got.len(), expected.len());
        for (g, e) in got.iter().zip(expected.iter()) {
            assert!(
                (g - e).abs() < 1e-5,
                "got {:?} expected {:?}",
                got,
                expected
            );
        }
    }

    #[test]
    fn test_pairwise_distances_mic_generality_across_different_system_sizes() {
        // The primitive must be correct regardless of how many points it's
        // given or what they represent — no hardcoded assumption about
        // chain/residue count or a fixed 2-chain topology. Exercise a small
        // 3-point "monomer-like" system and a differently-shaped 9-point
        // system through the same brute-force oracle used above.
        let box_dims = BoxDims::new([5.0, 5.0, 5.0]);

        let small_system: Vec<[f32; 3]> = vec![[0.1, 0.1, 0.1], [4.9, 0.1, 0.1], [2.5, 2.5, 2.5]];
        let large_system: Vec<[f32; 3]> = (0..9)
            .map(|k| {
                let f = k as f32;
                [(f * 0.7) % 5.0, (f * 1.3) % 5.0, (f * 2.1) % 5.0]
            })
            .collect();

        for system in [small_system, large_system] {
            let n = system.len();
            let got = pairwise_distances_mic(&system, Some(&box_dims));
            assert_eq!(got.len(), n * (n - 1) / 2);
            let mut idx = 0;
            for i in 0..n {
                for j in (i + 1)..n {
                    let expected = brute_force_min_image(&system[i], &system[j], &box_dims.lengths);
                    assert!(
                        (got[idx] - expected).abs() < 1e-4,
                        "n={} pair ({},{}): mic={} brute_force={}",
                        n,
                        i,
                        j,
                        got[idx],
                        expected
                    );
                    idx += 1;
                }
            }
        }
    }
}
