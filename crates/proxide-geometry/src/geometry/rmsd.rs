//! General (arbitrary-length) Kabsch RMSD.
//!
//! `proxide-frag`'s `kabsch_rmsd` (crates/proxide-frag/src/kabsch.rs) is a
//! *different*, shape-constrained function: it operates on fixed-N,
//! 4-atoms-per-residue backbone `Fragment<N, Centered>` values for its
//! MASTER-style fragment-search database. This module reimplements the same
//! rotation-fitting step (SVD of the cross-covariance matrix, with reflection
//! correction for improper rotations) for arbitrary-length `&[[f32; 3]]`
//! coordinate sets -- e.g. a whole MD frame vs. its crystal/native reference,
//! used as an MD trajectory frame-quality signal alongside
//! `radius_of_gyration`.
//!
//! Unlike `proxide-frag`'s version, the RMSD scalar here is computed by
//! directly measuring the aligned residuals (`sqrt(mean(|R*a_i - b_i|^2))`)
//! rather than via the algebraic norm-difference identity
//! (`(||A||^2+||B||^2)/N - 2*trace/N`). The two are mathematically
//! equivalent, but the identity subtracts two O(N) quantities that are
//! nearly equal whenever the true RMSD is small relative to the coordinate
//! magnitudes -- in f32 that catastrophic cancellation was measured to
//! produce residual RMSD values around 4e-3 for a *supposedly exact* pure
//! rotation on ~80 points with ~5 Angstrom-scale coordinates (confirmed via
//! an independent numpy reimplementation of the identity form, which
//! reproduced the identical artifact -- this is a property of the formula,
//! not a bug in either implementation). The direct-residual form has no such
//! cancellation and is exact to rounding at any RMSD scale, which matters
//! here given other parts of this codebase check reconstruction round-trips
//! to <=0.1 degree RMS.
//!
//! The two `kabsch_rmsd` functions are intentionally NOT unified: the fixed-N
//! version avoids heap allocation and is tuned for the fragment-search hot
//! path; this one prioritizes generality and numerical robustness near zero.
//! Do not confuse the two by name alone -- check which module you imported.

use super::transforms::center_coordinates;
use nalgebra::{Matrix3, Vector3};

/// Output of the Kabsch algorithm: optimal RMSD and rotation matrix.
#[derive(Debug, Clone)]
pub struct RmsdResult {
    /// Root-mean-square deviation between the two coordinate sets after
    /// optimal superposition. `f32::INFINITY` if the inputs are degenerate
    /// (mismatched length, empty, or SVD failure).
    pub rmsd: f32,
    /// Row-major 3x3 rotation matrix R such that R * a ~= b.
    pub rotation: [[f32; 3]; 3],
}

const IDENTITY: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// Compute the optimal RMSD and rotation between two ALREADY-CENTERED
/// (centroid subtracted), equal-length coordinate sets.
///
/// Most callers should prefer [`rmsd_with_centering`], which centers both
/// inputs first -- this raw version exists for callers that have already
/// centered once and want to avoid re-centering on every call (e.g. when
/// comparing many frames against the same pre-centered reference).
pub fn kabsch_rmsd(a: &[[f32; 3]], b: &[[f32; 3]]) -> RmsdResult {
    if a.is_empty() || a.len() != b.len() {
        return RmsdResult { rmsd: f32::INFINITY, rotation: IDENTITY };
    }
    let n = a.len() as f32;

    // Cross-covariance H = A^T B (3x3), built as a sum of outer products.
    let mut h = Matrix3::<f32>::zeros();
    for (pa, pb) in a.iter().zip(b.iter()) {
        let va = Vector3::new(pa[0], pa[1], pa[2]);
        let vb = Vector3::new(pb[0], pb[1], pb[2]);
        h += va * vb.transpose();
    }

    let svd = nalgebra::linalg::SVD::new(h, true, true);
    let (u, v_t) = match (svd.u, svd.v_t) {
        (Some(u), Some(v_t)) => (u, v_t),
        _ => return RmsdResult { rmsd: f32::INFINITY, rotation: IDENTITY },
    };

    // Reflection guard: if det(V U^T) < 0, the unconstrained SVD solution is
    // an improper rotation (a reflection); negate the last singular vector's
    // column to force a proper rotation, matching the standard Kabsch fix.
    let mut v = v_t.transpose();
    let d = (v * u.transpose()).determinant();
    if d < 0.0 {
        let col2 = v.column(2).clone_owned() * -1.0;
        v.set_column(2, &col2);
    }

    let r_mat = v * u.transpose();
    let rotation = [
        [r_mat[(0, 0)], r_mat[(0, 1)], r_mat[(0, 2)]],
        [r_mat[(1, 0)], r_mat[(1, 1)], r_mat[(1, 2)]],
        [r_mat[(2, 0)], r_mat[(2, 1)], r_mat[(2, 2)]],
    ];

    // RMSD via direct aligned residuals -- see module doc for why this is
    // preferred over the algebraic norm-difference identity.
    let sum_sq: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(pa, pb)| {
            let va = Vector3::new(pa[0], pa[1], pa[2]);
            let vb = Vector3::new(pb[0], pb[1], pb[2]);
            let rotated = r_mat * va;
            let diff = rotated - vb;
            diff.dot(&diff)
        })
        .sum();
    let rmsd = (sum_sq / n).sqrt();

    RmsdResult { rmsd, rotation }
}

/// Compute the optimal RMSD and rotation between two coordinate sets,
/// centering both (subtracting each set's own centroid) first.
///
/// Use this unless you have already centered both inputs yourself.
pub fn rmsd_with_centering(a: &[[f32; 3]], b: &[[f32; 3]]) -> RmsdResult {
    let mut a_c = a.to_vec();
    let mut b_c = b.to_vec();
    center_coordinates(&mut a_c);
    center_coordinates(&mut b_c);
    kabsch_rmsd(&a_c, &b_c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_coords_zero_rmsd() {
        let a = [[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0], [0.0, 0.0, 0.0]];
        let result = rmsd_with_centering(&a, &a);
        assert!(result.rmsd < 1e-4, "rmsd={}", result.rmsd);
    }

    #[test]
    fn test_pure_rotation_zero_rmsd() {
        // b is a rotated 90deg about z: (x,y,z) -> (-y,x,z).
        let a = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0], [2.0, -1.0, 1.0]];
        let b: Vec<[f32; 3]> = a.iter().map(|p| [-p[1], p[0], p[2]]).collect();
        let result = rmsd_with_centering(&a, &b);
        assert!(result.rmsd < 1e-3, "expected ~0 rmsd for a pure rotation, got {}", result.rmsd);
    }

    #[test]
    fn test_translation_invariance() {
        // Same shape, offset by a large translation -- centering should
        // remove the translation entirely, giving zero RMSD.
        let a = [[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0], [0.0, 0.0, 0.0]];
        let b: Vec<[f32; 3]> = a.iter().map(|p| [p[0] + 50.0, p[1] - 30.0, p[2] + 10.0]).collect();
        let result = rmsd_with_centering(&a, &b);
        assert!(result.rmsd < 1e-4, "rmsd={}", result.rmsd);
    }

    #[test]
    fn test_known_deviation() {
        // Two points, both already centered at the origin, NOT related by a
        // pure rotation (a genuine shape difference). Independently verified
        // via numpy (H = a^T b, SVD, same inner-product RMSD formula this
        // function uses) rather than assumed by hand -- an initial hand
        // guess that the optimal rotation is the identity was WRONG (gave
        // 1.0); the numpy cross-check gives the correct value below.
        // a: (+1,0,0), (-1,0,0)
        // b: (+1,1,0), (-1,-1,0)
        let a = [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]];
        let b = [[1.0, 1.0, 0.0], [-1.0, -1.0, 0.0]];
        let result = kabsch_rmsd(&a, &b);
        let expected = 0.41421356_f32; // sqrt(3 - 2*sqrt(2)), confirmed via numpy SVD
        assert!((result.rmsd - expected).abs() < 1e-3, "rmsd={}", result.rmsd);
    }

    #[test]
    fn test_mismatched_length_returns_infinity() {
        let a = [[0.0, 0.0, 0.0]];
        let b = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let result = kabsch_rmsd(&a, &b);
        assert!(result.rmsd.is_infinite());
    }

    #[test]
    fn test_empty_returns_infinity() {
        let a: [[f32; 3]; 0] = [];
        let b: [[f32; 3]; 0] = [];
        let result = kabsch_rmsd(&a, &b);
        assert!(result.rmsd.is_infinite());
    }
}
