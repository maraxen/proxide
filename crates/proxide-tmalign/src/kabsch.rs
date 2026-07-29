//! Kabsch superposition — optimal rotation + translation aligning two Cα
//! point sets of equal length.
//!
//! Generalizes the fixed-size, pre-centered algorithm in
//! `proxide-frag/src/kabsch.rs` (built for whole-fragment RMSD search over
//! a constant-size backbone window) to arbitrary-length point sets that
//! change every DP-refinement iteration, and additionally returns the
//! translation vector — `proxide-frag`'s version only reports rotation
//! since its fragments are pre-centered, but TM-align's own `-m
//! matrix.txt` output reports both `t[m]` and `u[m][0..2]`.
//!
//! Same core algorithm (SVD of the 3×3 cross-covariance matrix with the
//! standard `det(V·Uᵀ)<0` reflection correction), same `nalgebra`
//! dependency.

use nalgebra::{Matrix3, Vector3};

/// Optimal rotation + translation superposing `a` onto `b`:
/// `R·a[i] + t ≈ b[i]`.
#[derive(Debug, Clone)]
pub struct KabschResult {
    /// Root-mean-square deviation after optimal superposition.
    pub rmsd: f32,
    /// Row-major 3×3 rotation matrix R.
    pub rotation: [[f32; 3]; 3],
    /// Translation vector t.
    pub translation: [f32; 3],
}

fn degenerate() -> KabschResult {
    KabschResult {
        rmsd: f32::INFINITY,
        rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        translation: [0.0; 3],
    }
}

/// Compute the optimal RMSD, rotation, and translation superposing `a`
/// onto `b`. `a` and `b` must have equal, non-zero length; returns a
/// degenerate identity result (`rmsd = f32::INFINITY`) otherwise, or if
/// the SVD fails.
pub fn kabsch_superpose(a: &[Vector3<f32>], b: &[Vector3<f32>]) -> KabschResult {
    let n = a.len();
    if n == 0 || n != b.len() {
        return degenerate();
    }
    let n_f = n as f32;

    // Step 1 — centroids + centered point sets.
    let centroid_a: Vector3<f32> = a.iter().sum::<Vector3<f32>>() / n_f;
    let centroid_b: Vector3<f32> = b.iter().sum::<Vector3<f32>>() / n_f;

    // Step 2 — cross-covariance H = sum (a_i - centroid_a)(b_i - centroid_b)^T.
    let mut h = Matrix3::<f32>::zeros();
    let mut norm_sq_a = 0.0_f32;
    let mut norm_sq_b = 0.0_f32;
    for i in 0..n {
        let ca = a[i] - centroid_a;
        let cb = b[i] - centroid_b;
        h += ca * cb.transpose();
        norm_sq_a += ca.norm_squared();
        norm_sq_b += cb.norm_squared();
    }

    // Step 3 — SVD H = U Σ Vᵀ.
    let svd = nalgebra::linalg::SVD::new(h, true, true);
    let (u, v_t) = match (svd.u, svd.v_t) {
        (Some(u), Some(v_t)) => (u, v_t),
        _ => return degenerate(),
    };

    // Step 4 — reflection guard: negate V's 3rd column if det(V·Uᵀ) < 0.
    let mut v = v_t.transpose();
    let d = (v * u.transpose()).determinant();
    let signum = if d < 0.0 { -1.0_f32 } else { 1.0_f32 };
    if d < 0.0 {
        let col2 = v.column(2).clone_owned() * -1.0;
        v.set_column(2, &col2);
    }
    let sv = &svd.singular_values;
    let max_trace = sv[0] + sv[1] + signum * sv[2];

    // Step 5 — RMSD via the inner-product form (no need to explicitly
    // rotate every point first).
    let rmsd_sq = f32::max(
        0.0,
        (norm_sq_a + norm_sq_b) / n_f - 2.0 * max_trace / n_f,
    );
    let rmsd = rmsd_sq.sqrt();

    // Step 6 — rotation R = V Uᵀ (row-major); translation t = centroid_b - R·centroid_a.
    let r_mat = v * u.transpose();
    let rotation = [
        [r_mat[(0, 0)], r_mat[(0, 1)], r_mat[(0, 2)]],
        [r_mat[(1, 0)], r_mat[(1, 1)], r_mat[(1, 2)]],
        [r_mat[(2, 0)], r_mat[(2, 1)], r_mat[(2, 2)]],
    ];
    let t_vec = centroid_b - r_mat * centroid_a;
    let translation = [t_vec[0], t_vec[1], t_vec[2]];

    KabschResult {
        rmsd,
        rotation,
        translation,
    }
}

/// Apply a [`KabschResult`]'s rotation + translation to a single point:
/// `R·p + t`.
pub fn apply_transform(result: &KabschResult, p: Vector3<f32>) -> Vector3<f32> {
    let r = &result.rotation;
    Vector3::new(
        result.translation[0] + r[0][0] * p.x + r[0][1] * p.y + r[0][2] * p.z,
        result.translation[1] + r[1][0] * p.x + r[1][1] * p.y + r[1][2] * p.z,
        result.translation[2] + r[2][0] * p.x + r[2][1] * p.y + r[2][2] * p.z,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn tetrahedron() -> Vec<Vector3<f32>> {
        vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
        ]
    }

    #[test]
    fn identical_point_sets_yield_zero_rmsd() {
        let pts = tetrahedron();
        let result = kabsch_superpose(&pts, &pts);
        assert_relative_eq!(result.rmsd, 0.0, epsilon = 1e-4);
    }

    #[test]
    fn pure_translation_is_recovered_exactly() {
        let a = tetrahedron();
        let shift = Vector3::new(3.0, -2.0, 5.0);
        let b: Vec<_> = a.iter().map(|p| p + shift).collect();
        let result = kabsch_superpose(&a, &b);
        assert_relative_eq!(result.rmsd, 0.0, epsilon = 1e-4);
        assert_relative_eq!(result.translation[0], shift.x, epsilon = 1e-3);
        assert_relative_eq!(result.translation[1], shift.y, epsilon = 1e-3);
        assert_relative_eq!(result.translation[2], shift.z, epsilon = 1e-3);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(result.rotation[i][j], expected, epsilon = 1e-3);
            }
        }
    }

    #[test]
    fn pure_rotation_is_recovered() {
        // 90-degree rotation about z: (x,y,z) -> (-y, x, z)
        let a = vec![
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(1.0, 1.0, 1.0),
        ];
        let b: Vec<_> = a.iter().map(|p| Vector3::new(-p.y, p.x, p.z)).collect();
        let result = kabsch_superpose(&a, &b);
        assert_relative_eq!(result.rmsd, 0.0, epsilon = 1e-3);
        for p in &a {
            let transformed = apply_transform(&result, *p);
            let expected = Vector3::new(-p.y, p.x, p.z);
            assert_relative_eq!(transformed.x, expected.x, epsilon = 1e-3);
            assert_relative_eq!(transformed.y, expected.y, epsilon = 1e-3);
            assert_relative_eq!(transformed.z, expected.z, epsilon = 1e-3);
        }
    }

    #[test]
    fn mismatched_lengths_return_infinite_rmsd() {
        let a = vec![Vector3::new(0.0, 0.0, 0.0)];
        let b = vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0)];
        let result = kabsch_superpose(&a, &b);
        assert!(result.rmsd.is_infinite());
    }

    #[test]
    fn empty_inputs_return_infinite_rmsd() {
        let result = kabsch_superpose(&[], &[]);
        assert!(result.rmsd.is_infinite());
    }
}
