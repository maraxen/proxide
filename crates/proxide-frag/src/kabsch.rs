//! Kabsch RMSD algorithm — optimal rotation and RMSD between two centered
//! backbone fragments of the same length.
//!
//! Uses the inner-product form:
//!   RMSD² = (||A||² + ||B||²) / N_atoms − 2 · max_trace(SVD(H)) / N_atoms
//!
//! where H = Aᵀ B is the 3×3 cross-covariance matrix, and max_trace is the
//! sum of singular values with a reflection correction applied to the last
//! singular value when det(V Uᵀ) < 0.

use crate::fragment::{Centered, Fragment};
use nalgebra::{Matrix3, Vector3};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Output of the Kabsch algorithm: optimal RMSD and rotation matrix.
#[derive(Debug, Clone)]
pub struct KabschResult {
    /// Root-mean-square deviation between the two fragments after optimal superposition.
    pub rmsd: f32,
    /// Row-major 3×3 rotation matrix R such that R·a ≈ b.
    pub rotation: [[f32; 3]; 3],
}

// ---------------------------------------------------------------------------
// Public function
// ---------------------------------------------------------------------------

/// Compute the optimal RMSD and rotation between two centered fragments.
///
/// # Parameters
///
/// - `a`: first centered fragment
/// - `norm_sq_a`: precomputed `||A||²` (call [`Fragment::norm_sq`])
/// - `b`: second centered fragment
/// - `norm_sq_b`: precomputed `||B||²`
///
/// Returns a [`KabschResult`] with `rmsd = f32::INFINITY` if the SVD fails
/// (degenerate input).
pub fn kabsch_rmsd<const N: usize>(
    a: &Fragment<N, Centered>,
    norm_sq_a: f32,
    b: &Fragment<N, Centered>,
    norm_sq_b: f32,
) -> KabschResult {
    let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    // Step 1 — Build cross-covariance H = Aᵀ B (3×3).
    let mut h = Matrix3::<f32>::zeros();
    for r in 0..N {
        for at in 0..4 {
            let va = Vector3::from(a.coords[r][at]);
            let vb = Vector3::from(b.coords[r][at]);
            // outer product: H += va * vb^T
            h += va * vb.transpose();
        }
    }

    // Step 2 — Singular Value Decomposition: H = U Σ Vᵀ.
    let svd = nalgebra::linalg::SVD::new(h, true, true);

    // Step 3 — SVD failure guard.
    let (u, v_t) = match (svd.u, svd.v_t) {
        (Some(u), Some(v_t)) => (u, v_t),
        _ => {
            return KabschResult {
                rmsd: f32::INFINITY,
                rotation: identity,
            }
        }
    };

    // Step 4 — Reflection guard.
    // V is v_t.transpose() (nalgebra SVD decomposes H = U Σ Vᵀ,
    // so "v_t" is already Vᵀ and v_t.transpose() gives V).
    let mut v = v_t.transpose();
    let d = (v * u.transpose()).determinant();
    let signum = if d < 0.0 { -1.0_f32 } else { 1.0_f32 };
    if d < 0.0 {
        // Negate column 2 of V to fix reflection.
        let col2 = v.column(2).clone_owned() * -1.0;
        v.set_column(2, &col2);
    }
    let sv = &svd.singular_values; // [σ₀, σ₁, σ₂], descending
    let max_trace = sv[0] + sv[1] + signum * sv[2];

    // Step 5 — RMSD.
    let atoms_total = (N * 4) as f32;
    let rmsd_sq = f32::max(
        0.0,
        (norm_sq_a + norm_sq_b) / atoms_total - 2.0 * max_trace / atoms_total,
    );
    let rmsd = rmsd_sq.sqrt();

    // Step 6 — Rotation matrix R = V Uᵀ (row-major).
    let r_mat = v * u.transpose();
    let rotation = [
        [r_mat[(0, 0)], r_mat[(0, 1)], r_mat[(0, 2)]],
        [r_mat[(1, 0)], r_mat[(1, 1)], r_mat[(1, 2)]],
        [r_mat[(2, 0)], r_mat[(2, 1)], r_mat[(2, 2)]],
    ];

    KabschResult { rmsd, rotation }
}
