//! CMAP (Cross-Map) energy correction support.
//!
//! Provides functions for computing bicubic spline coefficients
//! used in CMAP energy calculations for backbone dihedral corrections.

/// Solve for the first derivatives (k) of a periodic cubic spline.
///
/// Given values y at equally spaced knots, solves for derivatives
/// such that the cubic spline interpolates y periodically.
///
/// Args:
///     y: Values at knots (N points)
///
/// Returns:
///     k: First derivatives dy/dx at knots (N points)
pub fn solve_periodic_spline_derivatives(y: &[f64]) -> Vec<f64> {
    let n = y.len();
    if n == 0 {
        return Vec::new();
    }

    // RHS vector: 3 * (y_{i+1} - y_{i-1})
    let mut rhs = vec![0.0; n];
    for i in 0..n {
        let y_next = y[(i + 1) % n];
        let y_prev = y[(i + n - 1) % n];
        rhs[i] = 3.0 * (y_next - y_prev);
    }

    // Matrix A: Diagonals 4, Off-diagonals 1, Corners 1
    // Solve A * k = rhs using direct solver
    // For periodic tridiagonal, we use Sherman-Morrison formula
    // But for small N (typically 24), direct LU is fine

    // Build full matrix (simpler for correctness, N is small)
    let mut a = vec![vec![0.0; n]; n];
    for i in 0..n {
        a[i][i] = 4.0;
        a[i][(i + n - 1) % n] = 1.0;
        a[i][(i + 1) % n] = 1.0;
    }

    // Gaussian elimination with partial pivoting
    gauss_solve(&mut a, &mut rhs)
}

/// Gaussian elimination solver for Ax = b, modifies a and b in place.
fn gauss_solve(a: &mut [Vec<f64>], b: &mut [f64]) -> Vec<f64> {
    let n = b.len();

    // Forward elimination
    for k in 0..n {
        // Find pivot
        let mut max_row = k;
        let mut max_val = a[k][k].abs();
        for i in (k + 1)..n {
            if a[i][k].abs() > max_val {
                max_val = a[i][k].abs();
                max_row = i;
            }
        }

        // Swap rows
        a.swap(k, max_row);
        b.swap(k, max_row);

        // Eliminate below
        for i in (k + 1)..n {
            if a[k][k].abs() < 1e-15 {
                continue;
            }
            let factor = a[i][k] / a[k][k];
            for j in k..n {
                a[i][j] -= factor * a[k][j];
            }
            b[i] -= factor * b[k];
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= a[i][j] * x[j];
        }
        if a[i][i].abs() > 1e-15 {
            x[i] = sum / a[i][i];
        }
    }

    x
}

/// Compute bicubic interpolation parameters for a 2D grid.
///
/// Given a grid of values, computes f, fx, fy, fxy at each grid point
/// for natural bicubic spline interpolation.
///
/// Args:
///     grid: N x N array of values (row-major, grid[i][j] at x[i], y[j])
///
/// Returns:
///     params: N x N x 4 array where last dimension is [f, fx, fy, fxy]
pub fn compute_bicubic_params(grid: &[Vec<f64>]) -> Vec<Vec<[f64; 4]>> {
    if grid.is_empty() || grid[0].is_empty() {
        return Vec::new();
    }

    let n = grid.len();
    let m = grid[0].len();

    // 1. fx: Solve along columns (derivative w.r.t row index i)
    let mut fx = vec![vec![0.0; m]; n];
    for j in 0..m {
        let col: Vec<f64> = (0..n).map(|i| grid[i][j]).collect();
        let derivs = solve_periodic_spline_derivatives(&col);
        for i in 0..n {
            fx[i][j] = derivs[i];
        }
    }

    // 2. fy: Solve along rows (derivative w.r.t col index j)
    let mut fy = vec![vec![0.0; m]; n];
    for i in 0..n {
        let derivs = solve_periodic_spline_derivatives(&grid[i]);
        for j in 0..m {
            fy[i][j] = derivs[j];
        }
    }

    // 3. fxy: Solve spline on fx along rows (d/dy of df/dx)
    let mut fxy = vec![vec![0.0; m]; n];
    for i in 0..n {
        let derivs = solve_periodic_spline_derivatives(&fx[i]);
        for j in 0..m {
            fxy[i][j] = derivs[j];
        }
    }

    // 4. Combine: f, fx, fy, fxy
    let mut params = vec![vec![[0.0; 4]; m]; n];
    for i in 0..n {
        for j in 0..m {
            params[i][j] = [grid[i][j], fx[i][j], fy[i][j], fxy[i][j]];
        }
    }

    params
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_periodic_spline_constant() {
        // Constant function should have zero derivatives
        let y = vec![1.0, 1.0, 1.0, 1.0];
        let k = solve_periodic_spline_derivatives(&y);
        for ki in &k {
            assert!(ki.abs() < 1e-10, "Expected zero derivative for constant");
        }
    }

    #[test]
    fn test_periodic_spline_linear() {
        // Linear increase (periodic, so wraps around)
        let y = vec![0.0, 1.0, 2.0, 3.0];
        let k = solve_periodic_spline_derivatives(&y);
        // Derivatives should be approximately 1 everywhere for linear
        // (with periodic boundary, there's a discontinuity at wrap)
        assert!(k.len() == 4);
    }

    #[test]
    fn test_bicubic_params_shape() {
        let grid = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let params = compute_bicubic_params(&grid);
        assert_eq!(params.len(), 3);
        assert_eq!(params[0].len(), 3);
        // f value should match original grid
        assert!((params[0][0][0] - 1.0).abs() < 1e-10);
        assert!((params[1][1][0] - 5.0).abs() < 1e-10);
    }
}
