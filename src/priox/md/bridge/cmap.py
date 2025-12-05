import numpy as np

def solve_periodic_spline_derivatives(y: np.ndarray) -> np.ndarray:
    """Solve for the first derivatives (k) of a periodic cubic spline.

    Args:
        y: (N,) values
    Returns:
        k: (N,) derivatives dy/dx at knots

    """
    n_points = len(y)
    # RHS vector: 3 * (y_{i+1} - y_{i-1})
    y_next = np.roll(y, -1)
    y_prev = np.roll(y, 1)
    rhs = 3.0 * (y_next - y_prev)

    # Matrix A: Diagonals 4, Off-diagonals 1, Corners 1
    matrix_a = np.zeros((n_points, n_points))
    for i in range(n_points):
        matrix_a[i, i] = 4.0
        matrix_a[i, (i-1)%n_points] = 1.0
        matrix_a[i, (i+1)%n_points] = 1.0

    # Solve A * k = rhs
    return np.linalg.solve(matrix_a, rhs)


def compute_bicubic_params(grid: np.ndarray) -> np.ndarray:
    """Compute f, fx, fy, fxy at each grid point for natural bicubic spline.

    Args:
        grid: (N, N) array of values. Assumes grid[i, j] is at x[i], y[j].

    Returns:
        params: (N, N, 4) array where last dim is [f, fx, fy, fxy]

    """
    n_points = grid.shape[0]
    f = grid

    # 1. fx: Solve along cols (derivative w.r.t row index i, which is x)
    fx = np.zeros_like(grid)
    for j in range(n_points):
        fx[:, j] = solve_periodic_spline_derivatives(grid[:, j])

    # 2. fy: Solve along rows (derivative w.r.t col index j, which is y)
    fy = np.zeros_like(grid)
    for i in range(n_points):
        fy[i, :] = solve_periodic_spline_derivatives(grid[i, :])

    # 3. fxy: Solve spline on fx along rows (d/dy of df/dx)
    fxy = np.zeros_like(grid)
    for i in range(n_points):
        fxy[i, :] = solve_periodic_spline_derivatives(fx[i, :])

    # Stack: f, fx, fy, fxy
    return np.stack([f, fx, fy, fxy], axis=-1)
