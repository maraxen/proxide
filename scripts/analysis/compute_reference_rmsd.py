#!/usr/bin/env python3
"""Compute reference Kabsch RMSD values for proxide-master integration test fixtures.

Generates the golden RMSD constants used in:
  - AC-3 Fixture A: residues 1-5 vs rotated residues 1-5 (should be ~0)
  - AC-3 Fixture B: residues 1-5 vs residues 6-10 of 1UBQ (reference value)

Usage:
    uv run python scripts/analysis/compute_reference_rmsd.py

Output:
    rmsd_fixture_a = <float>   (should be < 1e-4 — known-rotation test)
    rmsd_fixture_b = <float>   (reference value for AC-3 Fixture B)
"""

import argparse
import logging
import math
import sys

import numpy as np

# ---------------------------------------------------------------------------
# 1UBQ backbone coordinates (N, CA, C, O) for residues 1-10
# Source: PDB 1UBQ (https://www.rcsb.org/structure/1UBQ)
# ---------------------------------------------------------------------------

# Residues 1-5: MET, GLN, ILE, PHE, VAL
RESIDUES_1_5 = [
    # MET 1
    [[27.340, 24.430, 2.614],   # N
     [26.266, 25.413, 2.842],   # CA
     [26.913, 26.639, 3.531],   # C
     [27.886, 26.463, 4.263]],  # O
    # GLN 2
    [[26.335, 27.783, 3.258],   # N
     [26.850, 29.024, 3.898],   # CA
     [26.100, 29.200, 5.202],   # C
     [24.865, 29.378, 5.230]],  # O
    # ILE 3
    [[26.842, 29.241, 6.271],   # N
     [26.155, 29.362, 7.552],   # CA
     [26.633, 28.288, 8.497],   # C
     [26.882, 27.140, 8.054]],  # O
    # PHE 4
    [[26.850, 28.665, 9.756],   # N
     [27.303, 27.728, 10.726],  # CA
     [26.290, 26.630, 11.041],  # C
     [25.148, 26.680, 10.583]], # O
    # VAL 5
    [[26.720, 25.674, 11.780],  # N
     [25.855, 24.568, 12.116],  # CA
     [24.528, 24.935, 12.768],  # C
     [23.985, 26.049, 12.563]], # O
]

# Residues 6-10: LYS, THR, LEU, THR, GLY
RESIDUES_6_10 = [
    # LYS 6
    [[24.021, 24.049, 13.465],  # N
     [22.773, 24.262, 14.155],  # CA
     [21.836, 23.047, 14.055],  # C
     [21.876, 22.222, 14.979]], # O
    # THR 7
    [[20.988, 22.974, 13.014],  # N
     [20.085, 21.822, 12.849],  # CA
     [19.231, 21.603, 14.080],  # C
     [18.031, 21.940, 14.051]], # O
    # LEU 8
    [[19.849, 21.046, 15.115],  # N
     [19.074, 20.756, 16.318],  # CA
     [19.700, 19.613, 17.072],  # C
     [20.489, 18.855, 16.488]], # O
    # THR 9
    [[19.270, 19.440, 18.290],  # N
     [19.748, 18.333, 19.097],  # CA
     [20.001, 17.137, 18.268],  # C
     [20.815, 17.209, 17.325]], # O
    # GLY 10
    [[19.367, 16.101, 18.620],  # N
     [19.606, 14.907, 17.831],  # CA
     [20.356, 13.895, 18.618],  # C
     [20.079, 13.717, 19.795]], # O
]


def flatten(coords):
    """Flatten [N][4][3] to (N*4, 3) numpy array."""
    n = len(coords)
    arr = np.array(
        [[coords[r][a] for a in range(4)] for r in range(n)], dtype=np.float64
    ).reshape(-1, 3)
    return arr


def kabsch_rmsd(A_coords, B_coords):
    """Kabsch RMSD between two N-residue fragments (both auto-centered).

    Parameters
    ----------
    A_coords, B_coords : list[list[list[float]]]
        Backbone coords in [N][4][3] layout.

    Returns
    -------
    float
        RMSD in Angstroms.
    """
    A = flatten(A_coords)
    B = flatten(B_coords)

    # Center both fragments.
    A_c = A - A.mean(axis=0)
    B_c = B - B.mean(axis=0)

    # Cross-covariance matrix H = A^T B.
    H = A_c.T @ B_c

    # SVD.
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T

    # Reflection guard: correct d = det(V U^T) < 0.
    d = np.linalg.det(V @ U.T)
    sign = np.sign(d) if d != 0 else 1.0

    n_atoms = len(A)
    norm_sq_a = float(np.sum(A_c ** 2))
    norm_sq_b = float(np.sum(B_c ** 2))

    max_trace = S[0] + S[1] + sign * S[2]
    rmsd_sq = max(0.0, (norm_sq_a + norm_sq_b) / n_atoms - 2.0 * max_trace / n_atoms)
    return math.sqrt(rmsd_sq)


def rotation_matrix_z(theta_deg):
    """3x3 rotation matrix around the z-axis."""
    t = math.radians(theta_deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def apply_rotation(coords, R):
    """Apply rotation matrix R to [N][4][3] coords, returning the same shape."""
    A = flatten(coords)
    rotated = (R @ A.T).T
    n = len(coords)
    return rotated.reshape(n, 4, 3).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-level", default="INFO", help="Logging verbosity")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    log = logging.getLogger(__name__)

    # -----------------------------------------------------------------------
    # Fixture A: 1UBQ residues 1-5 vs 30-degree rotation of the same fragment.
    # Kabsch should recover the rotation and return RMSD ~ 0.
    # -----------------------------------------------------------------------
    R = rotation_matrix_z(30.0)
    rotated_1_5 = apply_rotation(RESIDUES_1_5, R)
    rmsd_a = kabsch_rmsd(RESIDUES_1_5, rotated_1_5)
    log.info("Fixture A RMSD (should be ~0): %.8f", rmsd_a)
    assert rmsd_a < 1e-4, f"Fixture A failed: rmsd={rmsd_a} >= 1e-4"
    print(f"rmsd_fixture_a = {rmsd_a:.8f}")

    # -----------------------------------------------------------------------
    # Fixture B: 1UBQ residues 1-5 vs 6-10.
    # This is the golden value embedded in the integration test.
    # -----------------------------------------------------------------------
    rmsd_b = kabsch_rmsd(RESIDUES_1_5, RESIDUES_6_10)
    log.info("Fixture B RMSD (residues 1-5 vs 6-10): %.6f", rmsd_b)
    print(f"rmsd_fixture_b = {rmsd_b:.6f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
