"""Tests for the kabsch_rmsd Rust binding (proxide._proxider).

Plain (non-JAX) Rust-backed function, mirroring test_radius_of_gyration.py's
approach -- covers the Python <-> Rust boundary on top of the synthetic
ground-truth cases already covered by the Rust unit tests in
proxide-geometry/src/geometry/rmsd.rs.
"""

from __future__ import annotations

import numpy as np
import pytest

from proxide import kabsch_rmsd


def test_identical_coords_zero_rmsd():
    coords = [[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0], [0.0, 0.0, 0.0]]
    result = kabsch_rmsd(coords, coords)
    assert result["rmsd"] == pytest.approx(0.0, abs=1e-4)


def test_pure_rotation_zero_rmsd():
    a = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0], [2.0, -1.0, 1.0]]
    b = [[-p[1], p[0], p[2]] for p in a]  # 90deg rotation about z
    result = kabsch_rmsd(a, b)
    assert result["rmsd"] == pytest.approx(0.0, abs=1e-3)


def test_known_deviation_matches_numpy_reference():
    # Independently verified via numpy (see rmsd.rs::test_known_deviation
    # for the derivation) -- NOT a hand-guessed value.
    a = [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
    b = [[1.0, 1.0, 0.0], [-1.0, -1.0, 0.0]]
    result = kabsch_rmsd(a, b)
    assert result["rmsd"] == pytest.approx(0.41421356, abs=1e-3)


def test_numpy_input_matches_list_input():
    a_list = [[1.0, 2.0, 3.0], [4.0, -1.0, 0.5], [0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]
    b_list = [[1.1, 2.1, 3.1], [4.2, -1.1, 0.4], [0.1, -0.1, 0.1], [2.2, 1.9, 2.1]]
    a_np = np.array(a_list, dtype=np.float32)
    b_np = np.array(b_list, dtype=np.float32)
    assert kabsch_rmsd(a_list, b_list)["rmsd"] == pytest.approx(
        kabsch_rmsd(a_np, b_np)["rmsd"], abs=1e-5,
    )


def test_rotation_shape():
    a = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    b = [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]
    result = kabsch_rmsd(a, b)
    assert len(result["rotation"]) == 3
    assert all(len(row) == 3 for row in result["rotation"])


def test_mismatched_length_raises():
    a = [[0.0, 0.0, 0.0]]
    b = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    with pytest.raises(ValueError):
        kabsch_rmsd(a, b)
