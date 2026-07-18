"""Tests for the radius_of_gyration Rust binding (proxide._proxider).

Unlike proxide.geometry.{transforms,alignment,metrics}, this is a plain
(non-JAX) Rust-backed function -- no chex/jit variants needed. These tests
cover the Python <-> Rust boundary (list vs. numpy input, error handling)
on top of the synthetic ground-truth cases already covered by the Rust
unit tests in proxide-geometry/src/geometry/radius_of_gyration.rs.
"""

from __future__ import annotations

import numpy as np
import pytest

from proxide import radius_of_gyration, weighted_radius_of_gyration


def test_radius_of_gyration_list_input():
    coords = [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]
    assert radius_of_gyration(coords) == pytest.approx(1.0, abs=1e-6)


def test_radius_of_gyration_numpy_input():
    coords = np.array(
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
        dtype=np.float32,
    )
    assert radius_of_gyration(coords) == pytest.approx(1.0, abs=1e-6)


def test_radius_of_gyration_list_and_numpy_agree():
    coords_list = [[1.0, 2.0, 3.0], [4.0, -1.0, 0.5], [0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]
    coords_np = np.array(coords_list, dtype=np.float32)
    assert radius_of_gyration(coords_list) == pytest.approx(
        radius_of_gyration(coords_np), abs=1e-5,
    )


def test_weighted_radius_of_gyration_uniform_weights_matches_unweighted():
    coords = [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]
    weights = [1.0, 1.0, 1.0, 1.0]
    assert weighted_radius_of_gyration(coords, weights) == pytest.approx(
        radius_of_gyration(coords), abs=1e-6,
    )


def test_weighted_radius_of_gyration_heavy_point_dominates():
    # A heavy point at the origin should pull Rg well below the unweighted value.
    coords = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]
    weights = [1000.0, 1.0]
    rg_weighted = weighted_radius_of_gyration(coords, weights)
    rg_unweighted = radius_of_gyration(coords)
    assert rg_weighted < rg_unweighted


def test_weighted_radius_of_gyration_length_mismatch_raises():
    coords = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    weights = [1.0]  # deliberately wrong length
    with pytest.raises(ValueError):
        weighted_radius_of_gyration(coords, weights)


def test_weighted_radius_of_gyration_nan_weight_propagates_nan():
    # Regression test (found by audit): a NaN weight must not silently read
    # as 0.0 -- for an MD frame-quality filter, a silent 0.0 reads as
    # "maximally compact" and would be KEPT by a >-threshold reject filter
    # instead of surfacing as bad input.
    coords = [[10.0, 0.0, 0.0], [-10.0, 0.0, 0.0]]
    weights = [float("nan"), 1.0]
    assert np.isnan(weighted_radius_of_gyration(coords, weights))


def test_weighted_radius_of_gyration_all_zero_weights_is_zero_not_nan():
    coords = [[10.0, 0.0, 0.0], [-10.0, 0.0, 0.0]]
    weights = [0.0, 0.0]
    assert weighted_radius_of_gyration(coords, weights) == 0.0
