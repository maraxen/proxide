"""Tests for proxide.io.fixtures tensor-bundle helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from proxide.io.fixtures import (
  assert_bundle_keys,
  flatten_tensor_dict,
  load_tensor_bundle,
  save_tensor_bundle,
  unflatten_tensor_dict,
)


def test_flatten_unflatten_roundtrip():
  nested = {
    "a": np.arange(4, dtype=np.float32),
    "b": {"c": np.ones((2, 2), dtype=np.float32)},
  }
  flat = flatten_tensor_dict(nested, prefix="feats")
  assert "feats/a" in flat
  assert "feats/b/c" in flat
  rebuilt = unflatten_tensor_dict(flat, prefix="feats")
  np.testing.assert_array_equal(rebuilt["a"], nested["a"])
  np.testing.assert_array_equal(rebuilt["b"]["c"], nested["b"]["c"])


def test_save_load_tensor_bundle(tmp_path: Path):
  nested = {"x": np.array([1.0, 2.0], dtype=np.float32)}
  npz = tmp_path / "bundle.npz"
  save_tensor_bundle(
    npz,
    nested,
    meta={"example_id": "toy", "n": 2},
    prefix="feats",
  )
  tensors, meta = load_tensor_bundle(npz, unflatten=True, prefix="feats")
  np.testing.assert_array_equal(tensors["x"], nested["x"])
  assert meta["example_id"] == "toy"
  assert meta["n"] == 2


def test_assert_bundle_keys_missing():
  with pytest.raises(KeyError, match="missing required keys"):
    assert_bundle_keys({"a": np.zeros(1)}, ["a", "b"])
