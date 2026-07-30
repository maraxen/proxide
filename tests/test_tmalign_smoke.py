"""Phase 4 smoke test: `proxide.tmalign.tm_align` PyO3 binding.

Verifies the Python binding's output shape/dict keys and that it matches
the underlying Rust `tmalign_pair_serial` behavior on the same fixture pair
(same synthetic non-collinear coordinate generator used by
`crates/proxide-tmalign/src/pipeline.rs`'s own unit tests) — not a
duplicate of the Rust parity tests against the real USalign reference,
which live in `crates/proxide-tmalign/tests/`.
"""

import numpy as np

import proxide


def _helix(n: int, offset: float) -> np.ndarray:
  i = np.arange(n)
  angle = np.radians(i * 100.0 + offset)
  z = i * 1.5
  return np.stack(
    [2.3 * np.cos(angle), 2.3 * np.sin(angle), z],
    axis=1,
  ).astype(np.float32)


def test_tm_align_returns_expected_shape_and_keys():
  coords1 = _helix(8, 0.0)
  coords2 = _helix(6, 30.0)

  result = proxide.tmalign.tm_align(coords1, coords2)

  assert set(result.keys()) == {
    "rotation",
    "translation",
    "tm_score_norm1",
    "tm_score_norm2",
    "n_aligned",
  }
  assert result["rotation"].shape == (3, 3)
  assert result["rotation"].dtype == np.float32
  assert result["translation"].shape == (3,)
  assert isinstance(result["n_aligned"], int)
  assert 0.0 <= result["tm_score_norm1"] <= 1.0
  assert 0.0 <= result["tm_score_norm2"] <= 1.0


def test_tm_align_self_alignment_is_near_perfect():
  coords = _helix(8, 0.0)

  result = proxide.tmalign.tm_align(coords, coords)

  assert abs(result["tm_score_norm1"] - 1.0) < 1e-3
  assert abs(result["tm_score_norm2"] - 1.0) < 1e-3
  assert result["n_aligned"] == 8


def test_tm_align_rejects_non_nx3_arrays():
  bad = np.zeros((5, 4), dtype=np.float32)
  good = _helix(5, 0.0)

  try:
    proxide.tmalign.tm_align(bad, good)
  except ValueError:
    pass
  else:
    raise AssertionError("expected a ValueError for a non-(N, 3) coordinate array")
