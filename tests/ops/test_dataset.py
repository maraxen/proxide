"""Tests for proxide.ops.dataset."""

import jax.numpy as jnp
import numpy as np
import pytest

from proxide.core.containers import Protein
from proxide.ops.dataset import _is_frame_valid


def _make_protein(*, coordinates: np.ndarray | jnp.ndarray) -> Protein:
  length = coordinates.shape[0]
  return Protein(
    coordinates=coordinates,
    aatype=jnp.zeros((length,), dtype=jnp.int32),
    one_hot_sequence=jnp.zeros((length, 21)),
    mask=jnp.ones((length,)),
    residue_index=jnp.arange(length),
    chain_index=jnp.zeros((length,)),
    atom_mask=jnp.ones((length, 37)),
  )


def test_is_frame_valid_accepts_jax_coordinates() -> None:
  frame = _make_protein(coordinates=jnp.ones((4, 37, 3)))
  assert _is_frame_valid(frame) == (True, "")


def test_is_frame_valid_rejects_jax_nan_coordinates() -> None:
  coords = jnp.ones((4, 37, 3))
  coords = coords.at[1, 0, 0].set(jnp.nan)
  frame = _make_protein(coordinates=coords)
  assert _is_frame_valid(frame) == (False, "NaN values in coordinates")


def test_is_frame_valid_accepts_numpy_coordinates() -> None:
  frame = _make_protein(coordinates=np.ones((4, 37, 3), dtype=np.float32))
  assert _is_frame_valid(frame) == (True, "")


def test_is_frame_valid_rejects_empty_structure() -> None:
  frame = _make_protein(coordinates=jnp.zeros((0, 37, 3)))
  assert _is_frame_valid(frame) == (False, "Empty structure")


def test_is_frame_valid_rejects_shape_mismatch() -> None:
  frame = _make_protein(coordinates=jnp.ones((4, 37, 3)))
  frame = frame.replace(aatype=jnp.zeros((3,), dtype=jnp.int32))
  assert _is_frame_valid(frame) == (
    False,
    "Shape mismatch between coordinates and aatype",
  )
