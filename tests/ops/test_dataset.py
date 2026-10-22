"""Tests for proxide.ops.dataset."""

import jax.numpy as jnp
import numpy as np
import pytest

from proxide import CoordFormat, OutputSpec, parse_structure
from proxide.chem.residues import atom_order
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


def _parsed_leu_frame() -> Protein:
  """A real parsed residue with no CD1 (unplaced slot: zero-filled, atom_mask 0)."""
  return parse_structure(
    "tests/data/leu_missing_cd1.pdb", OutputSpec(coord_format=CoordFormat.Atom37)
  )


def test_is_frame_valid_ignores_nan_in_unplaced_slot_of_parsed_structure() -> None:
  frame = _parsed_leu_frame()
  cd1 = atom_order["CD1"]
  assert not bool(frame.atom_mask[0, cd1])
  coords = np.array(frame.coordinates)
  coords[0, cd1] = np.nan
  assert _is_frame_valid(frame.replace(coordinates=coords)) == (True, "")


def test_is_frame_valid_rejects_nan_in_placed_atom_of_parsed_structure() -> None:
  frame = _parsed_leu_frame()
  ca = atom_order["CA"]
  assert bool(frame.atom_mask[0, ca])
  coords = np.array(frame.coordinates)
  coords[0, ca] = np.nan
  assert _is_frame_valid(frame.replace(coordinates=coords)) == (
    False,
    "NaN values in coordinates",
  )
