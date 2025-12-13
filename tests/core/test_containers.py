"""Tests for data structure definitions."""

from dataclasses import FrozenInstanceError

import jax.numpy as jnp
import numpy as np
import pytest

from proxide.core.containers import Protein


def test_protein_structure_frozen():
    """Test that Protein dataclass is immutable.

    Raises:
        FrozenInstanceError: If the dataclass is mutable.

    """
    p = Protein(
        coordinates=jnp.zeros((1, 1, 3)),
        aatype=jnp.zeros((1,)),
        one_hot_sequence=jnp.zeros((1, 21)),
        mask=jnp.zeros((1,)),
        residue_index=jnp.zeros((1,)),
        chain_index=jnp.zeros((1,)),
        atom_mask=jnp.zeros((1, 37)),
    )
    with pytest.raises(FrozenInstanceError):
        p.aatype = jnp.ones((1,))  # type: ignore[assignment]


def test_protein_creation_numpy():
    """Test creating Protein using NumPy arrays directly."""
    p = Protein(
        coordinates=np.ones((10, 37, 3), dtype=np.float32),
        aatype=np.ones(10, dtype=np.int8),
        atom_mask=np.ones((10, 37), dtype=np.float32),
        mask=np.ones(10, dtype=np.float32),
        one_hot_sequence=np.eye(21)[np.ones(10, dtype=np.int8)],
        residue_index=np.arange(10, dtype=np.int32),
        chain_index=np.zeros(10, dtype=np.int32),
    )

    assert isinstance(p, Protein)
    assert isinstance(p.coordinates, np.ndarray)
    assert isinstance(p.aatype, np.ndarray)
    assert isinstance(p.one_hot_sequence, np.ndarray)
    assert isinstance(p.mask, np.ndarray)
    assert isinstance(p.residue_index, np.ndarray)
    assert isinstance(p.chain_index, np.ndarray)

    assert p.coordinates.shape == (10, 37, 3)
    assert p.aatype.shape == (10,)
    assert p.one_hot_sequence.shape == (10, 21)
    assert p.mask.shape == (10,)
    assert p.residue_index.shape == (10,)
    assert p.chain_index.shape == (10,)


def test_protein_from_rust_dict():
    """Test creating Protein from a Rust dictionary output."""
    rust_dict = {
        "coordinates": np.ones((10, 37, 3), dtype=np.float32),
        "aatype": np.ones(10, dtype=np.int8),
        "atom_mask": np.ones((10, 37), dtype=np.float32),
        "residue_index": np.arange(10, dtype=np.int32),
        "chain_index": np.zeros(10, dtype=np.int32),
    }

    p = Protein.from_rust_dict(rust_dict, use_jax=False)

    assert isinstance(p, Protein)
    assert p.coordinates.shape == (10, 37, 3)
    assert p.aatype.shape == (10,)
    assert p.one_hot_sequence.shape == (10, 21)


def test_protein_replace():
    """Test that Protein.replace works correctly."""
    p = Protein(
        coordinates=np.ones((10, 37, 3), dtype=np.float32),
        aatype=np.ones(10, dtype=np.int8),
        atom_mask=np.ones((10, 37), dtype=np.float32),
        mask=np.ones(10, dtype=np.float32),
        one_hot_sequence=np.eye(21)[np.ones(10, dtype=np.int8)],
        residue_index=np.arange(10, dtype=np.int32),
        chain_index=np.zeros(10, dtype=np.int32),
    )

    new_aatype = np.zeros(10, dtype=np.int8)
    p_new = p.replace(aatype=new_aatype)

    assert isinstance(p_new, Protein)
    assert np.all(p_new.aatype == 0)
    assert np.all(p.aatype == 1)  # Original unchanged
