"""Tests for data structure definitions."""

from dataclasses import FrozenInstanceError

import jax.numpy as jnp
import numpy as np
import pytest

from priox.core.containers import Protein, ProteinTuple


def test_protein_structure_frozen():
    """Test that ProteinStructure dataclass is immutable.

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
    )
    with pytest.raises(FrozenInstanceError):
        p.aatype = jnp.ones((1,))  # type: ignore[assignment]


def test_protein_from_tuple_numpy():
    """Test creating Protein from ProteinTuple using NumPy factory."""
    p_tuple = ProteinTuple(
        coordinates=np.ones((10, 37, 3)),
        aatype=np.ones(10, dtype=np.int8),
        atom_mask=np.ones((10, 37)),
        residue_index=np.arange(10),
        chain_index=np.zeros(10, dtype=np.int32),
        dihedrals=None,
        source=None,
        mapping=None,
    )

    p = Protein.from_tuple_numpy(p_tuple)

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
