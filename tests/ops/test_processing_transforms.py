"""Tests for Grain operations for processing protein structures."""


import jax.numpy as jnp
import numpy as np
import pytest

from proxide.chem import residues as rc
from proxide.core.containers import Protein
from proxide.ops import transforms
from proxide.ops.transforms import pad_and_collate_proteins


class TestPadAndCollate:
    """Tests for the pad_and_collate_proteins function."""

    def test_pad_and_collate(self) -> None:
        """Test correct batching and padding of proteins."""
        p1_tuple = Protein(
            coordinates=np.ones((10, 37, 3)),
            aatype=np.ones(10, dtype=np.int8),
            one_hot_sequence=np.eye(21)[np.zeros(10, dtype=np.int32)],
            mask=np.ones((10,)),
            atom_mask=np.ones((10, 37)),
            residue_index=np.arange(10),
            chain_index=np.zeros(10, dtype=np.int32),
            dihedrals=None,
            mapping=None,
        )
        p2_tuple = Protein(
            coordinates=np.ones((15, 37, 3)),
            aatype=np.ones(15, dtype=np.int8),
            one_hot_sequence=np.eye(21)[np.zeros(15, dtype=np.int32)],
            mask=np.ones((15,)),
            atom_mask=np.ones((15, 37)),
            residue_index=np.arange(15),
            chain_index=np.zeros(15, dtype=np.int32),
            dihedrals=None,
            mapping=None,
        )

        elements: list[Protein] = [p1_tuple, p2_tuple]
        batch: Protein = pad_and_collate_proteins(elements)

        assert isinstance(batch, Protein)
        assert batch.coordinates.shape == (2, 15, 37, 3)
        assert batch.aatype.shape == (2, 15)
        assert batch.mask.shape == (2, 15)
        assert batch.residue_index.shape == (2, 15)
        assert batch.chain_index.shape == (2, 15)

        # Check that the first protein is padded correctly
        assert isinstance(batch.coordinates, np.ndarray)
        assert np.all(batch.coordinates[0, 10:] == 0)
        assert np.all(batch.aatype[0, 10:] == 0)

    def test_collate_empty_list_raises_error(self) -> None:
        """Test that collating an empty list raises a ValueError."""
        with pytest.raises(ValueError, match="Cannot collate an empty list"):
            pad_and_collate_proteins([])
