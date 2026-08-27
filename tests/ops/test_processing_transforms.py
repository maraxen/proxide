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

    def test_chain_ids_preserved_per_row_same_chain_count(self) -> None:
        """Regression: batching structures with the same chain COUNT but different chain
        LETTERS must keep each row's own chain_ids, not silently collapse to row 0's.

        Before the fix, `_stack_padded_proteins` passed `chain_ids` through the generic
        `jax.tree_util.tree_map(stack_fn, *padded_proteins)` call. `list` is a JAX pytree
        container, so tree_map recursed into the individual chain-letter strings at each
        matched position across proteins; strings have no `.shape`, so `stack_fn`'s fallback
        (`return first`) silently kept only the FIRST protein's letter at every position --
        batching a 2-chain {"A", "B"} structure with a 2-chain {"C", "D"} structure produced
        `chain_ids == ["A", "B"]` for BOTH rows, discarding the second structure's real chain
        identity with no exception, no warning, and a valid-shaped result.
        """
        p1 = Protein(
            coordinates=np.ones((4, 37, 3)),
            aatype=np.ones(4, dtype=np.int8),
            one_hot_sequence=np.eye(21)[np.zeros(4, dtype=np.int32)],
            mask=np.ones((4,)),
            atom_mask=np.ones((4, 37)),
            residue_index=np.arange(4),
            chain_index=np.array([0, 0, 1, 1], dtype=np.int32),
            chain_ids=["A", "B"],
            dihedrals=None,
            mapping=None,
        )
        p2 = Protein(
            coordinates=np.ones((4, 37, 3)),
            aatype=np.ones(4, dtype=np.int8),
            one_hot_sequence=np.eye(21)[np.zeros(4, dtype=np.int32)],
            mask=np.ones((4,)),
            atom_mask=np.ones((4, 37)),
            residue_index=np.arange(4),
            chain_index=np.array([0, 0, 1, 1], dtype=np.int32),
            chain_ids=["C", "D"],
            dihedrals=None,
            mapping=None,
        )

        batch: Protein = pad_and_collate_proteins([p1, p2])

        assert isinstance(batch, Protein)
        assert batch.chain_ids == [["A", "B"], ["C", "D"]], (
            "each row must keep its OWN chain_ids, not row 0's for every row"
        )

    def test_chain_ids_preserved_per_row_different_chain_count(self) -> None:
        """Regression: batching structures with DIFFERENT chain counts must not crash --
        before the fix, differing chain_ids list lengths made tree_map raise a pytree
        structure-mismatch ValueError before any stacking happened.
        """
        p1 = Protein(
            coordinates=np.ones((4, 37, 3)),
            aatype=np.ones(4, dtype=np.int8),
            one_hot_sequence=np.eye(21)[np.zeros(4, dtype=np.int32)],
            mask=np.ones((4,)),
            atom_mask=np.ones((4, 37)),
            residue_index=np.arange(4),
            chain_index=np.array([0, 0, 1, 1], dtype=np.int32),
            chain_ids=["A", "B"],
            dihedrals=None,
            mapping=None,
        )
        p2 = Protein(
            coordinates=np.ones((3, 37, 3)),
            aatype=np.ones(3, dtype=np.int8),
            one_hot_sequence=np.eye(21)[np.zeros(3, dtype=np.int32)],
            mask=np.ones((3,)),
            atom_mask=np.ones((3, 37)),
            residue_index=np.arange(3),
            chain_index=np.zeros(3, dtype=np.int32),
            chain_ids=["A"],
            dihedrals=None,
            mapping=None,
        )

        batch: Protein = pad_and_collate_proteins([p1, p2])

        assert isinstance(batch, Protein)
        assert batch.chain_ids == [["A", "B"], ["A"]]
