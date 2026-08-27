"""Tests for proxide.io.writing."""

import numpy as np
import pytest

from proxide.core.containers import Protein
from proxide.io.writing import write_mmcif, write_pdb


def _single_structure_protein(chain_ids: list[str] | None) -> Protein:
  """A real, non-batched 2-residue Protein for write_pdb/write_mmcif."""
  return Protein(
    coordinates=np.ones((2, 37, 3), dtype=np.float32),
    aatype=np.zeros(2, dtype=np.int8),
    residue_index=np.arange(2, dtype=np.int32),
    chain_index=np.array([0, 0], dtype=np.int32),
    chain_ids=chain_ids,
    full_coordinates=np.ones((2, 3), dtype=np.float32),
    atom_names=["CA", "CA"],
    res_names=["ALA", "ALA"],
    elements=["C", "C"],
  )


def _batched_protein() -> Protein:
  """A 2-row batched Protein, shaped like pad_and_collate_proteins's output."""
  return Protein(
    coordinates=np.ones((2, 2, 37, 3), dtype=np.float32),
    aatype=np.zeros((2, 2), dtype=np.int8),
    residue_index=np.tile(np.arange(2, dtype=np.int32), (2, 1)),
    chain_index=np.zeros((2, 2), dtype=np.int32),
    chain_ids=[["A", "B"], ["C", "D"]],
  )


class TestWritePdbRejectsBatched:
  """Regression: write_pdb must reject a batched Protein instead of corrupting output.

  Before this fix, a batched Protein's chain_ids (list[list[str]], one entry per batch
  row post the _stack_padded_proteins collision fix) was indexed the same way as a
  single structure's flat chain_ids -- silently stamping a Python list repr like
  "['A', 'B']" into the fixed-width PDB chain-ID column, corrupting the output file.
  """

  def test_rejects_nested_chain_ids(self, tmp_path) -> None:
    protein = _batched_protein()
    with pytest.raises(ValueError, match="batched"):
      write_pdb(protein, tmp_path / "out.pdb")

  def test_rejects_four_dim_coordinates(self, tmp_path) -> None:
    protein = _single_structure_protein(["A"]).replace(
      coordinates=np.ones((2, 2, 37, 3), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="batched"):
      write_pdb(protein, tmp_path / "out.pdb")

  def test_single_structure_still_writes(self, tmp_path) -> None:
    """A real, non-batched multi-chain Protein must still write without raising."""
    protein = _single_structure_protein(["A", "B"])
    out = write_pdb(protein, tmp_path / "out.pdb")
    assert out.exists()
    content = out.read_text()
    assert "ATOM" in content
    assert "END" in content

  def test_none_chain_ids_still_writes(self, tmp_path) -> None:
    protein = _single_structure_protein(None)
    out = write_pdb(protein, tmp_path / "out.pdb")
    assert out.exists()


class TestWriteMmcifRejectsBatched:
  """Same regression coverage as TestWritePdbRejectsBatched, for write_mmcif."""

  def test_rejects_nested_chain_ids(self, tmp_path) -> None:
    protein = _batched_protein()
    with pytest.raises(ValueError, match="batched"):
      write_mmcif(protein, tmp_path / "out.cif")

  def test_rejects_four_dim_coordinates(self, tmp_path) -> None:
    protein = _single_structure_protein(["A"]).replace(
      coordinates=np.ones((2, 2, 37, 3), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="batched"):
      write_mmcif(protein, tmp_path / "out.cif")

  def test_single_structure_still_writes(self, tmp_path) -> None:
    protein = _single_structure_protein(["A", "B"])
    out = write_mmcif(protein, tmp_path / "out.cif")
    assert out.exists()
    content = out.read_text()
    assert "_atom_site.group_PDB" in content
