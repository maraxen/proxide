"""Tests for proxide.io.writing."""

import numpy as np
import pytest

from proxide.core.containers import Protein
from proxide.io.writing import _resolve_chain_letters, write_mmcif, write_pdb


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


class TestResolveChainLetters:
  """Regression: chain_ids (per-CHAIN, Shape (N_chains,)) must be resolved through
  chain_index (per-RESIDUE, Shape (N_res,)), not indexed directly by a flattened
  per-atom-slot row index.

  Before this fix, write_pdb/write_mmcif did `chain_ids[i]` for `i` up to
  `len(full_coordinates) - 1` (== N_res * atoms_per_residue for Atom37/Atom14).
  Since `len(chain_ids) == N_chains` is far smaller, every atom past the first
  `N_chains` positions silently fell back to "A" -- meaning any real multi-chain,
  multi-residue Atom37 Protein got every residue after the first one or two
  mislabeled as chain "A", independent of the batching bug this PR also fixes.
  """

  def test_atom37_multi_chain_expands_correctly(self) -> None:
    protein = Protein(
      coordinates=np.ones((4, 37, 3), dtype=np.float32),
      aatype=np.zeros(4, dtype=np.int8),
      residue_index=np.arange(4, dtype=np.int32),
      chain_index=np.array([0, 0, 1, 1], dtype=np.int32),
      chain_ids=["A", "B"],
    )
    n_rows = 4 * 37
    resolved = _resolve_chain_letters(protein, n_rows)
    assert len(resolved) == n_rows
    assert resolved[:74] == ["A"] * 74, "residues 0-1 (74 atom slots) must be chain A"
    assert resolved[74:] == ["B"] * 74, "residues 2-3 (74 atom slots) must be chain B"

  def test_single_chain_still_resolves(self) -> None:
    protein = Protein(
      coordinates=np.ones((2, 37, 3), dtype=np.float32),
      aatype=np.zeros(2, dtype=np.int8),
      residue_index=np.arange(2, dtype=np.int32),
      chain_index=np.zeros(2, dtype=np.int32),
      chain_ids=["A"],
    )
    resolved = _resolve_chain_letters(protein, 2 * 37)
    assert resolved == ["A"] * 74

  def test_flat_full_format_is_already_aligned(self) -> None:
    """The flat "Full" format's chain_index is already per-atom (see
    Protein.from_rust_dict's Full-format branch), so no expansion is needed.
    """
    protein = Protein(
      coordinates=np.ones((5, 3), dtype=np.float32),
      aatype=np.zeros(5, dtype=np.int8),
      residue_index=np.arange(5, dtype=np.int32),
      chain_index=np.array([0, 0, 0, 1, 1], dtype=np.int32),
      chain_ids=["X", "Y"],
    )
    resolved = _resolve_chain_letters(protein, 5)
    assert resolved == ["X", "X", "X", "Y", "Y"]

  def test_no_chain_ids_defaults_to_a(self) -> None:
    protein = Protein(
      coordinates=np.ones((2, 37, 3), dtype=np.float32),
      aatype=np.zeros(2, dtype=np.int8),
      residue_index=np.arange(2, dtype=np.int32),
      chain_index=np.zeros(2, dtype=np.int32),
      chain_ids=None,
    )
    resolved = _resolve_chain_letters(protein, 2 * 37)
    assert resolved == ["A"] * 74

  def test_unalignable_row_count_degrades_to_a_rather_than_crash(self) -> None:
    protein = Protein(
      coordinates=np.ones((3, 37, 3), dtype=np.float32),
      aatype=np.zeros(3, dtype=np.int8),
      residue_index=np.arange(3, dtype=np.int32),
      chain_index=np.array([0, 0, 1], dtype=np.int32),
      chain_ids=["A", "B"],
    )
    # 100 does not divide evenly by n_res=3 -- cannot align, must not crash.
    resolved = _resolve_chain_letters(protein, 100)
    assert resolved == ["A"] * 100


class TestWritePdbChainLetterColumn:
  """End-to-end: write_pdb's actual output uses the resolved chain letter, not a
  raw index into chain_ids.
  """

  def test_real_two_chain_atom37_protein_writes_correct_chain_letters(self, tmp_path) -> None:
    n_res = 4
    n_slots = 37
    protein = Protein(
      coordinates=np.ones((n_res, n_slots, 3), dtype=np.float32),
      aatype=np.zeros(n_res, dtype=np.int8),
      residue_index=np.arange(n_res, dtype=np.int32),
      chain_index=np.array([0, 0, 1, 1], dtype=np.int32),
      chain_ids=["A", "B"],
      full_coordinates=np.ones((n_res * n_slots, 3), dtype=np.float32),
    )
    out = write_pdb(protein, tmp_path / "out.pdb")
    lines = [line for line in out.read_text().splitlines() if line.startswith("ATOM")]
    assert len(lines) == n_res * n_slots
    # Column 22 (1-indexed, per the PDB format) is the chain identifier.
    chain_col = [line[21] for line in lines]
    assert chain_col[: 2 * n_slots] == ["A"] * (2 * n_slots)
    assert chain_col[2 * n_slots :] == ["B"] * (2 * n_slots)
