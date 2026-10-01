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
      # Every slot resolved, so every slot is written (debt #1928: the writer
      # now emits only atom_mask-resolved Atom37 slots).
      atom_mask=np.ones((n_res, n_slots), dtype=np.float32),
    )
    out = write_pdb(protein, tmp_path / "out.pdb")
    lines = [line for line in out.read_text().splitlines() if line.startswith("ATOM")]
    assert len(lines) == n_res * n_slots
    # Column 22 (1-indexed, per the PDB format) is the chain identifier.
    chain_col = [line[21] for line in lines]
    assert chain_col[: 2 * n_slots] == ["A"] * (2 * n_slots)
    assert chain_col[2 * n_slots :] == ["B"] * (2 * n_slots)


def _atom_lines(path) -> list[str]:
  return [line for line in path.read_text().splitlines() if line.startswith("ATOM")]


def _flat_protein(n_atoms: int, **overrides) -> Protein:
  """A flat per-atom ("Full"-style) Protein with every per-atom field set."""
  fields = dict(
    coordinates=np.zeros((n_atoms, 37, 3), dtype=np.float32),
    aatype=np.zeros(n_atoms, dtype=np.int8),
    residue_index=np.arange(n_atoms, dtype=np.int32),
    chain_index=np.zeros(n_atoms, dtype=np.int32),
    chain_ids=["A"],
    full_coordinates=np.zeros((n_atoms, 3), dtype=np.float32),
    atom_names=["CA"] * n_atoms,
    res_names=["ALA"] * n_atoms,
    elements=["C"] * n_atoms,
  )
  fields.update(overrides)
  return Protein(**fields)


class TestWritePdbInventsNothing:
  """Debt #1928: write_pdb must not fabricate atoms, names, numbers or elements.

  The old writer filled gaps with "CA" / "UNK" / ``i + 1`` / ``atom_name[0]``:
  an Atom37 protein without per-atom names came out as 37 "CA" atoms per
  residue (unresolved zero-filled slots included), numbered 1..37*N.
  """

  def test_atom37_writes_only_resolved_slots_with_real_names(self, tmp_path) -> None:
    from proxide.chem.residues import atom_types, resnames, restype_order

    mask = np.zeros((2, 37), dtype=np.float32)
    mask[:, :5] = 1.0  # N, CA, C, CB, O
    gly = restype_order["G"]
    ala = restype_order["A"]
    protein = Protein(
      coordinates=np.ones((2, 37, 3), dtype=np.float32),
      aatype=np.array([ala, gly], dtype=np.int8),
      residue_index=np.array([10, 11], dtype=np.int32),
      chain_index=np.zeros(2, dtype=np.int32),
      chain_ids=["A"],
      atom_mask=mask,
    )
    lines = _atom_lines(write_pdb(protein, tmp_path / "out.pdb"))
    assert len(lines) == 10
    assert [line[12:16].strip() for line in lines[:5]] == atom_types[:5]
    assert {line[17:20] for line in lines[:5]} == {resnames[ala]}
    assert {line[17:20] for line in lines[5:]} == {resnames[gly]}
    assert [int(line[22:26]) for line in lines] == [10] * 5 + [11] * 5
    # Elements come from the Atom37 slot table, and one-letter-element names
    # start in column 14 (" CA "), so external readers don't see calcium.
    assert [line[76:78] for line in lines[:5]] == [" N", " C", " C", " C", " O"]
    assert [line[12:16] for line in lines[:5]] == [" N  ", " CA ", " C  ", " CB ", " O  "]

  def test_atom37_element_table_covers_exactly_the_vocabulary(self) -> None:
    from proxide.chem.residues import atom_types
    from proxide.io.writing import _ATOM37_ELEMENT

    assert set(_ATOM37_ELEMENT) == set(atom_types)
    assert set(_ATOM37_ELEMENT.values()) == {"N", "C", "O", "S"}

  @pytest.mark.parametrize("bad_aatype", [-1, 21])
  def test_atom37_out_of_range_aatype_raises(self, tmp_path, bad_aatype) -> None:
    mask = np.zeros((1, 37), dtype=np.float32)
    mask[0, 1] = 1.0
    protein = Protein(
      coordinates=np.ones((1, 37, 3), dtype=np.float32),
      aatype=np.array([bad_aatype], dtype=np.int8),
      residue_index=np.zeros(1, dtype=np.int32),
      chain_index=np.zeros(1, dtype=np.int32),
      chain_ids=["A"],
      atom_mask=mask,
    )
    with pytest.raises(ValueError, match="aatype"):
      write_pdb(protein, tmp_path / "out.pdb")

  def test_mmcif_rejects_empty_or_spaced_tokens(self, tmp_path) -> None:
    with pytest.raises(ValueError, match="atom name ''"):
      write_mmcif(_flat_protein(2, atom_names=["CA", ""]), tmp_path / "out.cif")
    with pytest.raises(ValueError, match="whitespace"):
      write_mmcif(_flat_protein(2, res_names=["ALA", "A A"]), tmp_path / "out.cif")
    assert not (tmp_path / "out.cif").exists()

  def test_mmcif_quotes_special_leading_characters(self, tmp_path) -> None:
    lines = _atom_lines(write_mmcif(_flat_protein(2, atom_names=["CA", "_X"]), tmp_path / "o.cif"))
    assert lines[1].split()[3] == "'_X'"

  def test_atom37_without_atom_mask_raises(self, tmp_path) -> None:
    protein = Protein(
      coordinates=np.ones((1, 37, 3), dtype=np.float32),
      aatype=np.zeros(1, dtype=np.int8),
      residue_index=np.zeros(1, dtype=np.int32),
      chain_index=np.zeros(1, dtype=np.int32),
    )
    with pytest.raises(ValueError, match="atom_mask"):
      write_pdb(protein, tmp_path / "out.pdb")
    assert not (tmp_path / "out.pdb").exists()

  def test_per_atom_missing_res_names_raises(self, tmp_path) -> None:
    with pytest.raises(ValueError, match="res_names"):
      write_pdb(_flat_protein(3, res_names=None), tmp_path / "out.pdb")

  def test_per_atom_length_mismatch_raises(self, tmp_path) -> None:
    with pytest.raises(ValueError, match="2 atom_names for 3 atoms"):
      write_pdb(_flat_protein(3, atom_names=["N", "CA"]), tmp_path / "out.pdb")

  def test_missing_elements_are_blank(self, tmp_path) -> None:
    lines = _atom_lines(write_pdb(_flat_protein(2, elements=None), tmp_path / "out.pdb"))
    assert {line[76:78] for line in lines} == {"  "}

  def test_given_elements_are_written(self, tmp_path) -> None:
    lines = _atom_lines(write_pdb(_flat_protein(2), tmp_path / "out.pdb"))
    assert {line[76:78] for line in lines} == {" C"}

  def test_mmcif_shares_the_same_rules(self, tmp_path) -> None:
    unmasked = Protein(
      coordinates=np.ones((1, 37, 3), dtype=np.float32),
      aatype=np.zeros(1, dtype=np.int8),
      residue_index=np.zeros(1, dtype=np.int32),
      chain_index=np.zeros(1, dtype=np.int32),
    )
    with pytest.raises(ValueError, match="write_mmcif: .*atom_mask"):
      write_mmcif(unmasked, tmp_path / "out.cif")
    lines = _atom_lines(write_mmcif(_flat_protein(2, elements=None), tmp_path / "out.cif"))
    # CIF's unknown marker, not atom_name[0].
    assert [line.split()[2] for line in lines] == ["?", "?"]


class TestWritePdbRejectsColumnOverflow:
  """Debt #1927: values that overflow a fixed-width column raise, and leave no file."""

  @pytest.mark.parametrize(
    ("overrides", "match"),
    [
      ({"residue_index": np.array([0, 10000], dtype=np.int32)}, "residue number 10000"),
      ({"residue_index": np.array([0, -1000], dtype=np.int32)}, "residue number -1000"),
      ({"atom_names": ["CA", "CAXYZ"]}, "atom name"),
      ({"res_names": ["ALA", "ALAX"]}, "residue name"),
      ({"elements": ["C", "XYZ"]}, "element"),
      ({"full_coordinates": np.array([[0, 0, 0], [10000.0, 0, 0]], np.float32)}, "coordinate"),
    ],
  )
  def test_overflow_raises(self, tmp_path, overrides, match) -> None:
    out = tmp_path / "out.pdb"
    with pytest.raises(ValueError, match=match):
      write_pdb(_flat_protein(2, **overrides), out)
    assert not out.exists()

  def test_serial_overflow_raises(self, tmp_path) -> None:
    n = 100_000
    protein = _flat_protein(
      n,
      coordinates=np.zeros((1, 37, 3), dtype=np.float32),
      residue_index=np.zeros(n, dtype=np.int32),
    )
    with pytest.raises(ValueError, match="serial > 99999"):
      write_pdb(protein, tmp_path / "out.pdb")

  @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
  def test_nonfinite_coordinates_raise_in_both_writers(self, tmp_path, bad) -> None:
    # "     nan" is exactly 8 chars, so the width check alone would let it through.
    coords = np.array([[0, 0, 0], [bad, 0, 0]], np.float32)
    protein = _flat_protein(2, full_coordinates=coords)
    with pytest.raises(ValueError, match="non-finite"):
      write_pdb(protein, tmp_path / "out.pdb")
    with pytest.raises(ValueError, match="non-finite"):
      write_mmcif(protein, tmp_path / "out.cif")
    assert not (tmp_path / "out.pdb").exists()
    assert not (tmp_path / "out.cif").exists()

  def test_boundary_values_still_write(self, tmp_path) -> None:
    protein = _flat_protein(
      2,
      residue_index=np.array([-999, 9999], dtype=np.int32),
      full_coordinates=np.array([[-999.999, 0, 0], [9999.999, 0, 0]], np.float32),
    )
    lines = _atom_lines(write_pdb(protein, tmp_path / "out.pdb"))
    assert [int(line[22:26]) for line in lines] == [-999, 9999]


class TestWritePdbOnRealParsedStructures:
  """The writer against Proteins as proxide's own parser produces them, not
  hand-built fixtures (review finding on task 260930_proxide-debt-sweep).
  """

  PDB = "tests/data/5awl.pdb"  # single model, 10 residues

  def test_multimodel_atom37_request_raises(self, tmp_path) -> None:
    # 1uao is an 18-model NMR file; an Atom37 request returns format="Full"
    # with 6660 flattened coordinates and no atom names. The old writer
    # emitted 6660 "CA"/"UNK" rows; refusing is the honest answer.
    from proxide import CoordFormat, OutputSpec, parse_structure

    protein = parse_structure("tests/data/1uao.pdb", OutputSpec(coord_format=CoordFormat.Atom37))
    with pytest.raises(ValueError, match="cannot name the atoms"):
      write_pdb(protein, tmp_path / "out.pdb")

  def test_atom37_round_trips_resolved_atoms(self, tmp_path) -> None:
    from proxide import CoordFormat, OutputSpec, parse_structure

    protein = parse_structure(self.PDB, OutputSpec(coord_format=CoordFormat.Atom37))
    lines = _atom_lines(write_pdb(protein, tmp_path / "out.pdb"))
    assert len(lines) == int(np.asarray(protein.atom_mask).sum())
    reparsed = parse_structure(
      str(tmp_path / "out.pdb"), OutputSpec(coord_format=CoordFormat.Atom37)
    )
    np.testing.assert_array_equal(np.asarray(reparsed.aatype), np.asarray(protein.aatype))
    np.testing.assert_array_equal(
      np.asarray(reparsed.residue_index), np.asarray(protein.residue_index)
    )
    np.testing.assert_allclose(
      np.asarray(reparsed.coordinates)[np.asarray(protein.atom_mask).astype(bool)],
      np.asarray(protein.coordinates)[np.asarray(protein.atom_mask).astype(bool)],
      atol=1e-3,
    )

  def test_full_format_raises_rather_than_writing_unk_residues(self, tmp_path) -> None:
    # parse_structure's Full output carries no per-atom residue names (and
    # from_rust_dict drops chain ids), so the old writer emitted every residue
    # as "UNK" in chain "A" with per-atom-misindexed residue numbers. Until
    # that data is plumbed through, refusing is the honest answer.
    from proxide import CoordFormat, OutputSpec, parse_structure

    protein = parse_structure(self.PDB, OutputSpec(coord_format=CoordFormat.Full))
    with pytest.raises(ValueError, match="res_names"):
      write_pdb(protein, tmp_path / "out.pdb")
