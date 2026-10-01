"""Tests for proxide.io.writing."""

from pathlib import Path

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
  """chain_ids (per-CHAIN vocabulary) are resolved through chain_index, which
  must already be aligned with the rows being labeled.

  Debt #2354: this used to return "A" when chain info was missing, misaligned
  or out of range, and to np.repeat a per-residue chain_index over atom rows
  whenever the counts happened to divide -- silently merging or mislabeling
  chains. Now: missing -> "" (unknown), misaligned/out-of-range -> ValueError.
  """

  @staticmethod
  def _protein(chain_index, chain_ids) -> Protein:
    n = len(chain_index)
    return Protein(
      coordinates=np.ones((n, 37, 3), dtype=np.float32),
      aatype=np.zeros(n, dtype=np.int8),
      residue_index=np.arange(n, dtype=np.int32),
      chain_index=np.asarray(chain_index, dtype=np.int32),
      chain_ids=chain_ids,
    )

  def test_multi_chain_resolves_per_row(self) -> None:
    protein = self._protein([0, 0, 1, 1], ["A", "B"])
    assert _resolve_chain_letters(protein, 4) == ["A", "A", "B", "B"]

  def test_no_chain_ids_is_unknown_not_a(self) -> None:
    protein = self._protein([0, 0], None)
    assert _resolve_chain_letters(protein, 2) == ["", ""]

  def test_misaligned_rows_raise_instead_of_repeating(self) -> None:
    # 4 residues, 148 atom rows: divisible, so the old code np.repeat-ed.
    protein = self._protein([0, 0, 1, 1], ["A", "B"])
    with pytest.raises(ValueError, match="refusing to guess an alignment"):
      _resolve_chain_letters(protein, 4 * 37)

  def test_unalignable_row_count_raises(self) -> None:
    protein = self._protein([0, 0, 1], ["A", "B"])
    with pytest.raises(ValueError, match="refusing to guess an alignment"):
      _resolve_chain_letters(protein, 100)

  def test_out_of_range_chain_index_raises(self) -> None:
    protein = self._protein([0, 1], ["B"])
    with pytest.raises(ValueError, match=r"chain_index values \[1\]"):
      _resolve_chain_letters(protein, 2)

  def test_unknown_chain_is_blank_in_pdb_and_refused_by_mmcif(self, tmp_path) -> None:
    protein = _flat_protein(2, chain_ids=None)
    pdb_lines = _atom_lines(write_pdb(protein, tmp_path / "out.pdb"))
    assert [line[21] for line in pdb_lines] == [" ", " "]
    # label_asym_id is mandatory and proxide's reader rejects "?", so mmCIF
    # cannot represent an unknown chain: refuse rather than write an
    # unreadable file.
    with pytest.raises(ValueError, match="chain ids are unknown"):
      write_mmcif(protein, tmp_path / "out.cif")
    assert not (tmp_path / "out.cif").exists()


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

  def test_model_stack_consumers_count_one_structure_or_refuse(self) -> None:
    # Review #9: consumers that assume one structure used to silently
    # mis-handle an 18-model stack (CLI info said 18 residues, num_atoms was
    # summed over all models, truncate_protein cropped the model axis).
    from proxide import CoordFormat, OutputSpec, parse_structure
    from proxide.ops.transforms import truncate_protein

    stack = parse_structure("tests/data/1uao.pdb", OutputSpec(coord_format=CoordFormat.Atom37))
    one = parse_structure(
      "tests/data/1uao.pdb", OutputSpec(coord_format=CoordFormat.Atom37, models=[1])
    )
    assert stack.n_models == 18 and one.n_models == 1
    assert stack.num_atoms == one.num_atoms  # one structure's atoms, not 18x
    with pytest.raises(ValueError, match="18 models"):
      _ = stack.atom_residue_ids
    with pytest.raises(ValueError, match="18 models"):
      truncate_protein(stack, max_length=5, strategy="center_crop")

  def test_cli_info_reports_residues_not_models(self) -> None:
    from typer.testing import CliRunner

    from proxide.cli.main import app

    result = CliRunner().invoke(app, ["info", "tests/data/1uao.pdb"])
    assert result.exit_code == 0, result.output
    assert "Number of Residues" in result.output
    residues_row = next(x for x in result.output.splitlines() if "Number of Residues" in x)
    assert "10" in residues_row and "18" not in residues_row, residues_row

  def test_multimodel_atom37_is_a_consistent_model_stack(self, tmp_path) -> None:
    # Debt #2355: 1uao is an 18-model NMR file. An Atom37 request used to
    # return format="Full" with 6660 flattened coordinates next to a
    # 77-entry mask. It is now a model stack whose mask matches it, the
    # writer refuses it as batched, and one selected model still writes.
    from proxide import CoordFormat, OutputSpec, parse_structure

    path = "tests/data/1uao.pdb"
    stack = parse_structure(path, OutputSpec(coord_format=CoordFormat.Atom37))
    assert stack.format == "Atom37"
    assert np.shape(stack.coordinates) == (18, 10, 37, 3)
    assert np.shape(stack.atom_mask) == (18, 10, 37)
    with pytest.raises(ValueError, match="batched"):
      write_pdb(stack, tmp_path / "out.pdb")

    one = parse_structure(path, OutputSpec(coord_format=CoordFormat.Atom37, models=[1]))
    lines = _atom_lines(write_pdb(one, tmp_path / "one.pdb"))
    assert len(lines) == int(np.asarray(one.atom_mask).sum())
    # Model 1 of the stack is exactly the single-model parse.
    np.testing.assert_array_equal(np.asarray(stack.atom_mask)[0], np.asarray(one.atom_mask))
    np.testing.assert_allclose(np.asarray(stack.coordinates)[0], np.asarray(one.coordinates))

  def test_multimodel_non_atom37_request_warns(self) -> None:
    from proxide import CoordFormat, OutputSpec, parse_structure

    with pytest.warns(UserWarning, match="18 models present"):
      parse_structure("tests/data/1uao.pdb", OutputSpec(coord_format=CoordFormat.Full))
    # A second parse (no cache for multi-model results) must warn again.
    with pytest.warns(UserWarning, match="18 models present"):
      parse_structure("tests/data/1uao.pdb", OutputSpec(coord_format=CoordFormat.Full))

  @staticmethod
  def _cache_probe(tmp_path, name: str) -> tuple[Path, str]:
    """A private copy of the fixture, plus the same file with x shifted.

    The format cache is keyed by path alone, so after overwriting the file a
    cache HIT still returns the old coordinates and a miss the new ones --
    which is how these tests know the second parse came from the cache.
    """
    src = Path(TestWritePdbOnRealParsedStructures.PDB).read_text()
    shifted = "".join(
      (x[:30] + f"{float(x[30:38]) + 50.0:8.3f}" + x[38:]) if x.startswith("ATOM") else x
      for x in src.splitlines(keepends=True)
    )
    path = tmp_path / name
    path.write_text(src)
    return path, shifted

  def test_full_format_cache_hit_returns_the_same_per_atom_fields(self, tmp_path) -> None:
    from proxide import CoordFormat, OutputSpec, parse_structure

    path, shifted = self._cache_probe(tmp_path, "full_cache.pdb")
    spec = OutputSpec(coord_format=CoordFormat.Full, enable_caching=True)
    first = parse_structure(str(path), spec)
    path.write_text(shifted)
    second = parse_structure(str(path), spec)
    # Proof of a cache hit: the old coordinates came back.
    np.testing.assert_allclose(np.asarray(second.coordinates), np.asarray(first.coordinates))
    for field in ("elements", "res_names", "atom_chain_ids", "atom_res_index"):
      a, b = getattr(first, field), getattr(second, field)
      assert a is not None and b is not None, field
      assert list(np.asarray(a)) == list(np.asarray(b)), field

  def test_atom37_cache_hit_keeps_the_chain_vocabulary(self, tmp_path) -> None:
    from proxide import CoordFormat, OutputSpec, parse_structure

    path, shifted = self._cache_probe(tmp_path, "a37_cache.pdb")
    spec = OutputSpec(coord_format=CoordFormat.Atom37, enable_caching=True)
    first = parse_structure(str(path), spec)
    path.write_text(shifted)
    second = parse_structure(str(path), spec)
    np.testing.assert_allclose(np.asarray(second.coordinates), np.asarray(first.coordinates))
    assert second.chain_ids == first.chain_ids == ["A"]

  def test_model_selected_parse_is_not_cached(self, tmp_path) -> None:
    # The cache key has no model selection: models=[2] must not be served to
    # (or from) an all-models request.
    from proxide import CoordFormat, OutputSpec, parse_structure

    path = tmp_path / "nmr.pdb"
    path.write_text(Path("tests/data/1uao.pdb").read_text())
    one = parse_structure(
      str(path), OutputSpec(coord_format=CoordFormat.Atom37, enable_caching=True, models=[2])
    )
    assert np.shape(one.coordinates) == (10, 37, 3)
    every = parse_structure(str(path), OutputSpec(coord_format=CoordFormat.Atom37, enable_caching=True))
    assert np.shape(every.coordinates) == (18, 10, 37, 3)

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

  def test_atom14_writes_the_same_atoms_as_atom37(self, tmp_path) -> None:
    # Backlog #5687: Atom14 slot names follow from aatype, so nothing has to
    # be invented. The two layouts of one structure must write identical atoms.
    from proxide import CoordFormat, OutputSpec, parse_structure

    def atoms(fmt, name):
      protein = parse_structure(self.PDB, OutputSpec(coord_format=fmt))
      return sorted(
        (line[17:20], int(line[22:26]), line[12:16].strip(), line[30:54], line[76:78])
        for line in _atom_lines(write_pdb(protein, tmp_path / name))
      )

    a14 = atoms(CoordFormat.Atom14, "a14.pdb")
    a37 = atoms(CoordFormat.Atom37, "a37.pdb")
    # Atom14 has no OXT slot, so the C-terminal OXT exists only in Atom37.
    assert a14 == [a for a in a37 if a[2] != "OXT"]
    assert len(a14) > 0 and len(a37) - len(a14) == 1

  def test_atom14_mask_on_a_slot_the_residue_lacks_raises(self, tmp_path) -> None:
    from proxide.chem.residues import restype_order

    mask = np.zeros((1, 14), dtype=np.float32)
    mask[0, :5] = 1.0  # N, CA, C, O, CB
    mask[0, 5] = 1.0  # ALA has no 6th atom14 slot
    protein = Protein(
      coordinates=np.ones((1, 14, 3), dtype=np.float32),
      aatype=np.array([restype_order["A"]], dtype=np.int8),
      residue_index=np.zeros(1, dtype=np.int32),
      chain_index=np.zeros(1, dtype=np.int32),
      chain_ids=["A"],
      atom_mask=mask,
    )
    with pytest.raises(ValueError, match="no atom for their residue type"):
      write_pdb(protein, tmp_path / "out.pdb")

  def test_chains_out_of_sorted_order_keep_their_letters(self, tmp_path) -> None:
    # Review finding 2026-10-01: unique_chain_ids was built in file order
    # while chain_index comes from a sorted map, so a file with chain B before
    # chain A swapped every atom's chain letter. Chain B is shifted +100 A in
    # x so each written atom's true chain is recoverable from its coordinate.
    from proxide import CoordFormat, OutputSpec, parse_structure

    src = [line for line in open(self.PDB) if line.startswith("ATOM")]

    def as_shifted_b(line: str) -> str:
      return line[:21] + "B" + line[22:30] + f"{float(line[30:38]) + 100.0:8.3f}" + line[38:]

    b_first = tmp_path / "b_first.pdb"
    b_first.write_text("".join(as_shifted_b(x) for x in src) + "TER\n" + "".join(src) + "END\n")
    for fmt in (CoordFormat.Atom37, CoordFormat.Atom14):
      protein = parse_structure(str(b_first), OutputSpec(coord_format=fmt))
      lines = _atom_lines(write_pdb(protein, tmp_path / f"{fmt}.pdb"))
      truth = ["B" if float(line[30:38]) > 50 else "A" for line in lines]
      assert [line[21] for line in lines] == truth, fmt

  def test_chain_filtered_structure_keeps_its_chain_letter(self, tmp_path) -> None:
    # Debt #2354: load_rust(chain_id="B") used to keep chain_index=1 while
    # setting chain_ids=["B"], so the writer emitted chain "A".
    from proxide.io.parsing.backend import load_rust

    src = [line for line in open(self.PDB) if line.startswith("ATOM")]
    two_chains = tmp_path / "two.pdb"
    two_chains.write_text(
      "".join(src) + "TER\n" + "".join(line[:21] + "B" + line[22:] for line in src) + "END\n"
    )
    protein_b = next(iter(load_rust(str(two_chains), chain_id="B")))
    assert protein_b.chain_ids == ["B"]
    # A chain that isn't there used to return the unfiltered structure.
    # (load_rust wraps the ValueError in its ParsingError.)
    from proxide.io.parsing.registry import ParsingError

    with pytest.raises(ParsingError, match=r"chain\(s\) \['Z'\] not in structure"):
      next(iter(load_rust(str(two_chains), chain_id="Z")))
    lines = _atom_lines(write_pdb(protein_b, tmp_path / "b.pdb"))
    assert len(lines) == len(src)
    assert {line[21] for line in lines} == {"B"}

  @staticmethod
  def _atom_key(line: str) -> tuple:
    # name, residue, chain, number, x, element -- everything write_pdb takes
    # from the structure (occupancy/B-factor are not carried, see debt).
    return (
      line[12:16].strip(), line[17:20], line[21], int(line[22:26]),
      round(float(line[30:38]), 2), line[76:78].strip(),
    )

  def test_full_format_round_trips_every_atom(self, tmp_path) -> None:
    # Backlog #5684: parse_structure's Full output now carries per-atom
    # elements, residue names and chain ids. It used to carry none, and the
    # writer emitted every residue as "UNK" in chain "A" (then, after debt
    # #1928, refused). Every source atom must come back identical.
    from proxide import CoordFormat, OutputSpec, parse_structure

    protein = parse_structure(self.PDB, OutputSpec(coord_format=CoordFormat.Full))
    written = _atom_lines(write_pdb(protein, tmp_path / "out.pdb"))
    source = [x for x in open(self.PDB) if x.startswith(("ATOM", "HETATM"))]
    assert sorted(map(self._atom_key, written)) == sorted(map(self._atom_key, source))

  def test_full_format_round_trips_two_chains(self, tmp_path) -> None:
    from proxide import CoordFormat, OutputSpec, parse_structure

    src = [x for x in open(self.PDB) if x.startswith("ATOM")]
    b_shifted = [
      x[:21] + "B" + x[22:30] + f"{float(x[30:38]) + 100.0:8.3f}" + x[38:] for x in src
    ]
    two = tmp_path / "two.pdb"
    two.write_text("".join(b_shifted) + "TER\n" + "".join(src) + "END\n")
    protein = parse_structure(str(two), OutputSpec(coord_format=CoordFormat.Full))
    written = _atom_lines(write_pdb(protein, tmp_path / "out.pdb"))
    assert sorted(map(self._atom_key, written)) == sorted(map(self._atom_key, b_shifted + src))

  def test_full_format_elements_come_from_the_parser_not_the_name(self, tmp_path) -> None:
    # Debt #2353: elements reach Python from the parser. A calcium ion named
    # "CA" (element column "CA") must stay calcium -- a first-letter rule in
    # Python would make it carbon.
    from proxide import CoordFormat, OutputSpec, parse_structure

    src = [x for x in open(self.PDB) if x.startswith("ATOM")]
    ca = "HETATM 9999 CA    CA B 900      30.000  30.000  30.000  1.00  0.00          CA\n"
    path = tmp_path / "with_ion.pdb"
    path.write_text("".join(src) + "TER\n" + ca + "END\n")
    protein = parse_structure(
      str(path), OutputSpec(coord_format=CoordFormat.Full, include_hetatm=True)
    )
    assert protein.elements is not None
    names = list(protein.atom_names)
    ion = [i for i, (n, r) in enumerate(zip(names, protein.res_names)) if r == "CA"]
    assert len(ion) == 1, (names[-3:], list(protein.res_names)[-3:])
    # The PDB reader keeps the element column's case as written ("CA"; see
    # the element-case debt). What matters here: calcium, not carbon.
    assert protein.elements[ion[0]].upper() == "CA"
