"""PDB/mmCIF writer options: the record rule, insertion codes, and goldens (spec 261123, T4).

Covers ``write_pdb``/``write_mmcif`` with ``record_type`` and ``insertion_codes``:

- every row of the record-rule table (spec D3), parametrised over both writers;
- PDB and mmCIF round trips of the fixture ``tests/data/icode_ligand.pdb`` (T1) through
  proxide's own readers, in Full and Atom37;
- every option value, an unknown value, ``hetero`` on Atom37, and invalid insertion codes;
- the byte goldens in ``tests/data/golden_writer`` (never regenerated here): (a) 5awl Atom37,
  (b) 5awl Atom14, (c) the hand-built Full protein, byte-identical with default options; (d) 5awl
  Full, which differs only in the record type of its 12 HOH rows under the default options and is
  byte-identical under ``record_type="residue_table"``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from proxide import CoordFormat, OutputSpec, parse_structure
from proxide.chem.residues import atom_order, resnames
from proxide.core.containers import Protein
from proxide.io.writing import write_mmcif, write_pdb

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "tests" / "data"
GOLDEN = DATA / "golden_writer"
FIXTURE = DATA / "icode_ligand.pdb"
GENERATOR = REPO / "scripts" / "golden" / "make_writer_goldens.py"

WRITERS = {"pdb": write_pdb, "cif": write_mmcif}
RECORD_TYPES = ("source", "residue_table")
INSERTION_CODE_OPTIONS = ("write", "blank")

# Residues of the T1 fixture, Atom37 order: 51, 52, 52A, 52B, MSE 53, 54, chain B 1 and 2.
T1_ATOM37_CODES = [" ", " ", "A", "B", " ", " ", " ", " "]
T1_ATOM37_RESNAMES = ["ALA", "GLY", "SER", "THR", "MSE", "ALA", "ALA", "GLY"]


def _generator():
  spec = importlib.util.spec_from_file_location("make_writer_goldens", GENERATOR)
  assert spec is not None and spec.loader is not None
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


def _parse(path: Path, layout: CoordFormat) -> Protein:
  return parse_structure(path, OutputSpec(coord_format=layout), use_jax=False)


@pytest.fixture(scope="module")
def t1_full() -> Protein:
  return _parse(FIXTURE, CoordFormat.Full)


@pytest.fixture(scope="module")
def t1_atom37() -> Protein:
  return _parse(FIXTURE, CoordFormat.Atom37)


def _full(res_names: list[str], hetero=None) -> Protein:
  """A hand-built Full Protein with one atom per entry of ``res_names``."""
  n = len(res_names)
  return Protein(
    coordinates=np.zeros((n, 37, 3), dtype=np.float32),
    aatype=np.zeros(n, dtype=np.int8),
    residue_index=np.arange(n, dtype=np.int32),
    chain_index=np.zeros(n, dtype=np.int32),
    chain_ids=["A"],
    full_coordinates=np.stack([np.arange(n, dtype=np.float32) * 1.5, np.zeros(n), np.zeros(n)], -1),
    atom_names=["CA"] * n,
    res_names=list(res_names),
    elements=["C"] * n,
    hetero=None if hetero is None else np.asarray(hetero, dtype=np.bool_),
  )


def _atom37(res_names: list[str]) -> Protein:
  """A hand-built Atom37 Protein, one resolved CA per residue, named by ``res_names``."""
  n = len(res_names)
  mask = np.zeros((n, 37), dtype=np.float32)
  mask[:, atom_order["CA"]] = 1.0
  coords = np.zeros((n, 37, 3), dtype=np.float32)
  coords[:, atom_order["CA"], 0] = np.arange(n) * 1.5
  aatype = np.array([resnames.index("MET" if r == "MSE" else r) for r in res_names], np.int8)
  return Protein(
    coordinates=coords,
    aatype=aatype,
    residue_index=np.arange(1, n + 1, dtype=np.int32),
    chain_index=np.zeros(n, dtype=np.int32),
    chain_ids=["A"],
    atom_mask=mask,
    residue_names=list(res_names),
  )


def _record_of(line: str, writer: str) -> str:
  """The record type (ATOM or HETATM) of one written atom row, for either format."""
  return line.split()[0] if writer == "cif" else line[:6].strip()


def _atom_rows(text: str, writer: str) -> list[str]:
  """Atom rows of a written file, without any header or footer lines."""
  if writer == "pdb":
    return [ln for ln in text.splitlines() if ln.startswith(("ATOM", "HETATM"))]
  return [ln for ln in text.splitlines() if ln.startswith(("ATOM ", "HETATM "))]


# (layout, record_type, res_name, hetero flag, expected record): every row of the D3 table, with
# the residue-table cases (MSE in and ALA out of the table) and the hetero values each row allows.
RECORD_RULE_ROWS = [
  ("Full", "source", "MSE", None, "HETATM"),
  ("Full", "source", "ALA", None, "ATOM"),
  ("Full", "source", "LIG", True, "HETATM"),
  ("Full", "source", "MSE", True, "HETATM"),
  ("Full", "source", "ALA", True, "HETATM"),
  ("Full", "source", "LIG", False, "ATOM"),
  ("Full", "source", "MSE", False, "HETATM"),
  ("Full", "residue_table", "MSE", None, "HETATM"),
  ("Full", "residue_table", "ALA", None, "ATOM"),
  ("Full", "residue_table", "LIG", True, "ATOM"),
  ("Full", "residue_table", "MSE", False, "HETATM"),
  ("Atom37", "source", "ALA", None, "ATOM"),
  ("Atom37", "source", "MSE", None, "HETATM"),
  ("Atom37", "residue_table", "ALA", None, "ATOM"),
  ("Atom37", "residue_table", "MSE", None, "HETATM"),
]


@pytest.mark.parametrize("writer", ["pdb", "cif"])
@pytest.mark.parametrize(("layout", "record_type", "res_name", "hetero", "expected"), RECORD_RULE_ROWS)
def test_record_rule_table_row(
  tmp_path: Path, writer: str, layout: str, record_type: str, res_name: str, hetero, expected: str
) -> None:
  if layout == "Full":
    protein = _full([res_name, "GLY"], hetero=None if hetero is None else [hetero, False])
  else:
    protein = _atom37([res_name, "GLY"])
  out = WRITERS[writer](protein, tmp_path / f"row.{writer}", record_type=record_type)
  rows = _atom_rows(out.read_text(), writer)
  assert len(rows) == 2
  assert _record_of(rows[0], writer) == expected, rows[0]
  # The second residue (GLY, no hetero flag) follows the table rule: always ATOM.
  assert _record_of(rows[1], writer) == "ATOM", rows[1]


@pytest.mark.parametrize("writer", ["pdb", "cif"])
def test_full_source_hetero_true_on_ligand_is_hetatm_and_residue_table_is_not(
  tmp_path: Path, writer: str
) -> None:
  protein = _full(["LIG"], hetero=[True])
  source = WRITERS[writer](protein, tmp_path / "s.cif").read_text()
  table = WRITERS[writer](protein, tmp_path / "t.cif", record_type="residue_table").read_text()
  assert _record_of(_atom_rows(source, writer)[0], writer) == "HETATM"
  assert _record_of(_atom_rows(table, writer)[0], writer) == "ATOM"


# --- PDB and mmCIF round trips of the T1 fixture -------------------------------------------------


def test_pdb_round_trip_full_keeps_record_types_icodes_and_ligand_atoms(tmp_path: Path, t1_full) -> None:
  out = write_pdb(t1_full, tmp_path / "t1_full.pdb")
  rows = _atom_rows(out.read_text(), "pdb")
  assert len(rows) == len(t1_full.atom_names)
  # Record types follow the parsed per-atom hetero flag: ligand, ion, waters and MSE are HETATM.
  assert [r[:6].strip() for r in rows] == ["HETATM" if h else "ATOM" for h in t1_full.hetero]
  assert any(r[17:20] == "LIG" and r.startswith("HETATM") for r in rows)
  assert all(r.startswith("HETATM") for r in rows if r[17:20] in ("LIG", "HOH", " ZN"))
  # Column 27 (index 26) carries the per-atom insertion code.
  assert [r[26] for r in rows] == list(t1_full.insertion_codes)

  back = _parse(out, CoordFormat.Full)
  assert back.insertion_codes == t1_full.insertion_codes
  assert list(back.hetero) == list(t1_full.hetero)
  assert list(back.res_names) == list(t1_full.res_names)
  assert list(back.atom_names) == list(t1_full.atom_names)


def test_pdb_round_trip_atom37_keeps_icodes_and_mse_hetatm(tmp_path: Path, t1_atom37) -> None:
  out = write_pdb(t1_atom37, tmp_path / "t1_atom37.pdb")
  rows = _atom_rows(out.read_text(), "pdb")
  mse_rows = [r for r in rows if r[17:20] == "MSE"]
  assert mse_rows and all(r.startswith("HETATM") for r in mse_rows)
  assert not any(r.startswith("HETATM") and r[17:20] != "MSE" for r in rows)
  # One code per residue, broadcast to that residue's atom rows.
  assert sorted({r[26] for r in rows}) == [" ", "A", "B"]
  assert t1_atom37.insertion_codes == T1_ATOM37_CODES

  back = _parse(out, CoordFormat.Atom37)
  assert back.insertion_codes == T1_ATOM37_CODES
  assert list(back.residue_names) == T1_ATOM37_RESNAMES


def test_cif_round_trip_full_keeps_52_52a_52b_distinct(tmp_path: Path, t1_full) -> None:
  out = write_mmcif(t1_full, tmp_path / "t1_full.cif")
  text = out.read_text()
  assert "_atom_site.pdbx_PDB_ins_code" in text
  back = _parse(out, CoordFormat.Full)
  assert back.insertion_codes == t1_full.insertion_codes
  assert list(back.hetero) == list(t1_full.hetero)
  codes_of_52 = {c for r, c in zip(back.atom_residue_ids, back.insertion_codes, strict=True) if r == 52}
  assert codes_of_52 == {" ", "A", "B"}


def test_cif_round_trip_atom37_keeps_52_52a_52b_distinct(tmp_path: Path, t1_atom37) -> None:
  out = write_mmcif(t1_atom37, tmp_path / "t1_atom37.cif")
  text = out.read_text()
  assert "_atom_site.pdbx_PDB_ins_code" in text
  assert any(ln.startswith("HETATM ") and " MSE " in ln for ln in text.splitlines())
  back = _parse(out, CoordFormat.Atom37)
  assert back.insertion_codes == T1_ATOM37_CODES
  assert list(back.residue_index).count(52) == 3


# --- Options ------------------------------------------------------------------------------------


@pytest.mark.parametrize("writer", ["pdb", "cif"])
@pytest.mark.parametrize("record_type", RECORD_TYPES)
@pytest.mark.parametrize("insertion_codes", INSERTION_CODE_OPTIONS)
def test_every_option_value_is_accepted(tmp_path: Path, writer, record_type, insertion_codes, t1_full) -> None:
  out = WRITERS[writer](
    t1_full, tmp_path / f"o.{writer}", record_type=record_type, insertion_codes=insertion_codes
  )
  assert out.exists()


@pytest.mark.parametrize("writer", ["pdb", "cif"])
def test_blank_insertion_codes_writes_column_27_blank_and_no_mmcif_column(
  tmp_path: Path, writer: str, t1_full
) -> None:
  out = WRITERS[writer](t1_full, tmp_path / f"b.{writer}", insertion_codes="blank")
  text = out.read_text()
  if writer == "pdb":
    assert all(r[26] == " " for r in _atom_rows(text, "pdb"))
  else:
    assert "pdbx_PDB_ins_code" not in text


@pytest.mark.parametrize("writer", ["pdb", "cif"])
@pytest.mark.parametrize(("option", "value", "accepted"), [
  ("record_type", "HETATM", "'source'"),
  ("record_type", "", "'residue_table'"),
  ("insertion_codes", "keep", "'write'"),
  ("insertion_codes", "True", "'blank'"),
])
def test_unknown_option_value_raises_and_names_the_accepted_values(
  tmp_path: Path, writer: str, option: str, value: str, accepted: str, t1_full
) -> None:
  target = tmp_path / f"u.{writer}"
  with pytest.raises(ValueError, match=f"{option}={value!r} is not accepted") as info:
    WRITERS[writer](t1_full, target, **{option: value})
  assert accepted in str(info.value)
  assert not target.exists()


def test_options_are_keyword_only(tmp_path: Path, t1_full) -> None:
  with pytest.raises(TypeError):
    write_pdb(t1_full, tmp_path / "x.pdb", "source")  # type: ignore[misc]


@pytest.mark.parametrize("writer", ["pdb", "cif"])
def test_hetero_on_atom37_raises(tmp_path: Path, writer: str, t1_atom37) -> None:
  protein = t1_atom37.replace(hetero=np.zeros(8, dtype=np.bool_))
  with pytest.raises(ValueError, match="hetero is set on an Atom37/Atom14"):
    WRITERS[writer](protein, tmp_path / f"h.{writer}")


@pytest.mark.parametrize("writer", ["pdb", "cif"])
@pytest.mark.parametrize("bad_code", ["AB", "é", ""])
def test_invalid_full_insertion_code_raises(tmp_path: Path, writer: str, bad_code: str, t1_full) -> None:
  codes = list(t1_full.insertion_codes)
  codes[2] = bad_code
  with pytest.raises(ValueError, match="insertion_codes"):
    WRITERS[writer](t1_full.replace(insertion_codes=codes), tmp_path / f"i.{writer}")


@pytest.mark.parametrize("writer", ["pdb", "cif"])
@pytest.mark.parametrize("bad_code", ["AB", "é"])
def test_invalid_atom37_insertion_code_raises(tmp_path: Path, writer: str, bad_code: str, t1_atom37) -> None:
  codes = list(T1_ATOM37_CODES)
  codes[2] = bad_code
  with pytest.raises(ValueError, match="insertion_codes"):
    WRITERS[writer](t1_atom37.replace(insertion_codes=codes), tmp_path / f"i.{writer}")


# --- Goldens ------------------------------------------------------------------------------------


def _golden_bytes(name: str) -> bytes:
  return (GOLDEN / name).read_bytes()


def _write_bytes(tmp_path: Path, name: str, protein: Protein, writer: str, **options) -> bytes:
  # mmCIF embeds data_<stem>, so the file is written under the golden's exact name.
  out = WRITERS[writer](protein, tmp_path / name, **options)
  return out.read_bytes()


@pytest.fixture(scope="module")
def gen():
  return _generator()


@pytest.fixture(scope="module")
def golden_proteins(gen) -> dict[str, Protein]:
  return {src: gen.load(src) for src in ("5awl:Atom37", "5awl:Atom14", "5awl:Full")}


def test_golden_a_5awl_atom37_pdb_and_cif_are_byte_identical(tmp_path: Path, golden_proteins) -> None:
  protein = golden_proteins["5awl:Atom37"]
  assert _write_bytes(tmp_path, "5awl_atom37.pdb", protein, "pdb") == _golden_bytes("5awl_atom37.pdb")
  assert _write_bytes(tmp_path, "5awl_atom37.cif", protein, "cif") == _golden_bytes("5awl_atom37.cif")


def test_golden_b_5awl_atom14_pdb_is_byte_identical(tmp_path: Path, golden_proteins) -> None:
  protein = golden_proteins["5awl:Atom14"]
  assert _write_bytes(tmp_path, "5awl_atom14.pdb", protein, "pdb") == _golden_bytes("5awl_atom14.pdb")


def test_golden_c_handbuilt_full_pdb_and_cif_are_byte_identical(tmp_path: Path, gen) -> None:
  protein = gen.handbuilt_full_protein()
  assert _write_bytes(tmp_path, "handbuilt_full.pdb", protein, "pdb") == _golden_bytes("handbuilt_full.pdb")
  assert _write_bytes(tmp_path, "handbuilt_full.cif", protein, "cif") == _golden_bytes("handbuilt_full.cif")


def _changed_line_indices(gold: list[str], new: list[str]) -> list[int]:
  assert len(gold) == len(new), f"line count changed: {len(gold)} -> {len(new)}"
  return [i for i, (g, n) in enumerate(zip(gold, new, strict=True)) if g != n]


def test_golden_d_5awl_full_pdb_differs_only_in_the_hoh_record_type(tmp_path: Path, golden_proteins) -> None:
  gold = _golden_bytes("5awl_full.pdb").decode().splitlines(keepends=True)
  new = _write_bytes(tmp_path, "5awl_full.pdb", golden_proteins["5awl:Full"], "pdb").decode().splitlines(
    keepends=True
  )
  changed = _changed_line_indices(gold, new)
  hoh_rows = [i for i, g in enumerate(gold) if g.startswith("ATOM") and g[17:20] == "HOH"]
  assert changed == hoh_rows
  assert len(changed) == 12
  for i in changed:
    assert gold[i].startswith("ATOM") and new[i].startswith("HETATM")
    assert new[i][6:] == gold[i][6:], "only the record type may change"


def test_golden_d_5awl_full_cif_differs_only_in_the_hoh_group_pdb(tmp_path: Path, golden_proteins) -> None:
  gold = _golden_bytes("5awl_full.cif").decode().splitlines(keepends=True)
  new = _write_bytes(tmp_path, "5awl_full.cif", golden_proteins["5awl:Full"], "cif").decode().splitlines(
    keepends=True
  )
  changed = _changed_line_indices(gold, new)
  hoh_rows = [i for i, g in enumerate(gold) if g.startswith("ATOM ") and g.split()[4] == "HOH"]
  assert changed == hoh_rows
  assert len(changed) == 12
  for i in changed:
    assert gold[i].startswith("ATOM ") and new[i].startswith("HETATM ")
    assert new[i][len("HETATM"):] == gold[i][len("ATOM"):], "only group_PDB may change"


def test_golden_d_residue_table_reproduces_both_goldens_byte_for_byte(tmp_path: Path, golden_proteins) -> None:
  protein = golden_proteins["5awl:Full"]
  assert (
    _write_bytes(tmp_path, "5awl_full.pdb", protein, "pdb", record_type="residue_table")
    == _golden_bytes("5awl_full.pdb")
  )
  assert (
    _write_bytes(tmp_path, "5awl_full.cif", protein, "cif", record_type="residue_table")
    == _golden_bytes("5awl_full.cif")
  )


# --- special one-character codes (T4 review) ----------------------------------------------------


@pytest.mark.parametrize("layout", [CoordFormat.Full, CoordFormat.Atom37])
@pytest.mark.parametrize("writer", ["pdb", "cif"])
@pytest.mark.parametrize("sym", ["'", "?", "."])
def test_cif_special_insertion_codes_round_trip(
  tmp_path: Path, layout: CoordFormat, writer: str, sym: str
) -> None:
  # "'" needs a double-quoted CIF token; "?" and "." must stay literal codes, not read back as
  # CIF unknown/inapplicable (blank).
  protein = _parse(FIXTURE, layout)
  codes = [sym if c == "A" else c for c in protein.insertion_codes]
  out = WRITERS[writer](protein.replace(insertion_codes=codes), tmp_path / f"out.{writer}")
  assert list(_parse(Path(out), layout).insertion_codes) == codes


@pytest.mark.parametrize("writer", ["pdb", "cif"])
def test_invalid_code_raises_even_with_blank_option_and_leaves_no_file(
  tmp_path: Path, writer: str, t1_atom37
) -> None:
  codes = ["AB" if c == "A" else c for c in t1_atom37.insertion_codes]
  target = tmp_path / f"out.{writer}"
  with pytest.raises(ValueError):
    WRITERS[writer](t1_atom37.replace(insertion_codes=codes), target, insertion_codes="blank")
  assert not target.exists()
