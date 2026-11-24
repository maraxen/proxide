"""Full-atom PDB writer acceptance tests, ported from aminx's LASEr PDB writer tests (spec 261123, T6).

Ported from aminx commit 095a49e0, tests/io/test_laser_pdb.py (a local-only tag). Each original
assertion this file keeps, or adapts, is quoted verbatim in the docstring of the test it became.

Four aminx tests are ported, by logic:

- test_heavy_layout_roundtrip
- test_protonated_layout_writes_non_nan_hydrogens
- test_identifiers_and_bfactors_use_pdb_columns
- test_ligand_passes_through_as_hetatm

Dropped, aminx-only (they test aminx code, not the writer): test_docstring_leaves_hydrogen_cleanup_open
(it reads an aminx module docstring) and test_featurizer_identifiers_reach_the_sample_helper (it runs
aminx's featurizer and sample helper).

Adaptations, all on proxide's Full path (``write_pdb`` on a Protein built directly, F7):

- Each Protein sets ``hetero`` and ``insertion_codes`` explicitly, per atom.
- aminx's NaN-padded absent atoms are not in the input: they are dropped before building (F8).
- aminx's padding widths (MAX_ATOMS, MAX_PROTONATED_ATOMS) do not exist here: one row per atom.
- prody is replaced by ``_columns``, which reads the PDB columns by position. The round trip uses
  proxide's ``parse_structure``.
- ``LASER_ALPHABET`` is copied from aminx ``featurize.py:22`` (aminx is not imported).
- Two-letter element names: proxide starts the name in column 13 (``"CL  "``) and writes the
  element upper-cased (``"CL"``). aminx wrote ``" CL "`` and ``"Cl"``. Both are asserted as proxide's
  convention, with the aminx value noted.
- An omitted B-factor is written blank, not ``0.00`` (``_pdb_occupancy_and_b_factor``, debt #2367).
- The segment ID (columns 73-76) is not ported: a Protein has no segment field, and the writer
  leaves those columns blank.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from proxide import CoordFormat, OutputSpec, parse_structure
from proxide.core.containers import Protein
from proxide.io.writing import write_pdb, write_pdb_reported

# aminx featurize.py:22 (copied; the order is the dataset's residue order).
LASER_ALPHABET = "ARNDCEQGHILKMFPSTWYVX"

# Upstream ``dataset_atom_order`` (heavy) and the hydrogens appended in
# ``hydrogen_extended_dataset_atom_order``. ``X`` follows alanine.
_HEAVY: dict[str, tuple[str, ...]] = {
  "G": ("N", "CA", "C", "O"),
  "X": ("N", "CA", "C", "O"),
  "A": ("N", "CA", "C", "O", "CB"),
  "S": ("N", "CA", "C", "O", "CB", "OG", "HG"),
  "C": ("N", "CA", "C", "O", "CB", "SG", "HG"),
  "T": ("N", "CA", "C", "O", "CB", "OG1", "CG2", "HG1"),
  "P": ("N", "CA", "C", "O", "CB", "CG", "CD"),
  "V": ("N", "CA", "C", "O", "CB", "CG1", "CG2"),
  "M": ("N", "CA", "C", "O", "CB", "CG", "SD", "CE"),
  "N": ("N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"),
  "I": ("N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"),
  "L": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"),
  "D": ("N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"),
  "E": ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"),
  "K": ("N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"),
  "Q": ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"),
  "H": ("N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2", "HD1", "HE2"),
  "F": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
  "R": ("N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"),
  "Y": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH", "HH"),
  "W": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE2", "CE3", "NE1", "CZ2", "CZ3", "CH2"),
}
_EXTRA_H: dict[str, tuple[str, ...]] = {
  "G": ("HA2", "HA3"),
  "A": ("HA", "HB1", "HB2", "HB3"),
  "S": ("HA", "HB2", "HB3"),
  "C": ("HA", "HB2", "HB3"),
  "T": ("HA", "HB", "HG21", "HG22", "HG23"),
  "P": ("HA", "HB2", "HB3", "HG2", "HG3", "HD2", "HD3"),
  "V": ("HA", "HB", "HG11", "HG12", "HG13", "HG21", "HG22", "HG23"),
  "M": ("HA", "HB2", "HB3", "HG2", "HG3", "HE1", "HE2", "HE3"),
  "N": ("HA", "HB2", "HB3", "HD21", "HD22"),
  "I": ("HA", "HB", "HG21", "HG22", "HG23", "HD11", "HD12", "HD13", "HG12", "HG13"),
  "L": ("HA", "HB2", "HB3", "HG", "HD11", "HD12", "HD13", "HD21", "HD22", "HD23"),
  "D": ("HA", "HB2", "HB3"),
  "E": ("HA", "HB2", "HB3", "HG2", "HG3"),
  "K": ("HA", "HB2", "HB3", "HG2", "HG3", "HD2", "HD3", "HE2", "HE3", "HZ1", "HZ2", "HZ3"),
  "Q": ("HA", "HB2", "HB3", "HG2", "HG3", "HE21", "HE22"),
  "H": ("HA", "HB2", "HB3", "HE1", "HD2"),
  "F": ("HA", "HB2", "HB3", "HD1", "HD2", "HE1", "HE2", "HZ"),
  "R": ("HA", "HB2", "HB3", "HG2", "HG3", "HD2", "HD3", "HE", "HH21", "HH22", "HH12", "HH11"),
  "Y": ("HA", "HB2", "HB3", "HD1", "HD2", "HE1", "HE2"),
  "W": ("HA", "HB2", "HB3", "HD1", "HE1", "HE3", "HZ2", "HZ3", "HH2"),
  "X": ("HA", "HB1", "HB2", "HB3"),
}
_RESNAME = {
  "A": "ALA",
  "R": "ARG",
  "N": "ASN",
  "D": "ASP",
  "C": "CYS",
  "E": "GLU",
  "Q": "GLN",
  "G": "GLY",
  "H": "HIS",
  "I": "ILE",
  "L": "LEU",
  "K": "LYS",
  "M": "MET",
  "F": "PHE",
  "P": "PRO",
  "S": "SER",
  "T": "THR",
  "W": "TRP",
  "Y": "TYR",
  "V": "VAL",
  "X": "XAA",
}


@dataclass(frozen=True, slots=True)
class _Row:
  """One input atom of the Full Protein built for a test."""

  chain: str
  resseq: int
  icode: str
  resname: str
  name: str
  element: str
  xyz: tuple[float, float, float]
  b: float
  occ: float


def _atom_lines(text: str) -> list[str]:
  return [line for line in text.splitlines() if line[:6].strip() in {"ATOM", "HETATM"}]


def _columns(line: str) -> dict[str, Any]:
  """The PDB fields of one atom record, read by 1-based column position (PDB v3.3)."""
  return {
    "record": line[0:6].strip(),
    "name": line[12:16].strip(),
    "resname": line[17:20].strip(),
    "chain": line[21].strip(),
    "resseq": int(line[22:26]),
    "icode": line[26],
    "xyz": (float(line[30:38]), float(line[38:46]), float(line[46:54])),
    "occ": float(line[54:60]),
    "b": float(line[60:66]),
    "element": line[76:78].strip(),
  }


def _slot_xyz(residue: int, slot: int) -> tuple[float, float, float]:
  return (
    (residue + 1) * 0.5 + slot * 0.125,
    slot * -0.25,
    residue * 0.25 - slot * 0.125,
  )


def _full(rows: list[_Row], *, hetero: list[bool] | None, with_b: bool = True) -> Protein:
  """A hand-built Full Protein, one entry per row; hetero and insertion codes are set explicitly."""
  n = len(rows)
  chain_ids = sorted({r.chain for r in rows})
  return Protein(
    coordinates=np.zeros((n, 37, 3), dtype=np.float32),
    aatype=np.zeros(n, dtype=np.int8),
    residue_index=np.array([r.resseq for r in rows], dtype=np.int32),
    chain_index=np.array([chain_ids.index(r.chain) for r in rows], dtype=np.int32),
    chain_ids=chain_ids,
    full_coordinates=np.array([r.xyz for r in rows], dtype=np.float32),
    atom_names=[r.name for r in rows],
    res_names=[r.resname for r in rows],
    elements=[r.element for r in rows],
    atom_chain_ids=[r.chain for r in rows],
    atom_res_index=np.array([r.resseq for r in rows], dtype=np.int32),
    b_factors=np.array([r.b for r in rows], dtype=np.float32) if with_b else None,
    occupancies=np.array([r.occ for r in rows], dtype=np.float32),
    insertion_codes=[r.icode for r in rows],
    hetero=None if hetero is None else np.asarray(hetero, dtype=np.bool_),
  )


def _row(
  chain: str,
  resseq: int,
  icode: str,
  resname: str,
  name: str,
  xyz: tuple[float, float, float],
  *,
  b: float = 0.0,
  occ: float = 1.0,
  element: str | None = None,
) -> _Row:
  return _Row(
    chain=chain,
    resseq=resseq,
    icode=icode,
    resname=resname,
    name=name,
    element=name[0] if element is None else element,
    xyz=xyz,
    b=b,
    occ=occ,
  )


def test_heavy_layout_roundtrip(tmp_path: Path) -> None:
  """Heavy atoms round-trip to 3 decimals and follow dataset_atom_order.

  Original aminx assertions (095a49e0, verbatim), with the adaptation noted:

    assert [(atom.name, atom.resname) for atom in atoms] == [
      (name, resname) for name, resname, _xyz in expected
    ]
    assert atom.xyz == pytest.approx(tuple(float(item) for item in xyz), abs=5e-4)
    assert atom.element == atom.name[0]
    assert atom.record == "ATOM"
    assert atom.bfactor == pytest.approx(0.0)
    assert tuple(by_residue[index + 1]) == _HEAVY[letter]   (for index, letter in the alphabet)
    assert tuple(by_residue[len(letters)]) == ("N", "CA", "C", "O")

  Adapted: ``atom.xyz`` / ``atom.bfactor`` / ``atom.record`` become ``_columns`` keys; the
  B-factor is set to 0.0 on the input (aminx wrote 0.0 when it was omitted). The aminx padded slot
  (residue 22, slot 4, CB) is absent from the input, so the last residue has four atoms (F8). The
  parse_structure round trip replaces aminx's prody cross-check.
  """
  letters = list(LASER_ALPHABET) + ["A"]
  rows: list[_Row] = []
  for resi, letter in enumerate(letters, start=1):
    for slot, name in enumerate(_HEAVY[letter]):
      if resi == len(letters) and slot == 4:  # aminx's dropped (CB of the final A)
        continue
      rows.append(_row("A", resi, " ", _RESNAME[letter], name, _slot_xyz(resi - 1, slot)))
  protein = _full(rows, hetero=[False] * len(rows))

  out = write_pdb(protein, tmp_path / "heavy.pdb")
  atoms = [_columns(line) for line in _atom_lines(out.read_text())]
  assert [(a["name"], a["resname"]) for a in atoms] == [(r.name, r.resname) for r in rows]
  for atom, row in zip(atoms, rows, strict=True):
    assert atom["xyz"] == pytest.approx(row.xyz, abs=5e-4)
    assert atom["element"] == atom["name"][0]
    assert atom["record"] == "ATOM"
    assert atom["b"] == pytest.approx(0.0)
  by_residue: dict[int, list[str]] = {}
  for atom in atoms:
    by_residue.setdefault(atom["resseq"], []).append(atom["name"])
  for index, letter in enumerate(LASER_ALPHABET):
    assert tuple(by_residue[index + 1]) == _HEAVY[letter]
  assert tuple(by_residue[len(letters)]) == ("N", "CA", "C", "O")

  # The same file through proxide's own reader.
  back = parse_structure(out, OutputSpec(coord_format=CoordFormat.Full), use_jax=False)
  assert list(back.atom_names) == [r.name for r in rows]
  # The written file says HIS (checked above, column by column). proxide's Full reader then names
  # a heavy-atom-only histidine HIP, a reader-side protonation label; only that rename is allowed.
  assert [("HIS" if n == "HIP" else n) for n in back.res_names] == [r.resname for r in rows]
  assert list(back.hetero) == [False] * len(rows)
  np.testing.assert_allclose(
    np.asarray(back.full_coordinates), np.array([r.xyz for r in rows]), atol=5e-4
  )


def test_protonated_layout_writes_non_nan_hydrogens(tmp_path: Path) -> None:
  """Protonated layout writes exactly the finite hydrogen-extended names.

  Original aminx assertions (095a49e0, verbatim):

    assert [atom.name for atom in atoms] == [name for name, _resname, _xyz in expected]
    assert "H" not in {atom.name for atom in atoms}
    written = [atom.name for atom in atoms if atom.resnum == tryptophan + 1]
    assert written == list(orders[tryptophan][:-1])
    assert len(atoms) == sum(len(names) for names in orders) - 1

  Adapted: the absent tryptophan hydrogen (its last slot, NaN in aminx) is not in the input, so
  the written atom count is the input count (F8). ``atom.resnum`` becomes ``resseq``. The writer
  raises on a NaN row, so a successful write is the non-NaN claim; the parsed rows are also checked
  finite.
  """
  letters = list(LASER_ALPHABET)
  orders = [_HEAVY[letter] + _EXTRA_H[letter] for letter in letters]
  tryptophan = LASER_ALPHABET.index("W")
  dropped = len(orders[tryptophan]) - 1
  rows: list[_Row] = []
  for residue, (letter, names) in enumerate(zip(letters, orders, strict=True)):
    for slot, name in enumerate(names):
      if (residue, slot) == (tryptophan, dropped):
        continue
      rows.append(_row("A", residue + 1, " ", _RESNAME[letter], name, _slot_xyz(residue, slot)))
  protein = _full(rows, hetero=[False] * len(rows))

  out = write_pdb(protein, tmp_path / "protonated.pdb")
  atoms = [_columns(line) for line in _atom_lines(out.read_text())]
  assert [atom["name"] for atom in atoms] == [r.name for r in rows]
  assert "H" not in {atom["name"] for atom in atoms}
  written = [atom["name"] for atom in atoms if atom["resseq"] == tryptophan + 1]
  assert written == list(orders[tryptophan][:-1])
  assert len(atoms) == sum(len(names) for names in orders) - 1
  assert all(np.isfinite(coord) for atom in atoms for coord in atom["xyz"])


def test_identifiers_and_bfactors_use_pdb_columns(tmp_path: Path) -> None:
  """Chain, residue number, insertion code, and B-factor sit in the PDB columns.

  Original aminx assertions (095a49e0, verbatim), with the adaptations noted:

    assert len(lines) == 9
    assert line[21] == "B"
    assert int(line[22:26]) == 42
    assert line[26] == "A"
    assert float(line[60:66]) == pytest.approx(0.5)
    assert line[17:20] == "ALA"
    assert line[76:78] == f"{line[12:16].strip()[0]:>2}"
    assert line[21] == "C"
    assert int(line[22:26]) == -3
    assert line[26] == " "
    assert float(line[60:66]) == pytest.approx(1.25)
    assert line[17:20] == "GLY"
    assert (atom.chain, atom.resnum, atom.icode) == ("B", 42, "A")
    assert atom.bfactor == pytest.approx(0.5)
    assert (atom.chain, atom.resnum, atom.icode) == ("C", -3, "")
    assert atom.bfactor == pytest.approx(1.25)
    for line in omitted.splitlines():
      if line.startswith("ATOM"):
        assert float(line[60:66]) == pytest.approx(0.0)

  Adapted: the insertion code "A" is the non-blank case in column 27 (the writer used to leave it
  blank, F3). The aminx omitted-B-factor assertion becomes a blank check: proxide writes an
  unmeasured B-factor blank (debt #2367). ``_columns`` reads the same columns as the aminx assertions.
  """
  coords_b = [(0.125, 0.25, 0.5), (1.5, 0.0, 0.0), (2.5, 0.5, 0.0), (2.25, 1.5, 0.0), (1.5, -1.25, 0.5)]
  coords_c = [(4.0, 0.0, 0.0), (5.25, 0.5, 0.0), (6.5, 0.0, 0.25), (6.75, -1.0, 0.5)]
  rows = [
    _row("B", 42, "A", "ALA", name, xyz, b=0.5)
    for name, xyz in zip(_HEAVY["A"], coords_b, strict=True)
  ] + [
    _row("C", -3, " ", "GLY", name, xyz, b=1.25)
    for name, xyz in zip(_HEAVY["G"], coords_c, strict=True)
  ]
  protein = _full(rows, hetero=[False] * len(rows))

  text = write_pdb(protein, tmp_path / "ids.pdb").read_text()
  lines = _atom_lines(text)
  assert len(lines) == 9
  for line in lines[:5]:
    assert line[21] == "B"
    assert int(line[22:26]) == 42
    assert line[26] == "A"
    assert float(line[60:66]) == pytest.approx(0.5)
    assert line[17:20] == "ALA"
    assert line[76:78] == f"{line[12:16].strip()[0]:>2}"
  for line in lines[5:]:
    assert line[21] == "C"
    assert int(line[22:26]) == -3
    assert line[26] == " "
    assert float(line[60:66]) == pytest.approx(1.25)
    assert line[17:20] == "GLY"
  atoms = [_columns(line) for line in lines]
  for atom in atoms[:5]:
    assert (atom["chain"], atom["resseq"], atom["icode"]) == ("B", 42, "A")
    assert atom["b"] == pytest.approx(0.5)
  for atom in atoms[5:]:
    assert (atom["chain"], atom["resseq"], atom["icode"]) == ("C", -3, " ")
    assert atom["b"] == pytest.approx(1.25)

  # Omitted B-factors: the Protein carries none, so columns 61-66 are blank, not 0.00.
  omitted = write_pdb(_full(rows, hetero=[False] * len(rows), with_b=False), tmp_path / "nob.pdb")
  for line in _atom_lines(omitted.read_text()):
    assert line[60:66] == " " * 6


def test_ligand_passes_through_as_hetatm(tmp_path: Path) -> None:
  """Ligand atoms are written as HETATM with the caller's names and columns.

  Original aminx assertions (095a49e0, verbatim), with the adaptations noted:

    assert [atom.record for atom in atoms] == ["ATOM", "ATOM", "ATOM", "ATOM", "HETATM", "HETATM"]
    assert [(atom.name, atom.element, atom.resname) for atom in het] == [
      ("C1", "C", "LIG"),
      ("CL", "Cl", "LIG"),
    ]
    assert het[0].xyz == pytest.approx((8.5, 1.25, -0.5), abs=5e-4)
    assert het[1].xyz == pytest.approx((9.0, 2.5, 0.75), abs=5e-4)
    assert (het[0].chain, het[0].resnum, het[0].icode) == ("B", 7, "")
    assert (het[1].chain, het[1].resnum, het[1].icode) == ("B", 7, "A")
    assert het[0].occupancy == pytest.approx(0.8)
    assert het[0].bfactor == pytest.approx(11.5)
    assert het[1].bfactor == pytest.approx(2.25)
    assert raw[0][12:16] == " C1 "
    assert raw[0][76:78] == " C"
    assert raw[1][12:16] == " CL "
    assert raw[1][76:78] == "Cl"
    assert raw[0][72:76] == "  LG"

  Adapted: the element is "CL" (proxide upper-cases the element column) and the two-letter name
  "CL" starts in column 13 (``"CL  "``), per proxide's column rule, not aminx's " CL ". The segment
  ID assertion (``raw[0][72:76]``) is dropped (see the module docstring). The ligand is marked with
  an explicit ``hetero`` flag; the table-rule case (``hetero=None``) is asserted as ATOM, and the
  unflagged protein rows are ATOM.
  """
  rows = [
    _row("A", 1, " ", "GLY", name, xyz)
    for name, xyz in zip(
      _HEAVY["G"],
      [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (2.5, 1.0, 0.0), (2.25, 2.0, 0.25)],
      strict=True,
    )
  ]
  rows += [
    _row("B", 7, " ", "LIG", "C1", (8.5, 1.25, -0.5), b=11.5, occ=0.8, element="C"),
    _row("B", 7, "A", "LIG", "CL", (9.0, 2.5, 0.75), b=2.25, occ=1.0, element="Cl"),
  ]
  hetero = [False] * 4 + [True, True]

  # The table rule with hetero=None: a non-table residue is ATOM. That is reported as the info
  # finding PROX-WRITE-HETERO-UNKNOWN, which never warns (spec D4).
  with warnings.catch_warnings():
    warnings.simplefilter("error")
    report = write_pdb_reported(_full(rows, hetero=None), tmp_path / "table.pdb")
  assert [_columns(ln)["record"] for ln in _atom_lines(report.path.read_text())] == ["ATOM"] * 6
  assert "PROX-WRITE-HETERO-UNKNOWN" in [f.code for f in report.findings]

  out = write_pdb(_full(rows, hetero=hetero), tmp_path / "ligand.pdb")
  lines = _atom_lines(out.read_text())
  atoms = [_columns(line) for line in lines]
  assert [atom["record"] for atom in atoms] == ["ATOM", "ATOM", "ATOM", "ATOM", "HETATM", "HETATM"]
  het = atoms[4:]
  assert [(atom["name"], atom["element"], atom["resname"]) for atom in het] == [
    ("C1", "C", "LIG"),
    ("CL", "CL", "LIG"),
  ]
  assert het[0]["xyz"] == pytest.approx((8.5, 1.25, -0.5), abs=5e-4)
  assert het[1]["xyz"] == pytest.approx((9.0, 2.5, 0.75), abs=5e-4)
  assert (het[0]["chain"], het[0]["resseq"], het[0]["icode"]) == ("B", 7, " ")
  assert (het[1]["chain"], het[1]["resseq"], het[1]["icode"]) == ("B", 7, "A")
  assert het[0]["occ"] == pytest.approx(0.8)
  assert het[0]["b"] == pytest.approx(11.5)
  assert het[1]["b"] == pytest.approx(2.25)
  raw = [line for line in lines if line.startswith("HETATM")]
  assert raw[0][12:16] == " C1 "
  assert raw[0][76:78] == " C"
  assert raw[1][12:16] == "CL  "
  assert raw[1][76:78] == "CL"

  # The parsed flag round-trips through proxide's reader.
  back = parse_structure(out, OutputSpec(coord_format=CoordFormat.Full), use_jax=False)
  assert list(back.hetero) == hetero
