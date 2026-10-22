"""Altloc resolution keys residues by identity, not by (chain, res_id, atom) alone.

Regression tests for debt #1930 review: under auth numbering, residues 17 and 17A
share (chain, res_id). Before the fix their altloc CA atoms were merged and the
structure lost residues. Also covers the temp-file round trip (serial numbers above
99 999 and HETATM records) and models that share residue numbers.
"""

import shutil
from pathlib import Path

import numpy as np
from proxide import OutputSpec, parse_structure
from proxide.io.parsing import backend

HEADER = """data_ALTLOC
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_seq_id
_atom_site.auth_asym_id
_atom_site.pdbx_PDB_model_num
"""

# Residue 17 (ALA) and residue 17A (GLY, label_seq 18) share auth chain A and
# auth number 17; both have altloc CA atoms.
INSERTION_CODE_ROWS = """ATOM 1 N N . ALA A 17 ? 0.0 0.0 0.0 1.00 10.0 17 A 1
ATOM 2 C CA A ALA A 17 ? 1.0 0.0 0.0 0.60 10.0 17 A 1
ATOM 3 C CA B ALA A 17 ? 1.1 0.0 0.0 0.40 10.0 17 A 1
ATOM 4 C C . ALA A 17 ? 2.0 0.0 0.0 1.00 10.0 17 A 1
ATOM 5 N N . GLY A 18 A 3.0 0.0 0.0 1.00 10.0 17 A 1
ATOM 6 C CA A GLY A 18 A 4.0 0.0 0.0 0.60 10.0 17 A 1
ATOM 7 C CA B GLY A 18 A 4.1 0.0 0.0 0.40 10.0 17 A 1
ATOM 8 C C . GLY A 18 A 5.0 0.0 0.0 1.00 10.0 17 A 1
"""

# One ALA residue (no altloc) plus a water with two altloc positions as HETATM.
# Serial 4 (occupancy 0.6) is kept; serial 5 (0.4) is dropped.
HETATM_ALTLOC_ROWS = """ATOM 1 N N . ALA A 1 ? 0.0 0.0 0.0 1.00 10.0 1 A 1
ATOM 2 C CA . ALA A 1 ? 1.0 0.0 0.0 1.00 10.0 1 A 1
ATOM 3 C C . ALA A 1 ? 2.0 0.0 0.0 1.00 10.0 1 A 1
HETATM 4 O O A HOH A . ? 6.0 0.0 0.0 0.60 10.0 201 A 1
HETATM 5 O O B HOH A . ? 6.1 0.0 0.0 0.40 10.0 201 A 1
"""

# One ALA residue with altloc CA atoms; serials 99999 and 100000..100002 span the
# 99 999 boundary. No residue number is shared, so only serial handling is tested.
SERIAL_ALTLOC_ROWS = """ATOM 99999 N N . ALA A 1 ? 0.0 0.0 0.0 1.00 10.0 1 A 1
ATOM 100000 C CA A ALA A 1 ? 1.0 0.0 0.0 0.60 10.0 1 A 1
ATOM 100001 C CA B ALA A 1 ? 1.1 0.0 0.0 0.40 10.0 1 A 1
ATOM 100002 C C . ALA A 1 ? 2.0 0.0 0.0 1.00 10.0 1 A 1
"""


def _write(tmp_path: Path, rows: str, name: str = "altloc.cif") -> Path:
  path = tmp_path / name
  path.write_text(HEADER + rows)
  return path


def _atoms_present(protein) -> int:
  return int(np.asarray(protein.atom_mask).sum())


def _parse_and_reparse_temp(monkeypatch, tmp_path: Path, path: Path):
  """Parse ``path`` and return ``(protein, raw)``.

  ``raw`` is the low-level atom dict (``serial_numbers``, ``is_hetatm``, ...) of the
  temporary file that the altloc path re-parsed. The spy copies that file while it
  still exists, so the assertions read what the round trip actually wrote.
  """
  seen: list[Path] = []
  real_parse_structure = backend._proxider.parse_structure

  def spy(file_path, spec, numbering):
    if Path(file_path) != path:
      copy = tmp_path / f"reparsed{Path(file_path).suffix}"
      shutil.copy(file_path, copy)
      seen.append(copy)
    return real_parse_structure(file_path, spec, numbering)

  monkeypatch.setattr(backend._proxider, "parse_structure", spy)
  protein = parse_structure(str(path), OutputSpec(), numbering="auth")
  assert len(seen) == 1, "the altloc path should re-parse exactly one temporary file"
  temp = seen[0]
  if temp.suffix == ".cif":
    raw = backend._proxider.parse_mmcif(str(temp), "auth")
  else:
    raw = backend._proxider.parse_pdb(str(temp))
  return protein, raw


def test_residues_17_and_17A_keep_their_own_altloc_atoms(tmp_path) -> None:
  path = _write(tmp_path, INSERTION_CODE_ROWS)
  protein = parse_structure(str(path), OutputSpec(), numbering="auth")
  # Two residues (17 and 17A), each with N, CA, C once: 6 atoms present.
  assert len(np.asarray(protein.residue_index)) == 2
  assert _atoms_present(protein) == 6


def test_label_numbering_gives_the_same_counts(tmp_path) -> None:
  path = _write(tmp_path, INSERTION_CODE_ROWS)
  protein = parse_structure(str(path), OutputSpec(), numbering="label")
  assert np.asarray(protein.residue_index).tolist() == [17, 18]
  assert _atoms_present(protein) == 6


def test_serials_above_99999_survive_the_altloc_round_trip(tmp_path, monkeypatch) -> None:
  path = _write(tmp_path, SERIAL_ALTLOC_ROWS, name="big_serial.cif")
  _, raw = _parse_and_reparse_temp(monkeypatch, tmp_path, path)
  # The altloc B atom (serial 100001, occupancy 0.4) is dropped; the others keep
  # their original serial numbers, including those above 99 999.
  assert sorted(np.asarray(raw["serial_numbers"]).tolist()) == [99999, 100000, 100002]


def test_hetatm_with_altloc_survives_the_altloc_round_trip(tmp_path, monkeypatch) -> None:
  path = _write(tmp_path, HETATM_ALTLOC_ROWS, name="hetatm_altloc.cif")
  _, raw = _parse_and_reparse_temp(monkeypatch, tmp_path, path)
  is_hetatm = np.asarray(raw["is_hetatm"]).tolist()
  # Exactly one HETATM survives: the kept water altloc (serial 4). Protein atoms stay ATOM.
  assert is_hetatm.count(True) == 1
  kept = is_hetatm.index(True)
  assert int(np.asarray(raw["serial_numbers"])[kept]) == 4
  assert str(np.asarray(raw["res_names"])[kept]).strip() == "HOH"


def test_models_sharing_a_residue_are_not_merged(tmp_path) -> None:
  rows = (
    "ATOM 1 N N . ALA A 1 ? 0.0 0.0 0.0 1.00 10.0 1 A 1\n"
    "ATOM 2 C CA . ALA A 1 ? 1.0 0.0 0.0 1.00 10.0 1 A 1\n"
    "ATOM 3 C C . ALA A 1 ? 2.0 0.0 0.0 1.00 10.0 1 A 1\n"
    "ATOM 4 N N . ALA A 1 ? 0.1 0.0 0.0 1.00 10.0 1 A 2\n"
    "ATOM 5 C CA A ALA A 1 ? 1.1 0.0 0.0 0.60 10.0 1 A 2\n"
    "ATOM 6 C CA B ALA A 1 ? 1.2 0.0 0.0 0.40 10.0 1 A 2\n"
    "ATOM 7 C C . ALA A 1 ? 2.1 0.0 0.0 1.00 10.0 1 A 2\n"
  )
  path = _write(tmp_path, rows, name="two_models.cif")
  protein = parse_structure(str(path), OutputSpec(), numbering="auth")
  assert protein.n_models == 2
  # Each model keeps N, CA, C once.
  assert _atoms_present(protein) == 6
