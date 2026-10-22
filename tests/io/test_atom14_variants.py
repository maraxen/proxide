"""Atom14/Atom37 formatting of MSE and unknown residue names (debt #2362).

MSE (selenomethionine) is typed as MET in the model-facing arrays (aatype 12).
Each array format's alias policy places its SE in the SD slot. Writers keep the
selenium: an MSE residue is written as HETATM MSE with an SE atom. A residue
with atoms but no layout (unknown names, ligands) is reported with a UserWarning by
Atom14. Atom37 places atoms by name and reports nothing.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from proxide import CoordFormat, OutputSpec, _proxider, parse_structure
from proxide.chem.residues import atom_types, resnames
from proxide.io.writing import (
  _HETATM_RESIDUES,
  _LAYOUT_ALIASES,
  write_mmcif,
  write_npz,
  write_pdb,
)
from proxide.ops.transforms import _pad_protein, _stack_padded_proteins, truncate_protein

_DATA = Path(__file__).resolve().parent.parent / "data"
_PROBE = _DATA / "mse_altloc.pdb"  # ALA, MSE with altlocs A/B on SE, GLY
_MET = resnames.index("MET")
_SD_ATOM37 = atom_types.index("SD")
_SD_ATOM14 = 6  # MET atom14 layout: N CA C O CB CG SD CE
_SE_A = [13.2, 16.1, 12.9]  # altloc A (occupancy 0.60) wins over B (0.40)
_FORMATS = [CoordFormat.Atom14, CoordFormat.Atom37]


def _parse(fmt: CoordFormat, path: Path):
  return parse_structure(str(path), OutputSpec(coord_format=fmt))


def _mse_index(protein) -> int:
  return int(np.flatnonzero(np.asarray(protein.residue_index) == 2)[0])


def _se_slot(fmt: CoordFormat) -> int:
  return _SD_ATOM14 if fmt == CoordFormat.Atom14 else _SD_ATOM37


def _write_residue_pdb(tmp_path: Path, res_name: str, record: str, atoms) -> Path:
  lines = []
  for serial, (name, element, x, y, z) in enumerate(atoms, start=1):
    lines.append(
      f"{record:<6}{serial:5d} {name:<4s} {res_name} A   1    "
      f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00          {element:>2s}\n"
    )
  path = tmp_path / f"{res_name}.pdb"
  path.write_text("".join(lines) + "END\n")
  return path


_XYZ_ATOMS = [
  ("N", "N", 0.0, 0.0, 0.0),
  ("CA", "C", 1.0, 0.0, 0.0),
  ("C", "C", 2.0, 0.0, 0.0),
]


@pytest.mark.parametrize("fmt", _FORMATS, ids=["atom14", "atom37"])
def test_mse_is_typed_met_with_se_in_sd(fmt):
  with warnings.catch_warnings():
    warnings.simplefilter("error", UserWarning)  # MSE must not be "unplaced"
    protein = _parse(fmt, _PROBE)
  idx = _mse_index(protein)
  assert int(np.asarray(protein.aatype)[idx]) == _MET
  mask = np.asarray(protein.atom_mask)[idx]
  assert float(mask.sum()) == 8.0  # N CA C O CB CG SE CE
  coords = np.asarray(protein.coordinates)[idx]
  np.testing.assert_allclose(coords[_se_slot(fmt)], _SE_A, atol=1e-3)


@pytest.mark.parametrize("fmt", _FORMATS, ids=["atom14", "atom37"])
def test_mse_round_trips_through_pdb_as_hetatm_se(fmt, tmp_path):
  protein = _parse(fmt, _PROBE)
  out = write_pdb(protein, tmp_path / "out.pdb")
  mse_lines = [x for x in out.read_text().splitlines() if x[17:20] == "MSE"]
  assert len(mse_lines) == 8
  assert all(x.startswith("HETATM") for x in mse_lines)
  se_lines = [x for x in mse_lines if x[12:16].strip() == "SE"]
  assert len(se_lines) == 1
  assert se_lines[0][76:78].strip().upper() == "SE"
  assert not any(x[12:16].strip() == "SD" for x in mse_lines)
  reparsed = _parse(fmt, out)
  np.testing.assert_allclose(
    np.asarray(reparsed.coordinates)[_mse_index(reparsed)][_se_slot(fmt)], _SE_A, atol=1e-3
  )


@pytest.mark.parametrize("fmt", _FORMATS, ids=["atom14", "atom37"])
def test_mse_round_trips_through_mmcif_as_hetatm_se(fmt, tmp_path):
  protein = _parse(fmt, _PROBE)
  out = write_mmcif(protein, tmp_path / "out.cif")
  mse_rows = [x.split() for x in out.read_text().splitlines() if " MSE " in x]
  assert len(mse_rows) == 8
  assert all(row[0] == "HETATM" for row in mse_rows)
  se_rows = [row for row in mse_rows if row[2] == "SE" and row[3] == "SE"]  # type, label atom
  assert len(se_rows) == 1
  reparsed = _parse(fmt, out)
  np.testing.assert_allclose(
    np.asarray(reparsed.coordinates)[_mse_index(reparsed)][_se_slot(fmt)], _SE_A, atol=1e-3
  )


def test_full_format_keeps_real_se_atom():
  protein = _parse(CoordFormat.Full, _PROBE)
  names = [str(n) for n in protein.atom_names]
  assert "SE" in names
  assert "SD" not in names


@pytest.mark.parametrize("fmt", [CoordFormat.Atom14], ids=["atom14"])
def test_unknown_residue_still_warns(fmt, tmp_path):
  # Negative control: a residue name with no alias and no layout keeps the warning.
  path = _write_residue_pdb(tmp_path, "XYZ", "ATOM", _XYZ_ATOMS)
  with pytest.warns(UserWarning, match="XYZ"):
    _parse(fmt, path)


_UBQ = Path(__file__).resolve().parents[2] / "crates" / "proxide-io" / "tests" / "data" / "1UBQ.cif"


def test_alias_variant_is_placed_like_its_parent(tmp_path):
  # Alias resolution only applies when the residue's atoms match the parent's
  # heavy-atom set, so this needs a REAL histidine: ubiquitin's single His68,
  # renamed to the CHARMM variant HSD. It must be placed exactly like HIS.
  his_cif = _UBQ.read_text()
  hsd_cif = tmp_path / "1UBQ_hsd.cif"
  hsd_cif.write_text(his_cif.replace(" HIS ", " HSD "))
  assert " HSD " in hsd_cif.read_text()

  spec = OutputSpec(coord_format=CoordFormat.Atom14)
  his = parse_structure(str(_UBQ), spec)
  with warnings.catch_warnings():
    warnings.simplefilter("error", UserWarning)  # HSD must not be "unplaced"
    hsd = parse_structure(str(hsd_cif), spec)
  np.testing.assert_array_equal(np.asarray(hsd.atom_mask), np.asarray(his.atom_mask))
  np.testing.assert_array_equal(np.asarray(hsd.aatype), np.asarray(his.aatype))
  his_idx = int(np.flatnonzero(np.asarray(his.residue_index) == 68)[0])
  assert float(np.asarray(hsd.atom_mask)[his_idx].sum()) == 10.0  # His heavy atoms


_MET = resnames.index("MET")


def _mse_protein(fmt: CoordFormat):
  return _parse(fmt, _PROBE)


def _mse_rows(text: str) -> list[str]:
  return [x for x in text.splitlines() if x[17:20] == "MSE"]


def test_writer_tables_equal_the_rust_policy():
  rust = _proxider._layout_aliases()
  assert set(_LAYOUT_ALIASES) == set(rust) == {"Atom14", "Atom37"}
  for fmt, table in rust.items():
    expected = {res: (layout, dict(renames)) for res, (layout, renames) in table.items()}
    assert _LAYOUT_ALIASES[fmt] == expected
  assert _HETATM_RESIDUES == frozenset(_proxider._hetatm_residues())
  assert _LAYOUT_ALIASES["Atom14"]["MSE"] == ("MET", {"SE": "SD"})
  assert _LAYOUT_ALIASES["Atom37"]["MSE"] == ("MET", {"SE": "SD"})


@pytest.mark.parametrize("fmt", _FORMATS, ids=["atom14", "atom37"])
def test_crop_then_write_keeps_mse_as_hetatm_se(fmt, tmp_path):
  protein = truncate_protein(_mse_protein(fmt), 2, "center_crop")  # ALA, MSE
  assert protein.residue_names == ["ALA", "MSE"]
  out = write_pdb(protein, tmp_path / "crop.pdb")
  rows = _mse_rows(out.read_text())
  assert len(rows) == 8
  assert all(x.startswith("HETATM") for x in rows)
  assert sum(x[12:16].strip() == "SE" for x in rows) == 1


@pytest.mark.parametrize("fmt", _FORMATS, ids=["atom14", "atom37"])
def test_pad_then_write_keeps_mse_as_hetatm_se(fmt, tmp_path):
  protein = _pad_protein(_mse_protein(fmt), 5)
  assert protein.residue_names == ["ALA", "MSE", "GLY", "", ""]
  out = write_pdb(protein, tmp_path / "pad.pdb")
  rows = _mse_rows(out.read_text())
  assert len(rows) == 8
  assert sum(x[12:16].strip() == "SE" for x in rows) == 1


def test_stack_keeps_each_rows_residue_names():
  full = _pad_protein(_mse_protein(CoordFormat.Atom14), 3)
  cropped = _pad_protein(truncate_protein(_mse_protein(CoordFormat.Atom14), 2, "center_crop"), 3)
  batched = _stack_padded_proteins([full, cropped])
  assert batched.residue_names == [["ALA", "MSE", "GLY"], ["ALA", "MSE", ""]]


def test_npz_round_trip_keeps_mse_residue_name(tmp_path):
  path = write_npz(_mse_protein(CoordFormat.Atom14), tmp_path / "mse.npz")
  with np.load(path) as data:
    names = [str(x) for x in data["residue_names"]]
    aatype = np.asarray(data["aatype"])
  assert names == ["ALA", "MSE", "GLY"]
  assert int(aatype[1]) == _MET


def test_met_sd_without_residue_names_warns_selenium_unrestorable(tmp_path):
  protein = _mse_protein(CoordFormat.Atom37).replace(residue_names=None)
  with pytest.warns(UserWarning, match="selenium"):
    write_pdb(protein, tmp_path / "no_names.pdb")


def test_residue_names_present_writes_without_warning(tmp_path):
  with warnings.catch_warnings():
    warnings.simplefilter("error", UserWarning)
    write_pdb(_mse_protein(CoordFormat.Atom37), tmp_path / "with_names.pdb")
