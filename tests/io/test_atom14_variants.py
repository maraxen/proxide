"""Atom14 formatting of variant / unknown residue names (debt #2362).

The Atom14 formatter used to pick each residue's atom layout by its as-read
name, so alias variants (HSD/HID/HIE -> HIS, CYX -> CYS, ...) and unknown
residues (MSE, ligands) fell to the empty UNK layout: the residue silently
vanished from the output. It now looks the layout up by residue type, and a
residue that still has no layout but does have atoms is reported with a
UserWarning naming it.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from proxide import CoordFormat, OutputSpec, parse_structure


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


# MSE with SE (selenium). MET layout is N CA C O CB CG SD CE, so SE belongs in SD.
_MSE_ATOMS = [
  ("N", "N", 0.0, 0.0, 0.0),
  ("CA", "C", 1.0, 0.0, 0.0),
  ("C", "C", 2.0, 0.0, 0.0),
  ("O", "O", 3.0, 0.0, 0.0),
  ("CB", "C", 1.0, 1.0, 0.0),
  ("CG", "C", 1.0, 2.0, 0.0),
  ("SE", "SE", 1.0, 3.0, 0.0),
  ("CE", "C", 2.0, 3.0, 0.0),
]


def test_mse_is_placed_via_met_layout_with_se_in_sd(tmp_path):
  path = _write_residue_pdb(tmp_path, "MSE", "HETATM", _MSE_ATOMS)
  spec = OutputSpec(coord_format=CoordFormat.Atom14)
  with warnings.catch_warnings():
    warnings.simplefilter("error", UserWarning)  # MSE must not be "unplaced"
    protein = parse_structure(str(path), spec)
  mask = np.asarray(protein.atom_mask)[0]
  assert float(mask.sum()) == float(len(_MSE_ATOMS))  # all 8 present atoms placed
  np.testing.assert_array_equal(mask[:8], np.ones(8, dtype=mask.dtype))
  assert float(mask[8:].sum()) == 0.0
  coords = np.asarray(protein.coordinates)[0]
  np.testing.assert_allclose(coords[6], [1.0, 3.0, 0.0])  # SD slot holds SE


def test_unknown_residue_is_reported_not_silently_dropped(tmp_path):
  # Negative control: a residue name with no alias and no layout keeps the warning.
  path = _write_residue_pdb(tmp_path, "XYZ", "ATOM", _MSE_ATOMS[:3])
  with pytest.warns(UserWarning, match="XYZ"):
    protein = parse_structure(str(path), OutputSpec(coord_format=CoordFormat.Atom14))
  assert float(np.asarray(protein.atom_mask)[0].sum()) == 0.0


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
