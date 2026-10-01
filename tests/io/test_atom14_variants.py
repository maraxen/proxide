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

_PDB = Path(__file__).resolve().parent.parent / "data" / "5awl.pdb"


def _with_first_residue_renamed(tmp_path: Path, new_name: str) -> tuple[Path, str]:
  lines = [x for x in _PDB.read_text().splitlines(keepends=True) if x.startswith("ATOM")]
  first = lines[0][22:27]  # residue number + insertion code of residue 0
  out = [x[:17] + f"{new_name:>3}" + x[20:] if x[22:27] == first else x for x in lines]
  path = tmp_path / f"{new_name}.pdb"
  path.write_text("".join(out) + "END\n")
  return path, lines[0][17:20]


def test_unknown_residue_is_reported_not_silently_dropped(tmp_path):
  path, _ = _with_first_residue_renamed(tmp_path, "MSE")
  with pytest.warns(UserWarning, match="MSE"):
    protein = parse_structure(str(path), OutputSpec(coord_format=CoordFormat.Atom14))
  # The residue really has no Atom14 atoms -- the warning is the only record.
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
