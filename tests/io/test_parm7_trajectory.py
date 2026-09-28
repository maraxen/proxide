"""parse_parm7 / parse_amber_trajectory on the alanine-dipeptide trajectory fixture.

The parm7 is generated here from ``trajectories/native.pdb`` (ACE-ALA-NME, 22 atoms, the
topology of ``trajectories/frame0.xtc``) so the expected values are known by construction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

proxide = pytest.importorskip("proxide")
_proxider = pytest.importorskip("proxide._proxider")
if not hasattr(_proxider, "parse_amber_trajectory"):
  pytest.skip("proxide built without parse_amber_trajectory", allow_module_level=True)

DATA = Path(__file__).resolve().parents[1] / "data" / "trajectories"
NATIVE = DATA / "native.pdb"
XTC = DATA / "frame0.xtc"  # 22 atoms, 501 frames (test.xtc is a 4-atom toy)
Z = {"H": 1, "C": 6, "N": 7, "O": 8}
MASS = {"H": 1.008, "C": 12.01, "N": 14.01, "O": 16.0}


def _block(flag: str, fmt: str, cells: list[str], per_line: int) -> str:
  lines = ["".join(cells[i : i + per_line]) for i in range(0, len(cells), per_line)] or [""]
  return "\n".join([f"%FLAG {flag}", f"%FORMAT({fmt})", *lines]) + "\n"


def _write_parm7(path: Path, drop_last_atom: bool = False) -> None:
  atoms = [line for line in NATIVE.read_text().splitlines() if line.startswith("ATOM")]
  if drop_last_atom:
    atoms = atoms[:-1]
  names = [a[12:16].strip() for a in atoms]
  resnames = [a[17:20].strip() for a in atoms]
  resseq = [int(a[22:26]) for a in atoms]
  elements = [n.lstrip("0123456789")[0] for n in names]
  res_first, res_labels = [], []
  for i, r in enumerate(resseq):
    if i == 0 or r != resseq[i - 1]:
      res_first.append(i + 1)
      res_labels.append(resnames[i])
  # Backbone + side-chain heavy-atom bonds are enough for connected components.
  index = {(resseq[i], n): i for i, n in enumerate(names)}
  pairs = [((1, "C"), (2, "N")), ((2, "N"), (2, "CA")), ((2, "CA"), (2, "C")), ((2, "C"), (3, "N")),
           ((1, "CH3"), (1, "C")), ((3, "N"), (3, "CH3"))]
  bonds = [(index[a], index[b]) for a, b in pairs if a in index and b in index]
  pointers = [0] * 31
  pointers[0], pointers[11], pointers[3] = len(atoms), len(res_labels), len(bonds)
  text = "%VERSION  VERSION_STAMP = V0001.000\n"
  text += _block("POINTERS", "10I8", [f"{v:8d}" for v in pointers], 10)
  text += _block("ATOM_NAME", "20a4", [f"{n:<4}" for n in names], 20)
  text += _block("CHARGE", "5E16.8", [f"{0.0:16.8E}" for _ in names], 5)
  text += _block("ATOMIC_NUMBER", "10I8", [f"{Z[e]:8d}" for e in elements], 10)
  text += _block("MASS", "5E16.8", [f"{MASS[e]:16.8E}" for e in elements], 5)
  text += _block("RESIDUE_LABEL", "20a4", [f"{r:<4}" for r in res_labels], 20)
  text += _block("RESIDUE_POINTER", "10I8", [f"{p:8d}" for p in res_first], 10)
  text += _block("BONDS_WITHOUT_HYDROGEN", "10I8", [f"{v:8d}" for a, b in bonds for v in (3 * a, 3 * b, 1)], 10)
  path.write_text(text)


@pytest.fixture
def parm7(tmp_path: Path) -> Path:
  path = tmp_path / "dipeptide.parm7"
  _write_parm7(path)
  return path


def test_parse_parm7_low_level(parm7: Path) -> None:
  topo = proxide.parse_parm7(parm7)
  assert topo["n_atoms"] == 22
  assert topo["res_names"] == ["ACE", "ALA", "NME"]
  assert topo["atom_names"][0] == "1HH3"
  assert topo["res_chain_ids"] == ["A", "A", "A"]
  assert topo["bonds"].shape[1] == 2


def test_frame0_matches_xtc_reader(parm7: Path) -> None:
  protein = proxide.parse_amber_trajectory(parm7, XTC, use_jax=False)
  aatype = np.asarray(protein.aatype)
  assert aatype.tolist() == [0], "only ALA is a protein residue (caps are not); AF index 0"
  ca = np.asarray(protein.coordinates)[0, 1]
  frame0 = np.asarray(proxide.parse_xtc(str(XTC))["coordinates"][0]).reshape(-1, 3)
  np.testing.assert_allclose(ca, frame0[8], atol=1e-4)  # atom 9 in native.pdb is ALA CA


def test_multiple_frames_are_stacked(parm7: Path) -> None:
  n_frames = proxide.frame_count(str(XTC))
  if n_frames < 2:
    pytest.skip("fixture trajectory has a single frame")
  protein = proxide.parse_amber_trajectory(parm7, XTC, frames=[0, -1], use_jax=False)
  coords = np.asarray(protein.coordinates)
  assert coords.shape[:3] == (2, 1, 37)
  assert not np.allclose(coords[0], coords[1])


def test_atom_count_mismatch_raises(tmp_path: Path) -> None:
  short = tmp_path / "short.parm7"
  _write_parm7(short, drop_last_atom=True)
  with pytest.raises(ValueError, match="atom count mismatch"):
    proxide.parse_amber_trajectory(short, XTC)


def test_out_of_range_frame_raises(parm7: Path) -> None:
  with pytest.raises(IndexError, match="out of range"):
    proxide.parse_amber_trajectory(parm7, XTC, frames=[10_000])


def test_unsupported_trajectory_format(parm7: Path) -> None:
  with pytest.raises(NotImplementedError, match=".xtc"):
    proxide.parse_amber_trajectory(parm7, DATA / "test.dcd")
