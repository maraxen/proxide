"""Parity tests for the lazy/parallel XTC reader Python bindings vs MDTraj.

Covers `proxide.read_xtc_lazy` (single-threaded, offset-cached `XtcReader`
cursor) and `proxide.read_xtc_parallel` (`read_frames_parallel`), each with
stride and atom-index selection, against `mdtraj.load(path, stride=...,
atom_indices=...)` as the reference oracle — mirroring
`tests/validation/test_trajectory_parity.py`'s conventions.

The synthetic fixture is generated at test time into `tmp_path` (never
checked into git — see `scripts/generate_trajectory_test_data.py`'s
`generate_large_xtc_fixture` for the pattern this follows, scaled up).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

try:
  import mdtraj

  MDTRAJ_AVAILABLE = True
except ImportError:
  MDTRAJ_AVAILABLE = False

# Small enough to run fast in CI, large enough to exercise stride/selection
# logic and the offset-scan/cache path meaningfully (unlike the tiny 5-frame
# test.xtc fixture).
N_FRAMES = 2500
N_ATOMS = 240
COORD_TOLERANCE_ANGSTROM = 0.01  # matches test_trajectory_parity.py — XTC's
# single-precision lossy-compression round-trip tolerance, not bit-exactness.


def _build_synthetic_xtc(path: Path):
  """Write a synthetic multi-thousand-frame/multi-hundred-atom XTC to `path`.

  Returns the `mdtraj.Topology` used, since XTC has no embedded topology and
  `mdtraj.load` requires one to be passed explicitly.
  """
  topology = mdtraj.Topology()
  chain = topology.add_chain()
  n_residues = N_ATOMS // 4
  atoms = []
  prev_c = None
  for _ in range(n_residues):
    residue = topology.add_residue("ALA", chain)
    n_atom = topology.add_atom("N", mdtraj.element.nitrogen, residue)
    ca_atom = topology.add_atom("CA", mdtraj.element.carbon, residue)
    c_atom = topology.add_atom("C", mdtraj.element.carbon, residue)
    o_atom = topology.add_atom("O", mdtraj.element.oxygen, residue)
    topology.add_bond(n_atom, ca_atom)
    topology.add_bond(ca_atom, c_atom)
    topology.add_bond(c_atom, o_atom)
    if prev_c is not None:
      topology.add_bond(prev_c, n_atom)
    prev_c = c_atom
    atoms.extend([n_atom, ca_atom, c_atom, o_atom])

  n_atoms = len(atoms)
  base_coords = np.zeros((n_atoms, 3), dtype=np.float32)
  for i in range(n_atoms):
    residue_idx, offset = divmod(i, 4)
    within_residue = np.array(
      [[0.00, 0.00, 0.00], [0.15, 0.00, 0.00], [0.25, 0.10, 0.00], [0.25, 0.20, 0.00]],
      dtype=np.float32,
    )[offset]
    base_coords[i] = within_residue + np.array([residue_idx * 0.4, 0.0, 0.0], dtype=np.float32)

  rng = np.random.default_rng(1337)
  coords = np.zeros((N_FRAMES, n_atoms, 3), dtype=np.float32)
  for i in range(N_FRAMES):
    coords[i] = base_coords + rng.standard_normal((n_atoms, 3)).astype(np.float32) * 0.01

  unitcell_lengths = np.array([[10.0, 10.0, 10.0]] * N_FRAMES, dtype=np.float32)
  unitcell_angles = np.array([[90.0, 90.0, 90.0]] * N_FRAMES, dtype=np.float32)

  traj = mdtraj.Trajectory(
    coords,
    topology,
    unitcell_lengths=unitcell_lengths,
    unitcell_angles=unitcell_angles,
  )
  traj.save_xtc(str(path))
  return topology


@pytest.fixture(scope="module")
def synthetic_xtc(tmp_path_factory):
  if not MDTRAJ_AVAILABLE:
    pytest.skip("MDTraj not installed")
  path = tmp_path_factory.mktemp("xtc_reader_parity") / "synthetic.xtc"
  topology = _build_synthetic_xtc(path)
  return path, topology


CASES = [
  pytest.param(1, None, id="full"),
  pytest.param(3, None, id="stride3"),
  pytest.param(1, [0, 4, 8, 40, 41, 200], id="atom_subset"),
  pytest.param(5, [1, 2, 3, 100], id="stride5_atom_subset"),
]


def _assert_parity(proxide_result: dict, mdtraj_traj) -> None:
  mdtraj_coords = mdtraj_traj.xyz * 10.0  # nm -> Angstroms
  mdtraj_box = mdtraj_traj.unitcell_vectors * 10.0  # nm -> Angstroms

  assert proxide_result["coordinates"].shape == mdtraj_coords.shape
  coord_diff = np.abs(proxide_result["coordinates"] - mdtraj_coords).max()
  assert coord_diff < COORD_TOLERANCE_ANGSTROM, f"coords differ by {coord_diff} Å"

  np.testing.assert_allclose(
    proxide_result["box_vectors"], mdtraj_box, atol=COORD_TOLERANCE_ANGSTROM
  )


@pytest.mark.skipif(not MDTRAJ_AVAILABLE, reason="MDTraj not installed")
@pytest.mark.parametrize("stride, atom_indices", CASES)
def test_read_xtc_lazy_matches_mdtraj(synthetic_xtc, stride, atom_indices):
  import proxide

  path, topology = synthetic_xtc
  result = proxide.read_xtc_lazy(str(path), stride=stride, atom_indices=atom_indices)
  reference = mdtraj.load(str(path), top=topology, stride=stride, atom_indices=atom_indices)
  _assert_parity(result, reference)


@pytest.mark.skipif(not MDTRAJ_AVAILABLE, reason="MDTraj not installed")
@pytest.mark.parametrize("stride, atom_indices", CASES)
def test_read_xtc_parallel_matches_mdtraj(synthetic_xtc, stride, atom_indices):
  import proxide

  path, topology = synthetic_xtc
  result = proxide.read_xtc_parallel(str(path), stride=stride, atom_indices=atom_indices)
  reference = mdtraj.load(str(path), top=topology, stride=stride, atom_indices=atom_indices)
  _assert_parity(result, reference)


@pytest.mark.skipif(not MDTRAJ_AVAILABLE, reason="MDTraj not installed")
def test_read_xtc_lazy_and_parallel_agree(synthetic_xtc):
  """Both new bindings should decode identical numbers, independent of mdtraj."""
  import proxide

  path, _topology = synthetic_xtc
  lazy = proxide.read_xtc_lazy(str(path), stride=2, atom_indices=[0, 5, 10])
  parallel = proxide.read_xtc_parallel(str(path), stride=2, atom_indices=[0, 5, 10])

  assert lazy["num_frames"] == parallel["num_frames"]
  assert lazy["num_atoms"] == parallel["num_atoms"]
  np.testing.assert_allclose(lazy["coordinates"], parallel["coordinates"])
  np.testing.assert_allclose(lazy["box_vectors"], parallel["box_vectors"])
