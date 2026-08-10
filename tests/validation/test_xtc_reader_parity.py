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
TIME_TOLERANCE_PS = 1e-4  # XTC frame times are stored as plain (not lossily
# compressed) floats, unlike coordinates — this is single-precision f32
# round-trip tolerance, not the coordinate compression tolerance above.


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
  # Ground truth for total_frames_on_disk: the true unstrided frame count,
  # established independently via a full stride=1 mdtraj load rather than
  # just trusting the N_FRAMES constant used to generate the fixture.
  true_total_frames = mdtraj.load(str(path), top=topology).n_frames
  return path, topology, true_total_frames


CASES = [
  pytest.param(1, None, id="full"),
  pytest.param(3, None, id="stride3"),
  pytest.param(1, [0, 4, 8, 40, 41, 200], id="atom_subset"),
  pytest.param(5, [1, 2, 3, 100], id="stride5_atom_subset"),
  # Unsorted atom_indices: mdtraj preserves the exact given order (fancy
  # indexing), so the bindings must too — a Mask-based selection (e.g.
  # molly's AtomSelection::from_index_list) would silently sort these and
  # diverge. (mdtraj.load rejects duplicate atom_indices outright, so that
  # case is covered separately below via a ground-truth numpy comparison
  # rather than against mdtraj.)
  pytest.param(1, [200, 0, 41, 8, 4, 40], id="atom_subset_unsorted"),
]


def _assert_parity(proxide_result: dict, mdtraj_traj, true_total_frames: int) -> None:
  mdtraj_coords = mdtraj_traj.xyz * 10.0  # nm -> Angstroms
  mdtraj_box = mdtraj_traj.unitcell_vectors * 10.0  # nm -> Angstroms

  assert proxide_result["coordinates"].shape == mdtraj_coords.shape
  coord_diff = np.abs(proxide_result["coordinates"] - mdtraj_coords).max()
  assert coord_diff < COORD_TOLERANCE_ANGSTROM, f"coords differ by {coord_diff} Å"

  np.testing.assert_allclose(
    proxide_result["box_vectors"], mdtraj_box, atol=COORD_TOLERANCE_ANGSTROM
  )

  assert proxide_result["times"].shape == (proxide_result["num_frames"],)
  np.testing.assert_allclose(
    proxide_result["times"], mdtraj_traj.time, atol=TIME_TOLERANCE_PS
  )

  assert proxide_result["total_frames_on_disk"] == true_total_frames


@pytest.mark.skipif(not MDTRAJ_AVAILABLE, reason="MDTraj not installed")
@pytest.mark.parametrize("stride, atom_indices", CASES)
def test_read_xtc_lazy_matches_mdtraj(synthetic_xtc, stride, atom_indices):
  import proxide

  path, topology, true_total_frames = synthetic_xtc
  result = proxide.read_xtc_lazy(str(path), stride=stride, atom_indices=atom_indices)
  reference = mdtraj.load(str(path), top=topology, stride=stride, atom_indices=atom_indices)
  _assert_parity(result, reference, true_total_frames)


@pytest.mark.skipif(not MDTRAJ_AVAILABLE, reason="MDTraj not installed")
@pytest.mark.parametrize("stride, atom_indices", CASES)
def test_read_xtc_parallel_matches_mdtraj(synthetic_xtc, stride, atom_indices):
  import proxide

  path, topology, true_total_frames = synthetic_xtc
  result = proxide.read_xtc_parallel(str(path), stride=stride, atom_indices=atom_indices)
  reference = mdtraj.load(str(path), top=topology, stride=stride, atom_indices=atom_indices)
  _assert_parity(result, reference, true_total_frames)


@pytest.mark.skipif(not MDTRAJ_AVAILABLE, reason="MDTraj not installed")
def test_read_xtc_lazy_and_parallel_agree(synthetic_xtc):
  """Both new bindings should decode identical numbers, independent of mdtraj."""
  import proxide

  path, _topology, true_total_frames = synthetic_xtc
  lazy = proxide.read_xtc_lazy(str(path), stride=2, atom_indices=[0, 5, 10])
  parallel = proxide.read_xtc_parallel(str(path), stride=2, atom_indices=[0, 5, 10])

  assert lazy["num_frames"] == parallel["num_frames"]
  assert lazy["num_atoms"] == parallel["num_atoms"]
  np.testing.assert_allclose(lazy["coordinates"], parallel["coordinates"])
  np.testing.assert_allclose(lazy["box_vectors"], parallel["box_vectors"])
  np.testing.assert_allclose(lazy["times"], parallel["times"])
  assert lazy["total_frames_on_disk"] == parallel["total_frames_on_disk"] == true_total_frames


@pytest.mark.skipif(not MDTRAJ_AVAILABLE, reason="MDTraj not installed")
@pytest.mark.parametrize("fn_name", ["read_xtc_lazy", "read_xtc_parallel"])
def test_atom_indices_preserve_order_and_duplicates(synthetic_xtc, fn_name):
  """atom_indices order/duplicates must be preserved exactly (numpy fancy-
  indexing semantics), matching mdtraj — not silently sorted/deduped by a
  Mask-style selection. mdtraj.load itself rejects duplicate atom_indices
  outright, so the ground truth here is the full (unselected) read, fancy-
  indexed in Python, rather than mdtraj.
  """
  import proxide

  path, _topology, _true_total_frames = synthetic_xtc
  fn = getattr(proxide, fn_name)
  requested = [5, 2, 5, 0, 41]

  full = fn(str(path))
  subset = fn(str(path), atom_indices=requested)

  expected_coords = full["coordinates"][:, requested, :]
  np.testing.assert_allclose(subset["coordinates"], expected_coords)
  assert subset["num_atoms"] == len(requested)


@pytest.mark.skipif(not MDTRAJ_AVAILABLE, reason="MDTraj not installed")
@pytest.mark.parametrize("fn_name", ["read_xtc_lazy", "read_xtc_parallel"])
def test_out_of_range_atom_index_raises(synthetic_xtc, fn_name):
  """An out-of-range atom index must raise a clean error, not panic."""
  import proxide

  path, _topology, _true_total_frames = synthetic_xtc
  fn = getattr(proxide, fn_name)
  with pytest.raises(ValueError, match="out of range"):
    fn(str(path), atom_indices=[0, N_ATOMS + 100])


@pytest.mark.skipif(not MDTRAJ_AVAILABLE, reason="MDTraj not installed")
def test_frame_count_matches_mdtraj(synthetic_xtc):
  """`proxide.frame_count` must agree with a full mdtraj load — and must not
  materialize any coordinate data to get there (unlike read_xtc_lazy/
  read_xtc_parallel, which decode every requested frame)."""
  import proxide

  path, _topology, true_total_frames = synthetic_xtc
  assert proxide.frame_count(str(path)) == true_total_frames


@pytest.mark.skipif(not MDTRAJ_AVAILABLE, reason="MDTraj not installed")
def test_n_atoms_matches_mdtraj(synthetic_xtc):
  import proxide

  path, topology, _true_total_frames = synthetic_xtc
  assert proxide.n_atoms(str(path)) == topology.n_atoms


@pytest.mark.skipif(not MDTRAJ_AVAILABLE, reason="MDTraj not installed")
def test_frame_count_excludes_truncated_trailing_frame(synthetic_xtc, tmp_path):
  """A trajectory whose last frame is only partially flushed — the normal
  state of the tail of a file a live simulation is still appending to — must
  not be counted. Regression coverage for the same guard exercised at the
  Rust level in `proxide-io`'s `test_xtc_reader_excludes_truncated_trailing_frame`,
  here through the actual Python-facing binding.
  """
  import proxide

  path, _topology, true_total_frames = synthetic_xtc

  full_bytes = path.read_bytes()
  truncated_path = tmp_path / "truncated.xtc"
  # Chop well inside the last frame's own compressed payload, not exactly on
  # a frame boundary.
  truncated_path.write_bytes(full_bytes[:-10])

  assert proxide.frame_count(str(truncated_path)) == true_total_frames - 1
