"""Parity tests for `proxide.read_xtc_ca_distogram`.

Covers the minimum-image-convention (MIC) pairwise-distance "distogram"
binding against two independent references:

1. A real-fixture cross-tool check against mdtraj: load the same frames via
   mdtraj, physically unwrap with `Trajectory.image_molecules()`, then
   compute naive Euclidean pairwise distances on the unwrapped coordinates.
   `read_xtc_ca_distogram` computes MIC directly on the *raw* (possibly
   PBC-split) coordinates, with no unwrap step at all — these are two
   independent approaches to the same physical quantity, so agreement here
   is real correctness signal, not a tautology.
2. An explicit `numpy.triu_indices`-order enumeration, since a silently
   wrong pair order would misassign every downstream residue-pair label
   without ever producing a shape mismatch or a NaN.

Mirrors `tests/validation/test_xtc_reader_parity.py`'s conventions (skip
cleanly if mdtraj isn't installed, Angstrom-scale coordinate tolerance
matching XTC's single-precision lossy-compression round-trip).
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

# Real, checked-in fixture: a 22-atom ACE-ALA-NME capped dipeptide, 501
# frames, with a real (small, ~25.7 A) periodic box — see
# tests/validation/test_xtc_reader_parity.py's synthetic fixtures for the
# larger-system case; this one is chosen specifically because it is real MD
# output (not synthetic) with a real topology usable by
# `Trajectory.image_molecules()`.
FIXTURE_DIR = Path(__file__).parent.parent / "data" / "trajectories"
XTC_PATH = FIXTURE_DIR / "frame0.xtc"
PDB_PATH = FIXTURE_DIR / "frame0.pdb"

COORD_TOLERANCE_ANGSTROM = 0.01  # XTC lossy-compression round-trip tolerance,
# matching test_xtc_reader_parity.py and test_trajectory_parity.py.
# MIC-vs-unwrap-then-naive distance tolerance: looser than the raw coordinate
# tolerance because a distance combines two independently-compressed
# coordinates (each with up to COORD_TOLERANCE_ANGSTROM error) through a
# sqrt(sum of squares), so worst-case error can be a small multiple of the
# per-coordinate tolerance.
DISTANCE_TOLERANCE_ANGSTROM = 0.05


def _skip_if_fixture_missing():
  if not XTC_PATH.exists() or not PDB_PATH.exists():
    pytest.skip(f"real fixture not found at {XTC_PATH} / {PDB_PATH}")


def _reference_distogram_via_unwrap_then_naive(
  xtc_path: Path, pdb_path: Path, atom_indices: list[int], stride: int
) -> np.ndarray:
  """Independent oracle: mdtraj load -> physical PBC unwrap
  (`image_molecules`) -> naive Euclidean pairwise distances on the unwrapped
  coordinates, restricted to `atom_indices` in the exact order given.

  This is architecturally the *opposite* strategy from
  `read_xtc_ca_distogram` (unwrap-first-distance-second vs.
  MIC-on-raw-coordinates) — the whole point of this test is that both
  should converge to the same physical answer despite starting from
  differently-processed coordinates.
  """
  traj = mdtraj.load(str(xtc_path), top=str(pdb_path), stride=stride)
  # This fixture is a single 22-atom capped dipeptide with no solvent —
  # mdtraj's default `guess_anchor_molecules` heuristic (built for
  # solute-in-solvent systems, "the anchor is whatever's much bigger than
  # everything else") has nothing to compare the one molecule against and
  # refuses to pick an anchor. Pass the single molecule as its own anchor
  # explicitly; this does not change what `image_molecules` computes, only
  # how it decides which molecule to treat as the fixed reference frame.
  (molecule,) = traj.topology.find_molecules()
  traj.image_molecules(inplace=True, anchor_molecules=[molecule])
  xyz_angstrom = traj.xyz * 10.0  # nm -> Angstrom

  n = len(atom_indices)
  iu, ju = np.triu_indices(n, k=1)
  selected = xyz_angstrom[:, atom_indices, :]  # (n_frames, n, 3)
  diffs = selected[:, iu, :] - selected[:, ju, :]  # (n_frames, n_pairs, 3)
  return np.sqrt((diffs**2).sum(axis=-1))  # (n_frames, n_pairs)


@pytest.mark.skipif(not MDTRAJ_AVAILABLE, reason="MDTraj not installed")
@pytest.mark.parametrize(
  "atom_indices, stride",
  [
    pytest.param([0, 5, 10, 15, 21], 1, id="5_atoms_stride1"),
    pytest.param([0, 2, 4, 6, 8, 10, 12, 21], 10, id="8_atoms_stride10"),
    pytest.param(list(range(22)), 25, id="all_atoms_stride25"),
  ],
)
def test_read_xtc_ca_distogram_matches_unwrap_then_naive_mdtraj(atom_indices, stride):
  """Cross-tool correctness: MIC-on-raw-coordinates (proxide) must agree
  with unwrap-then-naive-Euclidean (mdtraj) to within lossy-compression
  tolerance, across several atom subsets and strides.
  """
  _skip_if_fixture_missing()
  import proxide

  got = proxide.read_xtc_ca_distogram(str(XTC_PATH), atom_indices, stride=stride)
  expected = _reference_distogram_via_unwrap_then_naive(XTC_PATH, PDB_PATH, atom_indices, stride)

  n = len(atom_indices)
  n_pairs = n * (n - 1) // 2
  assert got.shape == expected.shape
  assert got.shape[1] == n_pairs

  max_diff = np.abs(got - expected).max()
  assert max_diff < DISTANCE_TOLERANCE_ANGSTROM, (
    f"MIC-on-raw (proxide) vs unwrap-then-naive (mdtraj) diverge by "
    f"{max_diff} Å (tolerance {DISTANCE_TOLERANCE_ANGSTROM} Å) for "
    f"atom_indices={atom_indices}, stride={stride}"
  )


@pytest.mark.skipif(not MDTRAJ_AVAILABLE, reason="MDTraj not installed")
def test_read_xtc_ca_distogram_sane_on_real_fixture():
  """Basic physical sanity on the full atom set: finite, non-negative,
  and within a plausible range for a 22-atom capped dipeptide (a handful
  of Angstroms end to end, certainly under 100 Å)."""
  _skip_if_fixture_missing()
  import proxide

  atom_indices = list(range(22))
  result = proxide.read_xtc_ca_distogram(str(XTC_PATH), atom_indices, stride=5)
  assert np.all(np.isfinite(result))
  assert np.all(result >= 0.0)
  assert np.all(result < 100.0)


@pytest.mark.skipif(not MDTRAJ_AVAILABLE, reason="MDTraj not installed")
def test_read_xtc_ca_distogram_synthetic_ground_truth_and_pair_order(tmp_path):
  """Synthetic ground-truth + explicit pair-order check, through the actual
  Python binding (not just the underlying Rust primitive — that's already
  covered exhaustively in `proxide-geometry`'s own Rust unit tests). Builds
  a tiny hand-picked 4-atom, 1-frame XTC with a known 10x10x10 A box, where
  every pairwise minimum-image distance can be computed by hand, including
  one pair (atoms 0 and 1) placed near opposite box faces along x — the
  case a naive non-PBC-aware distance calculation gets wrong (raw
  separation 9.0 A; true minimum-image separation 1.0 A).
  """
  import proxide

  topology = mdtraj.Topology()
  chain = topology.add_chain()
  residue = topology.add_residue("XXX", chain)
  for _ in range(4):
    topology.add_atom("C", mdtraj.element.carbon, residue)

  # Positions in Angstroms; box is 10x10x10 A (half-box = 5 A).
  #   0: (0.5, 5.0, 5.0)
  #   1: (9.5, 5.0, 5.0)  -- near-opposite face from atom 0 along x
  #   2: (5.0, 5.0, 5.0)  -- box center
  #   3: (5.0, 7.0, 5.0)
  # All pairwise |delta| are < 5 A (half-box) on every axis *except*
  # (0, 1)'s x-component (raw 9.0 A) -- the only pair this fixture wraps.
  positions_angstrom = np.array(
    [
      [0.5, 5.0, 5.0],
      [9.5, 5.0, 5.0],
      [5.0, 5.0, 5.0],
      [5.0, 7.0, 5.0],
    ],
    dtype=np.float32,
  )
  coords_nm = (positions_angstrom / 10.0).reshape(1, 4, 3)
  box_length_angstrom = 10.0
  traj = mdtraj.Trajectory(
    coords_nm,
    topology,
    unitcell_lengths=np.array([[box_length_angstrom / 10.0] * 3], dtype=np.float32),
    unitcell_angles=np.array([[90.0, 90.0, 90.0]], dtype=np.float32),
  )
  xtc_path = tmp_path / "synthetic_mic.xtc"
  traj.save_xtc(str(xtc_path))

  atom_indices = [0, 1, 2, 3]
  got = proxide.read_xtc_ca_distogram(str(xtc_path), atom_indices, stride=1)
  assert got.shape == (1, 6)

  # Hand-computed, in np.triu_indices(4, k=1) order: (0,1)(0,2)(0,3)(1,2)(1,3)(2,3)
  d01 = 1.0  # |0.5 - 9.5| = 9.0 raw -> minimum image 10.0 - 9.0 = 1.0
  d02 = 4.5  # |0.5 - 5.0|, no wrap (< 5 A half-box)
  d03 = float(np.hypot(4.5, 2.0))  # dx=|0.5-5.0|=4.5, dy=|5.0-7.0|=2.0, no wrap
  d12 = 4.5  # |9.5 - 5.0|, no wrap
  d13 = float(np.hypot(4.5, 2.0))  # dx=|9.5-5.0|=4.5, dy=2.0, no wrap
  d23 = 2.0  # |5.0 - 7.0|, no wrap
  expected = np.array([d01, d02, d03, d12, d13, d23], dtype=np.float32)

  np.testing.assert_allclose(
    got[0],
    expected,
    atol=COORD_TOLERANCE_ANGSTROM * 3,  # a distance combines 2 lossy coords
    err_msg="read_xtc_ca_distogram must match hand-computed minimum-image "
    "distances in exact np.triu_indices(4, k=1) order",
  )

  # Sanity: confirm this fixture really does distinguish MIC from a naive,
  # non-PBC-aware calculation — the (0, 1) raw separation really is 9.0 A,
  # not 1.0 A, so a bug that skipped wrapping entirely would be caught here.
  raw_d01 = float(np.linalg.norm(positions_angstrom[0] - positions_angstrom[1]))
  assert abs(raw_d01 - 9.0) < COORD_TOLERANCE_ANGSTROM
  assert abs(got[0, 0] - raw_d01) > 1.0, (
    "fixture failed to distinguish MIC from naive Euclidean distance"
  )
