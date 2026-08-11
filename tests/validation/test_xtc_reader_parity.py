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

import subprocess
import sys
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


# ---------------------------------------------------------------------------
# Memory-scaling regression for the read_frames_parallel OOM fix
# (praxia debt #1220): before the fix, `read_frames_parallel` decoded every
# requested frame at *full* atom count and collected all of them into one
# Vec before any atom-index filtering happened (that filtering lived only in
# the PyO3 binding layer, applied after collection). Peak memory was
# therefore ~independent of how many atoms the caller actually asked for.
# The fix filters inside each worker, immediately after that worker's single
# frame is decoded, so the collected result — and its peak memory — scales
# with `len(atom_indices)`, not the trajectory's full per-frame atom count.
#
# A large atom count with a modest frame count keeps fixture generation and
# decode time reasonable while still making the old-vs-new memory gap large
# enough to be unmistakable against ordinary Python/numpy/mdtraj process
# overhead.
MEMORY_TEST_N_ATOMS = 8_000
MEMORY_TEST_N_FRAMES = 1_500
# Selecting 5 atoms out of MEMORY_TEST_N_ATOMS should peak at roughly the
# Python/mdtraj-import baseline (empirically ~40MB on the dev machine this
# was calibrated on) — nowhere near the ~144MB that
# `MEMORY_TEST_N_ATOMS * MEMORY_TEST_N_FRAMES * 3 * 4` bytes would cost if
# the old bug's "collect every frame at full atom count, then filter" order
# were still in effect (empirically ~180MB on the same machine, before this
# fix). The ceiling below sits roughly in between — comfortably above the
# fixed-code baseline, comfortably below the old-code floor — so it
# discriminates the two rather than just confirming "some savings exist"
# (an earlier version of this test asserted a *relative* full-vs-subset
# delta instead of this absolute ceiling; that version passed even against
# the unfixed code, because the unfixed code's *full* read pays for an
# additional output-buffer copy on top of the same undischarged
# full-atom-count intermediate buffer its *subset* read also retains — so
# unfixed code shows a large full-vs-subset delta too, just for a different
# reason. The absolute ceiling on the subset call specifically is what
# actually catches the regression; verified by deliberately reverting the
# fix locally and confirming this assertion fails).
MEMORY_SUBSET_CEILING_KB = 100_000


def _build_large_atom_count_xtc(path: Path, n_atoms: int, n_frames: int, seed: int = 2024) -> None:
  """Write a synthetic XTC with many atoms but a single unbonded residue —
  cheap to construct a `mdtraj.Topology` for (no per-atom bond bookkeeping),
  since this fixture only needs a valid atom count, not realistic chemistry.
  """
  topology = mdtraj.Topology()
  chain = topology.add_chain()
  residue = topology.add_residue("XXX", chain)
  for _ in range(n_atoms):
    topology.add_atom("C", mdtraj.element.carbon, residue)

  rng = np.random.default_rng(seed)
  coords = (rng.standard_normal((n_frames, n_atoms, 3)) * 0.1).astype(np.float32)
  unitcell_lengths = np.array([[50.0, 50.0, 50.0]] * n_frames, dtype=np.float32)
  unitcell_angles = np.array([[90.0, 90.0, 90.0]] * n_frames, dtype=np.float32)

  traj = mdtraj.Trajectory(
    coords,
    topology,
    unitcell_lengths=unitcell_lengths,
    unitcell_angles=unitcell_angles,
  )
  traj.save_xtc(str(path))


@pytest.fixture(scope="module")
def large_atom_count_xtc(tmp_path_factory):
  if not MDTRAJ_AVAILABLE:
    pytest.skip("MDTraj not installed")
  path = tmp_path_factory.mktemp("xtc_memory_scaling") / "large_atom_count.xtc"
  _build_large_atom_count_xtc(path, MEMORY_TEST_N_ATOMS, MEMORY_TEST_N_FRAMES)
  return path


# NOTE on the measurement approach: this deliberately reads `VmHWM` from
# /proc/<pid>/status inside the *child* process itself, rather than the more
# obvious `resource.getrusage(RUSAGE_SELF).ru_maxrss` after a
# `subprocess.run(...)`. That more obvious approach was tried first and
# empirically found to be broken on this platform (WSL2): a freshly
# fork+exec'd child's own `ru_maxrss`, read immediately after `exec()`,
# reflects the *parent* process's resident set size at fork time, not
# anything the child itself allocated — confirmed by spawning a trivial
# child that does nothing but read its own `ru_maxrss` from a parent that
# had just touched a real ~300MB allocation, and seeing the child report
# that same ~300MB. `/proc/self/status`'s `VmHWM` does not have this
# problem (verified the same way): it reflects only the reading process's
# own peak resident set, seeded fresh after `execve()`. Per the "verify your
# measurement pipeline before trusting a research conclusion" discipline —
# this is exactly the kind of gotcha that check exists to catch.
_PEAK_VMHWM_CHILD_SCRIPT = """
import sys

import proxide

path = sys.argv[1]
raw_indices = sys.argv[2]
atom_indices = None if raw_indices == "None" else [int(x) for x in raw_indices.split(",")]

proxide.read_xtc_parallel(path, atom_indices=atom_indices)

with open("/proc/self/status") as f:
    for line in f:
        if line.startswith("VmHWM:"):
            print(int(line.split()[1]))
            break
""".strip()


def _peak_vmhwm_kb_for_read(path: Path, atom_indices: list[int] | None) -> int:
  """Run a single `proxide.read_xtc_parallel` call in a fresh subprocess and
  return that subprocess's own peak resident-set size (`VmHWM`, KB) as seen
  from inside the child itself.

  A fresh subprocess per call (rather than measuring both calls in this test
  process) is deliberate: `VmHWM` is a high-water mark that never decreases
  within a process, so measuring two calls back-to-back in one process would
  have the first call's allocation inflate the second call's reported peak
  regardless of which one actually needs more memory.
  """
  raw_indices = "None" if atom_indices is None else ",".join(str(i) for i in atom_indices)
  result = subprocess.run(
    [sys.executable, "-c", _PEAK_VMHWM_CHILD_SCRIPT, str(path), raw_indices],
    capture_output=True,
    text=True,
    check=True,
  )
  return int(result.stdout.strip().splitlines()[-1])


@pytest.mark.skipif(not MDTRAJ_AVAILABLE, reason="MDTraj not installed")
@pytest.mark.skipif(
  sys.platform != "linux",
  reason="/proc/<pid>/status VmHWM is Linux-specific (this repo's CI is Linux-only)",
)
def test_read_xtc_parallel_peak_memory_scales_with_atom_selection(large_atom_count_xtc):
  """Requesting a handful of atoms out of `MEMORY_TEST_N_ATOMS` must peak at
  roughly process-baseline memory, not memory proportional to the full atom
  count — the direct regression test for the OOM this fix addresses (praxia
  debt #1220). Before the fix, `read_frames_parallel` collected every
  requested frame's *full*-atom-count `MollyFrame` into one `Vec` — kept
  alive for the rest of the call — before any atom-index filtering ran (that
  filtering lived only in the PyO3 binding layer, applied after collection).
  A tiny `atom_indices` subset therefore used to pay for the same
  undiscarded full-atom-count buffer as an unselected read.

  Deliberately asserts an absolute ceiling on the *subset* call alone,
  rather than a full-vs-subset delta: the old, unfixed code also shows a
  large full-vs-subset delta (its full read pays for an extra output-buffer
  copy on top of the same undischarged full-atom-count buffer its subset
  read also retains), so a delta-only assertion cannot tell old from fixed
  behavior. This was caught empirically — by deliberately reverting the fix
  locally and rerunning this test — before landing on the ceiling below;
  see MEMORY_SUBSET_CEILING_KB's comment for the calibration.
  """
  path = large_atom_count_xtc

  subset_kb = _peak_vmhwm_kb_for_read(path, [0, 1, 2, 3, 4])
  assert subset_kb < MEMORY_SUBSET_CEILING_KB, (
    f"selecting 5/{MEMORY_TEST_N_ATOMS} atoms peaked at {subset_kb}KB, at or "
    f"above the {MEMORY_SUBSET_CEILING_KB}KB ceiling — peak memory for a "
    "tiny atom_indices subset should stay near process-baseline, not scale "
    "with the trajectory's full per-frame atom count"
  )

  # Sanity check that the full (unselected) read is still substantially
  # larger — confirms the fixture and read path are actually exercising a
  # real full-vs-subset size difference, not e.g. both calls degenerately
  # returning near-zero data.
  full_kb = _peak_vmhwm_kb_for_read(path, None)
  assert full_kb > subset_kb, (
    f"expected a full read (atom_indices=None) across {MEMORY_TEST_N_ATOMS} "
    f"atoms to peak higher than a 5-atom subset; full={full_kb}KB "
    f"subset={subset_kb}KB"
  )
