#!/usr/bin/env python3
"""Generate minimal trajectory test files for parity testing.

Creates XTC, DCD, and TRR files matching the 1crn.pdb topology (4 atoms).
Uses MDTraj for file generation.
"""

from pathlib import Path

import numpy as np

try:
  import mdtraj
except ImportError:
  print("MDTraj not installed. Run: uv pip install mdtraj")
  exit(1)


def generate_test_trajectories(output_dir: Path, num_frames: int = 10):
  """Generate test trajectory files.

  Args:
      output_dir: Directory to write trajectory files
      num_frames: Number of frames to generate

  """
  output_dir.mkdir(parents=True, exist_ok=True)

  # Create a simple 4-atom topology matching 1crn.pdb
  # N, CA, C, O of a single ALA residue
  topology = mdtraj.Topology()
  chain = topology.add_chain()
  residue = topology.add_residue("ALA", chain)

  n_atom = topology.add_atom("N", mdtraj.element.nitrogen, residue)
  ca_atom = topology.add_atom("CA", mdtraj.element.carbon, residue)
  c_atom = topology.add_atom("C", mdtraj.element.carbon, residue)
  o_atom = topology.add_atom("O", mdtraj.element.oxygen, residue)

  # Add bonds
  topology.add_bond(n_atom, ca_atom)
  topology.add_bond(ca_atom, c_atom)
  topology.add_bond(c_atom, o_atom)

  # Generate coordinates with slight motion
  # Base coordinates in nm (MDTraj internal unit)
  base_coords = np.array(
    [
      [0.00, 0.00, 0.00],  # N
      [0.15, 0.00, 0.00],  # CA (1.5 Å from N)
      [0.25, 0.10, 0.00],  # C
      [0.25, 0.20, 0.00],  # O
    ],
    dtype=np.float32,
  )

  # Create trajectory with small random motion
  np.random.seed(42)  # Reproducible
  coords = np.zeros((num_frames, 4, 3), dtype=np.float32)
  for i in range(num_frames):
    # Add small displacement (0.01 nm = 0.1 Å max)
    noise = np.random.randn(4, 3).astype(np.float32) * 0.01
    coords[i] = base_coords + noise

  # Create unit cell (orthorhombic box)
  # Lengths in nm, angles in degrees
  unitcell_lengths = np.array([[5.0, 5.0, 5.0]] * num_frames, dtype=np.float32)
  unitcell_angles = np.array([[90.0, 90.0, 90.0]] * num_frames, dtype=np.float32)

  # Create trajectory
  traj = mdtraj.Trajectory(
    coords,
    topology,
    unitcell_lengths=unitcell_lengths,
    unitcell_angles=unitcell_angles,
  )

  # Save in different formats
  print(f"Generating test trajectories with {num_frames} frames, 4 atoms each...")

  # XTC format (GROMACS compressed trajectory)
  xtc_path = output_dir / "test.xtc"
  traj.save_xtc(str(xtc_path))
  print(f"  ✓ XTC: {xtc_path}")

  # DCD format (CHARMM/NAMD)
  dcd_path = output_dir / "test.dcd"
  traj.save_dcd(str(dcd_path))
  print(f"  ✓ DCD: {dcd_path}")

  # TRR format (GROMACS full precision)
  trr_path = output_dir / "test.trr"
  traj.save_trr(str(trr_path))
  print(f"  ✓ TRR: {trr_path}")

  # Also update the 1crn.pdb to match this topology
  pdb_path = output_dir.parent / "1crn.pdb"
  traj[0].save_pdb(str(pdb_path))
  print(f"  ✓ PDB topology: {pdb_path}")

  # Verify files
  print("\nVerifying generated files...")
  for fmt, path in [("XTC", xtc_path), ("DCD", dcd_path), ("TRR", trr_path)]:
    loaded = mdtraj.load(str(path), top=str(pdb_path))
    print(f"  {fmt}: {loaded.n_frames} frames, {loaded.n_atoms} atoms")
    # Check coordinate range
    coord_range = (loaded.xyz.min(), loaded.xyz.max())
    print(f"       Coords: [{coord_range[0]:.3f}, {coord_range[1]:.3f}] nm")

  print("\n✅ Test trajectory files generated successfully!")
  return True


def generate_large_xtc_fixture(output_dir: Path, num_frames: int = 5000):
  """Generate a larger multi-thousand-frame XTC fixture.

  Same 4-atom topology as `generate_test_trajectories`; only the frame count
  differs. Exists to exercise/benchmark the offset-index scan, the on-disk
  offset cache, and parallel frame decoding on a trajectory too large for a
  single-digit frame count to meaningfully test.

  Args:
      output_dir: Directory to write the trajectory file
      num_frames: Number of frames to generate

  """
  output_dir.mkdir(parents=True, exist_ok=True)

  topology = mdtraj.Topology()
  chain = topology.add_chain()
  residue = topology.add_residue("ALA", chain)

  n_atom = topology.add_atom("N", mdtraj.element.nitrogen, residue)
  ca_atom = topology.add_atom("CA", mdtraj.element.carbon, residue)
  c_atom = topology.add_atom("C", mdtraj.element.carbon, residue)
  o_atom = topology.add_atom("O", mdtraj.element.oxygen, residue)

  topology.add_bond(n_atom, ca_atom)
  topology.add_bond(ca_atom, c_atom)
  topology.add_bond(c_atom, o_atom)

  base_coords = np.array(
    [
      [0.00, 0.00, 0.00],
      [0.15, 0.00, 0.00],
      [0.25, 0.10, 0.00],
      [0.25, 0.20, 0.00],
    ],
    dtype=np.float32,
  )

  np.random.seed(1337)
  coords = np.zeros((num_frames, 4, 3), dtype=np.float32)
  for i in range(num_frames):
    noise = np.random.randn(4, 3).astype(np.float32) * 0.01
    coords[i] = base_coords + noise

  unitcell_lengths = np.array([[5.0, 5.0, 5.0]] * num_frames, dtype=np.float32)
  unitcell_angles = np.array([[90.0, 90.0, 90.0]] * num_frames, dtype=np.float32)

  traj = mdtraj.Trajectory(
    coords,
    topology,
    unitcell_lengths=unitcell_lengths,
    unitcell_angles=unitcell_angles,
  )

  xtc_path = output_dir / "large.xtc"
  traj.save_xtc(str(xtc_path))
  print(f"  ✓ Large XTC ({num_frames} frames): {xtc_path}")

  pdb_path = output_dir.parent / "1crn.pdb"
  loaded = mdtraj.load(str(xtc_path), top=str(pdb_path))
  print(f"    Verified: {loaded.n_frames} frames, {loaded.n_atoms} atoms")

  return True


def generate_box_drift_xtc_fixture(output_dir: Path, num_frames: int = 4000):
  """Generate a many-frame XTC fixture with a per-frame VARYING, slightly
  triclinic box, for regression-testing box-vector decode. Built while
  investigating praxia debt #1237 / proxide#16, a reported box-vector
  "corruption" that turned out on deep root-causing to be a representation
  mismatch (mdtraj's high-level API reduces box vectors to lengths+angles
  and reconstructs them in its own canonical orientation) rather than an
  actual proxide decode bug — see the Rust test that consumes this fixture
  for the full writeup. Kept as a general-purpose decode regression guard.

  Unlike `generate_test_trajectories`/`generate_large_xtc_fixture` (both use a
  constant [5,5,5] nm orthorhombic box every frame), the ground truth here
  varies frame to frame — a static or purely-diagonal fixture can't catch a
  drift bug (an accumulating byte-offset error only shows up once later
  frames' bytes actually differ from frame 0's), and a purely-orthorhombic
  fixture can't catch a row/column transpose bug either (transposing a
  diagonal matrix is a no-op). This fixture varies box *lengths* linearly
  across frames AND introduces small, deterministic, per-frame *tilt*
  (off-diagonal) terms via `unitcell_vectors`, so both classes of bug are
  visible at every frame past the first, not just statically.

  Uses a larger (64-atom) topology than `generate_large_xtc_fixture`'s 4-atom
  one so frames go through XTC's real compressed-coordinate path (used for
  natoms > 9) as in a real MD trajectory, matching the conditions under which
  the corruption was originally found.

  Args:
      output_dir: Directory to write the trajectory file
      num_frames: Number of frames to generate

  """
  output_dir.mkdir(parents=True, exist_ok=True)

  n_atoms = 64
  topology = mdtraj.Topology()
  chain = topology.add_chain()
  for i in range(n_atoms):
    residue = topology.add_residue("ALA", chain, resSeq=i)
    topology.add_atom("CA", mdtraj.element.carbon, residue)

  # Simple extended chain, 3.8 Å CA-CA spacing, in nm.
  base_coords = np.zeros((n_atoms, 3), dtype=np.float32)
  base_coords[:, 0] = np.arange(n_atoms, dtype=np.float32) * 0.38

  np.random.seed(2026)
  coords = np.zeros((num_frames, n_atoms, 3), dtype=np.float32)
  for i in range(num_frames):
    noise = np.random.randn(n_atoms, 3).astype(np.float32) * 0.02
    coords[i] = base_coords + noise

  # Ground-truth box: unitcell_vectors directly (nm), one 3x3 matrix per
  # frame, row i = box vector i (mdtraj/GROMACS convention) — this is the
  # exact quantity proxide's `box_vectors` output must match once converted
  # to Angstroms.
  #
  # Lengths vary linearly over the trajectory (isotropic-NPT-like drift);
  # a small deterministic sinusoidal tilt is added to the off-diagonal
  # entries so every frame past 0 has nonzero off-diagonal ground truth,
  # bounded well clear of mdtraj's cell-validity requirements.
  t = np.arange(num_frames, dtype=np.float64) / max(num_frames - 1, 1)
  a_len = 5.0 + 0.5 * t  # 5.0 -> 5.5 nm
  b_len = 5.0 + 0.3 * t  # 5.0 -> 5.3 nm
  c_len = 6.0 + 0.8 * t  # 6.0 -> 6.8 nm
  tilt = 0.05 * np.sin(2 * np.pi * t * 7.0)  # nm, small deterministic wobble

  unitcell_vectors = np.zeros((num_frames, 3, 3), dtype=np.float32)
  unitcell_vectors[:, 0, 0] = a_len
  unitcell_vectors[:, 1, 0] = tilt  # b vector's x-component (tilt)
  unitcell_vectors[:, 1, 1] = b_len
  unitcell_vectors[:, 2, 0] = tilt * 0.5  # c vector's x-component (tilt)
  unitcell_vectors[:, 2, 1] = tilt * 0.5  # c vector's y-component (tilt)
  unitcell_vectors[:, 2, 2] = c_len

  traj = mdtraj.Trajectory(coords, topology)
  traj.unitcell_vectors = unitcell_vectors

  xtc_path = output_dir / "box_drift.xtc"
  traj.save_xtc(str(xtc_path))
  print(f"  ✓ Box-drift XTC ({num_frames} frames, {n_atoms} atoms): {xtc_path}")

  # Also persist the ground-truth box vectors (Angstroms) as a flat
  # little-endian f32 binary blob, shape (num_frames, 3, 3) row-major, that
  # the Rust regression test reads directly with no crate beyond std — no
  # need to re-derive or hardcode this array in Rust, and no .npy parsing
  # dependency required at test time.
  ground_truth_ang = (unitcell_vectors.astype(np.float64) * 10.0).astype("<f4")
  bin_path = output_dir / "box_drift_ground_truth_angstrom.bin"
  ground_truth_ang.tofile(bin_path)
  print(f"  ✓ Ground-truth box vectors (Angstrom, f32 LE flat): {bin_path}")

  # Sanity: round-trip through mdtraj itself must match what we asked for,
  # to floating-point precision (XTC box storage is uncompressed f32).
  reloaded = mdtraj.load(str(xtc_path), top=topology)
  max_diff = np.abs(reloaded.unitcell_vectors - unitcell_vectors).max()
  print(f"    mdtraj round-trip max abs diff (nm): {max_diff:.3e}")
  assert max_diff < 1e-4, "mdtraj itself failed to round-trip the box vectors we asked for"

  return True


if __name__ == "__main__":
  script_dir = Path(__file__).parent
  project_root = script_dir.parent
  output_dir = project_root / "tests" / "data" / "trajectories"

  generate_test_trajectories(output_dir)
  generate_large_xtc_fixture(output_dir)
  generate_box_drift_xtc_fixture(output_dir)
