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


if __name__ == "__main__":
  script_dir = Path(__file__).parent
  project_root = script_dir.parent
  output_dir = project_root / "tests" / "data" / "trajectories"

  generate_test_trajectories(output_dir)
