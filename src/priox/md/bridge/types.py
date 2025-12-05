from typing import TypedDict
import jax.numpy as jnp

class SystemParams(TypedDict):
  """Parameters for a JAX MD system."""

  charges: jnp.ndarray  # (N,)
  masses: jnp.ndarray # (N,) Atomic masses (amu)
  sigmas: jnp.ndarray  # (N,)
  epsilons: jnp.ndarray  # (N,)
  gb_radii: jnp.ndarray  # (N,) Implicit solvent radii (mbondi2)
  scaled_radii: jnp.ndarray # (N,) Scaled radii for descreening (OBC2)
  bonds: jnp.ndarray  # (N_bonds, 2)
  bond_params: jnp.ndarray  # (N_bonds, 2) [length, k]
  constrained_bonds: jnp.ndarray # (N_constraints, 2)
  constrained_bond_lengths: jnp.ndarray # (N_constraints,)
  angles: jnp.ndarray  # (N_angles, 3)
  angle_params: jnp.ndarray  # (N_angles, 2) [theta0, k]
  backbone_indices: jnp.ndarray  # (N_residues, 4) [N, CA, C, O] indices
  exclusion_mask: jnp.ndarray  # (N, N) boolean mask (True = interact, False = exclude)
  scale_matrix_vdw: jnp.ndarray # (N, N) scaling factors for VDW
  scale_matrix_elec: jnp.ndarray # (N, N) scaling factors for Electrostatics
  dihedrals: jnp.ndarray  # (N_dihedrals, 4)
  dihedral_params: jnp.ndarray  # (N_dihedrals, 3) [periodicity, phase, k]
  impropers: jnp.ndarray  # (N_impropers, 4)
  improper_params: jnp.ndarray  # (N_impropers, 3) [periodicity, phase, k]
  cmap_energy_grids: jnp.ndarray # (N_maps, Grid, Grid)
  cmap_indices: jnp.ndarray # (N_torsions,) map index for each torsion
  cmap_torsions: jnp.ndarray # (N_torsions, 5) [i, j, k, l, map_idx]
  cmap_coeffs: jnp.ndarray # (N_maps, Grid, Grid, 4) [f, fx, fy, fxy]
