from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
  from proxide.types import (
    AngleParams,
    Angles,
    BackboneIndices,
    BondParams,
    Bonds,
    Charges,
    CmapCoeffs,
    CmapIndices,
    CmapTorsions,
    ConstrainedBondLengths,
    ConstrainedBonds,
    DihedralParams,
    Dihedrals,
    EnergyGrids,
    Epsilons,
    ExclusionMask,
    ImproperParams,
    Impropers,
    Masses,
    Radii,
    ScaleMatrix,
    Sigmas,
    UreyBradleyBonds,
    UreyBradleyParams,
    VirtualSiteDef,
    VirtualSiteParams,
  )


class SystemParams(TypedDict):
  """Parameters for a JAX MD system."""

  charges: Charges  # (N,)
  masses: Masses  # (N,) Atomic masses (amu)
  sigmas: Sigmas  # (N,)
  epsilons: Epsilons  # (N,)
  gb_radii: Radii  # (N,) Implicit solvent radii (mbondi2)
  scaled_radii: Radii  # (N,) Scaled radii for descreening (OBC2)
  bonds: Bonds  # (N_bonds, 2)
  bond_params: BondParams  # (N_bonds, 2) [length, k]
  constrained_bonds: ConstrainedBonds  # (N_constraints, 2)
  constrained_bond_lengths: ConstrainedBondLengths  # (N_constraints,)
  angles: Angles  # (N_angles, 3)
  angle_params: AngleParams  # (N_angles, 2) [theta0, k]
  backbone_indices: BackboneIndices  # (N_residues, 4) [N, CA, C, O] indices
  exclusion_mask: ExclusionMask  # (N, N) boolean mask (True = interact, False = exclude)
  scale_matrix_vdw: ScaleMatrix  # (N, N) scaling factors for VDW
  scale_matrix_elec: ScaleMatrix  # (N, N) scaling factors for Electrostatics
  dihedrals: Dihedrals  # (N_dihedrals, 4)
  dihedral_params: DihedralParams  # (N_dihedrals, 3) [periodicity, phase, k]
  impropers: Impropers  # (N_impropers, 4)
  improper_params: ImproperParams  # (N_impropers, 3) [periodicity, phase, k]
  cmap_energy_grids: EnergyGrids  # (N_maps, Grid, Grid)
  cmap_indices: CmapIndices  # (N_torsions,) map index for each torsion
  cmap_torsions: CmapTorsions  # (N_torsions, 5) [i, j, k, l, map_idx]
  cmap_coeffs: CmapCoeffs  # (N_maps, Grid, Grid, 4) [f, fx, fy, fxy]
  urey_bradley_bonds: UreyBradleyBonds  # (N_ub, 2) [atom1, atom2] (1-3 interaction)
  urey_bradley_params: UreyBradleyParams  # (N_ub, 2) [d, k]
  virtual_site_def: VirtualSiteDef  # (N_vs, 4) [vs_idx, parent1, parent2, parent3]
  virtual_site_params: VirtualSiteParams  # (N_vs, 12)
