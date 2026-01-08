"""Modular components for Force Field parameters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
from flax.struct import dataclass, field
from jaxtyping import Float

if TYPE_CHECKING:
  from proxide.types import Charges, EnergyGrids, Epsilons, Radii, Scales, Sigmas


@dataclass(frozen=True)
class AtomTypeParams:
  """Parameters associated with atom types."""

  charges: Charges
  sigmas: Sigmas
  epsilons: Epsilons
  radii: Radii
  scales: Scales

  # Metadata maps
  atom_key_to_id: dict[tuple[str, str], int] = field(pytree_node=False)
  id_to_atom_key: list[tuple[str, str]] = field(pytree_node=False)
  atom_class_map: dict[str, str] = field(pytree_node=False)
  atom_type_map: dict[str, str] = field(pytree_node=False)


@dataclass(frozen=True)
class BondPotentialParams:
  """Parameters for bond potentials."""

  # (class1, class2, length, k)
  params: list[tuple[str, str, float, float]] = field(pytree_node=False)


@dataclass(frozen=True)
class AnglePotentialParams:
  """Parameters for angle potentials."""

  # (class1, class2, class3, theta, k)
  params: list[tuple[str, str, str, float, float]] = field(pytree_node=False)


@dataclass(frozen=True)
class DihedralPotentialParams:
  """Parameters for dihedral potentials (proper and improper)."""

  # List of dicts with 'classes' and 'terms'
  propers: list[dict[str, Any]] = field(pytree_node=False)
  impropers: list[dict[str, Any]] = field(pytree_node=False)


@dataclass(frozen=True)
class CMAPParams:
  """Parameters for CMAP potentials."""

  energy_grids: EnergyGrids  # (n_maps, grid_size, grid_size)
  torsions: list[dict[str, Any]] = field(pytree_node=False)


@dataclass(frozen=True)
class UreyBradleyParams:
  """Parameters for Urey-Bradley potentials."""

  # (class1, class2, length, k)
  params: list[tuple[str, str, float, float]] = field(pytree_node=False)


@dataclass(frozen=True)
class VirtualSiteParams:
  """Parameters for virtual sites."""

  # residue -> list of vs defs
  definitions: dict[str, list[dict[str, Any]]] = field(pytree_node=False)


@dataclass(frozen=True)
class NonbondedGlobalParams:
  """Global parameters for non-bonded interactions."""

  coulomb14scale: float = 0.833333
  lj14scale: float = 0.5
  cutoff_distance: float = 10.0  # Angstroms
  switch_distance: float = 9.0  # Angstroms (if using switch)
  use_dispersion_correction: bool = True
  use_pme: bool = False
  ewald_error_tolerance: float = 0.0005
  dielectric_constant: float = 78.5  # Solvent dielectric (usually implicit, or PME calc)


@dataclass(frozen=True)
class GAFFNonbondedParams:
  """GAFF-style nonbonded parameters indexed by atom type."""

  # Maps GAFF atom type (e.g., "ca", "c3") -> index
  type_to_index: dict[str, int] = field(pytree_node=False)
  sigmas: Float[jnp.ndarray, "n_types"]  # noqa: F821, UP037
  epsilons: Float[jnp.ndarray, "n_types"]  # noqa: F821, UP037
