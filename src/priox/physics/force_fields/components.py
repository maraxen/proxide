"""Modular components for Force Field parameters."""

from __future__ import annotations

from typing import Any, Optional

import equinox as eqx
import jax.numpy as jnp


class AtomTypeParams(eqx.Module):
    """Parameters associated with atom types."""
    charges: jnp.ndarray
    sigmas: jnp.ndarray
    epsilons: jnp.ndarray
    radii: jnp.ndarray
    scales: jnp.ndarray
    
    # Metadata maps
    atom_key_to_id: dict[tuple[str, str], int] = eqx.field(static=True)
    id_to_atom_key: list[tuple[str, str]] = eqx.field(static=True)
    atom_class_map: dict[str, str] = eqx.field(static=True)
    atom_type_map: dict[str, str] = eqx.field(static=True)


class BondPotentialParams(eqx.Module):
    """Parameters for bond potentials."""
    # (class1, class2, length, k)
    params: list[tuple[str, str, float, float]] = eqx.field(static=True)


class AnglePotentialParams(eqx.Module):
    """Parameters for angle potentials."""
    # (class1, class2, class3, theta, k)
    params: list[tuple[str, str, str, float, float]] = eqx.field(static=True)


class DihedralPotentialParams(eqx.Module):
    """Parameters for dihedral potentials (proper and improper)."""
    # List of dicts with 'classes' and 'terms'
    propers: list[dict[str, Any]] = eqx.field(static=True)
    impropers: list[dict[str, Any]] = eqx.field(static=True)


class CMAPParams(eqx.Module):
    """Parameters for CMAP potentials."""
    energy_grids: jnp.ndarray  # (n_maps, grid_size, grid_size)
    torsions: list[dict[str, Any]] = eqx.field(static=True)


class UreyBradleyParams(eqx.Module):
    """Parameters for Urey-Bradley potentials."""
    # (class1, class2, length, k)
    params: list[tuple[str, str, float, float]] = eqx.field(static=True)


class VirtualSiteParams(eqx.Module):
    """Parameters for virtual sites."""
    # residue -> list of vs defs
    definitions: dict[str, list[dict[str, Any]]] = eqx.field(static=True)


class NonbondedGlobalParams(eqx.Module):
    """Global parameters for non-bonded interactions."""
    coulomb14scale: float = 0.833333
    lj14scale: float = 0.5
    cutoff_distance: float = 10.0  # Angstroms
    switch_distance: float = 9.0   # Angstroms (if using switch)
    use_dispersion_correction: bool = True
    use_pme: bool = False
    ewald_error_tolerance: float = 0.0005
    dielectric_constant: float = 78.5  # Solvent dielectric (usually implicit, or PME calc)


class GAFFNonbondedParams(eqx.Module):
    """GAFF-style nonbonded parameters indexed by atom type."""
    # Maps GAFF atom type (e.g., "ca", "c3") -> index
    type_to_index: dict[str, int] = eqx.field(static=True)
    sigmas: jnp.ndarray  # (n_types,) in Angstroms
    epsilons: jnp.ndarray  # (n_types,) in kcal/mol

