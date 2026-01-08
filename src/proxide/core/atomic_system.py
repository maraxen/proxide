"""Atomic system definitions for PrioX.

This module defines the hierarchical dataclasses for atomic systems:
- MolecularTopology: Per-atom identity and connectivity
- AtomicState: Per-atom dynamic state (coordinates, velocities)
- AtomicConstants: Per-atom physics parameters
- AtomicSystem: Composite wrapper with backward-compatible delegation
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
from flax.struct import dataclass

if TYPE_CHECKING:
  pass

from proxide.core.types import (
  AngleParams,
  Angles,
  AtomMask,
  AtomsMask,
  AtomTypes,
  BondParams,
  Bonds,
  BoxVectors,
  Charges,
  CmapGrid,
  CmapIndices,
  Coordinates,
  DihedralParams,
  Elements,
  Epsilons,
  ImproperParams,
  Impropers,
  Masses,
  MoleculeType,
  PerAtomChainIndex,
  PerAtomResidueIndex,
  ProperDihedrals,
  Radii,
  Sigmas,
  Velocities,
)


@dataclass(kw_only=True)
class MolecularTopology:
  """Per-atom identity and connectivity data.

  All arrays are indexed by atom, with shape (N_atoms,) unless otherwise noted.
  For fields like residue_index and chain_index, a value of -1 indicates
  non-protein atoms (ligand, solvent, ion).

  Attributes:
    elements: Atomic element symbols, e.g., ["C", "N", "O"]. Shape (N_atoms,).
    atom_names: PDB atom names, e.g., ["CA", "CB", "N"]. Shape (N_atoms,).
    residue_index: Residue ID for each atom. -1 for non-protein. Shape (N_atoms,).
    chain_index: Chain ID for each atom. -1 for non-protein. Shape (N_atoms,).
    molecule_type: Molecule type per atom. 0=protein, 1=ligand, 2=solvent, 3=ion.
      Shape (N_atoms,).
    bonds: Covalent bond pairs as atom indices. Shape (N_bonds, 2).
    angles: Angle triplets as atom indices (i-j-k). Shape (N_angles, 3).
    proper_dihedrals: Proper dihedral quartets. Shape (N_dihedrals, 4).
    impropers: Improper dihedral quartets for planar groups. Shape (N_impropers, 4).

  """

  elements: Elements | None = None
  atom_names: Sequence[str] | None = None
  residue_index: PerAtomResidueIndex | None = None
  chain_index: PerAtomChainIndex | None = None
  molecule_type: MoleculeType | None = None
  bonds: Bonds | None = None
  angles: Angles | None = None
  proper_dihedrals: ProperDihedrals | None = None
  impropers: Impropers | None = None
  # Optional residue-level info for protein conversion
  res_names: Sequence[str] | None = None
  atom_types: AtomTypes | None = None  # Force field atom types (GAFF etc)
  # CMAP indices for backbone corrections
  cmap_indices: CmapIndices | None = None

  @property
  def protein_mask(self) -> AtomsMask | None:
    """Boolean mask for protein atoms (molecule_type == 0)."""
    if self.molecule_type is None:
      return None
    return self.molecule_type == 0

  @property
  def ligand_mask(self) -> AtomsMask | None:
    """Boolean mask for ligand atoms (molecule_type == 1)."""
    if self.molecule_type is None:
      return None
    return self.molecule_type == 1

  @property
  def solvent_mask(self) -> AtomsMask | None:
    """Boolean mask for solvent atoms (molecule_type == 2)."""
    if self.molecule_type is None:
      return None
    return self.molecule_type == 2

  @property
  def ion_mask(self) -> AtomsMask | None:
    """Boolean mask for ion atoms (molecule_type == 3)."""
    if self.molecule_type is None:
      return None
    return self.molecule_type == 3

  @property
  def num_bonds(self) -> int:
    """Number of bonds in the system."""
    if self.bonds is None:
      return 0
    return len(self.bonds)

  @property
  def num_angles(self) -> int:
    """Number of angles in the system."""
    if self.angles is None:
      return 0
    return len(self.angles)

  @property
  def num_dihedrals(self) -> int:
    """Number of proper dihedrals in the system."""
    if self.proper_dihedrals is None:
      return 0
    return len(self.proper_dihedrals)


@dataclass(kw_only=True)
class AtomicState:
  """Per-atom dynamic state data.

  Contains the time-varying properties of the system such as positions,
  velocities, and periodic box information.

  Attributes:
    coordinates: Cartesian atom positions in Angstroms. Shape (N_atoms, 3).
    box_vectors: Periodic box vectors for PBC. Shape (3, 3). None if non-periodic.
    velocities: Atom velocities in Angstroms/ps. Shape (N_atoms, 3). Optional.

  """

  coordinates: Coordinates
  box_vectors: BoxVectors | None = None
  velocities: Velocities | None = None


@dataclass(kw_only=True)
class AtomicConstants:
  """Per-atom physics parameters.

  Contains force field parameters and physical constants for each atom.
  All arrays have shape (N_atoms,) unless otherwise noted.

  Attributes:
    charges: Partial atomic charges in elementary charge units. Shape (N_atoms,).
    sigmas: Lennard-Jones sigma parameters in Angstroms. Shape (N_atoms,).
    epsilons: Lennard-Jones epsilon parameters in kcal/mol. Shape (N_atoms,).
    masses: Atomic masses in amu. Shape (N_atoms,).
    radii: Atomic radii (e.g., for GBSA). Shape (N_atoms,).
    bond_params: Bond force field params [length, k]. Shape (N_bonds, 2).
    angle_params: Angle force field params [theta, k]. Shape (N_angles, 2).
    dihedral_params: Dihedral params [periodicity, phase, k]. Shape (N_dihedrals, 3).
    improper_params: Improper dihedral params. Shape (N_impropers, 3).
    cmap_grid: CMAP energy grid for backbone corrections. Shape (grid_size, grid_size).

  """

  charges: Charges | None = None
  sigmas: Sigmas | None = None
  epsilons: Epsilons | None = None
  masses: Masses | None = None
  radii: Radii | None = None
  bond_params: BondParams | None = None
  angle_params: AngleParams | None = None
  dihedral_params: DihedralParams | None = None
  improper_params: ImproperParams | None = None
  cmap_grid: CmapGrid | None = None


@dataclass(kw_only=True)
class AtomicSystem:
  """Composite wrapper for atomic data.

  Provides a hierarchical structure separating topology, state, and constants.
  Backward-compatible attribute access is provided via __getattr__ delegation.

  Attributes:
    topology: Static connectivity and identity data (MolecularTopology).
    state: Dynamic positions and velocities (AtomicState).
    constants: Physics parameters like charges/LJ params (AtomicConstants).
    atom_mask: Per-atom validity mask. Shape (N_atoms,). 1.0=valid, 0.0=padding.
    source: Optional source file path for provenance.
    _rust_obj: Internal reference to Rust-side object if available.

  Example:
    >>> system = AtomicSystem(
    ...     topology=MolecularTopology(elements=["C", "N", "O"]),
    ...     state=AtomicState(coordinates=jnp.zeros((3, 3))),
    ... )
    >>> system.coordinates  # Delegates to system.state.coordinates
    >>> system.elements  # Delegates to system.topology.elements

  """

  topology: MolecularTopology
  state: AtomicState
  constants: AtomicConstants | None = None
  atom_mask: AtomMask | None = None
  source: str | None = None
  _rust_obj: Any = None

  def __getattr__(self, name: str) -> Any:
    """Delegate attribute access to sub-components for backward compatibility."""
    # Check sub-components in order: state, topology, constants
    for component in [self.state, self.topology, self.constants]:
      if component is not None and hasattr(component, name):
        return getattr(component, name)
    raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

  @property
  def num_atoms(self) -> int:
    """Total number of atoms in the system."""
    return self.state.coordinates.shape[0]

  @property
  def num_protein_atoms(self) -> int:
    """Number of protein atoms."""
    if self.topology.molecule_type is None:
      if self.atom_mask is not None:
        return int(jnp.sum(self.atom_mask))
      return self.num_atoms
    protein_mask = self.topology.molecule_type == 0
    if self.atom_mask is not None:
      return int(jnp.sum(protein_mask & (self.atom_mask > 0)))
    return int(jnp.sum(protein_mask))

  @property
  def num_ligand_atoms(self) -> int:
    """Number of ligand atoms."""
    if self.topology.molecule_type is None:
      return 0
    return int(jnp.sum(self.topology.molecule_type == 1))

  @property
  def has_ligands(self) -> bool:
    """Check if system contains any ligand atoms."""
    return self.num_ligand_atoms > 0

  @property
  def has_solvent(self) -> bool:
    """Check if system contains any solvent atoms."""
    if self.topology.molecule_type is None:
      return False
    return bool(jnp.any(self.topology.molecule_type == 2))

  @classmethod
  def from_arrays(
    cls,
    coordinates: Coordinates,
    atom_mask: AtomMask | None = None,
    elements: Elements | None = None,
    atom_names: Sequence[str] | None = None,
    residue_index: PerAtomResidueIndex | None = None,
    chain_index: PerAtomChainIndex | None = None,
    molecule_type: MoleculeType | None = None,
    bonds: Bonds | None = None,
    charges: Charges | None = None,
    sigmas: Sigmas | None = None,
    epsilons: Epsilons | None = None,
    radii: Radii | None = None,
    box_vectors: BoxVectors | None = None,
    source: str | None = None,
  ) -> AtomicSystem:
    """Factory to construct AtomicSystem from flat arrays.

    Convenience method to create an AtomicSystem by sorting inputs into
    the appropriate sub-components (topology, state, constants).

    Args:
      coordinates: Atom positions, shape (N_atoms, 3).
      atom_mask: Validity mask, shape (N_atoms,).
      elements: Element symbols.
      atom_names: PDB atom names.
      residue_index: Per-atom residue IDs.
      chain_index: Per-atom chain IDs.
      molecule_type: Per-atom molecule type (0=protein, 1=ligand, ...).
      bonds: Bond pairs, shape (N_bonds, 2).
      charges: Partial charges.
      sigmas: LJ sigma params.
      epsilons: LJ epsilon params.
      radii: Atomic radii.
      box_vectors: Periodic box, shape (3, 3).
      source: Source file path.

    Returns:
      AtomicSystem instance with populated sub-components.

    """
    topology = MolecularTopology(
      elements=elements,
      atom_names=atom_names,
      residue_index=residue_index,
      chain_index=chain_index,
      molecule_type=molecule_type,
      bonds=bonds,
    )
    state = AtomicState(
      coordinates=coordinates,
      box_vectors=box_vectors,
    )
    constants = None
    if any(x is not None for x in [charges, sigmas, epsilons, radii]):
      constants = AtomicConstants(
        charges=charges,
        sigmas=sigmas,
        epsilons=epsilons,
        radii=radii,
      )
    return cls(
      topology=topology,
      state=state,
      constants=constants,
      atom_mask=atom_mask,
      source=source,
    )

  def merge_with(self, other: AtomicSystem) -> AtomicSystem:
    """Merge this system with another AtomicSystem.

    Combines two systems (e.g., protein + ligand) into a single AtomicSystem.

    Args:
      other: Another AtomicSystem to merge.

    Returns:
      A new AtomicSystem containing atoms from both systems.

    """
    import numpy as np

    n_self = self.num_atoms
    n_other = other.num_atoms

    # Merge coordinates
    new_coords = jnp.concatenate([self.state.coordinates, other.state.coordinates], axis=0)

    # Merge atom_mask
    mask_self = self.atom_mask if self.atom_mask is not None else jnp.ones(n_self)
    mask_other = other.atom_mask if other.atom_mask is not None else jnp.ones(n_other)
    new_mask = jnp.concatenate([mask_self, mask_other], axis=0)

    # Merge topology elements
    def merge_seq(s1, s2, n1, n2):
      if s1 is None and s2 is None:
        return None
      l1 = list(s1) if s1 else ["X"] * n1
      l2 = list(s2) if s2 else ["X"] * n2
      return l1 + l2

    new_elements = merge_seq(self.topology.elements, other.topology.elements, n_self, n_other)
    new_atom_names = merge_seq(self.topology.atom_names, other.topology.atom_names, n_self, n_other)

    # Merge arrays (with offset for topology indices)
    def merge_arr(a1, a2):
      if a1 is None and a2 is None:
        return None
      parts = []
      if a1 is not None:
        parts.append(np.asarray(a1))
      if a2 is not None:
        parts.append(np.asarray(a2))
      return jnp.concatenate(parts, axis=0) if parts else None

    def merge_topo(a1, a2, offset):
      if a1 is None and a2 is None:
        return None
      parts = []
      if a1 is not None and len(a1) > 0:
        parts.append(np.asarray(a1))
      if a2 is not None and len(a2) > 0:
        parts.append(np.asarray(a2) + offset)
      return jnp.concatenate(parts, axis=0) if parts else None

    new_mol_type = merge_arr(self.topology.molecule_type, other.topology.molecule_type)
    new_res_idx = merge_arr(self.topology.residue_index, other.topology.residue_index)
    new_chain_idx = merge_arr(self.topology.chain_index, other.topology.chain_index)
    new_bonds = merge_topo(self.topology.bonds, other.topology.bonds, n_self)

    new_topology = MolecularTopology(
      elements=new_elements,
      atom_names=new_atom_names,
      residue_index=new_res_idx,
      chain_index=new_chain_idx,
      molecule_type=new_mol_type,
      bonds=new_bonds,
    )
    new_state = AtomicState(coordinates=new_coords)

    # Merge constants if present
    new_constants = None
    if self.constants or other.constants:
      new_constants = AtomicConstants(
        charges=merge_arr(
          self.constants.charges if self.constants else None,
          other.constants.charges if other.constants else None,
        ),
        sigmas=merge_arr(
          self.constants.sigmas if self.constants else None,
          other.constants.sigmas if other.constants else None,
        ),
        epsilons=merge_arr(
          self.constants.epsilons if self.constants else None,
          other.constants.epsilons if other.constants else None,
        ),
      )

    return AtomicSystem(
      topology=new_topology,
      state=new_state,
      constants=new_constants,
      atom_mask=new_mask,
    )


@dataclass(kw_only=True)
class Molecule(AtomicSystem):
  """Class representing a small molecule (ligand).

  Thin subclass of AtomicSystem for type clarity.
  """

  pass
