"""Atomic system definitions for PrioX.

This module defines the base classes for atomic systems, including
proteins, small molecules, and complexes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
from flax.struct import dataclass

if TYPE_CHECKING:
  from collections.abc import Sequence


@dataclass(kw_only=True)
class AtomicSystem:
  """Base class for any atomic system.

  This class acts as a JAX-compatible container for atomic data and can
  optionally wrap a Rust-backend AtomicSystem for performance-critical operations.

  Attributes:
      coordinates: Atom positions (N_atoms, 3)
      atom_mask: Binary mask for atom presence (N_atoms,)
      elements: Element symbols (N_atoms,)
      atom_names: Atom names (N_atoms,)
      molecule_type: Per-atom type (0=Protein, 1=Ligand, 2=Solvent, 3=Ion)
      atom_types: GAFF/ForceField atom types (N_atoms,)
      bonds: Bond indices (N_bonds, 2)
      angles: Angle indices (N_angles, 3)
      proper_dihedrals: Proper dihedral indices (N_dihedrals, 4)
      impropers: Improper dihedral indices (N_impropers, 4)
      charges: Atomic partial charges (N_atoms,)
      radii: Atomic radii (N_atoms,)
      _rust_obj: Optional reference to the Rust-side AtomicSystem object.

  """

  coordinates: jnp.ndarray
  atom_mask: jnp.ndarray
  elements: Sequence[str] | None = None
  atom_names: Sequence[str] | None = None

  # HETATM / Topology features
  molecule_type: jnp.ndarray | None = None  # (N_atoms,) 0=Protein, 1=Ligand, 2=Solvent, 3=Ion
  atom_types: Sequence[str] | None = None  # GAFF/ForceField atom types
  bonds: jnp.ndarray | None = None  # (N_bonds, 2) - atom index pairs
  angles: jnp.ndarray | None = None  # (N_angles, 3) - i-j-k with j as central atom
  proper_dihedrals: jnp.ndarray | None = None  # (N_dihedrals, 4) - proper torsions i-j-k-l
  impropers: jnp.ndarray | None = None  # (N_impropers, 4) - improper torsions for planar groups

  # Force field parameters
  bond_params: jnp.ndarray | None = None
  angle_params: jnp.ndarray | None = None
  dihedral_params: jnp.ndarray | None = None  # (N_dihedrals, 3) - [periodicity, phase, k]
  improper_params: jnp.ndarray | None = None  # (N_impropers, 3) - [periodicity, phase, k]

  # CMAP parameters for backbone correction
  cmap_indices: jnp.ndarray | None = None  # (N_cmap, 5) - C-N-CA-C-N atom indices for phi-psi
  cmap_grid: jnp.ndarray | None = None  # (grid_size, grid_size) raw energy grid in kJ/mol

  exclusion_mask: jnp.ndarray | None = None

  # Optional MD parameters common to all systems
  charges: jnp.ndarray | None = None
  sigmas: jnp.ndarray | None = None
  epsilons: jnp.ndarray | None = None
  radii: jnp.ndarray | None = None

  # Internal storage for Rust object (if available)
  _rust_obj: Any = None

  # --- Convenience properties for filtering by molecule type ---

  @property
  def protein_mask(self) -> jnp.ndarray | None:
    """Boolean mask for protein atoms (molecule_type == 0)."""
    if self.molecule_type is None:
      return None
    return self.molecule_type == 0

  @property
  def ligand_mask(self) -> jnp.ndarray | None:
    """Boolean mask for ligand atoms (molecule_type == 1)."""
    if self.molecule_type is None:
      return None
    return self.molecule_type == 1

  @property
  def solvent_mask(self) -> jnp.ndarray | None:
    """Boolean mask for solvent atoms (molecule_type == 2)."""
    if self.molecule_type is None:
      return None
    return self.molecule_type == 2

  @property
  def ion_mask(self) -> jnp.ndarray | None:
    """Boolean mask for ion atoms (molecule_type == 3)."""
    if self.molecule_type is None:
      return None
    return self.molecule_type == 3

  @property
  def has_ligands(self) -> bool:
    """Check if system contains any ligand atoms."""
    if self.molecule_type is None:
      return False
    return bool(jnp.any(self.molecule_type == 1))

  @property
  def has_solvent(self) -> bool:
    """Check if system contains any solvent atoms."""
    if self.molecule_type is None:
      return False
    return bool(jnp.any(self.molecule_type == 2))

  @property
  def num_protein_atoms(self) -> int:
    """Number of protein atoms."""
    if self.molecule_type is None:
      return int(jnp.sum(self.atom_mask))

    # Check if molecule_type is packed (size mismatch with atom_mask)
    if self.molecule_type.size != self.atom_mask.size:
      return int(jnp.sum(self.molecule_type == 0))

    return int(jnp.sum((self.molecule_type == 0) & (self.atom_mask > 0)))

  @property
  def num_ligand_atoms(self) -> int:
    """Number of ligand atoms."""
    if self.molecule_type is None:
      return 0

    if self.molecule_type.size != self.atom_mask.size:
      return int(jnp.sum(self.molecule_type == 1))

    return int(jnp.sum((self.molecule_type == 1) & (self.atom_mask > 0)))

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

  @property
  def num_impropers(self) -> int:
    """Number of improper dihedrals in the system."""
    if self.impropers is None:
      return 0
    return len(self.impropers)

  def to_openmm_topology(self):
    """Convert to an OpenMM Topology object.

    Requires openmm to be installed.

    Returns:
        openmm.app.Topology: OpenMM topology with atoms, residues, chains.

    Raises:
        ImportError: If openmm is not installed.

    """
    try:
      from openmm.app import Element, Topology
    except ImportError:
      raise ImportError(
        "OpenMM is required for to_openmm_topology(). "
        "Install with: conda install -c conda-forge openmm",
      )

    import numpy as np

    topology = Topology()
    chain = topology.addChain()

    # Get atom mask as numpy for indexing
    mask = np.asarray(self.atom_mask) > 0.5
    num_atoms = int(np.sum(mask))

    if num_atoms == 0:
      return topology

    # Create a single residue (simplified; could group by residue_index)
    residue = topology.addResidue("UNK", chain)

    # Get element info
    elements = self.elements if self.elements else ["C"] * num_atoms
    atom_names = self.atom_names if self.atom_names else [f"A{i}" for i in range(num_atoms)]

    # Add atoms
    atoms = []
    atom_idx = 0
    for i in range(len(mask)):
      if mask[i]:
        elem_str = elements[atom_idx] if atom_idx < len(elements) else "C"
        elem = Element.getBySymbol(elem_str) if len(elem_str) <= 2 else Element.getBySymbol("C")
        name = atom_names[atom_idx] if atom_idx < len(atom_names) else f"A{atom_idx}"
        atom = topology.addAtom(name, elem, residue)
        atoms.append(atom)
        atom_idx += 1

    # Add bonds if available
    if self.bonds is not None:
      bonds = np.asarray(self.bonds)
      for bond_idx in range(bonds.shape[0]):
        i, j = int(bonds[bond_idx, 0]), int(bonds[bond_idx, 1])
        if i < len(atoms) and j < len(atoms):
          topology.addBond(atoms[i], atoms[j])

    return topology

  def to_openmm_system(
    self,
    nonbonded_cutoff: float = 1.0,
    use_switching_function: bool = True,
    switch_distance: float = 0.9,
    coulomb14scale: float = 0.8333,
    lj14scale: float = 0.5,
  ):
    """Convert to an OpenMM System object with force field parameters.

    Requires openmm to be installed and force field parameters (charges, sigmas,
    epsilons) to be set on the AtomicSystem.

    Args:
        nonbonded_cutoff: Cutoff distance for nonbonded interactions in nm.
        use_switching_function: If True, use a switching function for LJ.
        switch_distance: Distance to begin switching function in nm.

    Returns:
        openmm.System: OpenMM system with NonbondedForce, HarmonicBondForce,
            and HarmonicAngleForce configured.

    Raises:
        ImportError: If openmm is not installed.
        ValueError: If required force field parameters are missing.

    Notes:
        Unit conversions:
        - Lengths: Å → nm (multiply by 0.1)
        - Bond force constants: kcal/mol/Å² → kJ/mol/nm² (multiply by 4.184 * 100)
        - Angle force constants: kcal/mol/rad² → kJ/mol/rad² (multiply by 4.184)
        - Torsion force constants: kcal/mol → kJ/mol (multiply by 4.184)
        - Energies: kcal/mol → kJ/mol (multiply by 4.184)

    """
    try:
      from openmm import (
        HarmonicAngleForce,
        HarmonicBondForce,
        NonbondedForce,
        PeriodicTorsionForce,
        System,
      )
      from openmm import unit as u
    except ImportError:
      raise ImportError(
        "OpenMM is required for to_openmm_system(). "
        "Install with: conda install -c conda-forge openmm",
      )

    import numpy as np

    # Handle Atom37 format masks: (N_res, 37) -> flatten to (N_res*37,)
    mask_raw = np.asarray(self.atom_mask)
    if mask_raw.ndim > 1:
      mask = mask_raw.flatten() > 0.5
    else:
      mask = mask_raw > 0.5
    n_atoms = int(np.sum(mask))

    if n_atoms == 0:
      raise ValueError("Cannot create OpenMM system with zero atoms")

    system = System()

    # Add atoms with masses (default to carbon mass if not specified)
    # Element masses in atomic mass units (amu)
    element_masses = {
      "H": 1.008,
      "C": 12.011,
      "N": 14.007,
      "O": 15.999,
      "S": 32.065,
      "P": 30.974,
      "F": 18.998,
      "Cl": 35.453,
      "Br": 79.904,
      "I": 126.904,
    }

    elements = self.elements if self.elements else ["C"] * n_atoms
    for i in range(n_atoms):
      elem = elements[i] if i < len(elements) else "C"
      mass = element_masses.get(elem, 12.011)
      system.addParticle(mass * u.amu)

    # Nonbonded Force (electrostatics + LJ)
    nonbonded = NonbondedForce()
    nonbonded.setNonbondedMethod(NonbondedForce.CutoffPeriodic)
    nonbonded.setCutoffDistance(nonbonded_cutoff * u.nanometer)

    if use_switching_function:
      nonbonded.setUseSwitchingFunction(True)
      nonbonded.setSwitchingDistance(switch_distance * u.nanometer)

    # Add particles to nonbonded force
    charges = np.asarray(self.charges) if self.charges is not None else np.zeros(n_atoms)
    sigmas = np.asarray(self.sigmas) if self.sigmas is not None else np.ones(n_atoms) * 0.3
    epsilons = np.asarray(self.epsilons) if self.epsilons is not None else np.zeros(n_atoms)

    # Convert units: priox uses Angstroms and kcal/mol, OpenMM uses nm and kJ/mol
    sigmas_nm = sigmas * 0.1  # Å to nm
    epsilons_kjmol = epsilons * 4.184  # kcal/mol to kJ/mol

    atom_idx = 0
    particle_params = []
    for i in range(len(mask)):
      if mask[i]:
        q = float(charges[atom_idx]) if atom_idx < len(charges) else 0.0
        sig = float(sigmas_nm[atom_idx]) if atom_idx < len(sigmas_nm) else 0.3
        eps = float(epsilons_kjmol[atom_idx]) if atom_idx < len(epsilons_kjmol) else 0.0
        nonbonded.addParticle(q, sig * u.nanometer, eps * u.kilojoule_per_mole)
        particle_params.append((q, sig, eps))
        atom_idx += 1

    system.addForce(nonbonded)

    # Harmonic Bond Force
    if self.bonds is not None and self.bond_params is not None:
      bond_force = HarmonicBondForce()
      bonds = np.asarray(self.bonds)
      bond_params = np.asarray(self.bond_params)

      for b_idx in range(len(bonds)):
        i, j = int(bonds[b_idx, 0]), int(bonds[b_idx, 1])
        if i < n_atoms and j < n_atoms:
          # bond_params: [length (Å), k (kcal/mol/Å²)]
          length = float(bond_params[b_idx, 0]) * 0.1  # Å to nm
          k = float(bond_params[b_idx, 1]) * 4.184 * 100  # kcal/mol/Å² to kJ/mol/nm²
          bond_force.addBond(i, j, length * u.nanometer, k * u.kilojoule_per_mole / u.nanometer**2)

      system.addForce(bond_force)

    # Harmonic Angle Force
    if self.angles is not None and self.angle_params is not None:
      angle_force = HarmonicAngleForce()
      angles = np.asarray(self.angles)
      angle_params = np.asarray(self.angle_params)

      for a_idx in range(len(angles)):
        i, j, k = int(angles[a_idx, 0]), int(angles[a_idx, 1]), int(angles[a_idx, 2])
        if i < n_atoms and j < n_atoms and k < n_atoms:
          # angle_params: [theta (rad), k (kcal/mol/rad²)]
          theta = float(angle_params[a_idx, 0])  # Already in radians
          k_angle = float(angle_params[a_idx, 1]) * 4.184  # kcal/mol/rad² to kJ/mol/rad²
          angle_force.addAngle(
            i,
            j,
            k,
            theta * u.radian,
            k_angle * u.kilojoule_per_mole / u.radian**2,
          )

      system.addForce(angle_force)

    # Periodic Torsion Force (proper dihedrals)
    if self.proper_dihedrals is not None and self.dihedral_params is not None:
      torsion_force = PeriodicTorsionForce()
      dihedrals = np.asarray(self.proper_dihedrals)
      dihedral_params = np.asarray(self.dihedral_params)

      for d_idx in range(len(dihedrals)):
        i, j, k, m = (
          int(dihedrals[d_idx, 0]),
          int(dihedrals[d_idx, 1]),
          int(dihedrals[d_idx, 2]),
          int(dihedrals[d_idx, 3]),
        )
        if i < n_atoms and j < n_atoms and k < n_atoms and m < n_atoms:
          # dihedral_params: [periodicity, phase (rad), k (kcal/mol)]
          periodicity = int(dihedral_params[d_idx, 0])
          phase = float(dihedral_params[d_idx, 1])  # radians
          k_torsion = float(dihedral_params[d_idx, 2]) * 4.184  # kcal/mol to kJ/mol
          torsion_force.addTorsion(
            i,
            j,
            k,
            m,
            periodicity,
            phase * u.radian,
            k_torsion * u.kilojoule_per_mole,
          )

      system.addForce(torsion_force)

    # Periodic Torsion Force (improper dihedrals)
    if self.impropers is not None and self.improper_params is not None:
      improper_force = PeriodicTorsionForce()
      impropers = np.asarray(self.impropers)
      improper_params = np.asarray(self.improper_params)

      for i_idx in range(len(impropers)):
        i, j, k, m = (
          int(impropers[i_idx, 0]),
          int(impropers[i_idx, 1]),
          int(impropers[i_idx, 2]),
          int(impropers[i_idx, 3]),
        )
        if i < n_atoms and j < n_atoms and k < n_atoms and m < n_atoms:
          # improper_params: [periodicity, phase (rad), k (kcal/mol)]
          periodicity = int(improper_params[i_idx, 0])
          phase = float(improper_params[i_idx, 1])  # radians
          k_improper = float(improper_params[i_idx, 2]) * 4.184  # kcal/mol to kJ/mol
          improper_force.addTorsion(
            i,
            j,
            k,
            m,
            periodicity,
            phase * u.radian,
            k_improper * u.kilojoule_per_mole,
          )

      system.addForce(improper_force)

    # CMAP Torsion Force (backbone corrections)
    if self.cmap_indices is not None and self.cmap_grid is not None:
      try:
        from openmm import CMAPTorsionForce
      except ImportError:
        CMAPTorsionForce = None

      if CMAPTorsionForce is not None:
        cmap_force = CMAPTorsionForce()
        cmap_indices = np.asarray(self.cmap_indices)
        cmap_grid = np.asarray(self.cmap_grid)

        # Add the CMAP (energy grid)
        grid_size = cmap_grid.shape[0]
        # Flatten grid for OpenMM (row-major)
        # Grid is in kJ/mol, OpenMM expects kJ/mol
        flat_grid = cmap_grid.flatten().tolist()
        cmap_idx = cmap_force.addMap(grid_size, flat_grid)

        # Add torsions that use this CMAP
        for t_idx in range(len(cmap_indices)):
          # cmap_indices: [C_prev, N, CA, C, N_next] for phi-psi pair
          atoms = [int(cmap_indices[t_idx, i]) for i in range(5)]
          if all(a < n_atoms for a in atoms):
            # Phi: C(i-1)-N(i)-CA(i)-C(i) -> atoms[0:4]
            # Psi: N(i)-CA(i)-C(i)-N(i+1) -> atoms[1:5]
            cmap_force.addTorsion(
              cmap_idx,
              atoms[0],
              atoms[1],
              atoms[2],
              atoms[3],  # phi atoms
              atoms[1],
              atoms[2],
              atoms[3],
              atoms[4],  # psi atoms
            )

        system.addForce(cmap_force)

    # Collect exclusions to avoid double counting
    # Set of sets/tuples of indices
    excluded_pairs = set()

    # Add exclusions for bonded atoms (1-2 pairs)
    if self.bonds is not None:
      bonds = np.asarray(self.bonds)
      for b_idx in range(len(bonds)):
        i, j = int(bonds[b_idx, 0]), int(bonds[b_idx, 1])
        if i < n_atoms and j < n_atoms:
          pair = tuple(sorted((i, j)))
          if pair not in excluded_pairs:
            nonbonded.addException(i, j, 0.0, 1.0, 0.0)
            excluded_pairs.add(pair)

    # Add exclusions for angle atoms (1-3 pairs)
    if self.angles is not None:
      angles = np.asarray(self.angles)
      for a_idx in range(len(angles)):
        i, k = int(angles[a_idx, 0]), int(angles[a_idx, 2])
        if i < n_atoms and k < n_atoms:
          pair = tuple(sorted((i, k)))
          if pair not in excluded_pairs:
            nonbonded.addException(i, k, 0.0, 1.0, 0.0)
            excluded_pairs.add(pair)

    # Add scaled interactions for dihedral atoms (1-4 pairs)
    if self.proper_dihedrals is not None:
      dihedrals = np.asarray(self.proper_dihedrals)
      for d_idx in range(len(dihedrals)):
        i, l = int(dihedrals[d_idx, 0]), int(dihedrals[d_idx, 3])
        if i < n_atoms and l < n_atoms:
          pair = tuple(sorted((i, l)))
          if pair not in excluded_pairs:
            # Get parameters
            q1, sig1, eps1 = particle_params[i]
            q2, sig2, eps2 = particle_params[l]

            # Calculate scaled parameters
            # Coulomb 1-4
            charge_prod = q1 * q2 * coulomb14scale

            # LJ 1-4 (Lorentz-Berthelot mixing + scaling)
            sigma_mix = (sig1 + sig2) * 0.5
            epsilon_mix = np.sqrt(eps1 * eps2) * lj14scale

            nonbonded.addException(
              i,
              l,
              charge_prod,
              sigma_mix * u.nanometer,
              epsilon_mix * u.kilojoule_per_mole,
            )
            # Mark as processed so we don't overwrite with another 1-4 or something else
            excluded_pairs.add(pair)

    return system

  def merge_with(self, other: AtomicSystem) -> AtomicSystem:
    """Merge this system with another AtomicSystem.

    Combines two systems (e.g., protein + ligand) into a single AtomicSystem.
    All topology indices (bonds, angles, dihedrals) from the second system
    are offset by the number of atoms in the first system.

    Args:
        other: Another AtomicSystem to merge with this one.

    Returns:
        A new AtomicSystem containing atoms from both systems.

    Example:
        >>> complex_system = protein.merge_with(ligand)

    """
    import numpy as np

    # Concatenate coordinates
    n_atoms_self = len(self.atom_mask)
    n_atoms_other = len(other.atom_mask)

    new_coords = jnp.concatenate([self.coordinates, other.coordinates], axis=0)
    new_mask = jnp.concatenate([self.atom_mask, other.atom_mask], axis=0)

    # Merge elements and atom_names (as lists)
    def merge_sequences(seq1, seq2):
      if seq1 is None and seq2 is None:
        return None
      s1 = list(seq1) if seq1 else ["X"] * n_atoms_self
      s2 = list(seq2) if seq2 else ["X"] * n_atoms_other
      return s1 + s2

    new_elements = merge_sequences(self.elements, other.elements)
    new_atom_names = merge_sequences(self.atom_names, other.atom_names)
    new_atom_types = merge_sequences(self.atom_types, other.atom_types)

    # Merge molecule_type (preserve from both)
    if self.molecule_type is not None or other.molecule_type is not None:
      mt1 = (
        self.molecule_type
        if self.molecule_type is not None
        else jnp.zeros(n_atoms_self, dtype=jnp.int32)
      )
      mt2 = (
        other.molecule_type
        if other.molecule_type is not None
        else jnp.ones(n_atoms_other, dtype=jnp.int32)
      )
      new_molecule_type = jnp.concatenate([mt1, mt2], axis=0)
    else:
      new_molecule_type = None

    # Helper to offset and merge topology arrays
    def merge_topology(arr1, arr2, offset: int):
      if arr1 is None and arr2 is None:
        return None
      parts = []
      if arr1 is not None and len(arr1) > 0:
        parts.append(np.asarray(arr1))
      if arr2 is not None and len(arr2) > 0:
        parts.append(np.asarray(arr2) + offset)
      if not parts:
        return None
      return jnp.array(np.concatenate(parts, axis=0))

    # Merge bonds, angles, dihedrals, impropers
    new_bonds = merge_topology(self.bonds, other.bonds, n_atoms_self)
    new_angles = merge_topology(self.angles, other.angles, n_atoms_self)
    new_dihedrals = merge_topology(self.proper_dihedrals, other.proper_dihedrals, n_atoms_self)
    new_impropers = merge_topology(self.impropers, other.impropers, n_atoms_self)

    # Merge parameter arrays (just concatenate, no offset)
    def merge_params(arr1, arr2):
      if arr1 is None and arr2 is None:
        return None
      parts = []
      if arr1 is not None and len(arr1) > 0:
        parts.append(np.asarray(arr1))
      if arr2 is not None and len(arr2) > 0:
        parts.append(np.asarray(arr2))
      if not parts:
        return None
      return jnp.array(np.concatenate(parts, axis=0))

    new_bond_params = merge_params(self.bond_params, other.bond_params)
    new_angle_params = merge_params(self.angle_params, other.angle_params)
    new_dihedral_params = merge_params(self.dihedral_params, other.dihedral_params)
    new_improper_params = merge_params(self.improper_params, other.improper_params)

    # Merge MD parameters
    new_charges = merge_params(self.charges, other.charges)
    new_sigmas = merge_params(self.sigmas, other.sigmas)
    new_epsilons = merge_params(self.epsilons, other.epsilons)
    new_radii = merge_params(self.radii, other.radii)

    # Merge CMAP (offset indices, keep grids separate - typically same grid)
    new_cmap_indices = merge_topology(self.cmap_indices, other.cmap_indices, n_atoms_self)
    # For cmap_grid, we use self's grid if available (ligands typically don't have CMAP)
    new_cmap_grid = self.cmap_grid if self.cmap_grid is not None else other.cmap_grid

    return AtomicSystem(
      coordinates=new_coords,
      atom_mask=new_mask,
      elements=new_elements,
      atom_names=new_atom_names,
      atom_types=new_atom_types,
      molecule_type=new_molecule_type,
      bonds=new_bonds,
      angles=new_angles,
      proper_dihedrals=new_dihedrals,
      impropers=new_impropers,
      bond_params=new_bond_params,
      angle_params=new_angle_params,
      dihedral_params=new_dihedral_params,
      improper_params=new_improper_params,
      cmap_indices=new_cmap_indices,
      cmap_grid=new_cmap_grid,
      charges=new_charges,
      sigmas=new_sigmas,
      epsilons=new_epsilons,
      radii=new_radii,
    )


@dataclass(kw_only=True)
class Molecule(AtomicSystem):
  """Class representing a small molecule (ligand).

  Thin subclass of AtomicSystem for type clarity.
  """
