"""System projection interface for converting AtomicSystem to various formats.

This module provides:
- MPNNBatch: Training-ready batch structure for PrxteinMPNN
- Projection functions to convert AtomicSystem to different output formats
- Registry pattern for extensible format support
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from flax.struct import dataclass

if TYPE_CHECKING:
  from proxide.core.atomic_system import AtomicSystem

from proxide.core.types import (
  NeighborIndices,
  PerAtomChainIndex,
  PhysicsFeatures,
  ProteinSequence,
  RBFFeatures,
  ResidueIndex,
  ResidueMask,
)


class OutputSpec(Protocol):
  """Protocol for output format specifications."""

  output_format_target: str


@dataclass
class MPNNBatch:
  """Training-ready batch for PrxteinMPNN.

  This is the canonical output format for training. It contains per-residue
  data with all derived features pre-computed. No coordinates are stored
  since RBF features encode geometric information directly.

  This structure is designed for efficient padding and batching since all
  fields are required (no Optional nesting) except physics_features.

  Attributes:
    aatype: Amino acid type for each residue (0-20). Shape (N_res,).
    residue_index: PDB residue numbering. Shape (N_res,).
    chain_index: Chain identifier per residue. Shape (N_res,).
    mask: Validity mask. 1.0=valid residue, 0.0=padding. Shape (N_res,).
    rbf_features: RBF distance features encoding backbone geometry.
      Shape (N_res, K_neighbors, F_rbf).
    neighbor_indices: Indices of K-nearest neighbors per residue.
      Shape (N_res, K_neighbors).
    physics_features: Combined electrostatic and vdW features. Optional.
      Shape (N_res, F_phys).

  Note:
    Coordinates are NOT included. Use rbf_features for geometric encoding.
    For coordinate-based operations (noising, sampling), use AtomicSystem directly.

  """

  aatype: ProteinSequence
  residue_index: ResidueIndex
  chain_index: PerAtomChainIndex
  mask: ResidueMask
  rbf_features: RBFFeatures
  neighbor_indices: NeighborIndices
  physics_features: PhysicsFeatures | None = None


# Registry of projector functions
_PROJECTORS: dict[str, Any] = {}


def register_projector(format_key: str):
  """Decorator to register a projector function.

  Args:
    format_key: String identifier for the output format.

  Returns:
    Decorator function.

  Example:
    >>> @register_projector("my_format")
    ... def project_my_format(system, spec):
    ...     return MyFormat(...)

  """

  def decorator(fn):
    _PROJECTORS[format_key] = fn
    return fn

  return decorator


def project(system: AtomicSystem, spec: OutputSpec) -> Any:
  """Project AtomicSystem to target format based on spec.

  Args:
    system: The AtomicSystem to project.
    spec: OutputSpec with output_format_target field.

  Returns:
    Projected output in the requested format.

  Raises:
    ValueError: If output format is not registered.

  """
  key = spec.output_format_target
  if key not in _PROJECTORS:
    msg = f"Unknown output format: {key}. Available: {list(_PROJECTORS.keys())}"
    raise ValueError(msg)
  return _PROJECTORS[key](system, spec)


@register_projector("mpnn")
def project_to_mpnn(system: AtomicSystem, spec: OutputSpec) -> MPNNBatch:
  """Project AtomicSystem to MPNNBatch for training.

  This function:
  1. Filters to protein atoms (molecule_type == 0)
  2. Groups atoms by residue to get per-residue features
  3. Computes RBF features from backbone geometry (via Rust)
  4. Computes physics features if requested

  Args:
    system: AtomicSystem with per-atom data.
    spec: OutputSpec controlling feature computation.

  Returns:
    MPNNBatch ready for training.

  Note:
    For best performance, use the Rust backend via:
    proxide._proxider.project_to_mpnn_batch()

  """
  # TODO: Implement full projection logic
  # For now, this is a placeholder that will be implemented with Rust backend
  raise NotImplementedError(
    "project_to_mpnn requires Rust backend. Use proxide._proxider.project_to_mpnn_batch() directly."
  )


@dataclass
class OpenMMSpec:
  """Specification for OpenMM projection."""

  output_format_target: str = "openmm"
  coulomb14scale: float = 1.0
  lj14scale: float = 1.0


@register_projector("openmm")
def project_to_openmm_system(
    system: AtomicSystem, spec: OpenMMSpec | None = None, **kwargs: Any
) -> Any:
  """Project AtomicSystem to OpenMM System.

  Args:
    system: AtomicSystem with topology and constants.
    spec: OpenMMSpec with scaling parameters.
    **kwargs: Additional parameters for OpenMM system (e.g., nonbonded_cutoff).

  Returns:
    openmm.System with forces configured.

  """
  try:
    from openmm import (  # type: ignore[unresolved-import]
      HarmonicAngleForce,
      HarmonicBondForce,
      NonbondedForce,
      PeriodicTorsionForce,
      System,
    )
    from openmm import unit as u  # type: ignore[unresolved-import]
  except ImportError as e:
    raise ImportError(
      "OpenMM is required for MD parameterization. Install with: micromamba install openmm"
    ) from e

  import numpy as np

  coulomb14scale = spec.coulomb14scale if spec is not None else 1.0
  lj14scale = spec.lj14scale if spec is not None else 1.0
  cutoff = kwargs.get("nonbonded_cutoff", 1.0)

  coords = np.asarray(system.state.coordinates)
  mask = np.asarray(system.atom_mask) if system.atom_mask is not None else np.ones(len(coords))
  mask = mask > 0.5
  n_atoms = int(np.sum(mask))

  if n_atoms == 0:
    raise ValueError("Cannot create OpenMM system with zero atoms")

  omm_system = System()

  # Element masses
  elements = system.topology.elements or ["C"] * n_atoms
  for i in range(n_atoms):
    elem = elements[i] if i < len(elements) else "C"
    # Basic mass approximation
    mass = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "S": 32.065}.get(elem, 12.011)
    omm_system.addParticle(mass * u.amu)

  # Nonbonded force
  nonbonded = NonbondedForce()
  nonbonded.setNonbondedMethod(NonbondedForce.CutoffPeriodic)
  nonbonded.setCutoffDistance(cutoff * u.nanometer)

  charges = (
    np.asarray(system.constants.charges)
    if system.constants is not None and system.constants.charges is not None
    else np.zeros(n_atoms)
  )
  sigmas = (
    np.asarray(system.constants.sigmas)
    if system.constants is not None and system.constants.sigmas is not None
    else np.ones(n_atoms) * 0.3
  )
  epsilons = (
    np.asarray(system.constants.epsilons)
    if system.constants is not None and system.constants.epsilons is not None
    else np.zeros(n_atoms)
  )

  # Convert units (assuming input in Angstrom/kcal/mol)
  sigmas_nm = sigmas * 0.1
  epsilons_kjmol = epsilons * 4.184

  for i in range(n_atoms):
    nonbonded.addParticle(
        float(charges[i]),
        float(sigmas_nm[i]) * u.nanometer,
        float(epsilons_kjmol[i]) * u.kilojoule_per_mole,
    )

  # Exclusions and 1-4 scaling
  if system.topology.bonds is not None:
    bonds = np.asarray(system.topology.bonds)
    # 1-2 exclusions
    for b in bonds:
      nonbonded.addException(int(b[0]), int(b[1]), 0.0, 0.0, 0.0)

    # 1-3 exclusions
    if system.topology.angles is not None:
      for angle in np.asarray(system.topology.angles):
        nonbonded.addException(int(angle[0]), int(angle[2]), 0.0, 0.0, 0.0)

    # 1-4 scaling
    if system.topology.proper_dihedrals is not None:
      for dihedral in np.asarray(system.topology.proper_dihedrals):
        idx_i, idx_l = int(dihedral[0]), int(dihedral[3])
        # Calculate scaling values
        charge_prod = float(charges[idx_i] * charges[idx_l] * coulomb14scale)
        combined_sigma = 0.5 * (float(sigmas_nm[idx_i]) + float(sigmas_nm[idx_l]))
        combined_epsilon = (
            np.sqrt(float(epsilons_kjmol[idx_i]) * float(epsilons_kjmol[idx_l])) * lj14scale
        )
        nonbonded.addException(idx_i, idx_l, charge_prod, combined_sigma, combined_epsilon)

  nonbonded.setReactionFieldDielectric(78.5)
  omm_system.addForce(nonbonded)

  # Bond force
  if (
    system.topology.bonds is not None
    and system.constants
    and system.constants.bond_params is not None
  ):
    bond_force = HarmonicBondForce()
    bonds = np.asarray(system.topology.bonds)
    bond_params = np.asarray(system.constants.bond_params)
    for b_idx in range(len(bonds)):
      i, j = int(bonds[b_idx, 0]), int(bonds[b_idx, 1])
      if i < n_atoms and j < n_atoms:
        length = float(bond_params[b_idx, 0]) * 0.1
        k = float(bond_params[b_idx, 1]) * 4.184 * 100
        bond_force.addBond(i, j, length * u.nanometer, k * u.kilojoule_per_mole / u.nanometer**2)
    omm_system.addForce(bond_force)

  # Angle force
  if (
    system.topology.angles is not None
    and system.constants
    and system.constants.angle_params is not None
  ):
    angle_force = HarmonicAngleForce()
    angles = np.asarray(system.topology.angles)
    angle_params = np.asarray(system.constants.angle_params)
    for a_idx in range(len(angles)):
      i, j, k = int(angles[a_idx, 0]), int(angles[a_idx, 1]), int(angles[a_idx, 2])
      if i < n_atoms and j < n_atoms and k < n_atoms:
        angle_rad = float(angle_params[a_idx, 0]) * np.pi / 180.0
        k_angle = float(angle_params[a_idx, 1]) * 4.184
        angle_force.addAngle(
            i, j, k, angle_rad * u.radian, k_angle * u.kilojoule_per_mole / u.radian**2
        )
    omm_system.addForce(angle_force)

  # Torsion force
  if (
      system.topology.proper_dihedrals is not None
      and system.constants
      and system.constants.dihedral_params is not None
  ):
    torsion_force = PeriodicTorsionForce()
    dihedrals = np.asarray(system.topology.proper_dihedrals)
    dihedral_params = np.asarray(system.constants.dihedral_params)
    for d_idx in range(len(dihedrals)):
      i_d, j_d, k_d, l_d = (
          int(dihedrals[d_idx, 0]),
          int(dihedrals[d_idx, 1]),
          int(dihedrals[d_idx, 2]),
          int(dihedrals[d_idx, 3]),
      )
      if i_d < n_atoms and j_d < n_atoms and k_d < n_atoms and l_d < n_atoms:
        if dihedral_params.ndim == 3:
          for term_idx in range(dihedral_params.shape[1]):
            periodicity = int(dihedral_params[d_idx, term_idx, 0])
            phase = float(dihedral_params[d_idx, term_idx, 1]) * np.pi / 180.0
            k_torsion = float(dihedral_params[d_idx, term_idx, 2]) * 4.184
            if abs(k_torsion) > 1e-6:
              torsion_force.addTorsion(
                  i_d,
                  j_d,
                  k_d,
                  l_d,
                  periodicity,
                  phase * u.radian,
                  k_torsion * u.kilojoule_per_mole,
              )
        else:
          periodicity = int(dihedral_params[d_idx, 0])
          phase = float(dihedral_params[d_idx, 1]) * np.pi / 180.0
          k_torsion = float(dihedral_params[d_idx, 2]) * 4.184
          torsion_force.addTorsion(
              i_d,
              j_d,
              k_d,
              l_d,
              periodicity,
              phase * u.radian,
              k_torsion * u.kilojoule_per_mole,
          )
    omm_system.addForce(torsion_force)

  # Improper force
  if (
      system.topology.impropers is not None
      and system.constants
      and system.constants.improper_params is not None
  ):
    improper_force = PeriodicTorsionForce()
    impropers = np.asarray(system.topology.impropers)
    improper_params = np.asarray(system.constants.improper_params)
    for d_idx in range(len(impropers)):
      i_d, j_d, k_d, l_d = (
          int(impropers[d_idx, 0]),
          int(impropers[d_idx, 1]),
          int(impropers[d_idx, 2]),
          int(impropers[d_idx, 3]),
      )
      if i_d < n_atoms and j_d < n_atoms and k_d < n_atoms and l_d < n_atoms:
        if improper_params.ndim == 3:
          for term_idx in range(improper_params.shape[1]):
            periodicity = int(improper_params[d_idx, term_idx, 0])
            phase = float(improper_params[d_idx, term_idx, 1]) * np.pi / 180.0
            k_torsion = float(improper_params[d_idx, term_idx, 2]) * 4.184
            if abs(k_torsion) > 1e-6:
              improper_force.addTorsion(
                  i_d,
                  j_d,
                  k_d,
                  l_d,
                  periodicity,
                  phase * u.radian,
                  k_torsion * u.kilojoule_per_mole,
              )
        else:
          periodicity = int(improper_params[d_idx, 0])
          phase = float(improper_params[d_idx, 1]) * np.pi / 180.0
          k_torsion = float(improper_params[d_idx, 2]) * 4.184
          improper_force.addTorsion(
              i_d,
              j_d,
              k_d,
              l_d,
              periodicity,
              phase * u.radian,
              k_torsion * u.kilojoule_per_mole,
          )
    omm_system.addForce(improper_force)

  return omm_system


@register_projector("openmm_topology")
def project_to_openmm_topology(system: AtomicSystem, spec: OutputSpec) -> Any:
  """Project AtomicSystem to OpenMM Topology.

  Args:
    system: AtomicSystem with topology info.
    spec: OutputSpec (unused).

  Returns:
    openmm.app.Topology.

  """
  try:
    from openmm.app import Element, Topology  # type: ignore[unresolved-import]
  except ImportError as e:
    raise ImportError(
      "OpenMM is required. Install with: conda install -c conda-forge openmm",
    ) from e

  import numpy as np

  topology = Topology()
  chain = topology.addChain()

  mask = np.asarray(system.atom_mask) if system.atom_mask is not None else np.ones(system.num_atoms)
  mask = mask > 0.5
  n_atoms = int(np.sum(mask))

  if n_atoms == 0:
    return topology

  residue = topology.addResidue("UNK", chain)
  elements = system.topology.elements or ["C"] * n_atoms
  atom_names = system.topology.atom_names or [f"A{i}" for i in range(n_atoms)]

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

  if system.topology.bonds is not None:
    bonds = np.asarray(system.topology.bonds)
    for b_idx in range(bonds.shape[0]):
      i, j = int(bonds[b_idx, 0]), int(bonds[b_idx, 1])
      if i < len(atoms) and j < len(atoms):
        topology.addBond(atoms[i], atoms[j])

  return topology
