"""Shared parsing utilities for structure files.

This module provides utilities for processing parsed structure data,
extracting features, and converting to Protein objects.

NOTE: This module maintains biotite dependency for MDTraj trajectory support.
For new code, prefer using the Rust parser directly via parse_structure().
"""

import logging
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import jax.numpy as jnp

# Biotite is still needed for MDTraj trajectory parsing
from biotite.structure import (
  AtomArray,
  AtomArrayStack,
  dihedral_backbone,
  get_residue_count,
  get_residue_positions,
  get_residue_starts,
  get_residues,
)

from proxide.chem.residues import atom_order
from proxide.core.containers import Protein, ProteinStream
from proxide.io.parsing.coords import process_coordinates
from proxide.io.parsing.mappings import atom_names_to_index, residue_names_to_aatype
from proxide.io.parsing.structures import ProcessedStructure
from proxide.io.parsing.types import TrajectoryStaticFeatures
from proxide.physics.constants import DEFAULT_EPSILON, DEFAULT_SIGMA

logger = logging.getLogger(__name__)


def _check_atom_array_length(atom_array: AtomArray | AtomArrayStack) -> None:
  """Check if the AtomArray has a valid length."""
  length = atom_array.array_length()
  logger.debug("Checking AtomArray length: %d", length)
  if length == 0:
    msg = "AtomArray is empty."
    logger.error(msg)
    raise ValueError(msg)


def _validate_atom_array_type(atom_array: Any) -> None:  # noqa: ANN401
  """Validate that the atom array is of the expected type."""
  logger.debug("Validating atom array type.")
  if not isinstance(atom_array, (AtomArray | AtomArrayStack)):
    msg = f"Expected AtomArray or AtomArrayStack, but got {type(atom_array)}."
    logger.error(msg)
    raise TypeError(msg)


def _get_chain_index(
  atom_array: AtomArray | AtomArrayStack,
) -> np.ndarray:
  """Get the chain index from the AtomArray."""
  if atom_array.chain_id is None:
    logger.debug("Chain ID not available, returning zeros for chain index.")
    return np.zeros(atom_array.array_length(), dtype=np.int32)

  if atom_array.chain_id.dtype != np.int32:
    logger.debug("Converting string chain IDs to integer indices (A=0, B=1, ...).")
    return np.asarray(
      np.char.encode(atom_array.chain_id.astype("U1")).view(np.uint8) - ord("A"),
      dtype=np.int32,
    )

  logger.debug("Using existing integer chain IDs.")
  return np.asarray(atom_array.chain_id, dtype=np.int32)


def _process_chain_id(
  atom_array: AtomArray | AtomArrayStack,
  chain_id: Sequence[str] | str | None = None,
) -> tuple[AtomArray | AtomArrayStack, np.ndarray]:
  """Process the chain_id of the AtomArray."""
  if chain_id is None:
    logger.debug("No chain_id specified. Using all available chains.")
    chain_index = _get_chain_index(atom_array)
    return atom_array, chain_index

  logger.info("Processing structure with specified chain_id(s): %s", chain_id)

  if isinstance(chain_id, str):
    chain_id = [chain_id]

  if not isinstance(chain_id, Sequence):
    msg = f"Expected chain_id to be a string or a sequence of strings, got {type(chain_id)}."
    logger.error(msg)
    raise TypeError(msg)

  if atom_array.chain_id is None:
    msg = "Chain ID is not available in the structure, but chain_id was specified."
    logger.error(msg)
    raise ValueError(msg)

  chain_mask = np.isin(atom_array.chain_id, chain_id)

  if not np.any(chain_mask):
    logger.warning("No atoms found for specified chain(s) %s.", chain_id)

  if isinstance(atom_array, AtomArrayStack):
    atom_array = cast("AtomArray | AtomArrayStack", atom_array[:, chain_mask])
  else:
    atom_array = cast("AtomArray | AtomArrayStack", atom_array[chain_mask])

  chain_index = _get_chain_index(atom_array)
  logger.debug("Filtered AtomArray to %d atoms for specified chains.", atom_array.array_length())
  return (
    atom_array,
    chain_index,
  )


def _extract_biotite_static_features(
  atom_array: AtomArray | AtomArrayStack,
  atom_map: dict[str, int] | None = None,
  chain_id: Sequence[str] | str | None = None,
) -> tuple[TrajectoryStaticFeatures, AtomArray | AtomArrayStack]:
  """Extract static features from a Biotite AtomArray."""
  logger.info("Extracting static features using Biotite.")
  if atom_map is None:
    atom_map = atom_order

  atom_array, chain_index = _process_chain_id(atom_array, chain_id)
  _check_atom_array_length(atom_array)
  num_residues_all_atoms = get_residue_count(atom_array)

  residue_indices, residue_names = get_residues(atom_array)
  logger.debug("Found %d residues in the processed AtomArray.", num_residues_all_atoms)

  residue_indices = np.asarray(residue_indices, dtype=np.int32)
  chain_index = chain_index[get_residue_starts(atom_array)]
  residue_inv_indices = get_residue_positions(
    atom_array,
    np.arange(atom_array.array_length()),
  )

  atom_names = atom_array.atom_name

  if atom_names is None:
    msg = "Atom names are not available in the structure."
    logger.error(msg)
    raise ValueError(msg)

  atom37_indices = atom_names_to_index(np.array(atom_names, dtype="U5"))

  atom_mask = atom37_indices != -1

  # This is the mask for 37 atoms for ALL residues (before filtering)
  atom_mask_37 = np.zeros((num_residues_all_atoms, 37), dtype=bool)

  res_indices_flat = np.asarray(residue_inv_indices)[atom_mask]
  atom_indices_flat = atom37_indices[atom_mask]

  atom_mask_37[res_indices_flat, atom_indices_flat] = 1

  aatype = residue_names_to_aatype(residue_names)
  nitrogen_mask = atom_mask_37[:, atom_map["N"]] == 1

  # Filter to residues that have an N atom (required for backbone trace)
  aatype = aatype[nitrogen_mask]
  atom_mask_37 = atom_mask_37[nitrogen_mask]
  residue_indices = residue_indices[nitrogen_mask]
  chain_index = chain_index[nitrogen_mask]

  num_residues = aatype.shape[0]
  logger.info("Filtered AtomArray to %d valid residues (those containing N).", num_residues)

  valid_residue_mask = nitrogen_mask[np.asarray(residue_inv_indices)]
  atom_mask &= valid_residue_mask

  return TrajectoryStaticFeatures(
    aatype=aatype,
    static_atom_mask_37=atom_mask_37,
    residue_indices=residue_indices,
    chain_index=chain_index,
    valid_atom_mask=atom_mask,
    nitrogen_mask=nitrogen_mask,
    num_residues=num_residues,
  ), atom_array


def atom_array_dihedrals(
  atom_array: AtomArray | AtomArrayStack,
) -> np.ndarray | None:
  """Compute backbone dihedral angles (phi, psi, omega) for the given AtomArray."""
  logger.debug("Computing backbone dihedral angles using Biotite.")
  phi, psi, omega = dihedral_backbone(atom_array)
  phi = np.asarray(phi)
  psi = np.asarray(psi)
  omega = np.asarray(omega)
  if (
    phi is None
    or psi is None
    or omega is None
    or np.all(np.isnan(phi))
    or np.all(np.isnan(psi))
    or np.all(np.isnan(omega))
  ):
    logger.warning("Dihedral calculation resulted in all NaN values or None.")
    return None

  dihedrals = np.stack([phi, psi, omega], axis=-1)

  clean_dihedrals = dihedrals[~np.any(np.isnan(dihedrals), axis=-1)]
  logger.debug("Calculated %d valid dihedral sets.", clean_dihedrals.shape[0])

  return clean_dihedrals


def _get_default_physics_parameters(
  elements: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Get default physics parameters based on element types.

  This is a fallback when force field loading fails.
  """
  n_atoms = len(elements)

  # Simple element-based defaults
  element_params = {
    "C": (1.908, 0.086),
    "N": (1.824, 0.170),
    "O": (1.661, 0.210),
    "S": (2.000, 0.250),
    "H": (0.600, 0.0157),
    "P": (2.100, 0.200),
  }

  charges = np.zeros(n_atoms, dtype=np.float32)
  sigmas = np.zeros(n_atoms, dtype=np.float32)
  epsilons = np.zeros(n_atoms, dtype=np.float32)

  for i, elem in enumerate(elements):
    sigmas[i], epsilons[i] = element_params.get(elem, (DEFAULT_SIGMA, DEFAULT_EPSILON))

  return charges, sigmas, epsilons


def _resolve_physics_parameters(
  processed_structure: ProcessedStructure,
  atom_array: AtomArray | AtomArrayStack,
  *,
  populate_physics: bool,
  force_field_name: str,  # noqa: ARG001
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
  """Resolve physics parameters from processed structure or populate them."""
  charges = processed_structure.charges
  radii = processed_structure.radii
  sigmas = processed_structure.sigmas
  epsilons = processed_structure.epsilons

  if populate_physics and (charges is None or sigmas is None or epsilons is None):
    logger.info("Populating missing physics parameters using element-based defaults")

    # Use first frame for parameter extraction if stack
    calc_array = (
      cast("AtomArray", atom_array[0]) if isinstance(atom_array, AtomArrayStack) else atom_array
    )

    elements = np.array(calc_array.element)
    charges_ff, sigmas_ff, epsilons_ff = _get_default_physics_parameters(elements)

    # Use defaults for missing parameters
    if charges is None:
      charges = charges_ff
    if sigmas is None:
      sigmas = sigmas_ff
    if epsilons is None:
      epsilons = epsilons_ff

    # Radii: use van der Waals radii if not present
    if radii is None:
      element_radii = {
        "H": 1.20,
        "C": 1.70,
        "N": 1.55,
        "O": 1.52,
        "S": 1.80,
        "P": 1.80,
      }
      radii = np.array(
        [element_radii.get(elem, 1.70) for elem in elements],
        dtype=np.float32,
      )
  return charges, radii, sigmas, epsilons


def processed_structure_to_protein_tuples(
  processed_structure: ProcessedStructure,
  source_name: str,
  *,
  extract_dihedrals: bool = False,
  populate_physics: bool = False,
  force_field_name: str = "ff14SB",
) -> ProteinStream:
  """Convert a ProcessedStructure into a stream of Protein instances.

  Args:
      processed_structure: The ProcessedStructure to convert
      source_name: Name of the source file
      extract_dihedrals: Whether to extract dihedral angles
      populate_physics: Whether to populate physics parameters if missing
      force_field_name: Force field to use for parameter population

  """
  atom_array = processed_structure.atom_array
  static_features, atom_array = _extract_biotite_static_features(atom_array, chain_id=None)
  charges, radii, sigmas, epsilons = _resolve_physics_parameters(
    processed_structure,
    atom_array,
    populate_physics=populate_physics,
    force_field_name=force_field_name,
  )
  num_frames = atom_array.stack_depth() if isinstance(atom_array, AtomArrayStack) else 1
  frame_count = 0

  def _yield_protein(frame: AtomArray) -> Protein:
    dihedrals = None
    if extract_dihedrals:
      dihedrals = atom_array_dihedrals(frame)

    coords = np.asarray(frame.coord)
    coords_37 = process_coordinates(
      coords,
      static_features.num_residues,
      static_features.static_atom_mask_37,
      static_features.valid_atom_mask,
    )
    atom_mask_2d = static_features.static_atom_mask_37.astype(np.float32)

    return Protein(
      coordinates=jnp.array(coords_37, dtype=jnp.float32),
      aatype=jnp.array(static_features.aatype, dtype=jnp.int8),
      atom_mask=jnp.array(atom_mask_2d, dtype=jnp.float32),
      mask=jnp.array(atom_mask_2d[:, atom_order["CA"]], dtype=jnp.float32),
      one_hot_sequence=jnp.eye(21)[static_features.aatype],
      residue_index=jnp.array(static_features.residue_indices, dtype=jnp.int32),
      chain_index=jnp.array(static_features.chain_index, dtype=jnp.int32),
      dihedrals=jnp.array(dihedrals, dtype=jnp.float32) if dihedrals is not None else None,
      full_coordinates=jnp.array(coords, dtype=jnp.float32),
      charges=jnp.array(charges, dtype=jnp.float32) if charges is not None else None,
      radii=jnp.array(radii, dtype=jnp.float32) if radii is not None else None,
      epsilons=jnp.array(epsilons, dtype=jnp.float32) if epsilons is not None else None,
      sigmas=jnp.array(sigmas, dtype=jnp.float32) if sigmas is not None else None,
      elements=None,
      atom_names=None,
    )

  if isinstance(atom_array, AtomArrayStack):
    for frame in atom_array:
      frame_count += 1
      logger.debug("Yielding frame %d of %d from Biotite stack.", frame_count, num_frames)
      yield _yield_protein(frame)

  elif isinstance(atom_array, AtomArray):
    frame_count += 1
    logger.debug("Yielding single frame from Biotite AtomArray.")
    yield _yield_protein(atom_array)
