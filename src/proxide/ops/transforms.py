"""Data operations for processing protein structures within a Grain pipeline.

This module implements `grain.transforms.Map` and `grain.IterOperation` classes
for parsing, transforming, and batching protein data.

Includes a padding registry pattern for extensible output format support.
"""

import warnings
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

import jax
import numpy as np

from proxide import md
from proxide.chem import residues as residue_constants
from proxide.core.containers import Protein
from proxide.core.projector import MPNNBatch
from proxide.physics.features import compute_electrostatic_node_features
from proxide.physics.force_fields import loader as force_fields

_MAX_TRIES = 5

# Type variable for batch output
T = TypeVar("T", Protein, MPNNBatch)

# Padding function registry
_PADDING_REGISTRY: dict[str, Callable[[Protein, int, int | None, dict | None], Any]] = {}


def register_padding(format_key: str) -> Callable:
  """Decorator to register a padding function for an output format.

  Example:
    @register_padding("mpnn")
    def _pad_to_mpnn(protein: Protein, max_len: int, ...) -> MPNNBatch:
      ...
  """

  def decorator(fn: Callable) -> Callable:
    _PADDING_REGISTRY[format_key] = fn
    return fn

  return decorator


def get_padding_fn(format_key: str) -> Callable | None:
  """Get a registered padding function by format key."""
  return _PADDING_REGISTRY.get(format_key)


def compute_rbf_features_rust(
  protein: Protein,
  num_neighbors: int = 30,
  noise_std: float | None = None,
  noise_seed: int | None = None,
  compute_physics: bool = False,
) -> Protein:
  """Compute RBF features using Rust backend and attach to Protein.

  This function uses the high-performance Rust implementation to compute
  RBF features and neighbor indices, optionally with Gaussian backbone noise.

  Args:
    protein: Input protein structure.
    num_neighbors: Number of K-nearest neighbors (default: 30).
    noise_std: Standard deviation for Gaussian backbone noise. None = no noise.
    noise_seed: Random seed for reproducible noising. None = random seed.
    compute_physics: Whether to compute physics features (charges, sigmas, epsilons).

  Returns:
    Protein with rbf_features and neighbor_indices populated.
  """
  try:
    from proxide import _oxidize
  except ImportError:
    # Fall back gracefully if Rust extension not available
    warnings.warn(
      "Rust _oxidize extension not available, skipping RBF precomputation",
      stacklevel=2,
    )
    return protein

  # Get source path if available
  source_path = getattr(protein, "source", None)
  if source_path is None:
    # Cannot use Rust projection without a file path
    warnings.warn(
      "Protein has no source path, skipping Rust RBF computation",
      stacklevel=2,
    )
    return protein

  try:
    # Call Rust projection
    result = _oxidize.project_to_mpnn_batch(
      source_path,
      num_neighbors,
      noise_std,
      noise_seed,
      compute_physics,
    )

    # Update protein with RBF features
    return protein.replace(  # type: ignore[attr-defined]
      rbf_features=result.get("rbf_features"),
      neighbor_indices=result.get("neighbor_indices"),
      physics_features=result.get("physics_features")
      if compute_physics
      else protein.physics_features,
    )
  except Exception as e:  # noqa: BLE001
    warnings.warn(
      f"Rust RBF computation failed for {source_path}: {e}",
      stacklevel=2,
    )
    return protein


def truncate_protein(
  protein: Protein,
  max_length: int | None,
  strategy: str = "none",
) -> Protein:
  """Truncate a protein to a maximum length.

  Args:
    protein: The protein to truncate.
    max_length: The maximum length. If None, no truncation is performed.
    strategy: The truncation strategy ("random_crop", "center_crop", "none").

  Returns:
    The truncated protein.

  """
  if max_length is None or strategy == "none":
    return protein

  length = protein.coordinates.shape[0]
  if length <= max_length:
    return protein

  if strategy == "center_crop":
    start = (length - max_length) // 2
  elif strategy == "random_crop":
    start = np.random.default_rng().integers(0, length - max_length + 1)
  else:
    msg = f"Unknown truncation strategy: {strategy}"
    raise ValueError(msg)

  end = start + max_length

  def slice_array(arr: Any | None) -> Any | None:
    if arr is None:
      return None
    # Assuming the first dimension is always the residue dimension for arrays that need slicing
    if hasattr(arr, "shape") and arr.shape[0] == length:
      return arr[start:end]
    return arr

  return protein.replace(  # type: ignore[attr-defined]
    coordinates=slice_array(protein.coordinates),
    aatype=slice_array(protein.aatype),
    mask=slice_array(protein.mask),
    one_hot_sequence=slice_array(protein.one_hot_sequence),
    atom_mask=slice_array(protein.atom_mask),
    residue_index=slice_array(protein.residue_index),
    chain_index=slice_array(protein.chain_index),
    full_coordinates=slice_array(protein.full_coordinates),
    dihedrals=slice_array(protein.dihedrals),
    mapping=slice_array(protein.mapping),
    charges=slice_array(protein.charges),
    radii=slice_array(protein.radii),
    sigmas=slice_array(protein.sigmas),
    epsilons=slice_array(protein.epsilons),
    physics_features=slice_array(protein.physics_features),
    # AtomicSystem fields (previously MD fields)
    bonds=slice_array(protein.bonds),
    bond_params=slice_array(protein.bond_params),
    angles=slice_array(protein.angles),
    angle_params=slice_array(protein.angle_params),
    exclusion_mask=slice_array(protein.exclusion_mask),
  )


def concatenate_proteins_for_inter_mode(elements: Sequence[Protein]) -> Protein:
  """Concatenate proteins for inter-chain mode (pass_mode='inter').

  Instead of padding and stacking, concatenate all structures along the residue dimension
  and remap chain IDs to ensure global uniqueness across all structures.

  Each structure's chain IDs are offset by the maximum chain ID from all previous structures,
  preserving the original chain relationships within each structure while ensuring no
  collisions across structures.

  The structure boundaries are stored in the `mapping` field as [0,0,0..., 1,1,1..., 2,2,2...]
  to enable "direct" tied_positions mode.

  Args:
    elements: List of protein tuples to concatenate.

  Returns:
    Protein: Single concatenated protein with globally unique chain IDs and structure mapping.

  Raises:
    ValueError: If the input list is empty.

  Example:
    >>> # Structure 1: chains [0,0,1,1], Structure 2: chains [0,0,2,2]
    >>> combined = concatenate_proteins_for_inter_mode([protein1, protein2])
    >>> # Result chains: [0,0,1,1,2,2,4,4] - each structure's chains are offset
    >>> # Result mapping: [0,0,0,0,1,1,1,1] - tracks which structure each residue came from

  """
  if not elements:
    msg = "Cannot concatenate an empty list of proteins."
    warnings.warn(msg, stacklevel=2)
    raise ValueError(msg)

  tries = 0
  while not all(isinstance(p, Protein) for p in elements):
    if any(isinstance(p, Sequence) for p in elements):
      elements = [p[0] if isinstance(p, Sequence) else p for p in elements]  # type: ignore[invalid-assignment]
      tries += 1
    if tries > _MAX_TRIES:
      msg = "Too many nested sequences in elements; cannot collate."
      warnings.warn(msg, stacklevel=2)
      raise ValueError(msg)

  proteins = list(elements)

  structure_indices = []
  for i, protein in enumerate(proteins):
    length = protein.coordinates.shape[0]
    structure_indices.append(np.full(length, i, dtype=np.int32))

  structure_mapping = np.concatenate(structure_indices, axis=0)
  remapped_chain_ids = []
  chain_offset = 0

  for protein in proteins:
    original_chains = protein.chain_index
    remapped_chains = original_chains + chain_offset
    remapped_chain_ids.append(remapped_chains)
    chain_offset = int(np.max(remapped_chains)) + 1

  chain_ids = np.concatenate(remapped_chain_ids, axis=0)
  concatenated = jax.tree_util.tree_map(lambda *x: np.concatenate(x, axis=0), *proteins)
  concatenated = concatenated.replace(chain_index=chain_ids, mapping=structure_mapping)  # type: ignore[attr-defined]
  return jax.tree_util.tree_map(lambda x: x[None, ...], concatenated)


def _validate_and_flatten_elements(
  elements: Sequence[Protein],
) -> list[Protein]:
  """Ensure all elements are Protein and flatten nested sequences.

  Args:
    elements (Sequence[Protein]): List of proteins to validate.

  Returns:
    list[Protein]: Validated and flattened list of Protein.

  Raises:
    ValueError: If the input list is empty or too deeply nested.

  """
  if not elements:
    msg = "Cannot collate an empty list of proteins."
    warnings.warn(msg, stacklevel=2)
    raise ValueError(msg)

  tries = 0
  while not all(isinstance(p, Protein) for p in elements):
    if any(isinstance(p, Sequence) for p in elements):
      elements = [p[0] if isinstance(p, Sequence) else p for p in elements]  # type: ignore[invalid-assignment]
      tries += 1
    if tries > _MAX_TRIES:
      msg = "Too many nested sequences in elements; cannot collate."
      warnings.warn(msg, stacklevel=2)
      raise ValueError(msg)
  return list(elements)


def _apply_electrostatics_if_needed(
  elements: list[Protein],
  *,
  use_electrostatics: bool,
  estat_noise: Sequence[float] | float | None = None,
  estat_noise_mode: str = "direct",
) -> list[Protein]:
  """Apply electrostatic features if requested.

  Args:
    elements (list[Protein]): List of proteins.
    use_electrostatics (bool): Whether to compute and add electrostatic features.
    estat_noise: Noise level(s) for electrostatics.
    estat_noise_mode: Mode for electrostatic noise ("direct" or "thermal").

  Returns:
    list[Protein]: Updated list with electrostatic features if requested.

  """
  if not use_electrostatics:
    return elements

  # Handle noise broadcasting if needed, or just pass single value if uniform
  # For now, assuming uniform noise for the batch or handling inside feature computation
  # compute_electrostatic_features_batch doesn't take noise yet, we need to update
  # it or call node features directly.
  # Actually compute_electrostatic_features_batch calls compute_electrostatic_node_features
  # per protein. We can pass the noise value there.

  noise_val = estat_noise
  if isinstance(noise_val, Sequence):
    noise_val = noise_val[0]  # Simple handling for now

  phys_feats = []
  for p in elements:
    feat = compute_electrostatic_node_features(
      p,
      noise_scale=noise_val,
      noise_mode=estat_noise_mode,
    )
    phys_feats.append(np.array(feat))

  return [
    p.replace(physics_features=feat)  # type: ignore[attr-defined]
    for p, feat in zip(elements, phys_feats, strict=False)
  ]


def _apply_md_parameterization(
  elements: list[Protein],
  *,
  use_md: bool,
) -> list[Protein]:
  """Parameterize proteins for MD simulation.

  Args:
      elements: List of protein tuples.
      use_md: Whether to apply MD parameterization.

  Returns:
      List of protein tuples with MD fields populated.

  """
  if not use_md:
    return elements

  # Load force field (cached)
  ff = force_fields.load_force_field("ff14SB")

  # Get residue names map
  # residue_constants.restypes is list of 20 AA.
  # We need to handle 'X' or others?
  # parameterize_system expects 3-letter codes.
  # residue_constants.restype_1to3 map.

  updated_elements = []
  for p in elements:
    # Convert aatype to residue names
    res_names = []
    for aa in p.aatype:
      if aa < len(residue_constants.restypes):
        res_1 = residue_constants.restypes[aa]
        res_3 = residue_constants.restype_1to3.get(res_1, "UNK")
      else:
        res_3 = "UNK"
      res_names.append(res_3)

    # Construct atom_names list
    atom_names = []
    for res_name in res_names:
      atoms = residue_constants.residue_atoms.get(res_name, [])
      atom_names.extend(atoms)

    params = md.parameterize_system(ff, res_names, atom_names)

    # Convert JAX arrays to numpy and populate ProteinTuple
    p_new = p.replace(  # type: ignore[attr-defined]
      bonds=np.array(params["bonds"]),
      bond_params=np.array(params["bond_params"]),
      angles=np.array(params["angles"]),
      angle_params=np.array(params["angle_params"]),
      backbone_indices=np.array(params["backbone_indices"]),
      exclusion_mask=np.array(params["exclusion_mask"]),
      charges=np.array(params["charges"]),
      sigmas=np.array(params["sigmas"]),
      epsilons=np.array(params["epsilons"]),
    )
    updated_elements.append(p_new)

  return updated_elements


def _pad_protein(  # noqa: C901
  protein: Protein,
  max_len: int,
  full_coords_max_len: int | None = None,
  md_dims: dict[str, int] | None = None,
) -> Protein:
  """Pad a single Protein to max_len using explicit field-based padding.

  Args:
    protein (Protein): Protein to pad.
    max_len (int): Maximum length to pad to.
    full_coords_max_len: Maximum length for full_coordinates field.
    md_dims: Dictionary of max dimensions for MD fields.

  Returns:
    Protein: Padded protein.

  Raises:
    ValueError: If the protein length exceeds max_len (negative padding).

  """
  protein_len = protein.coordinates.shape[0]
  pad_len = max_len - protein_len

  # Validate: protein must not exceed max_len
  if pad_len < 0:
    msg = (
      f"Protein length ({protein_len}) exceeds max_len ({max_len}). "
      f"Truncate the protein before padding, or increase max_len."
    )
    raise ValueError(msg)

  # Helper for residue-level padding (dim 0 = n_residues)
  def pad_residue_array(arr: np.ndarray | None) -> np.ndarray | None:
    if arr is None:
      return None
    arr = np.asarray(arr)
    if arr.ndim == 0:
      return arr
    return np.pad(arr, ((0, pad_len),) + ((0, 0),) * (arr.ndim - 1))

  # Helper for atomic-level padding (dim 0 = n_atoms, e.g., full_coordinates)
  full_coords_len = (
    protein.full_coordinates.shape[0] if protein.full_coordinates is not None else None
  )
  if full_coords_len is not None and full_coords_max_len is not None:
    full_coords_pad_len = full_coords_max_len - full_coords_len
    if full_coords_pad_len < 0:
      msg = (
        f"full_coordinates length ({full_coords_len}) exceeds "
        f"full_coords_max_len ({full_coords_max_len})."
      )
      raise ValueError(msg)
  else:
    full_coords_pad_len = 0

  def pad_atomic_array(arr: np.ndarray | None) -> np.ndarray | None:
    if arr is None:
      return None
    arr = np.asarray(arr)
    if arr.ndim == 0 or full_coords_pad_len == 0:
      return arr
    return np.pad(arr, ((0, full_coords_pad_len),) + ((0, 0),) * (arr.ndim - 1))

  # Explicitly pad each field based on its semantic type
  # Residue-level fields (shape[0] == n_residues)
  padded_coordinates = pad_residue_array(protein.coordinates)
  padded_aatype = pad_residue_array(protein.aatype)
  padded_mask = pad_residue_array(protein.mask)
  padded_one_hot = pad_residue_array(protein.one_hot_sequence)
  padded_atom_mask = pad_residue_array(protein.atom_mask)
  padded_residue_index = pad_residue_array(protein.residue_index)
  padded_chain_index = pad_residue_array(protein.chain_index)
  padded_dihedrals = pad_residue_array(protein.dihedrals)
  padded_mapping = pad_residue_array(protein.mapping)
  padded_physics_features = pad_residue_array(protein.physics_features)
  padded_rbf_features = pad_residue_array(protein.rbf_features)
  padded_vdw_features = pad_residue_array(protein.vdw_features)
  padded_electrostatic_features = pad_residue_array(protein.electrostatic_features)
  padded_neighbor_indices = pad_residue_array(protein.neighbor_indices)
  padded_backbone_indices = pad_residue_array(protein.backbone_indices)

  # Atomic-level fields (shape[0] == n_atoms, for full atomic representation)
  padded_full_coordinates = pad_atomic_array(protein.full_coordinates)
  padded_full_atom_mask = pad_atomic_array(protein.full_atom_mask)
  padded_charges = pad_atomic_array(protein.charges)
  padded_radii = pad_atomic_array(protein.radii)
  padded_sigmas = pad_atomic_array(protein.sigmas)
  padded_epsilons = pad_atomic_array(protein.epsilons)

  # Create the base padded protein
  padded_protein = protein.replace(  # type: ignore[attr-defined]
    coordinates=padded_coordinates,
    aatype=padded_aatype,
    mask=padded_mask,
    one_hot_sequence=padded_one_hot,
    atom_mask=padded_atom_mask,
    residue_index=padded_residue_index,
    chain_index=padded_chain_index,
    dihedrals=padded_dihedrals,
    mapping=padded_mapping,
    physics_features=padded_physics_features,
    rbf_features=padded_rbf_features,
    vdw_features=padded_vdw_features,
    electrostatic_features=padded_electrostatic_features,
    neighbor_indices=padded_neighbor_indices,
    backbone_indices=padded_backbone_indices,
    full_coordinates=padded_full_coordinates,
    full_atom_mask=padded_full_atom_mask,
    charges=padded_charges,
    radii=padded_radii,
    sigmas=padded_sigmas,
    epsilons=padded_epsilons,
  )

  # MD-specific fields with their own dimensions
  if md_dims:

    def pad_md_array(arr: np.ndarray | None, pad_amt: int) -> np.ndarray | None:
      if arr is None or pad_amt <= 0:
        return arr
      arr = np.asarray(arr)
      pads = [(0, pad_amt)] + [(0, 0)] * (arr.ndim - 1)
      return np.pad(arr, pads)

    # Bonds
    if padded_protein.bonds is not None:
      bonds_pad = md_dims.get("max_bonds", 0) - padded_protein.bonds.shape[0]
      if bonds_pad > 0:
        p_bonds = pad_md_array(padded_protein.bonds, bonds_pad)
        p_params = pad_md_array(padded_protein.bond_params, bonds_pad)
        padded_protein = padded_protein.replace(bonds=p_bonds, bond_params=p_params)  # type: ignore[attr-defined]

    # Angles
    if padded_protein.angles is not None:
      angles_pad = md_dims.get("max_angles", 0) - padded_protein.angles.shape[0]
      if angles_pad > 0:
        p_angles = pad_md_array(padded_protein.angles, angles_pad)
        p_params = pad_md_array(padded_protein.angle_params, angles_pad)
        padded_protein = padded_protein.replace(angles=p_angles, angle_params=p_params)  # type: ignore[attr-defined]

    # Exclusion mask (N_atoms, N_atoms) - special 2D padding
    if padded_protein.exclusion_mask is not None:
      curr = padded_protein.exclusion_mask.shape[0]
      target = md_dims.get("max_atoms", curr)
      amt = target - curr
      if amt > 0:
        mask = np.pad(
          padded_protein.exclusion_mask,
          ((0, amt), (0, amt)),
          constant_values=False,
        )
        padded_protein = padded_protein.replace(exclusion_mask=mask)  # type: ignore[attr-defined]

  return padded_protein


def _stack_padded_proteins(
  padded_proteins: list[Protein],
) -> Protein:
  """Stack a list of padded Proteins into a batch.

  Args:
    padded_proteins (list[Protein]): List of padded proteins.

  Returns:
    Protein: Batched protein.

  """

  def stack_fn(*arrays: np.ndarray | None) -> np.ndarray | None:
    """Stack arrays, handling None values and scalars."""
    non_none = [a for a in arrays if a is not None]
    if not non_none:
      return None
    first = non_none[0]
    if not hasattr(first, "shape") or first.ndim == 0:
      return first
    if not all(hasattr(a, "shape") and a.shape == first.shape for a in non_none):
      return None
    return np.stack(non_none, axis=0)

  return jax.tree_util.tree_map(stack_fn, *padded_proteins)


def pad_and_collate_proteins(
  elements: Sequence[Protein],
  *,
  use_electrostatics: bool = False,
  use_vdw: bool = False,  # noqa: ARG001
  estat_noise: Sequence[float] | float | None = None,
  estat_noise_mode: str = "direct",
  vdw_noise: Sequence[float] | float | None = None,  # noqa: ARG001
  vdw_noise_mode: str = "direct",  # noqa: ARG001
  backbone_noise_mode: str = "direct",
  max_length: int | None = None,
  output_format: str = "protein",
) -> Protein | MPNNBatch:
  """Batch and pad a list of Proteins.

  Take a list of individual `Protein`s and batch them together,
  padding them to a fixed length. Output format is configurable.

  Args:
    elements: List of proteins to collate.
    use_electrostatics: Whether to compute and add electrostatic features.
    use_vdw: Placeholder for van der Waals features (not implemented).
    estat_noise: Noise level(s) for electrostatics.
    estat_noise_mode: Mode for electrostatic noise.
    vdw_noise: Noise level(s) for vdW.
    vdw_noise_mode: Mode for vdW noise.
    backbone_noise_mode: Mode for backbone noise (e.g. "direct", "md").
    max_length: Fixed length to pad all proteins to. If None, pads to
      the maximum length in the batch.
    output_format: Output format key (e.g., "protein", "mpnn").
      Use "protein" for Protein batch (default).
      Use "mpnn" for MPNNBatch with precomputed RBF features.

  Returns:
    Batched and padded data in the specified format.

  Raises:
    ValueError: If the input list is empty.

  Example:
    >>> ensemble = pad_and_collate_proteins([protein1, protein2],
    use_electrostatics=True, max_length=512)

  """
  elements = _validate_and_flatten_elements(elements)
  elements = _apply_electrostatics_if_needed(
    elements,
    use_electrostatics=use_electrostatics,
    estat_noise=estat_noise,
    estat_noise_mode=estat_noise_mode,
  )

  # Apply MD parameterization if needed
  # We infer use_md from backbone_noise_mode?
  # Or we should add use_md arg?
  # The user didn't ask to add use_md arg to RunSpecification, but backbone_noise_mode.
  # But pad_and_collate doesn't take RunSpecification.
  # It takes args.
  # Let's assume if backbone_noise_mode (which is not passed here?)
  # Wait, pad_and_collate signature doesn't have backbone_noise_mode.
  # I should add it or use kwargs?
  # The signature in the file is:
  # def pad_and_collate_proteins(..., vdw_noise_mode: str = "direct", ...)
  # It doesn't have backbone_noise_mode.
  # I should add it.

  # But I can't change signature easily if it's used elsewhere.
  # However, I can add it as optional kwarg.
  # Or use `vdw_noise_mode`? No.

  # Let's add `backbone_noise_mode` to arguments.

  elements = _apply_md_parameterization(
    elements,
    use_md=(backbone_noise_mode == "md"),
  )

  proteins = list(elements)

  # Use fixed max_length if provided, otherwise use max in batch
  pad_len = max_length if max_length is not None else max(p.coordinates.shape[0] for p in proteins)

  # Calculate MD dims
  md_dims = {}
  if backbone_noise_mode == "md":
    max_bonds = 0
    max_angles = 0
    max_atoms = 0
    for p in proteins:
      if p.bonds is not None:
        max_bonds = max(max_bonds, p.bonds.shape[0])
      if p.angles is not None:
        max_angles = max(max_angles, p.angles.shape[0])
      if p.charges is not None:
        max_atoms = max(max_atoms, p.charges.shape[0])
    md_dims = {"max_bonds": max_bonds, "max_angles": max_angles, "max_atoms": max_atoms}

  # Calculate max full_coords length
  full_coords_max_len = 0
  atoms_per_res_ratios = set()

  # First pass to gather stats
  for p in proteins:
    if p.full_coordinates is not None:
      f_len = p.full_coordinates.shape[0]
      p_len = p.coordinates.shape[0]
      full_coords_max_len = max(full_coords_max_len, f_len)
      if p_len > 0 and f_len % p_len == 0:
        atoms_per_res_ratios.add(f_len // p_len)
      else:
        atoms_per_res_ratios.add(-1)  # Indicator for non-uniform/non-divisible

  # Determine target length
  # If we have a single consistent ratio (e.g. 37 for all proteins), we use it to extrapolate
  # based on pad_len (which is the target residue count).
  # This ensures JAX static shapes consistency (e.g. 512 * 37).
  if len(atoms_per_res_ratios) == 1:
    ratio = list(atoms_per_res_ratios)[0]
    if ratio > 0:
      full_coords_max_len = max(full_coords_max_len, pad_len * ratio)

  # For purely heterogeneous batches or fixed MD, full_coords_max_len is just the max seen in batch.

  # Dispatch based on output_format
  if output_format == "mpnn":
    # Use registered MPNN padding if available
    mpnn_pad_fn = get_padding_fn("mpnn")
    if mpnn_pad_fn is not None:
      return mpnn_pad_fn(proteins, pad_len, full_coords_max_len, md_dims)
    # Fallback: convert to MPNNBatch from padded Proteins
    padded_proteins = [_pad_protein(p, pad_len, full_coords_max_len, md_dims) for p in proteins]
    stacked = _stack_padded_proteins(padded_proteins)
    return _protein_batch_to_mpnn_batch(stacked)

  # Default: return Protein batch
  padded_proteins = [_pad_protein(p, pad_len, full_coords_max_len, md_dims) for p in proteins]
  return _stack_padded_proteins(padded_proteins)


def _protein_batch_to_mpnn_batch(batch: Protein) -> MPNNBatch:
  """Convert a batched Protein to MPNNBatch.

  Takes the relevant fields from the Protein batch and creates an MPNNBatch.
  RBF features must already be present in the Protein.
  """
  import jax.numpy as jnp

  return MPNNBatch(
    aatype=jnp.asarray(batch.aatype),
    residue_index=jnp.asarray(batch.residue_index),
    chain_index=jnp.asarray(batch.chain_index),
    mask=jnp.asarray(batch.mask),
    rbf_features=jnp.asarray(batch.rbf_features)
    if batch.rbf_features is not None
    else jnp.zeros((batch.mask.shape[0], batch.mask.shape[1], 30, 400)),
    neighbor_indices=jnp.asarray(batch.neighbor_indices)
    if batch.neighbor_indices is not None
    else jnp.zeros((batch.mask.shape[0], batch.mask.shape[1], 30), dtype=jnp.int32),
    physics_features=jnp.asarray(batch.physics_features)
    if batch.physics_features is not None
    else None,
  )


@register_padding("protein")
def _pad_to_protein(
  proteins: list[Protein],
  pad_len: int,
  full_coords_max_len: int | None,
  md_dims: dict[str, int] | None,
) -> Protein:
  """Default Protein padding - registered for consistency."""
  padded = [_pad_protein(p, pad_len, full_coords_max_len, md_dims) for p in proteins]
  return _stack_padded_proteins(padded)
