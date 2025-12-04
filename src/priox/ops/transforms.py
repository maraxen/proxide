"""Data operations for processing protein structures within a Grain pipeline.

This module implements `grain.transforms.Map` and `grain.IterOperation` classes
for parsing, transforming, and batching protein data.
"""

import warnings
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from priox.physics.force_fields import loader as force_fields
from priox.chem import residues as residue_constants
from priox.core.containers import Protein, ProteinTuple
from priox.md import jax_md_bridge
from priox.physics.features import compute_electrostatic_node_features

_MAX_TRIES = 5


def truncate_protein(
  protein: ProteinTuple,
  max_length: int | None,
  strategy: str = "none",
) -> ProteinTuple:
  """Truncate a protein to a maximum length.

  Args:
    protein: The protein tuple to truncate.
    max_length: The maximum length. If None, no truncation is performed.
    strategy: The truncation strategy ("random_crop", "center_crop", "none").

  Returns:
    The truncated protein tuple.

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

  def slice_array(arr: np.ndarray | None) -> np.ndarray | None:
    if arr is None:
      return None
    # Assuming the first dimension is always the residue dimension for arrays that need slicing
    if hasattr(arr, "shape") and arr.shape[0] == length:
      return arr[start:end]
    return arr

  return protein._replace(
    coordinates=slice_array(protein.coordinates),
    aatype=slice_array(protein.aatype),
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
    estat_backbone_mask=slice_array(protein.estat_backbone_mask),
    estat_resid=slice_array(protein.estat_resid),
    estat_chain_index=slice_array(protein.estat_chain_index),
    physics_features=slice_array(protein.physics_features),
  )


def concatenate_proteins_for_inter_mode(elements: Sequence[ProteinTuple]) -> Protein:
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
  while not all(isinstance(p, ProteinTuple) for p in elements):
    if any(isinstance(p, Sequence) for p in elements):
      elements = [p[0] if isinstance(p, Sequence) else p for p in elements]  # type: ignore[invalid-assignment]
      tries += 1
    if tries > _MAX_TRIES:
      msg = "Too many nested sequences in elements; cannot collate."
      warnings.warn(msg, stacklevel=2)
      raise ValueError(msg)

  proteins = [Protein.from_tuple_numpy(p) for p in elements]

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
  concatenated = concatenated.replace(chain_index=chain_ids, mapping=structure_mapping)
  return jax.tree_util.tree_map(lambda x: x[None, ...], concatenated)


def _validate_and_flatten_elements(
  elements: Sequence[ProteinTuple],
) -> list[ProteinTuple]:
  """Ensure all elements are ProteinTuple and flatten nested sequences.

  Args:
    elements (Sequence[ProteinTuple]): List of protein tuples to validate.

  Returns:
    list[ProteinTuple]: Validated and flattened list of ProteinTuple.

  Raises:
    ValueError: If the input list is empty or too deeply nested.

  """
  if not elements:
    msg = "Cannot collate an empty list of proteins."
    warnings.warn(msg, stacklevel=2)
    raise ValueError(msg)

  tries = 0
  while not all(isinstance(p, ProteinTuple) for p in elements):
    if any(isinstance(p, Sequence) for p in elements):
      elements = [p[0] if isinstance(p, Sequence) else p for p in elements]  # type: ignore[invalid-assignment]
      tries += 1
    if tries > _MAX_TRIES:
      msg = "Too many nested sequences in elements; cannot collate."
      warnings.warn(msg, stacklevel=2)
      raise ValueError(msg)
  return list(elements)


def _apply_electrostatics_if_needed(
  elements: list[ProteinTuple],
  *,
  use_electrostatics: bool,
  estat_noise: Sequence[float] | float | None = None,
  estat_noise_mode: str = "direct",
) -> list[ProteinTuple]:
  """Apply electrostatic features if requested.

  Args:
    elements (list[ProteinTuple]): List of protein tuples.
    use_electrostatics (bool): Whether to compute and add electrostatic features.
    estat_noise: Noise level(s) for electrostatics.
    estat_noise_mode: Mode for electrostatic noise ("direct" or "thermal").

  Returns:
    list[ProteinTuple]: Updated list with electrostatic features if requested.

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
      p, noise_scale=noise_val, noise_mode=estat_noise_mode,
    )
    phys_feats.append(np.array(feat))

  return [p._replace(physics_features=feat) for p, feat in zip(elements, phys_feats, strict=False)]


def _apply_md_parameterization(
  elements: list[ProteinTuple],
  *,
  use_md: bool,
) -> list[ProteinTuple]:
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
  ff = force_fields.load_force_field_from_hub("ff14SB")

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

    params = jax_md_bridge.parameterize_system(ff, res_names, atom_names)

    # Convert JAX arrays to numpy and populate ProteinTuple
    p_new = p._replace(
      md_bonds=np.array(params["bonds"]),
      md_bond_params=np.array(params["bond_params"]),
      md_angles=np.array(params["angles"]),
      md_angle_params=np.array(params["angle_params"]),
      md_backbone_indices=np.array(params["backbone_indices"]),
      md_exclusion_mask=np.array(params["exclusion_mask"]),

      charges=np.array(params["charges"]),
      sigmas=np.array(params["sigmas"]),
      epsilons=np.array(params["epsilons"]),
    )
    updated_elements.append(p_new)

  return updated_elements


def _pad_protein(  # noqa: C901
  protein: Protein,
  max_len: int,
  md_dims: dict[str, int] | None = None,
) -> Protein:
  """Pad a single Protein to max_len.

  Args:
    protein (Protein): Protein to pad.
    max_len (int): Maximum length to pad to.
    md_dims: Dictionary of max dimensions for MD fields.

  Returns:
    Protein: Padded protein.

  """
  pad_len = max_len - protein.coordinates.shape[0]
  protein_len = protein.coordinates.shape[0]
  full_coords_len = (
    protein.full_coordinates.shape[0] if protein.full_coordinates is not None else None
  )
  full_coords_pad_len = max_len - full_coords_len if full_coords_len is not None else 0

  # MD padding lengths
  md_pads = {}
  if md_dims:
    if protein.md_bonds is not None:
      md_pads["bonds"] = md_dims["max_bonds"] - protein.md_bonds.shape[0]
    if protein.md_angles is not None:
      md_pads["angles"] = md_dims["max_angles"] - protein.md_angles.shape[0]
    if protein.charges is not None:
      md_pads["atoms"] = md_dims["max_atoms"] - protein.charges.shape[0]

  def pad_fn(
    x: np.ndarray | None,
    *,
    pad_len: int = pad_len,
    protein_len: int = protein_len,
    full_coords_len: int | None = full_coords_len,
    full_coords_pad_len: int = full_coords_pad_len,
  ) -> np.ndarray | None:
    """Pad array along first dimension if it matches the protein residue count."""
    if x is None:
      return None
    if not hasattr(x, "shape") or not hasattr(x, "ndim"):
      return x
    if hasattr(x, "__array__"):
      x = np.asarray(x)
    if x.ndim == 0:
      return x

    # Handle MD fields explicitly by checking if they match known MD arrays
    # This is a bit hacky, better to match by name if tree_map passed keys.
    # But tree_map doesn't pass keys.
    # We can check if x is one of the MD arrays by identity? No, copies.
    # We can check shapes?
    # But `md_bonds` shape (N_bonds, 2) is unique?
    # N_bonds is arbitrary.

    # Better approach: Manually pad MD fields in `pad_and_collate` BEFORE calling `_stack`.
    # Or update `_pad_protein` to handle SPECIFIC fields if we could.
    # Since we can't easily identify fields in `pad_fn`, we should rely on `jax.tree_util.tree_map`
    # only for standard fields, and handle MD fields separately?
    # Or we can use `jax.tree_util.tree_map_with_path` (JAX 0.4.6+).
    # PrxteinMPNN uses JAX.

    # Let's try `tree_map_with_path` if available, or just manual padding for MD fields
    # inside `_pad_protein` before/after tree_map.
    # `Protein` is a dataclass. `tree_map` iterates fields.

    # Let's stick to shape-based heuristics for standard fields,
    # and manually pad MD fields in the wrapper `_pad_protein`.

    if full_coords_len is not None and x.shape[0] == full_coords_len:
      return np.pad(x, ((0, full_coords_pad_len),) + ((0, 0),) * (x.ndim - 1))

    if x.shape[0] == protein_len:
      return np.pad(x, ((0, pad_len),) + ((0, 0),) * (x.ndim - 1))

    return x

  # Pad standard fields
  padded_protein = jax.tree_util.tree_map(pad_fn, protein)

  # Manually pad MD fields if present
  # Manually pad MD fields if present
  if md_dims:
    def pad_array(arr: np.ndarray | None, pad_amt: int) -> np.ndarray | None:
      if arr is None:
          return None
      pads = [(0, pad_amt)] + [(0, 0)] * (arr.ndim - 1)
      return np.pad(arr, pads)

    # Bonds
    if padded_protein.md_bonds is not None:
      p_bonds = pad_array(padded_protein.md_bonds, md_pads.get("bonds", 0))
      p_params = pad_array(padded_protein.md_bond_params, md_pads.get("bonds", 0))
      padded_protein = padded_protein.replace(md_bonds=p_bonds, md_bond_params=p_params)

    # Angles
    if padded_protein.md_angles is not None:
      p_angles = pad_array(padded_protein.md_angles, md_pads.get("angles", 0))
      p_params = pad_array(padded_protein.md_angle_params, md_pads.get("angles", 0))
      padded_protein = padded_protein.replace(md_angles=p_angles, md_angle_params=p_params)



    # Exclusion mask (N_atoms, N_atoms)
    if padded_protein.md_exclusion_mask is not None:
      # Pad both dims
      curr = padded_protein.md_exclusion_mask.shape[0]
      target = md_dims["max_atoms"]
      amt = target - curr
      if amt > 0:
        mask = np.pad(
            padded_protein.md_exclusion_mask,
            ((0, amt), (0, amt)),
            constant_values=False,
        )
        padded_protein = padded_protein.replace(md_exclusion_mask=mask)

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
  elements: Sequence[ProteinTuple],
  *,
  use_electrostatics: bool = False,
  use_vdw: bool = False,  # noqa: ARG001
  estat_noise: Sequence[float] | float | None = None,
  estat_noise_mode: str = "direct",
  vdw_noise: Sequence[float] | float | None = None,  # noqa: ARG001
  vdw_noise_mode: str = "direct",  # noqa: ARG001
  backbone_noise_mode: str = "direct",
  max_length: int | None = None,
) -> Protein:
  """Batch and pad a list of ProteinTuples into a ProteinBatch.

  Take a list of individual `ProteinTuple`s and batch them together into a
  single `ProteinBatch`, padding them to a fixed length.

  Args:
    elements (list[ProteinTuple]): List of protein tuples to collate.
    use_electrostatics (bool): Whether to compute and add electrostatic features.
    use_vdw (bool): Placeholder for van der Waals features (not implemented).
    estat_noise: Noise level(s) for electrostatics.
    estat_noise_mode: Mode for electrostatic noise.
    vdw_noise: Noise level(s) for vdW.
    vdw_noise_mode: Mode for vdW noise.
    backbone_noise_mode: Mode for backbone noise (e.g. "direct", "md").
    max_length (int | None): Fixed length to pad all proteins to. If None, pads to
      the maximum length in the batch (variable per batch).

  Returns:
    Protein: Batched and padded protein ensemble.

  Raises:
    ValueError: If the input list is empty.

  Example:
    >>> ensemble = pad_and_collate_proteins([protein_tuple1, protein_tuple2],
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

  proteins = [Protein.from_tuple_numpy(p) for p in elements]

  # Use fixed max_length if provided, otherwise use max in batch
  pad_len = max_length if max_length is not None else max(p.coordinates.shape[0] for p in proteins)

  # Calculate MD dims
  md_dims = {}
  if backbone_noise_mode == "md":
      max_bonds = 0
      max_angles = 0
      max_atoms = 0
      for p in proteins:
          if p.md_bonds is not None:
              max_bonds = max(max_bonds, p.md_bonds.shape[0])
          if p.md_angles is not None:
              max_angles = max(max_angles, p.md_angles.shape[0])
          if p.charges is not None:
              max_atoms = max(max_atoms, p.charges.shape[0])
      md_dims = {"max_bonds": max_bonds, "max_angles": max_angles, "max_atoms": max_atoms}

  padded_proteins = [_pad_protein(p, pad_len, md_dims) for p in proteins]
  return _stack_padded_proteins(padded_proteins)
