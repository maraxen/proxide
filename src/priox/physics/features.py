"""Compute physics-based node features for protein structures."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from priox.geometry.transforms import compute_backbone_coordinates
from priox.physics.constants import BOLTZMANN_KCAL
from priox.physics.electrostatics import (
  compute_coulomb_forces_at_backbone,
  compute_noised_coulomb_forces_at_backbone,
)
from priox.physics.projections import project_forces_onto_backbone
from priox.physics.vdw import (
  compute_lj_forces_at_backbone,
  compute_noised_lj_forces_at_backbone,
)

if TYPE_CHECKING:
  from collections.abc import Sequence

  from priox.core.containers import ProteinTuple


def _resolve_sigma(
  value: float | jax.Array | None,
  mode: str,
) -> float | jax.Array:
  """Resolve the noise standard deviation (sigma) from the input value and mode.

  Args:
      value: The noise parameter.
             - If mode='direct', this is the raw sigma.
             - If mode='thermal', this is T (Kelvin).
      mode: 'direct' or 'thermal'.

  Returns:
      The calculated standard deviation (sigma).

  """
  # Treat None as 0.0
  if value is None:
    return 0.0

  val = jnp.asarray(value)

  if mode == "direct":
    return val

  if mode == "thermal":
    # Physics Formula: sigma = sqrt(0.5 * R * T)
    # We clamp T to 0.0 to prevent NaN from negative sqrt
    thermal_energy = jnp.maximum(0.5 * BOLTZMANN_KCAL * val, 0.0)
    return jnp.sqrt(thermal_energy)

  msg = f"Unknown noise mode: {mode}"
  raise ValueError(msg)




def _compute_electrostatic_features_raw(
  backbone_positions: jax.Array,
  all_positions: jax.Array,
  all_charges: jax.Array,
  noise_scale: float | jax.Array = 0.0,
  key: jax.Array | None = None,
) -> jax.Array:
  """Compute electrostatic features from raw arrays.

  Args:
      backbone_positions: (n_residues, 5, 3)
      all_positions: (n_atoms, 3)
      all_charges: (n_atoms,)
      noise_scale: Noise scale.
      key: PRNG key.

  Returns:
      Features (n_residues, 5)

  """
  n_residues = backbone_positions.shape[0]
  backbone_positions_flat = backbone_positions.reshape(-1, 3)  # (n_residues*5, 3)

  distances = jnp.linalg.norm(
    backbone_positions_flat[:, None, :] - all_positions[None, :, :],
    axis=-1,
  )

  closest_indices = jnp.argmin(distances, axis=1)
  backbone_charges_flat = all_charges[closest_indices]
  backbone_charges = backbone_charges_flat.reshape(n_residues, 5)

  if key is not None:
    forces_at_backbone = compute_noised_coulomb_forces_at_backbone(
      backbone_positions,
      all_positions,
      backbone_charges,
      all_charges,
      noise_scale=noise_scale,
      key=key,
    )
  else:
    # Validation logic: If user provided noise_scale but no key, this is an error
    # We try to detect this for concrete values (not tracers) to warn users
    # This prevents silent deterministic execution when stochastic was intended
    try:
      if float(noise_scale) > 0.0:
        msg = "Must provide key when noise_scale > 0"
        raise ValueError(msg)
    except (ValueError, TypeError):
      # Ignore if conversion fails (e.g. Tracer), rely on caller to be correct
      pass

    forces_at_backbone = compute_coulomb_forces_at_backbone(
      backbone_positions,
      all_positions,
      backbone_charges,
      all_charges,
    )

  return project_forces_onto_backbone(
    forces_at_backbone,
    backbone_positions,
  )


def compute_electrostatic_node_features(
  protein: ProteinTuple,
  *,
  noise_scale: float | jax.Array | None = None,
  noise_mode: str = "direct",
  key: jax.Array | None = None,
) -> jax.Array:
  """Compute SE(3)-invariant electrostatic features for each residue.

  See `_compute_electrostatic_features_raw` for implementation details.
  """
  if protein.charges is None:
    msg = "ProteinTuple must have charges (PQR data) to compute electrostatic features"
    raise ValueError(msg)

  if protein.full_coordinates is None:
    msg = "ProteinTuple must have full_coordinates to compute electrostatic features"
    raise ValueError(msg)

  backbone_positions = compute_backbone_coordinates(
    jnp.array(protein.coordinates),
  )
  all_positions = jnp.array(protein.full_coordinates).reshape(-1, 3)
  all_charges = jnp.array(protein.charges).reshape(-1)

  sigma = _resolve_sigma(noise_scale, noise_mode)

  return _compute_electrostatic_features_raw(
    backbone_positions,
    all_positions,
    all_charges,
    noise_scale=sigma,
    key=key,
  )


def _compute_vdw_features_raw(
  backbone_positions: jax.Array,
  all_positions: jax.Array,
  all_sigmas: jax.Array,
  all_epsilons: jax.Array,
  noise_scale: float | jax.Array = 0.0,
  key: jax.Array | None = None,
) -> jax.Array:
  """Compute vdW features from raw arrays.

  Args:
      backbone_positions: (n_residues, 5, 3)
      all_positions: (n_atoms, 3)
      all_sigmas: (n_atoms,)
      all_epsilons: (n_atoms,)
      noise_scale: Noise scale.
      key: PRNG key.

  Returns:
      Features (n_residues, 5)

  """
  n_residues = backbone_positions.shape[0]
  backbone_positions_flat = backbone_positions.reshape(-1, 3)

  distances = jnp.linalg.norm(
    backbone_positions_flat[:, None, :] - all_positions[None, :, :],
    axis=-1,
  )
  closest_indices = jnp.argmin(distances, axis=1)

  backbone_sigmas_flat = all_sigmas[closest_indices]
  backbone_epsilons_flat = all_epsilons[closest_indices]

  backbone_sigmas = backbone_sigmas_flat.reshape(n_residues, 5)
  backbone_epsilons = backbone_epsilons_flat.reshape(n_residues, 5)

  if key is not None:
    forces_at_backbone = compute_noised_lj_forces_at_backbone(
      backbone_positions,
      all_positions,
      backbone_sigmas,
      backbone_epsilons,
      all_sigmas,
      all_epsilons,
      noise_scale=noise_scale,
      key=key,
    )
  else:
    # Validation logic: If user provided noise_scale but no key, this is an error
    try:
      if float(noise_scale) > 0.0:
        msg = "Must provide key when noise_scale > 0"
        raise ValueError(msg)
    except (ValueError, TypeError):
      pass

    forces_at_backbone = compute_lj_forces_at_backbone(
      backbone_positions,
      all_positions,
      backbone_sigmas,
      backbone_epsilons,
      all_sigmas,
      all_epsilons,
    )

  return project_forces_onto_backbone(
    forces_at_backbone,
    backbone_positions,
  )


def compute_vdw_node_features(
  protein: ProteinTuple,
  *,
  noise_scale: float | jax.Array | None = None,
  noise_mode: str = "direct",
  key: jax.Array | None = None,
) -> jax.Array:
  """Compute SE(3)-invariant Van der Waals features for each residue.

  See `_compute_vdw_features_raw` for implementation details.
  """
  if protein.sigmas is None or protein.epsilons is None:
    msg = "ProteinTuple must have sigmas and epsilons to compute vdW features"
    raise ValueError(msg)

  if protein.full_coordinates is None:
    msg = "ProteinTuple must have full_coordinates to compute vdW features"
    raise ValueError(msg)

  backbone_positions = compute_backbone_coordinates(
    jnp.array(protein.coordinates),
  )
  all_positions = jnp.array(protein.full_coordinates).reshape(-1, 3)
  all_sigmas = jnp.array(protein.sigmas).reshape(-1)
  all_epsilons = jnp.array(protein.epsilons).reshape(-1)

  sigma = _resolve_sigma(noise_scale, noise_mode)

  return _compute_vdw_features_raw(
    backbone_positions,
    all_positions,
    all_sigmas,
    all_epsilons,
    noise_scale=sigma,
    key=key,
  )


def compute_electrostatic_features_batch(
  proteins: Sequence[ProteinTuple],
  max_length: int | None = None,
  *,
  pad_value: float = 0.0,
) -> tuple[jax.Array, jax.Array]:
  """Compute electrostatic features for a batch of proteins with padding.

  Args:
      proteins: List of ProteinTuple instances
      max_length: Maximum sequence length for padding. If None, uses the
        longest sequence in the batch.
      pad_value: Value to use for padding (default: 0.0)

  Returns:
      features: (batch_size, max_length, 5) padded feature arrays
      mask: (batch_size, max_length) binary mask (1.0 for real residues, 0.0 for padding)

  Example:
      >>> proteins = [load_pqr_file(f"protein_{i}.pqr") for i in range(4)]
      >>> features, mask = compute_electrostatic_features_batch(proteins, max_length=128)
      >>> print(features.shape, mask.shape)
      (4, 128, 5) (4, 128)

  """
  if not proteins:
    msg = "Must provide at least one protein"
    raise ValueError(msg)

  features_list = [compute_electrostatic_node_features(p) for p in proteins]

  lengths = [f.shape[0] for f in features_list]
  if max_length is None:
    max_length = max(lengths)
  elif max_length < max(lengths):
    msg = f"max_length={max_length} is less than longest sequence ({max(lengths)})"
    raise ValueError(msg)

  batch_size = len(proteins)
  n_features = 5

  features_padded = jnp.full((batch_size, max_length, n_features), pad_value)
  mask = jnp.zeros((batch_size, max_length))

  for i, (features, length) in enumerate(zip(features_list, lengths, strict=False)):
    features_padded = features_padded.at[i, :length, :].set(features)
    mask = mask.at[i, :length].set(1.0)

  return features_padded, mask
