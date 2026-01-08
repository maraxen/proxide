"""PQR file parsing utilities.

This module uses the Rust PQR parser for high-performance parsing.
PQR files contain atom coordinates along with partial charges and radii,
used for electrostatics calculations (Poisson-Boltzmann, etc).
"""

import logging
import pathlib
from collections.abc import Generator, Sequence
from typing import IO, Any

import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike

from proxide import _oxidize
from proxide.core.atomic_system import AtomicSystem
from proxide.io.parsing.registry import ParsingError, register_parser

logger = logging.getLogger(__name__)

# Type alias for PQR stream
PQRStream = Generator[AtomicSystem, None, None]


def parse_pqr_rust(file_path: str | pathlib.Path) -> dict[str, ArrayLike]:
  """Parse a PQR file using the Rust parser.

  Args:
      file_path: Path to PQR file

  Returns:
      Dictionary with parsed atom data including charges and radii

  Raises:
      ValueError: If parsing fails

  """
  path = str(file_path) if isinstance(file_path, pathlib.Path) else file_path
  return _oxidize.parse_pqr(path)


def _convert_rust_pqr_to_system(
  data: dict[str, Any],
  chain_id: str | Sequence[str] | None = None,
) -> AtomicSystem:
  """Convert Rust PQR parse result to AtomicSystem object.

  Args:
      data: Dictionary from Rust parse_pqr
      chain_id: Optional chain ID(s) to filter

  Returns:
      AtomicSystem object with charges and radii

  """
  # Extract data from Rust result
  num_atoms = data["num_atoms"]

  # Coords come as flattened array, reshape to (N, 3)
  coords_flat = np.array(data["coords"], dtype=np.float32)
  coords = coords_flat.reshape(num_atoms, 3)

  atom_names = list(data["atom_names"])
  res_names = list(data["res_names"])
  chain_ids = list(data["chain_ids"])
  elements = list(data["elements"])
  res_ids = np.array(data["res_ids"], dtype=np.int32)

  # PQR files have charge and radius
  charges = np.array(data.get("charges", []), dtype=np.float32)
  radii = np.array(data.get("radii", []), dtype=np.float32)

  # Filter by chain if requested
  if chain_id is not None:
    chain_set = {chain_id} if isinstance(chain_id, str) else set(chain_id)
    mask = np.array([c in chain_set for c in chain_ids])
    coords = coords[mask]
    atom_names = [a for a, m in zip(atom_names, mask, strict=False) if m]
    res_names = [r for r, m in zip(res_names, mask, strict=False) if m]
    chain_ids = [c for c, m in zip(chain_ids, mask, strict=False) if m]
    elements = [e for e, m in zip(elements, mask, strict=False) if m]
    res_ids = res_ids[mask]
    if len(charges) > 0:
      charges = charges[mask]
    if len(radii) > 0:
      radii = radii[mask]
    num_atoms = len(atom_names)

  if num_atoms == 0:
    msg = "No atoms found in PQR file after chain filtering."
    raise ValueError(msg)

  # Create atom mask (all atoms present in PQR)
  atom_mask = np.ones(num_atoms, dtype=np.float32)

  # Create chain index by mapping chain_ids to integers
  unique_chains = sorted(list(set(chain_ids)))
  chain_map = {c: i for i, c in enumerate(unique_chains)}
  chain_indices = np.array([chain_map[c] for c in chain_ids], dtype=np.int32)

  # Use from_arrays factory method
  return AtomicSystem.from_arrays(
    coordinates=jnp.array(coords),
    atom_mask=jnp.array(atom_mask),
    elements=elements,
    atom_names=atom_names,
    residue_index=jnp.array(res_ids),
    chain_index=jnp.array(chain_indices),
    charges=jnp.array(charges) if len(charges) > 0 else None,
    radii=jnp.array(radii) if len(radii) > 0 else None,
  )


@register_parser(["pqr"])
def load_pqr(
  file_path: str | pathlib.Path | IO[str],
  chain_id: str | Sequence[str] | None = None,
  *,
  extract_dihedrals: bool = False,  # noqa: ARG001
  populate_physics: bool = False,  # noqa: ARG001
  force_field_name: str = "ff14SB",  # noqa: ARG001
  return_type: str = "AtomicSystem",  # noqa: ARG001
  **kwargs: Any,  # noqa: ANN401, ARG001
) -> Any:  # Returns PQRStream (yields AtomicSystem)
  """Load a PQR file.

  Uses the Rust parser for high-performance parsing.
  PQR files already contain charge and radius information.

  Args:
      file_path: Path to PQR file
      chain_id: Optional chain ID(s) to filter
      extract_dihedrals: Unused (PQR doesn't have dihedrals)
      populate_physics: Unused (PQR already has physics params)
      force_field_name: Unused
      return_type: Ignored - PQR always returns AtomicSystem
      **kwargs: Additional arguments (ignored)

  Yields:
      AtomicSystem objects with charges and radii

  Raises:
      ParsingError: If parsing fails

  """
  # Handle file-like objects
  if hasattr(file_path, "read"):
    msg = "PQR Rust parser requires file path, not file-like object"
    raise ParsingError(msg)

  path = pathlib.Path(file_path) if isinstance(file_path, str) else file_path

  try:
    data = parse_pqr_rust(path)
    system = _convert_rust_pqr_to_system(data, chain_id)
    yield system
  except Exception as e:
    msg = f"Failed to parse PQR from source: {file_path}. {e}"
    raise ParsingError(msg) from e
