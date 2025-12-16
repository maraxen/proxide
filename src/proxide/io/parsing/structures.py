"""Structure data classes for parsing.

This module provides intermediate data structures used during parsing.
The ProcessedStructure class bridges between raw parsed data and
the final Protein container.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
  import numpy as np


@dataclasses.dataclass
class ProcessedStructure:
  """A structure that has been parsed and processed.

  This is an intermediate representation used by legacy parsers.
  New code should use Rust parsing directly via parse_structure().

  Attributes:
      atom_array: Raw atom data (coordinates, names, etc). Can be a dict
          from Rust or legacy biotite AtomArray for backward compatibility.
      r_indices: Residue indices for each atom
      chain_ids: Chain index for each atom
      charges: Partial charges (optional)
      radii: Van der Waals radii (optional)
      sigmas: LJ sigma parameters (optional)
      epsilons: LJ epsilon parameters (optional)

  """

  atom_array: Any  # Dict from Rust or legacy AtomArray
  r_indices: np.ndarray
  chain_ids: np.ndarray
  charges: np.ndarray | None = None
  radii: np.ndarray | None = None
  sigmas: np.ndarray | None = None
  epsilons: np.ndarray | None = None
