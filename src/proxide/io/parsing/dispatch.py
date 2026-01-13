"""Unified dispatch for parsing protein structures."""

from __future__ import annotations

import pathlib
from typing import IO, TYPE_CHECKING, Any

# Register parsers by importing modules lazily or when needed
# NOTE: biotite.py removed - Rust parser is now the primary parser for pdb/cif
from proxide.io.parsing.registry import FormatNotSupportedError, get_parser

if TYPE_CHECKING:
  from collections.abc import Sequence


def _infer_format(path: pathlib.Path | None) -> str | None:
  """Infer file format from path suffix."""
  if path is None:
    return None
  suffix = path.suffix.lower()
  if suffix == ".pdb":
    return "pdb"
  if suffix in (".cif", ".mmcif"):
    return "cif"
  if suffix == ".pqr":
    return "pqr"
  if suffix in (".fcz", ".foldcomp"):
    return "foldcomp"
  if suffix in (".dcd", ".xtc", ".h5", ".hdf5"):
    # Could be mdtraj or mdcath (handled inside mdtraj parser or check here if needed)
    # Assuming mdtraj for dispatch purposes, let parser handle specific HDF5 types if needed
    return "mdtraj"
  return None


def load_structure(
  file_path: str | pathlib.Path | IO[str],
  file_format: str | None = None,
  chain_id: str | Sequence[str] | None = None,
  return_type: str = "Protein",
  **kwargs: Any,  # noqa: ANN401
) -> Any:
  """Load a protein structure from a file.

  Args:
      file_path: Path to the file or file-like object.
      file_format: Format of the file (e.g., "pdb", "cif", "pqr", "foldcomp").
          If None, inferred from extension.
      chain_id: Chain ID(s) to load.
      **kwargs: Additional arguments passed to specific parsers.

  Returns:
      ProcessedStructure or ProteinStream (for FoldComp/Trajectories).

  """
  path = None
  if isinstance(file_path, str):
    path = pathlib.Path(file_path)
  elif isinstance(file_path, pathlib.Path):
    path = file_path

  if file_format is None:
    file_format = _infer_format(path)

  # Default to pdb for file-like objects (e.g. StringIO) if format not specified
  if file_format is None and path is None:
    file_format = "pdb"

  if file_format == "mdtraj":
    from proxide.io.parsing import mdtraj  # noqa: F401
  elif file_format == "pqr":
    from proxide.io.parsing import pqr  # noqa: F401
  elif file_format in ("pdb", "cif", "mmcif"):
    from proxide.io.parsing import backend as rust  # noqa: F401
  elif file_format == "foldcomp":
    # Foldcomp might be in its own module or handled by rust
    pass

  if file_format is None:
    msg = f"Failed to infer file format for: {file_path}"
    raise FormatNotSupportedError(msg)

  parser = get_parser(file_format)
  if not parser:
    msg = (
      f"Failed to parse structure from source: {file_path}. Unsupported file format: {file_format}"
    )
    raise FormatNotSupportedError(msg)

  return parser(file_path, chain_id=chain_id, return_type=return_type, **kwargs)


parse_input = load_structure
