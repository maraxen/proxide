"""Unified dispatch for parsing protein structures."""

from __future__ import annotations

import pathlib
from typing import IO, TYPE_CHECKING, Any

# Register parsers by importing modules
from priox.io.parsing import biotite, mdtraj, pqr  # noqa: F401
from priox.io.parsing.registry import FormatNotSupportedError, get_parser

if TYPE_CHECKING:
  from collections.abc import Sequence

  from priox.core.containers import ProteinStream


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
  **kwargs: Any,  # noqa: ANN401
) -> ProteinStream:
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

  parser = get_parser(file_format)
  if not parser:
      msg = (
        f"Failed to parse structure from source: {file_path}. "
        f"Unsupported file format: {file_format}"
      )
      raise FormatNotSupportedError(msg)

  return parser(file_path, chain_id=chain_id, **kwargs)


parse_input = load_structure
