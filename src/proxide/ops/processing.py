"""Pre-process various input formats into a single HDF5 file for efficient loading."""

import logging
import pathlib
import re
import warnings
from collections.abc import Generator, Iterator, Sequence
from typing import IO, Any, Literal

from proxide.core.containers import Protein
from proxide.io.parsing.dispatch import load_structure as parse_input
from proxide.io.parsing.foldcomp import FoldCompDatabase, get_protein_structures

# Instantiate the logger
logger = logging.getLogger(__name__)

# --- Regex patterns for ID matching ---
_FOLDCOMP_AFDB_PATTERN = re.compile(r"AF-[A-Z0-9]{6,10}-F[0-9]+-model_v[0-9]+")
_FOLDCOMP_ESM_PATTERN = re.compile(r"MGY[PC][0-9]{12,}(?:\.[0-9]+)?(?:_[0-9]+)?")
_FOLDCOMP_PATTERN = re.compile(
  rf"(?:{_FOLDCOMP_AFDB_PATTERN.pattern})|(?:{_FOLDCOMP_ESM_PATTERN.pattern})",
)
_PDB_PATTERN = re.compile(r"^[a-zA-Z0-9]{4}$")
_MD_CATH_PATTERN = re.compile(r"[a-zA-Z0-9]{5}[0-9]{2}")


def _resolve_inputs(  # noqa: C901
  inputs: Sequence[str | IO[str] | pathlib.Path],
  foldcomp_database: FoldCompDatabase | None = None,
  rcsb_format: Literal["mmcif", "pdb"] = "mmcif",
) -> Generator[str | pathlib.Path | IO[str] | Protein, None, None]:
  """Resolve a heterogeneous list of inputs into parseable sources.

  This generator categorizes each input and yields a source that `parse_input`
  can directly handle (file paths or StringIO objects). It fetches data for
  PDB, MD-CATH, and FoldComp IDs.

  Args:
      inputs: A sequence of input items.
      foldcomp_database: An optional FoldCompDatabase for resolving FoldComp IDs.
      rcsb_format: The format to fetch from RCSB ("mmcif" or "pdb").

  Yields:
      A parseable source (str, pathlib.Path, or StringIO).

  """
  from proxide.io.fetching import fetch_afdb, fetch_md_cath, fetch_rcsb

  foldcomp_ids = []

  # Configuration constant (TODO: move to config module)
  AFDB_FETCH_LIMIT = 50

  for item in inputs:
    try:
      if isinstance(item, str):
        if _FOLDCOMP_PATTERN.match(item):
          foldcomp_ids.append(item)
          continue
        if _PDB_PATTERN.match(item) and not pathlib.Path(item).exists():
          yield fetch_rcsb(item, format_type=rcsb_format)
          continue
        if _MD_CATH_PATTERN.match(item) and not pathlib.Path(item).exists():
          yield fetch_md_cath(item)
          continue

        path = pathlib.Path(item)
        if path.is_file():
          yield path
        elif path.is_dir():
          yield from (p for p in path.rglob("*") if p.is_file())
        else:
          warnings.warn(f"Input string '{item}' could not be categorized.", stacklevel=2)
      elif hasattr(item, "read"):  # StringIO-like
        yield item
      else:
        warnings.warn(f"Unsupported input type: {type(item)}", stacklevel=2)
    except Exception as e:  # noqa: BLE001
      warnings.warn(f"Failed to resolve input '{item}': {e}", stacklevel=2)

  if foldcomp_ids:
    # Logic: If database is provided and IDs > limit, use database.
    # Otherwise (or if no DB), use direct fetch.
    use_database = foldcomp_database is not None and len(foldcomp_ids) > AFDB_FETCH_LIMIT

    if use_database and foldcomp_database:
      yield from get_protein_structures(foldcomp_ids, foldcomp_database)
    else:
      # Fallback to direct fetching for each ID
      # Verify if IDs are compatible with AFDB fetching (Uniprot ID extraction might be needed)
      # _FOLDCOMP_PATTERN matches both AFDB and ESM IDs.
      # fetch_afdb expects Uniprot ID.
      # We need to extract Uniprot ID from AF-Identifier if possible.
      # Pattern: AF-{UNIPROT}-F1-model_v{VERSION}

      for fid in foldcomp_ids:
        # Try to parse AFDB ID
        af_match = _FOLDCOMP_AFDB_PATTERN.fullmatch(fid)
        if af_match:
          # Extract parts purely for verifying it looks like AFDB,
          # but fetch_afdb might need just the uniprot part?
          # Actually `fetch_afdb` implementation takes `uniprot_id`.
          # But the input here is the full ID like `AF-P12345-F1-model_v4`.
          # We should probably strip it?
          # Or implementation of fetch_afdb expects uniprot id.
          # Let's extract:
          parts = fid.split("-")
          if len(parts) >= 2 and parts[0] == "AF":
            uid = parts[1]
            # Version?
            # parts[-1] is model_v4
            # version extraction:
            version = 4  # Default
            try:
              v_str = parts[-1].split("_v")[1]
              version = int(v_str)
            except:
              pass

            try:
              yield fetch_afdb(uid, version=version)
            except Exception as e:
              warnings.warn(f"Failed to fetch AFDB ID {fid}: {e}", stacklevel=2)
          else:
            warnings.warn(
              f"Cannot fetch ID {fid} directly (only AFDB IDs supported for direct fetch).",
              stacklevel=2,
            )
        else:
          warnings.warn(
            f"Cannot fetch ID {fid} directly (ESM/other IDs not supported for direct fetch).",
            stacklevel=2,
          )


def _get_input_chain_pairs(
  inputs: Sequence[str | pathlib.Path | IO[str]],
  chain_id_arg: Sequence[str] | str | set[str] | dict[str, str] | None,
) -> Iterator[tuple[str | pathlib.Path | IO[str], Any]]:
  """Pair inputs with their corresponding chain_id based on the argument type."""
  if chain_id_arg is None or isinstance(chain_id_arg, str):
    return ((i, chain_id_arg) for i in inputs)

  if isinstance(chain_id_arg, set):
    chain_list = list(chain_id_arg)
    return ((i, chain_list) for i in inputs)

  if isinstance(chain_id_arg, dict):

    def _get_chain_from_dict(item: str | pathlib.Path | IO[str]) -> str | set[str]:
      key = str(item)
      if key in chain_id_arg:
        return chain_id_arg[key]

      if isinstance(item, pathlib.Path) and item.name in chain_id_arg:
        return chain_id_arg[item.name]

      msg = f"Input '{key}' not found in chain_id dictionary keys."
      raise ValueError(msg)

    return ((i, _get_chain_from_dict(i)) for i in inputs)

  if isinstance(chain_id_arg, Sequence):
    if len(chain_id_arg) != len(inputs):
      msg = (
        f"chain_id sequence length ({len(chain_id_arg)}) "
        f"does not match inputs length ({len(inputs)})."
      )
      raise ValueError(msg)
    return zip(inputs, chain_id_arg, strict=False)

  msg = f"Unsupported type for chain_id: {type(chain_id_arg)}"
  raise TypeError(msg)


def frame_iterator_from_inputs(
  inputs: Sequence[str | pathlib.Path | IO[str]],
  parse_kwargs: dict[str, Any] | None = None,
  foldcomp_database: FoldCompDatabase | None = None,
  rcsb_format: Literal["mmcif", "pdb"] = "mmcif",
) -> Iterator[Protein]:
  """Create a generator that yields Protein frames from mixed inputs.

  Supports flexible chain_id assignment:
  - None/str: Applies to all inputs.
  - Set: Applies as a filter (any of these chains) to all inputs.
  - Sequence: 1:1 mapping with inputs (lengths must match).
  - Dict: Maps input string representation to chain_id.
  """
  parse_kwargs = parse_kwargs or {}
  chain_id_arg = parse_kwargs.pop("chain_id", None)

  input_chain_pairs = _get_input_chain_pairs(inputs, chain_id_arg)

  for input_item, specific_chain_id in input_chain_pairs:
    resolved_sources = _resolve_inputs(
      [input_item],
      foldcomp_database,
      rcsb_format=rcsb_format,
    )

    for source in resolved_sources:
      if isinstance(source, Protein):
        yield source
      else:
        yield from parse_input(
          source,
          chain_id=specific_chain_id,
          **parse_kwargs,
        )
