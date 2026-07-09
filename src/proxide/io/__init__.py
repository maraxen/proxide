"""IO utilities: structure fetching/parsing, plus FASTA/A3M and Newick parsers."""

from __future__ import annotations

from typing import Any

from proxide._proxider import read_fasta, read_newick  # type: ignore[unresolved-import]
from proxide.io.fetching import (
  fetch_afdb,
  fetch_md_cath,
  fetch_rcsb,
)
from proxide.io.fixtures import (
  assert_bundle_keys,
  flatten_tensor_dict,
  load_tensor_bundle,
  save_tensor_bundle,
  unflatten_tensor_dict,
)

__all__ = [
  "fetch_afdb",
  "fetch_md_cath",
  "fetch_rcsb",
  "parse_structure",
  "load_structure",
  "parse_input",
  "read_fasta",
  "read_newick",
  "flatten_tensor_dict",
  "unflatten_tensor_dict",
  "save_tensor_bundle",
  "load_tensor_bundle",
  "assert_bundle_keys",
]


def __getattr__(name: str) -> Any:
  # Lazy: parsing pulls JAX; fixtures/FASTA must work in torch-only vendor venvs.
  if name in {"parse_structure", "load_structure", "parse_input"}:
    from proxide.io.parsing import load_structure, parse_input, parse_structure

    mapping = {
      "parse_structure": parse_structure,
      "load_structure": load_structure,
      "parse_input": parse_input,
    }
    return mapping[name]
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
