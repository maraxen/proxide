"""IO utilities: structure fetching/parsing, plus FASTA/A3M and Newick parsers."""

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
from proxide.io.parsing import load_structure, parse_input, parse_structure

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
