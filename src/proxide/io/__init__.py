"""IO utilities: structure fetching/parsing, plus FASTA/A3M and Newick parsers."""

from proxide._proxider import read_fasta, read_newick  # type: ignore[unresolved-import]
from proxide.io.fetching import (
  fetch_afdb,
  fetch_md_cath,
  fetch_rcsb,
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
]
