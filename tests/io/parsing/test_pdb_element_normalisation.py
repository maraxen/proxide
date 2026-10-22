"""Element column normalisation at parse time (debt #2366).

A filled PDB element column (cols 77-78) and a blank one (element inferred
from the atom name) must yield the same canonical symbol string. Before the
fix a filled ``CL`` stayed ``CL`` while the blank path gave ``Cl``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

try:
  from proxide.io.parsing.backend import is_rust_parser_available, parse_pdb_raw_rust

  RUST_AVAILABLE = is_rust_parser_available()
except ImportError:
  RUST_AVAILABLE = False

pytestmark = pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")


def _hetatm(serial: int, name: str, element_col: str) -> str:
  """One HETATM line with the PDB fixed columns laid out explicitly."""
  return (
    f"{'HETATM':<6}{serial:>5} {name:<4} {'ION':>3} A{serial:>4}    "
    f"{1.0:8.3f}{2.0:8.3f}{3.0:8.3f}{1.0:6.2f}{0.0:6.2f}"
    f"{'':10}{element_col:>2}"
  )


def test_filled_and_blank_element_columns_give_identical_symbols(tmp_path: Path):
  pdb = tmp_path / "ions.pdb"
  pdb.write_text(
    "\n".join(
      [
        _hetatm(1, "ZN", "ZN"),  # filled, upper case
        _hetatm(2, "CL", "CL"),  # filled, upper case
        _hetatm(3, "ZN", ""),  # blank: element inferred from name
        _hetatm(4, "CL", ""),  # blank: element inferred from name
        "END",
      ]
    )
    + "\n"
  )
  raw = parse_pdb_raw_rust(str(pdb))
  elements = list(raw.elements)
  assert elements == ["Zn", "Cl", "Zn", "Cl"]


def test_unknown_element_symbol_raises(tmp_path: Path):
  pdb = tmp_path / "bad.pdb"
  pdb.write_text(_hetatm(1, "XX", "XX") + "\nEND\n")
  with pytest.raises(Exception, match="PROX-IO-MALFORMED-RECORD"):
    parse_pdb_raw_rust(str(pdb))
