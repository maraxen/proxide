"""mmCIF chain and residue numbering: ``auth`` by default, ``label`` on request (debt #1930).

The fixture makes the two numberings disagree on both chain and residue number, so a
parse that reads the wrong one fails on the value, not just on the shape.
"""

from pathlib import Path

import numpy as np
import pytest
from proxide import OutputSpec, parse_structure
from proxide._proxider import _format_cache_stats, parse_mmcif

CIF = """data_TEST
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.auth_asym_id
_atom_site.auth_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
ATOM 1 N ALA B 1 A 7 0.000 0.000 0.000
ATOM 2 CA ALA B 1 A 7 1.000 0.000 0.000
ATOM 3 C ALA B 1 A 7 2.000 0.000 0.000
ATOM 4 O ALA B 1 A 7 2.500 1.000 0.000
ATOM 5 N GLY B 2 A 8 3.000 0.000 0.000
ATOM 6 CA GLY B 2 A 8 4.000 0.000 0.000
ATOM 7 C GLY B 2 A 8 5.000 0.000 0.000
ATOM 8 O GLY B 2 A 8 5.500 1.000 0.000
"""


@pytest.fixture
def cif_path(tmp_path: Path) -> Path:
  path = tmp_path / "label_differs_from_auth.cif"
  path.write_text(CIF)
  return path


def test_parse_mmcif_defaults_to_auth_numbering(cif_path: Path) -> None:
  raw = parse_mmcif(str(cif_path))
  assert set(raw["chain_ids"]) == {"A"}
  assert list(np.asarray(raw["res_ids"])) == [7, 7, 7, 7, 8, 8, 8, 8]


def test_parse_mmcif_label_numbering_reads_label_columns(cif_path: Path) -> None:
  raw = parse_mmcif(str(cif_path), numbering="label")
  assert set(raw["chain_ids"]) == {"B"}
  assert list(np.asarray(raw["res_ids"])) == [1, 1, 1, 1, 2, 2, 2, 2]


@pytest.mark.parametrize("bad", ["", "Auth", "label ", "both"])
def test_numbering_rejects_unknown_values(cif_path: Path, bad: str) -> None:
  with pytest.raises(ValueError, match=r'"auth" or "label"'):
    parse_mmcif(str(cif_path), numbering=bad)
  with pytest.raises(ValueError, match=r'"auth" or "label"'):
    parse_structure(str(cif_path), numbering=bad)


def test_numbering_is_part_of_the_format_cache_key(cif_path: Path) -> None:
  spec = OutputSpec(enable_caching=True)

  hits0, misses0, _ = _format_cache_stats()
  auth = parse_structure(str(cif_path), spec)
  hits1, misses1, _ = _format_cache_stats()
  assert (hits1 - hits0, misses1 - misses0) == (0, 1)

  # The label request must not be answered from the auth entry.
  label = parse_structure(str(cif_path), spec, numbering="label")
  hits2, misses2, _ = _format_cache_stats()
  assert (hits2 - hits1, misses2 - misses1) == (0, 1), "label parse shared the auth entry"
  assert np.asarray(label.residue_index).tolist() != np.asarray(auth.residue_index).tolist()

  # Each numbering is then served from its own entry.
  auth_again = parse_structure(str(cif_path), spec, numbering="auth")
  label_again = parse_structure(str(cif_path), spec, numbering="label")
  hits3, misses3, _ = _format_cache_stats()
  assert (hits3 - hits2, misses3 - misses2) == (2, 0)
  assert np.asarray(auth_again.residue_index).tolist() == np.asarray(auth.residue_index).tolist()
  assert np.asarray(label_again.residue_index).tolist() == np.asarray(label.residue_index).tolist()


def test_numbering_label_pins_1ubq_chains() -> None:
  # Real fixture: HOH waters are label chain B and auth chain A.
  path = "crates/proxide-io/tests/data/1UBQ.cif"
  assert set(parse_mmcif(path, numbering="label")["chain_ids"]) == {"A", "B"}
  assert set(parse_mmcif(path)["chain_ids"]) == {"A"}
