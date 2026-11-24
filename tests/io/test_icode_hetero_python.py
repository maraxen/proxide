"""Python side of insertion codes and the per-atom hetero flag (spec 261123, T3).

Covers the ``Protein`` fields ``insertion_codes`` and ``hetero`` on the fixture
``tests/data/icode_ligand.pdb`` (two chains; residues 52, 52A, 52B; one MSE; one ZN, one
LIG and two HOH HETATM groups; 53 atoms):

- ``parse_structure`` per layout, and ``from_rust_dict`` validation (D1);
- each propagation site of the D2 table: truncate, concat, pad, stack, chain filter;
- the npz round trip and the batched-input guard of the writer;
- parm7 (``hetero is None``) and PQR (raw hetero flag from the record types).

The Rust producers and the cache race are in ``test_icode_hetero_rust_producers.py``.
"""

from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path

import numpy as np
import pytest

import proxide._proxider as proxider
from proxide import CoordFormat, OutputSpec, parse_structure
from proxide.core.containers import Protein
from proxide.io.fixtures import load_npz_icode_fields
from proxide.io.parsing.backend import load_rust, parse_amber_trajectory
from proxide.io.writing import _reject_if_batched, write_npz
from proxide.ops.transforms import (
  _pad_protein,
  _stack_padded_proteins,
  concatenate_proteins_for_inter_mode,
  truncate_protein,
)

DATA = Path(__file__).resolve().parents[1] / "data"
FIXTURE = DATA / "icode_ligand.pdb"
PQR = DATA / "1a00.pqr"

# Residues of the fixture in order (spec T1): 51, 52, 52A, 52B, MSE 53, 54, chain B 1 and 2.
RESIDUE_CODES = [" ", " ", "A", "B", " ", " ", " ", " "]
CHAIN_A_RESIDUE_CODES = [" ", " ", "A", "B", " ", " "]


def _atom_lines() -> list[str]:
  return [ln for ln in FIXTURE.read_text().splitlines() if ln.startswith(("ATOM", "HETATM"))]


def _parse(layout: CoordFormat) -> Protein:
  return parse_structure(FIXTURE, OutputSpec(coord_format=layout), use_jax=False)


@pytest.fixture(scope="module")
def atom37() -> Protein:
  return _parse(CoordFormat.Atom37)


@pytest.fixture(scope="module")
def full() -> Protein:
  return _parse(CoordFormat.Full)


def _raw(layout: CoordFormat) -> dict:
  return proxider.parse_structure(str(FIXTURE), OutputSpec(coord_format=layout), "auth")


# --- parse_structure per layout ---------------------------------------------------------------


def test_atom37_codes_are_per_residue(atom37: Protein) -> None:
  assert atom37.insertion_codes == RESIDUE_CODES
  assert atom37.hetero is None


def test_atom14_codes_are_per_residue() -> None:
  p = _parse(CoordFormat.Atom14)
  assert p.insertion_codes == RESIDUE_CODES
  assert p.hetero is None


def test_full_codes_are_per_atom(full: Protein) -> None:
  expected = [line[26] for line in _atom_lines()]
  assert len(full.insertion_codes) == 53
  assert full.insertion_codes == expected
  assert "A" in full.insertion_codes and "B" in full.insertion_codes


def test_full_hetero_is_true_exactly_on_hetatm_rows(full: Protein) -> None:
  assert isinstance(full.hetero, np.ndarray)
  assert full.hetero.dtype == np.bool_
  assert full.hetero.shape == (53,)
  expected = np.array([ln.startswith("HETATM") for ln in _atom_lines()])
  np.testing.assert_array_equal(full.hetero, expected)
  # The HETATM groups named in the fixture are the ones flagged.
  flagged = {name for name, h in zip(full.res_names, full.hetero, strict=True) if h}
  assert flagged == {"MSE", "ZN", "LIG", "HOH"}


# --- from_rust_dict validation (D1) -----------------------------------------------------------


@pytest.mark.parametrize("bad", ["AB", "", "é", "\t"])
def test_invalid_insertion_code_raises(bad: str) -> None:
  raw = _raw(CoordFormat.Atom37)
  codes = list(raw["insertion_codes"])
  codes[2] = bad
  raw["insertion_codes"] = codes
  with pytest.raises(ValueError, match="insertion_codes"):
    Protein.from_rust_dict(raw, use_jax=False)


def test_insertion_codes_wrong_length_raises() -> None:
  raw = _raw(CoordFormat.Atom37)
  raw["insertion_codes"] = list(raw["insertion_codes"])[:-1]
  with pytest.raises(ValueError, match="expected 8"):
    Protein.from_rust_dict(raw, use_jax=False)


@pytest.mark.parametrize("bad", [np.arange(53, dtype=np.int8) % 2, np.ones(53, dtype=np.float32)])
def test_hetero_non_bool_raises(bad: np.ndarray) -> None:
  raw = _raw(CoordFormat.Full)
  raw["hetero"] = bad
  with pytest.raises(ValueError, match="bool"):
    Protein.from_rust_dict(raw, use_jax=False)


def test_hetero_wrong_length_raises() -> None:
  raw = _raw(CoordFormat.Full)
  raw["hetero"] = np.asarray(raw["hetero"], dtype=np.bool_)[:-1]
  with pytest.raises(ValueError, match="expected|shape"):
    Protein.from_rust_dict(raw, use_jax=False)


def test_full_hetero_is_bool_from_rust_dict() -> None:
  p = Protein.from_rust_dict(_raw(CoordFormat.Full), use_jax=False)
  assert p.hetero.dtype == np.bool_


# --- propagation sites (D2 table) -------------------------------------------------------------


def test_truncate_crops_insertion_codes_like_residue_names(atom37: Protein) -> None:
  # center_crop of 8 residues to 4 starts at 2: residues 52A, 52B, 53, 54.
  out = truncate_protein(atom37, 4, "center_crop")
  assert out.insertion_codes == ["A", "B", " ", " "]
  assert out.insertion_codes == list(atom37.insertion_codes)[2:6]
  assert len(out.residue_names) == 4


def test_truncate_full_drops_codes_and_hetero_with_warning(full: Protein) -> None:
  with pytest.warns(UserWarning, match="insertion_codes, hetero"):
    out = truncate_protein(full, 10, "center_crop")
  assert out.insertion_codes is None
  assert out.hetero is None


def test_concat_keeps_residue_names_and_insertion_codes_together(atom37: Protein) -> None:
  # Both string fields set at once: this used to raise TypeError in the batch-axis tree_map
  # ('MET'[None, ...]). residue_names carries a real residue name, not an empty one.
  p = atom37.replace(residue_names=["MET", "ALA", "GLY", "SER", "THR", "ALA", "ALA", "GLY"])
  out = concatenate_proteins_for_inter_mode([p, p])
  assert out.coordinates.shape[0] == 1
  assert out.coordinates.shape[1] == 16
  assert out.insertion_codes == RESIDUE_CODES + RESIDUE_CODES
  assert out.residue_names == list(p.residue_names) * 2


def test_concat_keeps_a_chain_ids_vocabulary_that_chain_index_resolves(atom37: Protein) -> None:
  # The normal parsed-protein path: chain_ids is set, so concat must neither warn nor drop it,
  # and every residue's remapped chain_index must name its original chain.
  import warnings as _warnings

  # A second input with two chains whose vocabulary lists one name beyond its max index.
  n = atom37.coordinates.shape[0]
  second = atom37.replace(
    chain_index=np.array([0] * (n // 2) + [1] * (n - n // 2), dtype=np.int32),
    chain_ids=["X", "Y", "UNUSED"],
  )
  with _warnings.catch_warnings():
    _warnings.simplefilter("error")
    out = concatenate_proteins_for_inter_mode([atom37, second])
  expected = [atom37.chain_ids[i] for i in np.asarray(atom37.chain_index)] + [
    ["X", "Y"][i] for i in np.asarray(second.chain_index)
  ]
  resolved = [out.chain_ids[i] for i in np.asarray(out.chain_index)[0]]
  assert resolved == expected
  assert "UNUSED" not in out.chain_ids


def test_concat_chain_ids_missing_on_one_input_is_none(atom37: Protein) -> None:
  out = concatenate_proteins_for_inter_mode([atom37, atom37.replace(chain_ids=None)])
  assert out.chain_ids is None


def test_concat_full_drops_codes_and_hetero_with_warning(full: Protein) -> None:
  with pytest.warns(UserWarning, match="insertion_codes, hetero"):
    out = concatenate_proteins_for_inter_mode([full, full])
  assert out.insertion_codes is None
  assert out.hetero is None


def test_pad_appends_blank_codes(atom37: Protein) -> None:
  padded = _pad_protein(atom37, 10)
  assert padded.insertion_codes == RESIDUE_CODES + [" ", " "]


def test_pad_full_drops_codes_and_hetero_with_warning(full: Protein) -> None:
  with pytest.warns(UserWarning, match="insertion_codes, hetero"):
    padded = _pad_protein(full, full.coordinates.shape[0] + 3)
  assert padded.insertion_codes is None
  assert padded.hetero is None


def test_stack_keeps_one_list_per_row(atom37: Protein) -> None:
  rows = [_pad_protein(atom37, 10), _pad_protein(atom37.replace(insertion_codes=["C"] * 8), 10)]
  batched = _stack_padded_proteins(rows)
  assert batched.insertion_codes == [rows[0].insertion_codes, rows[1].insertion_codes]


def test_stack_mixed_none_and_set_rows_drops_to_none_with_warning(atom37: Protein) -> None:
  rows = [_pad_protein(atom37, 10), _pad_protein(atom37.replace(insertion_codes=None), 10)]
  with pytest.warns(UserWarning, match="insertion_codes is missing on 1 of 2"):
    batched = _stack_padded_proteins(rows)
  assert batched.insertion_codes is None


def test_stack_drops_hetero_with_warning(full: Protein) -> None:
  with pytest.warns(UserWarning, match="hetero"):
    batched = _stack_padded_proteins([full, full])
  assert batched.hetero is None


def test_chain_filter_indexes_codes_with_the_residue_mask(monkeypatch) -> None:
  # load_rust builds its own OutputSpec with no coord_format (so Full), and the per-residue
  # filter runs only on Atom37. Pin Atom37 here so the residue-mask path is the one exercised.
  backend = importlib.import_module("proxide.io.parsing.backend")
  real_spec = backend.OutputSpec
  monkeypatch.setattr(
    backend, "OutputSpec", lambda **kw: real_spec(coord_format=CoordFormat.Atom37, **kw)
  )
  # Chain A only: the chain B residues drop out, and the codes follow the same mask.
  p = next(iter(load_rust(FIXTURE, chain_id="A", return_type="Protein")))
  assert p.insertion_codes == CHAIN_A_RESIDUE_CODES
  assert len(p.insertion_codes) == len(p.coordinates) == 6


# --- writer: npz round trip and batched guard -------------------------------------------------


def test_npz_round_trip_of_both_fields(tmp_path: Path, full: Protein) -> None:
  path = write_npz(full, tmp_path / "full.npz")
  fields = load_npz_icode_fields(path)
  assert fields["insertion_codes"] == full.insertion_codes
  assert isinstance(fields["insertion_codes"], list)
  assert fields["hetero"].dtype == np.bool_
  np.testing.assert_array_equal(fields["hetero"], full.hetero)


def test_npz_round_trip_of_atom37_codes(tmp_path: Path, atom37: Protein) -> None:
  path = write_npz(atom37, tmp_path / "atom37.npz")
  fields = load_npz_icode_fields(path)
  assert fields == {"insertion_codes": RESIDUE_CODES}


def test_npz_without_fields_writes_no_icode_keys(tmp_path: Path, atom37: Protein) -> None:
  path = write_npz(atom37.replace(insertion_codes=None), tmp_path / "bare.npz")
  assert load_npz_icode_fields(path) == {}


def test_reject_if_batched_fires_on_nested_insertion_codes(atom37: Protein) -> None:
  nested = atom37.replace(insertion_codes=[[c] for c in RESIDUE_CODES])
  with pytest.raises(ValueError, match="insertion_codes"):
    _reject_if_batched(nested, "write_pdb")


def test_reject_if_batched_does_not_fire_on_flat_codes(atom37: Protein) -> None:
  _reject_if_batched(atom37, "write_pdb")  # must not raise


# --- other producers --------------------------------------------------------------------------


def test_parm7_yields_hetero_none(tmp_path: Path) -> None:
  # There is no parm7 fixture file. The topology writer lives in test_parm7_trajectory.py,
  # so it is loaded by path rather than copied; the trajectory is the same one that module uses.
  spec = importlib.util.spec_from_file_location(
    "_parm7_traj", Path(__file__).with_name("test_parm7_trajectory.py")
  )
  assert spec is not None and spec.loader is not None
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  parm7 = tmp_path / "dipeptide.parm7"
  mod._write_parm7(parm7)
  # The raw dict has no hetero key at all: parm7 has no record types to derive it from.
  raw = proxider.parse_amber_trajectory(str(parm7), str(mod.XTC), [0], OutputSpec())
  assert "hetero" not in raw
  p = parse_amber_trajectory(parm7, mod.XTC, frames=[0], use_jax=False)
  p = p[0] if isinstance(p, (list, tuple)) else p
  assert p.hetero is None


def test_pqr_raw_hetero_comes_from_record_types() -> None:
  # PQR has no Protein path (it returns an AtomicSystem), so the flag is checked on the raw
  # dict that _proxider.parse_pqr returns, the one the Rust formatter reads.
  raw = proxider.parse_pqr(str(PQR))
  assert raw["hetero"] is not None
  hetero = np.asarray(raw["hetero"], dtype=np.bool_)
  assert hetero.shape == (raw["num_atoms"],)
  record_types = [ln.startswith("HETATM") for ln in PQR.read_text().splitlines() if ln[:6] in ("ATOM  ", "HETATM")]
  np.testing.assert_array_equal(hetero, np.asarray(record_types, dtype=np.bool_))


def test_protein_fields_are_declared_with_defaults() -> None:
  names = {f.name: f for f in dataclasses.fields(Protein)}
  assert "insertion_codes" in names
  assert "hetero" in names


# --- layout detection (Cursor review of PR #103) --------------------------------------------------


def _ala_plus_ligand_pdb(path: Path, n_ligand_atoms: int) -> Path:
  def atom(rec: str, serial: int, name: str, res: str, resi: int, x: float, el: str) -> str:
    return (
      f"{rec:<6}{serial:>5} {name:<4} {res:>3} A{resi:>4}    "
      f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 10.00          {el:>2}\n"
    )

  lines = [atom("ATOM", i + 1, n, "ALA", 1, float(i), n[0]) for i, n in enumerate(["N", "CA", "C", "O", "CB"])]
  lines += [
    atom("HETATM", 6 + k, f"C{k + 1}", "LIG", 2, 10.0 + k, "C") for k in range(n_ligand_atoms)
  ]
  path.write_text("".join(lines) + "END\n")
  return path


@pytest.mark.parametrize("n_ligand_atoms", [9, 32])  # 14 and 37 atoms over one residue
def test_full_parse_is_never_read_as_atom14_or_atom37(tmp_path: Path, n_ligand_atoms: int) -> None:
  # A flat Full buffer whose atom count equals num_residues * 14 (or * 37) used to be reshaped
  # as Atom14 (Atom37); with per-atom insertion codes it then raised on their length.
  path = _ala_plus_ligand_pdb(tmp_path / "p.pdb", n_ligand_atoms)
  protein = parse_structure(str(path), OutputSpec(coord_format=CoordFormat.Full))
  n_atoms = 5 + n_ligand_atoms
  assert protein.atom_names is not None and len(protein.atom_names) == n_atoms
  assert np.asarray(protein.coordinates).shape == (n_atoms, 3)
  assert len(protein.insertion_codes) == n_atoms
  assert list(np.asarray(protein.hetero)) == [False] * 5 + [True] * n_ligand_atoms
