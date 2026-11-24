"""Rust producers of insertion codes and the per-atom hetero flag (spec 261123, T2).

These cover the two Rust paths that are only reachable from Python:

- the Atom37 multi-model stacker in ``py_parsers.rs``: insertion codes are per
  residue and shared across models, so differing codes must raise;
- the parse cache: a cache hit must return the same ``insertion_codes`` and
  ``hetero`` as the cold parse that stored it.

The per-formatter expectations on ``tests/data/icode_ligand.pdb`` are Rust unit
tests (``crates/proxide-io/src/formatters``). The fixture is read in the test,
never hard-coded, except for the literal per-residue codes the spec names.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import proxide._proxider as proxider
from proxide import CoordFormat, OutputSpec

FIXTURE = Path(__file__).resolve().parents[1] / "data" / "icode_ligand.pdb"

# Protein residues of the fixture in residue order (spec T1): 51, 52, 52A, 52B,
# MSE 53, 54, then chain B 1 and 2.
EXPECTED_RESIDUE_CODES = [" ", " ", "A", "B", " ", " ", " ", " "]


def _fixture_lines() -> list[str]:
    return FIXTURE.read_text().splitlines()


def _atom_lines(lines: list[str]) -> list[str]:
    return [line for line in lines if line.startswith(("ATOM", "HETATM"))]


def _two_model_file(tmp_path: Path, name: str, second_model_edit=None) -> Path:
    """Write a two-model PDB: the fixture as model 1 and as model 2.

    ``second_model_edit`` maps each atom line of model 2 to its replacement.
    """
    atoms = _atom_lines(_fixture_lines())
    second = [second_model_edit(line) for line in atoms] if second_model_edit else atoms
    body = ["MODEL        1", *atoms, "ENDMDL", "MODEL        2", *second, "ENDMDL", "END"]
    path = tmp_path / name
    path.write_text("\n".join(body) + "\n")
    return path


def _set_icode(line: str, residue: tuple[str, str, str], icode: str) -> str:
    """Return ``line`` with column 27 set to ``icode`` if it belongs to ``residue``.

    ``residue`` is (chain, resSeq text, resName), read from the fixed columns.
    """
    chain, res_seq, res_name = residue
    if line[21] == chain and line[22:26].strip() == res_seq and line[17:20].strip() == res_name:
        return line[:26] + icode + line[27:]
    return line


def test_atom37_multimodel_equal_codes_stack(tmp_path: Path) -> None:
    path = _two_model_file(tmp_path, "icode_two_models_equal.pdb")

    spec = OutputSpec(coord_format=CoordFormat.Atom37)
    d = proxider.parse_structure(str(path), spec, "auth")

    assert d["n_models"] == 2
    assert d["coordinates"].shape[0] == 2
    assert list(d["insertion_codes"]) == EXPECTED_RESIDUE_CODES


def test_atom37_multimodel_differing_codes_raise(tmp_path: Path) -> None:
    # Model 2 moves SER 52 from insertion code A to C: same residue numbers,
    # different codes, so the stacked structure would be ill-defined.
    def edit(line: str) -> str:
        return _set_icode(line, ("A", "52", "SER"), "C")

    path = _two_model_file(tmp_path, "icode_two_models_differ.pdb", second_model_edit=edit)

    spec = OutputSpec(coord_format=CoordFormat.Atom37)
    with pytest.raises(ValueError, match="insertion codes differ"):
        proxider.parse_structure(str(path), spec, "auth")


@pytest.mark.parametrize("layout", ["atom37", "full"])
def test_cache_hit_returns_the_same_codes_and_hetero_as_cold_parse(
    tmp_path: Path, layout: str
) -> None:
    fmt = CoordFormat.Atom37 if layout == "atom37" else CoordFormat.Full
    # A unique copy of the fixture: the cache key is the path plus the file bytes,
    # so this first call is a miss and the second is a hit.
    path = tmp_path / f"icode_cache_{layout}.pdb"
    path.write_text(FIXTURE.read_text())
    spec = OutputSpec(coord_format=fmt, enable_caching=True)

    hits_before, misses_before, _ = proxider._format_cache_stats()
    cold = proxider.parse_structure(str(path), spec, "auth")
    hits_mid, misses_mid, _ = proxider._format_cache_stats()
    warm = proxider.parse_structure(str(path), spec, "auth")
    hits_after, _, _ = proxider._format_cache_stats()

    # The format-cache counters are process-global: another thread or a test sharing this
    # worker process can bump them between our snapshots, so exact deltas race under xdist.
    # The path is unique to this test, so our own calls are the only ones that can
    # produce these events: the cold call must add at least one miss and the warm call at
    # least one hit. The equality checks below carry the correctness claim.
    assert misses_mid >= misses_before + 1, "first call should miss"
    assert hits_after >= hits_mid + 1, "second call should hit"
    assert hits_mid >= hits_before

    assert list(cold["insertion_codes"]) == list(warm["insertion_codes"])
    if fmt is CoordFormat.Atom37:
        assert list(cold["insertion_codes"]) == EXPECTED_RESIDUE_CODES
    else:
        assert cold["hetero"] is not None
        assert list(cold["hetero"]) == list(warm["hetero"])
        assert len(cold["hetero"]) == len(warm["insertion_codes"])
        assert sum(bool(h) for h in warm["hetero"]) == sum(
            line.startswith("HETATM") for line in _atom_lines(_fixture_lines())
        )
