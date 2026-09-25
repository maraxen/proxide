"""Real-file atom-count regression tests (sprint 25, task 260922_autonomous-loop,
track a, decision i, debt #1919/#1920).

Cross-checks the Python-facing parsers against the SAME independent atom
counts established in Rust by `crates/proxide-io/tests/pqr_cif_snapshot.rs`
(Step 0): a real wwPDB file's reported `num_atoms` must equal an independent,
from-scratch count of its ATOM/HETATM rows -- for both the three new real
RCSB fixtures (1CRN.cif, 1CRN.pdb, 1UBQ.cif) and the existing 1a00.pqr.

Deviation from the fixer prompt's literal wording, reported per task
instructions: the prompt says "calling proxide's parse_input", but
`proxide.io.parsing.dispatch.parse_input` routes .cif/.pdb through
`backend.load_rust` -> `_proxider.parse_structure`, which applies
`add_hydrogens=True` and `remove_solvent=True` by default and reduces the
result to a residue-level atom37 `Protein` -- its atom count is structurally
NOT comparable to a raw ATOM/HETATM row count (hydrogens are added, HOH
removed, atoms bucketed into a fixed 37-slot layout). `test_rust_integration.py`
already establishes the precedent for this exact kind of raw-count assertion
via `proxide.io.parsing.backend.parse_pdb_raw_rust`/`parse_mmcif_rust` (the
same low-level RawAtomData wrappers around `_proxider.parse_pdb`/`parse_mmcif`
used by the Rust-side snapshot/independent-count tests), so this file uses
those instead, with `altloc="all"` to disable altloc deduplication (none of
these fixtures have altlocs, but "all" is the literal-count-preserving mode
by construction, matching the Rust side exactly). For 1a00.pqr,
`proxide.io.parsing.pqr.parse_pqr_rust` already returns the raw, undeduplicated
per-atom dict via `parse_input`'s own `pqr` dispatch path, so no such
substitution is needed there.
"""

import pathlib

import pytest

from proxide.io.parsing.backend import parse_mmcif_rust, parse_pdb_raw_rust
from proxide.io.parsing.pqr import parse_pqr_rust

REPO_ROOT = pathlib.Path(__file__).parents[3]
PROXIDE_IO_DATA = REPO_ROOT / "crates" / "proxide-io" / "tests" / "data"
PQR_1A00 = REPO_ROOT / "tests" / "data" / "1a00.pqr"

# Independent counts established in crates/proxide-io/tests/pqr_cif_snapshot.rs
# Step 0 (a from-scratch line/row count, NOT reusing any parser's tokenizer):
# independent_count_matches_1crn_cif / _1crn_pdb / _1ubq_cif, all passing
# against the (at Step 0, still unmodified) Rust parsers.
EXPECTED_NUM_ATOMS = {
    "1CRN.cif": 327,
    "1CRN.pdb": 327,
    "1UBQ.cif": 660,
}

pytestmark = pytest.mark.skipif(
    not PROXIDE_IO_DATA.is_dir(),
    reason=f"real fixture directory not found: {PROXIDE_IO_DATA}",
)


def test_1crn_cif_num_atoms_matches_independent_count():
    path = PROXIDE_IO_DATA / "1CRN.cif"
    raw = parse_mmcif_rust(path, altloc="all")
    assert raw.num_atoms == EXPECTED_NUM_ATOMS["1CRN.cif"]
    assert len(raw.atom_names) == EXPECTED_NUM_ATOMS["1CRN.cif"]


def test_1crn_pdb_num_atoms_matches_independent_count():
    path = PROXIDE_IO_DATA / "1CRN.pdb"
    raw = parse_pdb_raw_rust(path, altloc="all")
    assert raw.num_atoms == EXPECTED_NUM_ATOMS["1CRN.pdb"]
    assert len(raw.atom_names) == EXPECTED_NUM_ATOMS["1CRN.pdb"]


def test_1ubq_cif_num_atoms_matches_independent_count():
    path = PROXIDE_IO_DATA / "1UBQ.cif"
    raw = parse_mmcif_rust(path, altloc="all")
    assert raw.num_atoms == EXPECTED_NUM_ATOMS["1UBQ.cif"]
    assert len(raw.atom_names) == EXPECTED_NUM_ATOMS["1UBQ.cif"]


def test_1crn_cif_and_pdb_atom_count_parity():
    cif_raw = parse_mmcif_rust(PROXIDE_IO_DATA / "1CRN.cif", altloc="all")
    pdb_raw = parse_pdb_raw_rust(PROXIDE_IO_DATA / "1CRN.pdb", altloc="all")
    assert cif_raw.num_atoms == pdb_raw.num_atoms
    assert list(cif_raw.atom_names) == list(pdb_raw.atom_names)


@pytest.mark.skipif(not PQR_1A00.exists(), reason=f"PQR fixture not found: {PQR_1A00}")
def test_1a00_pqr_num_atoms_matches_independent_count():
    # Independent count: every ATOM/HETATM line in 1a00.pqr (8771 lines, all
    # well-formed 11-token lines -- see crates/proxide-io/src/formats/pqr.rs's
    # fail-loud rewrite, which leaves this fixture's parse byte-identical).
    data = parse_pqr_rust(PQR_1A00)
    assert data["num_atoms"] == 8771
    assert len(data["atom_names"]) == 8771
    assert len(data["charges"]) == 8771
    assert len(data["radii"]) == 8771
