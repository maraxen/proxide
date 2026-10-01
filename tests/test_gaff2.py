"""Test dataset for GAFF2 parameterization.

Validation strategy:
1. Masses/bonds/angles - compare against parsed dat file entries
2. Charges - verify charge conservation (sum ≈ 0)
3. Parameter lookups - verify substitution logic works for torsions

Reference values from gaff-2.2.20.dat. Atom-type-assignment correctness is
covered by tests/test_gaff2_golden.py (real `assert`-based tests against the
ATOMTYPE_GFF2.DEF rule table) -- this file's own former ATOM_TYPE_TESTS/
test_atom_types() were removed because (a) their expected values were wrong
relative to the actual DEF file (e.g. methane -> "cx", which requires a
3-membered ring methane doesn't have; ethanol O -> "op", a 3-membered-ring
oxygen type, not "oh") and (b) the test itself used `return failed == 0`
instead of `assert`, which pytest does not treat as a failure -- it could
never actually fail regardless of content.

debt #1900 (sprint-22 Track C): the remaining tests below had the exact same
`return failed == 0` bug -- fixed here to use real `assert`s so pytest
actually fails them on a regression. `parameterize_gaff_with_rdkit` needs the
ATOMTYPE_GFF2.DEF rule table (via assign_gaff2_atom_types) for every test in
this file; per this repo's rules that file is never fetched or stubbed by an
agent, so locally (without a fetched DEF) every test here raises
Gaff2DefMissingError -- expected, not a regression. CI's `tests` job fetches
the DEF before running (see .github/workflows/ci.yml).
"""

import pytest
import numpy as np

Chem = pytest.importorskip("rdkit.Chem")
from rdkit.Chem import AllChem

from proxide.chem.gaff2 import (
    parameterize_gaff_with_rdkit,
    load_gaff2_parameters,
)
from proxide.chem.partial_charges import CHARGE_SOURCE_ESPALOMA_AM1BCC


# Test molecules for parameter validation
PARAM_TESTS = [
    "C",  # methane
    "CC",  # ethane
    "CCC",  # propane
    "CCO",  # ethanol
    "CCCO",  # propanol
    "c1ccccc1",  # benzene
    "c1ccc(O)cc1",  # phenol
    "c1ccncc1",  # pyridine
    "CC(=O)C",  # acetone
    "CC(=O)O",  # acetic acid
]


def prepare_mol(smiles: str) -> Chem.Mol:
    """Prepare RDKit molecule with hydrogens."""
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.SanitizeMol(mol)
    return mol


def test_charge_conservation():
    """Charges sum to approximately zero for these (net-neutral) test molecules.

    charge_method is passed explicitly (rather than relying on the default)
    since this test cares about the numeric result, not about
    parameterize_gaff_with_rdkit's default-argument behavior.

    Note: under the pre-debt-#1900 silent-fallback implementation, this test
    would spuriously pass even when charge assignment failed entirely, because
    a failure there returned [0.0] * n_atoms -- whose sum is trivially 0.0,
    comfortably inside the tolerance below. The not-all-zero assertion is what
    catches that failure mode now.
    """
    for smiles in PARAM_TESTS:
        mol = prepare_mol(smiles)
        result = parameterize_gaff_with_rdkit(mol, charge_method="espaloma")

        assert result["charge_method"] == CHARGE_SOURCE_ESPALOMA_AM1BCC

        charges = result["charges"]
        assert not all(q == 0.0 for q in charges), (
            f"{smiles}: all-zero charges (this is the exact silent-fallback "
            "failure mode debt #1900 removed)"
        )

        charge_sum = sum(charges)
        assert abs(charge_sum) < 1e-5, f"{smiles}: charge sum {charge_sum:.2e} not ~0"


def test_parameter_lookups():
    """Parameters are looked up correctly for known types/pairs/triples/quads."""
    params = load_gaff2_parameters()

    # Mass lookups
    for atom_type in ["c3", "cp", "cx", "oh", "op", "n3", "ni", "hc"]:
        assert atom_type in params["masses"], f"missing mass lookup for {atom_type}"

    # Bond lookups
    for pair in [("c3", "c3"), ("cx", "op"), ("c3", "oh"), ("c3", "hc")]:
        key = tuple(sorted(pair))
        assert key in params["bonds"], f"missing bond lookup for {pair}"

    # Angle lookups
    for triple in [("c3", "c3", "c3"), ("c3", "c3", "oh"), ("c3", "c3", "hc")]:
        assert triple in params["angles"], f"missing angle lookup for {triple}"

    # Torsion lookups with our substitution logic
    # We substitute cx -> c3, x -> hc when looking up
    type_sub = lambda t: "c3" if t == "cx" else ("hc" if t == "x" else t)  # noqa: E731

    test_torsions = [
        (("cx", "cx", "cx", "x"), type_sub),  # propane H torsions become c3-c3-c3-hc
        (("c3", "c3", "c3", "c3"), lambda t: t),  # direct lookup
    ]

    for quad, subst in test_torsions:
        key = tuple(subst(t) for t in quad)
        assert key in params["torsions"] and params["torsions"][key], (
            f"missing torsion lookup for {quad}"
        )


def test_dat_loader_reads_amber_parm_dat_format():
    """Debt #2368: load_gaff2_parameters reads gaff-2.2.20.dat as AMBER parm.dat.

    Expected values are the same rows as read by ParmEd 4.3.1's
    AmberParameterSet on the identical file (the full-table comparison is the
    bathos-tracked scripts/validation/gaff2_dat_parmed_parity.py). Every
    assertion here failed on the previous whitespace-splitting loader:
    padded single-letter types ("c3-c -c3", "X -c -c -X") were misread as
    bonds and dropped; DIHE's IDIVF column was read as the periodicity, PK was
    never divided by IDIVF, and every IMPROPER row was dropped.
    """
    p = load_gaff2_parameters()
    # Angles with a padded middle type.
    assert p["angles"][("c3", "c", "c3")] == pytest.approx((59.15, 116.68))
    assert p["angles"][("c3", "c", "o")] == pytest.approx((76.45, 122.90))
    assert p["angles"][("ca", "ca", "nb")] == pytest.approx((68.17, 122.94))
    # Generic torsion: IDIVF=4, PK=1.2 -> barrier 0.3; periodicity from PN=2.
    assert p["torsions"][("X", "c", "c", "X")] == [pytest.approx((2, 0.3, 180.0))]
    # Specific torsion: IDIVF=1, PN=3 -> periodicity 3 (was read as 1).
    assert p["torsions"][("c3", "c3", "c3", "c3")] == [pytest.approx((3, 0.52, 0.0))]
    # Multi-term: a negative PN means another term for the same quartet follows.
    assert p["torsions"][("X", "c", "na", "X")] == [
        pytest.approx((2, 1.45, 180.0)), pytest.approx((4, 0.35, 180.0)),
    ]
    # Impropers (no IDIVF column): used to be dropped entirely.
    assert p["impropers"][("X", "X", "c", "o")] == pytest.approx((10.5, 180.0))
    # Table sizes: unique angle and torsion keys equal ParmEd's.
    assert len(p["angles"]) == 9712
    assert len(p["torsions"]) == 1341
    assert len(p["impropers"]) == 38  # 35 in ParmEd's permutation-canonical form
    assert len(p["vdw"]) == 97


def test_dat_loader_raises_on_a_malformed_row(tmp_path):
    """A row that does not fit its section raises instead of being skipped."""
    from pathlib import Path

    from proxide.chem import gaff2

    src = Path(gaff2.__file__).parent.parent / "assets" / "gaff" / "dat" / "gaff-2.2.20.dat"
    lines = src.read_text().split("\n")
    target = next(i for i, ln in enumerate(lines) if ln.startswith("c3-c -c3"))
    lines[target] = "c3-c -c3   not-a-number"
    bad = tmp_path / "bad.dat"
    bad.write_text("\n".join(lines))
    with pytest.raises(ValueError, match=r"bad\.dat:\d+: malformed ANGLE record"):
        load_gaff2_parameters(bad)


def test_parameter_values():
    """Parameter values match expected values from the dat file."""
    params = load_gaff2_parameters()

    # Known values from gaff-2.2.20.dat (what we actually parse).
    #
    # FIX (debt #1900, converting this file's return-failed==0 tests to real
    # asserts): the previous reference dict held VDW rmin_half values
    # (1.9069, 1.8606, 1.82, 1.7713 -- these are exactly params["vdw"]["c3"],
    # ["cp"], ["oh"], ["op"][0], measured 2026-09-23) under a dict literally
    # named `known_masses`, checked against `params["masses"]`. That
    # mismatch -- real atomic masses are 12.01 (c3, cp) / 16.0 (oh, op) amu,
    # measured against this repo's own parsed gaff-2.2.20.dat -- was
    # invisible under the old `return failed == 0` contract (pytest never
    # treats a returned False as a failure), so it silently "passed" for as
    # long as that bug existed. Corrected here to the real atomic masses;
    # this is a test-fixture correctness fix, not a re-fit to whatever the
    # code currently outputs -- amu values for C and O are settled physical
    # constants, independent of anything Track C touched.
    known_masses = {
        "c3": (12.01, 0.001),
        "cp": (12.01, 0.001),
        "oh": (16.0, 0.001),
        "op": (16.0, 0.001),
    }

    for atom_type, (expected, tolerance) in known_masses.items():
        assert atom_type in params["masses"], f"missing mass lookup for {atom_type}"
        actual = params["masses"][atom_type]
        assert abs(actual - expected) <= tolerance, (
            f"{atom_type} mass = {actual}, expected {expected}"
        )

    # Known bond values from what we actually load.
    #
    # FIX (same pass as above): the key was ("c3", "op"), which does not
    # exist in params["bonds"] (measured: params["bonds"][("c3","op")] is
    # absent; verified via `sorted(("c3","op"))`). The value pair
    # (273.64, 1.4368) is real and correct -- it's exactly
    # params["bonds"][("cx","op")], the same pair test_parameter_lookups
    # above already exercises (presence-only, no value check) -- so this was
    # a copy-paste key typo (c3 vs cx), not a wrong value. Same
    # invisible-under-return-False history as the masses fix.
    known_bonds = {
        ("cx", "op"): (273.64, 1.4368),  # alcohol O-H
    }

    for pair, (expected_kb, expected_r0) in known_bonds.items():
        key = tuple(sorted(pair))
        assert key in params["bonds"], f"missing bond lookup for {pair}"
        actual_kb, actual_r0 = params["bonds"][key]
        assert abs(actual_kb - expected_kb) <= 0.1 and abs(actual_r0 - expected_r0) <= 0.001, (
            f"{pair} bond = ({actual_kb}, {actual_r0}), expected ({expected_kb}, {expected_r0})"
        )

    # Verify we have substantial data loaded
    assert len(params["masses"]) >= 90, f"only {len(params['masses'])} masses loaded"
    assert len(params["bonds"]) >= 1000, f"only {len(params['bonds'])} bonds loaded"
    assert len(params["angles"]) >= 7000, f"only {len(params['angles'])} angles loaded"


def test_full_parameterization():
    """Full parameterization produces valid output, including charges/charge_method."""
    for smiles in PARAM_TESTS:
        mol = prepare_mol(smiles)
        result = parameterize_gaff_with_rdkit(mol, charge_method="espaloma")

        # Required fields -- charges and charge_method must always be present
        # (debt #1900: never silently omitted or left to a caller to infer).
        for field in (
            "atom_types",
            "charges",
            "charge_method",
            "masses",
            "bonds",
            "angles",
            "torsions",
        ):
            assert field in result, f"{smiles}: missing {field}"

        assert result["charge_method"] == CHARGE_SOURCE_ESPALOMA_AM1BCC, (
            f"{smiles}: charge_method = {result['charge_method']!r}, "
            f"expected {CHARGE_SOURCE_ESPALOMA_AM1BCC!r}"
        )
        assert not all(q == 0.0 for q in result["charges"]), (
            f"{smiles}: all-zero charges"
        )

        # Atom types length matches heavy atoms
        heavy_atoms = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() != 1)
        assert len(result["atom_types"]) >= heavy_atoms, (
            f"{smiles}: atom_types length mismatch: "
            f"{len(result['atom_types'])} < {heavy_atoms}"
        )


# Debt #1905: Observability of silent parameter fills


def test_acetone_central_angle_has_its_dat_parameters():
    """Acetone's c3-c-c3 angle carries gaff-2.2.20.dat's 59.15 kcal/mol/rad^2,
    116.68 deg (debt #2368).

    The old loader dropped this row ("c3-c -c3": padded single-letter type),
    so the angle was filled with 0.0 on every call -- this test's previous
    version asserted exactly that, as if it were correct.
    """
    result = parameterize_gaff_with_rdkit(prepare_mol("CC(=O)C"))
    central = [a for a in result["angles"] if sorted((a["types"][0], a["types"][2])) == ["c3", "c3"]
               and a["types"][1] == "c"]
    assert len(central) == 1, [a["types"] for a in result["angles"]]
    assert central[0]["kt"] == pytest.approx(59.15)
    assert central[0]["t0"] == pytest.approx(116.68)


def test_a_term_missing_from_the_table_raises(monkeypatch):
    """No term is ever filled with 0.0: a missing one raises, naming it (debt #1905)."""
    from proxide.chem import gaff2

    real = gaff2.load_gaff2_parameters()
    for key in [("c3", "c", "c3")]:
        real["angles"].pop(key, None)
        real["angles"].pop(key[::-1], None)
    monkeypatch.setattr(gaff2, "load_gaff2_parameters", lambda *a, **k: real)
    with pytest.raises(gaff2.Gaff2ParameterMissingError, match="angle c3-c-c3"):
        gaff2.parameterize_gaff_with_rdkit(prepare_mol("CC(=O)C"))


def test_substitutions_tracking():
    """substitutions tracks torsion type substitutions applied (debt #1905)."""
    # Any molecule where a torsion substitution actually applies
    # In the golden tests, no substitutions happen (all types are available)
    # But we can verify the key exists and is a list
    for smiles in PARAM_TESTS:
        mol = prepare_mol(smiles)
        result = parameterize_gaff_with_rdkit(mol)

        # substitutions should be present
        assert "substitutions" in result, f"{smiles}: missing substitutions key"
        assert isinstance(result["substitutions"], list), (
            f"{smiles}: substitutions is not a list"
        )

        # Each entry should have term="torsion", "from", and "to" keys
        for subst in result["substitutions"]:
            assert subst.get("term") == "torsion", (
                f"{smiles}: substitution term is not 'torsion': {subst}"
            )
            assert "from" in subst and isinstance(subst["from"], list), (
                f"{smiles}: substitution missing 'from' key: {subst}"
            )
            assert "to" in subst and isinstance(subst["to"], list), (
                f"{smiles}: substitution missing 'to' key: {subst}"
            )
            assert len(subst["from"]) == 4, (
                f"{smiles}: substitution 'from' should be 4-element torsion: {subst}"
            )
            assert len(subst["to"]) == 4, (
                f"{smiles}: substitution 'to' should be 4-element torsion: {subst}"
            )


def test_substitutions_key_always_present():
    """The substitutions record is always present, even when empty (debt #1905)."""
    for smiles in PARAM_TESTS:
        result = parameterize_gaff_with_rdkit(prepare_mol(smiles))
        assert isinstance(result.get("substitutions"), list), smiles
        assert "missing_params" not in result, "missing terms raise; there is no fill to record"