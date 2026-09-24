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