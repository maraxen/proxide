"""Orchestrator-owned invariant tests, Phase 4 of the GAFF2 bathos-literature-parity
campaign (see .praxia/docs/decisions/260818_gaff2-parity-verdict-policy.md and the
Phase 5 verdict report at .praxia/docs/audits/260820_gaff2-parity-verdict.md).

Per the campaign's Constraint 1 (orchestrator-owned re-derivation lock): Phase 3's
adversarial-refutation agents proposed two defects. These tests independently
re-derive both from the source, rather than trusting the agents' reports as-is.

Confirmed defects (xfail, documenting real bugs pending a follow-up fix -- NOT
accepted deviations like the naphthalene-bridgehead/AR1-AR3 tests in
test_gaff2_golden.py, which lock in a deliberate reading rather than a known bug):

1. `_bond_category_facts` computes lowercase `sb`/`db` (inclusive bond-category
   counts) identically to uppercase `SB`/`DB` (exact bond-type identity), silently
   dropping the "includes aromatic single/double" semantics the DEF footer
   specifies (ATOMTYPE_GFF2.DEF lines 411-412) and that the module's own docstring
   (gaff2.py lines 42-48) claims is implemented.
2. `_H_TYPE_BY_HEAVY` (the heavy-atom-type -> H-type lookup used by
   `build_gaff2_ffxml`) is missing most nitrogen ATD family types (only 9 of the
   ~20+ real N types map to "hn"); an amide nitrogen typed `nt`/`ns` silently
   falls through to the dict's default of "hc" (same type as a plain alkane C-H)
   instead of "hn". This is part of the already-deferred H-atom-typing rework
   (plan section B) but is now a concrete, reproducible instance of it.

Also includes a passing invariant confirming defect 1 does NOT corrupt any atom
type in the current benchmark set (Phase 3 Attacker #1's core finding).
"""

import pytest

Chem = pytest.importorskip("rdkit.Chem")
from rdkit.Chem import AllChem


def _types(smiles: str) -> list[tuple[str, str]]:
    """Return [(element, gaff2_type), ...] for heavy atoms in SMILES order."""
    from proxide.chem.gaff2 import assign_gaff2_atom_types

    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Failed to parse SMILES: {smiles}"
    mol = Chem.AddHs(mol)
    AllChem.SanitizeMol(mol)
    types = assign_gaff2_atom_types(mol)
    return [
        (atom.GetSymbol(), t)
        for atom, t in zip(mol.GetAtoms(), types)
        if atom.GetAtomicNum() != 1
    ]


@pytest.mark.xfail(
    reason=(
        "Confirmed defect (Phase 3 Attacker #1, re-derived here): "
        "_bond_category_facts's lowercase sb/db counts are computed identically "
        "to uppercase SB/DB, omitting the DEF footer's 'includes aromatic "
        "single/double' inclusive semantics. Toluene's ipso carbon has 3 heavy "
        "bonds (2 aromatic ring bonds + 1 single bond to the methyl carbon) and "
        "should satisfy a '3sb' (3-inclusive-single-bond) requirement, but the "
        "current tally only counts the literal RDKit SINGLE bond, giving sb=1."
    ),
    strict=True,
)
def test_bond_category_facts_sb_includes_aromatic_confirmed_defect() -> None:
    from proxide.chem.gaff2 import _atom_bond_facts, atomic_prop_matches, parse_atomic_prop

    mol = Chem.MolFromSmiles("Cc1ccccc1")  # toluene
    mol = Chem.AddHs(mol)
    AllChem.SanitizeMol(mol)
    ipso = mol.GetAtomWithIdx(1)  # ring carbon bonded to the methyl group
    assert ipso.GetSymbol() == "C" and ipso.GetIsAromatic()

    facts = _atom_bond_facts(ipso)
    assert facts.bond_counts["sb"] == 3, (
        f"expected ipso carbon's inclusive single-bond count (2 aromatic ring "
        f"bonds + 1 single C-CH3 bond) to be 3 per the DEF footer's 'sb includes "
        f"aromatic single' rule; got {facts.bond_counts}"
    )
    assert atomic_prop_matches(parse_atomic_prop("3sb"), facts)


def test_f8_bond_count_disambiguation_no_regression_on_h_ew_benchmark_molecules() -> None:
    """Locks in Phase 3 Attacker #1's finding: despite the sb/db defect above,
    the f8 bond-count disambiguation between carbonyl-type rules (`c` vs `cs`)
    is correct for every h_ew-targeting benchmark molecule, because none of
    their carbonyl carbons have aromatic bonds and cs/c are actually
    disambiguated by f9 (S vs O/S neighbor), not by which f8 branch fires.
    """
    for smiles, carbonyl_idx in [
        ("NC=O", 1),  # formamide
        ("CC(=O)NC", 1),  # N-methylacetamide
        ("CC(=O)C", 1),  # acetone
        ("CC(=O)[O-]", 1),  # acetate anion
    ]:
        heavy = _types(smiles)
        assert heavy[carbonyl_idx] == ("C", "c"), (
            f"{smiles}: expected carbonyl carbon (idx {carbonyl_idx}) to be 'c', "
            f"got {heavy}"
        )


@pytest.mark.xfail(
    reason=(
        "Confirmed defect (Phase 3 Attacker #3, re-derived here): "
        "_H_TYPE_BY_HEAVY only maps 9 nitrogen ATD types (n, n2, n3, na, nh, nb, "
        "nc, nd, n+) to 'hn'. Amide nitrogens typed 'nt'/'ns' are not in the "
        "dict, so build_gaff2_ffxml's H-typing loop silently falls through to "
        "the default 'hc' (same as a plain alkane C-H) instead of 'hn'. Part of "
        "the already-deferred H-atom-typing rework (plan section B); tracked "
        "here as a concrete, reproducible instance pending that follow-up."
    ),
    strict=True,
)
def test_h_type_by_heavy_missing_amide_n_types_confirmed_defect() -> None:
    from proxide.chem.gaff2 import _H_TYPE_BY_HEAVY, assign_gaff2_atom_types

    for smiles, n_idx in [("NC=O", 0), ("CC(=O)NC", 3)]:
        mol = Chem.MolFromSmiles(smiles)
        mol_heavy = Chem.AddHs(mol)
        AllChem.SanitizeMol(mol_heavy)
        mol_no_h = Chem.RemoveHs(mol_heavy)
        heavy_types = assign_gaff2_atom_types(mol_no_h)
        n_type = heavy_types[n_idx]
        assert mol_no_h.GetAtomWithIdx(n_idx).GetSymbol() == "N"

        h_type = _H_TYPE_BY_HEAVY.get(n_type, "hc")
        assert h_type == "hn", (
            f"{smiles}: amide nitrogen resolved to GAFF2 type {n_type!r}, whose "
            f"H should be 'hn' but _H_TYPE_BY_HEAVY has no entry for it -> "
            f"falls back to {h_type!r}"
        )
