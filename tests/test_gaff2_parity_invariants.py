"""Orchestrator-owned invariant tests, Phases 4-5 of the GAFF2 bathos-literature-parity
campaign (see .praxia/docs/decisions/260818_gaff2-parity-verdict-policy.md and the
Phase 5 verdict report at .praxia/docs/audits/260820_gaff2-parity-verdict.md).

Per the campaign's Constraint 1 (orchestrator-owned re-derivation lock): Phase 3's
adversarial-refutation agents proposed two defects. These tests independently
re-derive both from the source, rather than trusting the agents' reports as-is.

Defect 1 (FIXED, follow-up #1): `_bond_category_facts` used to compute lowercase
`sb`/`db` (inclusive bond-category counts) identically to uppercase `SB`/`DB`
(exact bond-type identity), silently dropping the "includes aromatic
single/double" semantics the DEF footer specifies (ATOMTYPE_GFF2.DEF lines
411-412). Fixed by Kekulizing a local copy before rule matching
(`assign_gaff2_atom_types`) so `bt.name` reflects the true per-bond Kekule
identity even for ring bonds, then deriving SB/DB as "kekule-exact AND NOT
aromatic" and sb/db as "kekule-exact regardless of aromaticity". Locked in below
as a passing test, not xfail.

Defect 2 (FIXED, follow-up #2): `_H_TYPE_BY_HEAVY` (the heavy-atom-type -> H-type
lookup used by `build_gaff2_ffxml`) used to be missing most nitrogen ATD family
types (only 9 of the ~20+ real N types mapped to "hn"); an amide nitrogen typed
`nt`/`ns` silently fell through to the dict's default of "hc" (same type as a
plain alkane C-H) instead of "hn". Fixed via `_H_TYPE_ELEMENT_DEFAULT`: per
ATOMTYPE_GFF2.DEF lines 79-82, hn/ho/hs/hp are each unconditional on the
*specific* N/O/S/P sub-type ("(N)"/"(O)"/"(S)"/"(P)", no further constraint),
so any heavy type not in `_H_TYPE_BY_HEAVY` now falls back to an
element-derived default instead of a blanket "hc". Locked in below as a
passing test, not xfail.

Also includes a passing invariant confirming the sb/db fix does NOT change any
atom type in the current h_ew benchmark set (Phase 3 Attacker #1's core finding,
still true after the fix since cs/c are actually disambiguated by f9, not f8).
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


def test_assign_gaff2_atom_types_kekulizes_before_matching(monkeypatch) -> None:
    """Regression guard for the fix itself, not just its output.

    PR-audit finding: the test below this one (and the whole golden suite)
    Kekulizes its OWN molecule and calls _atom_bond_facts directly -- it
    never actually exercises assign_gaff2_atom_types's internal Kekulize
    call, and no current benchmark molecule's *output* depends on it (cs/c
    are disambiguated by f9, not f8; verified via monkeypatching Chem.Kekulize
    to a no-op module-wide and confirming every golden-suite molecule's
    output is unchanged). That means a future accidental removal of the
    internal `Chem.Kekulize(mol_for_matching, clearAromaticFlags=False)` call
    in `assign_gaff2_atom_types` would pass every other test in this repo
    silently. This test spies on the real call (still lets it run -- not a
    no-op mock) to guard the wiring directly, independent of whether any
    molecule's typed output happens to depend on it today.
    """
    import proxide.chem.gaff2 as gaff2_module

    calls: list[tuple[bool | None]] = []
    real_kekulize = gaff2_module.Chem.Kekulize

    def spy_kekulize(mol, clearAromaticFlags=True, **kwargs):
        calls.append((clearAromaticFlags,))
        return real_kekulize(mol, clearAromaticFlags=clearAromaticFlags, **kwargs)

    monkeypatch.setattr(gaff2_module.Chem, "Kekulize", spy_kekulize)

    mol = Chem.MolFromSmiles("Cc1ccccc1")  # toluene: has aromatic bonds
    mol = Chem.AddHs(mol)
    AllChem.SanitizeMol(mol)
    gaff2_module.assign_gaff2_atom_types(mol)

    assert calls == [(False,)], (
        f"expected assign_gaff2_atom_types to call Chem.Kekulize exactly "
        f"once with clearAromaticFlags=False before rule matching; got "
        f"{calls}"
    )


def test_bond_category_facts_sb_db_include_aromatic_kekule_identity() -> None:
    """Locks in follow-up #1's fix: sb/db now reflect the true per-bond Kekule
    single/double identity (regardless of aromaticity), while SB/DB stay exact
    (non-aromatic only). Toluene's ipso carbon has 3 heavy bonds: 2 ring bonds
    (one Kekule-single, one Kekule-double in RDKit's chosen Kekule structure)
    plus 1 explicit single bond to the methyl carbon -- so sb=2 (the explicit
    single + the one Kekule-single ring bond), db=1 (the one Kekule-double ring
    bond), while SB stays 1 (only the explicit non-aromatic single bond).
    """
    from proxide.chem.gaff2 import _atom_bond_facts

    mol = Chem.MolFromSmiles("Cc1ccccc1")  # toluene
    mol = Chem.AddHs(mol)
    AllChem.SanitizeMol(mol)
    Chem.Kekulize(mol, clearAromaticFlags=False)
    ipso = mol.GetAtomWithIdx(1)  # ring carbon bonded to the methyl group
    assert ipso.GetSymbol() == "C" and ipso.GetIsAromatic()

    facts = _atom_bond_facts(ipso)
    assert facts.bond_counts["SB"] == 1, facts.bond_counts
    assert facts.bond_counts["sb"] == 2, facts.bond_counts
    assert facts.bond_counts["db"] == 1, facts.bond_counts


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


def test_h_type_by_heavy_amide_n_h_types_as_hn() -> None:
    """Exercises the real, fixed end-to-end path (build_gaff2_ffxml), not just
    the lookup table in isolation -- confirms the _H_TYPE_ELEMENT_DEFAULT
    fallback actually reaches the emitted FFXML for both formamide and
    N-methylacetamide's amide N-H.
    """
    import re

    from proxide.chem.gaff2 import build_gaff2_ffxml

    for smiles, n_idx in [("NC=O", 0), ("CC(=O)NC", 3)]:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.SanitizeMol(mol)
        n_atom = mol.GetAtomWithIdx(n_idx)
        assert n_atom.GetSymbol() == "N"
        h_idx = next(
            nb.GetIdx() for nb in n_atom.GetNeighbors() if nb.GetAtomicNum() == 1
        )

        charges = [0.0] * mol.GetNumAtoms()
        ffxml = build_gaff2_ffxml(mol, resname="LIG", charges=charges)

        type_name = f"LIG_{h_idx}"
        m = re.search(rf'<Type name="{type_name}" class="([^"]+)"', ffxml)
        assert m is not None, f"{smiles}: no AtomTypes entry for {type_name}"
        assert m.group(1) == "hn", (
            f"{smiles}: amide N-H (atom {h_idx}) resolved to class "
            f"{m.group(1)!r}, expected 'hn'"
        )
