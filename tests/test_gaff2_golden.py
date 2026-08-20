"""Golden examples for GAFF2 parameterization.

These values are reference standards from established sources:
1. GAFF2 specification (atom types from ATOMTYPE_GFF2.DEF)
2. gaff-2.2.20.dat parameter file
3. Known AM1-BCC charge values for common molecules
4. Published benchmark values from EspalomaCharge paper

These can be used to validate any GAFF2 implementation.
"""

import pytest

Chem = pytest.importorskip("rdkit.Chem")
from rdkit.Chem import AllChem

# Test molecules with expected atom types (from GAFF2 ATOMTYPE_GFF2.DEF)
# Per ATOMTYPE_GFF2.DEF (line numbers are from vendored asset):
# - cx (line 16) requires [RG3] (3-membered ring membership)
# - c3 (line 22) is for "other sp3 C" (non-ring)
# - cs (line 24-26) is for C=S (not C=C or C=O)
# - c (line 28-30) is for C=O (carbonyl carbon)
# - ca (line 35) is "pure aromatic atom" [AR1] -- no neighbor-count constraint
# - cp (line 33) is aromatic AND requires 3 aromatic (XX[AR1]) neighbors -- i.e.
#   a fused-ring junction (biphenyl connector, naphthalene bridgehead), NOT an
#   ordinary monosubstituted aromatic ring carbon (which has only 2 aromatic
#   heavy neighbors + one non-aromatic substituent or H)
# - cg (line 72-75) is "sp C of conjugated systems" (per the DEF file's own
#   comment) -- it requires a *single-bonded* neighbor that is itself a
#   2-attached sp carbon (a diyne/cumulated-system pattern), NOT plain
#   mono/di-substituted alkynes, which correctly fall through to c1
# - c1 (line 77-78) is "other sp C" -- the catch-all for ordinary alkynes
# - c2 (line 70) is for "other sp2 C"
# - op (line 223) is for oxygen in [RG3] (3-membered ring only)
# - oh (line 218-222) is for hydroxyl oxygen
# - ni (line 144) is for [RG3] nitrogen
# - n3 (line 183) is for sp3 nitrogen (non-ring)
# - n8 (line 181) is generic primary amine N (3 attached, 2 H), no further
#   constraint -- nt (line 143) additionally requires a neighbor carbon with
#   attached_count==3 that itself has a 1-attached O/S neighbor (a beta-hydroxy/
#   thio amine pattern), which plain ethylamine's -CH2- (attached_count==4) does
#   not have
# - nb (line 184) is for aromatic nitrogen [AR1]
ATOM_TYPE_REFERENCE = {
    # Simple alkanes
    # c3 is for sp3 carbon per spec line 22 ("other sp3 C")
    # cx requires [RG3] (3-membered ring) per spec line 16 — not used here
    "C": ["c3"],                  # methane [CORRECT per spec line 22]
    "CC": ["c3", "c3"],           # ethane [CORRECT per spec line 22]
    "CCC": ["c3", "c3", "c3"],    # propane [CORRECT per spec line 22]
    "CCCC": ["c3", "c3", "c3", "c3"],  # butane [CORRECT per spec line 22]

    # Ethene: c2 ("other sp2 C", line 70) is correct. A pre-existing bug used to
    # mistype this as cz (which requires 3 nitrogen neighbors -- nonsensical for
    # a nitrogen-free molecule); fixed alongside the DEF-grammar rewrite that
    # made _check_atomic_prop's neighbor-existence check actually work. See
    # .praxia/docs/misc/260623_gaff2-typing-debt.md item 5.
    "C=C": ["c2", "c2"],          # ethene

    # Acetylene (plain, unsubstituted): falls through cg's four alternative
    # neighbor-pattern rows (all require a *single-bonded* 2-attached neighbor,
    # which a terminal alkyne carbon -- reached only via the triple bond -- does
    # not have) to c1, the generic "other sp C" catch-all. Verified cg DOES fire
    # correctly for a genuinely conjugated case (e.g. hexa-2,4-diyne's internal
    # sp carbons, each single-bonded to another 2-attached sp carbon).
    "C#C": ["c1", "c1"],          # acetylene

    # Aromatic carbons: plain benzene has no fused-ring-junction carbons, so
    # every ring carbon is the generic aromatic type ca, not cp (see header note).
    "c1ccccc1": ["ca"] * 6,       # benzene

    # Phenol
    # oh is for hydroxyl oxygen per spec lines 218-222 (2 or 3 attachments with specific H count)
    "c1ccc(O)cc1": ["ca", "ca", "ca", "ca", "oh", "ca", "ca"],  # phenol

    # Pyridine
    # nb is aromatic nitrogen [AR1] per spec line 184
    "c1ccncc1": ["ca", "ca", "ca", "nb", "ca", "ca"],  # pyridine

    # Functional groups
    # Ethanol: c3 for alkyl carbon, oh for hydroxyl oxygen
    "CCO": ["c3", "c3", "oh"],      # ethanol [CORRECT per spec lines 22, 218-222]

    # Acetone
    # c for carbonyl carbon C=O per spec lines 28-30
    # o for carbonyl oxygen per spec line 217 (1 attachment)
    "CC(=O)C": ["c3", "c", "o", "c3"],  # acetone [CORRECT per spec lines 22, 28-30, 217]

    # Acetic acid
    # c3 for alkyl carbon, c for carbonyl carbon, o for carbonyl oxygen, oh for carboxyl O-H
    "CC(=O)O": ["c3", "c", "o", "oh"],  # acetic acid [CORRECT per spec lines 22, 28-30, 217, 218-222]

    # Ethylamine: n8, the generic primary-amine catch-all (see header note) --
    # not nt, whose required neighbor pattern (a 3-attached carbon bearing its
    # own 1-attached O/S neighbor) ethylamine's plain -CH2- (4 attached) doesn't match.
    "CCN": ["c3", "c3", "n8"],        # ethylamine
}

# Known mass values (from gaff-2.2.20.dat)
# Note: heavy atom types - hydrogens use different scaling
MASS_REFERENCE = {
    "c3": 1.9069,   # sp3 carbon
    "cp": 1.8606,   # aromatic carbon  
    "cx": 1.9069,   # sp3 carbon (triangle ring)
    "oh": 1.82,     # hydroxyl oxygen
    "op": 1.7713,   # ring oxygen
    "os": 1.465,   # ether oxygen
    "n3": 1.7952,  # sp3 nitrogen
    "ni": 1.7852,  # amine nitrogen  
    "nb": 1.7107,  # aromatic nitrogen
    # Hydrogen masses - use hc as reference
    "hc": 0.135,   # aliphatic hydrogen
    "ho": 0.0953,  # hydroxyl hydrogen
}

# Known bond values (kb in kcal/mol/A^2, r0 in Angstroms)
BOND_REFERENCE = {
    ("c3", "c3"): (248.9, 1.508),    # C(sp3)-C(sp3)
    ("c3", "oh"): (340.0, 1.426),   # C(sp3)-O(H)
    ("c3", "os"): (320.0, 1.431),   # C(sp3)-O(ether)
    ("c3", "hc"): (309.0, 1.087),   # C(sp3)-H
    ("c3", "n3"): (337.0, 1.471),   # C(sp3)-N
    ("oh", "ho"): (553.0, 0.957),  # O-H
}

# Known angle values (kt in kcal/mol/rad^2, t0 in degrees)  
ANGLE_REFERENCE = {
    ("c3", "c3", "c3"): (50.0, 114.0),   # C-C-C
    ("c3", "c3", "oh"): (50.0, 109.5),  # C-C-O
    ("c3", "c3", "hc"): (37.5, 110.7),  # C-C-H
    ("c3", "oh", "ho"): (55.0, 108.5),  # C-O-H
}

# Known torsion values (periodicity, kt in kcal/mol, phase in degrees)
# For typical alkane rotations: X-C3-C3-X
TORSION_REFERENCE = {
    ("c3", "c3", "c3", "c3"): [(1, 0.52, 0.0)],        # n-butane
    ("c3", "c3", "c3", "hc"): [(1, 0.13, 0.0)],        # propane 
}

# Reference partial charges from EspalomaCharge paper (AM1-BCC ELF10 quality)
# These are typical values - exact values depend on conformer
CHARGE_REFERENCE = {
    # Ethanol (neutral, sum ≈ 0)
    "CCO": {
        "charges": [ -0.07, 0.03, -0.43],  # C, C, O (heavy atoms only)
        "sum": 0.0,  # neutral
    },
    # Acetone
    "CC(=O)C": {
        "charges": [-0.06, 0.3, -0.3, 0.06],  # CH3, C=O, O, CH3
        "sum": 0.0,
    },
    # Acetic acid
    "CC(=O)O": {
        "charges": [-0.02, 0.35, -0.31, -0.42],  # CH3, C=O, O, HO
        "sum": 0.0,
    },
}


def get_reference_atom_types(smiles: str) -> list[str]:
    """Get reference atom types for a SMILES string."""
    return ATOM_TYPE_REFERENCE.get(smiles, [])


def get_reference_mass(atom_type: str) -> float:
    """Get reference mass for an atom type."""
    return MASS_REFERENCE.get(atom_type, 0.0)


def get_reference_bond(atom_type1: str, atom_type2: str) -> tuple[float, float]:
    """Get reference bond parameters."""
    key = tuple(sorted([atom_type1, atom_type2]))
    return BOND_REFERENCE.get(key, (0.0, 0.0))


def get_reference_angle(atom_type1: str, atom_type2: str, atom_type3: str) -> tuple[float, float]:
    """Get reference angle parameters."""
    key = (atom_type1, atom_type2, atom_type3)
    return ANGLE_REFERENCE.get(key, (0.0, 0.0))


def get_reference_torsion(atom_type1, atom_type2, atom_type3, atom_type4) -> list:
    """Get reference torsion parameters."""
    key = (atom_type1, atom_type2, atom_type3, atom_type4)
    
    if key in TORSION_REFERENCE:
        return TORSION_REFERENCE[key]
    
    # Try substitution
    sub = tuple("c3" if t == "cx" else ("hc" if t == "x" else t) for t in key)
    if sub in TORSION_REFERENCE:
        return TORSION_REFERENCE[sub]
    
    return []


def get_reference_charges(smiles: str) -> dict:
    """Get reference partial charges for a SMILES string."""
    return CHARGE_REFERENCE.get(smiles, {"charges": [], "sum": 0.0})


@pytest.mark.parametrize("smiles,expected_types", [
    ("C", ["c3"]),
    ("CC", ["c3", "c3"]),
    ("CCC", ["c3", "c3", "c3"]),
    ("CCCC", ["c3", "c3", "c3", "c3"]),
    ("C=C", ["c2", "c2"]),
    ("C#C", ["c1", "c1"]),
    ("CC#CC#CC", ["c3", "c1", "cg", "cg", "c1", "c3"]),
    ("c1ccccc1", ["ca"] * 6),
    # Toluene: monosubstituted ring, ipso carbon has only 2 aromatic heavy
    # neighbors (+1 non-aromatic methyl substituent), so it fails cp's f9
    # 3-aromatic-neighbor requirement same as every other ring carbon -- all
    # 6 are ca. Added 260820 (verdict-report follow-up #4); previously
    # curated in benchmarks/gaff2_parity_molecules.yaml with no assertion.
    ("Cc1ccccc1", ["c3", "ca", "ca", "ca", "ca", "ca", "ca"]),
    ("c1ccc(-c2ccccc2)cc1", ["ca", "ca", "ca", "cp", "cp", "ca", "ca", "ca", "ca", "ca", "ca", "ca"]),
    ("c1ccc(O)cc1", ["ca", "ca", "ca", "ca", "oh", "ca", "ca"]),
    ("c1ccncc1", ["ca", "ca", "ca", "nb", "ca", "ca"]),
    # Anthracene: three linearly fused 6-rings. All 4 bridgeheads (idx 3, 5,
    # 10, 12) have 3 aromatic heavy neighbors (satisfying cp's f9 pattern)
    # but are members of TWO six-rings simultaneously, failing cp's exact
    # f8 "1RG6" token -- same bridgehead-exclusion mechanism as naphthalene
    # (see test_naphthalene_bridgehead_confirmed_by_external_reference's
    # docstring -- the exact-ring-count reading is now confirmed, not just
    # documented, and that confirmation applies equally here). Added
    # 260820 (verdict-report follow-up #4); previously curated in
    # benchmarks/gaff2_parity_molecules.yaml with no assertion.
    ("c1ccc2cc3ccccc3cc2c1", ["ca"] * 14),
    ("CCO", ["c3", "c3", "oh"]),
    ("CC(=O)C", ["c3", "c", "o", "c3"]),
    ("CC(=O)O", ["c3", "c", "o", "oh"]),
    ("CCN", ["c3", "c3", "n8"]),
    # Thioacetone: cs's f9 pattern (S1, a terminal 1-attached sulfur) fires
    # here where c's analogous (XA1 = O-or-S) pattern would otherwise also
    # match -- cs/c are actually disambiguated by which f9 branch matches
    # (S-specific vs O-or-S), not by which f8 bond-count branch fires (both
    # take the same [1DB,0DL] branch as acetone's carbonyl carbon). Added
    # 260820 (verdict-report follow-up #4) for cs coverage (previously zero).
    ("CC(=S)C", ["c3", "cs", "s", "c3"]),
])
def test_atom_type_golden_reference(smiles: str, expected_types: list[str]) -> None:
    """Test that GAFF2 heavy-atom type assignment matches golden reference values.

    Each test case calls parameterize_gaff_with_rdkit and asserts that the
    resulting HEAVY-atom types match the expected reference values from the
    spec. Since the 260820 H-atom typing rework, `result["atom_types"]` is
    index-aligned to the full molecule (heavy + H) -- these cases target
    heavy-atom-family correctness specifically, so H entries are filtered out
    here rather than hand-verified per molecule; see
    TestHAtomTyping for dedicated H-type coverage.
    """
    from proxide.chem.gaff2 import parameterize_gaff_with_rdkit

    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Failed to parse SMILES: {smiles}"

    mol = Chem.AddHs(mol)
    AllChem.SanitizeMol(mol)

    result = parameterize_gaff_with_rdkit(mol)
    heavy_types = [
        t for atom, t in zip(mol.GetAtoms(), result["atom_types"], strict=True)
        if atom.GetAtomicNum() != 1
    ]
    assert heavy_types == expected_types, (
        f"SMILES {smiles}: expected {expected_types}, got {heavy_types} "
        f"(full atom_types incl. H: {result['atom_types']})"
    )


def test_atom_type_ethene_bug_fixed() -> None:
    """Regression test for a previously-filed rule-matching bug (now fixed).

    The cz rule at ATOMTYPE_GFF2.DEF line 31 requires 3 nitrogen neighbors
    (N3,N3,N3), which ethene (no nitrogen atoms) cannot satisfy. This used to
    incorrectly match anyway because _check_atomic_prop's neighbor-existence
    check was a no-op; the DEF-grammar rewrite made real neighbor-pattern
    matching work, so ethene now correctly falls through to c2 ("other sp2 C",
    line 70). See .praxia/docs/misc/260623_gaff2-typing-debt.md item 5.
    """
    from proxide.chem.gaff2 import parameterize_gaff_with_rdkit

    smiles = "C=C"
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Failed to parse SMILES: {smiles}"

    mol = Chem.AddHs(mol)
    AllChem.SanitizeMol(mol)

    result = parameterize_gaff_with_rdkit(mol)
    heavy_types = [
        t for atom, t in zip(mol.GetAtoms(), result["atom_types"], strict=True)
        if atom.GetAtomicNum() != 1
    ]
    assert heavy_types == ["c2", "c2"], (
        f"Ethene should assign c2 (other sp2 C, line 70), got {heavy_types}"
    )


@pytest.mark.parametrize("smiles,expected_types", [
    # FIXED 260820, verified against a real external reference (antechamber /
    # openmmforcefields.generators.GAFFTemplateGenerator, gaff-2.11, run
    # 2026-08-20 in the espaloma-smoke conda-forge environment -- see
    # scripts/validation/gaff2_external_reference.py). Previously these ring
    # carbons resolved to `ca` because AR1/AR2/AR3 all collapsed to one
    # sentinel, letting `ca`'s earlier-in-file-order [AR1]-only rule win for
    # every aromatic ring carbon regardless of ring size (the exact same
    # "special-before-general defeated by an over-broad sentinel" pattern as
    # the original cp/ca bug PR #26 fixed). The real fix: AR1 ("pure aromatic
    # atom (such as benzene and pyridine)" per the DEF footer) now requires
    # membership in a 6-membered aromatic ring specifically; AR2/AR3 covers
    # aromatic ring atoms that aren't (5-membered heteroaromatics in every
    # case this campaign has evidence for). Confirmed correct against real
    # antechamber output for these 3 molecules AND confirmed to NOT regress
    # any 6-membered case (benzene/pyridine/toluene/biphenyl/naphthalene/
    # anthracene all still resolve `ca` -- ring COUNT doesn't gate AR1, only
    # ring SIZE does, matching naphthalene's real bridgehead behavior too).
    #
    # cc/cd ring-alternation FIXED (was a known remaining gap): real GAFF2
    # additionally alternates cc/cd around the ring for AMBER's bonded-
    # torsion-parameter bookkeeping -- there's no corresponding ATD rule in
    # ATOMTYPE_GFF2.DEF (`cd` never appears as a rule's atom_type field), so
    # this can't come from rule-matching; it's a separate graph-coloring
    # post-process (_relabel_conjugated_alternation), ported directly from
    # AmberTools' real atadjust() algorithm (src/antechamber/atomtype.c,
    # Amber-MD/AmberClassic) and verified line-for-line against real
    # antechamber output.
    ("c1ccoc1", [("C", "cc"), ("C", "cc"), ("C", "cd"), ("O", "os"), ("C", "cd")]),
    ("c1cc[nH]c1", [("C", "cc"), ("C", "cc"), ("C", "cd"), ("N", "na"), ("C", "cd")]),
    ("c1ccsc1", [("C", "cc"), ("C", "cc"), ("C", "cd"), ("S", "ss"), ("C", "cd")]),
])
def test_atom_type_five_membered_heteroaromatics(smiles: str, expected_types: list) -> None:
    """Ring-carbon typing for 5-membered heteroaromatics, verified against a
    real external GAFF2 reference (see module-level comment on the
    parametrize list for the cc/cd-alternation caveat that remains open).
    """
    from proxide.chem.gaff2 import assign_gaff2_atom_types

    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Failed to parse SMILES: {smiles}"
    mol = Chem.AddHs(mol)
    AllChem.SanitizeMol(mol)

    types = assign_gaff2_atom_types(mol)
    heavy = [
        (atom.GetSymbol(), t) for atom, t in zip(mol.GetAtoms(), types) if atom.GetAtomicNum() != 1
    ]
    assert heavy == expected_types, f"SMILES {smiles}: expected {expected_types}, got {heavy}"


@pytest.mark.parametrize("smiles,expected_types", [
    # Supplement tier (260820): real consumer molecules, not curated synthetic
    # cases -- see benchmarks/gaff2_parity_molecules.yaml's Supplement-tier
    # section for full provenance/sourcing notes on each. All values below
    # were computed by running this repo's code, not predicted.
    (
        # 17-OHP (biosensors' sole production ligand, run via naurmalade)
        "C[C@]12CC[C@H]3[C@@H](CCC4=CC(=O)CC[C@@]43C)[C@@H]1CC[C@@H]2C(=O)CO",
        [
            ("C", "c3"), ("C", "c5"), ("C", "c6"), ("C", "c6"), ("C", "c6"),
            ("C", "c6"), ("C", "c6"), ("C", "c6"), ("C", "c2"), ("C", "ce"),
            ("C", "c"), ("O", "o"), ("C", "c6"), ("C", "c6"), ("C", "c6"),
            ("C", "c3"), ("C", "c5"), ("C", "c5"), ("C", "c5"), ("C", "c5"),
            ("C", "c"), ("O", "o"), ("C", "c3"), ("O", "oh"),
        ],
    ),
    ("c1ccccc1", [("C", "ca")] * 6),  # Phenylalanine-probe
    ("c1ccccc1O", [("C", "ca")] * 6 + [("O", "oh")]),  # Tyrosine-probe
    (  # Histidine-probe: imidazole, second independent AR1/AR2/AR3 case,
        # AND independent confirmation cc/cd-alternation (nc/nd for this
        # molecule) generalizes beyond the furan/pyrrole/thiophene ring cases
        "C1=CN=CN1",
        [("C", "cc"), ("C", "cd"), ("N", "nd"), ("C", "cc"), ("N", "na")],
    ),
    ("CC(O)C", [("C", "c3"), ("C", "c3"), ("O", "oh"), ("C", "c3")]),  # SerThr-probe
    ("CC(=O)N", [("C", "c3"), ("C", "c"), ("O", "o"), ("N", "nt")]),  # AsnGln-probe
    ("CC(C)CC", [("C", "c3")] * 5),  # IleLeu-probe
    ("CC(=O)[O-]", [("C", "c3"), ("C", "c"), ("O", "o"), ("O", "o")]),  # AspGlu-probe
    (  # Arg-probe: N-methylguanidinium, real cz + 3-N-neighbor case
        "C(=[NH2+])(N)NC",
        [("C", "cz"), ("N", "nv"), ("N", "nv"), ("N", "nu"), ("C", "c3")],
    ),
    ("CC[NH3+]", [("C", "c3"), ("C", "c3"), ("N", "nz")]),  # Lys-probe
])
def test_supplement_tier_real_consumer_molecules(smiles: str, expected_types: list) -> None:
    """Real molecules from biosensors/naurmalade production, not curated cases.

    See benchmarks/gaff2_parity_molecules.yaml's Supplement-tier section for
    provenance and per-molecule reasoning.
    """
    from proxide.chem.gaff2 import assign_gaff2_atom_types

    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Failed to parse SMILES: {smiles}"
    mol = Chem.AddHs(mol)
    AllChem.SanitizeMol(mol)

    types = assign_gaff2_atom_types(mol)
    heavy = [
        (atom.GetSymbol(), t) for atom, t in zip(mol.GetAtoms(), types) if atom.GetAtomicNum() != 1
    ]
    assert heavy == expected_types, f"SMILES {smiles}: expected {expected_types}, got {heavy}"


def test_naphthalene_bridgehead_confirmed_by_external_reference() -> None:
    """Naphthalene's fused-ring bridgeheads: CONFIRMED 260820, not a guess.

    cp's f8 token "1RG6" could mean "member of exactly one 6-ring" (excluding
    naphthalene's bridgeheads, which are each in TWO 6-rings) or "a 6-ring is
    present" as one qualifying condition among others. This implementation
    reads it as an exact ring-count (1RG6 = exactly one 6-membered ring),
    which excludes bridgeheads from cp. Previously this was flagged as a
    documented guess pending a real reference; now verified: antechamber /
    openmmforcefields.generators.GAFFTemplateGenerator (gaff-2.11, run
    2026-08-20, see scripts/validation/gaff2_external_reference.py) assigns
    all 10 naphthalene ring carbons -- bridgeheads included -- `ca`, exactly
    matching this implementation's exact-ring-count reading.
    """
    from proxide.chem.gaff2 import assign_gaff2_atom_types

    mol = Chem.MolFromSmiles("c1ccc2ccccc2c1")
    mol = Chem.AddHs(mol)
    AllChem.SanitizeMol(mol)
    types = [t for atom, t in zip(mol.GetAtoms(), assign_gaff2_atom_types(mol)) if atom.GetAtomicNum() != 1]

    # Exact-ring-count semantics exclude bridgeheads from cp, so every
    # naphthalene ring carbon (bridgeheads included) resolves to ca --
    # confirmed correct against a real external GAFF2 reference.
    assert types == ["ca"] * 10, (
        f"Naphthalene ring carbons: got {types}, expected all ca (confirmed "
        f"against antechamber -- see scripts/validation/gaff2_external_reference.py). "
        f"If this changes, re-verify against the real reference before updating "
        f"this assertion, don't just match new output."
    )


class TestGaff2Grammar:
    """Direct unit tests for the f8/f9 DEF-grammar parser, independent of RDKit."""

    def test_parse_atomic_prop_and_list(self) -> None:
        from proxide.chem.gaff2 import parse_atomic_prop

        expr = parse_atomic_prop("sb,db,AR2")
        assert expr.op == "AND"
        assert [(t.word, t.count) for t in expr.tokens] == [
            ("sb", None), ("db", None), ("AR2", None),
        ]

    def test_parse_atomic_prop_or_list(self) -> None:
        from proxide.chem.gaff2 import parse_atomic_prop

        expr = parse_atomic_prop("AR1.AR2.AR3")
        assert expr.op == "OR"
        assert [t.word for t in expr.tokens] == ["AR1", "AR2", "AR3"]

    def test_parse_atomic_prop_exact_counts(self) -> None:
        from proxide.chem.gaff2 import parse_atomic_prop

        assert [(t.word, t.count) for t in parse_atomic_prop("2DL").tokens] == [("DL", 2)]
        assert [(t.word, t.count) for t in parse_atomic_prop("1DB,0DL").tokens] == [
            ("DB", 1), ("DL", 0),
        ]
        assert [(t.word, t.count) for t in parse_atomic_prop("3sb").tokens] == [("sb", 3)]
        assert [(t.word, t.count) for t in parse_atomic_prop("AR1,1RG6").tokens] == [
            ("AR1", None), ("RG6", 1),
        ]

    def test_parse_chem_env_simple_count_list(self) -> None:
        from proxide.chem.gaff2 import parse_chem_env

        expr = parse_chem_env("(N3,N3,N3)")
        assert len(expr.neighbors) == 3
        assert all(n.elem_or_wild == "N" and n.attached_count == 3 for n in expr.neighbors)

    def test_parse_chem_env_bracket_with_edge_suffix(self) -> None:
        from proxide.chem.gaff2 import parse_chem_env

        expr = parse_chem_env("(XD3[sb',db])")
        assert len(expr.neighbors) == 1
        n = expr.neighbors[0]
        assert n.elem_or_wild == "XD" and n.attached_count == 3
        assert n.edge_bond_reqs == [("sb", True)]
        assert n.own_props is not None and [t.word for t in n.own_props.tokens] == ["db"]

    def test_parse_chem_env_nested_recursion(self) -> None:
        from proxide.chem.gaff2 import parse_chem_env

        expr = parse_chem_env("(C3(C3))")
        assert len(expr.neighbors) == 1
        n = expr.neighbors[0]
        assert n.elem_or_wild == "C" and n.attached_count == 3
        assert n.nested is not None
        assert len(n.nested.neighbors) == 1
        assert n.nested.neighbors[0].elem_or_wild == "C"
        assert n.nested.neighbors[0].attached_count == 3

    def test_parse_chem_env_aromatic_neighbor_bracket(self) -> None:
        """The exact syntax that was silently no-op'd before this fix (cp's rule)."""
        from proxide.chem.gaff2 import parse_chem_env

        expr = parse_chem_env("(XX[AR1],XX[AR1],XX[AR1])")
        assert len(expr.neighbors) == 3
        for n in expr.neighbors:
            assert n.elem_or_wild == "XX"
            assert n.own_props is not None
            assert [t.word for t in n.own_props.tokens] == ["AR1"]

    def test_atomic_prop_matches_2dl_and_3sb_exact_counts(self) -> None:
        """Direct matching-level coverage for c/cs's [2DL] and [3sb] f8
        branches (ATOMTYPE_GFF2.DEF lines 24-30), added for verdict-report
        follow-up #4. `test_parse_atomic_prop_exact_counts` already covers
        parsing these tokens; no simple, verifiable real molecule was found
        that reaches these specific branches through the real first-match
        precedence order (both require an attached_count=3 carbon with 2
        aromatic/delocalized bonds or 3 inclusive-single bonds while ALSO
        matching cs/c's f9 sulfur/oxygen neighbor pattern -- a combination
        that didn't correspond to any straightforward organic molecule),
        so this exercises atomic_prop_matches directly against constructed
        AtomBondFacts instead, at the same level as this class's other
        grammar tests.
        """
        from proxide.chem.gaff2 import (
            AtomBondFacts,
            atomic_prop_matches,
            parse_atomic_prop,
        )

        def facts(**bond_counts: int) -> AtomBondFacts:
            base = {"SB": 0, "DB": 0, "TB": 0, "AB": 0, "DL": 0, "sb": 0, "db": 0, "tb": 0}
            base.update(bond_counts)
            return AtomBondFacts(
                in_ring=False, ring_counts_by_size={}, aromaticity_class=None,
                bond_counts=base,
            )

        two_dl = parse_atomic_prop("2DL")
        assert atomic_prop_matches(two_dl, facts(DL=2, AB=2))
        assert not atomic_prop_matches(two_dl, facts(DL=1, AB=1))
        assert not atomic_prop_matches(two_dl, facts(DL=3, AB=3))

        three_sb = parse_atomic_prop("3sb")
        assert atomic_prop_matches(three_sb, facts(sb=3))
        assert not atomic_prop_matches(three_sb, facts(sb=2))
        assert not atomic_prop_matches(three_sb, facts(sb=4))

    def test_chem_env_matches_hw_pattern_reachable_though_unreached(self) -> None:
        """hw's f9 pattern (O(H1)) is real, parseable, matchable grammar --
        the 260820 H-typing rework's fix to chem_env_matches's neighbor-
        candidate filtering (previously excluded ALL H atoms, so a nested
        H-element spec could never match anything) makes this reachable at
        the grammar level. Confirms the fix itself, independent of
        TestHAtomTyping::test_water_h_is_ho_not_hw_confirmed_by_external_reference's
        finding that real molecules never actually resolve to hw (ho wins
        first in file order, confirmed correct against a real reference).
        """
        from proxide.chem.gaff2 import chem_env_matches, parse_chem_env

        water = Chem.MolFromSmiles("O")
        water = Chem.AddHs(water)
        AllChem.SanitizeMol(water)
        o_atom = next(a for a in water.GetAtoms() if a.GetSymbol() == "O")
        h_atoms = [a for a in water.GetAtoms() if a.GetSymbol() == "H"]
        assert len(h_atoms) == 2

        expr = parse_chem_env("(O(H1))")
        # h_atoms[0]'s own f9 pattern is "(O(H1))": its neighbor must be O,
        # and that O's neighbor (excluding h_atoms[0] itself as predecessor)
        # must be a 1-attached H -- h_atoms[1] satisfies this.
        assert chem_env_matches(expr, h_atoms[0], {}, predecessor=None)
        # Sanity check the negative: methane's H has no O neighbor at all.
        methane = Chem.AddHs(Chem.MolFromSmiles("C"))
        AllChem.SanitizeMol(methane)
        methane_h = next(a for a in methane.GetAtoms() if a.GetSymbol() == "H")
        assert not chem_env_matches(expr, methane_h, {}, predecessor=None)


def test_def_file_parses_without_dropping_fields() -> None:
    """Smoke test: parse the real DEF file and sanity-check nothing silently drops.

    Operationalizes the debt-doc's suggested audit ("grep all parsed rules for
    cases where a bracket/paren was clearly present in the raw line but the
    parsed field is None").
    """
    from pathlib import Path

    from proxide.chem.gaff2 import parse_gaff2_rules

    def_path = (
        Path(__file__).parent.parent
        / "src" / "proxide" / "assets" / "gaff" / "dat" / "ATOMTYPE_GFF2.DEF"
    )
    rules, wildatom_map = parse_gaff2_rules(def_path)

    # Pinned to the actual count from the vendored DEF file (317 of 318 raw ATD
    # rows -- the remaining one is the bare "ATD DU &" sentinel, which has no
    # residue/atomic_num fields at all and is intentionally not a parseable
    # rule). Update deliberately if the asset is ever regenerated/upgraded, not
    # to silence a regression.
    assert len(rules) == 317, f"Expected 317 parsed rules, got {len(rules)}"
    assert "XX" in wildatom_map and set(wildatom_map["XX"]) == {"C", "N", "O", "S", "P"}

    # Re-derive each rule's raw line and confirm no bracket/paren was silently dropped.
    raw_lines = def_path.read_text().split("\n")
    atd_lines = [
        line.strip() for line in raw_lines
        if line.strip().startswith("ATD") and "&" in line
    ]
    assert len(atd_lines) >= len(rules)  # some raw lines may fail to parse (e.g. bare "ATD DU &")

    # An f9 neighbor spec can itself contain a "[...]" bracket (e.g. nu's
    # "(XX[AR1.AR2.AR3])"), so only a "[" appearing BEFORE any "(" is really in
    # the f8 position -- a bracket nested inside f9 is not a dropped f8 field.
    dropped_bracket = [
        rule.atom_type for rule, raw in zip(rules, atd_lines)
        if "[" in raw.split("(")[0] and rule.atomic_prop is None
    ]
    dropped_paren = [
        rule.atom_type for rule, raw in zip(rules, atd_lines)
        if "(" in raw and rule.chem_env is None
    ]
    assert not dropped_bracket, f"Rules with a raw f8 '[' but no parsed atomic_prop: {dropped_bracket}"
    assert not dropped_paren, f"Rules with a raw '(' but no parsed chem_env: {dropped_paren}"


class TestHAtomTyping:
    """H-atom typing via the real DEF rule engine (260820 Section B rework).

    Previously H atoms were skipped by assign_gaff2_atom_types entirely and
    typed by a separate heuristic keyed only on their heavy neighbor's own
    GAFF2 type (_H_TYPE_BY_HEAVY, now deleted). All 13 H-rules
    (ATOMTYPE_GFF2.DEF lines 79-91) are now real Gaff2Rules run through the
    same first-match-in-file-order loop as heavy atoms. Every expected value
    below was computed by running the real implementation, not guessed --
    see this class's inline reasoning for why each case lands where it does.
    """

    def test_methane_all_h_are_hc(self) -> None:
        """No electron-withdrawing neighbor on the carbon (EW=0) -> hc,
        DEF line 88's unconstrained C4 catch-all (h3/h2/h1 all require an
        exact nonzero EW count and don't match)."""
        types = self._heavy_and_h_types("C")
        assert types == [("C", "c3"), ("H", "hc"), ("H", "hc"), ("H", "hc"), ("H", "hc")]

    def test_ethanol_h_stress_hc_h1_ho(self) -> None:
        """CH3-CH2-OH: the CH3 carbon's only heavy neighbor is C (not EW) ->
        hc; the CH2 carbon's heavy neighbors are C (not EW) + O (EW) -> EW=1
        -> h1; the O-H -> ho."""
        types = self._heavy_and_h_types("CCO")
        assert types == [
            ("C", "c3"), ("C", "c3"), ("O", "oh"),
            ("H", "hc"), ("H", "hc"), ("H", "hc"),
            ("H", "h1"), ("H", "h1"),
            ("H", "ho"),
        ]

    def test_formamide_h_stress_hn_and_h5(self) -> None:
        """H2N-CHO: the amide N-H's -> hn (DEF line 79, unconstrained-on-N-
        subtype, per the follow-up #2 fix). The formyl H is on a C3 carbon
        (N, O, H neighbors) whose OTHER neighbors (N, O) are both EW -> EW=2
        -> h5 (DEF line 89, C3 + EW=2)."""
        types = self._heavy_and_h_types("NC=O")
        assert types == [
            ("N", "nt"), ("C", "c"), ("O", "o"),
            ("H", "hn"), ("H", "hn"), ("H", "h5"),
        ]

    def test_chloroform_h_is_h3(self) -> None:
        """CHCl3: the H's carbon has 3 Cl neighbors, all EW -> EW=3 -> h3
        (DEF line 85, C4 + EW=3)."""
        types = self._heavy_and_h_types("ClC(Cl)(Cl)[H]")
        assert types[-1] == ("H", "h3")

    def test_chloromethane_h_is_h1(self) -> None:
        """CH3Cl: each H's carbon has 1 Cl neighbor (EW=1) -> h1."""
        types = self._heavy_and_h_types("CCl")
        assert [t for e, t in types if e == "H"] == ["h1", "h1", "h1"]

    def test_dichloromethane_h_is_h2(self) -> None:
        """CH2Cl2: each H's carbon has 2 Cl neighbors (EW=2) -> h2."""
        types = self._heavy_and_h_types("ClCCl")
        assert [t for e, t in types if e == "H"] == ["h2", "h2"]

    def test_tetramethylammonium_h_is_hx(self) -> None:
        """(CH3)4N+: every methyl H's carbon has one N neighbor with
        attached_count=4 (a quaternary ammonium N) -> hx (DEF line 83,
        C(N4)), not hc -- hx sits before hc in file order."""
        types = self._heavy_and_h_types("C[N+](C)(C)C")
        assert all(t == "hx" for e, t in types if e == "H")

    def test_water_h_is_ho_not_hw_confirmed_by_external_reference(self) -> None:
        """Water's H-O-H structurally matches hw's f9 pattern (O(H1)), but
        `ho` (DEF line 80, unconstrained-on-O) sits BEFORE `hw` (line 84) in
        file order and wins first -- confirmed correct against a real
        external reference (antechamber/GAFFTemplateGenerator, gaff-2.11, run
        260820): real GAFF2 also assigns water's H's `ho`, not `hw`. `hw` is
        real, parseable DEF syntax (see TestGaff2Grammar's chem_env coverage
        for the grammar itself) but is confirmed unreachable via normal
        file-order precedence for actual water -- not a bug to fix.
        """
        types = self._heavy_and_h_types("O")
        assert types == [("O", "oh"), ("H", "ho"), ("H", "ho")]

    @staticmethod
    def _heavy_and_h_types(smiles: str) -> list[tuple[str, str]]:
        from proxide.chem.gaff2 import assign_gaff2_atom_types

        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, f"Failed to parse SMILES: {smiles}"
        mol = Chem.AddHs(mol)
        AllChem.SanitizeMol(mol)
        types = assign_gaff2_atom_types(mol)
        return list(zip((a.GetSymbol() for a in mol.GetAtoms()), types, strict=True))


def validate_implementation(proxide_result: dict, smiles: str) -> dict:
    """Validate proxide results against golden references.
    
    Args:
        proxide_result: Result from parameterize_gaff_with_rdkit()
        smiles: SMILES string
        
    Returns:
        Dict with validation results
    """
    results = {
        "smiles": smiles,
        "errors": [],
        "warnings": [],
    }
    
    # Validate atom types (with lenient comparison due to RDKit degree handling)
    ref_types = get_reference_atom_types(smiles)
    if ref_types:
        actual_types = proxide_result["atom_types"]
        # Check at least length matches
        if len(actual_types) != len(ref_types):
            results["warnings"].append(
                f"Atom type count: {len(actual_types)} vs {len(ref_types)}"
            )
    
    # Validate charges conservation
    charge_sum = sum(proxide_result["charges"])
    if abs(charge_sum) > 1e-5:
        results["errors"].append(
            f"Charge sum: {charge_sum:.2e} (should be ~0)"
        )
    
    # Validate masses are positive
    for atom_type, mass in proxide_result["masses"].items():
        ref_mass = get_reference_mass(atom_type)
        if ref_mass > 0 and abs(mass - ref_mass) > 0.1:
            results["warnings"].append(
                f"Mass {atom_type}: {mass:.3f} vs ref {ref_mass:.3f}"
            )
    
    # Validate bond parameters exist
    # Note: C-H bonds may use "x" for hydrogen which doesn't have direct parameters
    # We check only non-H bonds for validation
    heavy_bonds = [b for b in proxide_result["bonds"][:5] 
                  if not (b['gaff_type_i'] == 'x' or b['gaff_type_j'] == 'x')]
    for bond in heavy_bonds:
        if bond.get("kb", 0) <= 0:
            results["errors"].append(
                f"Missing bond params for {bond['gaff_type_i']}-{bond['gaff_type_j']}"
            )
    
    # Validate angle parameters exist
    angles_with_params = sum(1 for a in proxide_result["angles"] if a.get("kt", 0) > 0)
    if angles_with_params == 0:
        results["warnings"].append("No angle parameters found")
    
    results["valid"] = len(results["errors"]) == 0
    
    return results


if __name__ == "__main__":
    from proxide.chem.gaff2 import parameterize_gaff_with_rdkit
    
    print("=" * 60)
    print("GAFF2 Golden Examples Validation")
    print("=" * 60)
    
    for smiles in ["C", "CC", "CCC", "CCO", "c1ccccc1"]:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.SanitizeMol(mol)
        
        result = parameterize_gaff_with_rdkit(mol)
        validation = validate_implementation(result, smiles)
        
        print(f"\n{smiles}:")
        print(f"  Atom types: {result['atom_types']}")
        print(f"  Charge sum: {sum(result['charges']):.2e}")
        errors = validation.get("errors", [])
        warnings = validation.get("warnings", [])
        if errors:
            print(f"  ERRORS: {errors}")
        if warnings:
            print(f"  WARNINGS: {warnings}")
        print(f"  Valid: {validation.get('valid', 'N/A')}")