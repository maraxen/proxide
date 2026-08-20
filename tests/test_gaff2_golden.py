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


@pytest.mark.parametrize("smiles,expected_types", [
    # 2-Pyranone: FOUND by a post-merge PARITY audit (260820), not a case
    # anyone had reason to construct beforehand -- falsifies the "6-membered
    # aromatic ring -> AR1/ca" proxy the AR1/AR2/AR3 fix originally shipped
    # with. Real antechamber's own AR3 test ("planar rings formed 'outside'
    # double bonds", ring.c) fires BEFORE the AR1 test: the ring carbon
    # bonded to the exocyclic carbonyl O via a double bond demotes the WHOLE
    # ring from AR1 to AR23, so the other 4 ring carbons resolve via the
    # cc/cd family (not ca), while the carbonyl carbon itself resolves via
    # the ordinary `c` rule (same mechanism as acetone/formamide) once `ca`
    # is no longer a candidate for it. Verified against real antechamber
    # output (gaff-2.11): o, c, cc, cd, cd, cc, os, in exactly this order.
    ("O=C1C=CC=CO1", [
        ("O", "o"), ("C", "c"), ("C", "cc"), ("C", "cd"),
        ("C", "cd"), ("C", "cc"), ("O", "os"),
    ]),
    # Boundary cases confirming the exocyclic-multibond check does NOT
    # over-fire when the double bond is one atom removed from the ring (a
    # SINGLE bond from the ring to a substituent that itself has its own
    # double bond) -- both stay ca, matching real antechamber exactly.
    ("O=Cc1ccccc1", [  # benzaldehyde
        ("O", "o"), ("C", "c"),
        ("C", "ca"), ("C", "ca"), ("C", "ca"), ("C", "ca"), ("C", "ca"), ("C", "ca"),
    ]),
    ("C=Cc1ccccc1", [  # styrene
        ("C", "c2"), ("C", "ce"),
        ("C", "ca"), ("C", "ca"), ("C", "ca"), ("C", "ca"), ("C", "ca"), ("C", "ca"),
    ]),
])
def test_ar1_exocyclic_multibond_demotion(smiles: str, expected_types: list) -> None:
    """AR1's exocyclic-double-bond exception (2-pyranone) and its boundary
    (benzaldehyde/styrene must NOT be demoted). See
    _ring_has_exocyclic_multibond's docstring for the AmberTools source this
    ports, and why the check must test the OTHER atom's total ring
    membership (not just "not in this specific ring") to avoid wrongly
    demoting fused-ring bridgeheads like naphthalene's.
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

    def test_relabel_conjugated_alternation_single_seed_only(self) -> None:
        """Confirmed post-merge PARITY audit (260820): real AmberTools'
        cpadjust() (governing cp/cq) has NO reseed mechanism -- it colors
        only the FIRST same-family connected component it finds and leaves
        every other disconnected component untouched, unlike atadjust()
        (governing cc/cd etc.), which does eventually seed every component
        via a `flag`-gated reseed in its relaxation loop. Zero real-molecule
        test previously exercised this (no biphenyl-type molecule with two
        disconnected cp systems is easy to construct), so this tests the
        generic mechanism directly against a small synthetic two-fragment
        molecule (two disconnected C=C pairs), matching this class's
        existing convention of unit-level coverage when no simple real
        molecule reaches a specific branch.
        """
        from proxide.chem.gaff2 import _relabel_conjugated_alternation

        # Two disconnected ethene-like fragments: C=C . C=C. Fake "family"
        # labels simulate being cc-eligible so the double bond triggers a
        # flip -- this tests the seeding mechanism itself, not real GAFF2
        # rule matching.
        mol = Chem.MolFromSmiles("C=C.C=C")
        mol = Chem.AddHs(mol)
        AllChem.SanitizeMol(mol)
        Chem.Kekulize(mol, clearAromaticFlags=False)
        pairs = {"X": "Y"}
        heavy_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() != 1]
        assert len(heavy_idx) == 4

        # single_seed_only=False (atadjust()-style): both components colored.
        types_multi = ["X"] * mol.GetNumAtoms()
        _relabel_conjugated_alternation(mol, types_multi, pairs, single_seed_only=False)
        assert [types_multi[i] for i in heavy_idx] == ["X", "Y", "X", "Y"]

        # single_seed_only=True (cpadjust()-style): only the first component
        # (lowest atom index) is colored; the second fragment is untouched.
        types_single = ["X"] * mol.GetNumAtoms()
        _relabel_conjugated_alternation(mol, types_single, pairs, single_seed_only=True)
        assert [types_single[i] for i in heavy_idx] == ["X", "Y", "X", "X"]


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

    def test_thiophene_beta_h_is_h4_not_ha(self) -> None:
        """Confirmed post-merge PARITY audit (260820): sulfur is a real
        electron-withdrawing atom per AmberTools' own source
        (`aromatic()`, src/antechamber/aromatic.c: `case 16: atom[i].ewd
        = 1; /* S is considered electron withdraw group */`) -- _EW_ATOMS
        previously omitted it, so the two ring H's beta to the thiophene
        sulfur (on the `cd`-typed carbons, each with EW=1 via the S
        neighbor) mistyped as `ha` (the unconstrained sp2 catch-all)
        instead of `h4` (DEF line 90, C3 + EW=1). The two H's alpha to
        sulfur (on the `cc`-typed carbons, whose OTHER heavy neighbor is
        just another ring carbon, not EW) correctly stay `ha`. Verified
        against real antechamber output (gaff-2.11): H1=ha, H2=ha, H3=h4,
        H4=h4, in exactly this atom order.
        """
        types = self._heavy_and_h_types("c1ccsc1")
        assert types == [
            ("C", "cc"), ("C", "cc"), ("C", "cd"), ("S", "ss"), ("C", "cd"),
            ("H", "ha"), ("H", "ha"), ("H", "h4"), ("H", "h4"),
        ]

    @staticmethod
    def _heavy_and_h_types(smiles: str) -> list[tuple[str, str]]:
        from proxide.chem.gaff2 import assign_gaff2_atom_types

        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, f"Failed to parse SMILES: {smiles}"
        mol = Chem.AddHs(mol)
        AllChem.SanitizeMol(mol)
        types = assign_gaff2_atom_types(mol)
        return list(zip((a.GetSymbol() for a in mol.GetAtoms()), types, strict=True))


class TestFullRuleCoverage:
    """Real-molecule, antechamber-verified coverage for GAFF2 organic atom types
    that had no test anywhere in this file before 260820's post-merge audit.

    Scope: proxide's ATOMTYPE_GFF2.DEF has 318 ATD rules / 193 atom-type tokens;
    193 - 91 are single-rule metal/lanthanide/actinide pass-throughs with no real
    combinatorial typing logic (out of scope for this class). Of the 91 "organic"
    (c/n/o/s/p/h-family) types, only 40 had ever been asserted anywhere in this
    file. This class closes that gap for every type where a real, chemically
    sensible molecule could be constructed and independently verified against
    real antechamber (gaff-2.11) output -- 50 types across 47 molecules below
    (some molecules exercise two target types at once, e.g. methylphosphine's
    hp H-type and p3 P-type).

    Explicitly excluded, not silently dropped:
    - `lp` (DEF: `ATD  lp * 0 1 &`, atomic_num=0): a virtual lone-pair
      pseudo-particle, not a real chemical element. No ordinary RDKit molecule
      from a SMILES/mol file can ever produce it -- there is no atom to type.
    - `p4`, `px`: both require a specific trivalent-P-with-a-double-bond
      structure (DEF fields: 3 total attachments, [db] fact, a specific
      chem_env). Every real SMILES tried either had RDKit expand P to
      hypervalent (5-attached, landing on p5) or fall through to the
      `pc`/`pe` conjugated-alternation family (which is earlier in file
      order and more general) rather than isolating p4/px's exact pattern --
      constructing the precise edge case would require an artificial
      formal-charge trick rather than a real, synthetically sensible ligand
      fragment, so these are deferred rather than forced.

    Finding this session: the `nv`/`nm`/`nn` mismatches below (aniline-type N
    adjacent to an aromatic ring, and its RG3/RG4 ring-nitrogen analogs) led to
    a real parser bug -- see `_parse_neighbor_spec`'s comment on the
    `(XX[AR1.AR2.AR3])` chem_env pattern for the fix (a "." was being silently
    treated as AND instead of the DEF footer's own documented OR, making these
    three rules permanently unsatisfiable and falling through to n8/np/nq).
    """

    @pytest.mark.parametrize("label,smiles,net_charge,expected_types", [
        # br
        ("bromobenzene", "Brc1ccccc1", 0, [("Br", "br"), ("C", "ca"), ("C", "ca"), ("C", "ca"), ("C", "ca"), ("C", "ca"), ("C", "ca"), ("H", "ha"), ("H", "ha"), ("H", "ha"), ("H", "ha"), ("H", "ha")]),
        # f
        ("fluorobenzene", "Fc1ccccc1", 0, [("F", "f"), ("C", "ca"), ("C", "ca"), ("C", "ca"), ("C", "ca"), ("C", "ca"), ("C", "ca"), ("H", "ha"), ("H", "ha"), ("H", "ha"), ("H", "ha"), ("H", "ha")]),
        # i
        ("iodobenzene", "Ic1ccccc1", 0, [("I", "i"), ("C", "ca"), ("C", "ca"), ("C", "ca"), ("C", "ca"), ("C", "ca"), ("C", "ca"), ("H", "ha"), ("H", "ha"), ("H", "ha"), ("H", "ha"), ("H", "ha")]),
        # cl
        ("chlorobenzene", "Clc1ccccc1", 0, [("Cl", "cl"), ("C", "ca"), ("C", "ca"), ("C", "ca"), ("C", "ca"), ("C", "ca"), ("C", "ca"), ("H", "ha"), ("H", "ha"), ("H", "ha"), ("H", "ha"), ("H", "ha")]),
        # hs
        ("ethanethiol", "CCS", 0, [("C", "c3"), ("C", "c3"), ("S", "sh"), ("H", "hc"), ("H", "hc"), ("H", "hc"), ("H", "h1"), ("H", "h1"), ("H", "hs")]),
        # hp + p3
        ("methylphosphine", "CP", 0, [("C", "c3"), ("P", "p3"), ("H", "hc"), ("H", "hc"), ("H", "hc"), ("H", "hp"), ("H", "hp")]),
        # n1
        ("acetonitrile", "CC#N", 0, [("C", "c3"), ("C", "c1"), ("N", "n1"), ("H", "hc"), ("H", "hc"), ("H", "hc")]),
        # nx
        ("trimethylammonium_cation", "C[NH+](C)C", 1, [("C", "c3"), ("N", "nx"), ("C", "c3"), ("C", "c3"), ("H", "hx"), ("H", "hx"), ("H", "hx"), ("H", "hn"), ("H", "hx"), ("H", "hx"), ("H", "hx"), ("H", "hx"), ("H", "hx"), ("H", "hx")]),
        # n4
        ("tetramethylammonium_cation", "C[N+](C)(C)C", 1, [("C", "c3"), ("N", "n4"), ("C", "c3"), ("C", "c3"), ("C", "c3"), ("H", "hx"), ("H", "hx"), ("H", "hx"), ("H", "hx"), ("H", "hx"), ("H", "hx"), ("H", "hx"), ("H", "hx"), ("H", "hx"), ("H", "hx"), ("H", "hx"), ("H", "hx")]),
        # n5
        ("aziridine", "C1CN1", 0, [("C", "cx"), ("C", "cx"), ("N", "n5"), ("H", "h1"), ("H", "h1"), ("H", "h1"), ("H", "h1"), ("H", "hn")]),
        # n6
        ("azetidine", "C1CCN1", 0, [("C", "cy"), ("C", "cy"), ("C", "cy"), ("N", "n6"), ("H", "h1"), ("H", "h1"), ("H", "hc"), ("H", "hc"), ("H", "h1"), ("H", "h1"), ("H", "hn")]),
        # n2
        ("n_methylethanimine", "CC=NC", 0, [("C", "c3"), ("C", "c2"), ("N", "n2"), ("C", "c3"), ("H", "hc"), ("H", "hc"), ("H", "hc"), ("H", "h4"), ("H", "h1"), ("H", "h1"), ("H", "h1")]),
        # nv
        ("aniline", "Nc1ccccc1", 0, [("N", "nv"), ("C", "ca"), ("C", "ca"), ("C", "ca"), ("C", "ca"), ("C", "ca"), ("C", "ca"), ("H", "hn"), ("H", "hn"), ("H", "ha"), ("H", "ha"), ("H", "ha"), ("H", "ha"), ("H", "ha")]),
        # ns
        ("beta_lactam", "O=C1CCN1", 0, [("O", "o"), ("C", "c"), ("C", "cy"), ("C", "cy"), ("N", "ns"), ("H", "hc"), ("H", "hc"), ("H", "h1"), ("H", "h1"), ("H", "hn")]),
        # np
        ("n_methylaziridine", "CN1CC1", 0, [("C", "c3"), ("N", "np"), ("C", "cx"), ("C", "cx"), ("H", "h1"), ("H", "h1"), ("H", "h1"), ("H", "h1"), ("H", "h1"), ("H", "h1"), ("H", "h1")]),
        # nq
        ("n_methylazetidine", "CN1CCC1", 0, [("C", "c3"), ("N", "nq"), ("C", "cy"), ("C", "cy"), ("C", "cy"), ("H", "h1"), ("H", "h1"), ("H", "h1"), ("H", "h1"), ("H", "h1"), ("H", "hc"), ("H", "hc"), ("H", "h1"), ("H", "h1")]),
        # nm
        ("n_phenylaziridine", "c1ccc(cc1)N1CC1", 0, [("C", "ca"), ("C", "ca"), ("C", "ca"), ("C", "ca"), ("C", "ca"), ("C", "ca"), ("N", "nm"), ("C", "cx"), ("C", "cx"), ("H", "ha"), ("H", "ha"), ("H", "ha"), ("H", "ha"), ("H", "ha"), ("H", "h1"), ("H", "h1"), ("H", "h1"), ("H", "h1")]),
        # nn
        ("n_phenylazetidine", "c1ccc(cc1)N1CCC1", 0, [("C", "ca"), ("C", "ca"), ("C", "ca"), ("C", "ca"), ("C", "ca"), ("C", "ca"), ("N", "nn"), ("C", "cy"), ("C", "cy"), ("C", "cy"), ("H", "ha"), ("H", "ha"), ("H", "ha"), ("H", "ha"), ("H", "ha"), ("H", "h1"), ("H", "h1"), ("H", "hc"), ("H", "hc"), ("H", "h1"), ("H", "h1")]),
        # no
        ("nitromethane", "C[N+](=O)[O-]", 0, [("C", "c3"), ("N", "no"), ("O", "o"), ("O", "o"), ("H", "h1"), ("H", "h1"), ("H", "h1")]),
        # ny
        ("dimethylammonium_cation", "C[NH2+]C", 0, [("C", "c3"), ("N", "ny"), ("C", "c3"), ("H", "hx"), ("H", "hx"), ("H", "hx"), ("H", "hn"), ("H", "hn"), ("H", "hx"), ("H", "hx"), ("H", "hx")]),
        # n7
        ("dimethylamine", "CNC", 0, [("C", "c3"), ("N", "n7"), ("C", "c3"), ("H", "h1"), ("H", "h1"), ("H", "h1"), ("H", "hn"), ("H", "h1"), ("H", "h1"), ("H", "h1")]),
        # n9
        ("ammonia", "N", 0, [("N", "n9"), ("H", "hn"), ("H", "hn"), ("H", "hn")]),
        # n+
        ("ammonium_cation", "[NH4+]", 1, [("N", "n+"), ("H", "hn"), ("H", "hn"), ("H", "hn"), ("H", "hn")]),
        # cu
        ("methylenecyclopropane", "C=C1CC1", 0, [("C", "c2"), ("C", "cu"), ("C", "cx"), ("C", "cx"), ("H", "ha"), ("H", "ha"), ("H", "hc"), ("H", "hc"), ("H", "hc"), ("H", "hc")]),
        # cv
        ("methylenecyclobutane", "C=C1CCC1", 0, [("C", "c2"), ("C", "cv"), ("C", "cy"), ("C", "cy"), ("C", "cy"), ("H", "ha"), ("H", "ha"), ("H", "hc"), ("H", "hc"), ("H", "hc"), ("H", "hc"), ("H", "hc"), ("H", "hc")]),
        # n
        ("n_n_dimethylacetamide", "CC(=O)N(C)C", 0, [("C", "c3"), ("C", "c"), ("O", "o"), ("N", "n"), ("C", "c3"), ("C", "c3"), ("H", "hc"), ("H", "hc"), ("H", "hc"), ("H", "h1"), ("H", "h1"), ("H", "h1"), ("H", "h1"), ("H", "h1"), ("H", "h1")]),
        # nu
        ("n_methylaniline", "CNc1ccccc1", 0, [("C", "c3"), ("N", "nu"), ("C", "ca"), ("C", "ca"), ("C", "ca"), ("C", "ca"), ("C", "ca"), ("C", "ca"), ("H", "h1"), ("H", "h1"), ("H", "h1"), ("H", "hn"), ("H", "ha"), ("H", "ha"), ("H", "ha"), ("H", "ha"), ("H", "ha")]),
        # oq
        ("oxetane", "C1CCO1", 0, [("C", "cy"), ("C", "cy"), ("C", "cy"), ("O", "oq"), ("H", "h1"), ("H", "h1"), ("H", "hc"), ("H", "hc"), ("H", "h1"), ("H", "h1")]),
        # s4
        ("dmso", "CS(=O)C", 0, [("C", "c3"), ("S", "s4"), ("O", "o"), ("C", "c3"), ("H", "h1"), ("H", "h1"), ("H", "h1"), ("H", "h1"), ("H", "h1"), ("H", "h1")]),
        # s6
        ("dimethyl_sulfone", "CS(=O)(=O)C", 0, [("C", "c3"), ("S", "s6"), ("O", "o"), ("O", "o"), ("C", "c3"), ("H", "h1"), ("H", "h1"), ("H", "h1"), ("H", "h1"), ("H", "h1"), ("H", "h1")]),
        # sp
        ("thiirane", "C1CS1", 0, [("C", "cx"), ("C", "cx"), ("S", "sp"), ("H", "h1"), ("H", "h1"), ("H", "h1"), ("H", "h1")]),
        # sq
        ("thietane", "C1CCS1", 0, [("C", "cy"), ("C", "cy"), ("C", "cy"), ("S", "sq"), ("H", "h1"), ("H", "h1"), ("H", "hc"), ("H", "hc"), ("H", "h1"), ("H", "h1")]),
        # p2
        ("phosphaacetylene", "C#P", 0, [("C", "c1"), ("P", "p2"), ("H", "ha")]),
        # p5
        ("tetramethylphosphonium_cation", "C[P+](C)(C)C", 1, [("C", "c3"), ("P", "p5"), ("C", "c3"), ("C", "c3"), ("C", "c3"), ("H", "hc"), ("H", "hc"), ("H", "hc"), ("H", "hc"), ("H", "hc"), ("H", "hc"), ("H", "hc"), ("H", "hc"), ("H", "hc"), ("H", "hc"), ("H", "hc"), ("H", "hc")]),
        # pb
        ("phosphinine", "c1ccpcc1", 0, [("C", "ca"), ("C", "ca"), ("C", "ca"), ("P", "pb"), ("C", "ca"), ("C", "ca"), ("H", "ha"), ("H", "ha"), ("H", "ha"), ("H", "ha"), ("H", "ha")]),
        # nc
        ("triazole", "c1nc[nH]n1", 0, [("C", "cc"), ("N", "nc"), ("C", "cd"), ("N", "na"), ("N", "nd"), ("H", "h5"), ("H", "h5"), ("H", "hn")]),
        # s2
        ("methylenemethylsulfonium_cation", "C[S+]=C", 1, [("C", "c3"), ("S", "s2"), ("C", "c2"), ("H", "h1"), ("H", "h1"), ("H", "h1"), ("H", "h4"), ("H", "h4")]),
        # nh
        ("n_n_dimethylaniline", "CN(C)c1ccccc1", 0, [("C", "c3"), ("N", "nh"), ("C", "c3"), ("C", "ca"), ("C", "ca"), ("C", "ca"), ("C", "ca"), ("C", "ca"), ("C", "ca"), ("H", "h1"), ("H", "h1"), ("H", "h1"), ("H", "h1"), ("H", "h1"), ("H", "h1"), ("H", "ha"), ("H", "ha"), ("H", "ha"), ("H", "ha"), ("H", "ha")]),
        # nj
        ("n_methyl_beta_lactam", "O=C1CCN1C", 0, [("O", "o"), ("C", "c"), ("C", "cy"), ("C", "cy"), ("N", "nj"), ("C", "c3"), ("H", "hc"), ("H", "hc"), ("H", "h1"), ("H", "h1"), ("H", "h1"), ("H", "h1"), ("H", "h1")]),
        # nk
        ("n_n_dimethylaziridinium_cation", "C[N+]1(C)CC1", 1, [("C", "c3"), ("N", "nk"), ("C", "c3"), ("C", "cx"), ("C", "cx"), ("H", "hx"), ("H", "hx"), ("H", "hx"), ("H", "hx"), ("H", "hx"), ("H", "hx"), ("H", "hx"), ("H", "hx"), ("H", "hx"), ("H", "hx")]),
        # nl
        ("n_n_dimethylazetidinium_cation", "C[N+]1(C)CCC1", 1, [("C", "c3"), ("N", "nl"), ("C", "c3"), ("C", "cy"), ("C", "cy"), ("C", "cy"), ("H", "hx"), ("H", "hx"), ("H", "hx"), ("H", "hx"), ("H", "hx"), ("H", "hx"), ("H", "hx"), ("H", "hx"), ("H", "hc"), ("H", "hc"), ("H", "hx"), ("H", "hx")]),
        # sx
        ("thiophene_1_oxide", "O=S1C=CC=C1", 0, [("O", "o"), ("S", "sx"), ("C", "ce"), ("C", "cf"), ("C", "cf"), ("C", "ce"), ("H", "h4"), ("H", "ha"), ("H", "ha"), ("H", "h4")]),
        # pc
        ("phosphole", "p1cc[nH]c1", 0, [("P", "pc"), ("C", "cc"), ("C", "cd"), ("N", "na"), ("C", "cd"), ("H", "ha"), ("H", "h4"), ("H", "hn"), ("H", "h4")]),
        # ne
        ("aza_diene_internal", "C=NC=C", 0, [("C", "c2"), ("N", "ne"), ("C", "ce"), ("C", "c2"), ("H", "h4"), ("H", "h4"), ("H", "h4"), ("H", "ha"), ("H", "ha")]),
        # pe
        ("phospha_diene_internal", "C=PC=C", 0, [("C", "c2"), ("P", "pe"), ("C", "ce"), ("C", "c2"), ("H", "ha"), ("H", "ha"), ("H", "ha"), ("H", "ha"), ("H", "ha")]),
        # sy
        ("thiophene_1_1_dioxide", "O=S1(=O)C=CC=C1", 0, [("O", "o"), ("S", "sy"), ("O", "o"), ("C", "ce"), ("C", "cf"), ("C", "cf"), ("C", "ce"), ("H", "h4"), ("H", "ha"), ("H", "ha"), ("H", "h4")]),
        # py
        ("phosphole_oxide", "O=P1C=CC=C1", 0, [("O", "o"), ("P", "py"), ("C", "ce"), ("C", "cf"), ("C", "cf"), ("C", "ce"), ("H", "hp"), ("H", "ha"), ("H", "ha"), ("H", "ha"), ("H", "ha")]),
    ])
    def test_full_rule_coverage(
        self, label: str, smiles: str, net_charge: int, expected_types: list
    ) -> None:
        # net_charge is unused here -- RDKit reads formal charge directly from
        # the SMILES bracket notation ([N+], [NH+], etc). It's carried in the
        # parametrize tuple only because the antechamber verification pass
        # that produced expected_types needed it (antechamber's -nc flag).
        del net_charge
        from proxide.chem.gaff2 import assign_gaff2_atom_types

        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, f"{label}: failed to parse SMILES {smiles!r}"
        mol = Chem.AddHs(mol)
        AllChem.SanitizeMol(mol)

        types = assign_gaff2_atom_types(mol)
        actual = list(zip((a.GetSymbol() for a in mol.GetAtoms()), types, strict=True))
        assert actual == expected_types, (
            f"{label} ({smiles}): expected {expected_types}, got {actual}"
        )

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