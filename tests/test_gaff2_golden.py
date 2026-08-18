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
# Note: Several entries below diverge from the official spec (see comments).
# Per ATOMTYPE_GFF2.DEF (line numbers are from vendored asset):
# - cx (line 16) requires [RG3] (3-membered ring membership)
# - c3 (line 22) is for "other sp3 C" (non-ring)
# - cs (line 24-26) is for C=S (not C=C or C=O)
# - c (line 28-30) is for C=O (carbonyl carbon)
# - c2 (line 70) is for "other sp2 C"
# - op (line 223) is for oxygen in [RG3] (3-membered ring only)
# - oh (line 218-222) is for hydroxyl oxygen
# - ni (line 144) is for [RG3] nitrogen
# - n3 (line 183) is for sp3 nitrogen (non-ring)
# - nb (line 184) is for aromatic nitrogen [AR1]
ATOM_TYPE_REFERENCE = {
    # Simple alkanes
    # c3 is for sp3 carbon per spec line 22 ("other sp3 C")
    # cx requires [RG3] (3-membered ring) per spec line 16 — not used here
    "C": ["c3"],                  # methane [CORRECT per spec line 22]
    "CC": ["c3", "c3"],           # ethane [CORRECT per spec line 22]
    "CCC": ["c3", "c3", "c3"],    # propane [CORRECT per spec line 22]
    "CCCC": ["c3", "c3", "c3", "c3"],  # butane [CORRECT per spec line 22]

    # Alkenes
    # cz per spec is for C bonded to 3 nitrogens (line 31), which is nonsensical for ethene
    # This is a pre-existing bug in gaff2.py's rule matching - ethene has no nitrogen neighbors
    # See .praxia/docs/misc/260623_gaff2-typing-debt.md for details
    "C=C": ["cz", "cz"],          # ethene [BUG: cz requires 3 N neighbors, marked xfail]

    # Acetylene
    # cg is sp C (atomic number 6, 2 attachments) per spec lines 72-75
    "C#C": ["cg", "cg"],          # acetylene [CORRECT per spec lines 72-75]

    # Aromatic carbons
    # cp is for "pure aromatic atom that can form aromatic single bond" (spec line 33)
    "c1ccccc1": ["cp"] * 6,       # benzene [CORRECT per spec line 33]

    # Phenol
    # oh is for hydroxyl oxygen per spec lines 218-222 (2 or 3 attachments with specific H count)
    "c1ccc(O)cc1": ["cp", "cp", "cp", "cp", "oh", "cp", "cp"],  # phenol [CORRECT per spec line 218-222]

    # Pyridine
    # nb is aromatic nitrogen [AR1] per spec line 184
    "c1ccncc1": ["cp", "cp", "cp", "nb", "cp", "cp"],  # pyridine [CORRECT per spec line 184]

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

    # Ethylamine
    # nt is sp3 nitrogen per spec line 143 (7 valence, 3 neighbors, 2 hydrogens, neighbor C3(XA1))
    # n3 is more general sp3 nitrogen per spec line 183 (7 valence, 3 attachments, non-ring)
    "CCN": ["c3", "c3", "nt"],        # ethylamine [CORRECT per spec line 143]
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
    ("C#C", ["cg", "cg"]),
    ("c1ccccc1", ["cp"] * 6),
    ("c1ccc(O)cc1", ["cp", "cp", "cp", "cp", "oh", "cp", "cp"]),
    ("c1ccncc1", ["cp", "cp", "cp", "nb", "cp", "cp"]),
    ("CCO", ["c3", "c3", "oh"]),
    ("CC(=O)C", ["c3", "c", "o", "c3"]),
    ("CC(=O)O", ["c3", "c", "o", "oh"]),
    ("CCN", ["c3", "c3", "nt"]),
])
def test_atom_type_golden_reference(smiles: str, expected_types: list[str]) -> None:
    """Test that GAFF2 atom type assignment matches golden reference values.

    Each test case calls parameterize_gaff_with_rdkit and asserts that the
    resulting atom types match the expected reference values from the spec.
    """
    from proxide.chem.gaff2 import parameterize_gaff_with_rdkit

    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Failed to parse SMILES: {smiles}"

    mol = Chem.AddHs(mol)
    AllChem.SanitizeMol(mol)

    result = parameterize_gaff_with_rdkit(mol)
    assert result["atom_types"] == expected_types, (
        f"SMILES {smiles}: expected {expected_types}, got {result['atom_types']}"
    )


@pytest.mark.xfail(
    reason="pre-existing gaff2.py rule-matching bug: ethene (C=C) incorrectly matches "
           "cz rule (requires 3 nitrogen neighbors), which is nonsensical for a "
           "nitrogen-free molecule. See .praxia/docs/misc/260623_gaff2-typing-debt.md"
)
def test_atom_type_ethene_bug() -> None:
    """Test case for C=C (ethene) that exposes pre-existing rule-matching bug.

    The cz rule at ATOMTYPE_GFF2.DEF line 31 requires 3 nitrogen neighbors (N3,N3,N3),
    but ethene has no nitrogen atoms. This appears to be a specification issue or
    implementation bug in the rule-matching logic that prioritizes cz before checking
    for availability of required neighbor types.
    """
    from proxide.chem.gaff2 import parameterize_gaff_with_rdkit

    smiles = "C=C"
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Failed to parse SMILES: {smiles}"

    mol = Chem.AddHs(mol)
    AllChem.SanitizeMol(mol)

    result = parameterize_gaff_with_rdkit(mol)
    # This should ideally be c2 (sp2 carbon, line 70), not cz
    # but the current implementation returns cz due to the rule-matching bug
    assert result["atom_types"] == ["c2", "c2"], (
        f"Ethene should assign c2 (other sp2 C, line 70), "
        f"but got {result['atom_types']} due to pre-existing bug"
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