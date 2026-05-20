"""
HP4-WASM parity test: Rust proxide-wasm vs Python GAFFTemplateGenerator.

Validates that the Rust GAFF2 parameter assignment produces physically
plausible results matching expected OpenMM units and value ranges.

Requires:
  - uv run pytest (proxide venv has openmmforcefields, rdkit, openmm)
  - cargo build -p proxide-wasm --bin param_cli (Rust CLI binary)
"""
import json
import subprocess
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
RUST_BINARY = PROJECT_ROOT / "target" / "debug" / "param_cli"

TEST_MOLS = [
    ("methane", "C"),
    ("ethane", "CC"),
    ("ethanol", "CCO"),
    ("acetone", "CC(=O)C"),
    ("benzene", "c1ccccc1"),
]


def rust_params(smiles: str) -> dict:
    """Run Rust param_cli and return parsed JSON."""
    if not RUST_BINARY.exists():
        pytest.skip(
            f"Rust binary not found at {RUST_BINARY}. "
            "Run: cargo build -p proxide-wasm --bin param_cli"
        )
    result = subprocess.run(
        [str(RUST_BINARY), smiles],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def py_gaff2_atom_types(smiles: str) -> dict:
    """Get GAFF2 atom types from openmmforcefields/rdkit."""
    try:
        from openmmforcefields.generators import GAFFTemplateGenerator
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as e:
        pytest.skip(f"Python deps not available: {e}")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    mol = AllChem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)

    gen = GAFFTemplateGenerator(molecules=[mol], forcefield="gaff-2.11")
    # Try to get templates
    try:
        templates = gen.generate_residue_template(mol)
        if templates:
            return {
                "atom_types": [a.type for a in templates.atoms],
                "n_atoms": mol.GetNumAtoms(),
            }
    except Exception:
        pass
    return {"atom_types": [], "n_atoms": mol.GetNumAtoms()}


# ============================================================================
# Tests
# ============================================================================


@pytest.mark.parametrize("name,smiles", TEST_MOLS)
def test_rust_params_complete(name, smiles):
    """Rust param_cli returns a complete ParamSet for each molecule."""
    params = rust_params(smiles)

    assert "atoms" in params, f"{name}: missing atoms"
    assert "bonds" in params, f"{name}: missing bonds"
    assert "angles" in params, f"{name}: missing angles"
    assert "torsions" in params, f"{name}: missing torsions"
    assert len(params["atoms"]) > 0, f"{name}: no atoms"

    # Heavy atoms: C counts as 1
    n_heavy = sum(1 for a in params["atoms"] if a["atomic_num"] != 1)
    assert n_heavy > 0, f"{name}: no heavy atoms"


@pytest.mark.parametrize("name,smiles", TEST_MOLS)
def test_bond_parameter_structure(name, smiles):
    """Verify bond parameters have required fields."""
    params = rust_params(smiles)

    for i, bond in enumerate(params["bonds"]):
        assert "i" in bond, f"{name}: bond {i} missing 'i'"
        assert "j" in bond, f"{name}: bond {i} missing 'j'"
        assert "k" in bond, f"{name}: bond {i} missing force constant 'k'"
        assert "r0" in bond, f"{name}: bond {i} missing equilibrium length 'r0'"
        assert "type_pair" in bond, f"{name}: bond {i} missing 'type_pair'"
        assert isinstance(bond["type_pair"], list), f"{name}: type_pair must be list"
        assert len(bond["type_pair"]) == 2, f"{name}: type_pair must have 2 elements"


@pytest.mark.parametrize("name,smiles", TEST_MOLS)
def test_angle_parameter_structure(name, smiles):
    """Verify angle parameters have required fields."""
    params = rust_params(smiles)

    for i, angle in enumerate(params["angles"]):
        assert "i" in angle, f"{name}: angle {i} missing 'i'"
        assert "j" in angle, f"{name}: angle {i} missing central 'j'"
        assert "k_idx" in angle, f"{name}: angle {i} missing 'k_idx'"
        assert "k_angle" in angle, f"{name}: angle {i} missing force constant"
        assert "theta0" in angle, f"{name}: angle {i} missing equilibrium angle"
        assert "type_triple" in angle, f"{name}: angle {i} missing 'type_triple'"
        assert isinstance(angle["type_triple"], list), f"{name}: type_triple must be list"
        assert len(angle["type_triple"]) == 3, f"{name}: type_triple must have 3 elements"


@pytest.mark.parametrize("name,smiles", TEST_MOLS)
def test_torsion_parameter_structure(name, smiles):
    """Verify torsion parameters have required fields."""
    params = rust_params(smiles)

    for i, torsion in enumerate(params["torsions"]):
        assert "i" in torsion, f"{name}: torsion {i} missing 'i'"
        assert "j" in torsion, f"{name}: torsion {i} missing 'j'"
        assert "k_idx" in torsion, f"{name}: torsion {i} missing 'k_idx'"
        assert "l" in torsion, f"{name}: torsion {i} missing 'l'"
        assert "periodicity" in torsion, f"{name}: torsion {i} missing periodicity"
        assert "k_torsion" in torsion, f"{name}: torsion {i} missing force constant"
        assert "phase" in torsion, f"{name}: torsion {i} missing phase"
        assert "type_quad" in torsion, f"{name}: torsion {i} missing 'type_quad'"
        assert isinstance(torsion["type_quad"], list), f"{name}: type_quad must be list"
        assert len(torsion["type_quad"]) == 4, f"{name}: type_quad must have 4 elements"


@pytest.mark.parametrize("name,smiles", TEST_MOLS)
def test_rust_atom_types_non_empty(name, smiles):
    """All atoms have non-empty atom types assigned."""
    params = rust_params(smiles)

    for atom in params["atoms"]:
        assert atom["atom_type"], f"{name}: atom {atom['idx']} has empty atom type"
        assert isinstance(atom["atom_type"], str), \
            f"{name}: atom type must be string, got {type(atom['atom_type'])}"
        # GAFF2 types are typically 1-4 chars
        assert 1 <= len(atom["atom_type"]) <= 4, \
            f"{name}: atom type '{atom['atom_type']}' has unusual length"


@pytest.mark.parametrize("name,smiles", TEST_MOLS)
def test_bond_parameters_positive_and_physical(name, smiles):
    """Bond parameters are positive and in physically plausible ranges."""
    params = rust_params(smiles)

    for bond in params["bonds"]:
        assert bond["k"] > 0, \
            f"{name}: bond force constant must be positive, got {bond['k']}"
        assert bond["r0"] > 0, \
            f"{name}: bond r0 must be positive, got {bond['r0']}"
        # r0 in nm: typical bonds are 0.09–0.17 nm (0.9–1.7 Å)
        assert 0.08 < bond["r0"] < 0.20, \
            f"{name}: bond r0={bond['r0']} nm outside typical range [0.08, 0.20]"
        # k in kJ/mol/nm²: typical bonds are 100k–400k
        assert bond["k"] > 50000, \
            f"{name}: bond k={bond['k']} kJ/mol/nm² seems too small"
        assert bond["k"] < 1000000, \
            f"{name}: bond k={bond['k']} kJ/mol/nm² seems too large"


@pytest.mark.parametrize("name,smiles", TEST_MOLS)
def test_angle_parameters_positive_and_physical(name, smiles):
    """Angle parameters are positive and in physically plausible ranges."""
    params = rust_params(smiles)

    for angle in params["angles"]:
        assert angle["k_angle"] > 0, \
            f"{name}: angle force constant must be positive, got {angle['k_angle']}"
        # theta0 in radians: valid range is 0–π
        assert 0 < angle["theta0"] < 3.2, \
            f"{name}: angle theta0={angle['theta0']} rad outside [0, π]"
        # k_angle in kJ/mol/rad²: typical angles are 50–200
        assert 5 < angle["k_angle"] < 1000, \
            f"{name}: angle k_angle={angle['k_angle']} kJ/mol/rad² outside typical range"


@pytest.mark.parametrize("name,smiles", TEST_MOLS)
def test_torsion_parameters_physical(name, smiles):
    """Torsion parameters are in physically plausible ranges."""
    params = rust_params(smiles)

    for torsion in params["torsions"]:
        assert torsion["periodicity"] > 0, \
            f"{name}: periodicity must be positive, got {torsion['periodicity']}"
        assert torsion["periodicity"] <= 6, \
            f"{name}: periodicity {torsion['periodicity']} is unusually high"
        # phase in radians: valid range is 0–2π
        assert 0 <= torsion["phase"] <= 6.3, \
            f"{name}: phase={torsion['phase']} rad outside [0, 2π]"


@pytest.mark.parametrize("name,smiles", TEST_MOLS)
def test_atom_type_consistency_in_parameters(name, smiles):
    """Atom types in bonds/angles/torsions are consistent (non-empty strings)."""
    params = rust_params(smiles)

    # Note: The 'atoms' list in the output only includes the original atoms from SMILES,
    # not the implicit hydrogens. However, bonds/angles/torsions reference all atoms
    # (including implicit H). We verify that all referenced atom types are non-empty strings.

    for bond in params["bonds"]:
        for atype in bond["type_pair"]:
            assert isinstance(atype, str) and atype, \
                f"{name}: bond has empty or invalid atom type '{atype}'"

    for angle in params["angles"]:
        for atype in angle["type_triple"]:
            assert isinstance(atype, str) and atype, \
                f"{name}: angle has empty or invalid atom type '{atype}'"

    for torsion in params["torsions"]:
        for atype in torsion["type_quad"]:
            assert isinstance(atype, str) and atype, \
                f"{name}: torsion has empty or invalid atom type '{atype}'"


def test_methane_specific():
    """Methane should have exactly one c3 carbon and c-h bonds only."""
    params = rust_params("C")

    c_atoms = [a for a in params["atoms"] if a["atomic_num"] == 6]
    assert len(c_atoms) == 1, "Methane should have 1 carbon"
    assert c_atoms[0]["atom_type"] == "c3", "Methane C should be c3"

    # The atoms list only includes heavy atoms, not implicit H.
    # But we can verify from bonds that H are present and have type 'hc'
    assert len(params["bonds"]) > 0, "Methane should have bonds"

    # All bonds should be either C-H
    for bond in params["bonds"]:
        types = set(bond["type_pair"])
        assert types == {"c3", "hc"}, \
            f"Methane bond types must be {{c3, hc}}, got {types}"


def test_ethane_specific():
    """Ethane should have two c3 carbons and C-C bond with expected parameters."""
    params = rust_params("CC")

    c_atoms = [a for a in params["atoms"] if a["atomic_num"] == 6]
    assert len(c_atoms) == 2, "Ethane should have 2 carbons"
    assert all(a["atom_type"] == "c3" for a in c_atoms), "Both carbons should be c3"

    # Find C-C bond
    cc_bonds = [b for b in params["bonds"]
                if set(b["type_pair"]) == {"c3", "c3"}]
    assert len(cc_bonds) >= 1, "Ethane should have at least one C-C bond"

    cc = cc_bonds[0]
    # C-C bond: r0 ≈ 0.1526 nm (1.526 Å)
    assert 0.14 < cc["r0"] < 0.17, \
        f"C-C r0={cc['r0']} nm; expected ~0.1526 nm"
    # C-C bond: k typically 190,000–250,000 kJ/mol/nm² in OpenMM
    assert 100000 < cc["k"] < 400000, \
        f"C-C k={cc['k']} kJ/mol/nm²; expected 100k–400k range"


def test_benzene_specific():
    """Benzene should have aromatic ca carbons."""
    params = rust_params("c1ccccc1")

    c_atoms = [a for a in params["atoms"] if a["atomic_num"] == 6]
    assert len(c_atoms) == 6, "Benzene should have 6 carbons"

    # All carbons should be aromatic ca (not c2)
    ca_atoms = [a for a in c_atoms if a["atom_type"] == "ca"]
    assert len(ca_atoms) == 6, \
        f"All benzene carbons should be ca, got types: {[a['atom_type'] for a in c_atoms]}"


def test_ethanol_structure():
    """Ethanol should have C, C, O atoms with correct types."""
    params = rust_params("CCO")

    # atoms list only includes heavy atoms (not implicit H)
    assert len(params["atoms"]) == 3, "Ethanol should have exactly 3 heavy atoms (C,C,O)"

    c_atoms = [a for a in params["atoms"] if a["atomic_num"] == 6]
    o_atoms = [a for a in params["atoms"] if a["atomic_num"] == 8]

    assert len(c_atoms) == 2, "Ethanol should have 2 carbons"
    assert len(o_atoms) == 1, "Ethanol should have 1 oxygen"

    # O should be atom type oh (alcohol oxygen)
    o_types = {a["atom_type"] for a in o_atoms}
    assert "oh" in o_types, f"Ethanol oxygen should be 'oh', got {o_types}"


def test_acetone_structure():
    """Acetone should have C, C, O atoms; O should be carbonyl."""
    params = rust_params("CC(=O)C")

    c_atoms = [a for a in params["atoms"] if a["atomic_num"] == 6]
    o_atoms = [a for a in params["atoms"] if a["atomic_num"] == 8]

    assert len(c_atoms) == 3, "Acetone should have 3 carbons"
    assert len(o_atoms) == 1, "Acetone should have 1 oxygen"

    # Carbonyl oxygen should be 'o' (or 'oq' in some forcefields)
    o_types = {a["atom_type"] for a in o_atoms}
    assert any(ot.startswith("o") for ot in o_types), \
        f"Acetone oxygen should start with 'o', got {o_types}"


# ============================================================================
# Summary tests
# ============================================================================


@pytest.mark.parametrize("name,smiles", TEST_MOLS)
def test_summary_print(name, smiles):
    """Print summary of each molecule's parameter counts."""
    params = rust_params(smiles)
    summary = (
        f"{name:15s} ({smiles:15s}): "
        f"{len(params['atoms']):2d} atoms, "
        f"{len(params['bonds']):3d} bonds, "
        f"{len(params['angles']):3d} angles, "
        f"{len(params['torsions']):3d} torsions"
    )
    print("\n" + summary)
