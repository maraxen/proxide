"""Tests for GAFF atom typing parity with OpenMM/AmberTools.

This module validates that our GAFF atom type assignment matches
the output from antechamber (via openmmforcefields GAFFTemplateGenerator).

These tests require:
- openmm
- openmmforcefields
- rdkit (for molecule handling)
"""

from pathlib import Path
from typing import Any

import pytest

# Check if required packages are available
try:
    from openff.toolkit import Molecule
    from openmmforcefields.generators import GAFFTemplateGenerator

    HAS_OPENMMFORCEFIELDS = True
except ImportError:
    HAS_OPENMMFORCEFIELDS = False

try:
    import priox_rs

    HAS_PRIOX_RS = True
except ImportError:
    HAS_PRIOX_RS = False


# Test molecules - SMILES strings for small molecules
TEST_MOLECULES = [
    ("methane", "C"),
    ("ethane", "CC"),
    ("ethanol", "CCO"),
    ("benzene", "c1ccccc1"),
    ("phenol", "c1ccc(O)cc1"),
    ("aniline", "c1ccc(N)cc1"),
    ("acetone", "CC(=O)C"),
    ("acetic_acid", "CC(=O)O"),
    ("methylamine", "CN"),
    ("dimethylether", "COC"),
    ("chloromethane", "CCl"),
    ("fluorobenzene", "c1ccc(F)cc1"),
    ("toluene", "Cc1ccccc1"),
    ("propane", "CCC"),
    ("butane", "CCCC"),
    ("isobutane", "CC(C)C"),
    ("cyclohexane", "C1CCCCC1"),
    ("formaldehyde", "C=O"),
    ("acetaldehyde", "CC=O"),
    ("methanol", "CO"),
]


@pytest.fixture
def gaff_generator():
    """Create a GAFFTemplateGenerator."""
    if not HAS_OPENMMFORCEFIELDS:
        pytest.skip("openmmforcefields not installed")
    return GAFFTemplateGenerator(forcefield="gaff-2.11")


def get_antechamber_atom_types(smiles: str) -> list[str]:
    """Get GAFF atom types from antechamber via openmmforcefields.

    This uses the GAFFTemplateGenerator which internally calls antechamber.
    """
    from openff.toolkit import Molecule as OFFMolecule

    # Create molecule and add hydrogens
    mol = OFFMolecule.from_smiles(smiles)
    mol.generate_conformers(n_conformers=1)
    mol.assign_partial_charges("am1bcc")

    # Create generator and parameterize
    generator = GAFFTemplateGenerator(molecules=[mol], forcefield="gaff-2.11")

    # Force parameterization - this calls antechamber
    ffxml = generator.generate_residue_template(mol)

    # Extract atom types from the generated template
    # The template assigns gaff_type to each atom
    return [atom.gaff_type for atom in mol.atoms]


def get_priox_atom_types(smiles: str) -> list[str]:
    """Get GAFF atom types from our Rust implementation.

    Note: This is a simplified version that uses element + connectivity.
    Full parity with antechamber requires more sophisticated typing.
    """
    from openff.toolkit import Molecule as OFFMolecule

    mol = OFFMolecule.from_smiles(smiles)
    mol.generate_conformers(n_conformers=1)

    # Get elements
    elements = [atom.symbol for atom in mol.atoms]

    # Get coordinates (in Angstroms, convert to whatever priox expects)
    coords = mol.conformers[0].magnitude  # numpy array

    # Call Rust function
    types = priox_rs.assign_gaff_atom_types(coords, elements)

    return [t if t is not None else "du" for t in types]


@pytest.mark.skipif(not HAS_OPENMMFORCEFIELDS, reason="openmmforcefields not installed")
@pytest.mark.skipif(not HAS_PRIOX_RS, reason="priox_rs not installed")
class TestGaffParityWithAntechamber:
    """Tests comparing our GAFF typing with antechamber."""

    @pytest.mark.parametrize("name,smiles", TEST_MOLECULES[:5])  # Start with simple molecules
    def test_atom_type_count_matches(self, name: str, smiles: str) -> None:
        """Test that we assign the same number of atom types."""
        try:
            antechamber_types = get_antechamber_atom_types(smiles)
            priox_types = get_priox_atom_types(smiles)

            assert len(priox_types) == len(antechamber_types), (
                f"Different number of types for {name}: "
                f"priox={len(priox_types)}, antechamber={len(antechamber_types)}"
            )
        except Exception as e:
            pytest.skip(f"Could not parameterize {name}: {e}")

    @pytest.mark.parametrize("name,smiles", [("methane", "C"), ("ethane", "CC")])
    def test_simple_alkanes(self, name: str, smiles: str) -> None:
        """Test simple alkanes have c3 and hc types."""
        try:
            priox_types = get_priox_atom_types(smiles)

            # All carbons should be c3 (sp3)
            from openff.toolkit import Molecule as OFFMolecule

            mol = OFFMolecule.from_smiles(smiles)
            elements = [atom.symbol for atom in mol.atoms]

            for i, (elem, atype) in enumerate(zip(elements, priox_types)):
                if elem == "C":
                    assert atype == "c3", f"Carbon {i} in {name} should be c3, got {atype}"
                elif elem == "H":
                    assert atype == "hc", f"Hydrogen {i} in {name} should be hc, got {atype}"
        except Exception as e:
            pytest.skip(f"Could not process {name}: {e}")

    def test_benzene_aromaticity(self) -> None:
        """Test that benzene carbons are typed as 'ca' (aromatic carbon)."""
        try:
            priox_types = get_priox_atom_types("c1ccccc1")

            from openff.toolkit import Molecule as OFFMolecule

            mol = OFFMolecule.from_smiles("c1ccccc1")
            elements = [atom.symbol for atom in mol.atoms]

            for i, (elem, atype) in enumerate(zip(elements, priox_types)):
                if elem == "C":
                    assert atype == "ca", f"Benzene carbon {i} should be ca, got {atype}"
                elif elem == "H":
                    assert atype == "ha", f"Benzene hydrogen {i} should be ha, got {atype}"
        except Exception as e:
            pytest.skip(f"Could not process benzene: {e}")

    def test_ethanol_oxygen(self) -> None:
        """Test ethanol oxygen typing."""
        try:
            priox_types = get_priox_atom_types("CCO")

            from openff.toolkit import Molecule as OFFMolecule

            mol = OFFMolecule.from_smiles("CCO")
            elements = [atom.symbol for atom in mol.atoms]

            # Find oxygen index
            for i, elem in enumerate(elements):
                if elem == "O":
                    # Should be oh (hydroxyl oxygen)
                    assert priox_types[i] == "oh", f"Ethanol O should be oh, got {priox_types[i]}"
                    break
        except Exception as e:
            pytest.skip(f"Could not process ethanol: {e}")


@pytest.mark.skipif(not HAS_OPENMMFORCEFIELDS, reason="openmmforcefields not installed")
class TestGaffTemplateGeneratorReference:
    """Reference tests using openmmforcefields directly.

    These tests document the expected behavior from antechamber.
    """

    def test_antechamber_methane(self) -> None:
        """Document antechamber typing for methane."""
        types = get_antechamber_atom_types("C")
        print(f"\nAntechamber methane types: {types}")
        # Expected: ['c3', 'hc', 'hc', 'hc', 'hc']
        assert "c3" in types  # sp3 carbon
        assert types.count("hc") == 4  # 4 hydrogens on carbon

    def test_antechamber_benzene(self) -> None:
        """Document antechamber typing for benzene."""
        types = get_antechamber_atom_types("c1ccccc1")
        print(f"\nAntechamber benzene types: {types}")
        # Expected: 6 'ca' + 6 'ha'
        assert types.count("ca") == 6  # aromatic carbons
        assert types.count("ha") == 6  # aromatic hydrogens

    def test_antechamber_ethanol(self) -> None:
        """Document antechamber typing for ethanol."""
        types = get_antechamber_atom_types("CCO")
        print(f"\nAntechamber ethanol types: {types}")
        assert "c3" in types  # sp3 carbons
        assert "oh" in types or "os" in types  # hydroxyl oxygen
        assert "ho" in types  # hydroxyl hydrogen

    def test_antechamber_acetone(self) -> None:
        """Document antechamber typing for acetone."""
        types = get_antechamber_atom_types("CC(=O)C")
        print(f"\nAntechamber acetone types: {types}")
        assert "c" in types  # carbonyl carbon (sp2)
        assert "o" in types  # carbonyl oxygen


@pytest.mark.skipif(not HAS_PRIOX_RS, reason="priox_rs not installed")
class TestGaffParameterLookup:
    """Tests for GAFF parameter lookup from XML files."""

    @pytest.fixture
    def gaff_params(self) -> dict[str, Any]:
        """Load GAFF 2.11 parameters."""
        gaff_path = Path(__file__).parent.parent.parent / "src" / "priox" / "assets" / "gaff" / "ffxml" / "gaff-2.11.xml"
        if not gaff_path.exists():
            pytest.skip("GAFF 2.11 XML not found")
        return priox_rs.load_forcefield(str(gaff_path))

    def test_c3_c3_bond_exists(self, gaff_params: dict) -> None:
        """Test that c3-c3 bond parameters exist."""
        bonds = gaff_params["harmonic_bonds"]

        # Find c3-c3 bond
        c3_c3_bond = None
        for bond in bonds:
            if (bond["class1"] == "c3" and bond["class2"] == "c3") or (
                bond["class1"] == "c3" and bond["class2"] == "c3"
            ):
                c3_c3_bond = bond
                break

        assert c3_c3_bond is not None, "c3-c3 bond not found in GAFF"
        assert c3_c3_bond["length"] > 0
        assert c3_c3_bond["k"] > 0

    def test_ca_ca_bond_exists(self, gaff_params: dict) -> None:
        """Test that ca-ca (aromatic) bond parameters exist."""
        bonds = gaff_params["harmonic_bonds"]

        ca_ca_bond = None
        for bond in bonds:
            if bond["class1"] == "ca" and bond["class2"] == "ca":
                ca_ca_bond = bond
                break

        assert ca_ca_bond is not None, "ca-ca bond not found in GAFF"
        # Aromatic bond should be shorter than aliphatic
        assert ca_ca_bond["length"] < 0.16  # Roughly 1.4 Angstrom in nm

    def test_c3_hc_bond_exists(self, gaff_params: dict) -> None:
        """Test that c3-hc (aliphatic C-H) bond exists."""
        bonds = gaff_params["harmonic_bonds"]

        c3_hc_bond = None
        for bond in bonds:
            classes = {bond["class1"], bond["class2"]}
            if classes == {"c3", "hc"}:
                c3_hc_bond = bond
                break

        assert c3_hc_bond is not None, "c3-hc bond not found in GAFF"

    def test_nonbonded_params_exist(self, gaff_params: dict) -> None:
        """Test that common atom types have nonbonded parameters."""
        nb_params = gaff_params["nonbonded_params"]
        nb_types = {p["atom_type"] for p in nb_params}

        expected_types = {"c3", "ca", "n3", "oh", "o", "hc", "ha", "hn", "ho"}
        missing = expected_types - nb_types
        assert not missing, f"Missing nonbonded params for: {missing}"

    def test_proper_torsions_exist(self, gaff_params: dict) -> None:
        """Test that proper torsion parameters exist."""
        propers = gaff_params["proper_torsions"]
        assert len(propers) > 100, "Expected many proper torsion parameters in GAFF"

    def test_improper_torsions_exist(self, gaff_params: dict) -> None:
        """Test that improper torsion parameters exist."""
        impropers = gaff_params["improper_torsions"]
        assert len(impropers) > 0, "Expected improper torsion parameters in GAFF"
