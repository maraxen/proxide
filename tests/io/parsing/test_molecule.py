"""Tests for molecule parsing (MOL2, SDF formats)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from proxide.io.parsing.molecule import Molecule

# Test MOL2 content (simplified benzene-like structure)
BENZENE_MOL2 = """\
@<TRIPOS>MOLECULE
benzene
 6 6 0 0 0
SMALL
bcc


@<TRIPOS>ATOM
      1 C1          1.2124    0.7000    0.0000 ca        1 LIG     -0.115000
      2 C2          1.2124   -0.7000    0.0000 ca        1 LIG     -0.115000
      3 C3          0.0000   -1.4000    0.0000 ca        1 LIG     -0.115000
      4 C4         -1.2124   -0.7000    0.0000 ca        1 LIG     -0.115000
      5 C5         -1.2124    0.7000    0.0000 ca        1 LIG     -0.115000
      6 C6          0.0000    1.4000    0.0000 ca        1 LIG     -0.115000
@<TRIPOS>BOND
     1     1     2 ar
     2     2     3 ar
     3     3     4 ar
     4     4     5 ar
     5     5     6 ar
     6     6     1 ar
@<TRIPOS>SUBSTRUCTURE
     1 LIG         1 TEMP              0 ****  ****    0 ROOT
"""


# Test SDF content (simplified methane)
METHANE_SDF = """\
methane
  RDKit          3D

  5  4  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    0.0000    1.0900 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.0267    0.0000   -0.3633 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.5133   -0.8892   -0.3633 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.5133    0.8892   -0.3633 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  1  3  1  0
  1  4  1  0
  1  5  1  0
M  END
$$$$
"""


class TestMoleculeFromMol2:
    """Tests for Molecule.from_mol2()."""
    
    def test_parse_benzene_mol2(self):
        """Test parsing a simple benzene MOL2 file."""
        with tempfile.NamedTemporaryFile(suffix=".mol2", mode="w", delete=False) as f:
            f.write(BENZENE_MOL2)
            mol2_path = f.name
        
        try:
            mol = Molecule.from_mol2(mol2_path)
            
            assert mol.name == "benzene"
            assert mol.n_atoms == 6
            assert mol.n_bonds == 6
            
            # Check atom types (should be GAFF 'ca' for aromatic carbon)
            assert all(t == "ca" for t in mol.atom_types)
            
            # Check elements
            assert all(e == "C" for e in mol.elements)
            
            # Check charges
            assert mol.charges.shape == (6,)
            np.testing.assert_allclose(mol.charges, -0.115, rtol=1e-3)
            
            # Check positions shape
            assert mol.positions.shape == (6, 3)
            
            # Check bonds (0-indexed)
            expected_bonds = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
            assert set(mol.bonds) == set(expected_bonds)
            
            # Check bond orders (aromatic -> 1)
            assert all(o == 1 for o in mol.bond_orders)

            # Check aromaticity is tracked separately from bond_orders --
            # this is what lets _to_rdkit() distinguish a real aromatic
            # ring from a plain single-bonded (e.g. cyclohexane) ring.
            assert all(mol.bond_aromatic)
        finally:
            Path(mol2_path).unlink()

    def test_to_rdkit_perceives_aromaticity(self):
        """`_to_rdkit()` must round-trip TRIPOS "ar" bonds as real RDKit
        aromaticity, not silently flatten them to a uniform-single-bonded
        (non-aromatic) ring -- confirmed post-260820 audit as a real,
        already-shipped bug affecting every existing `_to_rdkit()` caller
        (e.g. `proxide.chem.partial_charges`), not just new code. Without
        the fix, RDKit has no Kekule alternation to perceive aromaticity
        from, and `assign_gaff2_atom_types` mistypes every ring carbon.
        """
        pytest.importorskip("rdkit")
        from proxide.chem.gaff2 import assign_gaff2_atom_types

        with tempfile.NamedTemporaryFile(suffix=".mol2", mode="w", delete=False) as f:
            f.write(BENZENE_MOL2)
            mol2_path = f.name

        try:
            mol = Molecule.from_mol2(mol2_path)
            rdmol = mol._to_rdkit()

            assert all(atom.GetIsAromatic() for atom in rdmol.GetAtoms())

            types = assign_gaff2_atom_types(rdmol)
            assert types == ["ca"] * 6
        finally:
            Path(mol2_path).unlink()

    def test_to_rdkit_does_not_fabricate_implicit_hydrogens_when_file_has_explicit_h(self):
        """A real (explicit-hydrogen) mol2 lists every atom -- a terminal
        heteroatom with fewer bonds than RDKit's default neutral valence
        (e.g. a deprotonated carboxylate oxygen, single-bonded to its
        carbon with no H record anywhere in the file) is charged, not
        missing a real hydrogen. `_to_rdkit()` must not let RDKit's default
        implicit-valence model silently invent one -- confirmed 260821 as
        the actual cause of the geostd bulk-sample's two largest mismatch
        buckets ("o"->"oh", "c"->"c3" on carboxylate-type groups): the
        fabricated implicit H inflated the oxygen's degree from 1 to 2,
        which fails GAFF2's `ATD o * 8 1 &` rule (connum must be exactly 1)
        and falls through to the generic 2-connection "oh" rule instead.
        Real antechamber never does this since it treats an ac/mol2 file's
        atom list as complete.
        """
        pytest.importorskip("rdkit")

        acetate_mol2 = """\
@<TRIPOS>MOLECULE
acetate
 7 6 0 0 0
SMALL
bcc


@<TRIPOS>ATOM
      1 C1          0.0000    0.0000    0.0000 c3        1 LIG      0.000000
      2 H1          1.0000    0.0000    0.0000 hc        1 LIG      0.000000
      3 H2         -0.5000    0.8660    0.0000 hc        1 LIG      0.000000
      4 H3         -0.5000   -0.8660    0.0000 hc        1 LIG      0.000000
      5 C2         -0.5000    0.0000    1.4000 c         1 LIG      0.700000
      6 O1          0.2000    0.0000    2.3000 o         1 LIG     -0.700000
      7 O2         -1.8000    0.0000    1.5000 o         1 LIG     -0.700000
@<TRIPOS>BOND
     1     1     2 1
     2     1     3 1
     3     1     4 1
     4     1     5 1
     5     5     6 1
     6     5     7 1
@<TRIPOS>SUBSTRUCTURE
     1 LIG         1 TEMP              0 ****  ****    0 ROOT
"""
        with tempfile.NamedTemporaryFile(suffix=".mol2", mode="w", delete=False) as f:
            f.write(acetate_mol2)
            mol2_path = f.name

        try:
            mol = Molecule.from_mol2(mol2_path)
            rdmol = mol._to_rdkit()
            o1 = rdmol.GetAtomWithIdx(5)
            o2 = rdmol.GetAtomWithIdx(6)
            assert o1.GetSymbol() == "O" and o2.GetSymbol() == "O"
            assert o1.GetDegree() == 1 and o1.GetTotalNumHs() == 0
            assert o2.GetDegree() == 1 and o2.GetTotalNumHs() == 0
        finally:
            Path(mol2_path).unlink()

    def test_real_mol2_file(self):
        """Test parsing a real MOL2 file if available."""
        # Check if imatinib.mol2 exists
        imatinib_path = Path(__file__).parents[4] / "../../openmmforcefields/amber/files/imatinib.mol2"
        if not imatinib_path.exists():
            pytest.skip("imatinib.mol2 not found")
        
        mol = Molecule.from_mol2(imatinib_path)
        
        # Imatinib has 68 atoms
        assert mol.n_atoms == 68
        assert mol.n_bonds == 72
        
        # Should have various GAFF types
        assert "ca" in mol.atom_types  # Aromatic carbon
        assert "n" in mol.atom_types or "nh" in mol.atom_types or "n3" in mol.atom_types  # Nitrogen


class TestMoleculeFromSdf:
    """Tests for Molecule.from_sdf()."""
    
    def test_parse_methane_sdf(self):
        """Test parsing a simple methane SDF file."""
        with tempfile.NamedTemporaryFile(suffix=".sdf", mode="w", delete=False) as f:
            f.write(METHANE_SDF)
            sdf_path = f.name
        
        try:
            mol = Molecule.from_sdf(sdf_path)
            
            assert mol.name == "methane"
            assert mol.n_atoms == 5
            assert mol.n_bonds == 4
            
            # SDF doesn't have GAFF types
            assert all(t == "" for t in mol.atom_types)
            
            # Check elements
            assert mol.elements[0] == "C"
            assert mol.elements[1:] == ["H"] * 4
            
            # Check positions shape
            assert mol.positions.shape == (5, 3)
            
            # Carbon at origin
            np.testing.assert_allclose(mol.positions[0], [0, 0, 0], atol=1e-4)
        finally:
            Path(sdf_path).unlink()


class TestMoleculeProperties:
    """Tests for Molecule properties and methods."""
    
    def test_n_atoms_n_bonds(self):
        """Test n_atoms and n_bonds properties."""
        with tempfile.NamedTemporaryFile(suffix=".mol2", mode="w", delete=False) as f:
            f.write(BENZENE_MOL2)
            mol2_path = f.name
        
        try:
            mol = Molecule.from_mol2(mol2_path)
            assert mol.n_atoms == 6
            assert mol.n_bonds == 6
        finally:
            Path(mol2_path).unlink()


def _rdkit_available() -> bool:
    """Check if RDKit is available."""
    try:
        from rdkit import Chem  # noqa: F401
        return True
    except ImportError:
        return False


class TestMoleculeFromSmiles:
    """Tests for Molecule.from_smiles() (requires RDKit)."""
    
    def test_smiles_requires_rdkit(self):
        """Test that SMILES parsing gives helpful error without RDKit."""
        try:
            from rdkit import Chem  # noqa: F401
            rdkit_available = True
        except ImportError:
            rdkit_available = False
        
        if not rdkit_available:
            with pytest.raises(ImportError, match="RDKit"):
                Molecule.from_smiles("C")
    
    @pytest.mark.skipif(
        not _rdkit_available(),
        reason="RDKit not installed"
    )
    def test_smiles_to_molecule(self):
        """Test SMILES -> Molecule conversion with RDKit."""
        mol = Molecule.from_smiles("C")  # Methane
        
        assert mol.n_atoms == 5  # C + 4 H
        assert mol.n_bonds == 4
        assert "C" in mol.elements
        assert "H" in mol.elements

