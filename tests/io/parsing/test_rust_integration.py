"""Comprehensive integration tests for Rust parser.

Tests the complete Python → Rust → Python pipeline, including:
- mmCIF parsing
- Force field loading
- Parity tests with Biotite
- Capabilities API
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import warnings

try:
    from proxide.io.parsing.rust import (
        parse_pdb_to_protein as parse_pdb_rust,
        parse_pdb_to_protein,
        parse_pdb_raw_rust,
        parse_mmcif_rust,
        load_forcefield_rust,
        is_rust_parser_available,
        get_rust_capabilities,
        RawAtomData,
        ForceFieldData,
    )
    RUST_AVAILABLE = is_rust_parser_available()
except ImportError:
    RUST_AVAILABLE = False

pytestmark = pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")


# =============================================================================
# Capabilities Tests
# =============================================================================

class TestRustCapabilities:
    """Test the capabilities API."""
    
    def test_rust_available(self):
        """Verify Rust parser is available."""
        assert RUST_AVAILABLE
        
    def test_capabilities_function(self):
        """Test get_rust_capabilities returns expected functions."""
        caps = get_rust_capabilities()
        
        assert isinstance(caps, dict)
        assert caps["parse_pdb"] is True
        assert caps["parse_mmcif"] is True
        assert caps["parse_structure"] is True
        assert caps["load_forcefield"] is True


# =============================================================================
# mmCIF Parser Tests
# =============================================================================

class TestMmCIFParser:
    """Tests for mmCIF parsing."""
    
    def test_parse_mmcif_basic(self, tmp_path):
        """Test parsing a basic mmCIF file."""
        # Minimal mmCIF content
        cif_content = """
data_test
#
loop_
_atom_site.group_PDB 
_atom_site.id 
_atom_site.type_symbol 
_atom_site.label_atom_id 
_atom_site.label_comp_id 
_atom_site.label_asym_id 
_atom_site.label_seq_id 
_atom_site.Cartn_x 
_atom_site.Cartn_y 
_atom_site.Cartn_z 
_atom_site.occupancy 
_atom_site.B_iso_or_equiv 
_atom_site.auth_seq_id 
_atom_site.auth_asym_id 
ATOM   1  N N   ALA A 1 0.000 0.000 0.000 1.00 20.00 1 A
ATOM   2  C CA  ALA A 1 1.000 0.000 0.000 1.00 20.00 1 A
ATOM   3  C C   ALA A 1 2.000 0.000 0.000 1.00 20.00 1 A
ATOM   4  O O   ALA A 1 3.000 0.000 0.000 1.00 20.00 1 A
ATOM   5  C CB  ALA A 1 1.000 1.000 0.000 1.00 20.00 1 A
#
"""
        cif_file = tmp_path / "test.cif"
        cif_file.write_text(cif_content.strip())
        
        # Parse with Rust
        result = parse_mmcif_rust(cif_file)
        
        # Verify structure
        assert isinstance(result, RawAtomData)
        assert result.num_atoms == 5
        assert len(result.atom_names) == 5
        assert "CA" in result.atom_names
        assert result.coords.shape == (5, 3)
        
    def test_parse_mmcif_multi_chain(self, tmp_path):
        """Test mmCIF with multiple chains."""
        cif_content = """
data_test
#
loop_
_atom_site.group_PDB 
_atom_site.id 
_atom_site.type_symbol 
_atom_site.label_atom_id 
_atom_site.label_comp_id 
_atom_site.label_asym_id 
_atom_site.label_seq_id 
_atom_site.Cartn_x 
_atom_site.Cartn_y 
_atom_site.Cartn_z 
_atom_site.occupancy 
_atom_site.B_iso_or_equiv 
_atom_site.auth_seq_id 
_atom_site.auth_asym_id 
ATOM   1  C CA  ALA A 1 0.000 0.000 0.000 1.00 20.00 1 A
ATOM   2  C CA  GLY B 1 10.000 0.000 0.000 1.00 20.00 1 B
#
"""
        cif_file = tmp_path / "multi.cif"
        cif_file.write_text(cif_content.strip())
        
        result = parse_mmcif_rust(cif_file)
        
        assert result.num_atoms == 2
        # Different chains
        assert len(set(result.chain_ids)) == 2


# =============================================================================
# Force Field Loading Tests
# =============================================================================

class TestForceFieldLoader:
    """Tests for force field XML loading."""
    
    def test_load_forcefield_basic(self, tmp_path):
        """Test loading a basic force field XML file."""
        # Minimal OpenMM-style force field XML
        ff_xml = """<?xml version="1.0" encoding="UTF-8"?>
<ForceField>
  <AtomTypes>
    <Type name="protein-C" class="protein-C" element="C" mass="12.01"/>
    <Type name="protein-N" class="protein-N" element="N" mass="14.01"/>
    <Type name="protein-O" class="protein-O" element="O" mass="16.00"/>
  </AtomTypes>
  
  <Residues>
    <Residue name="ALA">
      <Atom name="N" type="protein-N" charge="-0.4157"/>
      <Atom name="CA" type="protein-C" charge="0.0337"/>
      <Atom name="C" type="protein-C" charge="0.5973"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="C"/>
    </Residue>
  </Residues>
  
  <HarmonicBondForce>
    <Bond class1="protein-C" class2="protein-N" length="0.1335" k="410000.0"/>
    <Bond class1="protein-C" class2="protein-C" length="0.1526" k="259400.0"/>
  </HarmonicBondForce>
  
  <HarmonicAngleForce>
    <Angle class1="protein-N" class2="protein-C" class3="protein-C" angle="1.9216" k="586.2"/>
  </HarmonicAngleForce>
</ForceField>
"""
        ff_file = tmp_path / "test_ff.xml"
        ff_file.write_text(ff_xml)
        
        # Load with Rust
        ff = load_forcefield_rust(ff_file)
        
        # Verify structure
        assert isinstance(ff, ForceFieldData)
        assert ff.num_atom_types == 3
        assert ff.num_residue_templates == 1
        assert ff.num_harmonic_bonds == 2
        assert ff.num_harmonic_angles == 1
        
    def test_load_forcefield_residue_lookup(self, tmp_path):
        """Test looking up residues in loaded force field."""
        ff_xml = """<?xml version="1.0" encoding="UTF-8"?>
<ForceField>
  <AtomTypes>
    <Type name="C" class="C" element="C" mass="12.01"/>
  </AtomTypes>
  <Residues>
    <Residue name="ALA">
      <Atom name="CA" type="C" charge="0.0"/>
    </Residue>
    <Residue name="GLY">
      <Atom name="CA" type="C" charge="0.0"/>
    </Residue>
  </Residues>
</ForceField>
"""
        ff_file = tmp_path / "ff.xml"
        ff_file.write_text(ff_xml)
        
        ff = load_forcefield_rust(ff_file)
        
        # Test residue lookup
        ala = ff.get_residue("ALA")
        assert ala is not None
        assert ala["name"] == "ALA"
        
        gly = ff.get_residue("GLY")
        assert gly is not None
        
        # Unknown residue
        unk = ff.get_residue("UNKNOWN")
        assert unk is None


# =============================================================================
# Raw PDB Parser Tests
# =============================================================================

class TestRawPDBParser:
    """Tests for raw PDB parsing (low-level)."""
    
    def test_parse_pdb_raw_basic(self, tmp_path):
        """Test raw PDB parsing returns correct atom data."""
        pdb_content = """
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N
ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00 15.00           C
ATOM      3  C   ALA A   1       2.000   0.000   0.000  1.00 18.00           C
END
"""
        pdb_file = tmp_path / "test.pdb"
        pdb_file.write_text(pdb_content.strip())
        
        result = parse_pdb_raw_rust(pdb_file)
        
        assert isinstance(result, RawAtomData)
        assert result.num_atoms == 3
        
        # Check coordinates
        np.testing.assert_array_almost_equal(
            result.coords[0], [0.0, 0.0, 0.0], decimal=3
        )
        np.testing.assert_array_almost_equal(
            result.coords[1], [1.0, 0.0, 0.0], decimal=3
        )
        
    def test_parse_pdb_raw_bfactors(self, tmp_path):
        """Test that B-factors are parsed correctly."""
        pdb_content = """
ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 25.00           C
ATOM      2  CB  ALA A   1       1.000   0.000   0.000  0.50 30.00           C
END
"""
        pdb_file = tmp_path / "test.pdb"
        pdb_file.write_text(pdb_content.strip())
        
        result = parse_pdb_raw_rust(pdb_file)
        
        # Check B-factors
        assert len(result.b_factors) == 2
        assert result.b_factors[0] == 25.0
        assert result.b_factors[1] == 30.0
        
        # Check occupancy
        assert result.occupancies[0] == 1.0
        assert result.occupancies[1] == 0.5


# =============================================================================
# High-Level Parser Tests
# =============================================================================

class TestHighLevelParser:
    """Tests for parse_pdb_rust (high-level with formatting)."""
    
    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_parse_pdb_returns_protein_tuple(self, tmp_path):
        """Test that parse_pdb_rust returns a proper Protein (deprecated)."""
        pdb_content = """
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N
ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00 20.00           C
ATOM      3  C   ALA A   1       2.000   0.000   0.000  1.00 20.00           C
ATOM      4  O   ALA A   1       3.000   0.000   0.000  1.00 20.00           O
ATOM      5  CB  ALA A   1       1.000   1.000   0.000  1.00 20.00           C
END
"""
        pdb_file = tmp_path / "test.pdb"
        pdb_file.write_text(pdb_content.strip())
        
        protein = parse_pdb_rust(pdb_file)
        
        # Should be Protein (deprecated)
        from proxide.core.containers import Protein
        assert isinstance(protein, Protein)
        
        # Check shapes
        assert protein.coordinates.shape == (1, 37, 3)
        assert protein.atom_mask.shape == (1, 37)
        assert len(protein.aatype) == 1
    
    def test_parse_pdb_to_protein_returns_protein(self, tmp_path):
        """Test that parse_pdb_to_protein returns a proper Protein object."""
        pdb_content = """
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N
ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00 20.00           C
ATOM      3  C   ALA A   1       2.000   0.000   0.000  1.00 20.00           C
ATOM      4  O   ALA A   1       3.000   0.000   0.000  1.00 20.00           O
ATOM      5  CB  ALA A   1       1.000   1.000   0.000  1.00 20.00           C
END
"""
        pdb_file = tmp_path / "test.pdb"
        pdb_file.write_text(pdb_content.strip())
        
        protein = parse_pdb_to_protein(pdb_file)
        
        # Should be Protein (not Protein)
        from proxide.core.containers import Protein
        assert isinstance(protein, Protein)
        
        # Check shapes
        assert protein.coordinates.shape == (1, 37, 3)
        assert len(protein.aatype) == 1
        
    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_parse_pdb_coordinate_accuracy(self, tmp_path):
        """Test parsed coordinates match input."""
        pdb_content = """
ATOM      1  CA  ALA A   1      12.345  67.890 -11.111  1.00 20.00           C
END
"""
        pdb_file = tmp_path / "test.pdb"
        pdb_file.write_text(pdb_content.strip())
        
        protein = parse_pdb_rust(pdb_file)
        
        # CA is index 1 in atom37
        from proxide.chem.residues import atom_order
        ca_idx = atom_order["CA"]
        
        np.testing.assert_array_almost_equal(
            protein.coordinates[0, ca_idx],
            [12.345, 67.890, -11.111],
            decimal=3
        )


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for error handling."""
    
    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_missing_file_pdb(self):
        """Test error when PDB file doesn't exist."""
        with pytest.raises(ValueError) as exc_info:
            parse_pdb_rust("/nonexistent/file.pdb")
        assert "parsing failed" in str(exc_info.value).lower() or "error" in str(exc_info.value).lower()
        
    def test_missing_file_mmcif(self):
        """Test error when mmCIF file doesn't exist."""
        with pytest.raises(ValueError):
            parse_mmcif_rust("/nonexistent/file.cif")
            
    def test_missing_file_forcefield(self):
        """Test error when force field file doesn't exist."""
        with pytest.raises(ValueError):
            load_forcefield_rust("/nonexistent/ff.xml")
            
    def test_invalid_pdb_content(self, tmp_path):
        """Test error handling for invalid PDB content."""
        invalid_file = tmp_path / "invalid.pdb"
        invalid_file.write_text("NOT A VALID PDB FILE\nJUST SOME TEXT")
        
        # Should either raise an error or return empty structure
        # depending on implementation
        try:
            result = parse_pdb_raw_rust(invalid_file)
            # If it doesn't raise, should have 0 atoms
            assert result.num_atoms == 0
        except ValueError:
            pass  # This is also acceptable


# =============================================================================
# Performance Smoke Tests
# =============================================================================

class TestPerformance:
    """Smoke tests for performance."""
    
    def test_parse_large_structure(self, tmp_path):
        """Test parsing a moderately large structure."""
        # Generate 500 residues
        lines = ["HEADER    TEST STRUCTURE"]
        atom_serial = 1
        for res_idx in range(1, 501):
            for atom_name, x_offset in [("N", 0), ("CA", 1), ("C", 2), ("O", 3), ("CB", 1)]:
                if atom_name == "CB" and res_idx % 8 == 0:  # Skip CB for GLY
                    continue
                x = res_idx * 3.8 + x_offset
                y = 0.0
                z = 0.0
                res_name = "GLY" if res_idx % 8 == 0 else "ALA"
                lines.append(
                    f"ATOM  {atom_serial:5d}  {atom_name:<3s} {res_name} A{res_idx:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           {atom_name[0]}"
                )
                atom_serial += 1
        lines.append("END")
        
        pdb_file = tmp_path / "large.pdb"
        pdb_file.write_text("\n".join(lines))
        
        # Should parse without error
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            protein = parse_pdb_rust(pdb_file)
        
        assert protein.coordinates.shape[0] == 500
        assert len(protein.aatype) == 500


# =============================================================================
# Physics Features Tests
# =============================================================================

class TestPhysicsFeatures:
    """Tests for physics feature computation."""
    
    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_vdw_features_basic(self, tmp_path):
        """Test VdW feature computation with compute_vdw=True."""
        pdb_content = """
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00 20.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00 20.00           C
ATOM      4  O   ALA A   1       1.244   2.390   0.000  1.00 20.00           O
ATOM      5  CB  ALA A   1       1.983  -0.743   1.225  1.00 20.00           C
ATOM      6  N   GLY A   2       3.314   1.552   0.000  1.00 20.00           N
ATOM      7  CA  GLY A   2       3.941   2.862   0.000  1.00 20.00           C
ATOM      8  C   GLY A   2       5.458   2.757   0.000  1.00 20.00           C
ATOM      9  O   GLY A   2       6.037   1.667   0.000  1.00 20.00           O
END
"""
        pdb_file = tmp_path / "test.pdb"
        pdb_file.write_text(pdb_content.strip())
        
        # Import OutputSpec from Rust
        from proxide import OutputSpec
        
        # Parse with VdW features enabled
        spec = OutputSpec(compute_vdw=True)
        protein = parse_pdb_rust(pdb_file, spec)
        
        # VdW features should be computed: (N_res, 5) for backbone atoms
        assert protein.vdw_features is not None
        assert protein.vdw_features.shape == (2, 5)  # 2 residues, 5 backbone atoms
        
        # Features for present atoms should be finite
        # (GLY has no CB, so position [1,3] may be NaN - that's expected)
        # Check ALA residue (index 0) - all atoms present
        assert np.all(np.isfinite(protein.vdw_features[0]))
    
    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_vdw_features_disabled_by_default(self, tmp_path):
        """Test that VdW features are not computed by default."""
        pdb_content = """
ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C
END
"""
        pdb_file = tmp_path / "test.pdb"
        pdb_file.write_text(pdb_content.strip())
        
        # Parse without VdW features
        protein = parse_pdb_rust(pdb_file)
        
        # VdW features should be None
        assert protein.vdw_features is None
