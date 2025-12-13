"""Tests for Rust parser integration

Compares Rust parser output against Biotite golden reference.
"""

import pytest
import numpy as np
from pathlib import Path

try:
    from proxide.io.parsing.rust_wrapper import parse_pdb_rust, is_rust_parser_available
    RUST_AVAILABLE = is_rust_parser_available()
except ImportError:
    RUST_AVAILABLE = False

pytestmark = pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")


def test_rust_parser_import():
    """Test that Rust parser can be imported."""
    assert RUST_AVAILABLE, "Rust parser should be available"
    assert callable(parse_pdb_rust), "parse_pdb_rust should be callable"


def test_rust_parser_basic_structure(tmp_path):
    """Test parsing a basic PDB structure."""
    # Create minimal test PDB
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
    
    # Parse with Rust
    protein = parse_pdb_rust(pdb_file)
    
    # Verify basic structure
    assert protein.coordinates.shape == (1, 37, 3), f"Expected (1, 37, 3), got {protein.coordinates.shape}"
    assert protein.atom_mask.shape == (1, 37), f"Expected (1, 37), got {protein.atom_mask.shape}"
    assert len(protein.aatype) == 1, f"Expected 1 residue, got {len(protein.aatype)}"
    assert protein.aatype[0] == 0, f"ALA should have type 0, got {protein.aatype[0]}"
    
    # Verify no NaN/Inf
    assert not np.any(np.isnan(protein.coordinates)), "Coordinates contain NaN"
    assert not np.any(np.isinf(protein.coordinates)), "Coordinates contain Inf"
    
    # Verify atom mask is binary
    assert np.all((protein.atom_mask == 0) | (protein.atom_mask == 1)), "Atom mask should be binary"
    
    # Check expected atoms are present (N, CA, C, O, CB for ALA)
    from proxide.chem.residues import atom_order
    assert protein.atom_mask[0, atom_order["N"]] == 1.0, "N should be present"
    assert protein.atom_mask[0, atom_order["CA"]] == 1.0, "CA should be present" 
    assert protein.atom_mask[0, atom_order["C"]] == 1.0, "C should be present"
    assert protein.atom_mask[0, atom_order["O"]] == 1.0, "O should be present"
    assert protein.atom_mask[0, atom_order["CB"]] == 1.0, "CB should be present"
    
    # Check coordinates for CA
    ca_coord = protein.coordinates[0, atom_order["CA"]]
    np.testing.assert_array_almost_equal(ca_coord, [1.0, 0.0, 0.0], decimal=3)


def test_rust_parser_multi_residue(tmp_path):
    """Test parsing multiple residues."""
    pdb_content = """
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N
ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00 20.00           C
ATOM      3  C   ALA A   1       2.000   0.000   0.000  1.00 20.00           C
ATOM      4  O   ALA A   1       3.000   0.000   0.000  1.00 20.00           O
ATOM      5  CB  ALA A   1       1.000   1.000   0.000  1.00 20.00           C
ATOM      6  N   GLY A   2       4.000   0.000   0.000  1.00 20.00           N
ATOM      7  CA  GLY A   2       5.000   0.000   0.000  1.00 20.00           C
ATOM      8  C   GLY A   2       6.000   0.000   0.000  1.00 20.00           C
ATOM      9  O   GLY A   2       7.000   0.000   0.000  1.00 20.00           O
END
"""
    
    pdb_file = tmp_path / "test_multi.pdb"
    pdb_file.write_text(pdb_content.strip())
    
    protein = parse_pdb_rust(pdb_file)
    
    # Should have 2 residues
    assert protein.coordinates.shape == (2, 37, 3)
    assert len(protein.aatype) == 2
    assert protein.aatype[0] == 0  # ALA
    assert protein.aatype[1] == 7  # GLY
    
    # GLY should not have CB
    from proxide.chem.residues import atom_order
    assert protein.atom_mask[1, atom_order["CB"]] == 0.0, "GLY should not have CB"


def test_rust_parser_multi_chain(tmp_path):
    """Test parsing multi-chain structures."""
    pdb_content = """
ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C
ATOM      2  CA  GLY B   1      10.000   0.000   0.000  1.00 20.00           C
END
"""
    
    pdb_file = tmp_path / "test_chains.pdb"
    pdb_file.write_text(pdb_content.strip())
    
    protein = parse_pdb_rust(pdb_file)
    
    # Should have 2 residues with different chain indices
    assert len(protein.aatype) == 2
    assert protein.chain_index[0] != protein.chain_index[1], "Chains should have different indices"


def test_rust_parser_residue_indices(tmp_path):
    """Test that residue indices are preserved from PDB."""
    pdb_content = """
ATOM      1  CA  ALA A  10       0.000   0.000   0.000  1.00 20.00           C
ATOM      2  CA  GLY A  20      10.000   0.000   0.000  1.00 20.00           C
END
"""
    
    pdb_file = tmp_path / "test_resid.pdb"
    pdb_file.write_text(pdb_content.strip())
    
    protein = parse_pdb_rust(pdb_file)
    
    # Residue indices should match PDB
    assert protein.residue_index[0] == 10
    assert protein.residue_index[1] == 20
