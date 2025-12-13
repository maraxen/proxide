"""Tests for PQR file parsing utilities (using Rust parser).

These tests verify the Rust-based PQR parser which returns AtomicSystem objects.
"""

import pathlib
import pytest
import numpy as np

# Import the new functions
from proxide.io.parsing.pqr import load_pqr, parse_pqr_rust
from proxide.core.atomic_system import AtomicSystem

TEST_PQR_PATH = pathlib.Path(__file__).parent.parent.parent / "data" / "1a00.pqr"

# Check if test file exists
pytestmark = pytest.mark.skipif(
    not TEST_PQR_PATH.exists(),
    reason=f"Test PQR file not found: {TEST_PQR_PATH}"
)


def test_parse_pqr_basic():
    """Test parsing a standard PQR file using Rust parser."""
    # Use the Rust parser directly
    data = parse_pqr_rust(TEST_PQR_PATH)
    
    # Verify basic structure
    assert "atom_names" in data
    assert "coords" in data
    assert "res_names" in data
    assert "num_atoms" in data
    assert data["num_atoms"] > 0
    assert len(data["atom_names"]) == data["num_atoms"]
    
    # Check that we have charges and radii (PQR specific)
    assert "charges" in data
    assert "radii" in data
    assert len(data["charges"]) == data["num_atoms"]
    

def test_load_pqr_basic():
    """Test the full load_pqr function."""
    systems = list(load_pqr(TEST_PQR_PATH))
    
    assert len(systems) > 0
    system = systems[0]
    
    # Verify AtomicSystem attributes
    assert isinstance(system, AtomicSystem)
    assert system.coordinates is not None
    assert system.atom_mask is not None
    assert len(system.atom_names) > 0
    
    # PQR should have charges and radii
    assert system.charges is not None
    assert system.radii is not None


def test_load_pqr_chain_selection():
    """Test parsing with chain selection."""
    systems = list(load_pqr(TEST_PQR_PATH, chain_id="A"))
    
    if systems:
        system = systems[0]
        # Verify system was created
        assert isinstance(system, AtomicSystem)
        assert len(system.atom_names) > 0


def test_parse_pqr_empty(tmp_path):
    """Test parsing an empty PQR file (should raise exception)."""
    empty_pqr = tmp_path / "empty.pqr"
    empty_pqr.write_text("")
    
    with pytest.raises(Exception):  # Could be ValueError or ParsingError
        parse_pqr_rust(empty_pqr)


def test_parse_pqr_insertion_codes(tmp_path):
    """Test parsing PQR file with residue insertion codes."""
    pqr_with_insertion = tmp_path / "insertion.pqr"
    pqr_content = """\
ATOM      1  N   ALA A  50      10.000  20.000  30.000  -0.500   1.850
ATOM      2  CA  ALA A  50      11.000  21.000  31.000   0.100   1.700
ATOM      3  N   ALA A  52      12.000  22.000  32.000  -0.500   1.850
ATOM      4  CA  ALA A  52      13.000  23.000  33.000   0.100   1.700
ATOM      5  N   ALA A  52A     14.000  24.000  34.000  -0.500   1.850
ATOM      6  CA  ALA A  52A     15.000  25.000  35.000   0.100   1.700
ATOM      7  N   ALA A  52B     16.000  26.000  36.000  -0.500   1.850
ATOM      8  CA  ALA A  52B     17.000  27.000  37.000   0.100   1.700
ATOM      9  N   ALA A  53      18.000  28.000  38.000  -0.500   1.850
"""
    pqr_with_insertion.write_text(pqr_content)
    
    data = parse_pqr_rust(pqr_with_insertion)
    
    # Should have 9 atoms
    assert data["num_atoms"] == 9
    assert len(data["atom_names"]) == 9


def test_load_pqr_returns_atomicsystem():
    """Test that load_pqr returns AtomicSystem with correct attributes."""
    systems = list(load_pqr(TEST_PQR_PATH))
    assert len(systems) == 1
    
    system = systems[0]
    
    # Check it's an AtomicSystem
    assert isinstance(system, AtomicSystem)
    
    # Check required fields
    assert system.coordinates.shape[1] == 3  # (N, 3)
    assert len(system.atom_mask) == system.coordinates.shape[0]
    
    # Check optional PQR-specific fields
    if system.charges is not None:
        assert len(system.charges) == system.coordinates.shape[0]
    if system.radii is not None:
        assert len(system.radii) == system.coordinates.shape[0]
