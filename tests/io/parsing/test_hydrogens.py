#!/usr/bin/env python3
"""Test hydrogen addition integration."""

from pathlib import Path

import pytest

from proxide import OutputSpec, parse_structure


def test_add_hydrogens_pipeline():
    """Test that add_hydrogens runs without crashing."""
    # Create a minimal test PDB
    pdb_path = Path("tests/data/test_ala.pdb")
    pdb_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(pdb_path, "w") as f:
        f.write("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n")
        f.write("ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C\n")
        f.write("ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C\n")
        f.write("ATOM      4  O   ALA A   1       1.246   2.390   0.000  1.00  0.00           O\n")
        f.write("ATOM      5  CB  ALA A   1       1.989  -0.760   1.220  1.00  0.00           C\n")
        f.write("END\n")
    
    # Test with hydrogens ON
    spec_on = OutputSpec(add_hydrogens=True, infer_bonds=True)
    result_on = parse_structure(str(pdb_path), spec_on)
    
    # Check that we got results
    assert "coordinates" in result_on, f"Missing coordinates key. Keys: {list(result_on.keys())}"
    
    # Test with hydrogens OFF
    spec_off = OutputSpec(add_hydrogens=False)
    result_off = parse_structure(str(pdb_path), spec_off)
    
    assert "coordinates" in result_off
    
    print(f"Test passed!")
    print(f"  With add_hydrogens=True: coordinates shape = {result_on['coordinates'].shape}")
    print(f"  With add_hydrogens=False: coordinates shape = {result_off['coordinates'].shape}")
    
    # Clean up
    pdb_path.unlink()

if __name__ == "__main__":
    test_add_hydrogens_pipeline()
