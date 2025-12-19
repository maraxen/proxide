
import pytest
import numpy as np
import tempfile
import os
from proxide.io.parsing.rust import parse_structure, OutputSpec, CoordFormat

# Create a minimal PDB file for testing
PDB_CONTENT = """
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N
ATOM      2  CA  ALA A   1       1.500   0.000   0.000  1.00 20.00           C
ATOM      3  C   ALA A   1       2.000   1.000   0.000  1.00 20.00           C
ATOM      4  O   ALA A   1       2.500   1.000   1.000  1.00 20.00           O
ATOM      5  CB  ALA A   1       1.500  -1.000   0.000  1.00 20.00           C
ATOM      6  N   GLY A   2       4.000   0.000   0.000  1.00 20.00           N
ATOM      7  CA  GLY A   2       5.500   0.000   0.000  1.00 20.00           C
ATOM      8  C   GLY A   2       6.000   1.000   0.000  1.00 20.00           C
ATOM      9  O   GLY A   2       6.500   1.000   1.000  1.00 20.00           O
"""

@pytest.fixture
def sample_pdb():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
        f.write(PDB_CONTENT)
        path = f.name
    yield path
    os.unlink(path)

def test_rbf_computation(sample_pdb):
    """Test that RBF features can be requested (even if not stored on Protein)."""
    spec = OutputSpec(
        coord_format=CoordFormat.Atom37,
        compute_rbf=True,
        rbf_num_neighbors=5
    )
    
    result = parse_structure(sample_pdb, spec)
    
    # RBF features are computed on-demand, not stored on Protein
    # Just verify parsing succeeds with compute_rbf flag
    assert result is not None
    assert hasattr(result, "coordinates")
    
def test_electrostatics_no_charges(sample_pdb):
    """Test that electrostatics flag can be set (even if not computed without charges)."""
    spec = OutputSpec(
        compute_electrostatics=True
    )
    
    result = parse_structure(sample_pdb, spec)
    
    # Just verify parsing succeeds
    assert result is not None
    assert hasattr(result, "coordinates")

def test_defaults(sample_pdb):
    """Test that default parsing works without feature computation flags."""
    spec = OutputSpec()
    result = parse_structure(sample_pdb, spec)
    
    # Verify basic structure is present
    assert result is not None
    assert hasattr(result, "coordinates")
    assert hasattr(result, "aatype")
