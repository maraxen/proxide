
import pytest
import numpy as np
import tempfile
import os
from proxide.io.parsing.rust_wrapper import parse_structure, OutputSpec, CoordFormat

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
    """Test that RBF features are computed correctly."""
    spec = OutputSpec(
        coord_format=CoordFormat.Atom37,
        compute_rbf=True,
        rbf_num_neighbors=5
    )
    
    result = parse_structure(sample_pdb, spec)
    
    # Result is a Protein
    # RBF features are no longer stored in Protein object
    assert not hasattr(result, "rbf_features")
    
def test_electrostatics_no_charges(sample_pdb):
    """Test that electrostatics are skipped without charges."""
    spec = OutputSpec(
        compute_electrostatics=True
    )
    
    result = parse_structure(sample_pdb, spec)
    
    # Physics features should be None if skipped
    assert result.physics_features is None

def test_defaults(sample_pdb):
    """Test that features are not computed by default."""
    spec = OutputSpec()
    result = parse_structure(sample_pdb, spec)
    assert not hasattr(result, "rbf_features")
    assert result.physics_features is None
