import pytest
import numpy as np
import tempfile
import os

from proxide import _oxidize

PDB_CONTENT = """ATOM      1  N   ALA A   1      -0.525   1.362   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       1.526   0.000   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       2.153   1.085   0.000  1.00  0.00           O
ATOM      5  CB  ALA A   1      -0.510  -0.778  -1.209  1.00  0.00           C
"""

@pytest.fixture
def pdb_file():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp:
        tmp.write(PDB_CONTENT)
        path = tmp.name
    try:
        yield path
    finally:
        if os.path.exists(path):
            os.remove(path)

def test_format_atom37(pdb_file):
    spec = _oxidize.OutputSpec()
    spec.coord_format = _oxidize.CoordFormat.Atom37
    result = _oxidize.parse_structure(pdb_file, spec)
    
    # Check basic fields
    assert "coordinates" in result
    coords = result["coordinates"]
    # Flat array in dict (N_res * 37 * 3) -> 1 * 37 * 3 = 111
    assert coords.shape == (111,)
    
    assert "atom_mask" in result
    assert result["atom_mask"].shape == (37,)

def test_format_atom14(pdb_file):
    spec = _oxidize.OutputSpec()
    spec.coord_format = _oxidize.CoordFormat.Atom14
    result = _oxidize.parse_structure(pdb_file, spec)
    
    coords = result["coordinates"]
    # 1 * 14 * 3 = 42
    assert coords.shape == (42,)

def test_format_backbone(pdb_file):
    spec = _oxidize.OutputSpec()
    spec.coord_format = _oxidize.CoordFormat.BackboneOnly
    result = _oxidize.parse_structure(pdb_file, spec)
    
    coords = result["coordinates"]
    # 1 * 4 * 3 = 12
    assert coords.shape == (12,)
    
    # Verify N, CA, C, O are present (mask)
    # They should be all 1s for ALA
    mask = result["atom_mask"]
    assert np.all(mask == 1.0)

def test_format_full(pdb_file):
    spec = _oxidize.OutputSpec()
    spec.coord_format = _oxidize.CoordFormat.Full
    result = _oxidize.parse_structure(pdb_file, spec)
    
    assert "coord_shape" in result
    shape = result["coord_shape"] # (N_atoms, 3, 1) for flat format
    assert shape[0] == 5  # 5 atoms in ALA
    assert shape[1] == 3  # 3D coords
    assert shape[2] == 1  # flat indicator
    
    assert "atom_names" in result
    atom_names = result["atom_names"]
    # Should be list of strings with N_atoms entries
    assert len(atom_names) == 5
    
    # Check first few names (PDB order: N, CA, C, O, CB)
    assert atom_names[0] == "N"
    assert atom_names[1] == "CA"

def test_caching(pdb_file):
    # Enable caching
    spec = _oxidize.OutputSpec()
    spec.enable_caching = True
    spec.coord_format = _oxidize.CoordFormat.Atom37
    
    # First call
    res1 = _oxidize.parse_structure(pdb_file, spec)
    
    # Second call
    res2 = _oxidize.parse_structure(pdb_file, spec)
    
    assert np.allclose(res1["coordinates"], res2["coordinates"])
    
    # Implicitly checks that 2nd call worked via cache (or at least works consistentely)
    # To really verify cache, we'd need to mock or inspect logs.
