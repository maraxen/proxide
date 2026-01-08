"""Equivalence test: Compare Rust FoldComp reader vs original PDB file."""
import pytest
import numpy as np
from pathlib import Path

# Test data paths
# NOTE: These files are currently missing in the monorepo structure.
# P3.2 goal will involve restoring/regenerating them.
TEST_DIR = Path(__file__).parent.parent.parent.parent / "proxide" / "oxidize" / "foldcomp" / "test"
FCZ_PATH = TEST_DIR / "test_af.fcz"
PDB_PATH = TEST_DIR / "test_af.pdb"
EXAMPLE_DB = TEST_DIR / "example_db"


@pytest.fixture
def rust_system():
    """Load structure using Rust oxidize parser."""
    import proxide._oxidize as _oxidize
    return _oxidize.parse_foldcomp(str(FCZ_PATH))


@pytest.fixture  
def reference_coords():
    """Extract backbone coords from original PDB file."""
    coords = []
    with open(PDB_PATH) as f:
        for line in f:
            if line.startswith("ATOM"):
                atom_name = line[12:16].strip()
                if atom_name in ("N", "CA", "C"):
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.extend([x, y, z])
    return np.array(coords, dtype=np.float32)


@pytest.mark.skipif(not FCZ_PATH.exists() or not PDB_PATH.exists(), reason="foldcomp test files not found")
def test_foldcomp_equivalence(rust_system, reference_coords):
    """Test that Rust FoldComp reader produces coordinates close to original PDB."""
    rust_coords = np.array(rust_system.coordinates, dtype=np.float32)
    
    # Should have same number of atoms
    assert len(rust_coords) == len(reference_coords), \
        f"Coordinate count mismatch: {len(rust_coords)} vs {len(reference_coords)}"
    
    # Compute RMSD
    diff = rust_coords - reference_coords
    rmsd = np.sqrt(np.mean(diff ** 2))
    
    print(f"RMSD between Rust reconstruction and original PDB: {rmsd:.4f} Å")
    
    # FoldComp introduces compression error, typically < 0.1 Å
    # Allow up to 0.5 Å for reconstruction + discretization error
    assert rmsd < 0.5, f"RMSD too high: {rmsd:.4f} Å"


@pytest.mark.skipif(not FCZ_PATH.exists(), reason="foldcomp test fcz not found")
def test_foldcomp_atom_count(rust_system):
    """Test that atom count matches expected backbone atoms."""
    # test_af has some number of residues, should have 3*N_res backbone atoms
    n_coords = len(rust_system.coordinates)
    n_atoms = n_coords // 3
    
    # Backbone: N, CA, C per residue
    assert n_atoms % 3 == 0, f"Atom count {n_atoms} not divisible by 3"
    n_residues = n_atoms // 3
    print(f"Structure has {n_residues} residues ({n_atoms} backbone atoms)")


@pytest.mark.skipif(not EXAMPLE_DB.exists(), reason="example_db not found")
def test_foldcomp_database_open():
    """Test FoldCompDatabase can open the example database."""
    import proxide._oxidize as _oxidize
    
    db = _oxidize.FoldCompDatabase(str(EXAMPLE_DB))
    assert len(db) > 0, "Database should have entries"
    print(f"Database contains {len(db)} entries")


@pytest.mark.skipif(not EXAMPLE_DB.exists(), reason="example_db not found")  
def test_foldcomp_database_get():
    """Test retrieving an entry from the database."""
    import proxide._oxidize as _oxidize
    
    db = _oxidize.FoldCompDatabase(str(EXAMPLE_DB))
    
    # Try to get first entry - read index to find a valid ID
    # We'll try with name "d1asha_" which is in the test fixtures
    if "d1asha_" in db:
        system = db.get_by_name("d1asha_")
        assert len(system.coordinates) > 0
        print(f"Retrieved d1asha_ with {len(system.coordinates)//3} atoms")


@pytest.mark.skipif(not FCZ_PATH.exists(), reason="foldcomp test fcz not found")
def test_benchmark_rust_foldcomp(rust_system, benchmark):
    """Benchmark Rust FoldComp parsing speed."""
    import proxide._oxidize as _oxidize
    
    def parse():
        return _oxidize.parse_foldcomp(str(FCZ_PATH))
    
    result = benchmark(parse)
    assert len(result.coordinates) > 0
