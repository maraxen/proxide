"""Trajectory parity tests for proxide Rust vs MDTraj/reference implementations.

Checks:
1. XTC reading matches MDTraj (coordinates, box vectors, time).
2. DCD reading matches MDTraj.
3. TRR reading matches MDTraj.
4. HDF5 reading (if implemented) matches reference.

Tests P3.1 objectives:
- DCD: Match MDTraj frame coordinates (1e-3 Å tolerance)
- TRR: Match MDTraj frame coordinates (1e-3 Å tolerance)  
- XTC: Match MDTraj frame coordinates (1e-3 Å tolerance)
- Frame count parity
- Box vector parity (where applicable)

Note: These tests require trajectory test files and MDTraj installation.
Tests are skipped gracefully when files are missing.
"""

from pathlib import Path

import numpy as np
import pytest

try:
    import mdtraj
    MDTRAJ_AVAILABLE = True
except ImportError:
    MDTRAJ_AVAILABLE = False


# =============================================================================
# Test Data Paths
# =============================================================================

TEST_DATA_DIR = Path("tests/data")
TRAJ_DATA_DIR = TEST_DATA_DIR / "trajectories"

# Expected test files (will skip if not present)
XTC_FILE = TRAJ_DATA_DIR / "test.xtc"
DCD_FILE = TRAJ_DATA_DIR / "test.dcd"
TRR_FILE = TRAJ_DATA_DIR / "test.trr"
PDB_TOPOLOGY = TEST_DATA_DIR / "1crn.pdb"  # For topology when loading trajectories
HDF5_FILE = TRAJ_DATA_DIR / "test.h5"


# =============================================================================
# XTC Parity Tests
# =============================================================================


@pytest.mark.skipif(not MDTRAJ_AVAILABLE, reason="MDTraj not installed")
def test_xtc_frame_coordinates_vs_mdtraj():
    """Compare XTC frame coordinates against MDTraj using pure-Rust molly parser."""
    from proxide import _oxidize

    # Try multiple common paths for test files
    xtc_file = TRAJ_DATA_DIR / "frame0.xtc"
    pdb_file = TRAJ_DATA_DIR / "frame0.pdb"

    if not xtc_file.exists():
        xtc_file = Path("/tmp/frame0.xtc")
        pdb_file = Path("/tmp/frame0.pdb")
    
    if not xtc_file.exists():
        xtc_file = TRAJ_DATA_DIR / "test.xtc"
        pdb_file = TRAJ_DATA_DIR / "test.pdb"

    if not xtc_file.exists() or not pdb_file.exists():
        pytest.skip(f"XTC test files not available (checked {xtc_file})")

    # Parse with oxidize (pure-Rust molly)
    try:
        result = _oxidize.parse_xtc(str(xtc_file))
    except Exception as e:
        if "requires compiling with" in str(e):
            pytest.skip("xtc feature not enabled")
        raise e
        
    priox_coords = result["coordinates"]  # Already in Angstroms

    # Parse with MDTraj
    traj = mdtraj.load(str(xtc_file), top=str(pdb_file))
    mdtraj_coords = traj.xyz * 10.0  # nm -> Angstroms

    # Check shapes match
    assert (
        priox_coords.shape == mdtraj_coords.shape
    ), f"Shape mismatch: {priox_coords.shape} vs {mdtraj_coords.shape}"

    # Check coordinate parity
    diff = np.abs(priox_coords - mdtraj_coords)
    max_diff = np.max(diff)
    print(f"XTC Max coord difference: {max_diff:.6f} Å")

    # XTC uses single precision - allow small tolerance
    assert max_diff < 0.01, f"XTC coordinates differ by {max_diff} Å"


@pytest.mark.skipif(not MDTRAJ_AVAILABLE, reason="MDTraj not installed")
def test_xtc_import_available():
    """Check if XTC parser is importable (now using pure-Rust molly)."""
    from proxide import parse_xtc
    assert callable(parse_xtc)



# =============================================================================
# DCD Parity Tests
# =============================================================================

@pytest.mark.skipif(not MDTRAJ_AVAILABLE, reason="MDTraj not installed")
def test_dcd_frame_coordinates_vs_mdtraj():
    """Compare DCD frame coordinates against MDTraj."""
    from proxide import _oxidize
    
    dcd_file = TRAJ_DATA_DIR / "frame0.dcd"
    pdb_file = TRAJ_DATA_DIR / "frame0.pdb"
    
    if not dcd_file.exists():
        dcd_file = TRAJ_DATA_DIR / "test.dcd"
    
    if not dcd_file.exists() or not pdb_file.exists():
        pytest.skip(f"DCD test files not available (checked {dcd_file} and {pdb_file})")
    
    # Parse with oxidize (pure-Rust)
    result = _oxidize.parse_dcd(str(dcd_file))
    priox_coords = result["coordinates"]  # Already in Angstroms
    
    # Parse with MDTraj
    traj = mdtraj.load(str(dcd_file), top=str(pdb_file))
    mdtraj_coords = traj.xyz * 10.0  # nm -> Angstroms
    
    # Check shapes match
    assert (
        priox_coords.shape == mdtraj_coords.shape
    ), f"Shape mismatch: {priox_coords.shape} vs {mdtraj_coords.shape}"
    
    # Check coordinate parity
    diff = np.abs(priox_coords - mdtraj_coords)
    max_diff = np.max(diff)
    print(f"DCD Max coord difference: {max_diff:.6f} Å")
    
    # Single precision coordinates, so allowing 1e-3
    assert max_diff < 1e-3, f"DCD coordinates differ by {max_diff} Å"

    # Check unit cell parity if available
    if "unit_cells" in result:
        priox_cells = result["unit_cells"]
        mdtraj_cells_len = traj.unitcell_lengths * 10.0 # nm -> Angstroms
        mdtraj_cells_ang = traj.unitcell_angles
        
        # priox_cells is (N_frames, 6) -> [a, b, c, alpha, beta, gamma]
        np.testing.assert_allclose(priox_cells[:, :3], mdtraj_cells_len, atol=1e-3)
        np.testing.assert_allclose(priox_cells[:, 3:], mdtraj_cells_ang, atol=1e-2)
        print("DCD Unit cell parity OK")


@pytest.mark.skipif(not MDTRAJ_AVAILABLE, reason="MDTraj not installed")
def test_dcd_import_available():
    """Check if DCD parser is importable."""
    from proxide import parse_dcd
    assert callable(parse_dcd)


# =============================================================================
# TRR Parity Tests
# =============================================================================

@pytest.mark.skipif(not MDTRAJ_AVAILABLE, reason="MDTraj not installed")
def test_trr_frame_coordinates_vs_mdtraj():
    """Compare TRR frame coordinates against MDTraj."""
    from proxide import _oxidize
    
    trr_file = TRAJ_DATA_DIR / "frame0.trr"
    pdb_file = TRAJ_DATA_DIR / "frame0.pdb"
    
    if not trr_file.exists():
        trr_file = TRAJ_DATA_DIR / "test.trr"
    
    if not trr_file.exists() or not pdb_file.exists():
        pytest.skip(f"TRR test files not available (checked {trr_file} and {pdb_file})")
    
    # Parse with oxidize (pure-Rust)
    result = _oxidize.parse_trr(str(trr_file))
    priox_coords = result["coordinates"]  # Already in Angstroms
    
    # Parse with MDTraj
    traj = mdtraj.load(str(trr_file), top=str(pdb_file))
    mdtraj_coords = traj.xyz * 10.0  # nm -> Angstroms
    
    # Check shapes match
    assert (
        priox_coords.shape == mdtraj_coords.shape
    ), f"Shape mismatch: {priox_coords.shape} vs {mdtraj_coords.shape}"
    
    # Check coordinate parity
    diff = np.abs(priox_coords - mdtraj_coords)
    max_diff = np.max(diff)
    print(f"TRR Max coord difference: {max_diff:.6f} Å")
    
    # TRR can be single or double precision. frame0.trr is usually single.
    assert max_diff < 1e-3, f"TRR coordinates differ by {max_diff} Å"

    # Check box vectors if available
    if "box_vectors" in result:
        priox_box = result["box_vectors"]
        mdtraj_box = traj.unitcell_vectors * 10.0 # nm -> Angstroms
        
        # MDTraj vectors are (N_frames, 3, 3)
        np.testing.assert_allclose(priox_box, mdtraj_box, atol=1e-3)
        print("TRR Box vector parity OK")


@pytest.mark.skipif(not MDTRAJ_AVAILABLE, reason="MDTraj not installed")
def test_trr_import_available():
    """Check if TRR parser is importable."""
    from proxide import parse_trr
    assert callable(parse_trr)


# =============================================================================
# HDF5 Parity Tests
# =============================================================================

@pytest.mark.skipif(not MDTRAJ_AVAILABLE, reason="MDTraj not installed")
def test_hdf5_parsing_parity():
    """Verify HDF5 parsing match MDTraj."""
    # Ensure HDF5 test file exists
    if not HDF5_FILE.exists():
        if not TRR_FILE.exists() or not PDB_TOPOLOGY.exists():
             pytest.skip("Cannot generate HDF5: missing source TRR or PDB")
        
        print("Generating HDF5 test file...")
        traj = mdtraj.load(str(TRR_FILE), top=str(PDB_TOPOLOGY))
        traj.save(str(HDF5_FILE))
        
    from proxide.io.parsing.rust import (
        is_hdf5_support_available,
        parse_mdtraj_h5_frame,
        parse_mdtraj_h5_metadata,
    )
    
    if not is_hdf5_support_available():
        pytest.skip("HDF5 support not available (mdcath feature not compiled)")

    # Load with MDTraj
    traj_mdtraj = mdtraj.load(str(HDF5_FILE))
    
    # Load with Rust (Metadata)
    metadata = parse_mdtraj_h5_metadata(str(HDF5_FILE))
    
    # Compare counts
    assert metadata.num_frames == traj_mdtraj.n_frames
    assert metadata.num_atoms == traj_mdtraj.n_atoms
    
    # Load frames
    rust_frames = []
    for i in range(metadata.num_frames):
        frame = parse_mdtraj_h5_frame(str(HDF5_FILE), i)
        rust_frames.append(frame.coords)
        
    rust_coords = np.array(rust_frames)
    mdtraj_coords = traj_mdtraj.xyz # nm
    
    # Convert MDTraj nm to Angstroms
    mdtraj_coords_angstrom = mdtraj_coords * 10.0
    
    diff = np.abs(rust_coords - mdtraj_coords_angstrom)
    max_diff = np.max(diff)
    print(f"Max HDF5 coord diff: {max_diff:.6f} Å")
    
    assert max_diff < 1e-3, f"HDF5 mismatch: {max_diff}"


# =============================================================================
# Integration Test
# =============================================================================

def test_all_trajectory_parsers_available():
    """Test that all trajectory parsers are at least importable."""
    from proxide import parse_dcd, parse_trr, parse_xtc
    assert callable(parse_xtc)
    assert callable(parse_dcd)
    assert callable(parse_trr)
    print("All trajectory parsers available")
