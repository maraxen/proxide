"""Integration tests for HDF5 parsing (MDTraj and MDCATH formats).

Tests require 'mdcath' feature to be enabled when compiling Rust extension.
Tests use synthetic HDF5 fixtures created with h5py.
"""

import pytest
import numpy as np
from pathlib import Path

try:
    from proxide.io.parsing.rust_wrapper import (
        parse_mdtraj_h5_metadata,
        parse_mdtraj_h5_frame,
        parse_mdcath_metadata,
        get_mdcath_replicas,
        parse_mdcath_frame,
        is_hdf5_support_available,
        is_rust_parser_available,
        get_rust_capabilities,
        MdtrajH5Data,
        MdcathData,
    )
    RUST_AVAILABLE = is_rust_parser_available()
except ImportError:
    RUST_AVAILABLE = False

# Check if h5py is available for creating test fixtures
try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not RUST_AVAILABLE, reason="Rust extension not available"
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mdtraj_h5_file(tmp_path):
    """Create a synthetic MDTraj-format HDF5 file for testing."""
    if not H5PY_AVAILABLE:
        pytest.skip("h5py not available for creating test fixtures")
    
    h5_path = tmp_path / "test_mdtraj.h5"
    
    num_frames = 10
    num_atoms = 50
    
    with h5py.File(h5_path, 'w') as f:
        # Coordinates: (num_frames, num_atoms, 3)
        coords = np.random.randn(num_frames, num_atoms, 3).astype(np.float32)
        f.create_dataset("coordinates", data=coords)
        
        # Time: (num_frames,)
        times = np.arange(num_frames, dtype=np.float64) * 0.5  # 0.5 ps timestep
        f.create_dataset("time", data=times)
        
        # Topology (simplified - MDTraj uses JSON inside an attribute)
        # For now, our Rust parser reads dummy topology
    
    return h5_path


@pytest.fixture
def mdcath_h5_file(tmp_path):
    """Create a synthetic MDCATH-format HDF5 file for testing."""
    if not H5PY_AVAILABLE:
        pytest.skip("h5py not available for creating test fixtures")
    
    h5_path = tmp_path / "test_mdcath.h5"
    
    domain_id = "1abc00"
    num_residues = 100
    temperatures = ["320", "348"]
    replicas = ["0", "1"]
    num_frames = 5
    
    with h5py.File(h5_path, 'w') as f:
        # Create domain group
        domain_grp = f.create_group(domain_id)
        
        # Residue names (fixed-length strings)
        resnames = np.array(["ALA"] * num_residues, dtype="S8")
        domain_grp.create_dataset("resname", data=resnames)
        
        # Chain IDs
        chain_ids = np.array(["A"] * num_residues, dtype="S8")
        domain_grp.create_dataset("chain", data=chain_ids)
        
        # Create temperature/replica/coords structure
        for temp in temperatures:
            temp_grp = domain_grp.create_group(temp)
            for replica in replicas:
                replica_grp = temp_grp.create_group(replica)
                
                # Coordinates: (num_frames, num_residues, 3) - CA only
                coords = np.random.randn(num_frames, num_residues, 3).astype(np.float32)
                replica_grp.create_dataset("coords", data=coords)
                
                # DSSP (optional)
                dssp = np.zeros((num_frames, num_residues), dtype=np.int8)
                replica_grp.create_dataset("dssp", data=dssp)
    
    return h5_path


# =============================================================================
# Capabilities Tests
# =============================================================================

class TestHDF5Capabilities:
    """Test HDF5 capability detection."""
    
    def test_capabilities_include_hdf5(self):
        """Test that get_rust_capabilities includes HDF5 functions."""
        caps = get_rust_capabilities()
        
        assert "parse_mdtraj_h5" in caps
        assert "parse_mdcath" in caps
        # These may be True or False depending on compilation features
        assert isinstance(caps["parse_mdtraj_h5"], bool)
        assert isinstance(caps["parse_mdcath"], bool)
    
    def test_is_hdf5_support_available_function(self):
        """Test is_hdf5_support_available returns boolean."""
        result = is_hdf5_support_available()
        assert isinstance(result, bool)


# =============================================================================
# MDTraj H5 Tests
# =============================================================================

@pytest.mark.skipif(not H5PY_AVAILABLE, reason="h5py not available")
class TestMdtrajH5Parser:
    """Tests for MDTraj HDF5 parsing."""
    
    @pytest.mark.skipif(
        RUST_AVAILABLE and not is_hdf5_support_available(),
        reason="HDF5 feature not enabled"
    )
    def test_parse_metadata(self, mdtraj_h5_file):
        """Test parsing MDTraj H5 metadata."""
        result = parse_mdtraj_h5_metadata(mdtraj_h5_file)
        
        assert isinstance(result, MdtrajH5Data)
        assert result.num_frames == 10
        assert result.num_atoms == 50
        assert len(result.atom_names) == 50
    
    @pytest.mark.skipif(
        RUST_AVAILABLE and not is_hdf5_support_available(),
        reason="HDF5 feature not enabled"
    )
    def test_parse_frame(self, mdtraj_h5_file):
        """Test parsing a single frame from MDTraj H5."""
        # First get metadata
        metadata = parse_mdtraj_h5_metadata(mdtraj_h5_file)
        
        # Then get frame
        frame_data = parse_mdtraj_h5_frame(mdtraj_h5_file, frame_idx=0)
        
        # Check returned RawAtomData
        assert frame_data.num_atoms == metadata.num_atoms
        assert frame_data.coords.shape == (50, 3)
    
    @pytest.mark.skipif(
        RUST_AVAILABLE and not is_hdf5_support_available(),
        reason="HDF5 feature not enabled"
    )
    def test_parse_multiple_frames(self, mdtraj_h5_file):
        """Test parsing multiple frames."""
        for frame_idx in range(3):
            frame_data = parse_mdtraj_h5_frame(mdtraj_h5_file, frame_idx=frame_idx)
            assert frame_data.coords.shape == (50, 3)
    
    @pytest.mark.skipif(
        RUST_AVAILABLE and not is_hdf5_support_available(),
        reason="HDF5 feature not enabled"
    )
    def test_frame_out_of_range(self, mdtraj_h5_file):
        """Test error when frame index is out of range."""
        with pytest.raises(ValueError):
            parse_mdtraj_h5_frame(mdtraj_h5_file, frame_idx=100)
    
    def test_missing_file_error(self):
        """Test error when file doesn't exist."""
        if not is_hdf5_support_available():
            with pytest.raises(ImportError):
                parse_mdtraj_h5_metadata("/nonexistent/file.h5")
        else:
            with pytest.raises(ValueError):
                parse_mdtraj_h5_metadata("/nonexistent/file.h5")


# =============================================================================
# MDCATH H5 Tests
# =============================================================================

@pytest.mark.skipif(not H5PY_AVAILABLE, reason="h5py not available")
class TestMdcathH5Parser:
    """Tests for MDCATH HDF5 parsing."""
    
    @pytest.mark.skipif(
        RUST_AVAILABLE and not is_hdf5_support_available(),
        reason="HDF5 feature not enabled"
    )
    def test_parse_domain_metadata(self, mdcath_h5_file):
        """Test parsing MDCATH domain metadata."""
        result = parse_mdcath_metadata(mdcath_h5_file)
        
        assert isinstance(result, MdcathData)
        assert result.domain_id == "1abc00"
        assert result.num_residues == 100
        assert len(result.resnames) == 100
        assert "320" in result.temperatures
        assert "348" in result.temperatures
    
    @pytest.mark.skipif(
        RUST_AVAILABLE and not is_hdf5_support_available(),
        reason="HDF5 feature not enabled"
    )
    def test_get_replicas(self, mdcath_h5_file):
        """Test getting replica list."""
        replicas = get_mdcath_replicas(mdcath_h5_file, "1abc00", "320")
        
        assert isinstance(replicas, list)
        assert "0" in replicas
        assert "1" in replicas
    
    @pytest.mark.skipif(
        RUST_AVAILABLE and not is_hdf5_support_available(),
        reason="HDF5 feature not enabled"
    )
    def test_parse_frame(self, mdcath_h5_file):
        """Test parsing a single frame from MDCATH."""
        frame = parse_mdcath_frame(
            mdcath_h5_file,
            domain_id="1abc00",
            temperature="320",
            replica="0",
            frame_idx=0,
        )
        
        assert isinstance(frame, dict)
        assert frame["temperature"] == "320"
        assert frame["replica"] == "0"
        assert frame["frame_idx"] == 0
        assert frame["coords"].shape == (100, 3)
    
    @pytest.mark.skipif(
        RUST_AVAILABLE and not is_hdf5_support_available(),
        reason="HDF5 feature not enabled"
    )
    def test_parse_different_temperatures(self, mdcath_h5_file):
        """Test parsing frames at different temperatures."""
        for temp in ["320", "348"]:
            frame = parse_mdcath_frame(
                mdcath_h5_file,
                domain_id="1abc00",
                temperature=temp,
                replica="0",
                frame_idx=0,
            )
            assert frame["temperature"] == temp
    
    @pytest.mark.skipif(
        RUST_AVAILABLE and not is_hdf5_support_available(),
        reason="HDF5 feature not enabled"
    )
    def test_frame_out_of_range(self, mdcath_h5_file):
        """Test error when frame index is out of range."""
        with pytest.raises(ValueError):
            parse_mdcath_frame(
                mdcath_h5_file,
                domain_id="1abc00",
                temperature="320",
                replica="0",
                frame_idx=100,
            )
    
    def test_missing_file_error(self):
        """Test error when file doesn't exist."""
        if not is_hdf5_support_available():
            with pytest.raises(ImportError):
                parse_mdcath_metadata("/nonexistent/file.h5")
        else:
            with pytest.raises(ValueError):
                parse_mdcath_metadata("/nonexistent/file.h5")


# =============================================================================
# Feature-Gate Tests
# =============================================================================

class TestFeatureGate:
    """Test behavior when HDF5 feature is not enabled."""
    
    def test_import_error_without_feature(self):
        """Test that proper ImportError is raised when feature not enabled."""
        if is_hdf5_support_available():
            pytest.skip("HDF5 feature is enabled")
        
        with pytest.raises(ImportError) as exc_info:
            parse_mdtraj_h5_metadata("/any/path.h5")
        
        assert "mdcath" in str(exc_info.value).lower() or "hdf5" in str(exc_info.value).lower()
