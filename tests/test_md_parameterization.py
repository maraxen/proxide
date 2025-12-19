"""Test MD parameterization integration with force field."""

import numpy as np
import pytest
from pathlib import Path

# Skip if Rust extension not available
pytest.importorskip("oxidize")

from proxide.io.parsing.rust import (
    parse_structure,
    OutputSpec,
    MissingResidueMode,
    is_rust_parser_available,
)


# Path to test data
TEST_DATA_DIR = Path(__file__).parent.parent / "data"
FF_XML_PATH = Path(__file__).parent.parent / "src" / "proxide" / "physics" / "force_fields" / "xml" / "protein.ff19SB.xml"


class TestMDParameterization:
    """Tests for MD parameterization from force field."""

    @pytest.fixture
    def simple_pdb(self, tmp_path):
        """Create a simple PDB file for testing."""
        pdb_content = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00 20.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00 20.00           C
ATOM      4  O   ALA A   1       1.251   2.390   0.000  1.00 20.00           O
ATOM      5  CB  ALA A   1       1.978  -0.760   1.230  1.00 20.00           C
ATOM      6  N   GLY A   2       3.320   1.520   0.000  1.00 20.00           N
ATOM      7  CA  GLY A   2       3.970   2.820   0.000  1.00 20.00           C
ATOM      8  C   GLY A   2       5.480   2.720   0.000  1.00 20.00           C
ATOM      9  O   GLY A   2       6.020   1.600   0.000  1.00 20.00           O
END
"""
        pdb_path = tmp_path / "test.pdb"
        pdb_path.write_text(pdb_content)
        return pdb_path

    def test_parameterization_basic(self, simple_pdb):
        """Test that parameterization produces charges and LJ params."""
        if not FF_XML_PATH.exists():
            pytest.skip(f"Force field file not found: {FF_XML_PATH}")
        
        # Create spec with parameterization enabled
        spec = OutputSpec(
            parameterize_md=True,
            force_field=str(FF_XML_PATH),
        )
        
        # Parse with parameterization
        import proxide_rs
        result = _oxidize.parse_structure(str(simple_pdb), spec)
        
        # Check that charges were assigned
        assert "charges" in result, "Missing charges in result"
        charges = result["charges"]
        assert len(charges) == 9, f"Expected 9 atoms, got {len(charges)}"
        
        # Check that sigmas and epsilons were assigned
        assert "sigmas" in result, "Missing sigmas in result"
        assert "epsilons" in result, "Missing epsilons in result"
        
        # Check that atom_types was assigned
        assert "atom_types" in result, "Missing atom_types in result"
        
        # Check parameterization stats
        assert "num_parameterized" in result
        assert "num_skipped" in result
        print(f"Parameterized: {result['num_parameterized']}, Skipped: {result['num_skipped']}")

    def test_parameterization_charges_nonzero(self, simple_pdb):
        """Test that backbone atoms get non-zero charges."""
        if not FF_XML_PATH.exists():
            pytest.skip(f"Force field file not found: {FF_XML_PATH}")
        
        spec = OutputSpec(
            parameterize_md=True,
            force_field=str(FF_XML_PATH),
        )
        
        import proxide_rs
        result = _oxidize.parse_structure(str(simple_pdb), spec)
        
        charges = result["charges"]
        
        # N should have negative charge, CA positive
        # Check that not all charges are zero
        assert not np.allclose(charges, 0.0), "All charges are zero - parameterization failed"
        
        # Check charge range is reasonable (-1 to +1 for amino acids)
        assert np.all(charges >= -2.0) and np.all(charges <= 2.0), \
            f"Charges out of expected range: min={charges.min()}, max={charges.max()}"

    def test_no_parameterization_by_default(self, simple_pdb):
        """Test that parameterization is disabled by default."""
        from proxide import _oxidize
        result = _oxidize.parse_structure(str(simple_pdb))
        
        # Should not have MD params when not requested
        assert "charges" not in result or result.get("charges") is None

    def test_missing_ff_warning(self, simple_pdb, caplog):
        """Test warning when parameterize_md=True but no force_field provided."""
        import logging
        caplog.set_level(logging.WARNING)
        
        spec = OutputSpec(
            parameterize_md=True,
            # force_field not set
        )
    
        from proxide import _oxidize
        result = _oxidize.parse_structure(str(simple_pdb), spec)
        
        # Should complete without error, but no charges assigned
        assert "charges" not in result or result.get("charges") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
