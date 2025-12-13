"""Extended tests for priox.io.parsing.pqr.

DEPRECATED: Many of these tests tested internal Python parsing functions
that have been replaced by the Rust parser. Those tests are now skipped.

The Rust PQR parser functionality is tested in test_pqr.py.
"""

import pytest
from proxide.io.parsing import pqr


# Skip tests that rely on removed Python implementation
pytestmark = pytest.mark.skip(
    reason="PQR parsing moved to Rust - internal Python functions removed"
)


class TestPQRExtended:
    """Extended tests for PQR parsing.
    
    Note: These tests are skipped because the internal parsing functions
    (_parse_atom_line, parse_pqr_to_processed_structure) have been replaced
    by the Rust implementation.
    """
    
    def test_parse_atom_line_invalid(self):
        """Test parsing invalid lines - NOW IN RUST."""
        pass

    def test_parse_atom_line_merged_fields(self):
        """Test parsing lines with merged fields - NOW IN RUST."""
        pass

    def test_parse_atom_line_water(self):
        """Test water skipping - NOW IN RUST."""
        pass

    def test_parse_pqr_file_object(self):
        """Test parsing from file-like object.
        
        Note: Rust parser requires file path, not file-like object.
        This is a known limitation.
        """
        pass

    def test_parse_pqr_chain_filtering_set(self):
        """Test chain filtering with a list/set - NOW IN PYTHON WRAPPER."""
        pass

    def test_parse_pqr_epsilon_lookup(self):
        """Test epsilon lookup based on element - NOW IN RUST."""
        pass
