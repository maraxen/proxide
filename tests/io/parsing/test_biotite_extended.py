"""Extended tests for legacy Biotite parsing utilities.

DEPRECATED: The biotite.py module has been removed.
These tests are now skipped.

The Rust parser (priox_rs) is now the primary parser for PDB/mmCIF files.
Use test_rust.py for testing the new parser.
"""

import pytest

# Skip all tests in this module - biotite.py was removed in Phase 5 migration
pytestmark = pytest.mark.skip(
    reason="biotite.py module removed - Rust parser is now primary"
)


def test_placeholder():
    """Placeholder test to prevent empty test module errors."""
    pass
