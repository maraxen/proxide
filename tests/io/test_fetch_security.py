"""Tests for fetching module security (path traversal, invalid IDs)."""

import pathlib
import pytest
from proxide.io import fetching

def test_fetch_rcsb_invalid_id():
    """Test that invalid RCSB ID raises an error (protection against path traversal)."""
    with pytest.raises(Exception) as excinfo:
        # PDB ID with path traversal
        fetching.fetch_rcsb("../../../etc/passwd")
    assert "Invalid ID" in str(excinfo.value)

def test_fetch_afdb_invalid_id():
    """Test that invalid AFDB ID raises an error."""
    with pytest.raises(Exception) as excinfo:
        fetching.fetch_afdb("P12345/../../bad")
    assert "Invalid ID" in str(excinfo.value)

def test_fetch_md_cath_invalid_id():
    """Test that invalid MD-CATH ID raises an error."""
    with pytest.raises(Exception) as excinfo:
        fetching.fetch_md_cath("1a/../../bad")
    assert "Invalid ID" in str(excinfo.value)

def test_fetch_md_cath_short_id():
    """Test that short MD-CATH ID (potential panic) is handled."""
    with pytest.raises(Exception) as excinfo:
        fetching.fetch_md_cath("a")
    # Should be caught by validate_id or the length check
    assert "Invalid ID" in str(excinfo.value) or "at least 3 characters" in str(excinfo.value)

def test_fetch_foldcomp_invalid_id():
    """Test that invalid FoldComp database name raises an error."""
    with pytest.raises(Exception) as excinfo:
        fetching.fetch_foldcomp_database("bad;rm -rf /")
    assert "Invalid ID" in str(excinfo.value)
