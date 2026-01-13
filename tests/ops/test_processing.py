"""Tests for the io.process module."""

import pathlib
import unittest
from io import StringIO
from unittest.mock import Mock, patch
from unittest.mock import mock_open as mock_file_open

import pytest
import requests

from proxide.core.containers import Protein
from proxide.io.parsing.structures import ProcessedStructure
from proxide.ops import processing
from proxide.ops.processing import _resolve_inputs, frame_iterator_from_inputs


class TestResolveInputs:
  """Tests for the _resolve_inputs function."""

  @patch("proxide.io.fetching.fetch_rcsb")
  @patch("proxide.ops.processing.pathlib.Path.exists")
  def test_resolve_pdb_id(self, mock_exists, mock_fetch_rcsb) -> None:
    """Test resolving a PDB ID input."""
    # checking input item existence -> False (so it treats as ID)
    # checking file cache existence -> False (so it calls fetch)
    mock_exists.return_value = False
    mock_fetch_rcsb.return_value = pathlib.Path("1abc.cif")
    
    result = list(_resolve_inputs(["1abc"]))

    assert len(result) == 1
    assert result[0] == pathlib.Path("1abc.cif")
    mock_fetch_rcsb.assert_called_once_with("1abc", format_type="mmcif")

  @patch("proxide.io.fetching.fetch_afdb")
  def test_resolve_afdb_id_direct(self, mock_fetch_afdb) -> None:
    """Test resolving an AFDB ID input (direct fetch)."""
    mock_fetch_afdb.return_value = pathlib.Path("AF-P12345-F1-model_v4.pdb")
    
    # Input is AF-P12345-F1-model_v4
    input_str = "AF-P12345-F1-model_v4"
    result = list(_resolve_inputs([input_str]))
    
    assert len(result) == 1
    assert result[0] == pathlib.Path("AF-P12345-F1-model_v4.pdb")
    mock_fetch_afdb.assert_called_once_with("P12345", version=4)

  def test_resolve_file_path(self, tmp_path: pathlib.Path) -> None:
    """Test resolving a file path input.

    Args:
      tmp_path: Pytest fixture for temporary directory.

    Returns:
      None

    Raises:
      AssertionError: If file path is not resolved correctly.

    Example:
      >>> test_resolve_file_path()

    """
    test_file = tmp_path / "test.pdb"
    test_file.write_text("ATOM   1  CA  ALA A   1")

    result = list(_resolve_inputs([str(test_file)]))

    assert len(result) == 1
    assert result[0] == test_file

  def test_resolve_stringio(self) -> None:
    """Test resolving a StringIO input.

    Args:
      None

    Returns:
      None

    Raises:
      AssertionError: If StringIO is not resolved correctly.

    Example:
      >>> test_resolve_stringio()

    """
    string_io = StringIO("ATOM   1  CA  ALA A   1")

    result = list(_resolve_inputs([string_io]))

    assert len(result) == 1
    assert result[0] is string_io

  def test_resolve_directory(self, tmp_path: pathlib.Path) -> None:
    """Test resolving a directory input (should yield all files in directory).

    Args:
      tmp_path: Pytest fixture for temporary directory.

    Returns:
      None

    Raises:
      AssertionError: If directory resolution is not correct.

    Example:
      >>> test_resolve_directory()

    """
    file1 = tmp_path / "test1.pdb"
    file2 = tmp_path / "test2.pdb"
    file1.write_text("ATOM 1")
    file2.write_text("ATOM 2")

    result = list(_resolve_inputs([str(tmp_path)]))

    expected_file_count = 2
    assert len(result) == expected_file_count
    assert all(isinstance(r, pathlib.Path) for r in result)


class TestFrameIteratorFromInputs:
  """Tests for the frame_iterator_from_inputs function."""

  def test_frame_iterator_basic(self, tmp_path: pathlib.Path) -> None:
    """Test basic frame iterator functionality.

    Args:
      tmp_path: Pytest fixture for temporary directory.

    Returns:
      None

    Raises:
      AssertionError: If frame iteration does not work correctly.

    Example:
      >>> test_frame_iterator_basic()

    """
    # This is a placeholder test - actual implementation would depend on
    # parse_input behavior and Protein structure
    test_file = tmp_path / "test.pdb"
    test_file.write_text("ATOM   1  CA  ALA A   1      10.000  20.000  30.000")

    with patch("proxide.ops.processing.parse_input") as mock_parse:
      mock_parse.return_value = iter([Mock()])  # Yield one mock protein

      result = list(frame_iterator_from_inputs([str(test_file)]))

      # Verify parse_input was called
      mock_parse.assert_called()
      assert len(result) == 1

  def test_frame_iterator_from_inputs_empty(self):
    """Test with empty input iterator."""
    with patch("proxide.ops.processing.parse_input") as mock_parse:
      mock_parse.return_value = iter([])  # Mock empty iterator

      iterator = processing.frame_iterator_from_inputs(
        ["dummy.pdb"],
        # chain_id_dict=None  # Removed
      )

      frames = list(iterator)
      assert len(frames) == 0