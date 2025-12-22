"""Tests for the fetching module."""

import pathlib
from unittest import mock

import pytest
import requests
from proxide.io import fetching


@mock.patch("proxide.io.fetching._oxidize")
def test_fetch_rcsb_delegation(mock_oxidize: mock.MagicMock, tmp_path: pathlib.Path) -> None:
  """Test that fetch_rcsb delegates to the Rust backend."""
  expected_path = str(tmp_path / "1abc.cif")
  mock_oxidize.fetch_rcsb.return_value = expected_path

  path = fetching.fetch_rcsb("1abc", format_type="mmcif", output_dir=str(tmp_path))

  assert path == expected_path
  mock_oxidize.fetch_rcsb.assert_called_once_with("1abc", str(tmp_path), "mmcif")


@mock.patch("proxide.io.fetching._oxidize")
def test_fetch_md_cath_delegation(mock_oxidize: mock.MagicMock, tmp_path: pathlib.Path) -> None:
  """Test that fetch_md_cath delegates to the Rust backend."""
  expected_path = str(tmp_path / "1a2b00.h5")
  mock_oxidize.fetch_md_cath.return_value = expected_path

  path = fetching.fetch_md_cath("1a2b00", output_dir=str(tmp_path))

  assert path == expected_path
  mock_oxidize.fetch_md_cath.assert_called_once_with("1a2b00", str(tmp_path))


@mock.patch("proxide.io.fetching._oxidize")
def test_fetch_afdb_delegation(mock_oxidize: mock.MagicMock, tmp_path: pathlib.Path) -> None:
  """Test that fetch_afdb delegates to the Rust backend."""
  expected_path = str(tmp_path / "AF-P12345-F1-model_v4.pdb")
  mock_oxidize.fetch_afdb.return_value = expected_path

  path = fetching.fetch_afdb("P12345", output_dir=str(tmp_path), version=4)

  assert path == expected_path
  mock_oxidize.fetch_afdb.assert_called_once_with("P12345", str(tmp_path), 4)
