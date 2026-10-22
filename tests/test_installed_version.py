"""The installed proxide package reports the version pyproject.toml declares."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def test_installed_proxide_matches_pyproject_when_metadata_present() -> None:
    proxide = pytest.importorskip("proxide")

    if proxide.__version__ == "0+unknown":
        pytest.skip("proxide is installed without metadata; nothing to compare")
    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert proxide.__version__ == pyproject["project"]["version"]
