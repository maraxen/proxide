"""Tests for scripts/sync_forcefields.py: the default asset path and OpenMM exclusions."""

from __future__ import annotations

import importlib.util
import pathlib
from types import ModuleType

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "sync_forcefields.py"


@pytest.fixture(scope="module")
def sync() -> ModuleType:
  spec = importlib.util.spec_from_file_location("sync_forcefields", SCRIPT)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


@pytest.fixture
def fake_openmm_data(tmp_path: pathlib.Path) -> pathlib.Path:
  data = tmp_path / "openmm_data"
  (data / "amber19").mkdir(parents=True)
  for rel in ("amber14-all.xml", "amber99_obc.xml", "amber19/protein.ff19SB.xml", "amber19/protein.ff19ipq.xml"):
    (data / rel).write_text("<ForceField/>\n")
  return data


def test_default_assets_dir_is_the_shipped_tree(sync: ModuleType) -> None:
  # The record that the acceptance program checks lives in the real asset tree.
  assert (REPO / sync.ASSETS_DIR / "FORCEFIELD-LICENCES.tsv").is_file()


def test_ff19ipq_is_skipped(sync: ModuleType, fake_openmm_data: pathlib.Path, tmp_path: pathlib.Path) -> None:
  assets = tmp_path / "assets"
  results = sync.copy_openmm_bundled_files(assets, openmm_data=fake_openmm_data)
  assert not (assets / "openmm_bundled/amber19/protein.ff19ipq.xml").exists()
  assert (assets / "openmm_bundled/amber19/protein.ff19SB.xml").is_file()
  assert [pathlib.Path(p).name for p in results["skipped"]] == ["protein.ff19ipq.xml"]
  # Positive control: the rest of the fake tree was copied.
  assert (assets / "openmm_bundled/amber14-all.xml").is_file()
  assert (assets / "implicit/amber99_obc.xml").is_file()


def test_top_level_exclusion_is_honoured(
  sync: ModuleType,
  fake_openmm_data: pathlib.Path,
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(sync, "EXCLUDED_OPENMM_FILES", frozenset({"amber14-all.xml"}))
  assets = tmp_path / "assets"
  results = sync.copy_openmm_bundled_files(assets, openmm_data=fake_openmm_data)
  assert not (assets / "openmm_bundled/amber14-all.xml").exists()
  assert [pathlib.Path(p).name for p in results["skipped"]] == ["amber14-all.xml"]
