"""The two Python-extension manifests must stay in step (debt #3181, ledger A5).

crates/proxide_py is the MIT-only manifest the wheels and the sdist are built
from; crates/proxide_py_gpl compiles the same sources with the GPL engines.
Two hand-maintained copies drift, and drift here is silent: a feature present
in only one manifest is just never compiled by the other.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / "crates/proxide_py/Cargo.toml"
GPL = ROOT / "crates/proxide_py_gpl/Cargo.toml"
GPL_CRATES = {"proxide-gaff2", "proxide-ligand-frame"}
GPL_FEATURES = {"gaff2-engine", "ligand-frame"}


def _load(path: Path) -> dict:
  return tomllib.loads(path.read_text())


@pytest.fixture(scope="module")
def manifests() -> tuple[dict, dict]:
  return _load(PY), _load(GPL)


def test_mit_manifest_names_no_gpl_crate(manifests) -> None:
  # The sdist omits the GPL crates; any mention here (even optional) breaks
  # Cargo resolution of the sdist, which is the #3181 failure.
  py, _ = manifests
  sections = [py.get("dependencies", {}), py.get("dev-dependencies", {})]
  sections += [t.get("dependencies", {}) for t in py.get("target", {}).values()]
  named = {name for section in sections for name in section}
  assert not (named & GPL_CRATES), f"proxide_py must not depend on {named & GPL_CRATES}"
  assert not (set(py.get("features", {})) & GPL_FEATURES)
  for name, members in py.get("features", {}).items():
    for member in members:
      assert not any(crate in member for crate in GPL_CRATES), (name, member)


def test_gpl_manifest_dependencies_superset(manifests) -> None:
  py, gpl = manifests
  for name, spec in py["dependencies"].items():
    assert name in gpl["dependencies"], f"proxide_py_gpl lacks dependency {name}"
    assert gpl["dependencies"][name] == spec, f"dependency {name} differs between manifests"
  assert GPL_CRATES <= set(gpl["dependencies"])


def test_gpl_manifest_features_superset(manifests) -> None:
  py, gpl = manifests
  for name, members in py["features"].items():
    if name == "default":
      continue
    assert gpl["features"].get(name) == members, f"feature {name} differs between manifests"
  assert set(py["features"]["default"]) <= set(gpl["features"]["default"])
  assert GPL_FEATURES <= set(gpl["features"])


def test_gpl_manifest_builds_the_same_sources(manifests) -> None:
  py, gpl = manifests
  assert gpl["lib"]["path"] == "../proxide_py/src/lib.rs"
  assert gpl["lib"]["crate-type"] == py["lib"]["crate-type"] == ["cdylib"]
  # Distinct lib names: a shared name collides in a workspace build's target dir.
  assert gpl["lib"]["name"] != py["lib"]["name"]
  assert gpl["package"]["publish"] is False
  assert py["package"]["publish"] is False
