"""Tests for scripts/check_versions.py.

Stdlib and pytest only: this module does not import proxide, and conftest.py
imports it lazily, so the file runs in an environment without proxide.
"""

from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "check_versions.py"
STATIC_FILES = ("pyproject.toml", "Cargo.toml", "src/proxide/__init__.py", "docs/source/conf.py")
PROXIDE_CORE_PIN = 'proxide-core = { path = "crates/proxide-core", version = "0.1.0-alpha.17" }'


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_versions", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_check(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root)],
        capture_output=True,
        text=True,
        check=False,
    )


def _make_tree(tmp_path: Path) -> Path:
    """Copy only the files the checker reads into tmp_path."""
    root = tmp_path / "repo"
    files = list(STATIC_FILES) + [
        p.relative_to(REPO).as_posix() for p in sorted((REPO / "crates").glob("*/Cargo.toml"))
    ]
    for rel in files:
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / rel, dest)
    return root


def _edit(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert text.count(old) == 1, f"{path}: expected exactly one {old!r}"
    path.write_text(text.replace(old, new), encoding="utf-8")


def _set_package_version(path: Path, value: str) -> None:
    text = path.read_text(encoding="utf-8")
    new, count = re.subn(
        r"^version(\.workspace)?\s*=.*$", f'version = "{value}"', text, count=1, flags=re.M
    )
    assert count == 1, f"{path}: no package version line found"
    path.write_text(new, encoding="utf-8")


def _retarget_cargo(root: Path, cargo_version: str) -> None:
    """Set the workspace version and rewrite every alpha.17 pin to match it."""
    for manifest in [root / "Cargo.toml", *sorted((root / "crates").glob("*/Cargo.toml"))]:
        text = manifest.read_text(encoding="utf-8")
        manifest.write_text(text.replace("0.1.0-alpha.17", cargo_version), encoding="utf-8")


def _mutate_root_stale_pin(root: Path) -> None:
    _edit(root / "Cargo.toml", PROXIDE_CORE_PIN, PROXIDE_CORE_PIN.replace("alpha.17", "alpha.15"))


def _mutate_crate_stale_sibling_pin(root: Path) -> None:
    _edit(
        root / "crates/proxide-wasm/Cargo.toml",
        'version = "0.1.0-alpha.17", default-features',
        'version = "0.1.0-alpha.9", default-features',
    )


def _mutate_target_table_stale_pin(root: Path) -> None:
    text = (root / "Cargo.toml").read_text(encoding="utf-8")
    text += (
        "\n[target.'cfg(unix)'.dependencies]\n"
        'proxide-core = { path = "crates/proxide-core", version = "0.1.0-alpha.15" }\n'
    )
    (root / "Cargo.toml").write_text(text, encoding="utf-8")


def _mutate_python_cargo_disagree(root: Path) -> None:
    _edit(root / "pyproject.toml", 'version = "0.1.0a17"', 'version = "0.1.0a18"')


def _mutate_crate_literal_package_version(root: Path) -> None:
    _set_package_version(root / "crates/proxide-core/Cargo.toml", "0.1.0-alpha.15")


def _mutate_single_quoted_dunder_version(root: Path) -> None:
    _edit(
        root / "src/proxide/__init__.py",
        '__version__ = importlib.metadata.version("proxide")',
        "__version__ = '0.1.0'",
    )


def _mutate_unsupported_pep440(root: Path) -> None:
    _edit(root / "pyproject.toml", 'version = "0.1.0a17"', 'version = "0.1.0.post1"')


def _mutate_missing_conf(root: Path) -> None:
    (root / "docs/source/conf.py").unlink()


def _mutate_versionless_path_dep(root: Path) -> None:
    _edit(root / "Cargo.toml", PROXIDE_CORE_PIN, 'proxide-core = { path = "crates/proxide-core" }')


def _mutate_non_member_path_dep(root: Path) -> None:
    text = (root / "Cargo.toml").read_text(encoding="utf-8")
    text += '\n[dependencies]\nstray = { path = "vendor/stray", version = "0.1.0-alpha.17" }\n'
    (root / "Cargo.toml").write_text(text, encoding="utf-8")


def _mutate_missing_member_manifest(root: Path) -> None:
    (root / "crates/proxide-core/Cargo.toml").unlink()


def _mutate_stray_conf_literal(root: Path) -> None:
    text = (root / "docs/source/conf.py").read_text(encoding="utf-8")
    (root / "docs/source/conf.py").write_text(text + '\nrelease = "0.1.0"\n', encoding="utf-8")


def _mutate_missing_pyproject(root: Path) -> None:
    (root / "pyproject.toml").unlink()


def _mutate_dynamic_version(root: Path) -> None:
    _edit(root / "pyproject.toml", 'version = "0.1.0a17"', 'dynamic = ["version"]')


MUTATIONS: list[tuple[str, Callable[[Path], None], str, str]] = [
    ("root-stale-workspace-pin", _mutate_root_stale_pin, "Cargo.toml", "proxide-core"),
    ("crate-stale-sibling-pin", _mutate_crate_stale_sibling_pin, "crates/proxide-wasm/Cargo.toml", "alpha.9"),
    ("target-table-stale-pin", _mutate_target_table_stale_pin, "Cargo.toml", "alpha.15"),
    ("python-cargo-disagree", _mutate_python_cargo_disagree, "pyproject.toml", "0.1.0a18"),
    ("crate-literal-package-version", _mutate_crate_literal_package_version, "crates/proxide-core/Cargo.toml", "alpha.15"),
    ("single-quoted-dunder-version", _mutate_single_quoted_dunder_version, "src/proxide/__init__.py", "'0.1.0'"),
    ("unsupported-pep440", _mutate_unsupported_pep440, "pyproject.toml", "0.1.0.post1"),
    ("missing-conf", _mutate_missing_conf, "docs/source/conf.py", "missing"),
    ("versionless-path-dep", _mutate_versionless_path_dep, "Cargo.toml", "proxide-core"),
    ("non-member-path-dep", _mutate_non_member_path_dep, "Cargo.toml", "vendor/stray"),
    ("missing-member-manifest", _mutate_missing_member_manifest, "crates/proxide-core/Cargo.toml", "has no manifest"),
    ("stray-conf-literal", _mutate_stray_conf_literal, "docs/source/conf.py", "'0.1.0'"),
    ("missing-pyproject", _mutate_missing_pyproject, "pyproject.toml", "file is missing"),
    ("dynamic-version", _mutate_dynamic_version, "pyproject.toml", "dynamic"),
]


def test_real_repo_versions_agree() -> None:
    result = _run_check(REPO)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("mutate", "offending_file", "fragment"),
    [pytest.param(m, f, s, id=name) for name, m, f, s in MUTATIONS],
)
def test_seeded_defect_is_reported(
    tmp_path: Path, mutate: Callable[[Path], None], offending_file: str, fragment: str
) -> None:
    root = _make_tree(tmp_path)
    mutate(root)

    result = _run_check(root)

    assert result.returncode == 1, result.stderr
    assert "Traceback" not in result.stderr
    assert offending_file in result.stderr
    assert fragment in result.stderr


@pytest.mark.parametrize(
    ("pyproject_version", "cargo_version", "exit_code", "fragment"),
    [
        pytest.param("0.1.0", "0.1.0-beta.1", 0, "cargo stays pre-release 0.1.0-beta.1", id="pyproject-final-cargo-prerelease-ok"),
        pytest.param("0.1.0", "0.1.1-beta.1", 1, "pyproject.toml", id="pyproject-final-cargo-other-release"),
        pytest.param("0.1.1", "0.1.1-dev.0", 0, "cargo stays pre-release 0.1.1-dev.0", id="pyproject-final-cargo-dev-ok"),
        pytest.param("0.1.1.dev0", "0.1.1-dev.0", 0, "all version strings agree", id="pyproject-dev-cargo-dev-exact-ok"),
        pytest.param("0.1.1.dev0", "0.1.1", 1, "pyproject.toml", id="pyproject-dev-cargo-final-mismatch"),
    ],
)
def test_final_pypi_with_prerelease_cargo_rule(
    tmp_path: Path, pyproject_version: str, cargo_version: str, exit_code: int, fragment: str
) -> None:
    root = _make_tree(tmp_path)
    _edit(root / "pyproject.toml", 'version = "0.1.0a17"', f'version = "{pyproject_version}"')
    _retarget_cargo(root, cargo_version)

    result = _run_check(root)

    assert result.returncode == exit_code, result.stderr
    assert fragment in result.stderr
    if exit_code == 1:
        assert "Cargo.toml" in result.stderr


@pytest.mark.parametrize(
    ("pep440", "cargo"),
    [
        ("0.1.0", "0.1.0"),
        ("0.1.0a17", "0.1.0-alpha.17"),
        ("0.1.0a07", "0.1.0-alpha.7"),
        ("0.1.0b1", "0.1.0-beta.1"),
        ("0.1.0rc1", "0.1.0-rc.1"),
        ("0.1.1.dev0", "0.1.1-dev.0"),
        ("0.1.1.dev12", "0.1.1-dev.12"),
    ],
)
def test_pep440_converts_to_cargo(pep440: str, cargo: str) -> None:
    assert _load_script().pep440_to_cargo(pep440) == cargo


@pytest.mark.parametrize("pep440", ["1!0.1.0", "0.1.0.post1", "0.1.0+local", "0.1"])
def test_unsupported_pep440_forms_raise(pep440: str) -> None:
    with pytest.raises(ValueError, match="unsupported PEP 440"):
        _load_script().pep440_to_cargo(pep440)
