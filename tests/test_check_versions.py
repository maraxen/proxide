"""Tests for scripts/check_versions.py: the real repo agrees, and each seeded defect is caught."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tomllib
from collections.abc import Callable
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "check_versions.py"
STATIC_FILES = ("pyproject.toml", "Cargo.toml", "src/proxide/__init__.py", "docs/source/conf.py")
PROXIDE_CORE_PIN = 'proxide-core = { path = "crates/proxide-core", version = "0.1.0-alpha.17" }'


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
    new, count = re.subn(r'^version(\.workspace)?\s*=.*$', f'version = "{value}"', text, count=1, flags=re.M)
    assert count == 1, f"{path}: no package version line found"
    path.write_text(new, encoding="utf-8")


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
    _edit(root / "pyproject.toml", 'version = "0.1.0a17"', 'version = "0.1.0.dev1"')


def _mutate_missing_conf(root: Path) -> None:
    (root / "docs/source/conf.py").unlink()


MUTATIONS: list[tuple[str, Callable[[Path], None], str, str]] = [
    ("root-stale-workspace-pin", _mutate_root_stale_pin, "Cargo.toml", "proxide-core"),
    ("crate-stale-sibling-pin", _mutate_crate_stale_sibling_pin, "crates/proxide-wasm/Cargo.toml", "alpha.9"),
    ("target-table-stale-pin", _mutate_target_table_stale_pin, "Cargo.toml", "alpha.15"),
    ("python-cargo-disagree", _mutate_python_cargo_disagree, "pyproject.toml", "0.1.0a18"),
    ("crate-literal-package-version", _mutate_crate_literal_package_version, "crates/proxide-core/Cargo.toml", "alpha.15"),
    ("single-quoted-dunder-version", _mutate_single_quoted_dunder_version, "src/proxide/__init__.py", "'0.1.0'"),
    ("unsupported-pep440", _mutate_unsupported_pep440, "pyproject.toml", "0.1.0.dev1"),
    ("missing-conf", _mutate_missing_conf, "docs/source/conf.py", "missing"),
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
    assert offending_file in result.stderr
    assert fragment in result.stderr


def test_installed_proxide_matches_pyproject_when_metadata_present() -> None:
    import proxide

    if proxide.__version__ == "0+unknown":
        pytest.skip("proxide is not installed; no metadata to compare")
    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert proxide.__version__ == pyproject["project"]["version"]


def _retarget_cargo(root: Path, cargo_version: str) -> None:
    """Set the workspace version and rewrite every alpha.17 pin to match it."""
    for manifest in [root / "Cargo.toml", *sorted((root / "crates").glob("*/Cargo.toml"))]:
        text = manifest.read_text(encoding="utf-8")
        manifest.write_text(text.replace("0.1.0-alpha.17", cargo_version), encoding="utf-8")


@pytest.mark.parametrize(
    ("pyproject_version", "cargo_version", "exit_code", "fragment"),
    [
        pytest.param(
            "0.1.0",
            "0.1.0-beta.1",
            0,
            "cargo stays pre-release 0.1.0-beta.1",
            id="pyproject-final-cargo-prerelease-ok",
        ),
        pytest.param(
            "0.1.0",
            "0.1.1-beta.1",
            1,
            "pyproject.toml",
            id="pyproject-final-cargo-other-release",
        ),
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
