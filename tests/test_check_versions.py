"""Tests for scripts/check_versions.py: the real repo agrees, and a bad pin is caught."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "check_versions.py"


def _run_check(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_real_repo_versions_agree() -> None:
    result = _run_check(REPO)
    assert result.returncode == 0, result.stderr


def test_mismatched_crate_pin_is_reported(tmp_path: Path) -> None:
    shutil.copy(REPO / "pyproject.toml", tmp_path / "pyproject.toml")
    cargo_text = (REPO / "Cargo.toml").read_text(encoding="utf-8")
    pin = 'proxide-core = { path = "crates/proxide-core", version = "0.1.0-alpha.17" }'
    assert cargo_text.count(pin) == 1
    broken = pin.replace('"0.1.0-alpha.17"', '"0.1.0-alpha.15"')
    (tmp_path / "Cargo.toml").write_text(cargo_text.replace(pin, broken), encoding="utf-8")

    result = _run_check(tmp_path)
    assert result.returncode == 1, result.stderr
    assert "proxide-core" in result.stderr
    assert "0.1.0-alpha.15" in result.stderr
