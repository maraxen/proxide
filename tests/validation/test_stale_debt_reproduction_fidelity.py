"""The stale-debt reproduction checks must observe the defect, not a text stand-in.

Debt #2363 (Rust) at 48b1c5c accepts malformed f8/f9 lines. A comment and two
absent substrings are not that acceptance: if the unmatched-"[" arm returns
None, the reproduction is gone even though the comment remains.

Debt #1905 at 1d4c925 fills a missing c3-c-c3 angle with kt = 0.0 and t0 = 0.0.
Matching any missing angle that merely contains "c3", or ignoring the constants
on the angle that was filled, reports the defect when it is absent.
"""

from __future__ import annotations

import importlib.util
import subprocess
import types
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO / "scripts" / "validation" / "verify_stale_debts.py"
_RS_COMMIT = "48b1c5c"
_RS_PATH = "crates/proxide-gaff2/src/def_parser.rs"
_UNMATCHED_COMMENT = 'unmatched "[" — leave remaining untouched'
_ABSENT_MARKERS = (
    "is_well_formed",
    "test_malformed_f8_f9_bodies_are_errors_like_python",
)


def _load_verifier():
    spec = importlib.util.spec_from_file_location("verify_stale_debts_fidelity", _SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _historical_def_parser() -> str:
    proc = subprocess.run(
        ["git", "show", f"{_RS_COMMIT}:{_RS_PATH}"],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout


def _reject_unclosed_bracket(src: str) -> str:
    """Make the unmatched-'[' arm return None. Leave the comment in place."""
    marker = "// else: unmatched"
    comment_at = src.find(marker)
    assert comment_at != -1
    close_at = src.rfind("        }\n", 0, comment_at)
    assert close_at != -1
    suffix = src[close_at:]
    assert suffix.startswith("        }\n")
    return (
        src[:close_at]
        + "        } else {\n            return None;\n        }\n"
        + suffix[len("        }\n") :]
    )


def test_debt_2363_rs_reproduction_false_when_unclosed_bracket_returns_none(monkeypatch):
    """Comment and absent substrings stay; the parser no longer keeps the rule."""
    src = _historical_def_parser()
    assert _UNMATCHED_COMMENT in src
    for marker in _ABSENT_MARKERS:
        assert marker not in src

    modified = _reject_unclosed_bracket(src)
    assert _UNMATCHED_COMMENT in modified
    for marker in _ABSENT_MARKERS:
        assert marker not in modified
    assert "return None;" in modified

    module = _load_verifier()
    real_run = subprocess.run

    def fake_run(cmd, *args, **kwargs):
        argv = list(cmd) if isinstance(cmd, (list, tuple)) else []
        if argv[:2] == ["git", "show"] and any(part.endswith(_RS_PATH) or _RS_PATH in part for part in argv):
            return subprocess.CompletedProcess(argv, 0, stdout=modified, stderr="")
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    assert module.verify_debt_2363_rs_reproduction() is False


class _HistoricalLoader:
    def __init__(self, result: dict):
        self._result = result

    def create_module(self, spec):
        return None

    def exec_module(self, module: types.ModuleType) -> None:
        result = self._result
        module._default_rules = None
        module._default_wildatom = None
        module.load_gaff2_parameters = lambda *a, **k: {"angles": {}}

        def parameterize(mol, charge_method="gasteiger"):
            return result

        module.parameterize_gaff_with_rdkit = parameterize


def _reproduction_with_result(monkeypatch, result: dict) -> bool:
    module = _load_verifier()
    real_spec = module.importlib.util.spec_from_file_location

    def fake_spec(name, location, *args, **kwargs):
        if name != "gaff2_prefix_1905":
            return real_spec(name, location, *args, **kwargs)
        return module.importlib.util.spec_from_loader(name, _HistoricalLoader(result))

    monkeypatch.setattr(module.importlib.util, "spec_from_file_location", fake_spec)
    return module.verify_debt_1905_reproduction()


def test_debt_1905_reproduction_false_when_c3_c_c3_is_restored(monkeypatch):
    """A different missing c3 angle is not the c3-c-c3 zero fill."""
    result = {
        "missing_params": [
            {"term": "angle", "types": ["c3", "c", "o"]},
        ],
        "angles": [
            {"types": ("c3", "c", "c3"), "kt": 59.15, "t0": 116.68},
            {"types": ("c3", "c", "o"), "kt": 0.0, "t0": 0.0},
        ],
    }
    assert _reproduction_with_result(monkeypatch, result) is False


def test_debt_1905_reproduction_false_when_c3_c_c3_fill_is_nonzero(monkeypatch):
    """Listing c3-c-c3 as missing is not the defect when the angle constants are not 0.0."""
    result = {
        "missing_params": [
            {"term": "angle", "types": ["c3", "c", "c3"]},
        ],
        "angles": [
            {"types": ("c3", "c", "c3"), "kt": 59.15, "t0": 116.68},
        ],
    }
    assert _reproduction_with_result(monkeypatch, result) is False
