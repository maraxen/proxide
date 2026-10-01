"""Conformance: Python code must not hand-roll element inference from atom names.

tests/test_element_inference_conformance.py guards the Rust tree; this is its
Python counterpart (debt #2353). Element symbols come from the parser (Rust,
through the canonical ``proxide_core::chem::masses::infer_element``) or are
unknown (``None``). Python never re-derives them from a name's first letter:
``name[0]`` reads "CL" as carbon, "NA" as nitrogen, "FE" as fluorine (ledger
A2), and the habitual ``else "C"`` turns an empty name into carbon (A1).

Detection is AST-based, on code only (comments and strings cannot switch it
on or off -- ledger B3), with two rules:

1. a target named ``element``/``elements``/``elem*`` assigned from an
   expression containing ``<name-like>[0]`` (optionally ``.upper()``-ed);
2. a conditional expression with ``<name-like>[0]`` in one branch and an
   element-symbol string literal in the other (the ``else "C"`` fallback).

"name-like" means a variable or attribute whose identifier contains "name".

CALIBRATION (ledger B2/B7): the detector must fire on the real historical
defects, copied verbatim into tests/data/element_inference_calibration/ with
their source commit. A detector that finds nothing there is broken, not
satisfied. The repository scan must also cover a non-trivial number of files,
so an empty scan cannot read as success.

Closed-vocabulary lookups that raise on an unknown key -- e.g. residues.py's
``van_der_waals_radius[atom1_name[0]]`` over the 20 standard residues' Atom14
names -- are not flagged: nothing is defaulted and nothing is silent.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src" / "proxide"
_CALIBRATION = _REPO / "tests" / "data" / "element_inference_calibration"
_ELEMENT_TARGET = re.compile(r"^elem", re.IGNORECASE)
_ELEMENT_LITERAL = re.compile(r"^[A-Z][a-z]?$")


def _is_name_prefix(node: ast.AST) -> bool:
  """``<name-like>[0]`` (or ``[:1]``/``[0:2]``) -- the first-letter idiom."""
  if not isinstance(node, ast.Subscript):
    return False
  base = node.value
  ident = base.id if isinstance(base, ast.Name) else getattr(base, "attr", None)
  if not ident or "name" not in ident.lower():
    return False
  s = node.slice
  if isinstance(s, ast.Constant) and s.value == 0:
    return True
  return isinstance(s, ast.Slice) and (
    s.lower is None or (isinstance(s.lower, ast.Constant) and s.lower.value == 0)
  )


def _contains_name_prefix(node: ast.AST) -> bool:
  return any(_is_name_prefix(n) for n in ast.walk(node))


def _is_element_literal(node: ast.AST) -> bool:
  return (
    isinstance(node, ast.Constant)
    and isinstance(node.value, str)
    and bool(_ELEMENT_LITERAL.match(node.value))
  )


def find_violations(source: str, label: str) -> list[str]:
  """Return one finding per hand-rolled element inference in ``source``."""
  findings: list[str] = []
  for node in ast.walk(ast.parse(source)):
    targets: list[ast.AST] = []
    value = None
    if isinstance(node, ast.Assign):
      targets, value = node.targets, node.value
    elif isinstance(node, (ast.AnnAssign, ast.AugAssign)) and node.value is not None:
      targets, value = [node.target], node.value
    for target in targets:
      if (
        isinstance(target, ast.Name)
        and _ELEMENT_TARGET.match(target.id)
        and value is not None
        and _contains_name_prefix(value)
      ):
        findings.append(f"{label}:{node.lineno}: '{target.id}' derived from a name's first letter")
    if isinstance(node, ast.IfExp):
      branches = (node.body, node.orelse)
      if any(_contains_name_prefix(b) for b in branches) and any(
        _is_element_literal(b) for b in branches
      ):
        findings.append(f"{label}:{node.lineno}: name-prefix element with a literal fallback")
  return findings


def _scan_src() -> tuple[int, list[str]]:
  files = sorted(_SRC.rglob("*.py"))
  findings: list[str] = []
  for path in files:
    findings += find_violations(path.read_text(), str(path.relative_to(_REPO)))
  return len(files), findings


def test_detector_fires_on_the_real_historical_defects() -> None:
  fixtures = sorted(_CALIBRATION.glob("*.py.txt"))
  assert len(fixtures) >= 2, "calibration fixtures are missing"
  per_fixture = {f.name: find_violations(f.read_text(), f.name) for f in fixtures}
  # containers.py @ 12c68e6: one assignment, which is also a literal fallback.
  assert len(per_fixture["containers_12c68e6.py.txt"]) == 2, per_fixture
  # writing.py @ d4ac812: the `elements = [...]` and `element = ... else atom_name[0]` lines.
  assert len(per_fixture["writing_d4ac812.py.txt"]) == 2, per_fixture


def test_detector_ignores_closed_vocabulary_lookups_and_comments() -> None:
  src = (
    "atom1_radius = van_der_waals_radius[atom1_name[0]]\n"
    "# elements = [name[0] for name in atom_names]\n"
    "doc = 'elements = [name[0] for name in atom_names]'\n"
  )
  assert find_violations(src, "snippet") == []


def test_no_hand_rolled_element_inference_in_src() -> None:
  n_files, findings = _scan_src()
  assert n_files > 50, f"scanned only {n_files} files under {_SRC}; the scan is not covering src"
  assert findings == [], "hand-rolled element inference:\n" + "\n".join(findings)
