"""Conformance: Python code must not hand-roll element inference or default elements.

tests/test_element_inference_conformance.py guards the Rust tree; this is its
Python counterpart (debt #2353). Element symbols come from the parser (Rust,
through the canonical ``proxide_core::chem::masses::infer_element``) or are
unknown (``None``) and handled loudly. Python never re-derives them from a
name's first letter -- ``name[0]`` reads "CL" as carbon, "NA" as nitrogen,
"FE" as fluorine (ledger A2) -- and never fills an unknown element with a
plausible symbol such as ``"C"`` (A1).

Detection is AST-based, on code only (comments and strings cannot switch it
on or off -- ledger B3). An *element-ish target* is a variable, attribute,
subscript key or keyword argument whose name starts with ``elem``. Rules:

1. an element-ish target assigned from an expression containing a
   *name-prefix* -- ``<name-like>[0]`` / ``[:1]`` (optionally ``.upper()``);
2. a conditional expression with a name-prefix in one branch and an
   element-symbol literal in the other (``name[0].upper() if name else "C"``);
3. an element-ish target given an element-literal fallback: a conditional
   with an element literal in a branch, ``x or ["C"] * n``, or
   ``x or "C"`` (the projector defect, review #8).

*Name-like*: an identifier containing "name" (variable or attribute), a
subscript or method call on one (``names[i][0]``, ``name.strip()[0]``), or a
comprehension variable iterating over one (``n[0] for n in atom_names``).

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


def _ident(node: ast.AST) -> str | None:
  if isinstance(node, ast.Name):
    return node.id
  if isinstance(node, ast.Attribute):
    return node.attr
  return None


def _is_element_target(node: ast.AST) -> bool:
  ident = _ident(node)
  if ident is None and isinstance(node, ast.Subscript):
    s = node.slice
    ident = s.value if isinstance(s, ast.Constant) and isinstance(s.value, str) else None
  return bool(ident and _ELEMENT_TARGET.match(ident))


def _is_element_literal(node: ast.AST) -> bool:
  return (
    isinstance(node, ast.Constant)
    and isinstance(node.value, str)
    and bool(_ELEMENT_LITERAL.match(node.value))
  )


class _Detector:
  def __init__(self, tree: ast.AST) -> None:
    # Comprehension variables iterating over a name-like iterable are name-like.
    self.name_vars: set[str] = set()
    for node in ast.walk(tree):
      if isinstance(node, ast.comprehension) and isinstance(node.target, ast.Name):
        if self._is_name_like(node.iter):
          self.name_vars.add(node.target.id)

  def _is_name_like(self, node: ast.AST) -> bool:
    ident = _ident(node)
    if ident is not None:
      return "name" in ident.lower() or ident in getattr(self, "name_vars", ())
    if isinstance(node, ast.Subscript):  # names[i]
      return self._is_name_like(node.value)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):  # name.strip()
      return self._is_name_like(node.func.value)
    return False

  def is_name_prefix(self, node: ast.AST) -> bool:
    if not isinstance(node, ast.Subscript) or not self._is_name_like(node.value):
      return False
    s = node.slice
    if isinstance(s, ast.Constant) and s.value == 0:
      return True
    return isinstance(s, ast.Slice) and (
      s.lower is None or (isinstance(s.lower, ast.Constant) and s.lower.value == 0)
    )

  def contains_name_prefix(self, node: ast.AST) -> bool:
    return any(self.is_name_prefix(n) for n in ast.walk(node))


def _has_literal_fallback(value: ast.AST) -> bool:
  """Rule 3: ``a if c else "C"``, ``x or ["C"] * n``, ``x or "C"``."""
  if isinstance(value, ast.IfExp):
    return _is_element_literal(value.body) or _is_element_literal(value.orelse)
  if isinstance(value, ast.BoolOp) and isinstance(value.op, ast.Or):
    for v in value.values[1:]:
      if _is_element_literal(v):
        return True
      if (
        isinstance(v, ast.BinOp)
        and isinstance(v.op, ast.Mult)
        and isinstance(v.left, ast.List)
        and any(_is_element_literal(e) for e in v.left.elts)
      ):
        return True
  return False


def find_violations(source: str, label: str) -> list[str]:
  """Return one finding per hand-rolled or defaulted element in ``source``."""
  tree = ast.parse(source)
  det = _Detector(tree)
  findings: list[str] = []

  def check(target: ast.AST, value: ast.AST | None, lineno: int) -> None:
    if value is None or not _is_element_target(target):
      return
    if det.contains_name_prefix(value):
      findings.append(f"{label}:{lineno}: element derived from a name's first letter")
    if _has_literal_fallback(value):
      findings.append(f"{label}:{lineno}: element given a literal fallback")

  for node in ast.walk(tree):
    if isinstance(node, ast.Assign):
      for target in node.targets:
        check(target, node.value, node.lineno)
    elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
      check(node.target, node.value, node.lineno)
    elif isinstance(node, ast.Call):
      for kw in node.keywords:
        if kw.arg and _ELEMENT_TARGET.match(kw.arg):
          check(ast.Name(id=kw.arg), kw.value, node.lineno)
    if isinstance(node, ast.IfExp):
      branches = (node.body, node.orelse)
      if any(det.contains_name_prefix(b) for b in branches) and any(
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
  assert len(fixtures) >= 3, "calibration fixtures are missing"
  per_fixture = {f.name: find_violations(f.read_text(), f.name) for f in fixtures}
  # containers.py @ 12c68e6: one line -- a name-prefix assignment (rule 1)
  # whose comprehension body is a name-prefix conditional with a literal
  # fallback (rule 2).
  assert len(per_fixture["containers_12c68e6.py.txt"]) == 2, per_fixture
  # writing.py @ d4ac812: `elements = [name[0] ...]` and
  # `element = ... else atom_name[0]`.
  assert len(per_fixture["writing_d4ac812.py.txt"]) == 2, per_fixture
  # projector.py @ 495ef10: four literal fallbacks (two `or ["C"] * n`, two
  # `else "C"`); the getBySymbol("C") line is a call, not a fallback literal
  # in an element-ish target, and is caught via `elem_str` upstream.
  assert len(per_fixture["projector_495ef10.py.txt"]) >= 4, per_fixture


def test_detector_ignores_closed_vocabulary_lookups_and_comments() -> None:
  src = (
    "atom1_radius = van_der_waals_radius[atom1_name[0]]\n"
    "# elements = [name[0] for name in atom_names]\n"
    "doc = 'elements = [name[0] for name in atom_names]'\n"
    "chain = x[0] if name else 'A'\n"
  )
  assert find_violations(src, "snippet") == []


def test_detector_catches_the_spellings_review_8_listed() -> None:
  for src in [
    "elements = [n[0] for n in atom_names]\n",
    "elements = [names[i][0] for i in range(3)]\n",
    "elem = name.strip()[0]\n",
    "self.elements = [n[0] for n in atom_names]\n",
    "d['element'] = atom_name[0]\n",
    "Protein(elements=[n[0] for n in atom_names])\n",
    "elements = topology.elements or ['C'] * n\n",
  ]:
    assert find_violations(src, "snippet"), src


def test_no_hand_rolled_or_defaulted_elements_in_src() -> None:
  n_files, findings = _scan_src()
  assert n_files > 50, f"scanned only {n_files} files under {_SRC}; the scan is not covering src"
  assert findings == [], "hand-rolled / defaulted elements:\n" + "\n".join(findings)
