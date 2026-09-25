"""Verify the mbondi2 table against the upstream source it cites.

`crates/proxide-physics/data/mbondi2.xml` carries a provenance block naming
OpenMM's `customgbforces.py` at a pinned revision. A citation nobody checks is
worth nothing -- ledger B6: an "uncited constant" counter goes down the moment
somebody types a citation string, so the gate has to require diff-reviewable
evidence rather than an assertion that the work was done.

So this test does not trust the citation. It re-derives the table directly from
that upstream file's AST and compares value by value. If upstream changes, or if
somebody edits our XML, this fails and names the element.

Parsing is done with `ast` rather than by importing OpenMM, so the check works
against a bare source checkout with no OpenMM installation and no import side
effects.

Locating the upstream source, in order:
  1. $PROXIDE_OPENMM_SOURCE -- path to customgbforces.py or an OpenMM checkout
  2. an importable `openmm` package
  3. known sibling checkouts on this machine

Run standalone:
    uv run --no-project python tests/test_gb_parameter_provenance.py
"""

from __future__ import annotations

import ast
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TABLE_PATH = _REPO_ROOT / "crates" / "proxide-physics" / "data" / "mbondi2.xml"

_RELATIVE_SOURCE = Path("openmm") / "app" / "internal" / "customgbforces.py"

# OpenMM spells elements as attributes of its `element` module; our table uses
# IUPAC symbols. Only the elements mbondi2 can mention need to be here.
_ELEMENT_SYMBOLS = {
    "hydrogen": "H",
    "deuterium": "D",
    "carbon": "C",
    "nitrogen": "N",
    "oxygen": "O",
    "fluorine": "F",
    "silicon": "Si",
    "phosphorus": "P",
    "sulfur": "S",
    "chlorine": "Cl",
}


class UpstreamNotFound(Exception):
    """The cited upstream source could not be located on this machine."""


def _candidate_paths() -> list[Path]:
    candidates: list[Path] = []

    env = os.environ.get("PROXIDE_OPENMM_SOURCE")
    if env:
        p = Path(env).expanduser()
        candidates.append(p if p.is_file() else p / _RELATIVE_SOURCE)

    try:
        import openmm  # noqa: PLC0415

        candidates.append(Path(openmm.__file__).parent / "app" / "internal" / "customgbforces.py")
    except Exception:
        pass

    # Sibling project checkouts. OpenMM is a heavy optional dependency and is
    # deliberately not added to this repo's environment just to run a gate.
    home = Path.home()
    candidates.append(
        home / "projects" / "prolix" / "submodules" / "openmm"
        / "wrappers" / "python" / _RELATIVE_SOURCE
    )
    for venv in sorted((home / "projects").glob("*/.venv/lib/python3*/site-packages")):
        candidates.append(venv / _RELATIVE_SOURCE)

    return candidates


def find_upstream_source() -> Path:
    for candidate in _candidate_paths():
        if candidate.is_file():
            return candidate
    raise UpstreamNotFound(
        "Could not locate OpenMM's customgbforces.py, so the mbondi2 table's "
        "citation could NOT be verified. Set PROXIDE_OPENMM_SOURCE to a path to "
        "that file or to an OpenMM checkout, or install openmm."
    )


def _element_symbol(node: ast.expr) -> str | None:
    """Map an `E.<element>` attribute node to its IUPAC symbol."""
    if isinstance(node, ast.Attribute):
        return _ELEMENT_SYMBOLS.get(node.attr)
    return None


def _number(node: ast.expr) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _number(node.operand)
        return None if inner is None else -inner
    return None


def parse_upstream(source_path: Path) -> dict[str, object]:
    """Extract mbondi2 radii, the catch-all, and OBC screening from upstream."""
    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    radii: dict[str, float] = {}
    default_radius: float | None = None
    hydrogen_rule: dict[str, float] = {}
    screen: dict[str, float] = {}
    default_screen: float | None = None

    for node in ast.walk(tree):
        # --- module-level _SCREEN_PARAMETERS ---------------------------------
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "_SCREEN_PARAMETERS" for t in node.targets
        ):
            if isinstance(node.value, ast.Dict):
                for key, value in zip(node.value.keys, node.value.values):
                    if not isinstance(value, ast.Tuple) or not value.elts:
                        continue
                    # Column 0 is the "normal" (OBC/HCT) column.
                    first = _number(value.elts[0])
                    if first is None:
                        continue
                    if key is None:
                        continue
                    if isinstance(key, ast.Constant) and key.value is None:
                        default_screen = first
                        continue
                    symbol = _element_symbol(key)
                    if symbol is not None:
                        screen[symbol] = first

        # --- def _mbondi2_radii(...) -----------------------------------------
        if isinstance(node, ast.FunctionDef) and node.name == "_mbondi2_radii":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Assign):
                    targets = [t.id for t in sub.targets if isinstance(t, ast.Name)]
                    if "default_radius" in targets:
                        default_radius = _number(sub.value)
                    if "element_to_const_radius" in targets and isinstance(sub.value, ast.Dict):
                        for key, value in zip(sub.value.keys, sub.value.values):
                            symbol = _element_symbol(key) if key is not None else None
                            number = _number(value)
                            if symbol is not None and number is not None:
                                radii[symbol] = number
                    # Carbon is special-cased in the body rather than tabled.
                    if (
                        len(sub.targets) == 1
                        and isinstance(sub.targets[0], ast.Subscript)
                        and isinstance(sub.targets[0].value, ast.Name)
                        and sub.targets[0].value.id == "radii"
                    ):
                        number = _number(sub.value)
                        if number is not None:
                            hydrogen_rule.setdefault("_seen", 0.0)
                            hydrogen_rule[f"assign_{len(hydrogen_rule)}"] = number

    return {
        "radii": radii,
        "default_radius": default_radius,
        "screen": screen,
        "default_screen": default_screen,
        "carbon_and_hydrogen_literals": sorted(
            {v for k, v in hydrogen_rule.items() if k != "_seen"}
        ),
    }


def parse_our_table() -> dict[str, object]:
    root = ET.parse(_TABLE_PATH).getroot()

    radii = {
        e.attrib["symbol"]: float(e.attrib["radius"])
        for e in root.findall("./Radii/Element")
    }
    screen = {
        e.attrib["symbol"]: float(e.attrib["screen"])
        for e in root.findall("./Screen/Element")
    }
    h_rule = root.find("./HydrogenRule")
    fallback = root.find("./Fallback")
    assert h_rule is not None, "table has no <HydrogenRule>"
    assert fallback is not None, "table has no <Fallback>"

    bonded_to = h_rule.find("./BondedTo")
    otherwise = h_rule.find("./Otherwise")
    assert bonded_to is not None and otherwise is not None

    return {
        "radii": radii,
        "screen": screen,
        "h_bonded_to_n": float(bonded_to.attrib["radius"]),
        "h_otherwise": float(otherwise.attrib["radius"]),
        "fallback_radius": float(fallback.attrib["radius"]),
        "fallback_screen": float(fallback.attrib["screen"]),
    }


def test_mbondi2_table_matches_cited_upstream() -> None:
    """Re-derive the table from the source it cites and diff it."""
    source = find_upstream_source()
    upstream = parse_upstream(source)
    ours = parse_our_table()

    errors: list[str] = []

    # --- radii, excluding carbon (special-cased upstream, tabled here) -------
    upstream_radii: dict[str, float] = dict(upstream["radii"])  # type: ignore[arg-type]
    our_radii: dict[str, float] = dict(ours["radii"])  # type: ignore[arg-type]

    our_radii_no_c = {k: v for k, v in our_radii.items() if k != "C"}
    if set(our_radii_no_c) != set(upstream_radii):
        only_ours = sorted(set(our_radii_no_c) - set(upstream_radii))
        only_theirs = sorted(set(upstream_radii) - set(our_radii_no_c))
        if only_ours:
            errors.append(
                f"our table defines radii upstream does NOT: {only_ours}. "
                "mbondi2 does not cover these -- adding a plausible value from "
                "another radius set is the silent substitution this gate exists "
                "to catch."
            )
        if only_theirs:
            errors.append(f"upstream defines radii we are missing: {only_theirs}")

    for symbol in sorted(set(our_radii_no_c) & set(upstream_radii)):
        if abs(our_radii_no_c[symbol] - upstream_radii[symbol]) > 1e-9:
            errors.append(
                f"radius[{symbol}]: ours={our_radii_no_c[symbol]} "
                f"upstream={upstream_radii[symbol]}"
            )

    # --- catch-all ----------------------------------------------------------
    if upstream["default_radius"] is None:
        errors.append("could not find `default_radius` in upstream _mbondi2_radii")
    elif abs(ours["fallback_radius"] - upstream["default_radius"]) > 1e-9:  # type: ignore[operator]
        errors.append(
            f"fallback radius: ours={ours['fallback_radius']} "
            f"upstream={upstream['default_radius']}"
        )

    if upstream["default_screen"] is None:
        errors.append("could not find the None-keyed default in _SCREEN_PARAMETERS")
    elif abs(ours["fallback_screen"] - upstream["default_screen"]) > 1e-9:  # type: ignore[operator]
        errors.append(
            f"fallback screen: ours={ours['fallback_screen']} "
            f"upstream={upstream['default_screen']}"
        )

    # --- screening ----------------------------------------------------------
    upstream_screen: dict[str, float] = dict(upstream["screen"])  # type: ignore[arg-type]
    our_screen: dict[str, float] = dict(ours["screen"])  # type: ignore[arg-type]
    # Upstream also keys deuterium; we resolve D as H and do not table it.
    upstream_screen.pop("D", None)

    if set(our_screen) != set(upstream_screen):
        errors.append(
            f"screening element coverage differs: ours={sorted(our_screen)} "
            f"upstream={sorted(upstream_screen)}"
        )
    for symbol in sorted(set(our_screen) & set(upstream_screen)):
        if abs(our_screen[symbol] - upstream_screen[symbol]) > 1e-9:
            errors.append(
                f"screen[{symbol}]: ours={our_screen[symbol]} "
                f"upstream={upstream_screen[symbol]}"
            )

    # --- hydrogen rule + carbon ---------------------------------------------
    # Upstream writes these as literals in the function body: 1.3 (H bonded to
    # N), 1.2 (H otherwise) and 1.7 (carbon). Assert each is present rather
    # than relying on ordering.
    literals = set(upstream["carbon_and_hydrogen_literals"])  # type: ignore[arg-type]
    for label, value in (
        ("hydrogen bonded to N", ours["h_bonded_to_n"]),
        ("hydrogen otherwise", ours["h_otherwise"]),
        ("carbon", our_radii.get("C")),
    ):
        if value is None or not any(abs(value - lit) < 1e-9 for lit in literals):
            errors.append(
                f"{label}: our value {value} does not appear among the literals "
                f"upstream assigns in _mbondi2_radii ({sorted(literals)})"
            )

    if errors:
        raise AssertionError(
            f"mbondi2 table disagrees with its cited upstream ({source}):\n  "
            + "\n  ".join(errors)
        )


def main() -> int:
    print("=" * 70)
    print("mbondi2 PROVENANCE — re-derive the table from its cited upstream")
    print("=" * 70)
    try:
        source = find_upstream_source()
    except UpstreamNotFound as exc:
        # Ledger B3: silence must never be indistinguishable from success.
        print(f"\n⚠ NOT VERIFIED — {exc}")
        print("\nThis is a SKIP, not a pass. The citation remains unchecked.")
        return 0

    print(f"\nupstream: {source}")
    try:
        test_mbondi2_table_matches_cited_upstream()
    except AssertionError as exc:
        print(f"\n✗ FAILED\n{exc}")
        return 1
    print("\n✓ PASS — every value matches the cited upstream implementation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
