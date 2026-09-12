"""Conformance: element inference from PDB atom names must use the canonical implementation.

WHY THIS TEST MATTERS

Inferring chemical elements from PDB atom names is subtle: PDB ATOM/HETATM records contain
atom names in ALL-CAPS (e.g., "CL" for chloride), which must resolve to title-case element
symbols (e.g., "Cl"). The most treacherous case is "CA" (alpha-carbon), which must stay
"C" (carbon), not become "Ca" (calcium).

proxide has ONE correct, two-letter-aware implementation in crates/proxide-core/src/chem/masses.rs:63.
Over time, SIX+ other modules reimplemented this by hand, and every one got it wrong.
This test guards against the seventh.

PRECEDENT

This test mirrors tests/test_alphabet_conformance.py: behavioural (do the constants agree?)
plus structural (are there duplicate implementations that might diverge?).

HOW TO RUN

  # Behavioural: Rust comprehensive corpus test (no build)
  cargo test -p proxide-core test_infer_element_comprehensive_corpus --lib

  # Structural: Python scanner (no Rust compilation)
  uv run --no-project python tests/test_element_inference_conformance.py

  # Both together as pytest plugin (requires maturin build):
  uv run pytest tests/test_element_inference_conformance.py::test_comprehensive_element_inference

  NOTE: Avoid bare `uv run pytest` (without --no-project). It triggers proxide_py build
  which fails because ATOMTYPE_GFF2.DEF is deliberately absent/gitignored.
"""

from __future__ import annotations

import subprocess
import re
from pathlib import Path


# Known violations accepted for a stated reason, keyed by path prefix.
#
# An entry here is a debt record, not a silence: the scan still finds the site,
# still prints it, and the justification must say what makes the deferral
# acceptable and what ends it. A path listed here that no longer violates fails
# the test -- an exemption outliving its cause is itself a silent hole.
# `expected_findings` bounds the exemption to the violations that were actually
# reviewed. A path prefix alone would blanket the file forever, so a violation
# added tomorrow would inherit today's justification silently. More findings
# than recorded fails as an unreviewed addition; zero fails as stale.
DEFERRED_VIOLATIONS: dict[str, dict[str, object]] = {
    # Empty on purpose. The gbsa.rs entry that lived here expired on
    # 2026-09-12: its blocking condition was 'authoritative mbondi2
    # parameters for Se, Na, Cu and Fe are sourced'. They were sourced, and
    # the authoritative answer is that mbondi2 does not define those
    # elements at all -- the reference implementation substitutes a
    # documented catch-all. gbsa.rs now routes through infer_element and
    # tags every catch-all it applies, so the violation is gone rather than
    # excused. See crates/proxide-physics/data/mbondi2.xml.
}


def test_comprehensive_element_inference() -> None:
    """Comprehensive element inference conformance (behavioural + structural).

    Can be run standalone (no pytest) or as pytest plugin.
    See module docstring for invocation.
    """
    # Part 1: Behavioural — Rust comprehensive corpus
    print("\n" + "=" * 70)
    print("1. BEHAVIOURAL TEST — Rust comprehensive corpus")
    print("=" * 70)

    result = subprocess.run(
        ["cargo", "test", "-p", "proxide-core", "test_infer_element_comprehensive_corpus", "--lib"],
        capture_output=True,
        text=True,
        env={**subprocess.os.environ, "RUSTC_WRAPPER": ""},
    )

    if result.returncode != 0:
        print("✗ FAILED")
        print(result.stderr)
        raise AssertionError("Comprehensive corpus test failed")

    for line in result.stdout.split("\n"):
        if "test result:" in line:
            print(f"✓ PASS — {line.strip()}")
            break

    # Part 2: Structural — scan for antipatterns
    print("\n" + "=" * 70)
    print("2. STRUCTURAL TEST — Scan for element inference antipatterns")
    print("=" * 70)

    violations = _scan_for_element_inference_antipatterns()

    unexempted: list[str] = []
    counts: dict[str, int] = {path: 0 for path in DEFERRED_VIOLATIONS}
    for violation in violations:
        for deferred_path in DEFERRED_VIOLATIONS:
            if violation.startswith(deferred_path):
                counts[deferred_path] += 1
                break
        else:
            unexempted.append(violation)

    if unexempted:
        msg = "\n".join(unexempted)
        print(f"✗ FAILED — Found {len(unexempted)} violations:\n{msg}")
        raise AssertionError(msg)

    for deferred_path, entry in sorted(DEFERRED_VIOLATIONS.items()):
        expected = entry["expected_findings"]
        found = counts[deferred_path]
        if found == 0:
            raise AssertionError(
                f"Stale entry in DEFERRED_VIOLATIONS: {deferred_path} no longer "
                "violates, so its exemption must be deleted."
            )
        if found != expected:
            raise AssertionError(
                f"{deferred_path} now yields {found} findings but its exemption "
                f"covers {expected}. An exemption only excuses the violations that "
                "were actually reviewed — fix the new one, or review it and update "
                "expected_findings with a reason."
            )
        print(f"[DEFERRED] {deferred_path} ({found} reviewed findings)\n    {entry['reason']}")

    print("✓ PASS — No unexempted hand-rolled element inference detected")


def _scan_for_element_inference_antipatterns() -> list[str]:
    """Scan crates for element-from-atom-name inference antipatterns.

    Detects two forms:

    PATTERN 1 (direct):
        let element = atom_name.chars().next().unwrap();
        match element { 'C' => ..., 'N' => ..., ... }

    PATTERN 2 (indirect, via intermediate):
        let trimmed = name.trim_start_matches(...);
        match trimmed.chars().next() { 'C' => ..., 'N' => ..., ... }

    Both bypass infer_element() and cause silent chemistry errors (Cl→C, bad radii).

    JUSTIFIED EXCLUSIONS:
    - Files importing infer_element() (doing it right)
    - masses.rs (the canonical implementation)
    - Single-char field extraction (alt_loc, i_code, etc. for metadata, not elements)
    - Residue-code lookup (RESTYPE_1TO3, three_to_one)
    - Element-symbol-to-number mapping ("CL" => Some(17) is lookup, not inference)
    - Atom-type dispatch on already-resolved tokens (gaff* files)
    - get_mass/get_radius calls (already-resolved element symbols)
    """
    crates_dir = Path(__file__).parent.parent / "crates"

    # Scan ALL Rust files, exclude by rule
    rust_files = sorted(crates_dir.rglob("*.rs"))
    rust_files = [f for f in rust_files if f.is_file() and not f.name.startswith(".")]

    violations = []

    for rust_file in rust_files:
        with open(rust_file, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            lines = content.split("\n")

        # Exclusion: file already uses canonical infer_element().
        #
        # Checked against comment-stripped source, not raw content. A substring
        # test over raw content lets a COMMENT switch the detector off for a
        # whole file: adding "// TODO: route through infer_element" to a
        # defective file makes its violations vanish, which then reads as
        # "fixed" to the stale-exemption check below and retires the exemption
        # on a file that never changed. The detector could not tell fixed from
        # invisible. Requiring a call or an import, in code, closes that.
        code_only = "\n".join(
            line for line in lines if not line.lstrip().startswith(("//", "*", "/*"))
        )
        if "infer_element(" in code_only or "use proxide_core::chem::masses" in code_only:
            continue

        # Exclusion: masses.rs is the canonical implementation
        if "masses.rs" in str(rust_file):
            continue

        # Scan for: .chars().next() on atom-name-derived variable with element dispatch
        for i, line in enumerate(lines, start=1):
            # Skip comments
            if line.strip().startswith("//") or line.strip().startswith("/*"):
                continue

            # Skip known safe patterns
            if any(safe in line for safe in [
                "alt_loc", "i_code", "label_alt_id", "pdbx_PDB_ins_code",
                "insertion code", "alt location", "RESTYPE_1TO3", "three_to_one",
                "one.chars()", "get_mass", "get_radius"
            ]):
                continue

            # Look for: .chars().next() on atom-name or derived variable
            if ".chars().next()" not in line:
                continue

            # Heuristic: variable name suggests atom-name derivation
            if not any(kw in line.lower() for kw in ["atom", "name", "element", "trimmed", "stripped"]):
                continue

            # Does context have element-symbol dispatch?
            context = "\n".join(lines[max(0, i - 2) : min(len(lines), i + 20)])
            if _has_element_symbol_dispatch(context):
                violations.append(
                    f"{rust_file.relative_to(crates_dir)}:{i}\n"
                    f"  {line.strip()}\n"
                    f"  → .chars().next() with element dispatch on atom-derived variable"
                )

        # Also check for indirect form: let trimmed = name.trim_*; ... trimmed.chars().next()
        for i, line in enumerate(lines, start=1):
            if ".trim" not in line or "name" not in line.lower():
                continue

            # Extract variable name from: let VAR = name.trim*(...)
            match = re.search(r"let\s+(\w+)\s*=\s*\w*name\w*\..*?(?:trim|slice)", line)
            if not match:
                continue

            var_name = match.group(1)
            context = "\n".join(lines[max(0, i) : min(len(lines), i + 20)])

            # Check if this variable is used in element dispatch
            if re.search(rf"{var_name}\s*\.chars\(\)\.next\(\)", context):
                if _has_element_symbol_dispatch(context) or _element_inference_in_context(context):
                    violations.append(
                        f"{rust_file.relative_to(crates_dir)}:{i}\n"
                        f"  {line.strip()}\n"
                        f"  → atom-name-derived var {var_name} used in element inference"
                    )

    return violations


def _has_element_symbol_dispatch(code: str) -> bool:
    """Check if code contains dispatch on element symbols (match arms).

    Looks for:
      'H' =>, 'C' =>, "Cl" =>, "Na" =>  (direct)
      Some('H') =>, Some('C') =>  (wrapped in Option)

    Excludes: "CL" => 17 (element-to-number, not inference).
    """
    # Pattern 1: Direct match arms: 'X' => or "Xx" => where X/Xx is element symbol
    if re.search(
        r"(['\"])([HCNOSFPIK]|Cl|Br|Na|Mg|Zn|Fe|Cu|Mn|Se|Ca)\1\s*=>",
        code,
    ):
        # But exclude element-to-number mappings ("CL" => 17)
        if not re.search(r"['\"]([A-Z]{2})['\"].*?=>\s*\d+", code):
            return True

    # Pattern 2: Option-wrapped match arms: Some('X') => or Some("Xx") =>
    if re.search(
        r"Some\s*\(\s*(['\"])([HCNOSFPIK]|Cl|Br|Na|Mg|Zn|Fe|Cu|Mn|Se|Ca)\1\s*\)\s*=>",
        code,
    ):
        return True

    return False


def _element_inference_in_context(code: str) -> bool:
    """Check for subtle element-inference patterns (non-match-statement forms).

    Detects patterns like:
      let upper = first_char.to_uppercase().to_string();
      return format!("{}{}", upper, second_char);

    This is element inference via string manipulation, not explicit dispatch.
    """
    # Pattern: to_uppercase() used on a char extracted from atom name,
    # then formatted/concatenated to build element symbols
    if "to_uppercase()" in code and ("format!" in code or "to_string()" in code):
        # Verify this is in the context of a function that looks like it's
        # inferring elements (name in function or variable names suggests it)
        if "extract_element" in code or "element" in code.lower():
            return True

    return False


if __name__ == "__main__":
    try:
        test_comprehensive_element_inference()
        print("\n" + "=" * 70)
        print("ALL TESTS PASSED ✓")
        print("=" * 70)
    except AssertionError as e:
        print(f"\n{e}")
        exit(1)
