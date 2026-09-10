"""Conformance: element inference from PDB atom names must use the canonical implementation.

WHY THIS TEST MATTERS

Inferring chemical elements from PDB atom names is subtle: PDB ATOM/HETATM records contain
atom names in ALL-CAPS (e.g., "CL" for chloride), which must resolve to title-case element
symbols (e.g., "Cl"). The most treacherous case is "CA" (alpha-carbon in protein backbones),
which must stay "C" (carbon), not become "Ca" (calcium).

proxide has ONE correct, two-letter-aware implementation:
`crates/proxide-core/src/chem/masses.rs:63` — `pub fn infer_element(atom_name: &str) -> &str`

Over time, at least SIX other modules independently reimplemented this inference by hand,
and every single one got it wrong. Three reimplementations were fixed in commit c24546a
(proxide-io parsers, where the most subtle bugs hide). Another agent is fixing three more
right now in proxide_fixer. But fixing instances does not stop the seventh from being written.

This test guards against the anti-pattern: any module that reimplements element inference
by dispatching on atom-name characters outside of `masses.rs` will fail this test with a
message pointing back to the canonical implementation.

PRECEDENT

This test mirrors `tests/test_alphabet_conformance.py`, which guards against silent
copy-drift in amino-acid ordering declarations. That test combines behavioural assertions
(do the constants agree with upstream?) and structural assertions (are there duplicate
declarations that might diverge?). This test applies the same two-pronged approach to
element inference, specific to the chemical space where the code actually operates.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


def test_infer_element_returns_correct_symbols() -> None:
    """Behavioural: infer_element() returns correct element symbols for all supported atoms.

    This test pins the correct behaviour for the critical atoms the codebase handles, with
    special attention to the two-letter elements (Cl, Br, Na, Mg, Zn, Fe, Cu, Mn, Se, I, K)
    and the alpha-carbon edge case (CA -> C, not Ca).

    The test runs via `cargo test` (disabling sccache to avoid permission issues).
    """
    # Use cargo test to run the infer_element tests in masses.rs
    env = {"RUSTC_WRAPPER": ""}  # Disable sccache to avoid permission issues in sandbox
    result = subprocess.run(
        ["cargo", "test", "-p", "proxide-core", "test_infer_element", "--lib", "--", "--nocapture"],
        cwd=Path(__file__).parent.parent,
        capture_output=True,
        text=True,
        env={**subprocess.os.environ, **env},
    )
    assert result.returncode == 0, f"infer_element tests failed:\n{result.stderr}\n{result.stdout}"


def test_no_hand_rolled_element_inference_outside_masses_rs() -> None:
    """Structural anti-duplication guard: scan for element inference reimplemented by hand.

    Searches Rust sources for the anti-pattern: dispatching on the first 1-2 characters of an
    atom-name-derived variable whose arms return element symbols. The canonical implementation
    is infer_element() in crates/proxide-core/src/chem/masses.rs, and it must be used instead.

    ALLOWED EXCEPTIONS (excluded from scan):
    - Single-character field extraction: .chars().next() on fields with fixed, known names
      (alt_loc, i_code, label_alt_id, etc.) — these extract single-char metadata, not elements.
    - Residue-code lookups (three_to_one, RESTYPE_1TO3): extracting amino-acid codes, not
      inferring elements from atom names.
    - Element-dispatch on already-resolved symbols: get_mass(element), get_radius(element),
      etc., where `element` is NOT from atom-name slicing.
    - Force-field property tables (proxide-gaff2, proxide-gaff): dispatch on atom types that
      are already classification tokens, not raw atom names.

    The check scans for the specific anti-pattern: slicing/extracting from variables containing
    "atom" or "name" in their identifier, then dispatching on single-letter or two-letter
    element symbols. This avoids false positives on legitimate element-symbol lookup tables.
    """
    crates_dir = Path(__file__).parent.parent / "crates"
    assert crates_dir.exists(), f"Expected crates/ directory at {crates_dir}"

    # Collect Rust files from parsers and physics — high-risk zones for atom-name slicing.
    # Exclude gaff/gaff2 (force-field typing), and skip masses.rs.
    high_risk_patterns = [
        "crates/proxide-io/**/*.rs",
        "crates/proxide-physics/**/*.rs",
        "crates/proxide_fixer/**/*.rs",
    ]

    rust_files = set()
    for pattern in high_risk_patterns:
        rust_files.update(crates_dir.parent.glob(pattern))

    # Filter out masses.rs and any file in gaff/gaff2
    rust_files = [
        f for f in rust_files
        if f.is_file()
        and "masses.rs" not in f.name
        and "gaff" not in str(f)
    ]

    violations = []

    for rust_file in rust_files:
        with open(rust_file, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            lines = content.split("\n")

        # Skip files that already import/use infer_element (they're doing it right)
        if "infer_element" in content:
            continue

        # Look for the specific anti-pattern: slicing an atom-name-derived variable,
        # then dispatching on element symbols.
        #
        # Key heuristic: search for .chars().next() or [..1]/[..2] on a variable
        # that contains "atom" or "name" in its identifier. This is much more
        # specific than looking for all .chars().next() calls (which has false
        # positives on insertion codes, alt locs, etc.).

        for i, line in enumerate(lines, start=1):
            # Skip comments and docstrings
            if line.strip().startswith("//"):
                continue

            # Skip known safe patterns
            if any(safe in line for safe in [
                "alt_loc", "i_code", "label_alt_id", "pdbx_PDB_ins_code",
                "three_to_one", "RESTYPE_1TO3", "get_mass", "get_radius",
                "insertion code", "alt loc"
            ]):
                continue

            # Look for atom-name-derived variables being sliced
            if ("atom" in line.lower() or "name" in line.lower()) and \
               ((".chars().next()" in line or "[..1]" in line or "[..2]" in line)):
                # Scan forward a few lines to see if there's element-symbol dispatch
                context_block = "\n".join(lines[max(0, i - 2) : min(len(lines), i + 15)])

                # Check for element symbol dispatch patterns:
                # - 'H' => ..., 'C' => ..., etc.
                # - "Cl" => ..., "Na" => ..., etc.
                has_element_arms = bool(
                    re.search(
                        r"(['\"])([HCNOSFPIK]|Cl|Br|Na|Mg|Zn|Fe|Cu|Mn|Se)\1\s*=>",
                        context_block,
                    )
                )

                if has_element_arms:
                    violations.append(
                        f"{rust_file.relative_to(crates_dir.parent)}:{i}\n"
                        f"  {line.strip()}\n"
                        f"  → element-like dispatch on atom-name-derived variable\n"
                    )

    if violations:
        msg = (
            "Found suspect element-inference patterns (element symbols dispatched on\n"
            "atom-name-derived variables) in:\n\n"
            + "".join(violations)
            + "\n"
            + "Element inference from PDB atom names is a COMMON BUG. Use the canonical\n"
            + "implementation: `proxide_core::chem::masses::infer_element()`.\n"
            + "\n"
            + "It correctly handles:\n"
            + "  • Two-letter elements: CL→Cl, BR→Br, NA→Na, MG→Mg, ZN→Zn, FE→Fe,\n"
            + "    CU→Cu, MN→Mn, SE→Se, I→I, K→K\n"
            + "  • Case insensitivity: CL, Cl, cl all → Cl\n"
            + "  • The alpha-carbon edge case: CA→C (not Ca)\n"
            + "\n"
            + "Reimplementing elsewhere guarantees silent physics errors.\n"
            + "See backlog #5052 (prolix) for the history of this bug.\n"
        )
        raise AssertionError(msg)
