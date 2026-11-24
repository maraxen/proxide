"""Stable codes for the PDB/mmCIF writer's findings (spec 261123 D4).

This is a literal table. No code is built dynamically (260910 OBS-110), so a grep for a code finds
every place it can be emitted. When 260910's ``diagnostics/registry.toml`` lands it imports this
table and does not duplicate it.

Experimental: not part of the stable API.
"""

from __future__ import annotations

from typing import Literal

Severity = Literal["info", "warning"]

HETERO_IGNORED = "PROX-WRITE-HETERO-IGNORED"
HETERO_UNKNOWN = "PROX-WRITE-HETERO-UNKNOWN"
ICODE_BLANKED = "PROX-WRITE-ICODE-BLANKED"
ICODE_UNKNOWN = "PROX-WRITE-ICODE-UNKNOWN"
ALIAS_ATOMS_MISSING = "PROX-WRITE-ALIAS-ATOMS-MISSING"

# (code, severity, meaning). Info findings never warn; warning findings warn once per write.
WRITE_CODES: tuple[tuple[str, Severity, str], ...] = (
  (
    HETERO_IGNORED,
    "warning",
    "record_type='residue_table' ignores hetero=True on rows whose residue is not in the table",
  ),
  (HETERO_UNKNOWN, "info", "Full Protein with hetero None: every row follows the residue table"),
  (
    ICODE_BLANKED,
    "warning",
    "insertion_codes='blank' writes column 27 blank although the Protein carries non-blank codes",
  ),
  (ICODE_UNKNOWN, "info", "Protein.insertion_codes is None: no row carries an insertion code"),
  (
    ALIAS_ATOMS_MISSING,
    "warning",
    "a residue typed other than its layout residue (modified_residue_typing='unknown') has no "
    "alias atoms (for MSE, SE) in the written records",
  ),
)

SEVERITY_OF: dict[str, Severity] = {code: severity for code, severity, _ in WRITE_CODES}
