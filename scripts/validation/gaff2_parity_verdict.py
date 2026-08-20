"""Phase 5 (Graded Verdict) of the GAFF2 bathos-literature-parity campaign.

Computes the final PARITY / PARTIAL / FAIL grade for proxide's GAFF2 atom-typing
implementation (`src/proxide/chem/gaff2.py`) using bathos's X1 cap-lattice grader
logic, fed by the evidence gathered across Phases 1-4 of the campaign (see
`.praxia/docs/decisions/260818_gaff2-parity-verdict-policy.md` and the verdict
report this script's output feeds: `.praxia/docs/audits/260820_gaff2-parity-verdict.md`).

The grading logic below is a direct, line-for-line transcription of
`bathos.parity.compute_grade` (bathos repo: `src/bathos/parity.py`, read
2026-08-20) rather than an import, because bathos requires Python>=3.13 while
this project's venv is pinned to Python 3.12 -- `uv run --with
/home/marielle/projects/bathos` fails to resolve for that reason. If bathos
ever supports 3.12, or proxide moves to 3.13+, prefer importing
`bathos.parity.ParityEvidence`/`compute_grade` directly over this transcription.

Usage:
    uv run python scripts/validation/gaff2_parity_verdict.py
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("gaff2_parity_verdict")


@dataclass
class ParityEvidence:
    """Mirrors bathos.parity.ParityEvidence (see module docstring)."""

    clause_parity_pct: float
    adversarial_survived: bool
    invariant_pass: bool
    reproduction_rung: str
    ambiguity_load: str


@dataclass
class ParityGradeResult:
    """Mirrors bathos.parity.ParityGradeResult."""

    grade: str
    ceilings: dict[str, str] = field(default_factory=dict)


_CLAUSE_PARITY_FAIL_THRESHOLD = 0.5
_CLAUSE_PARITY_PARTIAL_THRESHOLD = 1.0


def compute_grade(evidence: ParityEvidence) -> ParityGradeResult:
    """Verbatim transcription of bathos.parity.compute_grade's X1 cap-lattice."""
    ceilings: dict[str, str] = {}

    ceilings["invariant"] = "PARITY" if evidence.invariant_pass else "FAIL"

    if evidence.clause_parity_pct < _CLAUSE_PARITY_FAIL_THRESHOLD:
        ceilings["clause_parity"] = "FAIL"
    elif evidence.clause_parity_pct < _CLAUSE_PARITY_PARTIAL_THRESHOLD:
        ceilings["clause_parity"] = "PARTIAL"
    else:
        ceilings["clause_parity"] = "PARITY"

    ceilings["adversarial"] = "PARITY" if evidence.adversarial_survived else "FAIL"

    if evidence.reproduction_rung in ("R0", "R1"):
        ceilings["reproduction_rung"] = "PARITY"
    elif evidence.reproduction_rung in ("R2", "R3", "R4"):
        ceilings["reproduction_rung"] = "PARTIAL"
    else:
        ceilings["reproduction_rung"] = "PARTIAL"

    ceilings["ambiguity_load"] = (
        "PARTIAL" if evidence.ambiguity_load == "load_bearing" else "PARITY"
    )

    ceiling_order = {"FAIL": 0, "PARTIAL": 1, "PARITY": 2}
    min_ceiling = min(ceiling_order[c] for c in ceilings.values())
    grade_map = {0: "FAIL", 1: "PARTIAL", 2: "PARITY"}
    return ParityGradeResult(grade=grade_map[min_ceiling], ceilings=ceilings)


# --- Evidence, per Phase 1-4 findings + the PR #27/#28/#29 follow-up sprint
# (2026-08-20) -- see the verdict report for full derivation ---
#
# clause_parity_pct: of parity.bth.toml's 5 pre-registered hypotheses,
#   evaluated against the full history: Phase 3 findings, orchestrator
#   re-derivation (Phase 4), PR #27's follow-up fixes, and PR #28's real
#   external-reference run (antechamber/GAFFTemplateGenerator, gaff-2.11):
#     H1 (rule precedence for all benchmark molecules)        -> MATCH
#     H2 (f8 bond-count disambiguation)                        -> MATCH
#         (PR #27: Kekulization fix, sb/db now correctly inclusive)
#     H3 (f9 neighbor-pattern cp discrimination)               -> MATCH
#     H4 (AR1/AR2/AR3 sub-classification, pre-registered as a "KNOWN GAP,
#         not yet resolved")                                   -> MATCH (was
#         AMBIGUOUS). PR #28: confirmed WRONG (furan/pyrrole/thiophene should
#         be the cc family, not ca) against real antechamber output, then
#         fixed (AR1 now requires 6-membered-ring membership specifically,
#         matching the DEF footer's own "benzene and pyridine" citation) and
#         re-verified against the same reference. Independently confirmed to
#         generalize on a real production molecule outside the original 3
#         test cases (histidine-probe's imidazole, a two-heteroatom ring).
#     H5 (naphthalene bridgehead 1RG6 ambiguity resolved?)     -> MATCH (was
#         AMBIGUOUS). PR #28: real antechamber output confirms all 10
#         naphthalene ring carbons (bridgeheads included) resolve to ca,
#         exactly matching the existing exact-ring-count implementation --
#         no code change needed, just real confirmation replacing an
#         external-legend inference.
#   -> 5/5 MATCH = 1.0
#
#   EXPLICITLY OUT OF SCOPE for this clause count (real, documented gaps that
#   do NOT map onto any of the 5 registered hypotheses above, so their
#   resolution status does not affect this number -- see the verdict report's
#   "What PARITY does and doesn't cover" section for the honest accounting):
#   - cc/cd ring-alternation (torsion-parameter bookkeeping): there is no
#     `cd` ATD rule anywhere in ATOMTYPE_GFF2.DEF (confirmed by grep), so
#     this genuinely isn't part of "does the atom-typing RULE ENGINE
#     correctly implement the DEF file's rules" -- it's a downstream AMBER-
#     tooling convention external to the spec being validated here.
#   - H-atom typing (h1-h5's electron-withdrawing-atom element set): the 5
#     registered hypotheses only ever covered heavy-atom typing; H-atom
#     typing was implemented in PR #28 (Section B) but was never part of
#     this campaign's pre-registered scope, so its own judgment calls
#     (documented in gaff2.py's _EW_ATOMS) don't feed this grade either.
#
# adversarial_survived: True -- no operationalized hypothesis was mechanism-
#   nullified; both defects found (H2's sb/db bug, the H4 AR1/AR2/AR3 bug)
#   are now fixed outright and re-verified against real external evidence,
#   not merely documented or left as accepted deviations.
#
# invariant_pass: tests/test_gaff2_golden.py + tests/test_gaff2_parity_invariants.py's
#   full suite (59 tests as of PR #28) passes, zero xfail.
#
# reproduction_rung: R1 -- upgraded from R0 (text-parity only) after PR #28's
#   real external-reference run: proxide's atom-type-family output was
#   directly compared against an independent reference implementation
#   (antechamber/openmmforcefields GAFFTemplateGenerator, gaff-2.11) on
#   shared inputs (furan/pyrrole/thiophene/naphthalene), not merely read
#   against the spec text. Not R2+ (no *systematic*, full reproduction run
#   across the whole benchmark set against the reference tool -- 4 molecules
#   were spot-checked, not all 24).
#
# ambiguity_load: none -- upgraded from load_bearing. The genuine ambiguity
#   this dimension tracked (which AR-class reading is correct for 5-membered
#   heteroaromatics) is now resolved with real reference evidence, not just
#   documented as unresolved. The remaining cc/cd-alternation gap is a scoped
#   capability gap with a known, well-understood mechanism (not an unclear
#   spec interpretation) and sits outside this campaign's registered
#   hypotheses (see the clause_parity_pct note above) -- it does not belong
#   in this dimension either.

EVIDENCE = ParityEvidence(
    clause_parity_pct=1.0,
    adversarial_survived=True,
    invariant_pass=True,
    reproduction_rung="R1",
    ambiguity_load="none",
)


def main() -> None:
    result = compute_grade(EVIDENCE)
    logger.info("GAFF2 literature-parity verdict: %s", result.grade)
    logger.info("Ceilings:")
    for dim, ceiling in result.ceilings.items():
        logger.info("  %-18s %s", dim, ceiling)


if __name__ == "__main__":
    main()
