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


# --- Evidence, per Phase 1-4 findings (see the verdict report for full derivation) ---
#
# clause_parity_pct: of parity.bth.toml's 4 hypotheses, evaluated against Phase 3
#   findings + orchestrator re-derivation (Phase 4):
#     H1 (rule precedence for all benchmark molecules)        -> MATCH
#         Attacker #3's 14-molecule scorecard confirms every heavy-atom type is
#         internally correct per the DEF rules; the deviations it found were
#         benchmark-YAML documentation errors (Indene/Anthracene/Butadiene/etc.
#         notes claiming behavior the code never implemented), not code defects.
#     H2 (f8 bond-count disambiguation)                        -> DEVIATION
#         Confirmed defect (Attacker #1 + orchestrator re-derivation,
#         tests/test_gaff2_parity_invariants.py): lowercase sb/db bond-category
#         counts omit the DEF footer's "includes aromatic" inclusive semantics.
#         Scoped: does not corrupt any current benchmark molecule (re-derived
#         and locked in as a passing invariant), but is a real, reproducible bug.
#     H3 (f9 neighbor-pattern cp discrimination)               -> MATCH
#         Attacker #2 found no defect across 5 additional fused/bridgehead
#         molecules (anthracene, pyrene, triphenylene, triphenylbenzene,
#         terphenyl) beyond the existing naphthalene/biphenyl tests.
#     H4 (naphthalene bridgehead 1RG6 ambiguity resolved?)     -> AMBIGUOUS
#         Attacker #2 found supporting-but-not-conclusive external evidence
#         (openbabel/openbabel's independent gaff.dat legend) favoring the
#         current exact-count reading. Genuinely useful, not dispositive.
#   -> 2/4 MATCH = 0.5
#
# adversarial_survived: scoped to whether any of the 4 *operationalized*
#   hypotheses above was mechanism-nullified (per the decision doc's own FAIL
#   trigger: "adversarial attacks find mechanism-nullifying defects"). H2's
#   defect is real but classified MAJOR-not-critical (see 04_adjudicate.md's
#   severity rubric) because it does not corrupt the benchmark set the campaign
#   actually validates against -- represented via clause_parity_pct instead of
#   forcing a FAIL-by-design "landed refutation" here. True: no hypothesis was
#   mechanism-nullified.
#
#   NOTE: Attacker #3 also found a second, separate confirmed defect
#   (_H_TYPE_BY_HEAVY missing most nitrogen ATD types, causing amide N-H to
#   mistype as "hc" instead of "hn") -- deliberately EXCLUDED from this
#   evidence block. H-atom typing was explicitly scoped out of this campaign
#   (parity.bth.toml's hypotheses cover heavy-atom typing only; H-atom typing
#   was deferred as "Section B" in the original implementation plan). This
#   finding does not gate THIS grade, but must be addressed before any future
#   claim of H-atom-typing parity. See the verdict report.
#
# invariant_pass: tests/test_gaff2_parity_invariants.py's core, must-hold
#   invariant (no regression on the h_ew benchmark set despite the H2 defect)
#   passes. The two xfail tests document known, tracked, scoped defects rather
#   than an unexpected invariant failure.
#
# reproduction_rung: R0 -- text-parity only (ATOMTYPE_GFF2.DEF's own footer);
#   no local antechamber/OpenFF GAFF2 reference run exists to reach R1+.
#
# ambiguity_load: load_bearing -- AR1/AR2/AR3 aromaticity sub-classification
#   for 5-membered heteroaromatics remains genuinely unresolved (the DEF file
#   gives no algorithm to compute it), and this sits in the core aromatic-
#   typing mechanism, not an edge hyperparameter.

EVIDENCE = ParityEvidence(
    clause_parity_pct=0.5,
    adversarial_survived=True,
    invariant_pass=True,
    reproduction_rung="R0",
    ambiguity_load="load_bearing",
)


def main() -> None:
    result = compute_grade(EVIDENCE)
    logger.info("GAFF2 literature-parity verdict: %s", result.grade)
    logger.info("Ceilings:")
    for dim, ceiling in result.ceilings.items():
        logger.info("  %-18s %s", dim, ceiling)


if __name__ == "__main__":
    main()
