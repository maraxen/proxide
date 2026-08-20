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


# --- Evidence, per Phase 1-4 findings + verdict-report follow-ups #1/#2 (fixed
# 2026-08-20, same day as the original verdict) -- see the verdict report for
# full derivation ---
#
# clause_parity_pct: of parity.bth.toml's ACTUAL 5 pre-registered hypotheses
#   (PR-audit finding: an earlier version of this evidence block only counted 4,
#   silently dropping the AR1/AR2/AR3 hypothesis entirely rather than scoring it
#   -- corrected here to use the true denominator and the toml's own numbering),
#   evaluated against Phase 3 findings + orchestrator re-derivation (Phase 4) +
#   the follow-up fixes:
#     H1 (rule precedence for all benchmark molecules)        -> MATCH
#         Attacker #3's 14-molecule scorecard confirms every heavy-atom type is
#         internally correct per the DEF rules; the deviations it found were
#         benchmark-YAML documentation errors (Indene/Anthracene/Butadiene/etc.
#         notes claiming behavior the code never implemented, corrected as
#         follow-up #3), not code defects.
#     H2 (f8 bond-count disambiguation)                        -> MATCH (was
#         DEVIATION at the original verdict). Attacker #1's confirmed defect
#         (lowercase sb/db bond-category counts omitting the DEF footer's
#         "includes aromatic" inclusive semantics) is FIXED (follow-up #1):
#         assign_gaff2_atom_types now Kekulizes a local copy before matching,
#         so sb/db reflect the true per-bond Kekule identity. Verified via
#         tests/test_gaff2_parity_invariants.py
#         ::test_bond_category_facts_sb_db_include_aromatic_kekule_identity
#         (passing, not xfail) AND
#         ::test_assign_gaff2_atom_types_kekulizes_before_matching (a spy test
#         guarding the wiring itself, added after a PR audit found the first
#         test alone doesn't exercise assign_gaff2_atom_types's internal
#         Kekulize call -- no current benchmark molecule's typed OUTPUT
#         actually depends on this fix, so an output-only test can't catch a
#         regression here).
#     H3 (f9 neighbor-pattern cp discrimination)               -> MATCH
#         Attacker #2 found no defect across 5 additional fused/bridgehead
#         molecules (anthracene, pyrene, triphenylene, triphenylbenzene,
#         terphenyl) beyond the existing naphthalene/biphenyl tests.
#     H4 (AR1/AR2/AR3 sub-classification, pre-registered as a "KNOWN GAP, not
#         yet resolved") -> AMBIGUOUS. Genuinely unresolved; no algorithm
#         exists in the DEF file to compute it. This same fact ALSO drives
#         ambiguity_load=load_bearing below -- that is not double-counting:
#         clause_parity_pct measures per-hypothesis agreement, ambiguity_load
#         measures whether an unresolved ambiguity sits in the core mechanism.
#         Both are true of the same underlying gap; scoring it in both
#         dimensions is how the cap-lattice is designed to work, not a
#         self-serving inflation -- PR-audit finding: document this
#         explicitly rather than leave the relationship implicit.
#     H5 (naphthalene bridgehead 1RG6 ambiguity resolved?)     -> AMBIGUOUS
#         Attacker #2 found supporting-but-not-conclusive external evidence
#         (openbabel/openbabel's independent gaff.dat legend) favoring the
#         current exact-count reading. Genuinely useful, not dispositive.
#   -> 3/5 MATCH = 0.6
#
# adversarial_survived: True -- no operationalized hypothesis was mechanism-
#   nullified (H2's defect was real but scoped/non-critical, and is now fixed
#   outright rather than merely documented).
#
#   PR-audit flag (judgment call, not a bug): bathos's own compute_grade
#   docstring treats ANY landed refutation as FAIL-by-design ("it negates a
#   core claim"), and H2's refutation genuinely landed at the time Phase 4
#   adjudicated it. Scoring True here rests on the fix being independently
#   chemistry-verified afterward (a fresh audit agent re-derived all 7
#   DEF-file/RDKit claims from primary sources, not just re-running the
#   existing tests) rather than on a formal Phase-3-style adversarial
#   re-attack against the patched code. Recorded here explicitly for human
#   sign-off rather than silently defaulting to the more lenient reading.
#
#   NOTE: Attacker #3 also found a second, separate confirmed defect
#   (_H_TYPE_BY_HEAVY missing most nitrogen ATD types, causing amide N-H to
#   mistype as "hc" instead of "hn") -- deliberately EXCLUDED from this
#   evidence block, same as at the original verdict. H-atom typing was
#   explicitly scoped out of this campaign (parity.bth.toml's hypotheses cover
#   heavy-atom typing only; H-atom typing was deferred as "Section B" in the
#   original implementation plan). This defect is now ALSO fixed (follow-up
#   #2, via _H_TYPE_ELEMENT_DEFAULT) and verified end-to-end
#   (tests/test_gaff2_parity_invariants.py::test_h_type_by_heavy_amide_n_h_types_as_hn),
#   but stays excluded from this grade's evidence for the same scoping reason
#   as before -- fixing it doesn't retroactively bring it into this campaign's
#   hypothesis set, it just means the next H-atom-typing parity run (if one is
#   scoped) starts from a better baseline.
#
# invariant_pass: tests/test_gaff2_parity_invariants.py's core, must-hold
#   invariant (no regression on the h_ew benchmark set) passes. Zero xfail
#   tests remain -- both confirmed defects are fixed and locked in as passing.
#
# reproduction_rung: R0 -- text-parity only (ATOMTYPE_GFF2.DEF's own footer);
#   no local antechamber/OpenFF GAFF2 reference run exists to reach R1+.
#
# ambiguity_load: load_bearing -- AR1/AR2/AR3 aromaticity sub-classification
#   for 5-membered heteroaromatics remains genuinely unresolved (the DEF file
#   gives no algorithm to compute it), and this sits in the core aromatic-
#   typing mechanism, not an edge hyperparameter (this is H4 above). Unaffected
#   by the follow-up fixes; still caps the grade to PARTIAL even with
#   clause_parity at 0.6.

EVIDENCE = ParityEvidence(
    clause_parity_pct=0.6,
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
