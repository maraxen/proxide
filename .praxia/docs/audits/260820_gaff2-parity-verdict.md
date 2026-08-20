---
name: 260820_gaff2-parity-verdict
description: Phase 5 graded verdict (PARTIAL) for the GAFF2 atom-typing bathos-literature-parity campaign, closing out Phases 3-5
metadata:
  type: audit
  task_id: 260820_gaff2_parity_phase3
  status: final
---

# GAFF2 Atom-Typing Literature-Parity: Graded Verdict

**Date:** 2026-08-20
**Campaign:** `parity.bth.toml` (proxide repo root)
**Verdict:** **PARTIAL**
**Approver (accepted-PARTIAL sign-off):** Marielle

## Summary

PR #26 (merged 2026-08-19) rewrote proxide's GAFF2 DEF-grammar rule engine
(`src/proxide/chem/gaff2.py`), fixing a severe `cp`/`ca` aromatic-mistyping bug and
completing Phase 1 (blind reconstruction, N=3) and an informal Phase 2 (reconcile) of
this campaign. This report closes the campaign by running Phase 3 (adversarial
refutation, M=3), Phase 4 (orchestrator-owned adjudication), and Phase 5 (graded
verdict computation).

The grade is **PARTIAL**, computed via bathos's X1 cap-lattice
(`scripts/validation/gaff2_parity_verdict.py`, a transcription of
`bathos.parity.compute_grade` — see that script's docstring for why it's a
transcription rather than an import). Two ceilings are non-PARITY:

- **clause_parity = PARTIAL** (0.5): of the campaign's 4 hypotheses, 2 MATCH, 1 has a
  confirmed but scoped DEVIATION, 1 remains AMBIGUOUS.
- **ambiguity_load = PARTIAL**: the AR1/AR2/AR3 five-membered-heteroaromatic
  classification gap remains genuinely unresolved and sits in the core aromatic-typing
  mechanism.

No ceiling reached FAIL. The core mechanism this campaign exists to validate —
rule-precedence-correct heavy-atom typing, and `cp`/`ca` fused-ring discrimination
specifically — held up under adversarial attack across 14 benchmark molecules plus 5
additional fused/bridgehead PAHs the attackers introduced.

## Phase 3: Adversarial Refutation (M=3)

Three independent agents, each stating an assumption upfront and defaulting to
"deviation" on inconclusive evidence, per the honesty-tax protocol:

| Attacker | Lens | Target | Verdict |
|---|---|---|---|
| #1 | struct | H2: f8 bond-count disambiguation | **Defect found — major.** `_bond_category_facts` computes lowercase `sb`/`db` identically to uppercase `SB`/`DB`, dropping the DEF footer's "includes aromatic single/double" inclusive semantics. Demonstrated on toluene's ipso carbon (a real match/no-match flip); does not corrupt the `c`/`cs` benchmark molecules tested, because `cs`/`c` are actually disambiguated by f9 (S vs O/S neighbor), not by which f8 branch fires. |
| #2 | struct | H3+H4: cp neighbor-pattern, naphthalene bridgehead | **No defect found — moderate-high confidence.** No new mistyping across anthracene, pyrene, triphenylene, triphenylbenzene, terphenyl. No overcorrection (genuine biphenyl-type junctions still resolve to `cp`). Found supporting-but-not-conclusive external evidence (openbabel/openbabel's independently maintained `gaff.dat` legend: "`cp` — head sp2 C that connects two rings in biphenyl sys.") favoring the code's current exact-`1RG6`-count reading over the alternative. |
| #3 | stats | H1: full benchmark-set rule-precedence sweep | **Defect found (out-of-scope for this grade) + benchmark-authoring deviations.** Every heavy-atom type across all 14 curated molecules is internally correct per the DEF rules. A separate, code-confirmed defect was found in H-atom typing (`_H_TYPE_BY_HEAVY` missing most nitrogen ATD types — see "Excluded finding" below). Several `benchmarks/gaff2_parity_molecules.yaml` notes overclaim or misdescribe actual behavior (see Follow-ups). |

Full per-molecule scorecards and code/DEF-file citations are preserved in each
attacker's dispatch transcript (agent IDs referenced in this campaign's session log);
not reproduced verbatim here for length.

## Phase 4: Adjudication (orchestrator-owned re-derivation, Constraint 1)

Per the campaign's re-derivation lock, both claimed code defects were independently
re-derived — not trusted from the agent reports — via
`tests/test_gaff2_parity_invariants.py`:

1. `test_bond_category_facts_sb_includes_aromatic_confirmed_defect` (xfail, strict) —
   confirms Attacker #1's finding directly: toluene's ipso carbon has 3 heavy bonds (2
   aromatic + 1 single) and should satisfy `3sb`, but the current tally gives `sb=1`.
2. `test_f8_bond_count_disambiguation_no_regression_on_h_ew_benchmark_molecules`
   (passing) — confirms the defect above does **not** corrupt formamide,
   N-methylacetamide, acetone, or acetate's carbonyl-carbon typing.
3. `test_h_type_by_heavy_missing_amide_n_types_confirmed_defect` (xfail, strict) —
   confirms Attacker #3's H-typing finding directly: formamide/N-methylacetamide's
   amide nitrogen (`nt`/`ns`) is absent from `_H_TYPE_BY_HEAVY`, so its H silently
   types as `hc` instead of `hn`.

All three re-derivations reproduced the agents' claims exactly (2 xfail, 1 pass).
`tests/test_gaff2_golden.py`'s existing 28 tests remain green — no regression.

**Severity ranking** (per `04_adjudicate.md`'s rubric):

- Defect 1 (sb/db inclusive semantics): **major**. Real, reproducible, but scoped —
  does not corrupt the current benchmark set, and a clear fix direction exists
  (whether to include aromatic bonds in the `sb`/`db` tally, or to Kekulize before
  bond-category extraction — the latter is more precise but has broader blast radius
  and deserves its own scoped review rather than a fix bolted onto this verdict).
- Defect 2 (`_H_TYPE_BY_HEAVY` missing N types): **major, out of this grade's scope**.
  H-atom typing was never covered by `parity.bth.toml`'s hypotheses (heavy-atom typing
  only) and was explicitly deferred as a separate follow-up in the plan that produced
  PR #26. This finding does not gate the PARTIAL grade below, but must be fixed before
  any future claim of H-atom-typing parity — it directly undermines the "critical
  hydrogen typing on N adjacent to C=O" claim that Formamide/N-methylacetamide were
  specifically curated to test.

## Phase 5: Graded Verdict

Evidence (full derivation in `scripts/validation/gaff2_parity_verdict.py`'s comments):

| Dimension | Value | Ceiling |
|---|---|---|
| `invariant_pass` | True | PARITY |
| `clause_parity_pct` | 0.5 (2/4 hypotheses MATCH) | PARTIAL |
| `adversarial_survived` | True (no hypothesis mechanism-nullified) | PARITY |
| `reproduction_rung` | R0 (text-parity only; no local antechamber/OpenFF run) | PARITY |
| `ambiguity_load` | load_bearing (AR1/AR2/AR3 unresolved) | PARTIAL |

**Grade = min(ceilings) = PARTIAL.**

```
uv run python scripts/validation/gaff2_parity_verdict.py
GAFF2 literature-parity verdict: PARTIAL
```

### `[confounds.reference_parity]` block

Ready for the next Core-tier campaign's `claim.bth.toml` to reference (no existing
claim-tier campaign currently consumes GAFF2, so this is provided as a citable
artifact rather than wired into a specific file):

```toml
[[confounds]]
id = "C_baseline"
label = "GAFF2 atom typing is validated against its own DEF-file specification, with two documented residual gaps"
[confounds.reference_parity]
reference_paper = "AmberTools ATOMTYPE_GFF2.DEF (vendored spec; see parity.bth.toml citation_note)"
reference_metric = "exact atom-type match (equivalence_bound = 0.0)"
reference_value = 1.0
equivalence_bound = 0.0
parity_run_id = "260820_gaff2_parity_phase3"  # this campaign; no prior bth run/campaign_id existed to append to (see Provenance note below)
verdict = "PARTIAL"
```

## Sign-offs resolved

Both `[NEEDS HUMAN SIGN-OFF]` items in
`.praxia/docs/decisions/260818_gaff2-parity-verdict-policy.md` are now filled in (see
that file's diff in this same commit):

- **Approver for accepted-PARTIAL verdicts:** Marielle.
- **h_ew Option A vs. B:** resolved as **effectively satisfied, with a documented
  residual gap** — not the open A/B choice as originally framed. PR #26 correctly
  implemented f8's digit-count *arithmetic* and fixed the *disambiguation outcome* for
  every currently-curated h_ew molecule (re-verified here, not just trusted from the PR
  description). The one confirmed residual gap (lowercase `sb`/`db` inclusive-bond
  semantics) is scoped, tracked (xfail test above), and does not corrupt any current
  benchmark molecule — it does not block Core-tier campaigns under Option B's framing,
  but should be fixed before this file is next touched, since it's now a small, clearly
  specified, one-function change.

## Follow-ups (not blocking this verdict)

1. **Fix `_bond_category_facts`'s sb/db inclusive semantics** (Defect 1). Recommended
   direction: decide between (a) folding aromatic bonds into both `sb` and `db` tallies
   consistently with the existing `DL := is_aromatic` simplification (cheap, but
   over-counts — a 3-aromatic-bond atom would show both `sb=3` and `db=3`), or (b)
   Kekulizing before bond-category extraction to get the true per-bond Kekule identity
   while preserving `GetIsAromatic()` (more precise, broader blast radius — touches
   molecule preparation, deserves its own review). Un-xfail
   `test_bond_category_facts_sb_includes_aromatic_confirmed_defect` once fixed.
2. **Fix `_H_TYPE_BY_HEAVY`'s missing nitrogen ATD types** (Defect 2) as the first
   concrete item when Section B (H-atom typing rework, deferred in the plan that
   produced PR #26) is scoped. Un-xfail
   `test_h_type_by_heavy_missing_amide_n_types_confirmed_defect` once fixed.
3. **Correct `benchmarks/gaff2_parity_molecules.yaml`'s authoring deviations** (Attacker
   #3): Indene and Anthracene both claim `targets_gap: cp` but produce zero `cp` atoms
   (same bridgehead-exclusion mechanism as Naphthalene, just undocumented for these
   two); 1,3-Butadiene's terminal/internal `ce`/`cc` claim is exactly backwards;
   Divinyl-ketone/Acrolein overclaim `cc`/`cd`/`ce` coverage for terminal `=CH2`
   carbons that structurally can't get those types. None of these are code defects —
   the code is internally consistent; the curation notes are wrong or incomplete.
4. **Add golden-test coverage** for Toluene (already matches, cheap to lock in),
   Anthracene (currently zero coverage), and the `[2DL]`/`[3sb]`/`cs` branches of the
   f8 grammar (currently only `[1DB,0DL]` is exercised by a golden molecule).
5. **AR1/AR2/AR3 sub-classification** remains unresolved (no algorithm exists in the
   DEF file); this is the item keeping `ambiguity_load = load_bearing`. No action
   recommended beyond what's already documented in `tests/test_gaff2_golden.py`'s
   `test_atom_type_five_membered_heteroaromatics` and `parity.bth.toml`'s hypothesis 4.

## Provenance note

No bathos run/campaign was ever registered for Phase 1/2 of this campaign (they ran
informally, no `bth run` tracking — confirmed via `~/.bth/catalog`, zero GAFF2-tagged
runs). Per direction, this campaign starts run-tracking clean at Phase 3 rather than
fabricating retroactive Phase 1/2 records. PR #26 (`ba60145`, `fb95944`, merge
`a49ca94`) and the Phase 1 blind-reconstruction record are the prior evidence this run
builds on, cited here rather than represented as prior tracked runs.

## Reproduce this verdict

1. `uv run pytest tests/test_gaff2_golden.py tests/test_gaff2_parity_invariants.py -v`
   — expect 28 passed (golden) + 1 passed + 2 xfailed (invariants).
2. `uv run python scripts/validation/gaff2_parity_verdict.py` — expect `PARTIAL`.
3. If either follow-up fix (1 or 2 above) lands: update the corresponding evidence
   value in `scripts/validation/gaff2_parity_verdict.py` (clause_parity_pct moves
   toward 0.75 or 1.0 as hypotheses move from DEVIATION to MATCH), un-xfail the
   matching invariant test, and re-run this script — do not hand-edit the verdict
   without re-running the grader.
