---
name: 260820_gaff2-parity-verdict
description: Phase 5 graded verdict (PARITY, as of the PR #27/#28/#29 sprint) for the GAFF2 atom-typing bathos-literature-parity campaign
metadata:
  type: audit
  task_id: 260820_gaff2_parity_phase3
  status: final
---

# GAFF2 Atom-Typing Literature-Parity: Graded Verdict

**Date:** 2026-08-20 (original PARTIAL verdict; superseded same-day by Update 3 below)
**Campaign:** `parity.bth.toml` (proxide repo root)
**Verdict:** **PARITY** (was PARTIAL — see Update 3)
**Sign-off:** N/A for PARITY — the accepted-PARTIAL approver line below is historical,
kept for the record of what PR #27 originally required.
**Approver (accepted-PARTIAL sign-off, historical):** Marielle

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

- **clause_parity = PARTIAL** (originally 0.5 of 4 hypotheses; see the second Update
  below — the correct denominator is `parity.bth.toml`'s actual 5 pre-registered
  hypotheses).
- **ambiguity_load = PARTIAL**: the AR1/AR2/AR3 five-membered-heteroaromatic
  classification gap remains genuinely unresolved and sits in the core aromatic-typing
  mechanism.

No ceiling reached FAIL. The core mechanism this campaign exists to validate —
rule-precedence-correct heavy-atom typing, and `cp`/`ca` fused-ring discrimination
specifically — held up under adversarial attack across 14 benchmark molecules plus 5
additional fused/bridgehead PAHs the attackers introduced.

**Update, same day (2026-08-20):** follow-ups #1-#4 below all landed before merge.
Both confirmed defects (sb/db inclusive semantics; `_H_TYPE_BY_HEAVY` missing
nitrogen types) are now fixed and verified, not just documented. The grade is still
**PARTIAL** (`ambiguity_load` alone is enough to cap it there — AR1/AR2/AR3 is a
genuine, unresolved spec gap, not a bug), but on strictly better evidence than the
original run. See the updated Phase 5 table and Follow-ups section below; the Phase
3/4 sections are left as originally written (historical record of what the campaign
found) with pointers to what changed.

**Update 2, same day (2026-08-20), after subagent PR audit:** three independent
agents (code-correctness, chemistry/domain-accuracy, test-coverage) reviewed the full
PR before merge. Chemistry/domain-accuracy: all 7 spot-checked factual claims in this
report and the code independently re-verified from primary sources (raw DEF-file
text, fresh RDKit computations) — no discrepancies. Code-correctness: no high-
confidence bugs in either production fix (~85% confidence clean). Test-coverage found
three real issues, now fixed:
- The original `clause_parity_pct=0.75` used the wrong denominator — 4, not
  `parity.bth.toml`'s actual 5 pre-registered hypotheses (the AR1/AR2/AR3 hypothesis
  was silently excluded rather than scored, and the naphthalene-bridgehead hypothesis
  was mislabeled H4 when it's really the toml's H5). **Corrected to 3/5 = 0.6** — grade
  unaffected (still PARTIAL either way), but now traceable against the pre-registration.
- `test_bond_category_facts_sb_db_include_aromatic_kekule_identity` doesn't actually
  exercise `assign_gaff2_atom_types`'s internal Kekulize call (it Kekulizes its own
  molecule and calls `_atom_bond_facts` directly) — confirmed via monkeypatching the
  real call away and rerunning the whole golden suite with zero output changes. Added
  `test_assign_gaff2_atom_types_kekulizes_before_matching`, a spy test that guards the
  wiring itself; verified it fails when the internal call is removed.
- `_bond_category_facts`'s new Kekulization precondition was documented but
  unenforced. Added a runtime check that raises a clear, attributable error if it's
  ever violated, and changed `assign_gaff2_atom_types`'s `KekulizeException` handler
  to log and re-raise instead of silently degrading (silent degradation would have
  masked itself as a confusing failure at the new precondition check instead).

Also flagged (not changed, recorded for human awareness): `adversarial_survived=True`
is a defensible but real judgment call, not an automatic consequence of the evidence
— see the comment in `scripts/validation/gaff2_parity_verdict.py` above `EVIDENCE`.

**Update 3, same day (2026-08-20) — verdict upgraded to PARITY.** A follow-up sprint
(PR #28: AR1/AR2/AR3 external-reference fix + H-atom typing rework; PR #29: Supplement-
tier benchmark) closed the two remaining registered-hypothesis gaps with real evidence:

- **H4 (AR1/AR2/AR3), was AMBIGUOUS → now MATCH.** A real external reference
  (antechamber/`openmmforcefields.generators.GAFFTemplateGenerator`, gaff-2.11, run via
  a pre-existing conda-forge micromamba environment already on this machine — no new
  setup needed) **confirmed the prior AMBIGUOUS status was actually masking a real bug**:
  furan/pyrrole/thiophene's ring carbons should be the `cc` family, not `ca`. Root cause
  was the identical "special-before-general defeated by an over-broad sentinel" pattern
  PR #26 fixed for `cp`/`ca`: AR1/AR2/AR3 all collapsed into one match-anything
  sentinel. Fixed: AR1 now requires membership in a 6-membered aromatic ring
  specifically (matching the DEF footer's own "benzene and pyridine" citation for AR1);
  re-verified against the same reference, and confirmed to generalize on a real
  production molecule outside the original 3 test cases (histidine-probe's imidazole).
- **H5 (naphthalene bridgehead), was AMBIGUOUS → now MATCH.** The same external
  reference confirmed the *existing* exact-`1RG6`-ring-count implementation was already
  correct — no code change needed, just real confirmation replacing Attacker #2's
  external-legend inference.

`clause_parity_pct` moves from 0.6 (3/5) to **1.0 (5/5)**. `ambiguity_load` moves from
`load_bearing` to `none` (the genuine ambiguity is resolved, not just documented).
`reproduction_rung` moves from `R0` to `R1` (a real comparison against an independent
reference implementation now exists, on 4 of 24 benchmark molecules — not a full R2+
systematic run). Recomputed grade: **PARITY**, all five ceilings green.

**What PARITY does and doesn't cover — read this before treating it as unconditional.**
This grade covers exactly `parity.bth.toml`'s 5 pre-registered hypotheses (all
heavy-atom-typing questions). Two real, documented gaps sit outside that registered
scope and are NOT resolved by this grade:
- **cc/cd ring-alternation** (a torsion-parameter bookkeeping convention — real GAFF2
  alternates `cc`/`cd` around a conjugated ring; proxide gets the atom-type *family*
  right but not the alternation). There is no `cd` ATD rule anywhere in
  `ATOMTYPE_GFF2.DEF` (confirmed by grep), so this genuinely isn't part of "does the
  rule engine correctly implement the DEF file" — it needs a separate ring-traversal
  algorithm, tracked as a future item, not a defect in this verdict.
- **H-atom typing's electron-withdrawing-atom element set** (`_EW_ATOMS = {N, O, F, Cl,
  Br, I}`, PR #28): a real, DEF-file-verified implementation of the h1-h5/hx/hn/ho/hs/hp
  family, but the specific EW element set is a documented judgment call the DEF file
  itself never enumerates. H-atom typing was never part of this campaign's registered
  hypotheses (which only ever covered heavy atoms), so this doesn't affect the grade —
  but it's a real, standing judgment call worth knowing about if H-atom typing
  correctness matters for downstream work.

See `.praxia/docs/decisions/260818_gaff2-parity-verdict-policy.md` for what this PARITY
grade means for the demistify ligand-extension gate (idea-002) — per the original
cross-repo plan, a genuine PARITY verdict satisfies that gate outright, no
accepted-PARTIAL sign-off needed. That decision doc's update flags this explicitly for
human confirmation rather than assuming the gate should open silently.

## Phase 3: Adversarial Refutation (M=3)

Three independent agents, each stating an assumption upfront and defaulting to
"deviation" on inconclusive evidence, per the honesty-tax protocol:

| Attacker | Lens | Target | Verdict |
|---|---|---|---|
| #1 | struct | H2: f8 bond-count disambiguation | **Defect found — major.** `_bond_category_facts` computes lowercase `sb`/`db` identically to uppercase `SB`/`DB`, dropping the DEF footer's "includes aromatic single/double" inclusive semantics. Demonstrated on toluene's ipso carbon (a real match/no-match flip); does not corrupt the `c`/`cs` benchmark molecules tested, because `cs`/`c` are actually disambiguated by f9 (S vs O/S neighbor), not by which f8 branch fires. |
| #2 | struct | H3+H5: cp neighbor-pattern, naphthalene bridgehead | **No defect found — moderate-high confidence.** No new mistyping across anthracene, pyrene, triphenylene, triphenylbenzene, terphenyl. No overcorrection (genuine biphenyl-type junctions still resolve to `cp`). Found supporting-but-not-conclusive external evidence (openbabel/openbabel's independently maintained `gaff.dat` legend: "`cp` — head sp2 C that connects two rings in biphenyl sys.") favoring the code's current exact-`1RG6`-count reading over the alternative. |
| #3 | stats | H1: full benchmark-set rule-precedence sweep | **Defect found (out-of-scope for this grade) + benchmark-authoring deviations.** Every heavy-atom type across all 14 curated molecules is internally correct per the DEF rules. A separate, code-confirmed defect was found in H-atom typing (`_H_TYPE_BY_HEAVY` missing most nitrogen ATD types — see "Excluded finding" below). Several `benchmarks/gaff2_parity_molecules.yaml` notes overclaim or misdescribe actual behavior (see Follow-ups). |

Full per-molecule scorecards and code/DEF-file citations are preserved in each
attacker's dispatch transcript (agent IDs referenced in this campaign's session log);
not reproduced verbatim here for length.

## Phase 4: Adjudication (orchestrator-owned re-derivation, Constraint 1)

Per the campaign's re-derivation lock, both claimed code defects were independently
re-derived — not trusted from the agent reports — via
`tests/test_gaff2_parity_invariants.py`. **Both were then fixed the same day
(follow-ups #1/#2 below); the tests now lock in the fixes as passing assertions, not
xfail** — original wording preserved here for the historical record:

1. ~~`test_bond_category_facts_sb_includes_aromatic_confirmed_defect` (xfail,
   strict)~~ — confirmed Attacker #1's finding directly: toluene's ipso carbon has 3
   heavy bonds (2 aromatic + 1 single) and should satisfy `3sb`, but the tally gave
   `sb=1`. **FIXED**; now
   `test_bond_category_facts_sb_db_include_aromatic_kekule_identity` (passing).
2. `test_f8_bond_count_disambiguation_no_regression_on_h_ew_benchmark_molecules`
   (passing) — confirms the defect above did **not** corrupt formamide,
   N-methylacetamide, acetone, or acetate's carbonyl-carbon typing (still true after
   the fix).
3. ~~`test_h_type_by_heavy_missing_amide_n_types_confirmed_defect` (xfail,
   strict)~~ — confirmed Attacker #3's H-typing finding directly:
   formamide/N-methylacetamide's amide nitrogen (`nt`/`ns`) was absent from
   `_H_TYPE_BY_HEAVY`, so its H silently typed as `hc` instead of `hn`. **FIXED**;
   now `test_h_type_by_heavy_amide_n_h_types_as_hn` (passing).

All three re-derivations reproduced the agents' claims exactly (2 xfail, 1 pass) at
the time of the original verdict. `tests/test_gaff2_golden.py`'s 28 (now 32) tests
remained green throughout — no regression from either fix.

**Severity ranking** (per `04_adjudicate.md`'s rubric, as originally adjudicated):

- Defect 1 (sb/db inclusive semantics): **major**. Real, reproducible, but scoped —
  did not corrupt the current benchmark set. Fix direction chosen: Kekulize a local
  copy before rule matching (`assign_gaff2_atom_types`), giving the true per-bond
  Kekule single/double identity while preserving `GetIsAromatic()` for AB/DL — more
  precise than the alternative (folding all aromatic bonds into both `sb` and `db`,
  which would over-count).
- Defect 2 (`_H_TYPE_BY_HEAVY` missing N types): **major, was out of this grade's
  scope, now fixed anyway**. H-atom typing was never covered by
  `parity.bth.toml`'s hypotheses (heavy-atom typing only) and remains formally
  deferred as "Section B" of the plan that produced PR #26 (the *full* h1-h5/hx/ha
  sp2-vs-sp3 + electron-withdrawing-neighbor-count rework is still not done) — but
  this specific, narrow bug (missing dict entries silently defaulting to the wrong
  element's H type) was cheap and DEF-file-verified to fix (`ATOMTYPE_GFF2.DEF` lines
  79-82: `hn`/`ho`/`hs`/`hp` are each unconditional on the *specific* N/O/S/P
  sub-type), so it was fixed rather than left as a documented gap.

## Phase 5: Graded Verdict

**Historical record — superseded by Update 3 above.** The table, grade, and confounds
block below reflect the state at the ORIGINAL verdict (PARTIAL, `clause_parity_pct=0.6`,
`ambiguity_load=load_bearing`, `reproduction_rung=R0`). Current state:
`clause_parity_pct=1.0`, `ambiguity_load=none`, `reproduction_rung=R1`, grade **PARITY**
— see Update 3 for the full derivation and `scripts/validation/gaff2_parity_verdict.py`
for the current, executable source of truth. The confounds block immediately below has
been updated in place (not kept historical) since it's the citable artifact future
campaigns should actually reference.

Evidence (full derivation in `scripts/validation/gaff2_parity_verdict.py`'s comments):

| Dimension | Value | Ceiling |
|---|---|---|
| `invariant_pass` | True | PARITY |
| `clause_parity_pct` | 0.6 (3/5 hypotheses MATCH — H2 moved from DEVIATION to MATCH after follow-up #1's fix; H1/H3 MATCH; H4 AR1/AR2/AR3 and H5 naphthalene-bridgehead remain AMBIGUOUS) | PARTIAL |
| `adversarial_survived` | True (no hypothesis mechanism-nullified) | PARITY |
| `reproduction_rung` | R0 (text-parity only; no local antechamber/OpenFF run) | PARITY |
| `ambiguity_load` | load_bearing (AR1/AR2/AR3 unresolved — unaffected by the follow-up fixes) | PARTIAL |

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
label = "GAFF2 heavy-atom typing is validated against its own DEF-file specification (PARITY) and against a real external reference implementation on 4 of 24 benchmark molecules; H-atom typing and cc/cd ring-alternation sit outside the registered hypothesis scope -- see Update 3's 'What PARITY does and doesn't cover'"
[confounds.reference_parity]
reference_paper = "AmberTools ATOMTYPE_GFF2.DEF (vendored spec; see parity.bth.toml citation_note)"
reference_metric = "exact atom-type match (equivalence_bound = 0.0)"
reference_value = 1.0
equivalence_bound = 0.0
parity_run_id = "260820_gaff2_parity_phase3"  # this campaign; no prior bth run/campaign_id existed to append to (see Provenance note below)
verdict = "PARITY"
reproduction_rung = "R1"  # real antechamber/GAFFTemplateGenerator comparison, PR #28
```

## Sign-offs resolved

Both `[NEEDS HUMAN SIGN-OFF]` items in
`.praxia/docs/decisions/260818_gaff2-parity-verdict-policy.md` are now filled in (see
that file's diff in this same commit):

- **Approver for accepted-PARTIAL verdicts:** Marielle.
- **h_ew Option A vs. B:** resolved as **fully satisfied** — not the open A/B choice
  as originally framed, and no longer even the "residual gap" framing this section
  originally used. PR #26 correctly implemented f8's digit-count *arithmetic* and
  fixed the *disambiguation outcome* for every currently-curated h_ew molecule
  (re-verified here, not just trusted from the PR description); the one confirmed
  residual gap this section originally flagged (lowercase `sb`/`db` inclusive-bond
  semantics) is now fixed outright (follow-up #1), not just documented.

## Follow-ups

**Status as of Update 3: #1-#5 are all DONE.** See "What PARITY does and doesn't
cover" in Update 3 for two NEW, real, deliberately-out-of-scope items this sprint
surfaced (cc/cd ring-alternation; H-atom typing's EW-element-set judgment call) —
not listed as numbered follow-ups here since they were never part of this
campaign's registered hypotheses, but tracked and linked from Update 3 for anyone
who needs H-typing or torsion-parameter correctness specifically.

1. ~~Fix `_bond_category_facts`'s sb/db inclusive semantics~~ **DONE.**
   `assign_gaff2_atom_types` now Kekulizes a local copy
   (`clearAromaticFlags=False`) before rule matching, so `sb`/`db` reflect the true
   per-bond Kekule identity (including aromatic bonds) while `SB`/`DB` stay exact/
   non-aromatic. Never mutates the caller's molecule. Locked in as
   `test_bond_category_facts_sb_db_include_aromatic_kekule_identity` (passing).
2. ~~Fix `_H_TYPE_BY_HEAVY`'s missing nitrogen ATD types~~ **DONE** (narrow fix, not
   the full Section B rework). Unrecognized heavy types now fall back to an
   element-derived default (`_H_TYPE_ELEMENT_DEFAULT`: N→hn, O→ho, S→hs, P→hp,
   verified against `ATOMTYPE_GFF2.DEF` lines 79-82) instead of a blanket `hc`;
   carbon keeps its existing coarser `hc` default. Locked in as
   `test_h_type_by_heavy_amide_n_h_types_as_hn` (passing, exercises the real
   `build_gaff2_ffxml` path end-to-end via its emitted FFXML, not just the lookup
   table in isolation). The full h1-h5/hx/ha sp2-vs-sp3 + electron-withdrawing-
   neighbor-count rework (Section B) remains a separate, larger follow-up.
3. ~~Correct `benchmarks/gaff2_parity_molecules.yaml`'s authoring deviations~~
   **DONE.** Indene and Anthracene's false `cp` claims corrected with
   mechanism-specific notes (Indene fails cp's f9 3-aromatic-neighbor requirement,
   same as Toluene's ipso carbon; Anthracene fails f8's exact `1RG6` ring-count
   token, same as Naphthalene — these are two *different* mechanisms, verified
   separately rather than assumed identical). 1,3-Butadiene's backwards
   terminal/internal claim, and Divinyl-ketone/Acrolein's overclaimed `ce`/`cc`
   coverage for terminal `=CH2` carbons, corrected with a DEF-file-grounded
   explanation (`ce`'s f9 pattern requires a single-bonded heavy neighbor, which a
   double-bond-only terminal carbon structurally cannot have).
4. ~~Add golden-test coverage~~ **DONE.** Toluene and Anthracene locked in as
   parametrized golden cases; Thioacetone (`CC(=S)C`) added for first-ever `cs`
   coverage (verifies `cs`/`c` are disambiguated by f9's S-vs-O/S neighbor pattern,
   not by which f8 branch fires — both hit `[1DB,0DL]`). `[2DL]`/`[3sb]` got
   matching-level coverage (`atomic_prop_matches` against constructed
   `AtomBondFacts`) rather than a real-molecule test — no real molecule was found
   that reaches those specific `c`/`cs` branches through actual rule precedence
   without introducing an unverified claim; parsing-level coverage for these tokens
   already existed.
5. ~~AR1/AR2/AR3 sub-classification remains unresolved~~ **DONE** (PR #28). Closed
   exactly as this item predicted: via a real antechamber/OpenFF reference run
   (`scripts/validation/gaff2_external_reference.py`), which found the prior
   AMBIGUOUS status was masking a real bug (furan/pyrrole/thiophene should be `cc`,
   not `ca`) and fixed it (AR1 now requires 6-membered-ring membership). See
   Update 3 for the full derivation.

## Provenance note

No bathos run/campaign was ever registered for Phase 1/2 of this campaign (they ran
informally, no `bth run` tracking — confirmed via `~/.bth/catalog`, zero GAFF2-tagged
runs). Per direction, this campaign starts run-tracking clean at Phase 3 rather than
fabricating retroactive Phase 1/2 records. PR #26 (`ba60145`, `fb95944`, merge
`a49ca94`) and the Phase 1 blind-reconstruction record are the prior evidence this run
builds on, cited here rather than represented as prior tracked runs.

## Reproduce this verdict

1. `uv run pytest tests/test_gaff2_golden.py tests/test_gaff2_parity_invariants.py -v`
   — expect 55 passed, 0 xfailed (51 golden [incl. 10 Supplement-tier cases +
   TestHAtomTyping's 8 cases] + 4 invariants, all passing after PR #27/#28/#29's
   fixes landed).
2. `uv run python scripts/validation/gaff2_external_reference.py` — real
   antechamber-vs-proxide comparison for furan/pyrrole/thiophene/naphthalene; expect
   naphthalene MATCH, the three heteroaromatics DIFFER only on cc/cd alternation (the
   explicitly out-of-scope gap — atom-type family already matches).
3. `uv run python scripts/validation/gaff2_parity_verdict.py` — expect `PARITY`
   (`clause_parity_pct=1.0`, `ambiguity_load=none`, `reproduction_rung=R1`, all five
   ceilings green).
4. If cc/cd ring-alternation or H-atom typing's EW-element-set ever get scoped into a
   future campaign's registered hypotheses, update `parity.bth.toml` and
   `scripts/validation/gaff2_parity_verdict.py`'s evidence accordingly and re-run the
   grader — do not hand-edit the verdict.
