---
title: 'License provenance verification: two-tier lint + agentic check'
description: CI tripwire + agentic primary-source license verification for reimplemented/ported external code, plus the retroactive crate sweep that motivated it
status: draft
task_id: 260824_license-provenance-sweep
date: '260824'
backlog_ids: ''
adversarial_review: ''
---

# License provenance verification

## Context

Two real, already-shipped licensing defects surfaced this session, both the same
shape: a crate reimplements or ports code from an external repository, and its
`Cargo.toml` license declaration doesn't reflect that.

1. **`proxide-gaff2`** (in-progress Rust GAFF2 port, `gaff2-parity/rust-port`
   branch): embeds `ATOMTYPE_GFF2.DEF`, derived from AmberTools/antechamber
   (`Amber-MD/AmberClassic`). Confirmed **GPL-2.0-or-later** by fetching
   AmberClassic's actual root `LICENSE` file directly — not by triangulating
   web search results. Resolved this session: crate relicensed
   `GPL-2.0-or-later` with `NOTICE`/`LICENSE`, the engine gated behind a
   non-default `gaff2-engine` Cargo feature so the *published* wheel/crate
   stays MIT, and the DEF file itself moved to fetch-at-build-time
   (`scripts/fetch_amber_assets.py`, pinned commit + sha256) instead of being
   vendored in git.

2. **`proxide-confind`** and **`proxide-rotlib`** (already published to
   crates.io 2026-07-22, both `license.workspace = true` → MIT): both crates'
   own spec docs (`.praxia/docs/specs/260529_confind.md`,
   `.praxia/docs/specs/260529_rotlib.md`) cite `Grigoryanlab/Mosaist@450816a`
   as their **source reference**, with correspondences precise enough
   (`Residue::getPhi(false)`/`getPsi(false)`, `cache()` lines 158–160) to read
   as a port of the actual source, not an independent implementation from a
   paper. Fetched Mosaist's actual README directly: it is licensed
   **CC BY-NC-SA 4.0** (Attribution-NonCommercial-ShareAlike). This is a live,
   already-published mislicensing — worse than the GPL case, since
   NonCommercial can't be feature-gated around (it bars any commercial use of
   the derivative outright) and ShareAlike would require the derivative to
   carry the same restrictive terms. **Not yet resolved** — needs a business
   decision (contact Grigoryan lab for alternative terms, build an
   independent-implementation case, or rewrite/relicense) before anything
   else changes; out of scope for this doc.

Both were found by hand: reading each crate's spec doc for a "source
reference"-shaped citation, then fetching the *actual* upstream license from
a primary source. Nothing before this session checked for this class of
defect at all — there was no lint, no CI gate, no audit trail.

## Why a lint alone doesn't fix this

A regex/grep tripwire on a fixed phrase (`"Source reference:"`, `"port of"`,
etc.) has an unbounded false-negative surface — whatever phrasing convention
exists today, a future crate that doesn't happen to use it slips through
silently, and the two live incidents above were found via genuine judgment
(recognizing that citing exact function names and line numbers implies
source-level correspondence, not just "these two things are related") that a
deterministic pattern can't reproduce. The lint's job is narrower than that:
catch *candidates* cheaply and refuse to let them merge unverified. Judging
whether a candidate is actually a problem — what's the real upstream license,
is this a derivative or an independent implementation, is the declared
license compatible — is exactly the kind of primary-source research +
judgment call this session did by hand for GAFF2 and Mosaist. That has to
stay agentic; encoding it as more regex just moves the false-negative risk
around instead of closing it.

## Design: two tiers

### Tier 1 — CI lint (deterministic, recall-oriented, decides nothing)

Runs on every PR touching `crates/*/Cargo.toml`, `crates/*/src/lib.rs` (or
`src/main.rs`), or `.praxia/docs/specs/**`/`.praxia/docs/decisions/**`, plus a
weekly full-repo sweep as a backstop (a provenance signal can land in a spec
doc without touching the crate's own source in the same PR).

**Signal patterns** (case-insensitive, deliberately broad — false positives
are cheap, false negatives are the actual risk):
- Phrasing: `port(ed)? of`, `based on`, `derived from`, `reimplement(s|ation
  of)`, `ported from`, `matches .* exactly`, `source reference` (the existing
  hand-written convention, kept as one signal among several, not the only
  one)
- External repo URLs: `https?://(github|gitlab)\.com/[\w.-]+/[\w.-]+`
- Paper citations: DOI (`10\.\d{4,9}/\S+`) or arXiv (`arXiv:\d{4}\.\d{4,5}`)
  patterns

**Scan surfaces**: `crates/<crate>/Cargo.toml` `description` field,
`crates/<crate>/src/lib.rs`/`src/main.rs` leading `//!` doc block, and any
`.praxia/docs/{specs,decisions}/*.md` whose filename or content references
the crate.

**Fail condition**: a crate with a signal hit that has neither (a) a
crate-local `NOTICE`/`LICENSE` file, nor (b) a matching `append_audit` record
in `.praxia/audits.jsonl` dated after the signal was introduced. Also flags
the inverse case that actually happened here: a crate that *has* a `NOTICE`
file (i.e., was already flagged and given provenance docs) but whose
`Cargo.toml` still inherits `license.workspace = true` without an override —
catches exactly the "we wrote the NOTICE but forgot to flip the license
field" class of slip.

The lint never blocks on its own judgment of whether a signal is a real
problem — only on whether tier 2 has run and recorded a verdict for it.

### Tier 2 — Agentic verification (judgment, primary-source only)

Triggered by a tier-1 flag (new/changed crate), or run standalone for a
retroactive sweep (as done today — see Appendix). One agent dispatch per
flagged crate, using the same prompt shape used for today's sweep:

1. Read the crate's `Cargo.toml` description + declared license, and its
   `//!` doc block.
2. Find every provenance signal for that crate (spec docs, decision docs,
   doc comments).
3. For each cited external repo, fetch its *actual* license from primary
   source — `curl https://api.github.com/repos/<org>/<repo>` (the `license`
   field) plus a raw `LICENSE`/README fetch as fallback when the API field is
   null/`"other"`. **WebSearch triangulation is explicitly disallowed as the
   basis for the license determination** — it's what would have missed
   Mosaist's NC/SA clause, since a search-engine summary of "Mosaist license"
   doesn't reliably surface the specific CC variant.
4. If only a paper/method is cited with no code reference, note that
   distinction explicitly — algorithm-from-a-paper and port-of-source are
   different risk profiles (the former is generally not a copyright
   derivative-work concern; the latter is).
5. Compare upstream license against the crate's declared license and render
   a verdict.

**Verdict taxonomy** (no auto-resolution — a MISMATCH or NEEDS-HUMAN verdict
is a human decision, same as the GPL-gating and (pending) Mosaist decisions
this session):
- `CLEAN` — no external code provenance found (original work or algorithm
  from a paper without a source port)
- `OK` — external source cited, upstream license compatible with the
  declared license (e.g., MIT/BSD/Apache-2.0 upstream, MIT declared)
- `MISMATCH` — upstream license incompatible with what's declared (GPL,
  CC-NC/SA, proprietary, or unclear-but-restrictive)
- `NEEDS-HUMAN` — ambiguous correspondence (paper vs. source port unclear),
  upstream license itself ambiguous, or evidence couldn't be resolved

**Audit record**: write via `transduction_log(action="append_audit",
payload={audit_id, verdict, issues/findings})` — reuses the existing
`.praxia/audits.jsonl` infra (queryable via `transduction_query`) rather than
inventing a parallel record format. For `MISMATCH`/`NEEDS-HUMAN` verdicts,
additionally file a narrative doc under `.praxia/docs/audits/` per the
existing internal-docs convention, since those need a human-readable writeup
a reviewer can act on, not just a JSONL row.

## Rollout

- **Phase 0 (today)**: retroactive full-crate sweep, run manually via the
  Agent tool — see Appendix for results once complete. No CI wiring yet.
- **Phase 1 (available now, no new infra)**: for any new crate or provenance
  signal, dispatch tier 2 manually/on-demand — this is exactly what the sweep
  in the Appendix does, just not yet gated by a lint.
- **Phase 2 (needs a decision)**: wire tier 1 into CI (cheap — a grep-based
  GitHub Actions step) and tier 2 into an automated dispatch on flag. Tier 2
  in CI needs either headless Claude Code CI access (a `claude -p` invocation
  with a fixed prompt + structured output) or routing through
  `rig_run`/workflow infra if that's preferred to be kept in-house — this is
  an open decision, not resolved by this doc.

## Open decisions

- Does a `MISMATCH` verdict **block merge**, or just require a filed
  `.praxia/docs/audits/` writeup + explicit sign-off comment? (Recommend:
  block — matches how the GPL-gating and pending-Mosaist decisions were
  actually handled, as stop-and-ask, not silent-merge-with-a-note.)
- Phase 2 CI auth/dispatch mechanism (headless Claude Code vs. `rig_run`) —
  needs a call from whoever owns CI secrets/billing for this repo.
- Retention: audit records never expire, but should a crate's `NOTICE` state
  be considered stale after N months and re-verified? (Not urgent — flag for
  later, don't block Phase 1/2 on it.)

## Appendix: retroactive sweep inventory

Three parallel tier-2 passes dispatched over the remaining workspace crates
(13 crates, all outside the two already-known incidents). All three complete.

Already known from this session (not re-swept by these passes):

| Crate | Verdict | Upstream | Evidence |
|---|---|---|---|
| `proxide-gaff2` | Resolved | AmberTools/antechamber (`Amber-MD/AmberClassic`) | GPL-2.0-or-later, confirmed via AmberClassic's root `LICENSE`; crate relicensed + feature-gated this session |
| `proxide-confind` | **MISMATCH, unresolved** | `Grigoryanlab/Mosaist` (`mstcondeg`) | CC BY-NC-SA 4.0, confirmed via Mosaist's README; declared MIT; already published to crates.io |
| `proxide-rotlib` | **MISMATCH, unresolved** | `Grigoryanlab/Mosaist` (`mstrotlib.cpp`) | CC BY-NC-SA 4.0, confirmed via Mosaist's README; declared MIT; already published to crates.io |
| `proxide-tmalign` | OK | Zhang Lab USalign/TM-align (`pylelab/USalign`) | "Permissive custom text (not GPL)" per this crate's own spec (`260729_proxide-tmalign-phases-2-5.md`); porting fine with attribution — not independently re-verified against primary source in this pass |

Infra pass (complete):

| Crate | Verdict | Upstream | Evidence |
|---|---|---|---|
| `proxide-core` | CLEAN | — | Generic; the `forcefield` module parses bundled CHARMM/AMBER XML assets (asset-parsing code, not derived logic — see asset-tree flag below) |
| `proxide-geometry` | CLEAN | — | Generic geometric algorithms, no external source cited |
| `proxide-io` | CLEAN | — | Generic, no external source cited |
| `proxide-units` | CLEAN | — | Standard physical unit-conversion constants (GROMACS↔AMBER, nm/kJ·mol↔Å/kcal·mol) — not copyrightable expression from a specific codebase |

Incidental finding from this pass: `.praxia/docs/specs/260603_charmm-ic-sourcing.md`
(2026-06-03, predates this session) already states *"No `rotlib.bin` (CC
BY-NC-SA) committed or redistributed"* — the team already knew Mosaist's
`rotlib.bin` **binary asset** was CC BY-NC-SA and correctly avoided shipping
that file. That awareness evidently didn't extend to the **reimplemented
code itself** in `proxide-rotlib`/`proxide-confind`, which is the live,
unresolved defect above — i.e. this was a partial mitigation that missed the
code, not a blind first encounter with Mosaist's licensing.

Bindings/utility pass (complete):

| Crate | Verdict | Upstream | Evidence |
|---|---|---|---|
| `proxide_rs` | CLEAN | — | Pure internal re-export/glue layer, no external source |
| `proxide_py` | CLEAN | — | PyO3 binding glue, no algorithmic content |
| `proxide_fixer` | OK | PROPKA3 (LGPL-2.1), PDB2PQR (BSD), PDBFixer (MIT) | All three invoked as external subprocesses (`std::process::Command`), not linked/ported — wrapper staying MIT is the correct posture. Already documented in `.praxia/docs/specs/260618_system-prep-scope.md` (Rosetta explicitly excluded there for its non-commercial license) |
| `proxide-wasm` | OK | openmmforcefields (MIT) | `gaff2.rs` embeds `gaff-2.11.xml` via `include_str!`; matches this session's existing asset-provenance confirmation; crate declares `license = "MIT"` explicitly (not workspace-inherited) |
| `proxide-jaccard` | CLEAN | — | `distance.rs`'s merge-based Jaccard kernel is documented original work; "sourmash-style" refers only to input-file-format compatibility (parquet schema), not ported code |

Physics/algorithm pass (complete):

| Crate | Verdict | Upstream | Evidence |
|---|---|---|---|
| `proxide-gaff` (old coordinate-only heuristic, distinct from `proxide-gaff2`) | CLEAN | — | No embedded data files, no source-repo citation anywhere in src or specs |
| `proxide-physics` | CLEAN | — | No doc-comment provenance, no bundled data files, no spec citing an external source |
| `proxide-parallel-rt` | CLEAN | — | Minimal thread-count registry for wasm32 builds, pure infra |
| `proxide-frag` | **OK, weak NEEDS-HUMAN flag** | MASTER (Grigoryan lab, same lab as Mosaist) — cited by name in `.praxia/docs/specs/260602_proxide-master-spec.md` as "MASTER-style backbone fragment search" | Unlike confind/rotlib, this spec has **no** source-reference pin, no function/line correspondence to MASTER's source — it implements Kabsch RMSD (standard public-domain 1976 algorithm) over its own `proxide-confind` dependency. Reads as an independent implementation of a published *approach*, not a translation of copyrighted expression. Attempted to verify MASTER's own license directly (same primary-source method): **no `MASTER` repo exists under the `Grigoryanlab` GitHub org** (confirmed via `api.github.com/orgs/Grigoryanlab/repos`) — it isn't distributed there, so this can't be closed out the same way confind/rotlib were. Current evidence doesn't support treating this as a derivative, but the same-lab pattern (2-for-2 on Mosaist/CC-BY-NC-SA elsewhere) means someone should track down MASTER's actual distribution terms (lab website or direct contact) before calling this fully closed. |

**Cross-cutting flag surfaced by this pass, outside its assigned scope:**
`src/proxide/assets/` (`amber/`, `water/`, `implicit/`, `charmm/`,
`openmm_bundled/{amber14,amber19,charmm36,charmm36_2024,implicit}`,
`gaff/ffxml`, `gaff/dat`) has **no `LICENSE`/`NOTICE` files anywhere**,
despite `260618_system-prep-scope.md`'s own stated policy that each vendored
force field should live in its own dir with its upstream `LICENSE`/`NOTICE`.
Only `gaff/dat/ATOMTYPE_GFF2.DEF` has been resolved (this session). Both
`proxide_fixer` (`amber_ff` feature) and `proxide-wasm` (`gaff2.rs`) read
from this tree, and `proxide-physics` (still-running pass) almost certainly
does too. CHARMM in particular has historically carried more restrictive
academic-use terms than AMBER and hasn't been checked. **Recommend this as
its own follow-up sweep — not pulled into this doc's scope**, since it's a
distinct asset-tree audit rather than a code-provenance one.

Also surfaced by the physics/algorithm pass: `.praxia/docs/specs/260618_system-prep-scope.md`
already contains a prior ADR for exactly this class of problem on the
Python/asset side — *"Forcefield licensing — bundle per-asset under each FF's
own license. proxide code stays MIT; each vendored FF lives in its own dir
with its upstream LICENSE/NOTICE"* (CHARMM36 = public-domain+attribution,
Amber ff14SB/GAFF2 = GPL-in-subdir, OpenFF Sage = MIT). This spec's tier-1/
tier-2 design should be understood as **extending that existing policy to the
Rust crates and automating enforcement of both**, not inventing a parallel
convention — the project has done this diligence before, it just never
became a lint or an audit trail, and (per the asset-tree flag above) was
never actually executed for most of the bundled assets it describes.

### Summary verdict counts

19 crates total (2 known incidents + `proxide-gaff2` + 16 swept): 13 CLEAN,
3 OK, 2 **MISMATCH (unresolved)**, 1 OK-with-weak-flag pending an
unreachable primary source. No new MISMATCH beyond the two already known
(`proxide-confind`, `proxide-rotlib`) — the sweep's value was closing out
uncertainty on the other 16, not finding a third incident.
