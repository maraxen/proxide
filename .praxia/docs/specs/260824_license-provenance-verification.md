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
payload={audit_id, verdict, issues/findings})` when the MCP tool is bound to
this workspace, else append directly to `.praxia/audits.jsonl` in the same
shape — reuses existing infra rather than inventing a parallel record
format. The payload **must** carry `license_provenance_crate: "<crate>"` as
a top-level key — see the false-positive note below for why. For
`MISMATCH`/`NEEDS-HUMAN` verdicts, additionally file a narrative doc under
`.praxia/docs/audits/` per the existing internal-docs convention, since
those need a human-readable writeup a reviewer can act on, not just a JSONL
row.

**Both tiers are implemented and tested against this repo, not just
designed**: tier 1 is `scripts/check_license_provenance.py`; tier 2 is the
`.claude/skills/license-provenance-check` skill (fully specifies the method
above, including the exact audit-record shape). Building and testing tier 1
against the live repo surfaced two real bugs worth recording as evidence for
the "why a lint alone doesn't fix this" argument above, not just the abstract
case:

- **First cut matched "audited" by substring-containment** (crate name
  appears anywhere in an `.praxia/audits.jsonl` record) — this is a
  general project-wide log with plenty of unrelated entries that mention a
  crate name in passing (a functional code review, a bugfix), so it produced
  false "resolved" readings. Fixed by requiring an explicit
  `license_provenance_crate` marker field, not a substring match.
- **First cut matched provenance phrases anywhere in a spec doc that
  mentioned the crate name anywhere** — this doc's own Appendix names every
  crate in result tables and is full of exactly the phrase vocabulary
  ("reimplementation", "port of", "derived from") the lint watches for,
  so nearly every crate got flagged by this doc's own prose. Fixed by
  scoping signal-matching to the paragraph containing the crate-name
  mention, not the whole document.
- **Even after both fixes, the lint still misses `proxide-frag`'s real (weak)
  signal** — its spec says "MASTER-style backbone fragment search," which is
  a bare proper-noun allusion, not phrased as a porting claim the regex
  watches for. This is accepted, not treated as a bug to chase: catching
  every possible allusion to a tool name would need a maintained registry of
  "external tools this org's labs have built," which defeats tier 1's "cheap
  deterministic tripwire" purpose. It's exactly what tier 2's periodic
  full-sweep (not just flag-triggered) mode exists to catch instead — and did,
  in today's sweep.

## Rollout

- **Phase 0 (done today)**: retroactive full-crate sweep via the Agent tool
  (see Appendix) — no additional mismatches beyond the two already known.
- **Phase 1 (done today, no CI wiring)**: tier 1 (`scripts/check_license_provenance.py`)
  and tier 2 (`.claude/skills/license-provenance-check`) both exist and are
  developer-invoked — run the script locally/pre-push, and run the skill in
  a Claude Code session on any flag or before merging a crate that ports
  external code.
- **Phase 2 (deliberately deferred, tracked as debt #1397)**: CI-wiring was
  scoped and decided against for now, not left ambiguous. GitHub's Copilot
  coding agent needs a paid Copilot plan (Pro/Pro+/Business/Enterprise) plus
  explicit org policy enablement — not available on this repo's account
  tier, and headless Claude Code in CI was explicitly ruled out too. GitHub
  Models (`actions/ai-inference`) is a better structural fit — free, runs on
  the default `GITHUB_TOKEN` (`models:read` scope auto-granted to any
  Action), no paid plan or org policy gate the way Copilot coding agent
  needs — but it's a plain chat-completion API with no autonomous tool/web
  access, so a CI workflow would need a deterministic `curl` step to fetch
  primary-source license text first and hand it to the model for judgment
  only; that plumbing hasn't been built or live-tested end-to-end in this
  repo. Rather than block Phase 1 on resolving that, or wire something
  unverified into a merge gate, it's filed as debt (#1397) and tier 2 stays
  developer-side until someone picks it up.

## Open decisions

- Does a `MISMATCH` verdict **block merge**, or just require a filed
  `.praxia/docs/audits/` writeup + explicit sign-off comment? Moot for now
  with no CI gate to block — but worth deciding before Phase 2 is built, so
  debt #1397 inherits an answer instead of picking one implicitly.
  (Recommend: block — matches how the GPL-gating and pending-Mosaist
  decisions were actually handled, as stop-and-ask, not
  silent-merge-with-a-note.)
- Retention: audit records never expire, but should a crate's `NOTICE` state
  be considered stale after N months and re-verified? (Not urgent — flag for
  later, don't block Phase 1 on it.)

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

**Cross-cutting flag surfaced by this pass:** `src/proxide/assets/` had no
`LICENSE`/`NOTICE` files anywhere despite `260618_system-prep-scope.md`'s own
stated policy that each vendored force field should live in its own dir with
its upstream `LICENSE`/`NOTICE`. Followed up as its own audit — see
**Appendix B** below, which found a live, actively-shipping violation.

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

## Appendix B: `src/proxide/assets/` force-field data audit

Follow-up to the cross-cutting flag above. This is asset *data*, not crate
*code* — a distinct audit from the crate sweep, using the same primary-source
method. Three parallel passes (CHARMM, AMBER protein FF, water/implicit —
the last one still in flight as of this revision).

### CGenFF — live, actively-shipping violation, most urgent finding in this doc

`src/proxide/assets/charmm/charmm36_cgenff.xml` (2.3MB, git-tracked since the
repo's initial commit) is the CGenFF (CHARMM General Force Field)
parameter set, converted to OpenMM XML by openmmforcefields. Its actual
upstream terms, fetched directly from `mackerell.umaryland.edu/charmm_ff.shtml`
(primary source, quoted verbatim):

> "The CGenFF topology and parameter files are included with the full
> release of the CHARMM additive toppar files... Frequent users of the
> CGenFF program may wish to obtain a binary license. The procedure for
> obtaining a free-of-charge not-for-profit license is initiated by
> e-mailing us; it may take up to a few weeks and will require someone
> with signature authority at your institution to sign a license
> agreement... For-profit users may obtain the CGenFF program from
> SilcsBio, LLC."

This is a bilateral signed agreement, not a permissive or even a standard
copyleft license — not-for-profit use requires an institutional signature,
commercial use requires purchasing from a named third-party vendor. **And
this file is not just sitting in git**: `pyproject.toml`'s packaging config
(`include = ["src/proxide/**/*", "LICENSE"]`) wildcards in the entire assets
tree, so `charmm36_cgenff.xml` has been built into every published wheel
since the initial commit — this is actively distributed to every `pip
install proxide` user today, unconditionally, with no license gate. This is
the most urgent finding in this document: worse than `proxide-confind`/
`proxide-rotlib` in mechanism (compiled wheel distribution via PyPI, not just
source-visible-on-GitHub) and comparable or worse in severity (a required
signed agreement plus a named-vendor commercial carve-out, versus CC BY-NC-SA's
blanket noncommercial clause).

**Resolved for future releases (2026-08-24)**: `pyproject.toml`'s
`[tool.maturin]` now excludes the file from the built wheel (verified via a
real `maturin build` — confirmed absent from the resulting wheel, sibling
CHARMM files unaffected); the file was `git rm`'d from the working tree; and
`scripts/sync_forcefields.py`'s CHARMM mapping no longer references it, so a
future re-sync won't reintroduce it.

**Deliberately not purged from git history.** The file is present in the
repo's true initial commit and untouched since — removing it via
`git filter-repo` rewrites that commit's hash and therefore every
descendant: all 587 commits on `main` and every one of 60+ local/60+ remote
branches built on top, including 30+ worktrees active in this session alone.
Weighed against that cost: the working-tree removal + packaging fix already
stop it reaching anyone who clones or installs going forward, and anyone who
cloned before today already has a local copy regardless of what happens to
the remote's history — a full rewrite doesn't retroactively un-ship
anything already downloaded, including the PyPI wheels published before this
fix. Decided (2026-08-24, explicit choice, not a default) to hold off unless
a concrete reason forces it later — e.g. a specific request once in contact
with the MacKerell lab.

### Main CHARMM36 toppar (protein/nowaters/waters_ions) — NEEDS-HUMAN

Same two locations (`assets/charmm/charmm36{,_nowaters,_protein}.xml`,
`assets/charmm/waters_ions_*.xml`) checked against the MacKerell lab page and
openmmforcefields' own `charmm/README.md` (fetched from
`raw.githubusercontent.com/openmm/openmmforcefields/main/charmm/README.md`).
Neither states an explicit grant to redistribute the converted parameter
files — the lab page's only license language found is the CGenFF warning
above; the "free for nonprofit/academic use" statement associated with
CHARMM licenses the *program*, registered separately at
`brooks.chem.lsa.umich.edu/register/`, not the parameter data files
themselves. Widely redistributed by convention (openmmforcefields does
exactly this, and many other tools follow suit) but no positive written
grant found to point to. **Genuinely ambiguous — not a confirmed clean
permission and not a confirmed violation the way CGenFF is.**

### AMBER protein/nucleic force-field family — OK-by-absence-of-restriction

`assets/amber/*.xml` (ff14SB, ff19SB, ff99SB, lipid17/21, DNA/RNA-OL series)
and `assets/openmm_bundled/amber14/`, `amber19/` (confirmed exact file-list
match against `openmm/openmm`'s own `wrappers/python/openmm/app/data/`).
openmmforcefields' repo tree has exactly one license-related file — its root
MIT `LICENSE` — with no per-forcefield carve-out anywhere under
`ffxml/amber/`, unlike AmberClassic (whose own README explicitly flags
per-component license exceptions). OpenMM's own `docs-source/licenses/Licenses.txt`
itemizes component licenses (MIT for the API/Reference/CPU platforms, LGPL
for CUDA/OpenCL, separate notices for a few named third-party algorithms) and
has no item covering bundled force-field data — it ships as part of the
MIT-licensed `openmm.app` package with no carve-out. This is a real
primary-source basis (an upstream MIT project with no stated exception,
bundling and redistributing the data itself) but it's an *absence of a
restriction*, not a positive "yes this is MIT" statement the way
`ATOMTYPE_GFF2.DEF`'s GPL determination was positive — worth documenting the
reasoning explicitly in a NOTICE rather than treating it as fully closed.

Two exceptions inside this same family, **not** covered by the reasoning
above:
- `GLYCAM_06j-1.xml` — actual upstream is the Woods Lab (`glycam.org`), a
  different origin from AmberTools proper, with a documented history of its
  own separate academic-use terms. Not verified this pass (the lab's license
  page wasn't locatable at the URL guessed) — **NEEDS-HUMAN**.
- DNA/RNA-OL series (`DNA.OL15.xml`, `RNA.OL3.xml`, etc.) — separate
  publication lineage (Ivani/Orozco lab), not independently traced beyond
  the blanket AMBER-family reasoning — **NEEDS-HUMAN, lower priority** (no
  known history of restrictive terms, just genuinely unverified).

Also found: `assets/protein.ff19SB.xml` (top-level, outside `amber/`) is
byte-identical to `amber/protein.ff19SB.xml` — a duplicate for a different
load path. `assets/protein.ff14SB.xml` is a **different** file (230KB vs
343KB) from `amber/ff14SB.xml`, sizes suggesting it actually came from
`openmm_bundled/amber14/` instead — not a licensing concern (same
OK-by-absence-of-restriction verdict either way) but an untracked provenance
gap worth closing.

Separately: `scripts/sync_forcefields.py` clones openmmforcefields' default
branch with **no commit pin** — there is no record of which snapshot is
actually vendored right now, and re-running the script would silently drift
to whatever's current upstream. A packaging-hygiene gap, not a licensing one,
but worth fixing alongside.

### `openmm_bundled/charmm36{,_2024}/` — OK, lower risk

Distinct from the main CHARMM36 toppar above: confirmed via `openmm/openmm`'s
repo tree that these files exactly match OpenMM's own bundled
`wrappers/python/openmm/app/data/charmm36{,_2024}/` and are **water models
only** (spce, tip3p variants, tip4p variants, tip5p variants) — not
CHARMM protein or CGenFF content. Standard published water models, already
bundled and redistributed by the upstream OpenMM project itself. A NOTICE
crediting the bundling is good practice but not urgent.

### `water/` — OK, needs-NOTICE-only

`assets/water/*.xml` (tip3p/tip4pew/tip4pfb/spce/opc standard + ion
parameter files) — confirmed via openmmforcefields' full recursive repo tree
as their own XML-encoding of published water-model parameters
(TIP3P/TIP4P-Ew/OPC/SPC-E), not a redistribution of separately-licensed
software. Same root MIT `LICENSE` already confirmed for this repo, no
per-file carve-out exists. **Safe to vendor as-is — needs a NOTICE crediting
openmmforcefields/OpenMM, nothing more.**

### `implicit/` and `openmm_bundled/implicit/` — OK, needs-NOTICE-only

Neither comes from openmmforcefields (confirmed: zero "obc"/"implicit" hits
in its full tree). Both trace directly to `openmm/openmm`'s own repo —
`wrappers/python/openmm/app/data/{amber03_obc,amber10_obc,amber96_obc,amber99_obc}.xml`
and `implicit/{obc1,obc2}.xml`, exact filename match. OpenMM's GitHub API
license field reports `null` (same detector-miss pattern as Mosaist earlier
this session — the real answer isn't in a field GitHub can auto-detect).
The actual answer is in OpenMM's own `docs-source/licenses/Licenses.txt`
(OpenMM has no single root `LICENSE` — it's a multi-section document because
OpenMM genuinely mixes licenses by component), fetched directly: **Section 1
states "The OpenMM API, the Reference Platform, and the CPU platform may be
used under the terms of the MIT License"**, and these `app/data/` files are
Python-layer bundled data under that same component, not the separately
LGPL-licensed CUDA/OpenCL platform code (Section 2). **Safe to vendor as-is
— needs a NOTICE citing OpenMM `Licenses.txt` §1 specifically** (citing
"OpenMM is MIT" without the section would be wrong if ever applied to
anything actually sourced from the GPU platform code).

### Net effect

**One confirmed live violation** (CGenFF, actively shipping in every
published wheel — needs immediate, separate attention, not resolved by this
doc). **One genuinely ambiguous case** worth a NOTICE stating the ambiguity
honestly (main CHARMM36 toppar — no positive redistribution grant found, but
no confirmed block either). **Everything else in the tree is OK**, needing
only NOTICE files it currently lacks: AMBER protein family
(absence-of-restriction basis, two still-unverified carve-outs — GLYCAM,
DNA/RNA-OL lineage), `openmm_bundled/charmm36` water models, `water/`, and
`implicit/`+`openmm_bundled/implicit/`. None of this was caught by the crate
sweep above — it's a structurally different problem (bundled data files, not
ported/reimplemented code), which is itself evidence that tier 1/tier 2 as
scoped (crate-focused) needs a sibling pass over `src/proxide/assets/`
specifically, not just crates going forward.

## Appendix C: final architecture -- runtime-fetch + bring-your-own (2026-08-24)

Implemented and verified against the live repo, not just designed. Two
decisions that shaped this, both worth recording since they overturned an
earlier plan:

**Why not "get MIT a not-for-profit CGenFF license"?** The original plan
(draft an email requesting a not-for-profit license) doesn't actually solve
proxide's problem: proxide is a general-use open-source package, and a
license signed by one institution for its own use doesn't grant proxide the
right to redistribute to every downstream user, including commercial ones.
Getting the maintainer a personal license would have solved a problem
proxide doesn't have.

**Why not "fetch from openmmforcefields' GitHub at runtime instead of
bundling"?** This was floated as a way to avoid "redistributing" -- the
theory being that pointing a runtime fetch at openmmforcefields' own copy
means proxide never holds/ships the content itself. Rejected: (a)
openmmforcefields' own redistribution isn't established as authorized either
(Appendix B found no documented grant), so there's nothing clean to "just
link to"; (b) the functional-effect distinction that matters isn't which
server bytes come from, it's whether software automates the reproduction --
proxide's own code performing an HTTP fetch, writing bytes to disk, and
using them is a reproduction either way, timing (build vs. runtime) doesn't
change that. Recorded here because it's a tempting-sounding shortcut that
doesn't actually hold up, and a future contributor may reach for it again.

**What's actually built:**

- `proxide.assets._fetch`: fetch-and-cache for `amber/`, `water/`,
  `implicit/`, and the non-CHARMM parts of `openmm_bundled/` -- pinned
  commits (not branches) of `openmm/openmmforcefields` and `openmm/openmm`,
  cached under `platformdirs.user_cache_dir("proxide")` (overridable via
  `PROXIDE_ASSET_CACHE_DIR`). No per-file sha256 pinning (unlike
  `ATOMTYPE_GFF2.DEF`'s single-file case) -- a pinned commit SHA is already
  content-addressed, and hashing ~90 individual files for this batch wasn't
  proportionate; re-pin deliberately, don't float on a branch.
- `is_charmm_restricted(path)`: deny-by-default for anything under `charmm/`
  or with "charmm" in the path, with a narrow, explicitly-audited allowlist
  for exactly the two confirmed-water-only subdirectories
  (`openmm_bundled/charmm36/`, `openmm_bundled/charmm36_2024/`). Caught a
  real near-miss during implementation: `openmm_bundled/charmm36.xml` and
  `charmm36_2024.xml` (flat top-level files, distinct from the
  already-audited water-only *subdirectories* of the same name) turned out
  to contain the full CGenFF-including CHARMM36 force field -- confirmed by
  inspecting actual residue content (thousands of non-water residue names)
  before trusting the name similarity. A broad, path-based deny rule with a
  narrow allowlist is safer here than an enumerated restricted-file list.
- `resolve_charmm_toppar(name)`: bring-your-own via `PROXIDE_CHARMM_TOPPAR_DIR`;
  raises `CharmmLicenseRequiredError` with a pointer to this doc when unset.
- `proxide.assets._asset_index.ASSET_INDEX`: static, committed name to path
  map (76 entries) generated from the tree as it existed before conversion --
  lets `load_force_field("protein.ff14SB")`-style bare-name lookups resolve
  without any files present on disk (the old code used `rglob`, which can't
  find what isn't vendored anymore). Regenerate via
  `scripts/generate_asset_index.py` if the pinned commits are ever bumped.
- 116 files removed from git tracking and from the wheel (`amber/`, `water/`,
  `implicit/`, `openmm_bundled/`, all of `charmm/` including the two loose
  `protein.ff{14,19}SB.xml` duplicates) -- verified via a real `maturin build`
  that none of it ships anymore, `gaff/` unaffected.
- Fixed a real, unrelated bug found while rewriting the collection-time test
  (`tests/assets/test_load_all_forcefields.py` used to `rglob` the assets
  dir inside a `@pytest.mark.parametrize` decorator argument, which runs at
  *collection* time -- silently unworkable once these dirs aren't vendored):
  rewritten to parametrize over the static index (no I/O at collection
  time) and fetch lazily inside each test body, skipping (not failing) on
  network unavailability. Full pass, cold cache, real network: 96 passed, 25
  skipped, 0 failures.
- `sync_forcefields.py`'s role narrowed to "populate a local scratch copy for
  `generate_asset_index.py` to walk" (its actual asset-vendoring purpose is
  gone) -- also fixed a pre-existing dead `ASSETS_DIR = Path("src/priox/assets")`
  typo (stale project-rename leftover, unrelated to this task but found
  while touching the file) and removed a `generate_readme()` step that would
  have overwritten the now-hand-maintained `src/proxide/assets/README.md`
  with stale generic content on next run.
