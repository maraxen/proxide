---
name: license-provenance-check
description: Verify a crate's declared license against the actual upstream license of any external code it ports or reimplements, using primary-source evidence. Use when scripts/check_license_provenance.py flags a crate, when adding a new crate that ports/reimplements external code, or when auditing an existing one on request.
---

# License provenance check (tier 2)

This is the agentic half of the two-tier license-provenance design in
`.praxia/docs/specs/260824_license-provenance-verification.md`. Tier 1
(`scripts/check_license_provenance.py`) is a cheap, deterministic tripwire —
it catches *candidates* by regex and cannot judge them. This skill is the
judgment: given a flagged crate (or one a developer wants checked before
merging), determine what it actually derives from and whether the declared
license is correct.

**Not CI-wired.** Run this manually, in a Claude Code session, before merging
a crate that tier 1 flags (or that you know ports/reimplements something).
See the spec's "Rollout" section for why: this needs real tool access
(fetching primary sources) that a plain CI chat-completion call doesn't have
without significant extra plumbing, and that plumbing is filed as tech debt,
not built yet.

## Method

1. **Find every provenance signal for the crate.** Its `Cargo.toml`
   description, its `src/lib.rs`/`src/main.rs` doc comment, and any
   `.praxia/docs/{specs,decisions}/*.md` that discusses it. Don't stop at
   the first signal — a crate can cite more than one upstream (e.g. a
   dependency on a bundled data file *and* a ported algorithm).

2. **Distinguish "ported source" from "implements a published method."**
   These are different risk profiles. Citing exact function names, line
   numbers, or byte-for-byte output matching against an upstream binary is
   evidence of a source port (a real derivative-work question). Citing a
   paper or a named approach without any such correspondence is evidence of
   an independent implementation (generally not a copyright derivative-work
   concern — algorithms/ideas aren't copyrightable, expression is). When
   genuinely unclear, say so — don't force a clean answer.

3. **Fetch the actual upstream license from primary source. Never rely on
   web-search triangulation for the determination itself** — a search
   engine's summary of "X license" has already been wrong once this session
   (it's what would have missed Mosaist's CC BY-NC-SA NonCommercial clause
   specifically, versus just "some Creative Commons variant").
   - GitHub-hosted: `curl -sSL https://api.github.com/repos/<org>/<repo>`
     (check the `license` field — `"other"`/`"NOASSERTION"` means check by
     hand) and `curl -sSL https://raw.githubusercontent.com/<org>/<repo>/<ref>/LICENSE`
     (or `README.md`, since not every repo has a dedicated `LICENSE` file —
     Mosaist's terms are stated in its README, not a `LICENSE` file).
   - Non-GitHub (a lab's own distribution page, a journal page, etc.):
     `dangerouslyDisableSandbox: true` is usually needed, since the default
     sandbox's allowed network hosts are GitHub/PyPI/crates.io plus a short
     fixed list — say so when you use it.
   - If the upstream repo/page can't be found at all (as happened this
     session for `proxide-frag`'s "MASTER-style" citation — no `MASTER` repo
     exists under the citing lab's GitHub org), that's a real outcome to
     report, not a reason to guess.

4. **Compare and render a verdict** — no auto-resolution. A `MISMATCH` or
   `NEEDS-HUMAN` verdict is a business/legal decision for a human, the same
   way the GPL-gating decision (`proxide-gaff2`) and the still-open
   Mosaist decision (`proxide-confind`/`proxide-rotlib`) were handled this
   session: surface it, don't pick for them.

   - `CLEAN` — no external code provenance (original work, or an algorithm
     implemented from a paper with no source-level correspondence)
   - `OK` — external source cited, upstream license compatible with what's
     declared
   - `MISMATCH` — upstream license incompatible (GPL, CC-NC/SA, proprietary,
     or unclear-but-restrictive) with what's declared
   - `NEEDS-HUMAN` — ambiguous correspondence, ambiguous upstream license, or
     upstream couldn't be located to check at all

## Recording the result

Two places, both required when the verdict is anything other than a bare
`CLEAN` with nothing further to do:

1. **`transduction_log(action="append_audit", ...)`** (or, if that MCP tool
   isn't bound to this workspace in your session, append directly to
   `.praxia/audits.jsonl` in the same shape). The payload **must** include
   `license_provenance_crate: "<crate-dir-name>"` as a top-level payload
   key — this exact field name is what `scripts/check_license_provenance.py`
   looks for to mark a crate as resolved. Matching by "does the crate name
   appear anywhere in the audit log" was tried and produces false positives
   (the log is a general project log full of unrelated mentions) — use the
   explicit marker, not a substring match.

   ```json
   {
     "audit_id": "260824_license-provenance_proxide-confind",
     "payload": {
       "license_provenance_crate": "proxide-confind",
       "verdict": "MISMATCH",
       "upstream": "Grigoryanlab/Mosaist@450816a",
       "upstream_license": "CC BY-NC-SA 4.0",
       "evidence": "https://raw.githubusercontent.com/Grigoryanlab/Mosaist/450816a/README.md",
       "declared_license": "MIT (license.workspace = true)"
     }
   }
   ```

2. **For `MISMATCH`/`NEEDS-HUMAN` only**: file a narrative doc under
   `.praxia/docs/audits/` per the repo's internal-docs convention
   (`praxia docs add` / `docs(action="add")`, category `audits`) — a
   reviewer needs a readable writeup, not just a JSONL row, to act on it.

3. **If the verdict resolves a previously-flagged crate cleanly** (e.g. you
   add a `NOTICE`/`LICENSE` file), a NOTICE/LICENSE file alone already
   satisfies tier 1 — the audit record is still worth writing for the
   evidence trail, but isn't load-bearing for the lint to pass.

## What this skill does not do

It doesn't decide what to *do* about a MISMATCH (relicense, gate behind a
feature, contact the upstream author, rewrite) — that's the same kind of
call the GPL-gating decision was, made by the person who owns the tradeoff,
informed by this skill's evidence. It also doesn't scan the whole repo for
candidates — that's tier 1's job (or a manual full-crate sweep, as done once
this session).
