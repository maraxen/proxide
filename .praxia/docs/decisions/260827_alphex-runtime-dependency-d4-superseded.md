---
name: 260827_alphex-runtime-dependency-d4-superseded
description: alphex promoted from dev-only to runtime dependency in proxide, superseding decision D4 for this repo specifically
metadata:
  type: decision
  task_id: 260827_alphex-ecosystem-migration-plan
  status: accepted
---

# ADR: alphex promoted to a runtime dependency (supersedes D4, proxide-only)

**Date:** 2026-08-27
**Status:** Accepted
**Task:** 260827_alphex-ecosystem-migration-plan

---

## Context

Decision D4 (recorded in `tests/test_alphabet_conformance.py` and the `pyproject.toml`
comment it cited) scoped `alphex` to **dev-group-only, conformance-testing use**: proxide's
hand-rolled alphabet permutation tables (`chem/conversion.py`'s `MPNN_ALPHABET`/`AF_ALPHABET`
and the byte-identical duplicate pair in `io/parsing/mappings.py`) would be *tested against*
`alphex`'s declarations, but nothing under `src/` would import the library, so no runtime
dependency edge would be created.

That scoping was correct for Phase 0: `alphex` did not yet exist on PyPI (it was a local
editable path dependency, `abcdefghijk` → `alphex` renamed 2026-08-14, resolved from PyPI
2026-08-15), and the goal was narrowly to *prove* the two hand-rolled tables agreed with the
ecosystem's canonical declarations before trusting them further.

This work package (proxide alphex-ecosystem migration, Work Package 1) goes further: it
retires the hand-rolled literal construction in both `chem/conversion.py` and
`io/parsing/mappings.py`, replacing `[MPNN_ALPHABET.index(k) for k in AF_ALPHABET]`-style
Python loops with `alphex.perm(known.AF_X_21, known.MPNN_X_21, policy=Policy.RAISE, dtype=...)`
calls, and derives the alphabet strings themselves (`MPNN_ALPHABET`, `AF_ALPHABET`) from
`alphex.known.MPNN_X_21.symbols`/`AF_X_21.symbols` rather than from local string literals. That
requires `alphex` to be importable wherever `proxide.chem.conversion` or
`proxide.io.parsing.mappings` is imported — i.e. at runtime, for every consumer of proxide, not
just inside proxide's own dev/test environment.

## Decision

**`alphex` is promoted from `[dependency-groups] dev` to `[project] dependencies` in
proxide's `pyproject.toml`, superseding D4's dev-only scoping — for proxide specifically.**

This does not reopen D4 as a general ecosystem policy. Other repos in the `alphex` conformance
census (proteinsmc, aminx, asr) that only *test against* `alphex` without retiring their own
literal tables remain correctly dev-only under D4; this decision applies to proxide alone,
because proxide alone is retiring its literals at the source in this work package.

## Rationale

**proxide is the ecosystem's origin alphabet declaration site.** Per
`tests/test_alphabet_conformance.py`'s own docstring: "proxide is the origin.
`chem/residues.py:623`'s `restypes` is the AlphaFold-derived ordering that proteinsmc copied
verbatim, and `chem/conversion.py:16-17` is what aminx duplicated byte-for-byte." Being the
origin site carries a stronger obligation than being a downstream consumer: proxide should
retire its own duplicate literals at the source rather than merely test against `alphex`
indefinitely while leaving the two hand-rolled tables (`chem/conversion.py`,
`io/parsing/mappings.py` — finding F3 of the original census, a byte-for-byte duplicate pair
within proxide itself) in place as the thing being tested. Testing-only leaves the
duplication-and-drift risk exactly where it was; delegating construction to `alphex` removes it
structurally, the same way the census's other finding (a shape-valid but silently wrong permutation
table) motivated `alphex.perm`'s `(src.size,)`-shaped, `Policy.RAISE`-by-default contract in the
first place.

**No new transitive dependency risk.** `alphex` is numpy-only (`from alphex import Policy,
known, perm` pulls in nothing beyond `numpy`, already a runtime dependency of proxide via
`jax`/the base install). Promoting it to `[project] dependencies` adds no new transitive
dependency edge beyond `alphex` itself, which is already resolved in every proxide dev/test
environment today under D4.

**Scope stays narrow.** Only `conversion.py`'s and `mappings.py`'s `af_to_mpnn`/`mpnn_to_af`
permutation-table *construction* is delegated to `alphex.perm`. The string-conversion helpers
(`string_key_to_index`, `string_to_protein_sequence`, `protein_sequence_to_string`) stay
proxide-local: `alphex`'s `Alphabet.encode`/`.decode` have no unknown-symbol fallback and
cannot express arbitrary caller-supplied `aa_map` dicts, which these helpers support and are
tested doing (e.g. `tests/chem/test_conversion.py`'s `custom_map = {"A": 10, "R": 20}` case).
`residues.py`'s `MAP_HHBLITS_AATYPE_TO_OUR_AATYPE` (a third hand-rolled permutation table) is
explicitly out of scope: `alphex` ships no AF-ordered 22-state alphabet with X@20/GAP@21 — a
documented gap in `alphex.known`'s own module docstring ("there is no `restypes_with_x_and_gap`
row ... Recorded as a known gap rather than guessed at") — so there is nothing to delegate to
yet.

## Consequences

- Every consumer of `proxide` (not just its dev/test environment) now transitively depends on
  `alphex>=0.1.0a1`.
- The stale `# Phase 0 alphabet conformance (decision D4). DEV GROUP ONLY -- nothing under
  src/ imports this, so no runtime dependency edge is created.` comment in `pyproject.toml` is
  removed, since it is no longer true.
- `tests/test_alphabet_conformance.py`'s `pytest.importorskip("alphex", ...)` guard remains
  harmless (alphex is now unconditionally installed) and is left as-is; it still documents the
  historical dev-only default for repos that have not made this same promotion.
- If `alphex`'s API changes in a way that breaks `Policy.RAISE`/`perm`/`known.MPNN_X_21`/
  `known.AF_X_21`, proxide now fails at import time for all consumers, not just in its own test
  suite. This is treated as acceptable: the exhaustive round-trip parity test added alongside
  this migration (deriving "expected" from the pre-migration literal strings, not from `alphex`
  itself) is the regression gate that would catch such a break before it reaches a release.

## Related

- `tests/test_alphabet_conformance.py` — D4's original dev-only conformance tests (unchanged,
  now redundant-but-harmless with the runtime promotion)
- `src/proxide/chem/conversion.py`, `src/proxide/io/parsing/mappings.py` — the two migrated
  sites
- `src/proxide/chem/residues.py`'s `MAP_HHBLITS_AATYPE_TO_OUR_AATYPE` — explicitly out of scope
  (no matching shipped `alphex` declaration)
- task_id: 260827_alphex-ecosystem-migration-plan, Work Package 1
