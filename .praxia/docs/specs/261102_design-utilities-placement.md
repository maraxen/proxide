---
title: 'Shared design-screening utilities and lesson enforcement: placement across proxide, alynxr, aminx and project specs'
description: 'Where the TfR1 lessons live long term: inventory of proxide against our TfR1 script functions, placement per lesson, parity-test plan and open decisions; design only, no code'
status: draft
task_id: 261102_design-lessons-systemization
date: '261102'
backlog_ids: ''
adversarial_review: ''
---
# Shared design-screening utilities and lesson enforcement

Source of the lessons: `.praxia/docs/audits/261102_tfr1-epic-close-out.md` (L1-L13). Tags: **[V]** verified by reading the source this session, **[R]** from a Haiku recon report, not re-read, **[U]** unverified.

## 1. Principle

Lessons that are mechanical become library functions and linted spec fields; judgment lessons become a spec checklist and lesson records. Stateless sequence and structure utilities go to the lowest shared layer (`proxide`); anything that runs a predictor or knows about campaigns stays in `alynxr`; policy stays in specs. Scripts in project repos become thin wrappers.

## 2. Inventory: proxide vs our TfR1 functions

| Capability (TfR1 source) | In proxide today | Verdict |
|---|---|---|
| Composition vs background, net charge, aromatic fraction, entropy, GRAVY, pI (`tfr1_composition.py`: `freqs`, `charge`, `summarize`; `tfr1_mpnn_redesign.py`: `metrics`) | none [V] | **new** |
| Interface contacts between two chains, per-residue contact sets (`tfr1_sa_screen.py: interface_counts`, `tfr1_interface_charge.py: pair_counts`, `tfr1_p3_mouse.py: contacts_and_dock`; the S1 docking rule is a fourth copy) | `geometry/metrics.py` functions raise `NotImplementedError` (6 stubs) [V]; `run_confind` was completely deleted in commit `47c397d5b` (debt #1901) [V] | **new** (and a fix to the stub path; one implementation instead of four) |
| Ion-pair and like-charge contact classification (`pair_counts`) | none [V] | **new**, builds on the contact function |
| Contact-set Jaccard with alignment mapping (`tfr1_p3_mouse.py: jaccard, contact_overlap, aligned_pairs`) | `jaccard` module is MinHash-sketch distance over sourmash-style sketch Parquets, NOT contact sets [V] | **new** (name it so it cannot be confused with the sketch module) |
| TM-align, structure novelty | `tm_align, tm_scores_fixed_correspondence exist [V]` | **exists**; use instead of any foldseek-side reimplementation |
| AF vs MPNN alphabet conversion (`af_to_mpnn`, hit in the MPNN redesign) | `chem/conversion.py` (JAX) and `io/parsing/mappings.py` (NumPy) both derive permutation tables via runtime dependency `alphex.perm`, pinned by `tests/test_alphabet_conformance.py` [V] | **exists or partly**; aminx already uses its own `aminx.utils.aa_convert`; confirm which is canonical before adding anything |
| Vote logic with predictor family (`tfr1_p5b_check.py: vote, member_verdict`) | n/a | **not proxide**: belongs to `alynxr` arm metadata (see §3) |

## 3. Placement per lesson

- **L2 (sequence screen)** → `proxide` new pure-Python submodule (`proxide.seqqc`): functions in §2 rows 1, plus a `screen(sequences, background)` that returns a result dict shaped for a bathos `[result_schema]`.
- **Interface and docking utilities (L1 follow-through, L11)** → `proxide` (contacts, ion pairs, Jaccard). The S1 docking rule is policy: it stays a thin function in the project spec that calls the contact function.
- **L4 (predictor lineage, effective votes), L7 (pin tool commit, reader contract)** → `alynxr`: add `family` and `lineage` to the arm definition, a command that reports effective independent votes for a set of arms, and a reader that takes the unit-name contract from the campaign plan (the TfR1 F3 reader failed on A:A names and P3b arm keys drifted).
- **MPNN redesign with a fixed target chain** → `aminx`: a recipe or CLI feature for fixed-token specs. The CLI could not express them, so the TfR1 run used the Python API. Whether the Python-API driver matches the CLI was never compared **[U]**.
- **L1, L8, L9, L10 (gate bypass, validated submission artifact date, slate-freeze date, amendment budget, null-round limit)** → a design-campaign spec skeleton plus, if feasible, `spec_lint` rules. Whether `spec_lint` can express "gate with a bypass branch must list the decisions that become unvalidated" or "freeze date present" is **[U]**; probe before committing to it.
- **L3, L5, L6, L12 (generator diff, organiser extensions marked, speculative arm date, objective needs mechanism)** → the same skeleton, as checklist items.
- **All lessons** → also filed in praxia's lesson store so recon and spec agents see them.

## 4. Constraints on the extraction

1. **Registered runs are not rewritten.** `tfr1_composition.py`, `tfr1_interface_charge.py`, `tfr1_sa_screen.py` and `tfr1_p3_mouse.py` stay as committed. The library versions must reproduce the registered outputs (run ids dda7de75 composition, bba33e3a interface audit, 20aeafaf SA) as parity tests, then project scripts may switch to thin wrappers under a new sidecar and run.
2. **Every utility ships with positive and negative selftests** (as in `tfr1_interface_charge.selftest`) and returns a dict matching a documented result schema.
3. **Batched, not looped.** Interface and composition functions take arrays or lists of designs; test batched equals looped.
4. **No predictor calls in `proxide`**; no cluster calls.
5. **Dependency direction:** `alynxr`, `aminx` and project repos depend on `proxide`, never the reverse. `aminx` already declares `proxide` [V]; `alynxr` and `adaptyv` do not [V] and would add it.

## 5. Open decisions (the user's)

1. **Import constraint. DECIDED 2026-11-02 (user): accept the Rust build.** The sequence screen lives in `proxide` as-is; consumers (`alynxr`, this project) build the extension with maturin. Consequence to plan for: add `proxide` as a dependency of each consumer and document the build in each README; no lazy-import change. Original analysis: `proxide/__init__.py` imports `proxide._proxider` (the compiled Rust extension) at import time [V, line 18], so a pure-Python submodule cannot be imported or tested without building the extension. Options: (a) accept the Rust build for consumers (alynxr and adaptyv would need maturin builds), (b) make `__init__` import lazily so `proxide.seqqc` works without the extension, (c) put the sequence screen in a small separate pure-Python package and keep `proxide` for structure-heavy functions. This changes the cost of every consumer, so it should be decided before any code is written.
2. ~~Where the sequence screen lives if (c)~~: moot, option (a) chosen.
3. **Scope of the first slice.** Suggested: `seqqc` (composition, charge, aromatic, entropy) with parity against run dda7de75, plus the one contact function with parity against bba33e3a.
4. **Repo access.** Implementation touches `proxide` and `alynxr`, which are outside this worktree; each needs its own branch and worktree.

## 6. Out of scope

Editing global rules or CLAUDE.md (owner item); changing any registered TfR1 result; running any new fold.
