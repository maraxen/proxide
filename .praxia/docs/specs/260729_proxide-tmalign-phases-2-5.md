---
title: 'proxide-tmalign: Phases 2-5'
description: Remaining seeds, parity harness, orx-parallel, PyO3 bindings, bathos benchmark for the TM-align port
status: draft
task_id: 260729_tmalign_scaffold
date: '260729'
backlog_ids: ''
adversarial_review: ''
---
# proxide-tmalign: Phases 2-5

> Promoted from a personal, non-git-tracked plan file (`~/.claude/plans/purring-seeking-beaver.md`)
> so it's discoverable by future sessions and teammates. **Phase 0 (workspace scaffold) and Phase 1
> (core primitives: `structure.rs`, `seq.rs`, `d0.rs`, `kabsch.rs`, `nw.rs`) are complete** — merged
> via PR #8 (`feat/proxide-tmalign-scaffold`, rebased onto main 2026-07-29). This doc covers only
> the remaining Phases 2-5. For the line-referenced USalign algorithm detail needed to implement
> Phase 2's seeds and `DP_iter`, see
> [research/260729_tm-align-phase-2-algorithm-map](../research/260729_tm-align-phase-2-algorithm-map.md).

## Context

proxide has no structural-alignment scoring capability today — only fixed-length
backbone-fragment RMSD search (`proxide-frag`) and contact-degree analysis (`proxide-confind`).
TM-align (Zhang & Skolnick, *Nucleic Acids Res* 2005) is the standard tool for comparing two
protein structures independent of sequence identity, and its maintained superset **USalign**
(`pylelab/USalign`, permissively licensed) is the reference implementation the field cites. This
plan ports TM-align's core algorithm into a new Rust crate, adds `orx-parallel`-based parallelism
(the house parallel-iteration crate here), exposes it to Python via the existing `proxide_py`
bindings, and pre-registers a bathos parity experiment against the reference C++ binary — so the
port is trusted, not just "looks plausible."

Two scope decisions are locked in from user input:
- **v1 = single-pair TMalign only** (two Cα chains). MMalign (multi-chain), RNA mode, and circular
  permutation are explicitly deferred — this codebase's existing crates (`proxide-frag`,
  `proxide-confind`, `proxide-jaccard`) are each narrowly scoped to one algorithm, and MMalign's
  chain-assignment search is architecturally a separate layer on top of pairwise TM-align.
- **Parity reference numbers come from a locally-built USalign binary, frozen as committed Rust
  `const`s** — not a live `tmtools`/`usalign` test-time dependency (avoids a Python/native-build
  dependency in a pure-Rust test suite, and sidesteps `tmtools`'s GPLv3 wrapper layer).

## Established conventions this plan follows (verified directly, not assumed)

- **Workspace**: 16 crates under `crates/` (as of PR #8; was 15), hyphenated `proxide-<domain>`
  naming, declared in root `Cargo.toml` `[workspace] members` + `[workspace.dependencies]` at
  `0.1.0-alpha.15`. Crate: `proxide-tmalign` (already scaffolded).
- **Kabsch superposition precedent**: `crates/proxide-frag/src/kabsch.rs` — SVD on the 3×3
  cross-covariance `H = AᵀB` via `nalgebra::linalg::SVD`, with the standard `det(V·Uᵀ)<0`
  reflection-correction (negate column 2 of V). `proxide-tmalign/src/kabsch.rs` already generalizes
  this to variable-length `&[Vector3<f32>]` slices (Phase 1, done).
- **orx-parallel is the house parallel crate** (`rayon` only appears transitively via criterion,
  never used for production algorithms):
  - `crates/proxide-jaccard/src/matrix.rs:18-44` (`pairwise_jaccard_distance`) — row-parallel/pair-serial
    template: `(0..n).into_par().map(|i| (i+1..n).map(...).collect())`. Its sibling
    `pairwise_containment` (`matrix.rs:64-96`) is the template for **asymmetric** pairwise matrices
    (stores both directions per pair) — TM-score is asymmetric (`TM_norm1 != TM_norm2`) so
    cross-alignment batch scoring must follow `pairwise_containment`, not `pairwise_jaccard_distance`.
  - `crates/proxide-frag/src/search.rs:27-80` — dual serial/parallel API convention (`search` /
    `search_serial`, identical logic, `.par()` vs. plain iterator) for determinism/testability.
    `proxide-tmalign` follows this exact shape.
  - wasm gating: `#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]` +
    `proxide_parallel_rt::num_threads()`, as in `proxide-confind/src/parallel.rs:160-176`. Already
    present as a dependency block in `crates/proxide-tmalign/Cargo.toml`.
- **AA one-letter/three-letter codes**: `proxide_core::chem::residues::RESTYPE_1TO3` — reused by
  `proxide-tmalign/src/seq.rs` (Phase 1, done).
- **Python bindings**: single dedicated crate `crates/proxide_py` (lib `_proxider`,
  `crate-type=["cdylib"]`), one `#[pymodule]` entrypoint at `crates/proxide_py/src/lib.rs`
  registering everything via `m.add_function(wrap_pyfunction!(...))`. Concrete templates to copy
  (verified against actual code, not the stub crates):
  - `crates/proxide_py/src/py_jaccard.rs:26-74` (`jaccard_distance_matrix`) — the fully-wired
    numpy-in/dict-out template: heavy work inside `py.allow_threads(|| -> Result<_, String> {...})`
    (releases the GIL), errors mapped via
    `.map_err(|e| PyValueError::new_err(format!("... failed: {}", e)))` outside the closure, output
    built as a `PyDict` with `PyArray1::from_slice_bound(py, &flat).reshape((n, n))?` for array
    fields.
  - `crates/proxide_py/src/py_geometry.rs:53-72` (`kabsch_rmsd`) — the closer structural analog:
    normalizes coordinate input via the shared helper `crate::py_chemistry::extract_coords(py, &coords)?`
    (accepts a Python list-of-`[x,y,z]` *or* an `(N,3)` numpy array — **reuse this helper** for
    tmalign's Cα coordinate inputs rather than writing a new extractor), validates length equality
    with `PyValueError`, returns a `PyDict` with scalar + nested-list fields.
  - `py_confind.rs`/`py_rotlib.rs` are **stubs only** — not behavioral templates.
  - No `impl From<XError> for PyErr` exists yet anywhere in `proxide_py`; the idiom is inline
    `.map_err(...)` at each call site. `TmAlignError` (`crates/proxide-tmalign/src/error.rs`)
    should follow the `PyRuntimeError::new::<_, _>` direct-map pattern used for `fetch_*`
    (`crates/proxide_py/src/lib.rs:42,51,60,73`).
  - pyo3 `0.22` (`extension-module`, `abi3-py311`), `numpy = "0.22"`, built via `maturin` + `uv`.
    Re-export convention: thin `src/proxide/<name>/__init__.py` doing
    `from proxide._proxider import <fn>` (see `src/proxide/jaccard/__init__.py`).
  - **Not yet added** to `crates/proxide-tmalign/Cargo.toml`: `pyo3`, `numpy`, `insta`. `orx-parallel`
    and `approx` are already present.
- **Parity-test precedent**: `crates/proxide-confind/tests/test_parity_1dc7.rs` +
  `tests/common/mod.rs` — env-gated skip-if-absent loader helpers (e.g. `PDB_PATH`, `ROTLIB_PATH`),
  reference values as literal `const REF_*` arrays converted to a `HashMap` for lookup, tolerance
  via `const TOLERANCE: f64 = 5e-4` (epsilon compare, never exact match), skip via early `return`
  in the `#[test]` fn body (not `#[ignore]`). **This exact convention is already echoed into
  tmalign**: `crates/proxide-tmalign/src/structure.rs` has a `usalign_sample()` test helper using
  `USALIGN_REPO` (defaulting to `~/repos/USalign`) with the same skip-if-absent idiom — Phase 2
  extends this into a proper `tests/` parity harness, not more `src/`-inline tests. No bathos ties
  into proxide-confind's Rust tests (confirmed via grep) — bathos enters only at Phase 5.
- **bathos sidecar convention**: `.bth.toml` co-located with its driver script (e.g.
  `scripts/analysis/extract_rotlib_geometry.bth.toml`), schema `"bathos/0.3"`, `[experiment]` block
  (`slug`/`title`/`hypothesis`/`task_id`/`backlog`), `[outcomes.pass|marginal|fail]` each with
  `condition`+`reasoning`, `[result_schema]` with dotted-path keys, `[result_schema.provenance]`
  recording commit + input content hashes.
- **Tensor-bundle fixtures (`src/proxide/io/fixtures.py`) are Python-only, no Rust counterpart** —
  targets large nested-tensor E2E freezes (npz + JSON meta). TM-align's per-pair output is a
  handful of scalars, squarely in the confind "commit as literal consts" class — **do not** try to
  mirror the npz format in Rust for this harness.

## USalign/TMalign algorithm facts (grounding the port)

- **License**: permissive custom text (not GPL) — porting is fine with attribution + citing Zhang &
  Skolnick 2005 and related papers. `tmtools` (pybind11 wrapper, PyPI) bundles the same C++ code
  but its own wrapper is GPLv3 — reference its API shape only, don't copy its code.
- **Pipeline shape**: `TMalign_main()` runs 5 independent initial-seeding strategies (gapless
  threading, SS-seeded DP, local-structure superposition, SS+local combined, fragment gapless
  threading) with per-seed gap-open ranges over `{-0.6, 0}` — up to 10 `(seed, gap_open)`
  candidates total, each refined via `DP_iter` (sequential — Kabsch superposition + NW-DP,
  convergence-gated `|ΔTM|<1e-6`, up to 30 iterations) — best candidate kept by final TM-score.
  **This candidate set is the intra-alignment parallelism target (Phase 3a)**; `DP_iter` itself
  stays serial (inherent convergence dependency). Full algorithmic detail (exact function names,
  gap-open ranges per seed, iteration caps, the Gotoh-simplification quirk in `NWDP_TM` that must
  be replicated exactly) is in the companion research doc, not repeated here.
- **Confirmed zero threading anywhere in the reference C++** (no pthread/OpenMP/`#pragma omp`) —
  any Rust-side parallelism is new value-add, not a faithful port of concurrent reference behavior,
  and there's no thread-order nondeterminism to worry about when comparing outputs.
- **Final TM-score formula** (`param_set4final`, already ported in `d0.rs`): `D0_MIN=0.5`;
  `d0 = 0.5` if `Lnorm<=21` else `1.24*(Lnorm-15)^(1/3)-1.8`, clamped `>= D0_MIN`.
  `TM = (1/Lnorm) * Σ 1/(1+(d_i/d0)²)` over aligned pairs. Search-phase uses a looser d0
  (`parameter_set4search`).
- **Parseable CLI reference outputs**: `-outfmt 2` → one tab-separated line
  `name1 name2 TM2 TM1 rmsd seqID1 seqID2 seqID_ali xlen ylen n_ali8`; `-m matrix.txt` →
  rotation+translation block (`t[m]`, `u[m][0..2]`). Both are ideal, stable formats for generating
  frozen parity consts.

## Crate skeleton

```
crates/proxide-tmalign/
  Cargo.toml           # done — rlib, wasm cfg block present; [[bin]]/[[bench]] deferred to Phase 3b
  src/
    lib.rs              # done
    structure.rs        # done — CaTrace + extraction
    seq.rs               # done
    d0.rs                 # done
    kabsch.rs             # done — generalized (variable-N) Kabsch
    nw.rs                  # done — generic affine-gap NW DP core
    score.rs              # Phase 2 — TM-score evaluation given alignment + rotation
    seed/
      mod.rs               # Phase 2 — SeedKind enum + run_seed() dispatch (closed set of 5 — enum, not dyn trait)
      gapless.rs, secondary_structure.rs, local_structure.rs, ss_plus.rs, fragment_gapless.rs  # Phase 2
    refine.rs             # Phase 2 — DP_iter port, sequential, convergence-gated
    pipeline.rs            # Phase 2 (tmalign_pair_serial) / Phase 3a (tmalign_pair parallel twin)
    parallel.rs             # Phase 3b — cross-alignment batch scoring
    error.rs              # done
    bin/tmalign.rs          # Phase 3b — thin CLI driver, mirrors confind's bin
  tests/
    test_kabsch_general.rs, test_seed_gapless.rs, test_dp_refine.rs   # Phase 2
    test_parity_<pair>.rs (×2-3)   # Phase 2 — committed reference consts + TOLERANCE, confind-style
    common/mod.rs                  # Phase 2 — env-var-gated USALIGN_BIN regen helper
  benches/bench_pairwise.rs        # Phase 3b
```

Core result type (maps directly onto `-outfmt 2` fields):
```rust
pub struct TmAlignResult {
    pub alignment: Vec<(Option<usize>, Option<usize>)>, // residue-index pairs, None = gap
    pub rotation: [[f32; 3]; 3],
    pub translation: [f32; 3],
    pub rmsd: f32,
    pub tm_score_norm1: f32,  // normalized by structure 1 length
    pub tm_score_norm2: f32,  // normalized by structure 2 length
    pub seq_id1: f32, pub seq_id2: f32, pub seq_id_ali: f32,
    pub n_aligned: usize,
}
```

## Parallelism plan (staged, serial-correctness-first)

1. **Serial only** (Phase 2): `tmalign_pair_serial` iterates the up-to-10 `(seed, gap_open)`
   combinations sequentially, `max_by` on final TM-score. Parity is only meaningful once results
   are deterministic and serial-verified — don't introduce `orx-parallel` before this is green.
2. **Intra-alignment parallelism** (Phase 3a): `tmalign_pair` — `.par()` over the candidates (outer
   loop), `DP_iter` stays sequential per candidate. Dual-API convention exactly as
   `proxide-frag/src/search.rs`. Regression gate: `tmalign_pair` and `tmalign_pair_serial` must
   agree (within float-reduction-order tolerance) on the Phase-2 fixture set.
3. **Cross-alignment/batch parallelism** (Phase 3b): `pairwise_tm_scores(&[CaTrace]) -> Array2<f32>`-shaped
   API, row-parallel/pair-serial per `proxide-jaccard/src/matrix.rs:18-44`, calling
   `tmalign_pair_serial` in the inner loop (never nest `tmalign_pair`'s own parallelism). **Must
   follow the asymmetric `pairwise_containment` pattern** (two directional values per pair), not
   the symmetric `pairwise_jaccard_distance` pattern.
4. wasm-gate both `.par()` call sites uniformly with `proxide_parallel_rt::num_threads()`, matching
   sibling crates, even though there's no current tmalign+wasm consumer.

## Parity-testing harness (Phase 2)

- Build USalign locally from the `~/repos/USalign` clone (dev-only, not committed) to generate
  reference numbers.
- Commit only the derived scalars (TM-score, RMSD, rotation/translation, seqID fields) as literal
  `const`s in `tests/test_parity_<pair>.rs`, following `proxide-confind/tests/test_parity_1dc7.rs`
  exactly.
- Commit a small set of permissively-licensed (public-domain RCSB) PDB structure pairs directly
  under `tests/data/` — no env-var gating needed for the *inputs* (unlike confind's
  non-redistributable Mosaist fixtures), only for the *regeneration* step (`USALIGN_BIN` env var,
  env-gated `tests/regen_parity_consts.rs` that shells out to the local reference binary and
  asserts the committed consts haven't silently drifted — skipped in CI).
- Tolerances (accounting for `-ffast-math` in the reference build): TM-score `1e-4` absolute; RMSD
  `1e-3` Å; rotation matrix elements `1e-3` per-element plus an independent `RᵀR ≈ I`
  orthogonality self-check (`1e-4`); exact alignment-path match only asserted for unambiguous
  "easy" cases, informational-only for "hard" near-tied cases.
- Benchmark set (6-8 curated pairs, not a full published corpus — avoid premature scope): 2 "easy"
  (same fold, high identity), 2 "hard" (same fold, low/no identity — exercises the SS/fragment
  seeds), 2 "different-length" (tests d0 asymmetry), 1-2 unrelated-fold negative controls (TM-score
  near the ~0.17 random floor).

## bathos-tracked experiment (Phase 5)

`scripts/analysis/tmalign_reference_parity.py` + `scripts/analysis/tmalign_reference_parity.py.bth.toml`
(schema `bathos/0.3`):
- `hypothesis`: Rust `tmalign_pair_serial` reproduces reference USalign `-outfmt 2`
  TM-score/RMSD/rotation within the tolerances above across the curated benchmark set.
- `task_id = "260729_tmalign_scaffold"`.
- `[outcomes.pass/marginal/fail]`: pass = all pairs within tolerance; marginal = only "hard" pairs
  exceed tolerance (points to seed-selection/refinement edge case); fail = systematic deviation
  (points to a d0/DP-scoring/Kabsch-sign bug).
- `[result_schema]`: dotted paths per pair (`pairs.<id>.tm_score_norm1`, `.rmsd`,
  `.rotation_deviation_max`, `.reference_binary_version`).
- `[result_schema.provenance]`: script commit, crate commit, **pylelab/USalign git SHA used to
  build the reference binary** (USalign is actively developed; formulas could shift between
  versions), input PDB content hashes.
- Run this after Phase 2 (first honest parity claim), and again at full benchmark scale in Phase 5.

## Python bindings (Phase 4)

New `crates/proxide_py/src/py_tmalign.rs`, add `proxide-tmalign.workspace = true` to
`crates/proxide_py/Cargo.toml` (first time it's added), register via
`m.add_function(wrap_pyfunction!(py_tmalign::tm_align, m)?)?;` in `_proxider`, thin re-export
`src/proxide/tmalign/__init__.py`. API shape (numpy-in/dict-out, `tmtools.tm_align` as an interface
reference only):
```
proxide._proxider.tm_align(coords1: (N1,3) f32, coords2: (N2,3) f32, seq1: str, seq2: str) -> dict
  -> {"rotation": (3,3), "translation": (3,), "tm_score_norm1": float, "tm_score_norm2": float,
      "rmsd": float, "seq_id1": float, "seq_id2": float, "seq_id_ali": float, "n_aligned": int}
```
Defer `.pyi` stub coverage (consistent with the existing, accepted gap — jaccard/confind/frag have
none either). Note this explicitly as a known pre-existing gap in the Phase 4 PR description, not a
regression.

## Phase sequencing (each phase independently mergeable)

- ~~**Phase 0** — Clone+build `pylelab/USalign` locally, scaffold `crates/proxide-tmalign/`.~~ **Done, PR #8.**
- ~~**Phase 1** — `structure.rs`, `seq.rs`, `d0.rs`, generalized `kabsch.rs`, `nw.rs`.~~ **Done, PR #8.**
- **Phase 2** — Remaining 4 seeds; full up-to-10-candidate `tmalign_pair_serial`; parity harness v1
  (`tests/data/` + `test_parity_<pair>.rs` + `common/mod.rs` env-gated regen helper) against 2-3
  fixture pairs. **First honest parity-verified milestone.** Trigger first bathos run here.
- **Phase 3a** — `tmalign_pair` (orx-parallel intra-alignment twin) + serial/parallel equivalence
  tests.
- **Phase 3b** — `parallel.rs::pairwise_tm_scores` (cross-alignment, asymmetric-matrix pattern) +
  `benches/bench_pairwise.rs` (criterion, mirroring `proxide-jaccard`/`proxide-frag` bench
  conventions).
- **Phase 4** — `proxide_py` wiring (`py_tmalign.rs`, `src/proxide/tmalign/__init__.py`). Verify:
  `maturin develop` builds; Python smoke test matches Rust result on the Phase-2 fixture pair.
- **Phase 5** — Expand benchmark set to full curated 6-8 pairs; run/register the bathos experiment
  at full scale; classify pass/marginal/fail.

## Verification

- Each phase: `cargo check -p proxide-tmalign` / `cargo test -p proxide-tmalign` (narrow, local —
  do not run the full workspace suite per this machine's compute limits).
- Phase 2 boundary: parity tests pass within stated tolerances against the frozen reference consts.
- Phase 3 boundary: parallel/serial twin-equivalence tests pass; `cargo bench -p proxide-tmalign`
  shows wall-clock improvement on a ≥10-structure batch.
- Phase 4 boundary: `uv run maturin develop` builds the extension; a Python-side smoke test
  (`tests/` root, pytest) calls `proxide.tmalign.tm_align(...)` and matches the Rust-side result on
  the same fixture pair.
- Phase 5 boundary: bathos run recorded via the `using-bathos` skill/MCP tooling, outcome
  classified against the pre-registered pass/marginal/fail conditions.
