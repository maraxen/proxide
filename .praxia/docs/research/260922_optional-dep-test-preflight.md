---
title: Optional-Dependency Test Pre-Flight on Python 3.11
description: Assessment of optional-dependency-gated test execution when rdkit, mdtraj, h5py, and tables are installed at uv.lock versions on Python 3.11
date: 2026-09-22
task_id: 260922_autonomous-loop
status: complete
---

## Executive Summary

Pre-flight research for debt #909 phase 1: assessment of which optional-dependency-gated tests pass, fail, or skip when rdkit/mdtraj/h5py/tables are installed on Python 3.11 at uv.lock versions.

**Result:** 76 previously-skipped test instances now execute, 76 of 76 pass (100%); 3 pre-existing failures persist (not new); no failures introduced by optional dependencies.

**Critical finding:** Test failures (92 test instances across 3 files) are NOT regressions — they result from missing ATOMTYPE_GFF2.DEF file in this checkout (CI fetches via `scripts/fetch_amber_assets.py` before pytest). The missing DEF causes `_get_default_rules()` to silently return an empty rule set, so all GAFF2 type assignments fall back to 'c3' (sp3) defaults. This exposes production bug debt #1896 (silent GAFF2 rule fallback). Phase 2 must (a) confirm all 92 tests pass in real CI with fetched DEF, (b) fix debt #1896 so missing DEF raises instead of silently mistyping.

## Environment Setup

### Command Sequence

```bash
cd /home/marielle/projects/proxide/.claude/worktrees/wt-20260910-164718

# Create scratch venv
uv venv -p 3.11 target/preflight/venv

# Export dependencies with all extras
uv export --frozen --no-hashes --extra dev --extra molecules --extra trajectories --no-emit-project > target/preflight/req.txt

# Install base dependencies
OMP_NUM_THREADS=2 timeout 1200 uv pip install --python target/preflight/venv/bin/python -r target/preflight/req.txt

# Build and install proxide in preflight venv (with RUSTC_WRAPPER disabled to avoid sccache sandbox issue)
RUSTC_WRAPPER="" OMP_NUM_THREADS=2 timeout 600 uv pip install --python target/preflight/venv/bin/python -e .
```

### Installed Versions (Preflight Environment)

- **Python**: 3.11.15
- **rdkit**: 2026.03.1 (uv.lock: 2026.3.1) ✓
- **mdtraj**: 1.11.0 (uv.lock: 1.11.0) ✓
- **h5py**: 3.15.1 (uv.lock: 3.15.1) ✓
- **tables**: 3.10.2 (uv.lock: 3.10.2) ✓
- **proxide**: 0.1.0-alpha.16 (built successfully)

## Test Coverage

### Gated Test Files: 23 Files, 287 Test Instances

23 test files containing `importorskip()`, `HAS_H5PY`, `HAS_MDTRAJ`, or `_AVAILABLE` markers were executed in both environments.

Total test instances:
- Baseline: 260 (110 PASS, 92 FAIL, 58 SKIP)
- Preflight: 287 (186 PASS, 92 FAIL, 9 SKIP)
- Net change: +27 test instances executing (previously skipped)

## Results Table

| File | Baseline | Preflight | Status | Notes |
|------|----------|-----------|--------|-------|
| test_hdf5_integration.py | P:2 F:0 S:12 | P:13 F:0 S:1 | No change | 11 h5py-gated tests now execute |
| test_xtc_reader_parity.py | P:0 F:0 S:20 | P:20 F:0 S:0 | SKIP → PASS | All 20 mdtraj-gated tests now pass |
| test_partial_charges.py | P:20 F:0 S:0 | P:20 F:0 S:0 | No change | Already ran, all pass |
| test_trajectory_parity.py | P:1 F:0 S:7 | P:7 F:0 S:1 | No change | 6 mdtraj-gated tests now execute |
| test_xtc_distogram_parity.py | P:2 F:0 S:7 | P:9 F:0 S:0 | No change | 7 h5py-gated tests now execute |
| test_rust_parser.py | P:5 F:0 S:0 | P:5 F:0 S:0 | No change | Already ran, all pass |
| test_rust_integration.py | P:18 F:0 S:0 | P:18 F:0 S:0 | No change | Already ran, all pass |
| test_reference_frame.py | P:0 F:0 S:3 | P:0 F:0 S:3 | No change | 3 partial-charges-gated tests remain skipped (require rdkit feature) |
| test_physics_parity.py | P:4 F:0 S:2 | P:5 F:0 S:1 | No change | 1 mdtraj-gated test now executes, passes |
| test_openmm_roundtrip.py | P:6 F:0 S:0 | P:6 F:0 S:0 | No change | Already ran, all pass |
| test_energy_relaxation.py | P:3 F:0 S:0 | P:3 F:0 S:0 | No change | Already ran, all pass |
| test_atomic_system_openmm_export.py | P:1 F:0 S:0 | P:1 F:0 S:0 | No change | Already ran, all pass |
| test_molecule.py | P:6 F:1 S:1 | P:6 F:1 S:1 | No change | **1 failing** — GAFF2 aromaticity fallback (env-only) |
| test_dispatch.py | P:0 F:0 S:1 | P:23 F:0 S:0 | SKIP → PASS | All 23 mdtraj-dispatch tests now pass |
| test_md_parameterization.py | P:5 F:0 S:2 | P:5 F:0 S:2 | No change | Already ran, all pass |
| test_gb_provenance_bindings.py | P:7 F:0 S:0 | P:7 F:0 S:0 | No change | Already ran, all pass |
| test_gaff2_parity_invariants.py | P:2 F:2 S:0 | P:2 F:2 S:0 | No change | **2 failing** — GAFF2 empty rules (env-only, DEF not fetched) |
| test_gaff2_golden.py | P:17 F:89 S:0 | P:17 F:89 S:0 | No change | **89 failing** — GAFF2 empty rules (env-only, DEF not fetched) |
| test_gaff2.py | P:4 F:0 S:0 | P:4 F:0 S:0 | No change | Already ran, all pass |
| test_alphabet_conformance.py | P:7 F:0 S:0 | P:7 F:0 S:0 | No change | Already ran, all pass |
| test_mdcath_extended.py | P:0 F:0 S:1 | P:4 F:0 S:0 | SKIP → PASS | All 4 mdtraj-streaming tests now pass |
| test_mdcath.py | P:0 F:0 S:1 | P:2 F:0 S:0 | SKIP → PASS | All 2 mdtraj-streaming tests now pass |
| test_mdtraj.py | P:0 F:0 S:1 | P:2 F:0 S:0 | SKIP → PASS | All 2 mdtraj parsing tests now pass |

## Detailed Findings

### Newly Executing Tests (Previously Skipped)

5 test files went from all-skipped to all-passing:

1. **test_xtc_reader_parity.py** (20 tests)
   - Status: SKIP (baseline) → PASS (preflight)
   - Reason: mdtraj availability check; all tests pass with mdtraj 1.11.0
   - Import: `pytest.importorskip("mdtraj")`

2. **test_dispatch.py** (23 tests)
   - Status: SKIP (baseline) → PASS (preflight)
   - Reason: h5py/mdtraj availability via module-level `pytest.importorskip()` calls in `tests/io/parsing/test_dispatch.py` itself (lines 11-12); h5py fires first. `tests/io/parsing/conftest.py` uses a try/except ImportError `HAS_H5PY` flag, not `importorskip`.
   - All dispatch tests execute and pass

3. **test_mdcath_extended.py** (4 tests)
   - Status: SKIP (baseline) → PASS (preflight)
   - Reason: mdtraj availability check
   - All mdtraj-based streaming tests pass

4. **test_mdcath.py** (2 tests)
   - Status: SKIP (baseline) → PASS (preflight)
   - Reason: mdtraj availability check
   - All streaming tests pass

5. **test_mdtraj.py** (2 tests)
   - Status: SKIP (baseline) → PASS (preflight)
   - Reason: module-level `pytest.importorskip("mdtraj")`
   - All mdtraj parsing tests pass

### Partially Executing Test Files

Several files have mixed skip/pass status; partial execution unlocks more tests:

- **test_hdf5_integration.py**: 12 skipped → 1 skipped (11 newly executing)
  - h5py-gated fixtures and test methods now execute
  - All new tests pass
  
- **test_trajectory_parity.py**: 7 skipped → 1 skipped (6 newly executing)
  - mdtraj-dependent tests now execute
  - All new tests pass
  
- **test_xtc_distogram_parity.py**: 7 skipped → 0 skipped (7 newly executing)
  - All h5py/mdtraj-dependent tests execute and pass

- **test_physics_parity.py**: 2 skipped → 1 skipped (1 newly executing)
  - mdtraj-dependent test now executes and passes

### Pre-Existing Failures (Not Introduced by Optional Deps)

**3 test files with 92 failing test instances** — these failures exist in both baseline and preflight, NOT caused by optional dependencies. **Critical: All GAFF2 failures are environment-only in this checkout.**

#### GAFF2 Failures: Root Cause

ATOMTYPE_GFF2.DEF is not present in this checkout (CI fetches it via `scripts/fetch_amber_assets.py` before pytest). When the file is missing, `src/proxide/chem/gaff2.py:1234-1235` silently returns an empty rule set:

```python
if rules_path.exists():
    _default_rules, _default_wildatom = parse_gaff2_rules(rules_path)
else:
    _default_rules = []  # ← SILENT FALLBACK
    _default_wildatom = {}
```

With no rules loaded, `assign_gaff2_atom_types()` has no patterns to match, so all atoms fall back to 'c3' (sp3 carbon) defaults. This manifests as:
- test_gaff2_golden.py C=C: expected ['c2', 'c2'], got ['c3', 'c3'] (tests/test_gaff2_golden.py:268)
- test_gaff2_parity_invariants.py formamide: expected carbonyl 'c', got 'c3' (line 134)
- test_molecule.py benzene: expected aromatic 'ca', got 'c3' (line 173)

**Phase 2 must verify:** GAFF2 golden/invariants/molecule tests pass once CI fetches the DEF file AND installs rdkit (debt #909 phase 2d) — the DEF alone is not sufficient, since these tests import rdkit and CI's `tests` job installs only `.[dev]`.

#### Environment-Only Classification

1. **test_molecule.py** (1 failing: `test_to_rdkit_perceives_aromaticity`)
   - Line 173: `assert types == ["ca"] * 6 + ["ha"] * 6`
   - Actual: `['c3', 'c3', 'c3', 'c3', 'c3', 'c3', 'ha', 'ha', 'ha', 'ha', 'ha', 'ha']`
   - Cause: GAFF2 rules empty (DEF not fetched locally), benzene carbons get 'c3' default
   - Classification: **Env-only locally (DEF not fetched); NOT exercised in CI even after debt #1896's fix packages/loads the DEF — these tests import rdkit, and CI's `tests` job installs only `.[dev]` (rdkit is gated behind the `molecules`/`espaloma` extras); unexercised until debt #909 phase 2d installs rdkit there**
   - Related: Production bug debt #1896 (silent empty GAFF2 rules)

2. **test_gaff2_parity_invariants.py** (2 failing)
   - `test_f8_bond_count_disambiguation_no_regression_on_h_ew_benchmark_molecules` (line 134)
     - Failure: formamide NC=O: expected carbonyl carbon to be 'c', got 'c3'
     - Cause: GAFF2 rules empty (DEF not fetched)
   - `test_h_type_by_heavy_amide_n_h_types_as_hn` (line 166)
     - Failure: formamide NC=O: amide N-H resolved to 'ha', expected 'hn'
     - Cause: GAFF2 rules empty → no amide-H-specific rule matches → falls back to generic H default 'ha'
   - Classification: **Env-only locally (DEF not fetched); NOT exercised in CI even after debt #1896's fix packages/loads the DEF — these tests import rdkit, and CI's `tests` job installs only `.[dev]` (rdkit is gated behind the `molecules`/`espaloma` extras); unexercised until debt #909 phase 2d installs rdkit there**
   - Related: Production bug debt #1896 (silent empty GAFF2 rules)

3. **test_gaff2_golden.py** (89 failing out of 106 tests)
   - Failure pattern: Tests with unsaturated carbons, aromatics, heteroaromatics, carbonyls all fail
   - Examples: C=C expects ['c2','c2'] got ['c3','c3'], c1ccccc1 expects ['ca']*6 got ['c3']*6
   - Root cause: GAFF2 rules empty (DEF not fetched locally)
   - Classification: **Env-only locally (DEF not fetched); NOT exercised in CI even after debt #1896's fix packages/loads the DEF — these tests import rdkit, and CI's `tests` job installs only `.[dev]` (rdkit is gated behind the `molecules`/`espaloma` extras); unexercised until debt #909 phase 2d installs rdkit there**
   - Related: Production bug debt #1896 (silent empty GAFF2 rules)

**Did NOT verify these pass with DEF present** — fetching ATOMTYPE_GFF2.DEF is disallowed by the task constraints. Phase 2 must confirm in real CI.

### Remaining Skipped Tests (Unfixed)

1. **test_reference_frame.py** (3 tests remain skipped)
   - Skip reason: Requires rdkit feature flag or additional rdkit setup
   - These are gated by partial-charges-specific flags beyond `HAS_RDKIT`

## Phase 2 Implications

### Tests Now Available for Execution

- **76 previously-skipped test instances** now execute with optional deps installed
- **76 of 76 newly-running tests pass** (100% success rate on formerly-skipped tests)
- No new failures introduced by optional dependencies
- Tests newly executable: test_xtc_reader_parity (20 tests), test_dispatch (23 tests), test_mdcath_extended (4 tests), test_mdcath (2 tests), test_mdtraj (2 tests), plus 25 more in partially-skipped files (test_hdf5_integration 11, test_trajectory_parity 6, test_xtc_distogram_parity 7, test_physics_parity 1)

### Existing Defects: Environment-Only vs. Production

Pre-existing failures (92 test instances) break down by classification:

| Issue | Count | Classification | CI Impact | Category |
|-------|-------|-----------------|-----------|----------|
| GAFF2 golden tests | 89 | Env-only (DEF not fetched locally) | Still NOT exercised in CI — needs rdkit, not installed by `.[dev]` | Type inference |
| GAFF2 amide typing | 2 | Env-only (DEF not fetched locally) | Still NOT exercised in CI — needs rdkit, not installed by `.[dev]` | Type inference |
| rdkit aromaticity | 1 | Env-only (DEF not fetched locally) | Still NOT exercised in CI — needs rdkit, not installed by `.[dev]` | GAFF2 fallback |
| Remaining skip gates | 3 | Unfixed, test data missing | Persists in CI | Test fixtures |

**CORRECTED (260922, debt #1896 fix / spec-adversarial FATAL #1): all 92 failures root to the missing ATOMTYPE_GFF2.DEF file locally, but fetching the DEF alone does NOT make these tests run in CI. These are rdkit-requiring GAFF2 typing tests, and CI's `tests` job installs only `.[dev]` — rdkit lives behind the `molecules`/`espaloma` extras, neither installed there. Debt #1896's fix makes CI fetch and package the DEF (so a post-install `load_gaff2_rules()` smoke check can run there), but the GAFF2 *typing* tests below remain unexercised in CI until debt #909 phase 2d installs rdkit in the `tests` job.**

### Action Items for Phase 2

**Core:** Confirm GAFF2 typing-test failures are environment-only, and separately confirm they will actually run in CI once rdkit is installed (debt #909 phase 2d) — fetching the DEF (debt #1896) is necessary but not sufficient for that.

1. **Phase 2a — Verify in real CI**: Run tests in CI with `scripts/fetch_amber_assets.py` executed before pytest AND rdkit installed (debt #909 phase 2d)
   - Confirms: test_gaff2_golden.py (89 tests) pass with DEF fetched + rdkit installed
   - Confirms: test_gaff2_parity_invariants.py (2 tests) pass with DEF fetched + rdkit installed
   - Confirms: test_molecule.py benzene test passes with DEF fetched + rdkit installed
   - Until then: debt #1896 lands a CI-covered DEF-loading check (`tests/chem/test_gaff2_def_loading.py` + a post-install `load_gaff2_rules()` smoke step) that does NOT need rdkit, but does not by itself exercise the typing tests above
   - Timeline: 1 CI run after debt #909 phase 2d

2. **Phase 2b — Fix debt #1896** (production bug — silent GAFF2 rule fallback)
   - **Problem:** When ATOMTYPE_GFF2.DEF is missing, `_get_default_rules()` silently returns `[]` instead of raising
   - **Solution:** Modify `src/proxide/chem/gaff2.py:1231-1235` to raise `FileNotFoundError` or `RuntimeError` when DEF is missing
   - **Benefit:** Failures will be loud and immediate, not silent mistyping buried in parameterisation
   - Acceptance: Missing DEF raises, test fails at import-time rather than on first type assignment

3. **Phase 2c — Module-level guard + allowlist**
   - Gate optional-dep tests behind a module-keyed guard that counts and allows specific exceptions
   - Apply to 76 newly-running tests (they all pass)
   - Apply to 3 GAFF2-dependent tests (with explicit allowlist noting env-only cause)
   - Apply to test_reference_frame.py's 3 skips

4. **Phase 2d — CI integration**
   - Enable optional-dep test runs: `uv pip install .[dev,molecules,trajectories]` (ci.yml:60 currently installs only `.[dev]`)
   - Optional deps remain optional for baseline (no new CI job required), but enable the 76 tests when installed

## Observations

- **Build environment**: sccache sandbox issue forced RUSTC_WRAPPER="" for maturin build; no functional impact
- **Python 3.11 compatibility**: All optional deps (rdkit 2026.3.1, mdtraj 1.11.0, h5py 3.15.1, tables 3.10.2) install cleanly on Python 3.11.15
- **Test infrastructure**: No pytest, conftest, or fixture issues encountered when optional deps present
- **Gating quality**: Skip markers work correctly; all expected tests gate properly
- **Newly-passing quality**: All 76 previously-skipped tests pass without modification, indicating that gating logic was the only blocker

## Summary Statistics

### Test Instance Counts (not file counts)

| Metric | Baseline | Preflight | Change |
|--------|----------|-----------|--------|
| **Total test instances** | 260 | 287 | +27 (+10.4%) |
| **PASS instances** | 110 | 186 | +76 |
| **FAIL instances** | 92 | 92 | ±0 |
| **SKIP instances** | 58 | 9 | -49 |
| Test files (file count) | 23 | 23 | ±0 |
| Files with passes and zero failures (file count) | 14 | 19 | +5 |
| Files with any fail (file count) | 3 | 3 | ±0 |

### Newly Passing Test Instances (76 total)

- test_xtc_reader_parity: 20 tests (was all-skip, now all-pass)
- test_dispatch: 23 tests (was all-skip, now all-pass)
- test_mdcath_extended: 4 tests (was all-skip, now all-pass)
- test_mdcath: 2 tests (was all-skip, now all-pass)
- test_mdtraj: 2 tests (was all-skip, now all-pass)
- test_hdf5_integration: 11 partial (was 12 skip, now 11 pass)
- test_trajectory_parity: 6 partial (was 7 skip, now 6 pass)
- test_xtc_distogram_parity: 7 partial (was 7 skip, now 7 pass)
- test_physics_parity: 1 partial (was 2 skip, now 1 pass)

**76 of 76 newly-executing tests pass (100% success rate).**

### Failing Test Instances (92 total, all environment-only)

- test_gaff2_golden: 89 failures (env-only: DEF not fetched)
- test_gaff2_parity_invariants: 2 failures (env-only: DEF not fetched)
- test_molecule: 1 failure (env-only: GAFF2 fallback due to missing DEF)

**All 92 failures are environment-only to this checkout in the sense that they stem from the missing DEF, not a code defect.** CORRECTED (260922): fetching ATOMTYPE_GFF2.DEF before pytest (debt #1896) does NOT make these 92 GAFF2-typing tests pass in CI on its own — they import rdkit, which CI's `tests` job does not install (only `.[dev]`, not the `molecules`/`espaloma` extras). They remain unexercised in CI until debt #909 phase 2d installs rdkit there. What debt #1896's fix does add to CI coverage is a DEF-loading check (no rdkit needed): the DEF is now fetched before install so the wheel packages it, and a post-install step asserts `load_gaff2_rules()` returns a non-empty ruleset.
