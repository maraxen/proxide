# Comprehensive Deployment Plan: Proxide Monorepo

**Date:** 2026-04-16
**Status:** Revision 2 — Addressed Oracle Feedback

---

## Executive Summary

This plan consolidates the `deployment_plan.md` and `phase2_plan.md` into a single, codebase-grounded roadmap. It prioritizes critical blockers (broken imports, missing CLI, security vulnerabilities, missing GIL releases) before moving to performance work (rayon, cell-list integration, zero-copy wrappers) and finally deployment hardening (CI/CD, dual ecosystem publish). It is organized into **5 sequential phases**, each independently testable.

---

## 1. Current State Assessment (Evidence-Grounded)

### 1.1 What is Working
| Component | Status | Evidence |
|-----------|--------|----------|
| Rust core (proxide_rs) | Compiles, rlib | Cargo.toml, src/ layout |
| PyO3 bindings (proxide_py) | Compiled .abi3.so exists | _oxidize.abi3.so in src/proxide/ |
| PDB/mmCIF/PQR/FoldComp parsing | Implemented | formats/, py_parsers.rs |
| Trajectory parsing (XTC/DCD/TRR/MDC) | Implemented | py_trajectory.rs |
| HDF5/mdcath support | Optional feature | py_hdf5.rs, mdcath feature flag |
| MD parameterization | Implemented | physics/md_params.rs |
| Hydrogen addition | Implemented | geometry/hydrogens.rs |
| Cell list spatial hash | Exists | geometry/cell_list.rs |
| Multi-platform CI wheel builds | Linux + macOS | ci.yml |
| Trusted PyPI publishing | OIDC configured | ci.yml release job |
| Test suite | ~6,389 lines, 28+ files | tests/ |

### 1.2 Critical Blockers (Must Fix Before Any Release)

| # | Blocker | Location | Impact |
|---|---------|----------|--------|
| B1 | proxide.cli:app entry point references non-existent src/proxide/cli.py | pyproject.toml:32 | proxide CLI command fails completely |
| B2 | __proxider (double underscore) vs _proxider (single underscore) import inconsistency across 7 Python files | Multiple | Runtime ImportError in many code paths |
| B3 | Path traversal in fetch_rcsb, fetch_afdb, fetch_md_cath — no ID sanitization | io/fetching.rs:16-123 | Security vulnerability |
| B4 | Panic in fetch_md_cath for IDs <3 chars (&id[1..3]) | io/fetching.rs:86 | Process abort on malformed input |
| B5 | No GIL release around Rust compute sections; PyO3 locks thread | py_parsers.rs, py_forcefield.rs | Python threading broken |

### 1.3 Significant Warnings

| # | Warning | Location | Impact |
|---|---------|----------|--------|
| W1 | trajectory_parity.yml references cd oxidize (stale directory) | .github/workflows/ | CI workflow fails |
| W2 | rayon declared in Cargo.toml but zero par_iter calls in source | All .rs files | HPC claims currently false |
| W3 | PyAtomicSystem wraps AtomicSystem via copy, not zero-copy | bindings/atomic_system.rs | Performance claims violated |
| W4 | hdf5-metno is default in CI — requires libhdf5 dev headers | Cargo.toml, ci.yml | Fragile CI on clean runners |
| W5 | No Windows wheel builds in CI | ci.yml | Windows users unsupported |
| W6 | cargo test never runs in CI | ci.yml | Rust unit tests untested |
| W7 | _oxidize.abi3.so is a stale non-workspace build artifact | src/proxide/ | Wrong module loaded at runtime |
| W8 | proxide_py lib name _proxide_rs mismatches pyproject.toml _proxider | Cross-file | Module load fails on fresh build |

### 1.4 Suggestions / Future Work
- Crates.io publish workflow for proxide_rs (source publish)
- Cell list integration into neighbor search hot path (replaces O(N^2))
- no_std refactor for math/geometry sub-modules
- Windows CPU wheel builds
- pytest-benchmark integration
- SLURM/Apptainer packaging guide for HPC users

---

## 2. Architecture: Target State

```
proxide (monorepo)
├── crates/
│   ├── proxide_rs/          [rlib] Pure Rust core
│   │   └── src/
│   │       ├── io/fetching.rs  <- SANITIZE IDs, NO PYO3
│   │       └── geometry/       <- Integrate cell_list + rayon par_iter
│   └── proxide_py/          [cdylib] Thin PyO3 layer
│       ├── Cargo.toml          <- lib.name = "_proxider" (fix W8)
│       └── src/
│           ├── lib.rs          
│           ├── py_fetch.rs     <- PyO3 wrappers for fetching relocated here
│           └── bindings/atomic_system.rs <- Arc<AtomicSystem>
├── src/proxide/
│   ├── cli.py               <- CREATE (fixes B1)
│   ├── io/fetching.py       <- fix __proxider -> _proxider (B2)
│   └── chem/properties.py   <- fix __proxider (B2)
├── pyproject.toml           <- module-name = "proxide._proxider"
└── .github/workflows/
    ├── ci.yml               <- add cargo test, Windows, smoke post-build
    └── trajectory_parity.yml <- fix stale cd oxidize (W1)
```

### 2.1 Module Name: Standardize on proxide._proxider

Recommendation: Use single underscore _proxider throughout.
- Fix crates/proxide_py/Cargo.toml: name = "_proxider" (was "_proxide_rs")
- Fix all Python imports: use _proxider not __proxider
- This matches pyproject.toml module-name = "proxide._proxider"

---

## 3. Execution Phases

### Phase 0: Critical Bug Fixes (Week 1)

#### 0.1 Fix Module Name Inconsistency (B2, W8)
Files to update:
- crates/proxide_py/Cargo.toml: name = "_proxider"
- src/proxide/io/fetching.py: __proxider -> _proxider
- src/proxide/chem/properties.py: __proxider -> _proxider
- src/proxide/ops/transforms.py: __proxider -> _proxider
- src/proxide/io/parsing/molecule.py: __proxider -> _proxider
- src/proxide/io/parsing/backend.py: __proxider -> _proxider
- src/proxide/core/projector.py: __proxider -> _proxider
Verification: python -c "import proxide; print(proxide._proxider.__doc__)"

#### 0.2 Fix Path Traversal + Panic in Fetch (B3, B4)
Add to crates/proxide_rs/src/io/fetching.rs:

```rust
pub fn validate_id(id: &str) -> Result<(), String> {
    if id.is_empty() || id.len() > 20 {
        return Err(format!("Invalid ID length: {}", id.len()));
    }
    if !id.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-') {
        return Err(format!("Invalid ID characters: {}", id));
    }
    Ok(())
}
```

- Call `validate_id()` in all three fetch core functions.
- Replace `&md_cath_id[1..3]` with `md_cath_id.get(1..3).ok_or("ID too short")?`
- Wrap path join: use `.file_name()` check to prevent path component injection.

#### 0.3 Create CLI Module (B1)
Create src/proxide/cli.py with commands: fetch, info, convert, validate, bench, parameterize.
Use typer + rich + tqdm (all already in pyproject.toml dependencies).

#### 0.4 Fix trajectory_parity.yml (W1)
Replace `cd oxidize` with correct maturin develop call at repo root.
Fix PATH setup to include $HOME/.cargo/bin.

#### 0.5 Partial GIL Release (File I/O only) (B5)
In `py_parsers.rs`, `parse_structure` currently does Python object manipulation midway through.
We cannot simply wrap the entire outer function with `allow_threads`.
- Extracted Rust-only parse/compute step into `py.allow_threads(|| { ... parse I/O ... })`.
- Re-acquire the GIL correctly when calling `PyArray1::from_slice_bound!` and constructing Python Dicts.
  
Apply similar safe boundaries to `parse_pdb`, `parse_mmcif`, `parse_xtc`, `parse_dcd`, `parse_trr`, and fetch wrappers. Complete GIL restructuring is deferred to Phase 3.

#### 0.6 Remove Stale `_oxidize.abi3.so` (W7)
Audit and remove `_oxidize.abi3.so` from `src/proxide/`.
Add `*.so` and `*.pyd` to `.gitignore` inside the source directory.
Verification: Ensure `ls src/proxide/*.so` shows only `_proxider.abi3.so` post-build. 

#### 0.7 Phase 0 Verification Gate
- Add `tests/io/test_fetch_security.py` directly testing for exceptions on invalid IDs like `../../etc/passwd`.
- Run pytest `smoke` suite.

---

### Phase 1: Architecture Hardening (Week 1-2)

#### 1.1 PyAtomicSystem: Arc-based wrapper (W3)
Intermediate step before full zero-copy:
```rust
#[pyclass]
pub struct PyAtomicSystem {
    pub inner: Arc<AtomicSystem>,
}
```
**Clarification:** `Arc<AtomicSystem>` eliminates Rust-side cloning when the object is shared across threads. It does NOT eliminate the initial Python→Rust copy at construction, which remains until Phase 3 (ndarray::Array1 storage internments).

#### 1.2 Fix Cargo.toml lib name (W8)
crates/proxide_py/Cargo.toml:
```toml
[lib]
name = "_proxider"
crate-type = ["cdylib"]
```
Verify the `#[pymodule]` function name in `lib.rs` matches exactly.

#### 1.3 HDF5 Build Hardening (W4)
Add to `ci.yml` for Linux jobs using RHEL8 compatible `manylinux: 2_28` to ensure newer HDF5 availability:
```yaml
- name: Build wheels
  uses: PyO3/maturin-action@v1
  with:
    manylinux: 2_28
```

#### 1.4 Add cargo test to CI (W6)
New CI job:
```yaml
rust-tests:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: dtolnay/rust-toolchain@stable
    - run: sudo apt-get install -y libhdf5-dev pkg-config
    - run: cargo test --workspace --features full
```

#### 1.5 Decouple PyO3 from Pure Rust Core
Move fetching PyO3 wrappers `fetch_rcsb`, `fetch_afdb`, `fetch_md_cath` out of `proxide_rs/src/io/fetching.rs` into `crates/proxide_py/src/py_fetch.rs`. 
Keep `validate_id()` and the native Rust fetch logic in `proxide_rs`. 

---

### Phase 2: CLI Implementation (Week 2-3)

#### 2.1 Command Specifications

| Command | Key Options | Backend |
|---------|-------------|---------|
| proxide fetch <id> | --source rcsb/afdb/mdcath --dir . --format mmcif | _proxider.fetch_rcsb/afdb/fetch_md_cath |
| proxide info <path> | | parse_structure, print atom/residue/chain counts |
| proxide convert <in> <out> | --spec spec.json | parse_structure |
| proxide validate <path> | --strict | Parse validation only |
| proxide bench <path> | --n 10 --format table/json | Timer + tqdm |
| proxide parameterize <path> | --ff protein.ff14SB.xml --output params.npz | Full MD param pipeline |

Paths passed from Typer CLI will use `pathlib.Path` definitions for correct cross-platform compatibility.

#### 2.2 HPC Streaming Strategy
- Never hold full trajectory in memory
- proxide convert: use streaming frame-by-frame parsing
- proxide bench: progress bars via tqdm on frame counts

#### 2.3 CLI Safety: Input Validation (Python Layer)
```python
import re
from pathlib import Path
ID_PATTERN = re.compile(r'^[A-Za-z0-9_-]{1,20}$')

def validate_id(id: str) -> str:
    if not ID_PATTERN.match(id):
        raise typer.BadParameter(f"Invalid ID format: {id!r}")
    return id
```

---

### Phase 3: Parallelism and Performance (Week 3-4)

#### 3.1 Rayon Integration in Neighbor Search (W2)
Target: `geometry/neighbors.rs`, `find_k_nearest_neighbors`
Integration plan utilizes `par_iter()` on the existing `CellList` queries. Add `query_neighbors` method explicitly using read-only borrows.

#### 3.2 Thread Pool Control from Python
Register parallel pool configuration in `_proxider` library.

#### 3.3 Full GIL Restructure and True Zero-Copy
Replace `Vec<f32>` arrays in `AtomicSystem` natively with `ndarray::Array1<f32>`. Complete GIL separation so memory isn't bounced between Python context arbitrarily.

---

### Phase 4: CI/CD Modernization (Week 4)

#### 4.1 Windows Wheel Builds (W5)
Disable the `mdcath` HDF5 feature for the Windows matrices by default to prevent dragging down pipelines with VCPKG HDF5 compilation (~20m).
```yaml
args: --release --out dist --find-interpreter --no-default-features --features xtc
```

#### 4.2 Crates.io Publish Workflow
New file: `.github/workflows/publish-crate.yml`
Uses OIDC trusted publishing. Manual publish check before CI rollout: `proxide-core`.

#### 4.3 Smoke Test Post Wheel Build
Verify standard `.abi3.so` install natively inside Python.

---

### Phase 5: Release Readiness (Week 5)

#### 5.1 Version Management
Sync pyproject.toml version with workspace.package.version via maturin dynamic versioning or manual sync script.
Tag format: v0.1.0 triggers PyPI + Crates.io publish.

#### 5.2 Documentation Updates
- README: CLI reference section
- Performance notes: thread control, GIL behavior, memory model
- Security: ID format requirements for fetch commands

#### 5.3 Full Verification Matrix

| Test | Command | Pass Criteria |
|------|---------|---------------|
| Rust unit tests | cargo test --workspace | All pass |
| Python smoke | pytest -m smoke | All pass |
| CLI entry | proxide --help | Shows help text |
| CLI fetch | proxide fetch 1UBQ | Downloads .cif file |
| CLI security | proxide fetch "../../etc" | Validation error, no file access |
| Module import | python -c "from proxide._proxider import parse_pdb" | No error |
| Wheel install | pip install dist/proxide-*.whl && python -c "import proxide" | No error |
| GIL release | Python threading test with concurrent parse_structure calls | No deadlock |

---

## 4. Decision Points for User Review

> [!IMPORTANT]
> **Decision 1 — Windows HDF5 in CI:** Building Windows wheels requires HDF5 via VCPKG (~20min CI time) OR disabling `mdcath` feature on Windows builds. **Recommendation:** Disable `mdcath` on Windows CI initially; document HDF5 requirement for users who need it.

> [!IMPORTANT]
> **Decision 2 — Zero-Copy Timeline:** Phase 1 implements `Arc<AtomicSystem>` to prevent thread copies. True buffer zero-copy avoids Python-to-Rust conversion completely, but necessitates transitioning components to `ndarray::Array1`. Deferred to Phase 3. Are you aligned with this staging?

> [!WARNING]
> **Decision 3 — GIL Release Scope:** `parse_structure` currently interweaves Rust compute logic and `PyArray` object creation. A complete GIL extraction to fully exploit threading is pushed to Phase 3; Phase 0 handles wrapping file-parsing and pure string-handling. Is this acceptable?

> [!NOTE]
> **Decision 4 — `validate` CLI Command Scope:** The `validate` command could range from "does it parse without error" to "full chemical validity check (bond lengths, chirality, residue completeness)." **Recommendation:** Start with parse-only validation; add chemical checks incrementally.

> [!NOTE]
> **Decision 5 — Crates.io Name Conflict:** The name `proxide_rs` should be checked for existing registration on crates.io before publishing. Alternative names: `proxide-core`, `proxider-rs`. **Recommendation:** Check availability and register with the preferred name before any release.

---

## 5. Oracle-Critique Cycle Plan

| Cycle | Focus | Evidence Sources |
|-------|-------|-----------------|
| 1 | Completed | Codebase audit |
| 2 (Post Phase 0) | Security + module fix correctness | Code diffs, pytest -m smoke |
| 3 (Post Phase 2) | CLI utility + HPC usability | CLI end-to-end, memory profile |

---

## 6. Operational Rules of Engagement
- **Git Safety**: No commits without explicit user authorization
- **Validation Gate**: Every phase must pass its verification matrix before proceeding
- **Audit Trail**: All operations logged to `.agent/docs/daily/YYMMDD.md`
- **Transparency**: Explain before acting; brief summary post-action
- **Security First**: Phase 0 security fixes deploy before any other code changes
