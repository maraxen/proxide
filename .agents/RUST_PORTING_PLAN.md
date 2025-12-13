# Rust Porting Plan

This document outlines the plan to refactor `priox` to fully leverage the `priox_rs` Rust extension, replacing slower legacy Python implementations (Biotite, Hydride, pure-Python force field parsing).

## 1. Objective

Replace performance-critical Python components with their Rust counterparts to improve speed, memory usage, and maintainability.

## 2. Scope

| Component | Current Implementation | Target Implementation | Status |
|-----------|------------------------|-----------------------|--------|
| Structure Parsing | `biotite.structure.io` (Python/Cython) | `priox_rs.parse_structure` (Rust) | Ready to integrate |
| Hydrogen Addition | `hydride` (Python) | `priox_rs` (`geometry::hydrogens`) | Ready to integrate |
| Force Field Parsing | Python XML parsing / OpenMM | `priox_rs.load_forcefield` (Rust) | Ready to integrate |
| MD Parameterization | `priox.md.bridge` (Python) | `priox_rs` (`physics::md_params`) | Ready to integrate |
| Trajectory Reading | `mdtraj` / `biotite` | `priox_rs` (`formats::xtc`, etc.) | Partial (XTC/HDF5 done) |
| Topology/Bonds | `biotite` bond perception | `priox_rs` (`geometry::topology`) | Ready to integrate |

## 3. Implementation Steps

### Phase 1: Structure Parsing & Preprocessing

**Goal:** Replace `priox.io.parsing.biotite` with a new `priox.io.parsing.rust` module.

1. **Create `priox.io.parsing.rust`:**
    * Implement `load_structure_rs(path, ...)` wrapping `priox_rs.parse_structure`.
    * Ensure `OutputSpec` is correctly configured to match requested features (add_hydrogens, etc.).
    * Convert the returned dictionary into `ProcessedStructure` or directly translate to `Protein`/`AtomicSystem` containers.

2. **Update `priox.io.parsing.registry`:**
    * Register the new Rust-based parser.
    * Deprecate or lower priority of `biotite` parser.

3. **Validate Hydrogen Addition:**
    * Rust extension has built-in hydrogen addition with relaxation.
    * Verify it matches or exceeds `hydride` quality (already tested in `test_hydrogen_parity.py`).

### Phase 2: Force Field & MD Parameterization

**Goal:** Move force field loading and system parameterization to Rust.

1. **Update `priox.physics.data.force_fields`:**
    * Use `priox_rs.load_forcefield` to load XML files.
    * This returns a dictionary of atom types, bonds, etc.
    * Update the `FullForceField` Python class (if it exists) to be initialized from this dictionary instead of parsing XML itself.

2. **Integrate Parameterization:**
    * `priox_rs.parse_structure` can optionally `parameterize_md` if provided a force field path.
    * Update `AtomicSystem` construction to utilize this. Pass the force field path to the parser and receive `charges`, `sigmas`, `epsilons`, `atom_types` directly.
    * This eliminates the need for the Python-side `parameterize_system` logic in `priox.md.bridge` (which matches atoms to types via string manipulation). Rust does this faster.

### Phase 3: Trajectory & cleanup

**Goal:** Finalize trajectory support and remove legacy deps.

1. **Trajectory Reader:**
    * Ensure `priox.io.trajectory.read_trajectory` prefers `priox_rs` for supported formats (XTC, TRR, DCD, HDF5).
    * Fallback to MDTraj only if necessary (or remove MDTraj dep if Rust coverage is sufficient).

2. **Remove Dependencies:**
    * Remove `biotite`, `hydride`, `openmm` (if used only for parsing) from `pyproject.toml` dependencies.
    * Clean up `priox.io.parsing.biotite.py`.

## 4. Workflows & Validation

* **Validation:** Use existing parity tests (e.g. `test_hydrogen_parity.py`, `test_load_all_forcefields.py`) to ensure Rust implementation matches or beats legacy.
* **Benchmarks:** Measure parse time for large structures (e.g. 1CRN, larger proteins).

## 5. Next Actions

1. Verify `priox_rs` build and import.
2. Implement `priox.io.parsing.rust` to replace `biotite.py`.
3. Switch default parser in `priox.io.load_protein`.
