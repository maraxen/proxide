# Technical Debt Tracker

**Last Updated:** 2025-12-12

This document tracks known technical debt, incomplete features, and deferred work items.

---

## ✅ Recently Completed (Dec 2025)

### OpenMM Export ✅ COMPLETE

**Status:** Fully implemented and validated

- [x] `AtomicSystem.to_openmm_topology()` - Converts to OpenMM Topology
- [x] `AtomicSystem.to_openmm_system()` - Full force field export:
  - NonbondedForce (charges, LJ with switching function)
  - HarmonicBondForce (bonds with length/k)
  - HarmonicAngleForce (angles with theta/k)
  - PeriodicTorsionForce (proper dihedrals)
  - PeriodicTorsionForce (improper dihedrals)
  - 1-2 exclusions for bonded atoms
  - Proper unit conversions (Å→nm, kcal/mol→kJ/mol)
- [x] Round-trip validation tests (`tests/validation/test_openmm_roundtrip.py`)
- [x] `dihedral_params` and `improper_params` added to AtomicSystem

### Force Field Assets ✅ COMPLETE

- [x] `protein.ff14SB.xml` - Downloaded from openmmforcefields
- [x] `protein.ff19SB.xml` - Already present
- [x] `load_force_field()` function working
- [x] Force field tests updated (`tests/assets/test_forcefields.py`)

### Code Cleanup & Protein API Simplification ✅ COMPLETE

- [x] Remove extra dataclasses from `containers.py`
- [x] Simplify `Protein` API
- [x] Move IO-specific types to `priox.io`
- [x] ProteinTuple deprecation and removal
- [x] AtomicSystem architecture implementation

---

## 🔴 High Priority (Blocking Validation)

### Hydrogen Addition Geometry ✅ COMPLETE

**Status:** Fully validated (2024-12-12)

**Current state:**

- ✅ Hydrogen templates for amino acids
- ✅ `add_hydrogens()` function works
- ✅ Energy relaxation implemented
- ✅ Bond length geometry validated
- ✅ All `test_hydrogen_parity.py` tests passing

### Trajectory Format Integration Tests

**Status:** XTC working, DCD/TRR need pure-Rust alternatives

**Files:**

- `formats/dcd.rs` - implemented (chemfiles-based, crashes)
- `formats/trr.rs` - implemented (chemfiles-based, crashes)
- `formats/xtc.rs` - ✅ working (pure-Rust via `molly` crate)

**Current State:**

- ✅ XTC: Fixed using pure-Rust `molly` crate (`xtc-pure` feature)
- ❌ DCD: Blocked by chemfiles SIGFPE crash (no pure-Rust alternative found)
- ❌ TRR: Blocked by chemfiles SIGFPE crash (`groan_rs` has TRR support but complex API)

**Future Options for DCD/TRR:**

1. Use `groan_rs` crate for TRR (pure Rust, complex API)
2. Implement custom DCD parser (DCD is a simple binary format)
3. Wait for chemfiles fix upstream

---

## 🟡 Medium Priority (Not Blocking)

### GAFF Atom Typing ✅ COMPLETE

**Status:** Fully implemented (2024-12-12)

**Implementation:**

- ✅ `assign_gaff_atom_types()` function in `priox_rs`
- ✅ GAFF parameter loading from `gaff.dat`
- ✅ Rust-native atom type assignment from topology
- ✅ Tests in `tests/assets/test_gaff_parity.py`

### Documentation Refresh

**Status:** Pending

**Tasks:**

- [ ] Update `docs/` folder with current API
- [ ] Add usage examples for OpenMM export
- [ ] Review/update docstring citations
- [ ] Add examples for `add_hydrogens`, `relax_hydrogens`, and MD parameterization

### Test Suite Cleanup

**Status:** Some outdated tests found

**Issues Found:**

- `tests/io/parsing/test_dispatch.py` - Tests for `estat_backbone_mask` attribute (not on Protein)
- `tests/io/parsing/test_foldcomp_extended.py` - Tests for `source` attribute (not on Protein)

**Tasks:**

- [ ] Review and update/remove outdated tests
- [ ] Ensure all tests pass or are properly skipped

---

## ⬜ Deferred (Future Work)

### Performance Benchmarking

**Status:** Not started

**Tasks:**

- [ ] Create `benchmarks/rust_vs_python.py`
- [ ] Measure parsing speed (PDB/mmCIF) vs Biotite
- [ ] Measure formatting speed (Atom37) vs original priox
- [ ] Measure hydrogen addition time vs hydride

### 1-3 and 1-4 Exclusions ✅ COMPLETE

**Status:** Fully implemented in `rust_ext/src/forcefield/exclusions.rs`

- ✅ 1-2 exclusions from bonds
- ✅ 1-3 exclusions from angles
- ✅ 1-4 pairs from dihedrals
- ✅ `coulomb14scale` and `lj14scale` read from force field XML

### CMAPForce Support

**Status:** Not implemented

**Note:** CMAP is important for accurate backbone sampling in protein force fields (ff14SB, ff19SB)

### Pure-Rust TRR and DCD Trajectory Readers

**Status:** Deferred (XTC working, TRR/DCD blocked by chemfiles crash)

**Background:**

The `chemfiles` library crashes with SIGFPE on this environment when opening trajectory files.
XTC was fixed by using the pure-Rust `molly` crate.

**Options for TRR:**

1. Use `groan_rs` crate - has TrrReader but requires System struct (complex API)
2. Implement minimal TRR parser using XDR format specification

**Options for DCD:**

1. No pure-Rust DCD crate found
2. Implement custom parser - DCD is a simple binary format (CHARMM/NAMD origin)

**Tasks:**

- [ ] Research `groan_rs` TrrReader low-level API
- [ ] Evaluate implementing minimal DCD parser
- [ ] Add `trr-pure` and `dcd-pure` feature flags when ready

---

## Code Health Notes

### Lint Warnings (Acceptable)

The `to_openmm_system()` method has complexity warnings (22 > 10, 21 branches, 83 statements). These are acceptable for a comprehensive export method - splitting would reduce readability.

### Test Coverage

Current test status:

- Core tests: ~140 passing
- Validation tests: Some require OpenMM or trajectory features
- Skip patterns working correctly for optional dependencies
