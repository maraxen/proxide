# Proxide Technical Debt Tracker

## ✅ Recently Completed (Dec 2025)

### Phase 5: Python/Biotite Migration ✅ COMPLETE

- [x] PQR parsing, mass assignment, and structure processing ported to Rust.
- [x] Biotite dependency minimized.
- [x] `protein.ff14SB.xml` - Downloaded from openmmforcefields
- [x] `protein.ff19SB.xml` - Already present
- [x] `load_force_field()` function working
- [x] Force field tests updated (`tests/assets/test_forcefields.py`)

### OpenMM Export & Physics ✅ COMPLETE

- [x] `to_openmm_system()` supports all standard MD forces.
- [x] CMAP support implemented.
- [x] 1-3/1-4 exclusions implemented.

### Code Cleanup & Protein API Simplification ✅ COMPLETE

- [x] Remove extra dataclasses from `containers.py`
- [x] Simplify `Protein` API
- [x] Move IO-specific types to `proxide.io`
- [x] ProteinTuple deprecation and removal
- [x] AtomicSystem architecture implementation

### Trajectory Format Integration ✅ COMPLETE

- [x] Pure-Rust implementations for XTC (via `molly`), DCD, and TRR.
- [x] `chemfiles` dependency completely removed (no more SIGFPE crashes).
- [x] Simplified features: `xtc`, `mdcath`, `full`. Baseline DCD/TRR.
- [x] Parity validated against MDTraj for all formats.

---

## � High Priority

### Documentation Refresh

- [ ] Update `docs/` folder with current API.
- [ ] Add examples for `add_hydrogens` and MD parameterization.

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
- [ ] Measure formatting speed (Atom37) vs original implementation
- [ ] Measure hydrogen addition time vs hydride

### Ligand Charge Assignment

**Status:** Deferred (parameterize_molecule returns zero charges)

**Background:**

GAFF provides LJ parameters and atom types but NOT partial charges.
`parameterize_molecule()` currently returns zero charges for all atoms.

**Options:**

1. Integrate with AM1-BCC via antechamber (external tool call)
2. Accept user-provided charges as parameter
3. Use Gasteiger charges (less accurate but pure-computation)
4. Add charge derivation from OpenMM ForceField templates

**Tasks:**

- [ ] Add `charges` optional parameter to `parameterize_molecule()`
- [ ] Document charge assignment workflow for ligands
- [ ] Consider AM1-BCC subprocess wrapper

---

### Foldcomp Direct Parsing

**Status:** Not started

**Tasks:**

- [ ] Implement direct parsing of foldcomp, implementing their C++ logic in Rust
- [ ] Direct python binding to our Rust implementation

## Code Health Notes

### Lint Warnings (Acceptable)

The `to_openmm_system()` method has complexity warnings (22 > 10, 21 branches, 83 statements). These are acceptable for a comprehensive export method - splitting would reduce readability.

### Test Coverage

Current test status (December 2025):

- **442 tests passing**
- **61 tests skipped** (expected - optional dependencies)
- Skip patterns working correctly for GAFF/antechamber, HDF5, OpenMM features
