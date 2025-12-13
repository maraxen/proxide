# Comprehensive Rust Port Plan - ALL OPERATIONS

**Goal:** Port ALL priox feature extraction operations to Rust for a complete end-to-end pipeline.

**Status:** Phase 7 In Progress - Library renamed to proxide/oxidize

---

## Completed Phases

### ✅ Phase 1: Core Parsing

- [x] PDB/mmCIF parsing via `parse_structure()`
- [x] RawAtomData structure with zero-copy NumPy conversion
- [x] Removed legacy `biotite.py`

### ✅ Phase 2: Force Fields & MD Parameterization

- [x] OpenMM XML force field parsing (`load_forcefield`)
- [x] GAFF atom typing (`gaff.rs`)
- [x] Protein MD parameterization (`md_params.rs`)
- [x] Removed legacy `core.py`

### ✅ Phase 3: Coordinate Formatters

- [x] Atom37Formatter, Atom14Formatter, BackboneFormatter
- [x] Format caching layer

### ✅ Phase 4: Physics Modules

- [x] `gbsa.rs` - mbondi2 radii, OBC2 scaling
- [x] `water.rs` - TIP3P/SPCE/TIP4P water models
- [x] `cmap.rs` - Bicubic spline for CMAP energy
- [x] Removed Python bridge modules: `gbsa.py`, `water.py`, `cmap.py`, `complex.py`, `ligand.py`
- [x] `parameterize_molecule()` - GAFF-based ligand parameterization
- [x] `AtomicSystem.merge_with()` - Protein + ligand complex building
- [x] PyO3 bindings: `assign_mbondi2_radii`, `assign_obc2_scaling_factors`, `get_water_model`, `compute_bicubic_params`, `parameterize_molecule`

---

## Remaining Work

### Phase 5: Trajectory Formats

- [x] XTC (via pure-Rust molly crate)
- [x] HDF5 (mdcath feature)
- [ ] DCD (deferred - needs pure Rust implementation)
- [ ] TRR (deferred - needs pure Rust implementation)

### Phase 5b: Biotite Removal & Cleanup

**Audit Results (Dec 2025):**

- 6 files still import biotite
- MDTraj only needed for DCD fallback
- md/bridge can be mostly removed

**Tasks:**

- [ ] Port PQR parsing to Rust (`formats/pqr.rs`)
- [ ] Delete `physics_utils.py` (use Rust parameterization)
- [ ] Refactor `utils.py` to remove biotite dependency
- [ ] Simplify `mdtraj.py` - remove biotite conversion layer
- [ ] Port `assign_masses()` to Rust
- [ ] Update `structures.py` type hints

---

## Current Rust Modules

```
rust_ext/src/physics/
├── mod.rs
├── constants.rs
├── electrostatics.rs
├── vdw.rs
├── md_params.rs      # Protein MD parameterization
├── gbsa.rs           # NEW: GBSA radii/scaling
├── water.rs          # NEW: Water models
└── cmap.rs           # NEW: CMAP bicubic spline
```

---

## Python Entry Points

```python
# Structure parsing with MD parameterization
from priox.io.parsing.rust import parse_structure, OutputSpec
spec = OutputSpec()
spec.parameterize_md = True
spec.force_field = "protein.ff14SB"
protein = parse_structure("structure.pdb", spec)
```

---

## Bridge Directory (Minimal)

Only 2 files remain in `src/priox/md/bridge/`:

- `types.py` - TypedDict definitions for SystemParams
- `utils.py` - Mass assignment helper (~25 lines)

---

## Next Steps: Phase 6

### Performance & Polish

- [ ] Add Rayon parallelism for large structures
- [ ] Implement CustomFormatter for flexible output
- [ ] Benchmark against reference implementations
- [ ] Documentation refresh

### Advanced Features

- [ ] Pure-Rust DCD reader
- [ ] Pure-Rust TRR reader (groan_rs or custom)
- [x] CMAP energy integration in `to_openmm_system()`
- [ ] Periodic boundary condition support

---

## Phase 7: Finalization (COMPLETED Dec 2025)

### 7.1 Eliminate Python Fallback Logic ✅

Removed all `RUST_AVAILABLE` checks and fallback code:

```python
# REMOVED THIS PATTERN:
try:
    import priox_rs
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    priox_rs = None
```

**Files updated:**

- [x] `src/proxide/io/parsing/rust.py` - Main parser wrapper
- [x] `src/proxide/io/parsing/pqr.py` - PQR parser
- [x] `src/proxide/md/bridge/utils.py` - Mass assignment

**Result:** Rust extension (`oxidize`) is now a hard dependency.

---

### 7.2 Rename Libraries ✅

**New naming scheme (Dec 2025):**

| Component | Old Name | New Name |
|-----------|----------|----------|
| Python Package | `priox` | `proxide` |
| Rust Extension | `priox_rs` | `oxidize` |

**Files updated:**

- [x] `rust_ext/Cargo.toml`: `name = "oxidize"`
- [x] `rust_ext/pyproject.toml`: `name = "oxidize"`
- [x] `rust_ext/src/lib.rs`: `fn oxidize(...)`
- [x] `pyproject.toml`: `name = "proxide"`
- [x] Renamed `src/priox/` → `src/proxide/`
- [x] Updated all internal imports

---

### 7.3 Split `lib.rs` Into Modules

Current `lib.rs` is ~1670 lines. Recommended split:

```
rust_ext/src/
├── lib.rs           # Module exports, pymodule registration only
├── py_parsers.rs    # parse_pdb, parse_structure, parse_mmcif, parse_pqr
├── py_trajectory.rs # parse_xtc, parse_dcd, parse_trr
├── py_forcefield.rs # load_forcefield
├── py_hdf5.rs       # HDF5 parsing functions (feature-gated)
└── py_chemistry.rs  # assign_masses, assign_gaff_atom_types, etc.
```

- [ ] Create module files with function implementations
- [ ] Update `lib.rs` to import and re-export
- [ ] Verify with `maturin develop`
- [ ] Run full test suite

---

### 7.4 Test Suite Cleanup

- [ ] Remove deprecated test files for removed Python modules
- [ ] Remove tests for biotite-based parsing (if any remain)
- [ ] Consolidate Rust parser tests in `tests/io/parsing/`
- [ ] Add coverage for all Rust-exposed functions
- [ ] Remove/update tests that expect Python fallback behavior

---

### 7.5 Code Cleanup

- [ ] Remove deprecated Python files:
  - `src/priox/md/bridge/` (if fully replaced)
  - Legacy parsing modules
- [ ] Update all type hints for Rust-returned objects
- [ ] Add `Protein.format` attribute (Literal type) to track coord format

**Completed:**

- [x] `Protein.format` added with values: `"Atom37"`, `"Atom14"`, `"Full"`, `"BackboneOnly"`

---

### 7.6 Test Coverage Finalization

Target: **95%+ coverage** on Rust-exposed functionality

- [ ] Run `pytest --cov=priox tests/`
- [ ] Identify uncovered code paths
- [ ] Add missing tests
- [ ] Document any intentionally uncovered code
