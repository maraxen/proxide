# Comprehensive Rust Port Plan - ALL OPERATIONS

**Goal:** Port ALL priox feature extraction operations to Rust for a complete end-to-end pipeline.

**Status:** Phase 4 Complete (95% Complete)

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
