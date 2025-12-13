# Phase 7: Rust Port Finalization

## Context

You're continuing the Rust port finalization for the **priox** library - a high-performance protein structure parsing toolkit built on a Rust extension (`priox_rs`).

## First Steps

1. **Review the `.agents` directory** for project context:

   ```
   .agents/
   ├── COMPLETE_RUST_PORT.md     # ← START HERE - Phase 7 details
   ├── TECHNICAL_DEBT.md         # Known issues
   ├── VALIDATION_ROADMAP.md     # Testing status
   └── RUST_PORTING_PLAN.md      # Original port plan
   ```

2. Review `COMPLETE_RUST_PORT.md` Phase 7 specifically - it contains the detailed task list.

## Phase 7 Tasks (Priority Order)

### 7.1 Eliminate Python Fallback Logic

Remove all `RUST_AVAILABLE` checks:

```python
# REMOVE THIS PATTERN everywhere:
try:
    import priox_rs
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
```

Key files:

- `src/priox/io/parsing/rust.py`
- `src/priox/io/parsing/dispatch.py`
- Any other files with this pattern

### 7.2 Rename Library to "proxide"

- Update `rust_ext/Cargo.toml`: `name = "proxide"`
- Update all Python imports from `priox_rs` to `proxide`
- Update `pyproject.toml`

### 7.3 Split `lib.rs` Into Modules

Current: `rust_ext/src/lib.rs` (~1670 lines)

Target structure:

```
rust_ext/src/
├── lib.rs           # Module exports only
├── py_parsers.rs    # parse_pdb, parse_structure, parse_mmcif, parse_pqr
├── py_trajectory.rs # parse_xtc, parse_dcd, parse_trr
├── py_forcefield.rs # load_forcefield
├── py_hdf5.rs       # HDF5 functions (feature-gated)
└── py_chemistry.rs  # assign_masses, assign_gaff_atom_types, etc.
```

### 7.4 Test Suite Cleanup

- Remove deprecated test files
- Consolidate Rust parser tests

### 7.5 Code Cleanup

- Add `Protein.format` attribute (Literal["Atom37", "Atom14", "Full", "BackboneOnly"])
- Remove deprecated Python files

### 7.6 Test Coverage

- Target 95%+ coverage on Rust-exposed functionality

## Build & Test Commands

```bash
# Build Rust extension
cd rust_ext && /home/marielle/workspace/priox/.venv/bin/maturin develop

# Run tests
cd /home/marielle/workspace/priox
.venv/bin/pytest tests/io/parsing/test_dispatch.py -v
.venv/bin/pytest tests/ -v  # Full suite
```

## Current State

- ✅ `lib.rs` is compilable and functional
- ✅ All 23 dispatch tests passing
- ⚠️ 20 compiler warnings (unused variables) - minor cleanup needed
