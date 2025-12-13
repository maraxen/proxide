# Agent Development Guidelines

This document provides essential guidelines for AI agents working on the priox codebase.

## Current Status (Dec 2025)

### ✅ Core Implementation Complete

- **Rust Extension:** PDB, mmCIF, PQR parsers
- **Trajectory Formats:**
  - ✅ XTC: Pure-Rust via `molly` crate (no chemfiles crash)
  - ⏳ DCD/TRR: Deferred (blocked by chemfiles SIGFPE, need pure-Rust impl)
  - ✅ HDF5: MDTraj/mdCATH formats working
- **AtomicSystem Architecture:** Base class with Protein/Molecule inheritance
- **Force Field Loading:** ff14SB, ff19SB from assets
- **OpenMM Export:** Full force field export with validation tests
- **Physics Features:** Electrostatics, VdW, RBF, dihedrals
- **Hydrogen Addition:** ✅ add_hydrogens(), relax_hydrogens() validated
- **GAFF Atom Typing:** ✅ assign_gaff_atom_types() working
- **1-3/1-4 Exclusions:** ✅ Full exclusion list generation

### 🔄 Remaining Work

- **Pure-Rust TRR/DCD:** Researching alternatives (groan_rs, custom parsers)
- **Documentation Refresh:** Update docs/ folder with current API
- **CMAPForce:** Not yet implemented (for ff14SB/ff19SB backbone)

---

## Command Execution

### Python Scripts

**Always use `uv run` to execute Python scripts:**

```bash
# ✅ Correct
uv run python script.py

# ❌ Incorrect (may fail or use wrong environment)
python script.py
```

### Timeout Prevention

**Always wrap long-running commands with `timeout` to prevent stalling:**

```bash
# ✅ Correct - 30 second timeout
timeout 30 uv run python script.py

# ✅ Correct - 60 second timeout for longer operations
timeout 60 uv run pytest tests/ -v

# ❌ Incorrect - may stall indefinitely
uv run python long_script.py
```

### Recommended Timeouts

| Operation | Timeout |
|-----------|---------|
| Simple scripts | 30s |
| Unit tests | 60s |
| Integration tests | 120s |
| Rust builds | 180s |
| maturin develop | 180s |

### Multi-line Commands

**Avoid inline multi-line Python commands.** Instead:

1. Write the script to a file
2. Execute it with `uv run python script.py`

```bash
# ✅ Correct
echo 'print("hello")' > test.py && timeout 10 uv run python test.py

# ❌ Problematic - shell quoting issues cause stalling
python -c "
print('hello')
"
```

## Rust Extension Development

### Building

```bash
# Build with maturin (installs into current venv)
timeout 180 cd rust_ext && maturin develop --release

# Run Rust tests only
timeout 60 cd rust_ext && cargo test --lib
```

### Testing

```bash
# Test the Rust extension from Python
timeout 30 uv run python -c "import priox_rs; print(priox_rs)"
```

## Environment

- Use `uv` for Python package management
- Rust toolchain via rustup
- maturin for building Python extensions from Rust

---

## Key Files Reference

### Core Classes

| Class | Location | Description |
|-------|----------|-------------|
| `AtomicSystem` | `src/priox/core/atomic_system.py` | Base class for all atomic systems |
| `Protein` | `src/priox/core/containers.py` | Protein structure (inherits AtomicSystem) |
| `Molecule` | `src/priox/core/atomic_system.py` | Small molecule/ligand |
| `FullForceField` | `src/priox/physics/force_fields/loader.py` | Force field parameter container |

### Key Methods

| Method | Class | Description |
|--------|-------|-------------|
| `to_openmm_topology()` | AtomicSystem | Export to OpenMM Topology |
| `to_openmm_system()` | AtomicSystem | Export to OpenMM System with forces |
| `load_force_field()` | module function | Load force field from assets |
| `from_rust_dict()` | Protein | Create Protein from Rust parser output |

### Test Locations

| Test Area | Location |
|-----------|----------|
| Core tests | `tests/core/` |
| Physics tests | `tests/physics/` |
| Force field tests | `tests/assets/test_forcefields.py` |
| OpenMM validation | `tests/validation/test_openmm_roundtrip.py` |
| Hydrogen parity | `tests/validation/test_hydrogen_parity.py` |

---

## Documentation Files

| File | Purpose |
|------|---------|
| `121211.md` | Master task list with current priorities |
| `ROADMAP.md` | Long-term architecture roadmap |
| `TECHNICAL_DEBT.md` | Known issues and deferred work |
| `VALIDATION_ROADMAP.md` | Parity testing checklist |
| `ATOMIC_SYSTEM_ARCHITECTURE.md` | AtomicSystem design docs |

---

## Quick Reference: Running Tests

```bash
# All core and physics tests
uv run pytest tests/core/ tests/physics/ -v

# Force field tests
uv run pytest tests/assets/test_forcefields.py -v

# OpenMM validation (requires openmm)
uv run pytest tests/validation/test_openmm_roundtrip.py -v

# Full test suite (some may skip/fail due to missing features)
uv run pytest tests/ -x --tb=short
```
