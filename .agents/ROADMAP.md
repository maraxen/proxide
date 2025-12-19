# Rust Extension Roadmap - ALL-RUST PIPELINE

**Last Updated:** 2025-12-19 (All Phases Complete)  
**Architecture:** Parse → Format → Geometry Ops → AtomicSystem (ALL in Rust)

---

## Executive Summary

**Revised Goal:** All processing in Rust until the final `Protein` is handed to Python.

**Pipeline:**

```
┌──────────────┐
│ Rust Parser  │  PDB/mmCIF → Raw

AtomData (ALL atoms, ALL models by default)
└──────┬───────┘
       ↓
┌──────────────┐
│ Rust Formatter│  RawAtomData + OutputSpec → Formatted data (atom37/atom14/etc)
└──────┬───────┘
       ↓
┌──────────────┐
│ Rust Geometry│  Solvent removal, bond inference, hydrogen addition
└──────┬───────┘
       ↓
┌──────────────┐
│ Protein      │  Final output to Python (zero-copy NumPy arrays)
└──────────────┘
```

**Key Principles:**

- ✅ Parse ALL models by default, filter via OutputSpec
- ✅ All formatting logic in Rust (atom37, atom14, custom)
- ✅ All geometry operations in Rust (solvent, bonds, H-addition)
- ✅ Python receives finished Protein only
- ✅ Zero-copy transfer of NumPy arrays
- ✅ Caching of formatted outputs in Rust

---

## 1. Data Structures

### RawAtomData (Already Implemented ✅)

Variable-length atom list with ALL fields:

```rust
pub struct RawAtomData {
    coords: Vec<f32>,           // Flat (N_atoms * 3)
    atom_names: Vec<String>,
    elements: Vec<String>,
    res_names: Vec<String>,
    res_ids: Vec<i32>,
    chain_ids: Vec<String>,
    b_factors: Vec<f32>,
    occupancy: Vec<f32>,
    // ... all PDB fields
}
```

### ProcessedStructure (NEW - in Rust)

Intermediate structure with computed indices:

```rust
pub struct ProcessedStructure {
    raw_atoms: RawAtomData,
    
    // Computed indices for efficient formatting
    residue_starts: Vec<usize>,      // Index of first atom in each residue
    residue_atom_counts: Vec<usize>, // Atoms per residue
    unique_residues: Vec<ResidueInfo>,
    unique_chains: Vec<String>,
    chain_to_index: HashMap<String, i32>,
    
    // Metadata
    num_residues: usize,
    num_chains: usize,
}
```

### OutputSpec (NEW - in Rust)

User specifies exactly what they want:

```rust
pub struct OutputSpec {
    // Format
    coord_format: CoordFormat,  // Atom37, Atom14, Full, BackboneOnly
    
    // Filtering
    models: Option<Vec<usize>>,      // Which models to include (default: all)
    chains: Option<Vec<String>>,      // Which chains
    remove_hetatm: bool,
    remove_solvent: bool,
    residue_range: Option<(i32, i32)>,
    
    // Processing
    add_hydrogens: bool,
    infer_bonds: bool,
    
    // Optional fields
    include_b_factors: bool,
    include_occupancy: bool,
    include_physics_params: bool,
    
    // Error handling
    error_mode: ErrorMode,  // Warn, Skip, Fail
    
    // Performance
    enable_caching: bool,
}

pub enum CoordFormat {
    Atom37,           // (N_res, 37, 3)
    Atom14,           // (N_res, 14, 3)  
    Full,             // Variable-length, all atoms
    BackboneOnly,     // (N_res, 4, 3) - N, CA, C, O
    Custom(Vec<String>),  // User-specified atom names
}

pub enum ErrorMode {
    Warn,   // Log warnings, continue
    Skip,   // Skip problematic atoms/residues
    Fail,   // Fail entire structure
}
```

### FormattedStructure (NEW - in Rust)

Output after formatting:

```rust
pub struct FormattedStructure {
    coordinates: Vec<f32>,     // Shape depends on format
    aatype: Vec<i8>,
    atom_mask: Vec<f32>,
    residue_index: Vec<i32>,
    chain_index: Vec<i32>,
    
    // Optional fields
    b_factors: Option<Vec<f32>>,
    occupancy: Option<Vec<f32>>,
    charges: Option<Vec<f32>>,
    
    // Geometry data
    bonds: Option<Vec<(usize, usize)>>,
    
    // Metadata
    coord_shape: (usize, usize, usize),  // e.g. (10, 37, 3)
    format: CoordFormat,
}
```

---

## 2. Implementation Phases (REVISED)

### Phase 1: Infrastructure ✅ COMPLETE

### Phase 2: Raw Atom Parser ✅ COMPLETE

- [x] RawAtomData structure
- [x] PDB parser returning all atoms
- [x] All atom-level fields (coords, names, elements, b_factors, etc.)

### Phase 3: Structure Processing ✅ COMPLETE

- [x] Implement `ProcessedStructure::from_raw()`
- [x] Residue grouping and indexing
- [x] Chain mapping utilities
- [x] Multi-model handling (`processing/models.rs`)

### Phase 4: Formatters ✅ COMPLETE

- [x] Implement `OutputSpec` struct
- [x] `Atom37Formatter` - map to (N_res, 37, 3)
- [x] `Atom14Formatter` - reduced representation
- [x] `FullFormatter` - keep all atoms (ragged or padded)
- [x] `BackboneFormatter` - N, CA, C, O only
- [x] Custom formatter (`formatters/custom.rs`)
- [x] Format caching layer

### Phase 5: Geometry Operations ✅ COMPLETE

- [x] Solvent removal (`geometry/solvent.rs`)
- [x] HETATM filtering (via `molecule_type`)
- [x] Bond inference (`geometry/topology.rs`)
- [x] Cell list algorithm (`geometry/cell_list.rs`)
- [x] Hydrogen addition (`geometry/hydrogens.rs`)
- [x] Coordinate transformations (`geometry/transforms.rs`)
- [x] Sequence alignment (`geometry/alignment.rs`)

### Phase 6: Extended Formats ✅ COMPLETE (needs testing)

- [x] PQR parser (`formats/pqr.rs`)
- [x] mmCIF parser (all models by default)
- [x] XTC trajectory parsing (via `chemfiles` crate)
- [x] DCD format (`formats/dcd.rs`) ⚠️ Needs integration tests
- [x] TRR format (`formats/trr.rs`) ⚠️ Needs integration tests
- [x] MDTraj HDF5 (.h5) parsing (feature-gated)
- [x] MDCATH HDF5 (.h5) parsing (feature-gated)
- [x] PyO3 bindings for HDF5 parsers
- [x] Python wrapper functions for HDF5

### Phase 7: Python Interface

- [x] Single function: `parse_structure(path, spec=None) -> ProteinTuple`
- [x] Thin Python wrapper (just calls Rust)
- [x] Automatic NumPy array conversion
- [x] Error handling/logging integration

---

## 3. Design Decisions - ANSWERED

### 1. Ragged Arrays for Full Format

**Best Practice:** Use **padded arrays with masks** for ML/JAX compatibility:

```rust
// Option A (Recommended): Padded with mask
FormattedStructure {
    coordinates: (N_res, max_atoms_per_res, 3),  // Padded to max
    atom_mask: (N_res, max_atoms_per_res),       // 1 = present, 0 = padding
}

// Option B: Return list of arrays (Python-side handling)
// Less efficient for batch processing

// Option C: Flat array with indices
// More complex for downstream use
```

**Decision:** Use **Option A** - padded arrays with masks. This is:

- JAX/ML friendly (fixed shapes)
- Easy to batch
- Minimal overhead (Rust handles padding efficiently)

### 2. Caching Formatted Outputs

**YES** - Implement caching in Rust:

```rust
pub struct StructureCache {
    atom37: Option<FormattedStructure>,
    atom14: Option<FormattedStructure>,
    full: Option<FormattedStructure>,
}
```

Benefits:

- Avoid re-formatting same structure

- Automatic invalidation on spec changes
- Minimal memory overhead (lazy evaluation)

### 3. Multi-Model Handling

**Parse ALL models by default, filter via OutputSpec:**

```rust
pub struct OutputSpec {
    models: Option<Vec<usize>>,  // None = all models, Some([0]) = first only
}
```

Rationale:

- Parsing all models has minimal overhead in Rust
- Simpler code path (no conditionals during parsing)
- Filtering is fast compared to I/O
- Users specify what they want explicitly

### 4. Hydrogens

**Port to Rust eventually:**

- Phase 1: Keep Python hydride as fallback
- Phase 2: Implement basic H-addition in Rust (geometric reconstruction)
- Phase 3: Full hydride port for advanced cases

### 5. Error Handling

**Configurable via ErrorMode:**

```rust
match spec.error_mode {
    ErrorMode::Warn => {
        log::warn!("Missing atom {} in residue {}", atom, res);
        continue;  // Skip atom
    },
    ErrorMode::Skip => {
        // Skip entire residue silently
    },
    ErrorMode::Fail => {
        return Err(format!("Missing required atom"));
    },
}
```

**Defaults:**

- Atom/residue issues → Warn
- Structural issues (no atoms, corrupt file) → Fail

---

## 4. Rust Module Structure

```
rust_ext/
├── src/
│   ├── lib.rs              # PyO3 entry point
│   │
│   ├── structure.rs        # RawAtomData, ProcessedStructure
│   │
│   ├── spec.rs             # OutputSpec, CoordFormat, ErrorMode
│   │
│   ├── formats/
│   │   ├── mod.rs
│   │   ├── pdb.rs          # PDB parser
│   │   ├── pqr.rs          # PQR parser
│   │   ├── mmcif.rs        # mmCIF parser
│   │   └── trajectory.rs   # XTC/DCD/TRR
│   │
│   ├── processing/
│   │   ├── mod.rs
│   │   ├── residue.rs      # Residue grouping
│   │   ├── chain.rs        # Chain mapping
│   │   └── models.rs       # Multi-model handling
│   │
│   ├── formatters/
│   │   ├── mod.rs
│   │   ├── atom37.rs       # Atom37Formatter
│   │   ├── atom14.rs       # Atom14Formatter
│   │   ├── full.rs         # FullFormatter
│   │   └── cache.rs        # Format caching
│   │
│   ├── geometry/
│   │   ├── mod.rs
│   │   ├── solvent.rs      # Solvent removal
│   │   ├── bonds.rs        # Bond inference
│   │   ├── hydrogens.rs    # H-addition
│   │   └── neighbors.rs    # Cell list algorithm
│   │
│   └── output/
│       ├── mod.rs
│       └── protein.rs  # Convert to Protein
```

---

## 5. Python API

**Single entry point:**

```python
from oxidize import parse_structure, OutputSpec, CoordFormat

# Simple case (defaults to atom37, all models, remove solvent)
protein = parse_structure("structure.pdb")

# Custom spec
spec = OutputSpec(
    coord_format=CoordFormat.Atom14,
    models=[0],  # First model only
    remove_solvent=True,
    add_hydrogens=True,
    include_b_factors=True,
    error_mode="warn",
)

protein = parse_structure("structure.pdb", spec)

# Returns Protein directly
print(protein.coordinates.shape)  # (N_res, 14, 3) for atom14
print(protein.atom_mask.shape)     # (N_res, 14)
```

**Backward compatibility wrapper:**

```python
# src/priox/io/parsing/rust_wrapper.py
def load_structure_rust(path, **kwargs):
    """Backward-compatible wrapper."""
    spec = OutputSpec(**kwargs)
    return parse_structure(path, spec)
```

---

## 6. Performance Targets

| Operation | Current (Python) | Target (Rust) | Speedup |
|-----------|------------------|---------------|---------|
| PDB parse | 50ms | 2ms | 25x |
| mmCIF parse | 500ms | 20ms | 25x |
| Atom37 format | 10ms | 0.5ms | 20x |
| Solvent removal | 15ms | 1ms | 15x |
| H-addition | 100ms | 10ms | 10x |
| **End-to-end** | **175ms** | **15ms** | **12x** |

---

### Phase 8: Documentation & Polish

- [ ] Thorough code review and documentation update
- [ ] Validated citations for all algorithms/methods (Google-style: Author, Year)
- [ ] Docstring improvements with examples

---

### Phase 10: Validation & Parity Testing

- [ ] PDB coordinate parity vs original priox
- [ ] Atom37/Atom14 format parity vs original priox
- [x] Bond inference parity vs Biotite
- [ ] RBF expansion parity vs original priox
- [ ] Dihedral angle parity vs MDTraj
- [x] Force field XML parsing parity vs OpenMM
- [ ] Trajectory format parity (XTC/DCD/TRR) vs MDTraj
- [ ] HDF5 parsing parity (MDTraj/MDCATH) vs reference loaders
- [x] Hydrogen addition parity vs hydride/PDBFixer

---

### Phase 11: OpenMM Export ✅ COMPLETE (Dec 2025)

- [x] `to_openmm_topology()` method on AtomicSystem/Protein
- [x] `to_openmm_system()` method with force field integration:
  - NonbondedForce (charges, LJ with switching function)
  - HarmonicBondForce (bonds with length/k)
  - HarmonicAngleForce (angles with theta/k)
  - PeriodicTorsionForce (proper dihedrals)
  - PeriodicTorsionForce (improper dihedrals)
  - 1-2 exclusions for bonded atoms
  - Proper unit conversions (Å→nm, kcal/mol→kJ/mol)
- [x] Round-trip validation: `tests/validation/test_openmm_roundtrip.py`
- [x] Platform-agnostic simulation setup (tested with energy minimization)

---

### Phase 12: MD Physics Extensions ✅ COMPLETE (Dec 2025)

- [x] `parameterize_molecule()` - GAFF-based ligand parameterization
- [x] `AtomicSystem.merge_with()` - System merging for complexes
- [x] `assign_mbondi2_radii()` - GBSA mbondi2 radii (PyO3)
- [x] `assign_obc2_scaling_factors()` - OBC2 scaling (PyO3)
- [x] `get_water_model()` - TIP3P/SPCE/TIP4PEW (PyO3)
- [x] `compute_bicubic_params()` - CMAP splines (PyO3)

---

### Phase 9: Data Structure Unification (ProteinTuple Deprecation)

**Goal**: Simplify the codebase by removing the redundant `ProteinTuple` container. **(COMPLETED)**

- [x] Mark `ProteinTuple` as deprecated with warnings
- [x] Update all parsers to return `Protein` directly (using `from_rust_dict`)
- [x] Add `parse_pdb_to_protein()` function returning `Protein` directly
- [x] Add `ProteinLike` union type for backward compatibility
- [x] Add `ensure_protein()` helper function
- [x] Migrate existing code using `ProteinTuple` to use `Protein`
- [x] Remove `Protein.from_tuple()` and `Protein.from_tuple_numpy()` methods
- [x] Delete `ProteinTuple` class entirely
- [x] Update tests to use `Protein` exclusively
- [x] Simplify `Protein` to rely more on `AtomicSystem` inheritance

## 7. Migration Strategy

**Phase A:** Rust available, Python default

- Keep biotite/mdtraj
- `PRIOX_USE_RUST=1` to enable

**Phase B:** Gradual rollout  

- Rust default for PDB
- Biotite fallback for edge cases
- Extensive testing

**Phase C:** Full Rust

- All formats in Rust
- Remove biotite/mdtraj dependencies
- Major version bump

---

## Summary

This all-Rust approach:

- ✅ Maximizes performance (no Python overhead)
- ✅ Simplifies Python codebase  
- ✅ Enables advanced optimizations (SIMD, parallelism)
- ✅ Single source of truth for processing logic
- ✅ Easier to maintain (one language for core logic)

Python becomes a thin wrapper that just receives finished Proteins.
