# Comprehensive Rust Port Plan - ALL OPERATIONS

**Goal:** Port ALL priox feature extraction operations to Rust for a complete end-to-end pipeline.

---

## Current Architecture (Python/JAX)

```
PDB File → Parse → Protein (JAX) → Feature Extraction (JAX) → ML Model
                                    ├─ Radial Basis Functions
                                    ├─ Sequence Alignment (SW/NW)
                                    ├─ Electrostatic Forces
                                    ├─ vdW Forces
                                    ├─ MD Parameterization
                                    └─ Geometry Transforms
```

## Target Architecture (All-Rust + OutputSpec)

```
Input File + OutputSpec → RUST → Protein (with ALL requested features)
                          ├─ Parse (PDB/mmCIF/XTC)
                          ├─ Format (atom37/14/custom)
                          ├─ Geometry (RBF, alignment, transforms)
                          ├─ Physics (electrostatics, vdW, MD params)
                          └─ Zero-copy → NumPy arrays
```

**Python receives finished Protein with all features pre-computed.**

---

## Operations to Port

### ✅ Phase 1: COMPLETED

- [x] Raw atom parsing (PDB)
- [x] RawAtomData structure
- [x] Zero-copy NumPy conversion

### 📋 Phase 2: Structure Processing & Formatting

#### A. Residue/Chain Processing

- [ ] Residue grouping from raw atoms
- [ ] Chain ID mapping
- [ ] Multi-model handling
- [ ] Residue type mapping (3-letter → index)

#### B. Coordinate Formatters

- [x] Atom37Formatter - (N_res, 37, 3) with standard atom ordering
- [x] Atom14Formatter - (N_res, 14, 3) reduced representation
- [x] BackboneFormatter - (N_res, 4, 3) N, CA, C, O only
- [ ] CustomFormatter - user-specified atoms
- [x] Format caching layer

### 📋 Phase 3: Geometry Operations

#### A. Radial Basis Functions ⭐ (Completed)

**Source:** `priox/geometry/radial_basis.py`

**What it does:**

- Computes RBF encoding of inter-atomic distances
- 16 radial basis functions (2-22 Å range)
- For all 25 backbone atom pairs (N-N, CA-CA, etc.)
- Returns (N_res, K_neighbors, 25×16) features

**Rust implementation:**

```rust
pub fn compute_radial_basis(
    backbone_coords: &[(f32, f32, f32)],  // (N_res * 5, 3) - N,CA,C,CB,O
    neighbor_indices: &[usize],            // K nearest neighbors per residue
) -> Vec<f32> {  // (N_res, K, 400) flattened
    // 1. Compute pairwise distances for all backbone pairs
    // 2. Gather neighbors using indices
    // 3. Apply Gaussian RBF: exp(-(d - center)^2 / sigma^2)
    // 4. Return concatenated features
}
```

**Dependencies:**

- Backbone coordinate extraction
- Distance calculations
- Gaussian functions

---

#### B. Sequence Alignment Algorithms ⭐⭐

**Source:** `priox/geometry/alignment.py`

**What it does:**

- **Smith-Waterman** (local alignment) - 3 variants:
  - No gap penalty
  - Linear gap penalty  
  - Affine gap penalty (most accurate)
- **Needleman-Wunsch** (global alignment)
- Differentiable via JAX autodiff for soft alignment
- Returns alignment traceback for position mapping

**Key challenge:** These use dynamic programming with JAX's scan/vmap for differentiation.

**Rust implementation strategy:**

```rust
// Traditional DP (non-differentiable, fast)
pub fn smith_waterman_affine(
    seq_a: &[u8],      // Amino acid indices
    seq_b: &[u8],
    gap_open: f32,
    gap_extend: f32,
) -> AlignmentResult {
    // Standard 3-matrix affine gap DP
    // M[i,j] = match/mismatch
    // I[i,j] = insertion
    // D[i,j] = deletion
}

// Soft/differentiable version (for gradient-based learning)
pub fn smith_waterman_soft(
    seq_a: &[u8],
    seq_b: &[u8],
    temperature: f32,  // For soft-max
) -> Vec<f32> {  // Gradient-compatible traceback
    // Use log-sum-exp for soft maximum
    // Return position mapping matrix
}
```

**Options:**

1. **Port just the core DP** (non-differentiable) → Fast, simple
2. **Port with autodiff** using autodiff crates → Complex, maintains differentiability
3. **Hybrid:** Rust for inference, JAX for training → Practical

**Recommendation:** Start with Option 1 (non-differentiable), add Option 3 if gradient needed.

---

#### C. Geometry Transforms

**Source:** `priox/geometry/transforms.py`

- Backbone coordinate extraction (N, CA, C, CB, O)
- Rotation/translation transforms
- Distance/angle calculations
- Coordinate frame conversions

**Rust:**

```rust
pub fn compute_backbone_coordinates(
    coords_atom37: &[f32],  // (N_res, 37, 3)
    atom_mask: &[f32],      // (N_res, 37)
) -> Vec<f32> {  // (N_res, 5, 3) for N,CA,C,CB,O
    // Extract indices 0,1,2,3,4 from atom37
}

pub fn apply_rotation(coords: &mut [f32], rotation: &[f32; 9]) { ... }
pub fn compute_distances(coords_a: &[f32], coords_b: &[f32]) -> Vec<f32> { ... }
```

---

### ✅ Phase 4: Physics Features (COMPLETED)

#### A. Electrostatic Features

- [x] Coulomb forces
- [x] Force projections
- [x] Noise scaling

#### B. Van der Waals Features

- [x] Lennard-Jones parameters
- [x] Force calculation
- [x] Integration with defaults

#### C. MD Parameterization

- [x] Atom type assignment
- [x] Bond/Angle/Dihedral inference
- [x] Charge assignment
- [x] Exclusion masks (1-2, 1-3, 1-4)
- [x] GAFF integration

### ✅ Phase 5: Force Field Integration (COMPLETED)

#### XML Parser

- [x] OpenMM XML parser implemented (AtomTypes, Residues, NonbondedForce)
- [x] Topology generation (Bonds, Angles, Dihedrals)
- [x] Integration with `OutputSpec`

#### GAFF Support

- [x] GAFF atom typing
- [x] Parameter loading from internal assets

---

### 📋 Phase 6: Optimization & Polish (Ongoing)

#### Coordinate Formatters

- [x] Atom37Formatter
- [x] Atom14Formatter
- [x] BackboneFormatter
- [x] FullFormatter (for MD/OpenMM export)
- [ ] CustomFormatter

#### Performance

- [x] Caching layer
- [ ] SIMD optimizations
- [ ] Further Rayon parallelization

---

## Unified OutputSpec (Extended)

```rust
pub struct OutputSpec {
    // === STRUCTURE ===
    pub coord_format: CoordFormat,        // Atom37, Atom14, Full, etc.
    pub models: Option<Vec<usize>>,       // Which models (default: all)
    pub chains: Option<Vec<String>>,
    pub remove_hetatm: bool,
    pub remove_solvent: bool,
    
    // === GEOMETRY FEATURES ===
    pub compute_rbf: bool,                // Radial basis functions
    pub rbf_num_neighbors: usize,         // K neighbors for RBF (default: 30)
    pub compute_alignment: bool,          // Smith-Waterman for multi-protein
    pub alignment_algorithm: AlignmentAlg, // SW_Affine, NW, etc.
    
    // === PHYSICS FEATURES ===
    pub compute_electrostatics: bool,
    pub electrostatics_noise: Option<f32>,
    pub compute_vdw: bool,
    pub vdw_noise: Option<f32>,
    
    // === MD PARAMETERIZATION ===
    pub parameterize_md: bool,
    pub force_field: Option<String>,      // "ff14SB", "ff19SB", etc.
    pub infer_bonds: bool,
    pub add_hydrogens: bool,
    
    // === OPTIONAL FIELDS ===
    pub include_b_factors: bool,
    pub include_occupancy: bool,
    pub include_dihedrals: bool,
    
    // === ERROR HANDLING ===
    pub error_mode: ErrorMode,
    
    // === PERFORMANCE ===
    pub enable_caching: bool,
    pub num_threads: usize,               // Rayon parallelism
}
```

---

## Rust Module Structure (Complete)

```
rust_ext/src/
├── lib.rs                    # PyO3 entry, parse_structure()
│
├── structure.rs              # RawAtomData, ProcessedStructure
├── spec.rs                   # OutputSpec, enums
│
├── formats/                  # === PARSING ===
│   ├── pdb.rs               # ✅ Done
│   ├── pqr.rs
│   ├── mmcif.rs
│   └── xtc.rs               # Trajectory
│
├── processing/              # === POST-PARSE ===
│   ├── residues.rs          # Grouping, chain mapping
│   ├── models.rs            # Multi-model handling
│   └── filters.rs           # Solvent/HETATM removal
│
├── formatters/              # === COORDINATE FORMATTING ===
│   ├── atom37.rs
│   ├── atom14.rs
│   ├── full.rs
│   └── cache.rs             # Format caching
│
├── geometry/                # === GEOMETRY OPERATIONS ===
│   ├── radial_basis.rs      # RBF encoding ⭐
│   ├── alignment.rs         # Smith-Waterman, Needleman-Wunsch ⭐⭐
│   ├── transforms.rs        # Rotations, distances, frames
│   ├── backbone.rs          # Backbone extraction
│   └── neighbors.rs         # KNN, cell lists
│
├── physics/                 # === PHYSICS FEATURES ===
│   ├── electrostatics.rs    # Coulomb forces ⭐⭐
│   ├── vdw.rs              # Lennard-Jones forces
│   ├── projections.rs       # Force projections
│   ├── constants.rs         # Physical constants
│   └── md_params.rs         # MD parameterization ⭐⭐⭐
│
├── forcefield/              # === FORCE FIELD ===
│   ├── xml_parser.rs        # OpenMM XML parser
│   ├── residue_templates.rs # Template matching
│   ├── topology.rs          # Bond/angle inference
│   └── parameters.rs        # LJ, charges, etc.
│
└── output/
    └── protein_tuple.rs     # Final conversion to Protein
```

---

## Implementation Priority

### Tier 1: Core Infrastructure (Weeks 1-2)

1. ✅ Raw parsing (PDB) - Done
2. Residue processing & chain mapping
3. Atom37 formatter
4. Basic geometry transforms (backbone extraction, distances)

### Tier 2: Physics Essentials (Weeks 3-4)

5. Force field XML parser ⭐⭐⭐ (BLOCKER for physics)
6. MD parameterization (bonds, angles, charges)
7. Electrostatic features
8. vdW features

### Tier 3: Advanced Geometry (Weeks 5-6)

9. Radial basis functions
10. Smith-Waterman alignment
11. Needleman-Wunsch alignment
12. Neighbor search (KNN, cell lists)

### Tier 4: Optimization & Polish (Weeks 7-8)

13. Format caching
14. Rayon parallelization
15. SIMD optimizations
16. Extended formats (mmCIF, XTC, TRR, MDTraj HDF5 ✅, MDCATH HDF5 ✅)
17. Comprehensive testing

---

## External Crate Dependencies

```toml
[dependencies]
pyo3 = { version = "0.20", features = ["extension-module"," abi3-py311"] }
numpy = "0.20"
rayon = "1.8"              # Parallelism
nalgebra = "0.32"          # Linear algebra for transforms
quick-xml = "0.31"         # Force field XML parsing
kdtree = "0.7"             # K-nearest neighbors
fnv = "1.0"                # Fast HashMap
log = "0.4"                # Logging
thiserror = "1.0"          # Error types

# Optional for alignment
# autodiff = "0.7"         # If we want differentiable alignment
```

---

## Python API (Final)

```python
from priox_rs import parse_structure, OutputSpec

# Simple case
protein = parse_structure("1uao.pdb")

# With physics
spec = OutputSpec(
    coord_format="atom37",
    compute_electrostatics=True,
    compute_rbf=True,
    parameterize_md=True,
    force_field="ff14SB",
    include_b_factors=True,
)
protein = parse_structure("structure.pdb", spec)

# Returns Protein (dataclass) with:
print(protein.coordinates.shape)      # (N_res, 37, 3)
print(protein.physics_features.shape) # (N_res, 5) - electrostatic projections
print(protein.bonds.shape)            # (N_bonds, 2)
print(protein.charges.shape)          # (N_atoms,)
```

---

## Performance Targets

| Operation | Python/JAX | Rust Target | Expected Speedup |
|-----------|------------|-------------|------------------|
| PDB parse | 50ms | 2ms | 25x |
| Atom37 format | 10ms | 0.5ms | 20x |
| RBF computation | 100ms | 5ms | 20x |
| Smith-Waterman | 150ms | 10ms | 15x |
| Electrostatics | 80ms | 8ms | 10x |
| MD params | 200ms | 15ms | 13x |
| **Total Pipeline** | **590ms** | **40ms** | **15x** |

---

## Testing Strategy

### Unit Tests (Rust)

- Each module has tests in Rust
- Compare against hand-calculated values
- Edge cases (empty structures, single atom, etc.)

### Integration Tests (Python)

- Golden master: Compare Rust vs JAX output
- Numerical tolerance for physics (1e-5)
- Exact match for topology (bonds, angles)

### Benchmarks

- `criterion.rs` for Rust benchmarks
- Compare with `pytest-benchmark` for Python

---

## Summary

This plan ports **100% of priox feature extraction** to Rust:

- ✅ **14 major operation categories**
- ✅ **Complete end-to-end pipeline**
- ✅ **Zero-copy to Python**
- ✅ **Configurable via OutputSpec**

Python becomes a thin wrapper receiving fully-featured Proteins.
