# AtomicSystem Architecture (Implemented)

## Status

**Implemented** (December 2025)

## Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                    AtomicSystem (base)                       │
│  - coordinates: (N_atoms, 3)                                │
│  - atom_mask: (N_atoms,)                                    │
│  - atom_names: List[str]                                    │
│  - elements: List[str]                                      │
│  - charges: Optional[ndarray]                               │
│  - sigmas/epsilons: Optional[ndarray]                       │
│  - bonds: Optional[(N_bonds, 2)]                            │
│  - bond_params: Optional[(N_bonds, 2)]                      │
│  - angle_params: Optional[(N_angles, 2)]                    │
│  - exclusion_mask: Optional[(N_atoms, N_atoms)]             │
└──────────────────────────┬──────────────────────────────────┘
                           │
                       │
          ┌────────────┴────┐
          ▼                 ▼
┌─────────────────┐ ┌─────────────────┐
│    Protein      │ │    Molecule     │
│  - aatype       │ │  - smiles       │
│  - residue_idx  │ │  - atom_types   │
│  - chain_idx    │ │  - stereochem   │
│  - atom37/14    │ │                 │
│  - dihedrals    │ │                 │
│  - physics_feat │ │                 │
└─────────────────┘ └─────────────────┘
```

---

## Key Design Decisions

### 1. AtomicSystem Base Class

```python
@dataclass
class AtomicSystem:
    """Base class for any atomic system."""
    
    # Core coordinates (all atoms)
    coordinates: np.ndarray  # (N_atoms, 3)
    atom_mask: np.ndarray    # (N_atoms,) - 1.0 for real, 0.0 for padding
    
    # Atom identity
    atom_names: list[str]    # Per-atom names
    elements: list[str]      # Element symbols
    
    # MD parameters (optional)
    charges: np.ndarray | None = None
    sigmas: np.ndarray | None = None
    epsilons: np.ndarray | None = None
    radii: np.ndarray | None = None
    atom_types: list[str] | None = None  # Force field atom types
    
    # Topology (optional)
    bonds: np.ndarray | None = None         # (N_bonds, 2) indices
    bond_params: np.ndarray | None = None   # (N_bonds, 2) k, r0
    angle_params: np.ndarray | None = None
    exclusion_mask: np.ndarray | None = None

    # Metadata
    source: str | None = None
```

### 2. Protein (Residue-Based)

```python
@dataclass
class Protein(AtomicSystem):
    """Residue-organized protein structure."""
    
    # Residue organization
    aatype: np.ndarray            # (N_res,) residue types 0-20
    residue_index: np.ndarray     # (N_res,) PDB numbering
    chain_index: np.ndarray       # (N_res,)
    
    # Formatted coordinates (for ML)
    atom37_coords: np.ndarray | None = None  # (N_res, 37, 3)
    atom37_mask: np.ndarray | None = None    # (N_res, 37)
    
    # Features
    one_hot_sequence: np.ndarray | None = None
    dihedrals: np.ndarray | None = None
    physics_features: np.ndarray | None = None
    backbone_indices: np.ndarray | None = None
```

### 3. Molecule (Small Molecule/Ligand)

```python
@dataclass
class Molecule(AtomicSystem):
    """Small molecule or ligand."""
    
    smiles: str | None = None
    mol_name: str | None = None
    
    # GAFF-style atom types for non-protein
    gaff_types: list[str] | None = None
```

### 4. Component Access (Properties)

Instead of separate container classes, components are accessed via dynamic properties on `AtomicSystem`.

---

## Rust Parser Changes

### OutputSpec Extensions

```rust
pub enum SystemType {
    ProteinOnly,      // Current behavior
    FullSystem,       // Everything as AtomicSystem
}

pub struct OutputSpec {
    // ... existing fields ...
    
    // System type handling
    pub system_type: SystemType,
    
    // Non-protein handling
    pub include_hetatm: bool,       // Include HETATM records
    pub include_solvent: bool,      // Include HOH/WAT
    pub parse_ligands: bool,        // Separate ligand parsing
    
    // MD bridge
    pub compute_topology: bool,     // Infer bonds/angles
    pub ligand_force_field: Option<String>,  // GAFF, etc.
}
```

---

## Migration Path

### Phase 1: Current (Completed)

- [x] ProteinTuple + Protein for residue-based parsing
- [x] Rust formatters (Atom37, Atom14, Backbone, Full)
- [x] MD parameterization for proteins

### Phase 2: AtomicSystem Foundation ✅ COMPLETE

- [x] Create `AtomicSystem` base class (`src/priox/core/atomic_system.py`)
- [x] Define `Molecule` class for ligands
- [x] Refactor `Protein` to inherit from `AtomicSystem`
- [x] Keep backward compatibility with existing API
- [x] Expose Rust `AtomicSystem`, `Molecule` structs via PyO3

### Phase 3: Non-Protein Support

- [x] Add `Molecule` class for ligands
- [x] Rust: `include_hetatm` flag in OutputSpec
- [x] Rust: Separate HETATM grouping

### Phase 4: MD Bridge ✅ COMPLETE

- [x] Bond inference for all atom types
- [x] GAFF parameterization for ligands
- [x] Topology builder for OpenMM export

### Phase 5: Full Integration

- [x] Unified MD simulation interface (`relax_hydrogens` uses this)
- [ ] ML model support for protein-ligand

---

## Benefits

1. **Cleaner separation** - Proteins, ligands, solvent as distinct types
2. **MD-ready** - Direct path from parsed structure to simulation
3. **Extensible** - Easy to add new molecule types
4. **Backward compatible** - Protein still works as before
5. **ML-friendly** - Formatted coordinates for protein, raw for others

---

## Implementation Priority

| Priority | Item | Effort | Status |
|----------|------|--------|--------|
| P1 | AtomicSystem base class | Low | ✅ Done |
| P1 | Protein inherits AtomicSystem | Medium | ✅ Done |
| P2 | include_hetatm in Rust | Low | ✅ Done |
| P2 | HETATM grouping in Rust | Medium | ✅ Done |
| P3 | Molecule class | Low | ✅ Done |
| P4 | GAFF integration | High | ✅ Done |
| P4 | Topology builder | High | ✅ Done |
