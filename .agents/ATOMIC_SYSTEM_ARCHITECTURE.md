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

---

## Benefits

1. **Cleaner separation** - Proteins, ligands, solvent as distinct types
2. **MD-ready** - Direct path from parsed structure to simulation
3. **Extensible** - Easy to add new molecule types
4. **Backward compatible** - Protein still works as before
5. **ML-friendly** - Formatted coordinates for protein, raw for others
