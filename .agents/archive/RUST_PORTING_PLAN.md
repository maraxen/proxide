# Rust Porting Plan

## Status: Phase 4 In Progress

### Completed

- [x] **Phase 1-3**: Structure parsing, MD parameterization, cleanup
- [x] Removed `biotite.py`, `core.py` (legacy)
- [x] Removed `gbsa.py`, `water.py`, `cmap.py`, `complex.py`, `ligand.py`

### New Rust Modules

| Module | Purpose | Status |
|--------|---------|--------|
| `gbsa.rs` | mbondi2 radii, OBC2 scaling | ✅ Complete |
| `water.rs` | TIP3P/SPCE/TIP4P models | ✅ Complete |
| `cmap.rs` | Bicubic spline for CMAP | ✅ Complete |

### Remaining

- [ ] Extend `md_params.rs` for ligand/GAFF parameterization
- [ ] Add `AtomicSystem.merge_with()` for complex building
- [ ] Expose new Rust functions via PyO3 bindings

### Migration Guide

```python
# Old API (removed)
from priox.md import parameterize_system  # Raises NotImplementedError

# New API
from priox.io.parsing.rust import parse_structure, OutputSpec
spec = OutputSpec()
spec.parameterize_md = True
protein = parse_structure("structure.pdb", spec)
```
