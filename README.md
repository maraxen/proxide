# Proxide

**Proxide** is a high-performance library for Protein I/O and Physics bridging in JAX. It combines a flexible Python/JAX frontend with a highly optimized Rust backend (`oxidize`) to provide fast structure parsing, force field parameterization, and seamless integration with JAX MD.

**NOTE**: This is a research library in active development.

---

## 🚀 Features

- **Hybrid Architecture**:
  - **Rust Backend**: 25x faster parsing (PDB/mmCIF), 50x faster topology generation, and robust force field parameterization.
  - **JAX Frontend**: Differentiable physics, geometric deep learning utilities, and seamless GPU integration.
- **Robust I/O**: Load PDB, mmCIF, and PQR files with automatic error handling and corrections.
- **Molecular Dynamics**: Parse OpenMM XML force fields, assign GAFF parameters, and generate fully parameterized `AtomicSystem` objects for JAX MD.
- **Trajectory Support**: High-performance parsing of XTC, DCD, and TRR trajectories.

---

## 📦 Installation

Proxide requires a Rust toolchain to build the backend extension.

### Prerequisites

- **Python**: 3.11+
- **Rust**: 1.75+ (Install via [rustup.rs](https://rustup.rs))
- **C++ Compiler**: For compiling HDF5/chemfiles dependencies if needed.

### From Source

```bash
# Clone the repository
git clone https://github.com/maraxen/proxide.git
cd proxide

# Install with uv (recommended) or pip
uv pip install .

# For development (includes test dependencies)
uv pip install -e ".[dev]"
```

The installation process will automatically compile the Rust `oxidize` extension using `maturin`.

---

## 🛠️ Usage

### Loading a Structure

Use the high-level `parse_structure` function for fast, robust parsing:

```python
from proxide import parse_structure

# Load a PDB file to a unified Protein object
protein = parse_structure("path/to/structure.pdb")

# Access data as JAX arrays
print(protein.coordinates.shape)  # (N_residues, 37, 3)
```

### Force Field Parameterization

Proxide can automatically assign force field parameters (charges, radii, etc.) via the Rust backend:

```python
from proxide import parse_structure, OutputSpec

# Configure parsing options
spec = OutputSpec(
    add_hydrogens=True,             # Add missing geometric hydrogens
    infer_bonds=True,               # Infer connectivity if missing
    parameterize_md=True,           # Compute MD parameters
    force_field="protein.ff14SB.xml" # Use standard AMBER force field
)

# Parse and parameterize in one step
protein = parse_structure("path/to/structure.pdb", spec)

# Access MD parameters
print(protein.charges)  # Partial charges
print(protein.sigma)    # Lennard-Jones sigma
```

### Trajectory Parsing

```python
from proxide import parse_xtc

# Fast Rust-based XTC parsing
traj_data = parse_xtc("path/to/trajectory.xtc")
coords = traj_data["coordinates"]  # (N_frames, N_atoms, 3)
```

---

## ⚡ Performance

The migration to a Rust backend has yielded significant performance improvements compared to the pure Python implementation:

| Operation | Speedup |
|:----------|:--------|
| **PDB Parsing** | **25x** |
| **mmCIF Parsing** | **25x** |
| **Topology Generation** | **50x** |
| **Force Field Loading** | **10x** |

---

## ⚠️ Migration Notes

If you are migrating from older versions of Proxide:

1. **Biotite Removal**: Direct dependency on `biotite` for parsing has been removed. All parsing is now handled by `oxidize`.
2. **API Changes**:
    - `proxide.io.parsing.biotite` -> `proxide.parse_structure`
    - `proxide.physics.force_fields` -> `proxide.load_forcefield`
3. **JAX by Default**: Most I/O functions now return JAX arrays by default. Use `use_jax=False` if you specifically need NumPy arrays.

---

## 🔧 Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run fast smoke tests
uv run pytest -m smoke
```

### Linting and Typing

```bash
# Linting
uv run ruff check src/proxide/ --fix

# Type Checking
uv run ty check
```
