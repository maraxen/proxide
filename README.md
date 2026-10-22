# Proxide

> **Pre-release toward 0.1.0.** The [Stability](#stability) section is the single source for what is stable and what may change in a minor release.

**Proxide** is a high-performance library for Protein I/O and Physics bridging in JAX. It combines a flexible Python/JAX frontend with a highly optimized Rust backend (`_proxider`) to provide fast structure parsing, force field parameterization, and seamless integration with JAX MD.

**NOTE**: This is a research library in active development.

---

## 🚀 Features

- **Hybrid Architecture**:
  - **Rust Backend**: 25x faster parsing (PDB/mmCIF), 50x faster topology generation, and robust force field parameterization.
  - **JAX Frontend**: Differentiable physics, geometric deep learning utilities, and seamless GPU integration.
- **Robust I/O**: Load PDB, mmCIF, and PQR files with automatic error handling and corrections.
- **Molecular Dynamics**: Parse OpenMM XML force fields, assign GAFF parameters (experimental), and generate fully parameterized `AtomicSystem` objects for JAX MD.
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

The installation process will automatically compile the Rust `_proxider` extension using `maturin`.

### Optional: Espaloma Charge (ML partial charges)

Inference uses the JAX/Equinox port [`expaloma`](https://github.com/maraxen/expaloma) (bundled weights; no PyTorch/DGL at runtime). The **`[espaloma]`** extra pulls **`expaloma`** from **PyPI** (and `rdkit`).

```bash
uv pip install -e ".[espaloma]"
```

For **local development** against an editable `expaloma` checkout, use a path override (e.g. `uv.sources` in your workspace) or:

```bash
uv pip install -e ../expaloma
uv pip install -e ".[dev]"
```

**OpenFF toolkit wrapper** (legacy upstream `espaloma_charge`): `uv pip install -e ".[espaloma-openff]"`.

API: `proxide.chem.partial_charges` (`assign_espaloma_charges_rdkit`, etc.). Validation uses **golden** vectors under `tests/data/espaloma_golden/`; refresh those files when bumping `expaloma`.

```bash
pytest -m espaloma
```

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

### Fast Partial Charges (Espaloma)

Proxide provides access to high-performance Expaloma graph-neural-network inference for charge assignment using native Rust embedded weights:

```bash
# Assign charges with the fast Rust backend
proxide charges molecule.sdf --backend rust
```

---

## ⚡ Performance

The migration to a Rust backend has yielded significant performance improvements compared to the pure Python implementation:

| Operation | Speedup | Example Metric |
|:----------|:--------|:---------------|
| **PDB Parsing** | **25x** | - |
| **mmCIF Parsing** | **25x** | - |
| **Topology Generation** | **50x** | - |
| **Force Field Loading** | **10x** | - |
| **Espaloma Inference** | **7.5x**| 13.6ms (JAX) ➔ 1.8ms (Rust) |

---

## 🚀 Deployment & CI/CD

Proxide uses **Maturin** to seamlessly build Python wheels and publish them to PyPI. To publish `proxide` releases securely without secrets, we utilize [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC).
Do **not** configure a `PYPI_API_TOKEN` GitHub Secret. Instead, create a Trusted Publisher in the PyPI dashboard bound to this repository to allow `.github/workflows/publish.yml` to authenticate.

---

## ⚠️ Migration Notes

If you are migrating from older versions of Proxide:

1. **Biotite Removal**: Direct dependency on `biotite` for parsing has been removed. All parsing is now handled by `_proxider`.
2. **API Changes**:
    - `proxide.io.parsing.biotite` -> `proxide.parse_structure`
    - `proxide.physics.force_fields` -> `proxide.load_forcefield`
3. **JAX by Default**: Most I/O functions now return JAX arrays by default. Use `use_jax=False` if you specifically need NumPy arrays.

---

## Stability

Only core IO is stable at 0.1.0:

- structure parsers and writers: `parse_pdb`, `parse_mmcif`, `parse_pqr`, `parse_structure`, `parse_parm7`, `parse_amber_trajectory`, `write_pdb`, `write_mmcif`, `write_npz`, `write_dcd`;
- the `Protein` result type returned by the parsers;
- parse options: `OutputSpec` and its enums (`CoordFormat`, `ErrorMode`, `HydrogenSource`), which `parse_structure` takes;
- trajectory readers: `parse_xtc`, `parse_dcd`, `parse_trr`, `read_xtc_lazy`, `read_xtc_parallel`, `read_xtc_ca_distogram`, `frame_count`, `n_atoms`, `TrajectoryStream`, and the MDTraj and mdCATH HDF5 readers;
- structure fetching: `fetch_rcsb`, `fetch_afdb`, `fetch_foldcomp_database`, and the `proxide.io` submodule, except the tensor-bundle helpers below.

`OutputSpec` is stable, but options that enable experimental features are experimental: `parameterize_md`, `force_field`, `water_model`, `strict_parameterization`, `compute_electrostatics`, `electrostatics_noise`, `compute_vdw`, `compute_rbf`, `rbf_num_neighbors`, `relax_hydrogens`, `relax_max_iterations`, `auto_terminal_caps`, `missing_residue_mode` and `unit_system`. The types `UnitSystem` and `MissingResidueMode` are experimental too. `hydrogen_source` is accepted, but no code path reads it yet.

Everything else exported from `proxide` is experimental and may change in a minor release. That covers GAFF2 typing (`assign_gaff_atom_types`), force field loading (`load_forcefield`, `parameterize_molecule`), atomic masses, GB radii and OBC2 scaling, water models, CMAP parameters, geometry helpers, the Jaccard and TM-align modules, the `iterload` function (top level only; the `io` submodule does not export it), and `proxide.AtomicSystem`. `proxide.__experimental__` lists them.

The tensor-bundle helpers in `proxide.io` (`save_tensor_bundle`, `load_tensor_bundle`, `flatten_tensor_dict`, `unflatten_tensor_dict`, `assert_bundle_keys`) are experimental.

confind, rotlib and CCD/fixer are experimental too; they have no top-level Python exports at this version.

## 🔧 Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run fast smoke tests
uv run pytest -m smoke
```

### Building Wheels Locally

Before pushing a release to GitHub, test the wheel build locally to catch platform-specific issues early:

```bash
# Test the build locally (validates HDF5 static compilation and workflow config)
bash scripts/test-wheel-build.sh

# This will:
# 1. Check build prerequisites (cargo, rustc, python3)
# 2. Build the Rust extension with static HDF5 compilation
# 3. Verify the GitHub workflow is correctly configured
```

**Note**: The Rust extension uses static HDF5 compilation via the `hdf5-metno` crate with the `static` feature enabled. This means HDF5 is compiled from source during the build and included in the binary, eliminating platform-specific library dependencies.

### Linting and Typing

```bash
# Linting
uv run ruff check src/proxide/ --fix

# Type Checking
uv run ty check
```
