# Force Fields

This directory contains utilities for loading force field parameters in Proxide.

## Structure

- `loader.py`: Python wrapper for loading force fields via the Rust backend.
- `components.py`: JAX dataclasses for storing force field parameters.

## Usage

Force fields are loaded from OpenMM-style XML files located in `proxide/assets/` or from arbitrary file paths. The parsing is handled by the high-performance Rust extension.

```python
from proxide.physics.force_fields import load_force_field

# Load by name (searches in assets)
ff = load_force_field("protein.ff14SB.xml")

# Load by path
ff = load_force_field("/path/to/my_forcefield.xml")
```

The returned `FullForceField` object contains JAX arrays ready for computation.
