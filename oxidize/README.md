# Oxidize

**Oxidize** is the high-performance Rust extension for Proxide, providing efficient structure parsing, hydrogen addition, and MD parameterization. It is built and distributed as part of the `proxide` package.

> [!NOTE]
> This directory is a subdirectory of `proxide`. Build and install from the parent directory using `maturin develop` or `uv pip install .`

## Building

From the `proxide` root directory:

```bash
cd proxide
maturin develop --release
```

## Python API

### OutputSpec Configuration

```python
from oxidize import OutputSpec, CoordFormat, HydrogenSource

spec = OutputSpec()
spec.coord_format = CoordFormat.Full
spec.parameterize_md = True
spec.force_field = "path/to/forcefield.xml"

# Hydrogen addition
spec.add_hydrogens = True
spec.hydrogen_source = HydrogenSource.FragmentLibrary
spec.relax_hydrogens = True  # Energy minimize after placement
```

### HydrogenSource Enum

| Variant | Description |
|---------|-------------|
| `ForceFieldFirst` | Default. Use FF templates, fallback to fragment library |
| `FragmentLibrary` | Geometric placement via Kabsch alignment |
| `ForceFieldOnly` | FF templates only (error if undefined) |

### Parsing Structures

```python
from oxidize import parse_structure

result = parse_structure("protein.pdb", spec)
# Returns dict with coordinates, atom_mask, charges, bonds, angles, dihedrals, etc.
```

## Testing

Parity tests validate energy calculations against OpenMM:

```bash
pytest tests/physics/test_explicit_parity.py -v
```

Tests include:

- Bond energy parity
- Angle energy parity  
- Dihedral energy parity
- Nonbonded energy validation

## Architecture

```text
oxidize/

├── src/
│   ├── lib.rs           # PyO3 module entry point
│   ├── spec.rs          # OutputSpec, CoordFormat, HydrogenSource
│   ├── formatters/      # Atom37, Atom14, Full coordinate formatters
│   ├── geometry/        # Hydrogen addition, bond inference, relaxation
│   └── physics/         # MD parameterization, force field parsing
└── Cargo.toml
```
