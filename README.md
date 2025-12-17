# Proxide

**Proxide** is a specialized library for Protein I/O and Physics bridging in JAX. It provides efficient tools for loading, processing, and converting protein structure data in the JAX ecosystem, as well as bridging with MD engines like JAX MD.

**NOTE**: This is a work-in-progress library and is not yet ready for production use. It is currently in active development and subject to change.

## Features

- **Robust I/O**: Load protein structures from PDB, CIF, and other formats using Biotite and MDTraj backends.
- **JAX Integration**: Seamlessly convert data to JAX arrays for high-performance computing.
- **Physics Bridge**: Tools to interface with JAX MD for molecular dynamics simulations.
- **Chemical Utilities**: Comprehensive conversion tools for amino acid sequences and structural representations.

## Installation

### From GitHub (Recommended)

```bash
pip install git+https://github.com/maraxen/proxide.git@main
```

### From Source

To install Proxide from source, clone the repository and run:

```bash
git clone https://github.com/maraxen/proxide.git
cd proxide
pip install .
```

For development installation with test dependencies:

```bash
pip install -e ".[dev]"
```

## Usage

### Loading a Structure

```python
from proxide.io.parsing import biotite as bio

# Load a PDB file
structure = bio.load_structure("path/to/file.pdb")
print(structure)
```

### Hydrogen Addition

Proxide supports automatic hydrogen placement with configurable source selection:

```python
from oxidize import OutputSpec, HydrogenSource, CoordFormat
from proxide.io.parsing.rust import parse_structure

spec = OutputSpec(
    coord_format=CoordFormat.Full,
    add_hydrogens=True,
    hydrogen_source=HydrogenSource.FragmentLibrary,  # Geometric placement
    relax_hydrogens=True  # Energy minimize after placement
)

protein = parse_structure("path/to/file.pdb", spec)
```

**HydrogenSource Options:**

- `ForceFieldFirst` (default): Use FF templates, fallback to fragment library
- `FragmentLibrary`: Use geometric placement via Kabsch alignment
- `ForceFieldOnly`: Use FF templates only (fails if undefined)

### Sequence Conversion

```python
from proxide.chem import conversion

# Convert sequence string to integer encoding
seq_ints = conversion.string_to_protein_sequence("ACDEF")
```

## Development

### Running Tests

Run the full test suite:

```bash
pytest
```

Run fast "smoke" tests to verify core functionality:

```bash
pytest -m smoke
```

### Linting and Typing

Check code quality:

```bash
ruff check .
ty check
```
