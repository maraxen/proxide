# Priox

**Priox** is a specialized library for Protein I/O and Physics bridging in JAX. It provides efficient tools for loading, processing, and converting protein structure data in the JAX ecosystem, as well as bridging with MD engines like JAX MD.

**NOTE**: This is a work-in-progress library and is not yet ready for production use. It is currently in active development and subject to change.

## Features

- **Robust I/O**: Load protein structures from PDB, CIF, and other formats using Biotite and MDTraj backends.
- **JAX Integration**: Seamlessly convert data to JAX arrays for high-performance computing.
- **Physics Bridge**: Tools to interface with JAX MD for molecular dynamics simulations.
- **Chemical Utilities**: Comprehensive conversion tools for amino acid sequences and structural representations.

## Installation

First, ensure you have [uv](https://github.com/astral-sh/uv) installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then, to install Priox, simply run:

```bash
uv pip install .
```

For development installation with test dependencies:

```bash
uv pip install -e ".[dev]"
```

## Usage

### Loading a Structure

```python
from priox.io.parsing import biotite as bio

# Load a PDB file
structure = bio.load_structure("path/to/file.pdb")
print(structure)
```

### Sequence Conversion

```python
from priox.chem import conversion

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
