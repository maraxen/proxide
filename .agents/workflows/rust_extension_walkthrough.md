# Priox Rust Extension Walkthrough

**Last Updated:** 2025-12-13

This walkthrough covers the main features of the `priox_rs` Rust extension and how to use them effectively.

---

## Table of Contents

1. [Installation & Setup](#installation--setup)
2. [Structure Parsing](#structure-parsing)
3. [PQR Files with Charges](#pqr-files-with-charges)
4. [Hydrogen Addition](#hydrogen-addition)
5. [Force Field Parameterization](#force-field-parameterization)
6. [Mass Assignment](#mass-assignment)
7. [OpenMM Export](#openmm-export)
8. [GBSA & Water Models](#gbsa--water-models)

---

## Installation & Setup

```bash
# Build the Rust extension in development mode
cd /path/to/priox
uv run maturin develop --release -m rust_ext/Cargo.toml

# Verify installation
uv run python -c "import priox_rs; print('priox_rs installed!')"
```

---

## Structure Parsing

### Basic PDB/mmCIF Parsing

```python
import priox_rs

# Create output specification
spec = priox_rs.OutputSpec()
spec.infer_bonds = True  # Infer bonds from distances

# Parse structure
result = priox_rs.parse_structure("protein.pdb", spec)

# Access results
print(f"Residues: {len(result['aatype'])}")
print(f"Coordinates shape: {result['coordinates'].shape}")
print(f"Bonds: {len(result['bonds'])}")
```

### With MD Parameterization

```python
import priox_rs

spec = priox_rs.OutputSpec()
spec.parameterize_md = True
spec.force_field = "path/to/protein.ff14SB.xml"
spec.infer_bonds = True

result = priox_rs.parse_structure("protein.pdb", spec)

# Now has force field parameters
print(f"Charges: {len(result['charges'])}")
print(f"Angles: {len(result['angles'])}")
print(f"Dihedrals: {len(result['dihedrals'])}")
```

### Adding Hydrogens

```python
import priox_rs

spec = priox_rs.OutputSpec()
spec.add_hydrogens = True  # Add missing hydrogens
spec.relax_hydrogens = True  # Energy-minimize hydrogen positions

result = priox_rs.parse_structure("protein.pdb", spec)
```

---

## PQR Files with Charges

PQR files contain atomic charges and radii for electrostatics calculations.

```python
import priox_rs
from priox.io.parsing.pqr import load_pqr

# Direct Rust access
data = priox_rs.parse_pqr("structure.pqr")
print(f"Atoms: {data['num_atoms']}")
print(f"Charges: {data['charges'][:5]}")
print(f"Radii: {data['radii'][:5]}")

# Python wrapper (returns AtomicSystem)
systems = list(load_pqr("structure.pqr"))
system = systems[0]
print(f"Coordinates: {system.coordinates.shape}")
print(f"Charges: {system.charges.shape}")
```

---

## Hydrogen Addition

The Rust extension provides fragment-based hydrogen placement with optional energy relaxation.

```python
import priox_rs

# Option 1: During parsing
spec = priox_rs.OutputSpec()
spec.add_hydrogens = True

result = priox_rs.parse_structure("protein.pdb", spec)

# Option 2: Standalone function
# (if you have raw atom data already)
# raw_data = priox_rs.parse_pdb("protein.pdb")
# with_hydrogens = priox_rs.add_hydrogens(raw_data)
```

### With Energy Relaxation

Energy relaxation improves hydrogen geometry using a quick energy minimization.

```python
import priox_rs

spec = priox_rs.OutputSpec()
spec.add_hydrogens = True
spec.relax_hydrogens = True  # OpenMM-based relaxation

result = priox_rs.parse_structure("protein.pdb", spec)
```

---

## Force Field Parameterization

Load OpenMM-compatible force field XML files and assign parameters.

```python
import priox_rs

# Load force field
ff = priox_rs.load_forcefield("path/to/protein.ff14SB.xml")

# Access parameters
print(f"Atom types: {len(ff['atoms'])}")
print(f"Bonds: {len(ff['bonds'])}")
print(f"Angles: {len(ff['angles'])}")
```

### Python Force Field Loader

```python
from priox.physics.force_fields.loader import load_force_field

# Loads from priox/assets/ or full path
ff = load_force_field("protein.ff14SB")

# Get parameters for an atom
charge = ff.get_charge("ALA", "CA")
sigma, epsilon = ff.get_lj_params("ALA", "CA")
```

---

## Mass Assignment

Assign atomic masses based on element type.

```python
import priox_rs

# Rust implementation (fast)
masses = priox_rs.assign_masses(["N", "CA", "C", "O", "H"])
print(f"Masses: {masses}")
# Output: [14.007, 12.011, 12.011, 15.999, 1.008]

# Python wrapper (uses Rust internally)
from priox.md.bridge.utils import assign_masses
masses = assign_masses(["N", "CA", "C"])
```

---

## OpenMM Export

Convert AtomicSystem/Protein to OpenMM for simulation.

```python
from priox.core.containers import Protein

# Parse with parameterization
protein = Protein.from_rust_dict(
    priox_rs.parse_structure("protein.pdb", spec)
)

# Convert to OpenMM
topology = protein.to_openmm_topology()
system = protein.to_openmm_system(
    nonbonded_cutoff=1.0,  # nm
    use_switching_function=True,
    switch_distance=0.9,  # nm
    coulomb14scale=0.8333,
    lj14scale=0.5,
)

# Run simulation
from openmm import LangevinIntegrator, Platform
from openmm.app import Simulation

integrator = LangevinIntegrator(300, 1.0, 0.002)
simulation = Simulation(topology, system, integrator)
simulation.minimizeEnergy()
```

---

## GBSA & Water Models

### GBSA Radii and Scaling

```python
import priox_rs

elements = ["N", "C", "C", "O", "H"]
atom_names = ["N", "CA", "C", "O", "H"]
charges = [0.0, 0.1, -0.1, 0.0, 0.0]

# Assign mbondi2 radii
radii = priox_rs.assign_mbondi2_radii(elements, atom_names, charges)

# Assign OBC2 scaling factors
scaling = priox_rs.assign_obc2_scaling_factors(elements)
```

### Water Models

```python
import priox_rs

# Get TIP3P water model parameters
water = priox_rs.get_water_model("tip3p")
print(f"O charge: {water['o_charge']}")
print(f"H charge: {water['h_charge']}")
print(f"O-H distance: {water['oh_distance']}")
```

### CMAP Backbone Corrections

```python
import priox_rs

# Compute bicubic interpolation parameters for CMAP grid
grid_1d = [...]  # Energy values on phi-psi grid
coefficients = priox_rs.compute_bicubic_params(grid_1d)
```

---

## API Reference

### OutputSpec Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `coord_format` | `str` | `"atom37"` | Output coordinate format |
| `models` | `list[int]` | `None` | Models to include (None = all) |
| `chains` | `list[str]` | `None` | Chains to include (None = all) |
| `remove_hetatm` | `bool` | `False` | Remove HETATM records |
| `remove_solvent` | `bool` | `True` | Remove water/solvent |
| `add_hydrogens` | `bool` | `False` | Add missing hydrogens |
| `relax_hydrogens` | `bool` | `False` | Energy-minimize H positions |
| `infer_bonds` | `bool` | `False` | Infer bonds from distances |
| `parameterize_md` | `bool` | `False` | Assign FF parameters |
| `force_field` | `str` | `None` | Path to force field XML |

### Main Functions

| Function | Description |
|----------|-------------|
| `parse_structure(path, spec)` | Parse PDB/mmCIF with options |
| `parse_pdb(path)` | Parse PDB file (raw) |
| `parse_mmcif(path)` | Parse mmCIF file (raw) |
| `parse_pqr(path)` | Parse PQR file with charges/radii |
| `load_forcefield(path)` | Load OpenMM XML force field |
| `assign_masses(atom_names)` | Get atomic masses |
| `assign_mbondi2_radii(...)` | Get GBSA radii |
| `assign_obc2_scaling_factors(...)` | Get OBC2 scaling |
| `get_water_model(name)` | Get water model params |

---

## Next Steps

After completing this walkthrough, see:

- **ROADMAP.md** - Overall development roadmap
- **VALIDATION_ROADMAP.md** - Parity testing status
- **TECHNICAL_DEBT.md** - Known issues and deferred work
