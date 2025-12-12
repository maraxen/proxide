# Validation Roadmap

**Purpose:** Verify parity between Rust implementations and reference implementations (original priox Python, Biotite, OpenMM, MDTraj)

---

## 1. Structure Parsing Parity

### 1.1 PDB Parser

| Test | Rust Function | Reference | Tolerance | Status |
|------|--------------|-----------|-----------|--------|
| Atom coordinates | `parse_pdb_file()` | Original priox `parse_pdb()` | 1e-3 Å | ⬜ |
| Residue grouping | `ProcessedStructure::from_raw()` | Original priox processing | Exact | ⬜ |
| Chain mapping | `chain_indices` | Original priox chains | Exact | ⬜ |
| HETATM classification | `molecule_type` | Original priox HETATM | Exact | ⬜ |
| Multi-model handling | `filter_models()` | Biotite `get_structure()` | Exact | ⬜ |

**Test Files:**

- `1CRN.pdb` — Small protein, no HETATM
- `1ATP.pdb` — Protein with ATP ligand
- `2NRL.pdb` — Multi-model NMR structure
- `3HTB.pdb` — Complex with waters/ions

### 1.2 mmCIF Parser

| Test | Rust Function | Reference | Tolerance | Status |
|------|--------------|-----------|-----------|--------|
| Atom coordinates | `parse_mmcif_file()` | Biotite mmCIF parser | 1e-3 Å | ⬜ |
| Entity/chain mapping | chain handling | Biotite | Exact | ⬜ |

### 1.3 PQR Parser

| Test | Rust Function | Reference | Tolerance | Status |
|------|--------------|-----------|-----------|--------|
| Coordinates | `parse_pqr_file()` | PDB2PQR output | 1e-3 Å | ⬜ |
| Charges | charges field | PDB2PQR output | 1e-4 | ⬜ |
| Radii | radii field | PDB2PQR output | 1e-4 | ⬜ |

---

## 2. Geometry Operations Parity

### 2.1 Distance Calculations

| Test | Rust Function | Reference | Tolerance | Status |
|------|--------------|-----------|-----------|--------|
| Euclidean distance | `euclidean_distance()` | NumPy `linalg.norm` | 1e-6 | ⬜ |
| Distance matrix | `distance_matrix()` | Original priox `distances.py` | 1e-5 | ⬜ |

### 2.2 Dihedral Angles

| Test | Rust Function | Reference | Tolerance | Status |
|------|--------------|-----------|-----------|--------|
| Phi/Psi angles | `compute_dihedral()` | MDTraj `compute_phi/psi` | 1e-4 rad | ⬜ |
| Omega angles | dihedral calculation | MDTraj `compute_omega` | 1e-4 rad | ⬜ |

### 2.3 Bond Inference

| Test | Rust Function | Reference | Tolerance | Status |
|------|--------------|-----------|-----------|--------|
| Covalent bond detection | `infer_bonds()` | Biotite `connect_via_distances()` | Exact | ⬜ |
| Bond count per residue | topology bonds | Standard amino acid | ±1 bond | ⬜ |

### 2.4 Neighbor Search

| Test | Rust Function | Reference | Tolerance | Status |
|------|--------------|-----------|-----------|--------|
| K-nearest neighbors | `find_k_nearest_neighbors()` | Biotite `CellList` | Exact | ⬜ |
| Cutoff neighbors | `find_neighbors_within_cutoff()` | Biotite `CellList.get_atoms()` | Exact | ⬜ |
| Cell list algorithm | `CellList::query_neighbors()` | Brute force reference | Exact | ⬜ |

### 2.5 Sequence Alignment

| Test | Rust Function | Reference | Tolerance | Status |
|------|--------------|-----------|-----------|--------|
| Smith-Waterman score | `smith_waterman_affine()` | Biotite `align_optimal()` | Exact | ⬜ |
| Needleman-Wunsch score | `needleman_wunsch()` | Biotite `align_optimal()` | Exact | ⬜ |
| BLOSUM62 matrix | `substitution_score()` | Standard BLOSUM62 | Exact | ⬜ |

---

## 3. Coordinate Formatters Parity

### 3.1 Atom37 Format

| Test | Rust Function | Reference | Tolerance | Status |
|------|--------------|-----------|-----------|--------|
| Coordinate ordering | `Atom37Formatter::format()` | Original priox `Protein` | 1e-3 Å | ⬜ |
| Atom mask | atom_mask output | Original priox | Exact | ⬜ |
| Residue type indices | aatype output | Original priox | Exact | ⬜ |

### 3.2 Atom14 Format

| Test | Rust Function | Reference | Tolerance | Status |
|------|--------------|-----------|-----------|--------|
| Coordinate ordering | `Atom14Formatter::format()` | Original priox/AlphaFold | 1e-3 Å | ⬜ |
| Sidechain atoms | SC atom positions | AlphaFold conventions | Exact | ⬜ |

### 3.3 Backbone Format

| Test | Rust Function | Reference | Tolerance | Status |
|------|--------------|-----------|-----------|--------|
| N-CA-C-O ordering | `BackboneFormatter::format()` | Original priox backbone | 1e-3 Å | ⬜ |

---

## 4. Physics Calculations Parity

### 4.1 Electrostatics

| Test | Rust Function | Reference | Tolerance | Status |
|------|--------------|-----------|-----------|--------|
| Coulomb potential | `coulomb_potential()` | Original priox `features.py` | 1e-4 kJ/mol | ⬜ |
| Coulomb forces | `coulomb_forces()` | OpenMM NonbondedForce | 1e-4 kJ/mol/nm | ⬜ |

### 4.2 Van der Waals

| Test | Rust Function | Reference | Tolerance | Status |
|------|--------------|-----------|-----------|--------|
| LJ energy | `lj_energy()` | Original priox `features.py` | 1e-4 kJ/mol | ⬜ |
| LJ forces | `lj_forces()` | OpenMM NonbondedForce | 1e-4 kJ/mol/nm | ⬜ |

### 4.3 Radial Basis Functions

| Test | Rust Function | Reference | Tolerance | Status |
|------|--------------|-----------|-----------|--------|
| RBF expansion | `apply_rbf()` | Original priox `rbf.py` | 1e-5 | ⬜ |
| RBF centers | `generate_rbf_centers()` | Original priox | 1e-5 | ⬜ |

---

## 5. Force Field Integration Parity

### 5.1 XML Parser

| Test | Rust Function | Reference | Tolerance | Status |
|------|--------------|-----------|-----------|--------|
| Atom types | `parse_forcefield_xml()` | OpenMM ForceField | Exact | ✅ |
| Residue templates | residue parsing | OpenMM | Exact | ✅ |
| Nonbonded params | sigma/epsilon | OpenMM | 1e-6 | ✅ |

### 5.2 Topology Generation

| Test | Rust Function | Reference | Tolerance | Status |
|------|--------------|-----------|-----------|--------|
| Bond list | `Topology::from_coords()` | OpenMM Modeller | Exact | ✅ |
| Angle list | angle generation | OpenMM | Exact | ✅ |
| Dihedral list | dihedral generation | OpenMM | Exact | ✅ |

### 5.3 Exclusion Lists

| Test | Rust Function | Reference | Tolerance | Status |
|------|--------------|-----------|-----------|--------|
| 1-2 exclusions | `Exclusions::exclusions_12` | OpenMM | Exact | ✅ |
| 1-3 exclusions | `Exclusions::exclusions_13` | OpenMM | Exact | ✅ |
| 1-4 pairs | `Exclusions::exclusions_14` | OpenMM | Exact | ✅ |

### 5.4 GAFF Atom Typing

| Test | Rust Function | Reference | Tolerance | Status |
|------|--------------|-----------|-----------|--------|
| Atom type assignment | `assign_gaff_types()` | Antechamber | Exact | ✅ |
| LJ parameters | `GaffParameters` | gaff.dat | 1e-4 | ✅ |

---

## 5.5 Trajectory Format Parity

### XTC Format

| Test | Rust Function | Reference | Tolerance | Status |
|------|--------------|-----------|-----------|--------|
| Frame coordinates | `parse_xtc_file()` | MDTraj `load_xtc()` | 1e-3 Å | ⬜ |
| Frame count | num_frames | MDTraj | Exact | ⬜ |
| Unitcell/box | box_vectors | MDTraj | 1e-4 nm | ⬜ |

### DCD Format

| Test | Rust Function | Reference | Tolerance | Status |
|------|--------------|-----------|-----------|--------|
| Frame coordinates | `parse_dcd_file()` | MDTraj `load_dcd()` | 1e-3 Å | ⬜ |
| Header parsing | DcdHeader | MDTraj | Exact | ⬜ |

### TRR Format

| Test | Rust Function | Reference | Tolerance | Status |
|------|--------------|-----------|-----------|--------|
| Frame coordinates | `parse_trr_file()` | MDTraj `load_trr()` | 1e-3 Å | ⬜ |
| Velocities | velocities field | MDTraj | 1e-4 | ⬜ |
| Forces | forces field | MDTraj (if present) | 1e-4 | ⬜ |

---

## 5.6 HDF5 Format Parity

### MDTraj HDF5

| Test | Rust Function | Reference | Tolerance | Status |
|------|--------------|-----------|-----------|--------|
| Coordinates | `parse_mdtraj_h5()` | MDTraj `load()` | 1e-3 Å | ⬜ |
| Topology | chain/residue info | MDTraj | Exact | ⬜ |
| Atom names | atom_names | MDTraj | Exact | ⬜ |

### MDCATH HDF5

| Test | Rust Function | Reference | Tolerance | Status |
|------|--------------|-----------|-----------|--------|
| Coordinates | `parse_mdcath_h5()` | Original priox loader | 1e-3 Å | ⬜ |
| Domain metadata | domain_id, superfamily | Original loader | Exact | ⬜ |

---

## 5.7 Hydrogen Addition Parity

> **Implementation Status:** ✅ COMPLETE & VALIDATED (2024-12-12)
>
> - Fixed fragment library loading (embedded at compile time)
> - Implemented energy relaxation using OpenMM (via Python integration)
> - Added `add_hydrogens_with_relax()` function
> - Exposed via `OutputSpec.relax_hydrogens` flag
> - Validated successful energy relaxation on 5awl.pdb and 1uao.pdb

| Test | Rust Function | Reference | Tolerance | Status |
|------|--------------|-----------|-----------|--------|
| Hydrogen count | `add_hydrogens()` | hydride.add_hydrogen() | Exact | ✅ |
| N-H bond length | geometry check | 1.01 Å ideal | 0.05 Å | ✅ |
| C-H bond length (sp3) | geometry check | 1.09 Å ideal | 0.05 Å | ✅ |
| Backbone H placement | N-H direction | hydride | 10° angle | ✅ |
| Sidechain H placement | all amino acids | hydride | 0.1 Å | ✅ |
| Energy relaxation | `relax_hydrogens()` | OpenMM | 0.1 Å | ✅ |

**Test Cases:**

- Alanine dipeptide (minimal, all H types)
- Glycine (special case: 2 HA)
- Proline (no amide H)
- Charged residues (LYS, ARG, GLU, ASP)
- Aromatic residues (PHE, TYR, TRP)

---

## 6. Test Infrastructure

### 6.1 Test Data

```text
tests/data/
├── pdb/
│   ├── 1crn.pdb      # Small protein (46 residues)
│   ├── 1atp.pdb      # Protein + ATP ligand
│   ├── 2nrl.pdb      # NMR multi-model
│   └── 3htb.pdb      # Complex with solvent
├── mmcif/
│   └── 1crn.cif
├── pqr/
│   └── 1crn.pqr
└── reference/
    ├── 1crn_mdtraj_dihedrals.npy
    ├── 1crn_openmm_energy.json
    └── 1crn_priox_original.json
```

### 6.2 Test Script Template

```python
# tests/validation/test_parity.py

import numpy as np
import priox_rs
import biotite.structure.io.pdb as pdb
import mdtraj

def test_coordinate_parity():
    """Verify Rust coordinates match Biotite and original priox."""
    # Rust
    rust_result = priox_rs.parse_structure("tests/data/pdb/1crn.pdb", spec)
    
    # Biotite reference
    pdb_file = pdb.PDBFile.read("tests/data/pdb/1crn.pdb")
    structure = pdb_file.get_structure()[0]
    biotite_coords = structure.coord
    
    np.testing.assert_allclose(
        rust_result["coordinates"], biotite_coords, atol=1e-3
    )

def test_dihedral_parity():
    """Verify Rust dihedrals match MDTraj."""
    # Reference from MDTraj
    traj = mdtraj.load("tests/data/pdb/1crn.pdb")
    ref_phi = mdtraj.compute_phi(traj)[1]
    
    # Rust implementation
    rust_phi = compute_phi_rust(...)
    
    np.testing.assert_allclose(rust_phi, ref_phi, atol=1e-4)
```

### 6.3 Benchmark Integration

```python
# benchmarks/parity_benchmarks.py

def benchmark_parsing():
    """Compare Rust vs original priox parsing speed with parity check."""
    # Verify same output first
    assert_coordinates_match(rust_result, original_priox_result)
    
    # Then benchmark
    rust_time = timeit(priox_rs.parse_structure, ...)
    python_time = timeit(original_priox_parse, ...)
    
    print(f"Rust: {rust_time:.3f}s, Python: {python_time:.3f}s")
    print(f"Speedup: {python_time/rust_time:.1f}x")
```

---

## 7. CI Integration

### 7.1 GitHub Actions Workflow

```yaml
# .github/workflows/parity_tests.yml
name: Parity Tests

on: [push, pull_request]

jobs:
  parity:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install dependencies
        run: |
          pip install biotite mdtraj openmm numpy scipy
          maturin develop --release
      - name: Run parity tests
        run: pytest tests/validation/ -v
```

### 7.2 Coverage Tracking

Track which parity tests are passing in CI:

```text
✅ PDB parsing (coordinates)
✅ PDB parsing (residues)
✅ Bond inference
⬜ Dihedral angles (pending MDTraj comparison)
⬜ OpenMM energy parity
```

---

## 8. Execution Status

### Phase 1: Core Parsing & Structure (✅ Completed)

- PDB/mmCIF Parsing
- ProcessedStructure Mapping
- Bond Inference

### Phase 2: Hydrogen & Force Fields (✅ Completed)

- Hydrogen placement & relaxation
- Force Field (OpenMM XML) parsing
- GAFF atom typing

### Phase 3: Extended Parity & Format Validation (✅ Done with issues)

- [x] Structure Parity (Atom37/14/Backbone)
- [x] Physics Parity (Dihedrals, RBF, Electrostatics)
- [x] Trajectory Support
  - HDF5: ✅ Verified
  - XTC/DCD/TRR: ⚠️ Blocked by upstream `chemfiles` crash (SIGFPE)

### Phase 4: Optimization & Polish (Next)

- [ ] CustomFormatter (20x speedup)
- [ ] Rayon Parallelism
- [ ] Documentation & Final Review
