# proxide-gaff

> **Alpha — experimental, no API stability guarantees.** This crate is part of an active research project; APIs may change without notice between versions. Not recommended for production use.

General Amber Force Field (GAFF) implementation for protein structures

Part of the [proxide](https://github.com/maraxen/proxide) workspace.

## Overview

`proxide-gaff` implements the General Amber Force Field (GAFF) for small molecules and non-standard residues that are not covered by standard protein force fields. It performs GAFF atom typing based on element, hybridization, ring membership, and local chemical environment, then assigns and looks up bonded/nonbonded parameters from versioned GAFF XML files. It also computes nonbonded exclusion lists (1-2, 1-3, 1-4) from molecular topology.

## Key Types

- `GaffAtomType` — represents a typed GAFF atom with its assigned type string and chemical context
- `GaffParameters` — top-level container for GAFF parameter tables (bonds, angles, torsions, nonbonded)
- `GaffTypeRule` — a single typing rule matching element, hybridization, ring membership, and neighbor count to a GAFF type string
- `GaffAtomTyper` — assigns GAFF atom types to all atoms in a molecule given elements and topology
- `GaffTemplateGenerator` — high-level entry point: loads a versioned GAFF XML file, performs atom typing, and generates residue templates with full parameter assignment
- `GaffError` — error type covering XML parse failures, missing parameters, and typing failures
- `Exclusions` — nonbonded exclusion list (1-2, 1-3, 1-4 pairs) derived from molecular topology
- `AssignedBond` — a bond term with assigned GAFF types and looked-up force constant / equilibrium length
- `AssignedAngle` — an angle term with assigned GAFF types and looked-up force constant / equilibrium angle
- `AssignedDihedral` — a proper or improper dihedral term with GAFF types and periodicity/phase/barrier parameters
- `AssignedNonbonded` — per-atom nonbonded parameters (epsilon, rmin_half) after GAFF type lookup
- `MoleculeParameters` — collected bonded and nonbonded parameters for a complete molecule

## Usage

```rust
use proxide_gaff::{GaffTemplateGenerator, Exclusions};

// Load GAFF 2.11 parameters from the bundled XML
let gen = GaffTemplateGenerator::new("gaff-2.11", None)?;

// Assign GAFF atom types given element symbols and topology
let atom_types = gen.assign_atom_types(&elements, &topology);

// Generate a residue template (e.g., for a ligand named "LIG") with partial charges
let template = gen.generate_template("LIG", &elements, &topology, Some(&charges))?;

// Retrieve all bonded and nonbonded parameters for the molecule
let params = gen.get_molecule_parameters(&elements, &topology);

// Build the nonbonded exclusion list (1-2, 1-3, 1-4)
let excl = Exclusions::from_topology(&topology);
assert!(!excl.is_excluded(0, 5));
assert!(excl.is_14_pair(0, 3));
```

## Dependencies

This crate depends on: proxide-core, proxide-geometry
