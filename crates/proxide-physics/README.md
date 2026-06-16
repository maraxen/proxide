# proxide-physics

> **Alpha — experimental, no API stability guarantees.** This crate is part of an active research project; APIs may change without notice between versions. Not recommended for production use.

Physics and forcefield modules for protein structures

Part of the [proxide](https://github.com/maraxen/proxide) workspace.

## Overview

`proxide-physics` assigns force field parameters (charges, Lennard-Jones, GBSA radii, bonded terms, CMAP) to processed protein structures and small molecules, and computes raw physics quantities (Coulomb potentials/forces, LJ forces/energies, SE(3)-invariant electrostatic and vdW features at backbone positions) for use in downstream ML pipelines or MD engines. It bridges `proxide-core`'s `ForceField`/`Topology` types and `proxide-geometry`'s topology inference with per-atom parameter arrays ready for OpenMM or JAX consumption. A secondary concern is providing water model parameters (TIP3P, SPC/E, TIP4P-Ew) for explicit solvent setups.

## Key Types

- **`MDParameters`** — per-atom parameter arrays (charges, sigmas, epsilons, bonds, dihedrals, resolved nonbonded 1-4 params, etc.) produced by parameterization and consumed by MD engines or JAX pipelines
- **`ParamOptions`** — configuration for parameterization: controls terminal cap auto-assignment and missing-residue handling mode
- **`ParamError`** — error type returned by parameterization routines
- **`MissingResidueMode`** — enum governing behavior when a residue is absent from the force field (e.g., `SkipWarn`, `Fail`)
- **`BackboneFrame`** — SE(3) frame derived from backbone atom positions, used as the local coordinate system for invariant feature computation
- **`ProjectedForces`** — forces projected onto backbone frame axes, suitable for rotationally equivariant ML input
- **`WaterModel`** — parameters for explicit-solvent water models (TIP3P, SPC/E, TIP4P-Ew), including geometry and charges

## Usage

```rust
use proxide_physics::{parameterize_structure, parameterize_molecule, ParamOptions};
use proxide_physics::{compute_electrostatic_features, compute_backbone_frame};

// Parameterize a processed protein structure
let options = ParamOptions::default(); // auto_terminal_caps=true, missing_mode=SkipWarn
let md_params = parameterize_structure(&processed_structure, &topology, &force_field, &options)?;
// md_params.charges, .sigmas, .epsilons, .bonds, .dihedrals, .resolved_nonbonded_14_params, etc.

// For small molecules / ligands
let md_params = parameterize_molecule(&coords, &elements, 1.3)?;

// SE(3)-invariant backbone electrostatic features
let features = compute_electrostatic_features(&backbone_5, &all_positions, &all_charges);
```

## Dependencies

This crate depends on: `proxide-core`, `proxide-geometry`, `proxide-gaff`, `proxide-units`.
