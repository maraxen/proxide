# proxide_rs

> **Alpha — experimental, no API stability guarantees.** This crate is part of an active research project; APIs may change without notice between versions. Not recommended for production use.

High-performance Rust parsing core for protein structures

Part of the [proxide](https://github.com/maraxen/proxide) workspace.

## Overview

`proxide_rs` is the top-level aggregation crate that re-exports the full proxide library surface — core chemistry/structure types, unit systems, geometry algorithms, physics, I/O formats, and GAFF force-field data — through a single namespace. It also provides higher-level processing logic not present in the sub-crates: hydrogen placement via fragment library + Kabsch superimposition, hydrogen position relaxation via UFF energy minimization, protonation state determination (HIS/CYS/ASP/GLU/LYS with pH and structural heuristics), multi-model NMR/trajectory splitting, coordinate noising for data augmentation, and projection of atomic structures to MPNN batch feature tensors for protein design models.

## Key Types

- `RotatableGroup` — represents a rotatable bond group used during hydrogen relaxation
- `RelaxOptions` — configuration for UFF energy minimization during hydrogen relaxation
- `EnergyMinimizer` — UFF-based energy minimizer for local hydrogen position optimization
- `MPNNBatchResult` — output tensor batch in ProteinMPNN feature format
- `ProjectionError` — error type for MPNN batch projection failures
- `FragmentLibrary` (re-exported via `proxide_geometry`) — fragment library for hydrogen placement via Kabsch superimposition
- `ProcessedStructure` (re-exported via `proxide_core`) — fully parsed and annotated protein structure
- `UnitSystem` (re-exported via `proxide_units`) — unit system definitions for physical quantities

## Feature Flags

| Flag | Description |
|------|-------------|
| `default` | Enables `xtc` and `parallel` features |
| `fetching` | Enables remote PDB/structure fetching via reqwest; gates `proxide_io::io` re-export |
| `hdf5` | Enables HDF5 trajectory format support via proxide-io |
| `foldcomp` | Enables FoldComp compressed structure format support via proxide-io |
| `xtc` | Enables XTC trajectory format support via proxide-io (on by default) |
| `parallel` | Enables parallel I/O processing via proxide-io (on by default) |
| `full` | Enables all optional features: `fetching`, `hdf5`, `foldcomp`, `xtc`, `parallel` |

## Usage

```rust
use proxide_rs::{
    geometry::hydrogens::init_fragment_library,
    add_hydrogens_with_relax,
    project_to_mpnn_batch,
};

// Call once at startup — avoids GIL deadlock in PyO3 bindings
init_fragment_library();

// Add and relax hydrogens in one step
// Returns (num_added, num_relaxed, final_energy)
let (num_added, num_relaxed, energy) =
    add_hydrogens_with_relax(&mut structure, &mut bonds, true, None)?;

// Project to ProteinMPNN-style batched feature tensors
// with optional Gaussian coordinate noise (std=0.1 Å) and physics features
let batch = project_to_mpnn_batch(&structure, 30, Some(0.1), seed, true)?;
```

## Dependencies

This crate depends on: proxide-core, proxide-geometry, proxide-physics, proxide-gaff, proxide-io, proxide-units
