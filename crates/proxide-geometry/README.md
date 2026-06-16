# proxide-geometry

> **Alpha — experimental, no API stability guarantees.** This crate is part of an active research project; APIs may change without notice between versions. Not recommended for production use.

Geometric algorithms for protein structures

Part of the [proxide](https://github.com/maraxen/proxide) workspace.

## Overview

proxide-geometry provides a collection of pure-Rust geometric algorithms for protein structure analysis. It covers distance calculations, backbone dihedral angles (phi/psi/omega), spatial neighbor search via cell lists and k-NN, radial basis function encoding for ML feature extraction, coordinate transformations, bond inference via covalent radii, sequence alignment (Smith-Waterman / Needleman-Wunsch), solvent/ion filtering, hydrogen position estimation via a fragment library with Kabsch superimposition, and the NeRF algorithm for placing atoms from internal coordinates.

## Key Types

- `AlignmentResult` — result of a pairwise sequence alignment (Smith-Waterman or Needleman-Wunsch), including aligned sequences and score
- `BackboneDihedrals` — per-residue phi/psi/omega dihedral angles in f32
- `BackboneDihedrals64` — per-residue phi/psi/omega dihedral angles in f64
- `CellList` — spatial index for fast neighbor lookup within a cutoff distance
- `FragmentKey` — lookup key for a hydrogen fragment in the fragment library
- `Fragment` — a single hydrogen placement fragment (center + heavy-atom geometry)
- `FragmentLibrary` — binary-encoded library of hydrogen placement fragments, loadable via `from_binary`
- `RBFResult` — radial basis function encoding output including feature vectors and shape metadata
- `Nerf` — NeRF (Natural Extension Reference Frame) atom placer; constructs Cartesian coordinates from internal coordinates

## Feature Flags

| Flag | Description |
|------|-------------|
| `serde` | Derives serde Serialize/Deserialize for public types and enables proxide-core/serde; off by default |

## Usage

```rust
use proxide_geometry::{
    compute_backbone_dihedrals, find_k_nearest_neighbors,
    compute_radial_basis_with_shape, CellList, Nerf,
    FragmentLibrary, smith_waterman_affine,
};

// Backbone dihedrals from N/CA/C triples:
let dihedrals = compute_backbone_dihedrals(&backbone_coords);

// KNN for RBF features:
let neighbors = find_k_nearest_neighbors(&ca_coords, 16);
let rbf = compute_radial_basis_with_shape(&backbone5_coords, &neighbors);

// Bond inference via covalent radii + cell list:
let bonds = infer_bonds(&coords, &elements, 1.3);

// Hydrogen placement via fragment library:
let lib = FragmentLibrary::from_binary(include_bytes!("...")).unwrap();
let h_positions = calculate_hydrogen_positions(&lib, "C", 0, 0, vec![1,1], center, &heavy);
```

## Dependencies

This crate depends on: proxide-core
