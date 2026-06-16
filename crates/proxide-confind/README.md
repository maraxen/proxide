# proxide-confind

> **Alpha — experimental, no API stability guarantees.** This crate is part of an active research project; APIs may change without notice between versions. Not recommended for production use.

ConFind contact-degree algorithm — Rust/orx-parallel reimplementation of Mosaist mstcondeg

Part of the [proxide](https://github.com/maraxen/proxide) workspace.

## Overview

`proxide-confind` implements the ConFind contact-degree metric for protein residue pairs, porting Mosaist's `mstcondeg` to Rust with orx-parallel. Given a protein backbone and a rotamer library, it runs a three-phase pipeline: Phase A prunes backbone-clashing rotamers per residue (crowdedness), Phase B enumerates cross-rotamer clashes in parallel to compute collision probabilities per residue pair, and Phase C aggregates those probabilities into a scalar rotamer-freedom value per residue. Contact-degree values are used downstream (e.g., by dTERMen) to identify residue pairs that are "poised to interact".

## Key Types

- `ConFind` — main entry point; holds the rotamer library, protein backbone, and all cached phase results
- `ResidueIndex` — strongly-typed index into the backbone residue array
- `ResidueBackbone` — per-residue backbone geometry (coordinates, amino-acid identity)
- `ProteinBackbone` — full-protein backbone container; constructed from PDB input
- `ContactList` — collection of residue-pair contact-degree values with filtering and lookup utilities
- `ConFindError` — error type covering library mismatches, missing residues, and I/O failures

## Feature Flags

| Flag | Description |
|------|-------------|
| `serde` | Derives serde `Serialize`/`Deserialize` on public types via the optional serde dependency |

## Usage

```rust
use std::path::Path;
use std::sync::Arc;
use proxide_rotlib::RotamerLibrary;
use proxide_confind::{ConFind, ResidueIndex, load_pdb_f64};

const CONTACT_THRESHOLD: f64 = 0.02; // canonical dTERMen threshold

let rotlib = Arc::new(RotamerLibrary::load(Path::new("rotlib.bin")).unwrap());
let backbone = Arc::new(load_pdb_f64(Path::new("protein.pdb")).unwrap());
let cf = ConFind::new(rotlib, backbone, false);

let residues = vec![ResidueIndex(0), ResidueIndex(1)];
let contacts = cf.contacts(&residues, 0.0).unwrap();
let filtered = contacts.filter(CONTACT_THRESHOLD);
```

## Dependencies

This crate depends on: proxide-core, proxide-geometry, proxide-io, proxide-rotlib.
