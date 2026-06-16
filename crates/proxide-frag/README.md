# proxide-frag

> **Alpha — experimental, no API stability guarantees.** This crate is part of an active research project; APIs may change without notice between versions. Not recommended for production use.

Backbone fragment search — Kabsch RMSD over fixed-length fragment databases (MASTER-style)

Part of the [proxide](https://github.com/maraxen/proxide) workspace.

## Overview

proxide-frag implements a MASTER-style backbone fragment search engine. It stores fixed-length protein backbone fragments (N, CA, C, O atoms) in an immutable database of centered coordinates, then searches that database in parallel using the Kabsch SVD algorithm to find all fragments within an RMSD threshold of a query fragment. An optional Cauchy-Schwarz norm-bound pre-filter avoids expensive SVD calls for entries that cannot match.

## Key Types

- `Fragment<const N: usize, State>` — a fixed-length backbone fragment parameterized by length and state (raw or centered)
- `Raw` — state marker indicating fragment coordinates have not been centered
- `Centered` — state marker indicating fragment coordinates have been mean-centered
- `BackboneAtom` — enum of backbone atom positions: N=0, CA=1, C=2, O=3
- `FragmentDb<const N: usize>` — immutable, searchable database of centered fragments
- `FragmentDbBuilder<const N: usize>` — builder that accumulates raw fragments and constructs a `FragmentDb`
- `SourceLabel` — label attached to each database entry to identify its origin structure
- `KabschResult` — output of the Kabsch SVD alignment, including RMSD
- `SearchResult` — a database hit: source label, index, and RMSD to the query
- `PersistError` — error type for save/load operations
- `AlreadyCenteredError` — error returned when centering is attempted on an already-centered fragment

## Usage

```rust
use proxide_frag::{Fragment, FragmentDb, FragmentDbBuilder, SourceLabel};

// Build a database of 5-residue fragments (4 atoms per residue = 20 atoms x 3 coords)
let mut builder = FragmentDbBuilder::<5>::new();

// Add raw fragments with source labels
let coords: [[f32; 3]; 20] = /* backbone N, CA, C, O coords for 5 residues */ todo!();
let frag = Fragment::<5, _>::new(coords);
let label = SourceLabel::new("1abc", 0);
builder.add_fragment(frag, label).expect("centering failed");

let db = builder.build();

// Save to disk and reload
db.save(std::path::Path::new("frags.bin")).expect("save failed");
let db = FragmentDb::<5>::load(std::path::Path::new("frags.bin")).expect("load failed");

// Search: center a query fragment, then search within epsilon Angstroms RMSD
let query_coords: [[f32; 3]; 20] = todo!();
let query_raw = Fragment::<5, _>::new(query_coords);
let (query, _centroid) = query_raw.center().expect("centering failed");

// Use search_prefiltered() for large databases to skip SVD via norm-bound pre-filter
let hits = db.search_prefiltered(&query, 1.0); // epsilon = 1.0 Å
for result in &hits {
    println!("Hit: {:?} RMSD={:.3}", result.label, result.rmsd);
}
```
