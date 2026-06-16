# proxide-io

> **Alpha — experimental, no API stability guarantees.** This crate is part of an active research project; APIs may change without notice between versions. Not recommended for production use.

IO and parsing modules for protein structures

Part of the [proxide](https://github.com/maraxen/proxide) workspace.

## Overview

proxide-io provides parsers for all major protein structure and trajectory file formats (PDB, mmCIF, DCD, XTC, TRR, FASTA/A3M, Newick, Foldcomp, mdcath/mdtraj HDF5, PQR) and coordinate formatters that convert a ProcessedStructure into standardized array representations (Atom37, Atom14, Backbone, Full). It also includes an optional HTTP fetching layer for downloading structures from RCSB, AlphaFold DB, and mdcath, and an LRU cache to avoid re-formatting the same structure multiple times.

## Key Types

- `FormattedAtom37` — Atom37 coordinate array output from the Atom37 formatter
- `FormattedAtom14` — Atom14 coordinate array output from the Atom14 formatter
- `FormattedBackbone` — Backbone coordinate array output from the Backbone formatter
- `FormattedFull` — Full coordinate array output from the Full formatter
- `Atom37Formatter` — Formats a ProcessedStructure into Atom37 representation
- `Atom14Formatter` — Formats a ProcessedStructure into Atom14 representation
- `BackboneFormatter` — Formats a ProcessedStructure into Backbone representation
- `FullFormatter` — Formats a ProcessedStructure into Full coordinate representation
- `CacheKey` — Key type for the LRU format cache
- `CachedStructure` — Cached formatted structure value stored in the LRU cache
- `CachedStructureArgs` — Arguments used to construct or look up a cached structure
- `FormatCache` — LRU cache mapping CacheKey to CachedStructure
- `XtcTrajectory` — XTC trajectory reader
- `DcdTrajectory` — DCD trajectory reader
- `DcdHeader` — Header parsed from a DCD trajectory file
- `DcdFrame` — Single frame read from a DCD trajectory
- `FrameWithBox` — Trajectory frame bundled with periodic box vectors
- `DcdError` — Error type for DCD parsing failures
- `BackboneChain` — Backbone atom chain extracted from a parsed structure
- `TokenizedMSA` — Tokenized multiple sequence alignment from FASTA/A3M
- `TreeArrays` — Array representation of a parsed Newick phylogenetic tree
- `FastaError` — Error type for FASTA/A3M parsing failures
- `NewickError` — Error type for Newick parsing failures
- `IOParseError` — General IO parse error (ssbond feature)
- `CIFRegistry` — Registry for CIF-based disulfide bond records (ssbond feature)
- `MdcathDomain` (hdf5) — Domain-level metadata from an mdcath HDF5 file
- `MdcathFrame` (hdf5) — Single frame from an mdcath HDF5 trajectory
- `MdtrajFrame` (hdf5) — Single frame from an mdtraj HDF5 trajectory
- `MdtrajH5Result` (hdf5) — Full result parsed from an mdtraj HDF5 file

## Feature Flags

| Flag | Description |
|------|-------------|
| `pdb` | Enables PDB and mmCIF parsers, all formatters (Atom37, Atom14, Backbone, Full); included in default |
| `cif` | CIF-related parsing gate; included in default |
| `ssbond` | Enables IOParseError and CIFRegistry for disulfide-bond CIF processing |
| `xtc` | XTC trajectory parsing via the molly crate; included in default |
| `parallel` | Parallel processing via orx-parallel and proxide-parallel-rt; included in default |
| `foldcomp` | Foldcomp compressed structure format via flate2 |
| `fetching` | HTTP downloading of structures from RCSB, AlphaFold DB, and mdcath via reqwest |
| `hdf5` | HDF5-based trajectory formats: mdcath_h5 (MdcathDomain/MdcathFrame) and mdtraj_h5 (MdtrajFrame/MdtrajH5Result) |
| `mmap` | Memory-mapped file access via memmap2 |
| `full` | Meta-feature enabling all optional features: pdb, cif, ssbond, xtc, parallel, foldcomp, fetching, hdf5, mmap |

## Usage

```rust
// Parse a PDB file and format as Atom37
let (raw, model_ids) = proxide_io::formats::pdb::parse_pdb_file("1abc.pdb")?;
// ... process into ProcessedStructure via proxide-core ...
let atom37 = proxide_io::formatters::Atom37Formatter::format(&processed, &spec)?;
// atom37.coordinates is a flat Vec<f32> of shape (N_res * 37 * 3)
```

## Dependencies

This crate depends on: proxide-core, proxide-geometry, proxide-parallel-rt
