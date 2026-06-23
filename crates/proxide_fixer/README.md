# proxide_fixer

> **Alpha — experimental, no API stability guarantees.** This crate is part of an active research project; APIs may change without notice between versions. Not recommended for production use.

Protein structure sanitization and repair for proxide

Part of the [proxide](https://github.com/maraxen/proxide) workspace.

## Overview

`proxide_fixer` provides protein structure sanitization and repair operations, including missing-atom detection against residue templates, hydrogen placement, steric clash resolution, chain-terminus capping, and stereochemistry validation. It operates on a hierarchical Topology model (Chain → Residue → Atom) and bridges to `proxide_rs` `AtomicSystem` for downstream physics calculations. Sanitization passes are gated behind optional Cargo features (`protonation`, `capping`, `stereo`) and exposed through the `Sanitizer` trait.

## Key Types

- `Topology` — hierarchical protein structure model (Chain → Residue → Atom); primary entry point for sanitization
- `Chain` — a single polypeptide chain within a Topology
- `Residue` — a single amino acid residue within a Chain
- `Atom` — an individual atom within a Residue, carrying position and element data
- `MissingAtom` — describes an atom present in a residue template but absent from the parsed structure
- `ResidueLibrary` — collection of residue templates used for missing-atom detection and hydrogen placement
- `ResidueTemplate` — canonical definition of a residue's atoms and connectivity
- `TemplateAtom` — a single atom entry within a ResidueTemplate
- `Builder` — utilities for mapping SEQRES records and resolving steric clashes
- `CappingSanitizer` — applies N/C-terminal capping groups to chain termini (requires `capping` feature)
- `CappingError` — error type returned by CappingSanitizer
- `ProtonationSanitizer` — adds or removes hydrogens according to a protonation strategy (requires `protonation` feature)
- `ProtonationError` — error type returned by ProtonationSanitizer
- `ProtonationStrategy` — enum controlling pH-dependent or fixed protonation behavior
- `StereoSantizer` — validates and corrects chiral-center stereochemistry (requires `stereo` feature)
- `Sanitizer` (trait) — unified interface implemented by all sanitizer types

## Feature Flags

| Flag | Description |
|------|-------------|
| `protonation` | Enables `ProtonationSanitizer` and `Sanitizer::protonate`; gates the `sanitizers::protonation` module |
| `capping` | Enables `CappingSanitizer` for N/C-terminal capping and `Sanitizer::cap`; gates the `sanitizers::capping` module |
| `stereo` | Enables `StereoSantizer` for chiral-center validation and `Sanitizer::fix_stereo`; gates the `sanitizers::stereo` module |

## Optional External Dependencies

### Modeller (loop modeling, C10)

Loop modeling via C10 requires **Modeller** — an external tool not bundled with proxide.

- Set `MODELLER_EXEC` to the Modeller executable path, or add it to `PATH`
- Set `MODELLER_KEY` to your Modeller license key
- Without these, `SystemPrepConfig { model_loops: true, .. }` returns `LoopModellingError::ModelerNotInstalled` or `MissingLicenseKey`
- Download Modeller: https://salilab.org/modeller/

## Usage

```rust
use proxide_fixer::{Topology, ResidueLibrary, Builder};
use proxide_fixer::sanitizers::capping::CappingSanitizer;

// Build topology from raw parsed atom data
let mut topology = Topology::from_raw_atom_data(&raw);

// Detect missing atoms against the standard residue library
let library = ResidueLibrary::new_standard();
let missing = topology.find_missing_atoms(&library);

// Cap chain termini (requires `capping` feature)
let mut cap = CappingSanitizer::new(&mut topology);
cap.run().unwrap();

// Convert to AtomicSystem for downstream physics
let mut system = topology.to_atomic_system();

// Resolve steric clashes with a 1.5 Å radius
Builder::resolve_clashes(&mut system, 1.5);
```

## Dependencies

This crate depends on: `proxide_rs`, `proxide-geometry`
