# proxide-core

> **Alpha — experimental, no API stability guarantees.** This crate is part of an active research project; APIs may change without notice between versions. Not recommended for production use.

Foundational data structures for protein parsing

Part of the [proxide](https://github.com/maraxen/proxide) workspace.

## Overview

proxide-core provides the foundational data structures, chemistry constants, and force field types used throughout the proxide protein parsing stack. It defines the raw and processed atom/residue representations (matching biotite's AtomArray format), the full molecular topology layer (bonds, angles, dihedrals), and an OpenMM-style XML force field parser, together with output specification enums that control coordinate formatting, hydrogen handling, and MD parameterization.

## Key Types

- `RawAtomData` — flat columnar store for raw parsed atom records before grouping
- `AtomRecord` — a single atom record as parsed from a structure file
- `AtomicSystem` — coordinate array with atom metadata; supports noise injection and coordinate updates
- `AtomicSystemArgs` — builder arguments for constructing an `AtomicSystem`
- `ProcessedStructure` — residue-grouped view of a structure derived from `RawAtomData`
- `ResidueInfo` — metadata for a single residue (name, chain, insertion code, sequence number)
- `ResidueId` — unique identifier for a residue within a structure
- `LigandInfo` — metadata for small-molecule ligands
- `OutputSpec` — controls downstream coordinate format, hydrogen handling, and physics feature computation
- `CoordFormat` — enum of coordinate representations (Atom37, Atom14, Full, BackboneOnly)
- `OutputFormatTarget` — target output format for coordinate serialization
- `ErrorMode` — controls how parsing errors are surfaced (fail-fast vs. accumulate)
- `MissingResidueMode` — policy for handling gaps in residue numbering
- `HydrogenSource` — source for hydrogen coordinates (explicit, inferred, or force field)
- `ForceField` — parsed OpenMM XML force field with indexed residue templates and atom types
- `AtomType` — force field atom type with mass, charge, and LJ parameters
- `ResidueTemplate` — force field residue definition with atoms, bonds, and external bonds
- `ResidueAtom` — atom entry within a residue template
- `HarmonicBondParam` — harmonic bond stretch parameters (k, r0)
- `HarmonicAngleParam` — harmonic angle bend parameters (k, theta0)
- `ProperTorsionParam` — proper dihedral torsion parameters with periodicity terms
- `ImproperTorsionParam` — improper dihedral parameters for planarity restraints
- `TorsionTerm` — a single periodicity term within a torsion parameter
- `NonbondedParam` — nonbonded (LJ + charge) parameters for an atom type
- `NonbondedException` — explicit 1-4 exception overrides for nonbonded interactions
- `GBSAOBCParam` — Generalized Born / Surface Area (OBC variant) solvation parameters
- `CMAPData` — container for all CMAP correction grids in a force field
- `CMAPGrid` — a single CMAP correction energy grid
- `CMAPTorsion` — atom-type pattern defining which residue torsions use a CMAP grid
- `Bond` — bonded atom pair with bond order
- `Angle` — three-atom angle in the molecular topology
- `Dihedral` — four-atom dihedral in the molecular topology
- `ParseError` — error type for structure and force field parsing failures

## Feature Flags

| Flag | Description |
|------|-------------|
| `serde` | Enables serde Serialize/Deserialize derives on core data types |
| `testing` | Exposes the testing module (also enabled under `#[cfg(test)]`) |

## Usage

```rust
use proxide_core::{
    AtomRecord, RawAtomData, ProcessedStructure,
    parse_forcefield_xml, OutputSpec, CoordFormat,
};

// Build a raw atom store by appending parsed records
let mut raw = RawAtomData::with_capacity(1024);
for record in parsed_records {
    raw.add_atom(record);
}

// Group atoms into residues
let structure = ProcessedStructure::from_raw(raw);

// Load an OpenMM XML force field
let ff = parse_forcefield_xml("amber14-all.xml")?;
let template = ff.get_residue("ALA").expect("ALA not in force field");
let atom_type = ff.get_atom_type("CT").expect("atom type not found");

// Control output format
let spec = OutputSpec {
    coord_format: CoordFormat::Atom37,
    ..Default::default()
};
```

## Dependencies

This crate depends on: proxide-units
