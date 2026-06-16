# proxide-units

> **Alpha — experimental, no API stability guarantees.** This crate is part of an active research project; APIs may change without notice between versions. Not recommended for production use.

Unit conversion constants and specifications for proxide parameterizer output

Part of the [proxide](https://github.com/maraxen/proxide) workspace.

## Overview

`proxide-units` provides unit conversion constants and structured conversion specs for translating parameterizer output from GROMACS-internal units (nm, kJ/mol) to target unit systems such as AMBER (Å, kcal/mol), GROMACS (identity), and OpenMM (nm, kJ/mol). Its scope is explicitly limited to the parameterizer output dict emitted by `py_parsers.rs`; the proxide-physics internals (kcal/mol + Å) are a separate domain. The crate has no internal proxide dependencies.

## Key Types

- `UnitSystem` — enum with variants `Amber`, `Gromacs`, and `OpenMM`; derives `Default` (= `Amber`)
- `UnitConversionSpec` — struct holding scalar conversion factors: `length: f32`, `energy: f32`, `bond_k: f32`, `angle_k: f32`

## Usage

```rust
use proxide_units::UnitSystem;

let spec = UnitSystem::Amber.conversion_spec();
let length_angstrom = length_nm * spec.length;
let energy_kcal = energy_kj * spec.energy;
```
