//! Unit conversion constants and specifications for proxide parameterizer output.
//!
//! This crate provides conversion factors and unit system specifications for translating
//! from GROMACS-internal units (nm, kJ/mol) to various target unit systems (e.g., AMBER).
//!
//! **Scope**: Applies only to the parameterizer output dict emitted by py_parsers.rs.
//! The proxide-physics internals (kcal/mol + Å) are a separate domain.

pub mod constants;
pub mod phantom;
pub mod registry;

pub use constants::{
    KCAL_TO_KJ, KJ_TO_KCAL, NM_TO_ANGSTROM, ANGSTROM_TO_NM, BOND_K_FACTOR,
};
pub use registry::{UnitSystem, UnitConversionSpec};
