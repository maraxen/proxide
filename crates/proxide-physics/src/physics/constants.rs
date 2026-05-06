//! Physical constants for molecular simulations
//!
//! Port of proxide/physics/constants.py
//!
//! Note: These constants are for physics engine internals (kcal/mol + Å domain).
//! For parameterizer output unit conversions (GROMACS nm/kJ → target), see proxide-units.

#![allow(dead_code)]

/// Coulomb constant for electrostatics (kcal/mol·Å·e⁻²)
/// Standard value for protein force fields
pub const COULOMB_CONSTANT_KCAL: f32 = 332.0636;

/// Coulomb constant in atomic units (Hartree)
pub const COULOMB_CONSTANT_ATOMIC: f32 = 1.0;

/// Default Coulomb constant (kcal/mol units)
pub const COULOMB_CONSTANT: f32 = COULOMB_CONSTANT_KCAL;

/// Boltzmann constant in kcal/(mol·K) (Molar Gas Constant R)
pub const BOLTZMANN_KCAL: f32 = 0.0019872;

/// Minimum distance to avoid division by zero (Angstroms)
pub const MIN_DISTANCE: f32 = 1e-7;

/// Maximum force magnitude (kcal/mol/Å) for clamping
pub const MAX_FORCE: f32 = 1e6;

// Unit conversions (re-exported from proxide-units — single source of truth)
pub use proxide_units::constants::{KCAL_TO_KJ, KJ_TO_KCAL, NM_TO_ANGSTROM, ANGSTROM_TO_NM};

// Lennard-Jones defaults

/// Default sigma (Angstroms) - typical for carbon
pub const DEFAULT_SIGMA: f32 = 3.5;

/// Default epsilon (kcal/mol) - typical for nonpolar atoms
pub const DEFAULT_EPSILON: f32 = 0.1;

// Generalized Born constants

/// Dielectric constant of water at 298K
pub const DIELECTRIC_WATER: f32 = 78.5;

/// Dielectric constant inside protein
pub const DIELECTRIC_PROTEIN: f32 = 1.0;

/// Probe radius (Angstroms) for solvent-accessible surface
pub const PROBE_RADIUS: f32 = 1.4;

/// Surface tension (kcal/mol/Å²) - matches OpenMM OBC2 default
pub const SURFACE_TENSION: f32 = 0.0054;

/// Dielectric offset (Angstroms) - matches OpenMM OBC2 default
pub const DIELECTRIC_OFFSET: f32 = 0.009;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coulomb_constant() {
        // Should be close to standard literature value
        assert!((COULOMB_CONSTANT_KCAL - 332.0).abs() < 1.0);
    }

    #[test]
    fn test_unit_conversions() {
        // Round trip conversion
        let val = 10.0;
        let converted = val * KCAL_TO_KJ * KJ_TO_KCAL;
        assert!((converted - val).abs() < 1e-6);
    }
}
