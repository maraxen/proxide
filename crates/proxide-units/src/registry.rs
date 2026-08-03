use crate::constants::{BOND_K_FACTOR, KJ_TO_KCAL, NM_TO_ANGSTROM};

/// Conversion factors from GROMACS-internal units (nm, kJ/mol) to a target unit system.
///
/// Scope: applies only to the parameterizer output dict emitted by py_parsers.rs.
/// proxide-physics internals (kcal/mol + Å) are a separate domain and out of scope.
///
/// # Future
/// A full graph-based registry (BFS transitive derivation) is planned for custom unit systems.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UnitConversionSpec {
    /// Factor: nm → target length unit
    pub length: f32,
    /// Factor: kJ/mol → target energy unit
    pub energy: f32,
    /// Factor: kJ/mol/nm² → target bond force constant unit
    pub bond_k: f32,
    /// Factor: kJ/mol/rad² → target angle force constant unit
    pub angle_k: f32,
}

impl UnitConversionSpec {
    /// Create conversion spec for AMBER units (Å + kcal/mol).
    pub fn amber() -> Self {
        Self {
            length: NM_TO_ANGSTROM,
            energy: KJ_TO_KCAL,
            bond_k: BOND_K_FACTOR,
            angle_k: KJ_TO_KCAL,
        }
    }

    /// Create conversion spec for GROMACS units (nm + kJ/mol).
    /// This is the identity conversion (no-op).
    pub fn gromacs() -> Self {
        Self {
            length: 1.0,
            energy: 1.0,
            bond_k: 1.0,
            angle_k: 1.0,
        }
    }

    /// Create conversion spec for OpenMM units (nm + kJ/mol).
    /// OpenMM uses nm/kJ internally — same as GROMACS for parameterizer output.
    pub fn openmm() -> Self {
        Self::gromacs()
    }
}

/// Unit system for parameterizer output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UnitSystem {
    /// Angstroms + kcal/mol (default, AMBER convention)
    #[default]
    Amber,
    /// Nanometers + kJ/mol (GROMACS convention)
    Gromacs,
    /// Nanometers + kJ/mol (OpenMM internal convention)
    OpenMM,
}

impl UnitSystem {
    /// Get the conversion spec for this unit system.
    pub fn conversion_spec(&self) -> UnitConversionSpec {
        match self {
            Self::Amber => UnitConversionSpec::amber(),
            Self::Gromacs => UnitConversionSpec::gromacs(),
            Self::OpenMM => UnitConversionSpec::openmm(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gromacs_identity() {
        let spec = UnitConversionSpec::gromacs();
        assert_eq!(spec.length, 1.0);
        assert_eq!(spec.energy, 1.0);
        assert_eq!(spec.bond_k, 1.0);
        assert_eq!(spec.angle_k, 1.0);
    }

    #[test]
    fn test_amber_length() {
        let spec = UnitConversionSpec::amber();
        assert_eq!(spec.length, 10.0);
    }

    #[test]
    fn test_amber_energy() {
        let spec = UnitConversionSpec::amber();
        let expected = KJ_TO_KCAL;
        let diff = (spec.energy - expected).abs();
        assert!(
            diff < 1e-8,
            "AMBER energy should be KJ_TO_KCAL; got {}",
            spec.energy
        );
    }

    #[test]
    fn test_unit_system_amber_conversion_spec() {
        let spec = UnitSystem::Amber.conversion_spec();
        assert_eq!(spec.length, 10.0);
    }

    #[test]
    fn test_unit_system_gromacs_conversion_spec() {
        let spec = UnitSystem::Gromacs.conversion_spec();
        assert_eq!(spec.length, 1.0);
    }

    #[test]
    fn test_unit_system_openmm_conversion_spec() {
        let spec = UnitSystem::OpenMM.conversion_spec();
        assert_eq!(spec.length, 1.0);
    }

    #[test]
    fn test_openmm_same_as_gromacs() {
        let openmm_spec = UnitConversionSpec::openmm();
        let gromacs_spec = UnitConversionSpec::gromacs();
        assert_eq!(openmm_spec, gromacs_spec);
    }
}
