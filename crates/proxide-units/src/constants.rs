/// Energy conversion: kcal/mol → kJ/mol
pub const KCAL_TO_KJ: f32 = 4.184;

/// Energy conversion: kJ/mol → kcal/mol
pub const KJ_TO_KCAL: f32 = 1.0 / 4.184;

/// Length conversion: nanometers → angstroms
pub const NM_TO_ANGSTROM: f32 = 10.0;

/// Length conversion: angstroms → nanometers
pub const ANGSTROM_TO_NM: f32 = 0.1;

/// Bond force constant conversion factor: kJ/mol/nm² → kcal/mol/Å²
/// Combines energy (kJ/mol → kcal/mol) and length (nm² → Å²) conversions.
pub const BOND_K_FACTOR: f32 = KJ_TO_KCAL / (NM_TO_ANGSTROM * NM_TO_ANGSTROM);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_energy_roundtrip() {
        let val = 2.5;
        let roundtrip = (val * KCAL_TO_KJ * KJ_TO_KCAL - val).abs();
        assert!(roundtrip < 1e-5, "Energy roundtrip should be precise; got {}", roundtrip);
    }

    #[test]
    fn test_bond_k_factor_sanity() {
        // Expected value: KJ_TO_KCAL / 100.0 ≈ 0.002390057
        let expected = KJ_TO_KCAL / 100.0;
        let diff = (BOND_K_FACTOR - expected).abs();
        assert!(diff < 1e-8, "BOND_K_FACTOR should be ~0.002390057; got {}", BOND_K_FACTOR);
    }
}
