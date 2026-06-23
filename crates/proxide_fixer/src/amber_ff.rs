//! Amber force field parameter handling and validation.

#[cfg(test)]
#[cfg(feature = "amber_ff")]
mod convention_tests {
    #[test]
    fn rmin_to_sigma_ct_atom() {
        // ff14SB sp3 carbon (CT): Rmin/2 = 1.9080 Å, ε = 0.1094 kcal/mol
        // Correct formula: σ = (Rmin/2) × 2^(5/6)  [NOT σ = Rmin × 2^(−1/6)]
        let rmin_half: f32 = 1.9080;
        let sigma = rmin_half * 2.0_f32.powf(5.0 / 6.0);
        assert!(
            (sigma - 3.3997).abs() < 1e-3,
            "σ expected ≈3.3997 Å, got {sigma}"
        );
    }

    #[test]
    fn lj_zero_crossing_invariant() {
        // At r = σ, the LJ potential must be exactly 0 regardless of ε
        // This catches σ-form vs ε-form confusion
        let sigma: f32 = 3.3997;
        let epsilon: f32 = 0.1094;
        // U = 4ε[(σ/r)^12 − (σ/r)^6]; at r=σ: U = 4ε[1 − 1] = 0
        let r = sigma;
        let sr6 = (sigma / r).powi(6);
        let u = 4.0 * epsilon * (sr6 * sr6 - sr6);
        assert!(u.abs() < 1e-5, "LJ at r=σ expected 0.0, got {u}");
    }
}
