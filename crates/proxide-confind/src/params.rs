//! Physical constants, amino acid propensities, and parameters for ConFind.
//!
//! Citations:
//! - Zheng, F., Zhang, J., & Grigoryan, G. (2015). "Tertiary Structural Propensities Reveal
//!   Fundamental Sequence/Structure Relationships in Proteins." *Structure*, 23(5), 961–971.
//! - Zheng, W., & Grigoryan, G. (2017). "Sequence statistics of tertiary structural motifs
//!   reflect protein stability." *PLoS ONE*, 12(5), e0178272.

/// Heavy-atom clash distance threshold between sidechain and backbone atoms (Å).
/// Sidechains within 2.0 Å of the backbone represent steric clash and are pruned.
pub const CLASH_DIST: f64 = 2.0;

/// Heavy-atom contact distance threshold between sidechain atoms (Å).
/// Sidechains within 3.0 Å are considered to be in steric contact.
pub const CONT_DIST: f64 = 3.0;

/// Backbone neighbor search cutoff distance between Cα atoms (Å).
/// Pairs separated by more than 25.0 Å cannot physically form sidechain contacts.
pub const DCUT: f64 = 25.0;

/// Low-collision probability threshold for structural freedom calculation.
pub const LO_COLL_PROB_CUT: f64 = 0.5;

/// High-collision probability threshold for structural freedom calculation.
pub const HI_COLL_PROB_CUT: f64 = 2.0;

/// The 18 canonical amino acids with discrete rotameric sidechain degrees of freedom,
/// excluding Glycine (no Cβ/sidechain) and Proline (covalent ring constraint).
pub const AA_NAMES: &[&str] = &[
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "HIS", "ILE", "LEU", "LYS", "MET", "PHE",
    "SER", "THR", "TRP", "TYR", "VAL",
];

/// Background empirical amino acid propensity P(a) in percent from Zheng et al. (2015) Table S1.
///
/// Returns percentage values (e.g., ALA = 7.73%).
pub fn aa_propensity(aa: &str) -> f64 {
    match aa {
        "ALA" => 7.73,
        "ARG" => 5.03,
        "ASN" => 4.50,
        "ASP" => 5.82,
        "CYS" | "CYX" => 1.84,
        "GLN" => 3.94,
        "GLU" => 6.61,
        "GLY" => 7.11,
        "HIS" | "HSD" | "HID" | "HIE" | "HIP" => 2.35,
        "ILE" => 5.66,
        "LEU" => 8.83,
        "LYS" => 6.27,
        "MET" | "MSE" => 2.08,
        "PHE" => 4.05,
        "PRO" => 4.52,
        "SER" => 6.13,
        "THR" => 5.53,
        "TRP" => 1.51,
        "TYR" => 3.54,
        "VAL" => 6.91,
        _ => panic!("no propensity for amino acid {aa}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aa_propensity_sum() {
        let standard_20 = [
            "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS",
            "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
        ];
        let total: f64 = standard_20.iter().map(|&aa| aa_propensity(aa)).sum();
        // Zheng et al. Table S1 values sum to 99.96% (~100% within rounding)
        assert!((total - 99.96).abs() < 1e-4);
    }

    #[test]
    fn test_constants() {
        assert_eq!(CLASH_DIST, 2.0);
        assert_eq!(CONT_DIST, 3.0);
        assert_eq!(DCUT, 25.0);
        assert_eq!(LO_COLL_PROB_CUT, 0.5);
        assert_eq!(HI_COLL_PROB_CUT, 2.0);
        assert_eq!(AA_NAMES.len(), 18);
    }
}
