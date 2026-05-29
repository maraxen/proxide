pub const DCUT: f64 = 25.0;
pub const CLASH_DIST: f64 = 2.0;
pub const CONT_DIST: f64 = 3.0;
pub const LO_COLL_PROB: f64 = 0.5;
pub const HI_COLL_PROB: f64 = 2.0;

/// 18 amino acids placed as rotamers (GLY and PRO excluded).
pub const AA_NAMES: [&str; 18] = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "HIS", "ILE", "LEU", "LYS", "MET", "PHE",
    "SER", "THR", "TRP", "TYR", "VAL",
];

/// Background propensity (percent) matching Mosaist's `aaProp` table.
pub fn aa_propensity(aa: &str) -> f64 {
    match aa {
        "ALA" => 7.73,
        "ARG" => 5.03,
        "ASN" => 4.50,
        "ASP" => 5.82,
        "CYS" => 1.84,
        "GLN" => 3.94,
        "GLU" => 6.61,
        "GLY" => 7.11,
        "HIS" | "HSD" => 2.35,
        "ILE" => 5.66,
        "LEU" => 8.83,
        "LYS" => 6.27,
        "MET" => 2.08,
        "PHE" => 4.05,
        "PRO" => 4.52,
        "SER" => 6.13,
        "THR" => 5.53,
        "TRP" => 1.51,
        "TYR" => 3.54,
        "VAL" => 6.91,
        _ => panic!("no propensity for {aa}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aa_names_count() {
        assert_eq!(AA_NAMES.len(), 18);
        // GLY and PRO absent
        assert!(!AA_NAMES.contains(&"GLY"));
        assert!(!AA_NAMES.contains(&"PRO"));
    }

    #[test]
    fn aa_propensity_spot_checks() {
        assert!((aa_propensity("ALA") - 7.73).abs() < 1e-10);
        assert!((aa_propensity("VAL") - 6.91).abs() < 1e-10);
        assert!((aa_propensity("GLY") - 7.11).abs() < 1e-10);
        assert!((aa_propensity("HIS") - 2.35).abs() < 1e-10);
        assert!((aa_propensity("HSD") - 2.35).abs() < 1e-10);
    }

    #[test]
    #[should_panic(expected = "no propensity")]
    fn aa_propensity_unknown() {
        aa_propensity("XYZ");
    }

    #[test]
    fn propensity_sum_roughly_100() {
        let s: f64 = AA_NAMES.iter().map(|aa| aa_propensity(aa)).sum::<f64>()
            + aa_propensity("GLY")
            + aa_propensity("PRO");
        assert!((s - 100.0).abs() < 1.0, "propensity sum = {s}");
    }
}
