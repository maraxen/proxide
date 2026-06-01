use proxide_confind::{aa_propensity, AA_NAMES};

#[test]
fn aa_propensity_spot_checks() {
    assert!((aa_propensity("ALA") - 7.73).abs() < 1e-10);
    assert!((aa_propensity("ARG") - 5.03).abs() < 1e-10);
    assert!((aa_propensity("VAL") - 6.91).abs() < 1e-10);
    assert!((aa_propensity("GLY") - 7.11).abs() < 1e-10);
    assert!((aa_propensity("PRO") - 4.52).abs() < 1e-10);
    assert!((aa_propensity("HIS") - 2.35).abs() < 1e-10);
    assert!((aa_propensity("HSD") - 2.35).abs() < 1e-10);
}

#[test]
fn aa_names_count() {
    assert_eq!(AA_NAMES.len(), 18);
}

#[test]
fn gly_and_pro_absent() {
    assert!(!AA_NAMES.contains(&"GLY"));
    assert!(!AA_NAMES.contains(&"PRO"));
}

#[test]
fn propensity_sum_approximately_100() {
    let sum: f64 = AA_NAMES.iter().map(|aa| aa_propensity(aa)).sum::<f64>()
        + aa_propensity("GLY")
        + aa_propensity("PRO");
    assert!((sum - 100.0).abs() < 1.0, "propensity sum = {sum}");
}

#[test]
#[should_panic(expected = "no propensity")]
fn aa_propensity_panics_on_unknown_aa() {
    aa_propensity("XYZ");
}
