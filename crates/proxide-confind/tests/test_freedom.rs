use proxide_confind::compute_freedom;
use proxide_rotlib::RotamerId;
use std::collections::HashMap;
use std::sync::Arc;

fn make_map(entries: &[(u32, f64)]) -> HashMap<Arc<RotamerId>, f64> {
    entries
        .iter()
        .map(|&(i, v)| {
            (
                Arc::new(RotamerId {
                    aa: format!("AA{i}"),
                    bin_index: 0,
                    rot_index: i,
                }),
                v,
            )
        })
        .collect()
}

#[test]
fn freedom_type2_all_uncontested() {
    // Type 2, empty collision_prob, n_surviving=10, n_library=10.
    // All uncontested → freedom = 1.0.
    let cp: HashMap<Arc<RotamerId>, f64> = HashMap::new();
    let f = compute_freedom(&cp, 10, 10, 0.5, 2.0, 2);
    assert!((f - 1.0).abs() < 1e-10, "f={f}, expected 1.0");
}

#[test]
fn freedom_type2_all_contested_high() {
    // Type 2, all rotamers contested with prob >> hi_cut (300 >> 2.0*100).
    // n_uncontested=0, n1=0, n2=0 → freedom = 0.0.
    let cp = make_map(&[(0, 300.0), (1, 300.0), (2, 300.0)]);
    let f = compute_freedom(&cp, 3, 3, 0.5, 2.0, 2);
    assert!(f.abs() < 1e-10, "f={f}, expected 0.0");
}

#[test]
fn freedom_type2_mixed_collisions() {
    // Type 2: n_surviving=2, n_library=4, coll_prob has 2 entries.
    // entry 0: cp=40.0 → 40/100=0.4 < 0.5 AND < 2.0
    // entry 1: cp=150.0 → 150/100=1.5 >= 0.5 AND < 2.0
    // n_uncontested = 2 - 2 = 0
    // n1 = 0 + 1 (only 40/100 < 0.5) = 1
    // n2 = 0 + 2 (both 0.4 < 2.0 and 1.5 < 2.0) = 2
    // expected = sqrt((1^2 + 2^2)/2) / 4 = sqrt(5/2) / 4
    let cp = make_map(&[(0, 40.0), (1, 150.0)]);
    let f = compute_freedom(&cp, 2, 4, 0.5, 2.0, 2);
    let expected = ((1.0_f64 * 1.0 + 2.0 * 2.0) / 2.0).sqrt() / 4.0;
    assert!((f - expected).abs() < 1e-10, "f={f}, expected={expected}");
}

#[test]
fn freedom_type1() {
    // Type 1: n_surviving=3, n_library=4, coll_prob has 1 entry.
    // entry 0: cp=40.0 → 40/100=0.4 < 0.5
    // n_uncontested = 3 - 1 = 2
    // count < 0.5: 1 (just the 40/100)
    // n = 2 + 1 = 3
    // f = 3 / 4 = 0.75
    let cp = make_map(&[(0, 40.0)]);
    let f = compute_freedom(&cp, 3, 4, 0.5, 2.0, 1);
    let expected = 3.0 / 4.0;
    assert!((f - expected).abs() < 1e-10);
}

#[test]
fn freedom_type3() {
    // Type 3: uses different formula with n1 and n2 normalized.
    // Let's test with same data as type 2 mixed.
    // n_library=4, n_uncontested=0
    // n1_norm = (0 + 1) / 4 = 0.25
    // n2_norm = (0 + 2) / 4 = 0.5
    // f = sqrt((n2^2 + n2*n1) / 2) = sqrt((0.5^2 + 0.5*0.25) / 2) = sqrt((0.25 + 0.125) / 2)
    let cp = make_map(&[(0, 40.0), (1, 150.0)]);
    let f = compute_freedom(&cp, 2, 4, 0.5, 2.0, 3);
    let n1_norm: f64 = 1.0 / 4.0;
    let n2_norm: f64 = 2.0 / 4.0;
    let expected: f64 = ((n2_norm * n2_norm + n2_norm * n1_norm) / 2.0).sqrt();
    assert!((f - expected).abs() < 1e-10, "f={f}, expected={expected}");
}

#[test]
fn freedom_zero_library() {
    // n_library=0 → should return 0.0 immediately.
    let cp: HashMap<Arc<RotamerId>, f64> = HashMap::new();
    let f = compute_freedom(&cp, 0, 0, 0.5, 2.0, 2);
    assert_eq!(f, 0.0);
}

#[test]
fn freedom_type2_mixed_with_uncontested() {
    // Test case: n_surviving=5, n_library=10.
    // coll_prob has 2 entries: cp=30.0 (< lo_cut), cp=150.0 (>= lo_cut, < hi_cut)
    // n_uncontested = 5 - 2 = 3
    // n1 = 3 + 1 (30/100=0.3 < 0.5) = 4
    // n2 = 3 + 2 (both < 2.0) = 5
    // f = sqrt((4^2 + 5^2)/2) / 10 = sqrt(41/2) / 10
    let cp = make_map(&[(0, 30.0), (1, 150.0)]);
    let f = compute_freedom(&cp, 5, 10, 0.5, 2.0, 2);
    let expected = ((4.0_f64.powi(2) + 5.0_f64.powi(2)) / 2.0).sqrt() / 10.0;
    assert!((f - expected).abs() < 1e-10, "f={f}, expected={expected}");
}

#[test]
#[should_panic(expected = "unknown freedom type")]
fn freedom_invalid_type() {
    let cp: HashMap<Arc<RotamerId>, f64> = HashMap::new();
    compute_freedom(&cp, 5, 10, 0.5, 2.0, 99);
}
