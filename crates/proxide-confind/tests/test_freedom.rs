use proxide_confind::{compute_freedom, FreedomMode};

#[test]
fn freedom_all_uncontested() {
    // Empty collision counts, n_surviving=10, n_library=10.
    // All uncontested → freedom = 1.0.
    let cp: Vec<f64> = Vec::new();
    let f = compute_freedom(FreedomMode::Rms, &cp, 10, 10, 0.5, 2.0);
    assert!((f - 1.0).abs() < 1e-10, "f={f}, expected 1.0");
}

#[test]
fn freedom_all_contested_high() {
    // All rotamers contested with prob > hi_cut (3.0 > 2.0).
    // n_uncontested=0, n1=0, n2=0 → freedom = 0.0.
    let cp = vec![3.0, 3.0, 3.0];
    let f = compute_freedom(FreedomMode::Rms, &cp, 3, 3, 0.5, 2.0);
    assert!(f.abs() < 1e-10, "f={f}, expected 0.0");
}

#[test]
fn freedom_mixed_collisions() {
    // n_surviving=2, n_library=4, 2 collision counts:
    // entry 0: cp=0.4 (< 0.5 and < 2.0)
    // entry 1: cp=1.5 (>= 0.5 and < 2.0)
    // n_uncontested = 2 - 2 = 0
    // n1 = 0 + 1 (only 0.4 < 0.5) = 1
    // n2 = 0 + 2 (both < 2.0) = 2
    // expected = sqrt((1^2 + 2^2)/2) / 4 = sqrt(5/2) / 4
    let cp = vec![0.4, 1.5];
    let f = compute_freedom(FreedomMode::Rms, &cp, 2, 4, 0.5, 2.0);
    let expected = ((1.0_f64 * 1.0 + 2.0 * 2.0) / 2.0).sqrt() / 4.0;
    assert!((f - expected).abs() < 1e-10, "f={f}, expected={expected}");
}

#[test]
fn freedom_zero_library() {
    // n_library=0 → returns 0.0 immediately.
    let f = compute_freedom(FreedomMode::Rms, &[], 0, 0, 0.5, 2.0);
    assert_eq!(f, 0.0);
}

#[test]
fn freedom_mixed_with_uncontested() {
    // n_surviving=5, n_library=10.
    // 2 collision counts: cp=0.3 (< 0.5), cp=1.5 (>= 0.5, < 2.0)
    // n_uncontested = 5 - 2 = 3
    // n1 = 3 + 1 = 4
    // n2 = 3 + 2 = 5
    // expected = sqrt((4^2 + 5^2)/2) / 10 = sqrt(41/2) / 10
    let cp = vec![0.3, 1.5];
    let f = compute_freedom(FreedomMode::Rms, &cp, 5, 10, 0.5, 2.0);
    let expected = ((4.0_f64.powi(2) + 5.0_f64.powi(2)) / 2.0).sqrt() / 10.0;
    assert!((f - expected).abs() < 1e-10, "f={f}, expected={expected}");
}
