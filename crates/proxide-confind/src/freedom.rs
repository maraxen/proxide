//! Structural freedom calculation from sidechain collision probabilities.
//!
//! Citations:
//! - Zheng, W., & Grigoryan, G. (2017). "Sequence statistics of tertiary structural motifs
//!   reflect protein stability." *PLoS ONE*, 12(5), e0178272, S1 Text.

/// Compute residue structural freedom from surviving rotamer collision probabilities.
///
/// Under the RMS freedom formulation (spec §2.5, §3.3):
/// - Uncontested rotamers (zero collision with neighbors) = n_surviving - collision_counts.len()
/// - n_1 = n_uncontested + count(P_coll < lo_cut)
/// - n_2 = n_uncontested + count(P_coll < hi_cut)
/// - Freedom = sqrt((n_1^2 + n_2^2) / 2) / n_library
pub fn compute_freedom(
    collision_counts: &[f64],
    n_surviving: usize,
    n_library: usize,
    lo_cut: f64,
    hi_cut: f64,
) -> f64 {
    if n_library == 0 {
        return 0.0;
    }

    let n_uncontested = n_surviving.saturating_sub(collision_counts.len());
    let mut n1 = n_uncontested;
    let mut n2 = n_uncontested;

    for &c in collision_counts {
        if c < lo_cut {
            n1 += 1;
        }
        if c < hi_cut {
            n2 += 1;
        }
    }

    let n1_f = n1 as f64;
    let n2_f = n2 as f64;
    let rms = ((n1_f * n1_f + n2_f * n2_f) / 2.0).sqrt();
    (rms / (n_library as f64)).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_library_returns_zero() {
        assert_eq!(compute_freedom(&[], 0, 0, 0.5, 2.0), 0.0);
    }

    #[test]
    fn test_uncontested_freedom() {
        // 10 surviving rotamers out of 10 library rotamers, zero collisions
        let f = compute_freedom(&[], 10, 10, 0.5, 2.0);
        assert!((f - 1.0).abs() < 1e-6);

        // 5 surviving out of 10 library, zero collisions
        let f5 = compute_freedom(&[], 5, 10, 0.5, 2.0);
        assert!((f5 - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_high_collision_freedom() {
        // 10 surviving, all 10 have severe collisions (> 2.0)
        let collisions = vec![3.0; 10];
        let f = compute_freedom(&collisions, 10, 10, 0.5, 2.0);
        assert_eq!(f, 0.0);
    }

    #[test]
    fn test_moderate_collision_freedom() {
        // 10 surviving, 5 with P_coll = 1.0 (between 0.5 and 2.0), 5 with P_coll = 0.2 (< 0.5)
        let collisions = vec![0.2, 0.2, 0.2, 0.2, 0.2, 1.0, 1.0, 1.0, 1.0, 1.0];
        let f = compute_freedom(&collisions, 10, 10, 0.5, 2.0);
        // n1 = 5, n2 = 10
        // rms = sqrt((25 + 100) / 2) = sqrt(62.5) ≈ 7.90569
        // f = 7.90569 / 10 ≈ 0.79057
        assert!((f - (62.5f64.sqrt() / 10.0)).abs() < 1e-6);
    }

    #[test]
    fn test_freedom_bounded_in_zero_one() {
        let f1 = compute_freedom(&[0.1, 0.4], 5, 10, 0.5, 2.0);
        assert!((0.0..=1.0).contains(&f1));
    }
}
