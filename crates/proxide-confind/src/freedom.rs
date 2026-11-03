//! Structural freedom calculation from sidechain collision probabilities.
//!
//! Citations:
//! - Zheng, W., & Grigoryan, G. (2017). "Sequence statistics of tertiary structural motifs
//!   reflect protein stability." *PLoS ONE*, 12(5), e0178272, S1 Text.

use crate::config::FreedomMode;

/// Compute residue structural freedom from surviving-rotamer collision probabilities.
///
/// - `p_coll`: collision mass of the *contested* surviving rotamers only (those with
///   `P_coll > 0`).
/// - uncontested = `n_surviving.saturating_sub(p_coll.len())` (`P_coll = 0 < tau` for any
///   `tau > 0`).
/// - `N_lo = uncontested + count(p < tau_lo)`; `N_hi = uncontested + count(p < tau_hi)`
///   (strict inequality).
/// - [`FreedomMode::Rms`]: `sqrt((N_lo^2 + N_hi^2) / 2) / n_library`.
/// - [`FreedomMode::SingleThreshold`]: `N_lo / n_library` (`tau_hi` unused).
/// - `n_library == 0` returns `0.0` by convention.
pub fn compute_freedom(
    mode: FreedomMode,
    p_coll: &[f64],
    n_surviving: usize,
    n_library: usize,
    tau_lo: f64,
    tau_hi: f64,
) -> f64 {
    if n_library == 0 {
        return 0.0;
    }

    let n_uncontested = n_surviving.saturating_sub(p_coll.len());
    let mut n_lo = n_uncontested;
    let mut n_hi = n_uncontested;

    for &c in p_coll {
        if c < tau_lo {
            n_lo += 1;
        }
        if c < tau_hi {
            n_hi += 1;
        }
    }

    let n_lo_f = n_lo as f64;
    let n_hi_f = n_hi as f64;
    let numerator = match mode {
        FreedomMode::Rms => ((n_lo_f * n_lo_f + n_hi_f * n_hi_f) / 2.0).sqrt(),
        FreedomMode::SingleThreshold => n_lo_f,
    };
    (numerator / (n_library as f64)).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use FreedomMode::{Rms, SingleThreshold};

    #[test]
    fn zero_library_returns_zero() {
        assert_eq!(compute_freedom(Rms, &[], 0, 0, 0.5, 2.0), 0.0);
        assert_eq!(compute_freedom(SingleThreshold, &[], 0, 0, 0.5, 2.0), 0.0);
    }

    #[test]
    fn no_neighbours_spec_example() {
        // N_lib = 4, N_surv = 2, no neighbours: F = sqrt((4 + 4)/2)/4 = 0.5.
        assert!((compute_freedom(Rms, &[], 2, 4, 0.5, 2.0) - 0.5).abs() < 1e-9);
        assert!((compute_freedom(SingleThreshold, &[], 2, 4, 0.5, 2.0) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn worked_example_sqrt37() {
        // N_lib = 10, N_surv = 8; P_coll = {0.3, 0.7, 1.5, 2.5}, four uncontested:
        // N_lo = 5, N_hi = 7, F = sqrt(37)/10; single threshold 0.5.
        let p = [0.3, 0.7, 1.5, 2.5];
        let f = compute_freedom(Rms, &p, 8, 10, 0.5, 2.0);
        assert!((f - 37f64.sqrt() / 10.0).abs() < 1e-9);
        let f1 = compute_freedom(SingleThreshold, &p, 8, 10, 0.5, 2.0);
        assert!((f1 - 0.5).abs() < 1e-9);
    }

    #[test]
    fn boundary_is_strict() {
        // P_coll = 0.5 is excluded from N_lo but included in N_hi.
        // N_lo = 0, N_hi = 1 over N_lib = 1: Rms = sqrt(1/2).
        let f = compute_freedom(Rms, &[0.5], 1, 1, 0.5, 2.0);
        assert!((f - (0.5f64).sqrt()).abs() < 1e-12);
        assert_eq!(
            compute_freedom(SingleThreshold, &[0.5], 1, 1, 0.5, 2.0),
            0.0
        );
    }

    #[test]
    fn zero_survivors_gives_zero() {
        assert_eq!(compute_freedom(Rms, &[], 0, 7, 0.5, 2.0), 0.0);
    }

    #[test]
    fn invariants_over_pseudo_random_inputs() {
        // Deterministic LCG; checks 0 <= F <= 1, F <= N_surv/N_lib, F1 <= F and
        // monotone non-increase when one P_coll grows.
        let mut s: u64 = 0x1234_5678_9abc_def0;
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f64) / ((1u64 << 31) as f64)
        };
        for _ in 0..500 {
            let n_lib = 1 + (next() * 40.0) as usize;
            let n_surv = ((next() * (n_lib as f64 + 1.0)) as usize).min(n_lib);
            let n_contested = ((next() * (n_surv as f64 + 1.0)) as usize).min(n_surv);
            let p: Vec<f64> = (0..n_contested).map(|_| 0.01 + next() * 4.0).collect();
            let f = compute_freedom(Rms, &p, n_surv, n_lib, 0.5, 2.0);
            let f1 = compute_freedom(SingleThreshold, &p, n_surv, n_lib, 0.5, 2.0);
            assert!((0.0..=1.0).contains(&f));
            assert!(f <= n_surv as f64 / n_lib as f64 + 1e-12);
            assert!(f1 <= f + 1e-12);
            if !p.is_empty() {
                let mut q = p.clone();
                q[0] += 1.0;
                assert!(compute_freedom(Rms, &q, n_surv, n_lib, 0.5, 2.0) <= f + 1e-12);
            }
        }
    }
}
