use std::collections::HashMap;
use std::sync::Arc;
use proxide_rotlib::RotamerId;

/// Compute the freedom score for one residue.
///
/// `coll_prob`: map from rotamer ID → accumulated collision probability.
/// `n_surviving`: surviving rotamer count for this residue.
/// `n_library`: total library rotamer count for this residue.
///
/// Returns freedom type 2 by default (used by ConFind v1).
pub fn compute_freedom(
    coll_prob: &HashMap<Arc<RotamerId>, f64>,
    n_surviving: usize,
    n_library: usize,
    lo_cut: f64,
    hi_cut: f64,
    freedom_type: u8,
) -> f64 {
    if n_library == 0 {
        return 0.0;
    }
    let n_uncontested = n_surviving.saturating_sub(coll_prob.len()) as f64;

    match freedom_type {
        1 => {
            let n = n_uncontested
                + coll_prob.values().filter(|&&v| v / 100.0 < 0.5).count() as f64;
            n / n_library as f64
        }
        2 => {
            let n1 = n_uncontested
                + coll_prob.values().filter(|&&v| v / 100.0 < lo_cut).count() as f64;
            let n2 = n_uncontested
                + coll_prob.values().filter(|&&v| v / 100.0 < hi_cut).count() as f64;
            ((n1 * n1 + n2 * n2) / 2.0).sqrt() / n_library as f64
        }
        3 => {
            let n1 = (n_uncontested
                + coll_prob.values().filter(|&&v| v / 100.0 < lo_cut).count() as f64)
                / n_library as f64;
            let n2 = (n_uncontested
                + coll_prob.values().filter(|&&v| v / 100.0 < hi_cut).count() as f64)
                / n_library as f64;
            ((n2 * n2 + n2 * n1) / 2.0).sqrt()
        }
        _ => panic!("unknown freedom type {freedom_type}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn freedom_type2_no_collisions() {
        // All surviving rotamers uncontested → freedom = 1.0.
        let cp: HashMap<Arc<RotamerId>, f64> = HashMap::new();
        let f = compute_freedom(&cp, 10, 10, 0.5, 2.0, 2);
        assert!((f - 1.0).abs() < 1e-10, "f={f}");
    }

    #[test]
    fn freedom_type2_all_contested() {
        // All rotamers contested with prob >> hi_cut (200 >> 2.0*100 = 200).
        let cp = make_map(&[(0, 300.0), (1, 300.0), (2, 300.0)]);
        let f = compute_freedom(&cp, 3, 3, 0.5, 2.0, 2);
        // n1 = 0, n2 = 0, n_uncontested = 0 → f = 0.
        assert!(f.abs() < 1e-10, "f={f}");
    }

    #[test]
    fn freedom_type2_mixed() {
        // 2 surviving, 1 with cp=40 (40/100=0.4 < 0.5 AND < 2.0), 1 with cp=150 (1.5 < 2.0 but ≥ 0.5).
        // n_library=4, n_uncontested = 2 - 2 = 0.
        // n1 = 0 + 1 (only 40/100=0.4 < 0.5) = 1
        // n2 = 0 + 2 (both 0.4 < 2.0 and 1.5 < 2.0) = 2
        let cp = make_map(&[(0, 40.0), (1, 150.0)]);
        let f = compute_freedom(&cp, 2, 4, 0.5, 2.0, 2);
        let expected = ((1.0_f64 * 1.0 + 2.0 * 2.0) / 2.0).sqrt() / 4.0;
        assert!((f - expected).abs() < 1e-10, "f={f} expected={expected}");
    }

    #[test]
    fn freedom_type1() {
        let cp = make_map(&[(0, 40.0)]);
        let f = compute_freedom(&cp, 3, 4, 0.5, 2.0, 1);
        // n_uncontested=2, pass < 0.5: 1 (40/100=0.4). n = 3, lib=4.
        let expected = 3.0 / 4.0;
        assert!((f - expected).abs() < 1e-10);
    }

    #[test]
    fn freedom_zero_library() {
        let cp: HashMap<Arc<RotamerId>, f64> = HashMap::new();
        let f = compute_freedom(&cp, 0, 0, 0.5, 2.0, 2);
        assert_eq!(f, 0.0);
    }
}
