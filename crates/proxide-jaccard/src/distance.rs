//! Jaccard similarity/distance between two scaled-MinHash sketches.
//!
//! Sketches are sorted, deduplicated `i64` hash sets (sourmash-style "scaled"
//! MinHash signatures). At equal scale, the stored hash sets are themselves
//! an unbiased sample of the underlying k-mer sets, so the Jaccard estimate
//! can be read directly off the intersection/union of the two sets — no
//! resampling to a common size is needed. Both inputs must already be sorted
//! ascending and free of duplicate values; callers (see `sketch.rs`) are
//! responsible for that invariant.

/// Number of elements shared between two sorted, deduplicated slices.
///
/// O(len_a + len_b) via a merge walk; no allocation.
fn intersection_count(a: &[i64], b: &[i64]) -> usize {
    debug_assert!(
        a.windows(2).all(|w| w[0] < w[1]),
        "sketch `a` must be sorted+deduped"
    );
    debug_assert!(
        b.windows(2).all(|w| w[0] < w[1]),
        "sketch `b` must be sorted+deduped"
    );

    let mut i = 0;
    let mut j = 0;
    let mut count = 0;
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                count += 1;
                i += 1;
                j += 1;
            }
        }
    }
    count
}

/// Jaccard similarity |A ∩ B| / |A ∪ B| in [0.0, 1.0].
///
/// Two empty sketches are defined as similarity 0.0 (rather than the
/// mathematically-undefined 0/0) so the result is always a finite f64.
pub fn jaccard_similarity(a: &[i64], b: &[i64]) -> f64 {
    let union = a.len() + b.len();
    if union == 0 {
        return 0.0;
    }
    let intersection = intersection_count(a, b);
    intersection as f64 / (union - intersection) as f64
}

/// Jaccard distance 1 - similarity, in [0.0, 1.0].
pub fn jaccard_distance(a: &[i64], b: &[i64]) -> f64 {
    1.0 - jaccard_similarity(a, b)
}

/// All pairwise overlap statistics derivable from one merge pass.
/// Containment is "free" once `intersection` is known — it doesn't add a
/// second O(len_a + len_b) pass on top of Jaccard, since the expensive
/// part (the merge walk) is shared.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Overlap {
    /// |A ∩ B|.
    pub intersection: usize,
    /// |A ∩ B| / |A| — fraction of `a` contained in `b`. 0.0 if `a` is empty.
    pub containment_a_in_b: f64,
    /// |A ∩ B| / |B| — fraction of `b` contained in `a`. 0.0 if `b` is empty.
    pub containment_b_in_a: f64,
    /// |A ∩ B| / |A ∪ B|.
    pub jaccard_similarity: f64,
}

/// Computes intersection, both directional containments, and Jaccard
/// similarity in a single O(len_a + len_b) merge pass.
///
/// Containment is the more informative metric when sketch sizes differ a
/// lot (e.g. a virus genome vs. a vertebrate genome at the same minhash
/// scale): Jaccard penalizes the size mismatch even when one sketch's
/// content is essentially a subset of the other's, while containment
/// answers "what fraction of A's k-mers are also in B?" directly.
///
/// Containment is asymmetric (`containment_a_in_b != containment_b_in_a`
/// in general) and is *not* a metric — it does not satisfy the triangle
/// inequality, unlike Jaccard distance, which does. Algorithms that
/// require a symmetric distance (e.g. HDBSCAN / mutual-reachability
/// clustering) need either `jaccard_similarity`/`jaccard_distance`, or
/// [`overlap_coefficient`] (a symmetrized containment).
pub fn overlap(a: &[i64], b: &[i64]) -> Overlap {
    let intersection = intersection_count(a, b);
    let containment_a_in_b = if a.is_empty() {
        0.0
    } else {
        intersection as f64 / a.len() as f64
    };
    let containment_b_in_a = if b.is_empty() {
        0.0
    } else {
        intersection as f64 / b.len() as f64
    };
    let union = a.len() + b.len();
    let jaccard_similarity = if union == 0 {
        0.0
    } else {
        intersection as f64 / (union - intersection) as f64
    };
    Overlap {
        intersection,
        containment_a_in_b,
        containment_b_in_a,
        jaccard_similarity,
    }
}

/// |A ∩ B| / |A| — the fraction of `a` contained in `b`, in [0.0, 1.0].
/// Asymmetric: `containment(a, b) != containment(b, a)` in general. 0.0 if
/// `a` is empty, rather than the mathematically-undefined 0/0.
///
/// If you also need Jaccard or the reverse containment for the same
/// pair, prefer [`overlap`] — it computes all of them from one merge
/// pass instead of repeating the O(len_a + len_b) walk per metric.
pub fn containment(a: &[i64], b: &[i64]) -> f64 {
    if a.is_empty() {
        return 0.0;
    }
    intersection_count(a, b) as f64 / a.len() as f64
}

/// The overlap coefficient `min(containment(a,b), containment(b,a))` —
/// symmetric, unlike raw containment, so it's a drop-in alternative
/// wherever a symmetric distance/similarity is required (e.g. as a
/// clustering metric: `1.0 - overlap_coefficient(a, b)`).
pub fn overlap_coefficient(a: &[i64], b: &[i64]) -> f64 {
    let o = overlap(a, b);
    o.containment_a_in_b.min(o.containment_b_in_a)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// Reference implementation via HashSet, for differential testing
    /// against the merge-based kernel.
    fn naive_jaccard(a: &[i64], b: &[i64]) -> f64 {
        let sa: HashSet<i64> = a.iter().copied().collect();
        let sb: HashSet<i64> = b.iter().copied().collect();
        let union = sa.union(&sb).count();
        if union == 0 {
            return 0.0;
        }
        sa.intersection(&sb).count() as f64 / union as f64
    }

    #[test]
    fn identical_sets_have_similarity_one() {
        let a = [1, 2, 3, 4, 5];
        assert_eq!(jaccard_similarity(&a, &a), 1.0);
        assert_eq!(jaccard_distance(&a, &a), 0.0);
    }

    #[test]
    fn disjoint_sets_have_similarity_zero() {
        let a = [1, 2, 3];
        let b = [4, 5, 6];
        assert_eq!(jaccard_similarity(&a, &b), 0.0);
        assert_eq!(jaccard_distance(&a, &b), 1.0);
    }

    #[test]
    fn both_empty_is_zero_not_nan() {
        let a: [i64; 0] = [];
        assert_eq!(jaccard_similarity(&a, &a), 0.0);
    }

    #[test]
    fn one_empty_is_zero() {
        let a: [i64; 0] = [];
        let b = [1, 2, 3];
        assert_eq!(jaccard_similarity(&a, &b), 0.0);
    }

    #[test]
    fn partial_overlap_matches_naive() {
        let a = [1, 2, 3, 4, 5, 10, 20];
        let b = [3, 4, 5, 6, 7, 20, 30];
        assert_eq!(jaccard_similarity(&a, &b), naive_jaccard(&a, &b));
        // intersection {3,4,5,20} = 4, union = 7+7-4 = 10
        assert!((jaccard_similarity(&a, &b) - 0.4).abs() < 1e-12);
    }

    #[test]
    fn negative_hash_values_are_handled() {
        // i64 minhash values can be negative depending on the hash function;
        // ordering/intersection must not assume non-negativity.
        let a = [-100, -5, 0, 5, 100];
        let b = [-100, -5, 1, 5, 200];
        assert_eq!(jaccard_similarity(&a, &b), naive_jaccard(&a, &b));
    }

    #[test]
    fn fuzz_against_naive_reference() {
        // Deterministic pseudo-random fuzz (no external RNG dependency at
        // this layer): generate sorted, deduplicated slices via a simple
        // LCG and differential-test against the HashSet reference.
        let mut state: u64 = 0x2545F4914F6CDD1D;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as i64
        };
        for _ in 0..200 {
            let na = (next().unsigned_abs() % 50) as usize;
            let nb = (next().unsigned_abs() % 50) as usize;
            let mut a: Vec<i64> = (0..na).map(|_| next() % 1000).collect();
            let mut b: Vec<i64> = (0..nb).map(|_| next() % 1000).collect();
            a.sort_unstable();
            a.dedup();
            b.sort_unstable();
            b.dedup();
            let expected = naive_jaccard(&a, &b);
            let actual = jaccard_similarity(&a, &b);
            assert!(
                (expected - actual).abs() < 1e-12,
                "mismatch for a={a:?} b={b:?}: naive={expected} kernel={actual}"
            );
        }
    }

    fn naive_containment(a: &[i64], b: &[i64]) -> f64 {
        if a.is_empty() {
            return 0.0;
        }
        let sa: HashSet<i64> = a.iter().copied().collect();
        let sb: HashSet<i64> = b.iter().copied().collect();
        sa.intersection(&sb).count() as f64 / a.len() as f64
    }

    #[test]
    fn containment_is_asymmetric_for_subset() {
        // a is fully contained in b, but b is mostly not contained in a.
        let a = [1, 2, 3];
        let b = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        assert_eq!(containment(&a, &b), 1.0); // all of a is in b
        assert!((containment(&b, &a) - 0.3).abs() < 1e-12); // 3 of 10 elements of b are in a
    }

    #[test]
    fn containment_matches_naive_reference() {
        let a = [1, 2, 3, 4, 5, 10, 20];
        let b = [3, 4, 5, 6, 7, 20, 30];
        assert_eq!(containment(&a, &b), naive_containment(&a, &b));
        assert_eq!(containment(&b, &a), naive_containment(&b, &a));
    }

    #[test]
    fn containment_of_empty_a_is_zero() {
        let a: [i64; 0] = [];
        let b = [1, 2, 3];
        assert_eq!(containment(&a, &b), 0.0);
    }

    #[test]
    fn overlap_struct_matches_individual_functions() {
        let a = [1, 2, 3, 4, 5, 10, 20];
        let b = [3, 4, 5, 6, 7, 20, 30];
        let o = overlap(&a, &b);
        assert_eq!(o.containment_a_in_b, containment(&a, &b));
        assert_eq!(o.containment_b_in_a, containment(&b, &a));
        assert_eq!(o.jaccard_similarity, jaccard_similarity(&a, &b));
        assert_eq!(o.intersection, 4); // {3,4,5,20}
    }

    #[test]
    fn overlap_coefficient_is_symmetric_and_equals_smaller_containment() {
        let a = [1, 2, 3];
        let b = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let coeff_ab = overlap_coefficient(&a, &b);
        let coeff_ba = overlap_coefficient(&b, &a);
        assert_eq!(coeff_ab, coeff_ba, "overlap coefficient must be symmetric");
        // a is fully contained in b, so the coefficient is min(1.0, 0.3) = 0.3.
        assert!((coeff_ab - 0.3).abs() < 1e-12);
    }

    #[test]
    fn fuzz_overlap_against_naive_reference() {
        let mut state: u64 = 0xA1F2E3D4C5B6978;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as i64
        };
        for _ in 0..200 {
            let na = (next().unsigned_abs() % 50) as usize;
            let nb = (next().unsigned_abs() % 50) as usize;
            let mut a: Vec<i64> = (0..na).map(|_| next() % 1000).collect();
            let mut b: Vec<i64> = (0..nb).map(|_| next() % 1000).collect();
            a.sort_unstable();
            a.dedup();
            b.sort_unstable();
            b.dedup();

            let o = overlap(&a, &b);
            assert!(
                (o.containment_a_in_b - naive_containment(&a, &b)).abs() < 1e-12,
                "containment(a,b) mismatch for a={a:?} b={b:?}"
            );
            assert!(
                (o.containment_b_in_a - naive_containment(&b, &a)).abs() < 1e-12,
                "containment(b,a) mismatch for a={a:?} b={b:?}"
            );
            assert!(
                (o.jaccard_similarity - naive_jaccard(&a, &b)).abs() < 1e-12,
                "jaccard mismatch for a={a:?} b={b:?}"
            );
        }
    }
}
