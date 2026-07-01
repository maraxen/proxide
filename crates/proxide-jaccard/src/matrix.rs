//! Parallel pairwise Jaccard distance matrix over a [`SketchStore`].
//!
//! The O(n²) triangle is the actual computational bottleneck (the merge
//! kernel itself is O(sketch_len) per pair); IO and parsing are comparatively
//! cheap for the subset sizes this is built for (an explicit accession list,
//! not a full shard scan). Each row `i` is dispatched as one `orx-parallel`
//! task computing distances to all `j > i`; row lengths shrink linearly as
//! `i` grows, and orx-parallel's work-stealing scheduler absorbs that
//! imbalance without any manual chunking.

use crate::distance::{jaccard_distance, overlap};
use crate::sketch::SketchStore;
use ndarray::Array2;
use orx_parallel::{IntoParIter, ParIter};

/// Dense, symmetric n×n Jaccard distance matrix (f32) in accession order
/// matching `store.accessions()`. Diagonal is 0.0 (self-distance).
pub fn pairwise_jaccard_distance(store: &SketchStore) -> Array2<f32> {
    let n = store.len();
    let mut mat = Array2::<f32>::zeros((n, n));
    if n < 2 {
        return mat;
    }

    let rows: Vec<Vec<f32>> = (0..n)
        .into_par()
        .map(|i| {
            let a = store.sketch(i);
            ((i + 1)..n)
                .map(|j| jaccard_distance(a, store.sketch(j)) as f32)
                .collect()
        })
        .collect();

    for (i, row) in rows.into_iter().enumerate() {
        for (offset, dist) in row.into_iter().enumerate() {
            let j = i + 1 + offset;
            mat[[i, j]] = dist;
            mat[[j, i]] = dist;
        }
    }

    mat
}

/// Dense n×n pairwise containment matrix: `mat[[i, j]]` is the fraction
/// of accession `i`'s hashes contained in accession `j`'s
/// (`distance::containment(store.sketch(i), store.sketch(j))`).
///
/// Unlike [`pairwise_jaccard_distance`], this matrix is **not**
/// symmetric — `mat[[i, j]] != mat[[j, i]]` in general. Containment is
/// the more informative metric when sketch sizes differ a lot (e.g.
/// comparing a virus genome against a vertebrate genome at the same
/// minhash scale): Jaccard punishes the size mismatch even when one
/// sketch is essentially a subset of the other, while containment
/// answers "is A contained in B?" directly, per direction.
///
/// Computes both directions from a single merge pass per pair (via
/// [`crate::distance::overlap`]), not two — same total kernel work as
/// `pairwise_jaccard_distance`, just twice the output. Diagonal is
/// always 1.0 (every sketch fully contains itself), bypassing the
/// general kernel's 0/0 convention for empty sketches, the same way
/// `pairwise_jaccard_distance`'s diagonal is always 0.0.
pub fn pairwise_containment(store: &SketchStore) -> Array2<f32> {
    let n = store.len();
    let mut mat = Array2::<f32>::zeros((n, n));
    for i in 0..n {
        mat[[i, i]] = 1.0;
    }
    if n < 2 {
        return mat;
    }

    let rows: Vec<Vec<(f32, f32)>> = (0..n)
        .into_par()
        .map(|i| {
            let a = store.sketch(i);
            ((i + 1)..n)
                .map(|j| {
                    let o = overlap(a, store.sketch(j));
                    (o.containment_a_in_b as f32, o.containment_b_in_a as f32)
                })
                .collect()
        })
        .collect();

    for (i, row) in rows.into_iter().enumerate() {
        for (offset, (a_in_b, b_in_a)) in row.into_iter().enumerate() {
            let j = i + 1 + offset;
            mat[[i, j]] = a_in_b;
            mat[[j, i]] = b_in_a;
        }
    }

    mat
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::jaccard_distance;

    fn store_from(rows: &[(&str, &[i64])]) -> SketchStore {
        // Builds a store without going through parquet, for matrix-level
        // unit tests — exercises the same `sketch()`/`accessions()` surface
        // the parallel fill consumes.
        crate::sketch::SketchStore::from_rows_for_test(rows)
    }

    #[test]
    fn matrix_is_symmetric_with_zero_diagonal() {
        let store = store_from(&[("a", &[1, 2, 3]), ("b", &[2, 3, 4]), ("c", &[10, 20, 30])]);
        let mat = pairwise_jaccard_distance(&store);
        assert_eq!(mat.shape(), &[3, 3]);
        for i in 0..3 {
            assert_eq!(mat[[i, i]], 0.0);
        }
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(mat[[i, j]], mat[[j, i]]);
            }
        }
        let expected_ab = jaccard_distance(&[1, 2, 3], &[2, 3, 4]) as f32;
        assert_eq!(mat[[0, 1]], expected_ab);
    }

    #[test]
    fn single_accession_is_trivial() {
        let store = store_from(&[("a", &[1, 2, 3])]);
        let mat = pairwise_jaccard_distance(&store);
        assert_eq!(mat.shape(), &[1, 1]);
        assert_eq!(mat[[0, 0]], 0.0);
    }

    #[test]
    fn empty_store_yields_empty_matrix() {
        let store = store_from(&[]);
        let mat = pairwise_jaccard_distance(&store);
        assert_eq!(mat.shape(), &[0, 0]);
    }

    #[test]
    fn containment_matrix_has_unit_diagonal_and_is_asymmetric() {
        // "small" is fully contained in "big"; "big" is mostly not
        // contained in "small" — the matrix must reflect that asymmetry.
        let store = store_from(&[
            ("small", &[1, 2, 3]),
            ("big", &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        ]);
        let mat = pairwise_containment(&store);
        assert_eq!(mat.shape(), &[2, 2]);
        assert_eq!(mat[[0, 0]], 1.0);
        assert_eq!(mat[[1, 1]], 1.0);
        assert_eq!(mat[[0, 1]], 1.0); // all of "small" is in "big"
        assert!((mat[[1, 0]] - 0.3).abs() < 1e-6); // 3 of 10 of "big" is in "small"
        assert_ne!(mat[[0, 1]], mat[[1, 0]]);
    }

    #[test]
    fn containment_matrix_matches_direct_overlap_calls() {
        let store = store_from(&[
            ("a", &[1, 2, 3, 4, 5, 10, 20]),
            ("b", &[3, 4, 5, 6, 7, 20, 30]),
        ]);
        let mat = pairwise_containment(&store);
        let o = crate::distance::overlap(&[1, 2, 3, 4, 5, 10, 20], &[3, 4, 5, 6, 7, 20, 30]);
        assert!((mat[[0, 1]] - o.containment_a_in_b as f32).abs() < 1e-6);
        assert!((mat[[1, 0]] - o.containment_b_in_a as f32).abs() < 1e-6);
    }

    #[test]
    fn containment_single_accession_is_self_contained() {
        let store = store_from(&[("a", &[1, 2, 3])]);
        let mat = pairwise_containment(&store);
        assert_eq!(mat.shape(), &[1, 1]);
        assert_eq!(mat[[0, 0]], 1.0);
    }

    #[test]
    fn containment_empty_store_yields_empty_matrix() {
        let store = store_from(&[]);
        let mat = pairwise_containment(&store);
        assert_eq!(mat.shape(), &[0, 0]);
    }
}
