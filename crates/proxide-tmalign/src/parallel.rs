//! Cross-alignment batch scoring: dense pairwise TM-align over a set of
//! structures.
//!
//! Phase 3b of `.praxia/docs/specs/260729_proxide-tmalign-phases-2-5.md`.
//! The O(n²) all-pairs alignment is the bottleneck (each `tmalign_pair_serial`
//! call is itself far more expensive than a single sketch-distance kernel
//! invocation), so this follows `proxide_jaccard::matrix`'s row-parallel/
//! pair-serial split: row `i` is one `orx-parallel` task computing alignments
//! against all `j > i`, and the inner loop always calls
//! [`tmalign_pair_serial`] — never [`crate::pipeline::tmalign_pair`]'s own
//! `.into_par()`, to avoid nesting orx-parallel dispatches.

use crate::error::TmAlignError;
use crate::pipeline::tmalign_pair_serial;
use crate::structure::CaTrace;
use ndarray::Array2;
use orx_parallel::{IntoParIter, ParIter};

/// Dense n×n pairwise TM-score matrix over `traces`.
///
/// Following `proxide_jaccard::matrix::pairwise_containment`'s asymmetric
/// two-directional-value-per-pair pattern (one alignment pass per pair
/// produces both matrix entries, not two independent passes):
///
/// - `mat[[i, j]]` — TM-score of the `i`/`j` alignment normalized by trace
///   `i`'s own length (`tm_score_norm1` from a single
///   `tmalign_pair_serial(&traces[i].coords, &traces[j].coords)` call).
/// - `mat[[j, i]]` — the same alignment's TM-score normalized by trace
///   `j`'s length (`tm_score_norm2` from that same call).
///
/// `mat[[i, j]] != mat[[j, i]]` in general — this is not a symmetric
/// distance matrix, matching TM-align's own convention that `TM1`/`TM2`
/// differ whenever the two structures differ in length.
///
/// Diagonal is always `1.0` (every structure aligns perfectly against
/// itself), bypassing the general kernel the same way
/// `pairwise_containment`'s diagonal is fixed rather than computed.
///
/// # Errors
///
/// Returns [`TmAlignError::EmptyStructure`] if any trace has no residues,
/// checked up front (before dispatching any alignment work) so a single
/// bad input can't waste an O(n²) batch.
pub fn pairwise_tm_scores(traces: &[CaTrace]) -> Result<Array2<f32>, TmAlignError> {
    let n = traces.len();
    if traces.iter().any(CaTrace::is_empty) {
        return Err(TmAlignError::EmptyStructure);
    }

    let mut mat = Array2::<f32>::zeros((n, n));
    for i in 0..n {
        mat[[i, i]] = 1.0;
    }
    if n < 2 {
        return Ok(mat);
    }

    let par = (0..n)
        .into_par()
        .map(|i| -> Result<Vec<(f32, f32)>, TmAlignError> {
            ((i + 1)..n)
                .map(|j| {
                    tmalign_pair_serial(&traces[i].coords, &traces[j].coords)
                        .map(|result| (result.tm_score_norm1, result.tm_score_norm2))
                })
                .collect()
        });
    #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
    let par = par.num_threads(proxide_parallel_rt::num_threads());
    let rows: Vec<Result<Vec<(f32, f32)>, TmAlignError>> = par.collect();

    for (i, row) in rows.into_iter().enumerate() {
        for (offset, (norm1, norm2)) in row?.into_iter().enumerate() {
            let j = i + 1 + offset;
            mat[[i, j]] = norm1;
            mat[[j, i]] = norm2;
        }
    }

    Ok(mat)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seq::three_to_one;
    use nalgebra::Vector3;
    use proxide_core::processing::residues::ResidueId;

    fn make_trace(coords: Vec<Vector3<f32>>) -> CaTrace {
        let n = coords.len();
        CaTrace {
            seq: vec![three_to_one("ALA") as u8; n],
            res_ids: (0..n as i32)
                .map(|i| ResidueId {
                    chain_id: "A".to_string(),
                    res_id: i + 1,
                    insertion_code: ' ',
                })
                .collect(),
            coords,
        }
    }

    /// Non-collinear helical coordinate generator, same shape as
    /// `pipeline.rs`'s own unit-test helper, parameterized by a seed offset
    /// so distinct structures in a batch aren't literally identical.
    fn helix(n: usize, seed: f32) -> CaTrace {
        make_trace(
            (0..n)
                .map(|i| {
                    let angle = (i as f32 * 100.0 + seed).to_radians();
                    let z = i as f32 * 1.5;
                    Vector3::new(2.3 * angle.cos(), 2.3 * angle.sin(), z)
                })
                .collect(),
        )
    }

    #[test]
    fn matrix_shape_and_unit_diagonal() {
        let traces = vec![helix(8, 0.0), helix(6, 30.0), helix(10, 60.0)];
        let mat = pairwise_tm_scores(&traces).expect("non-empty traces");
        assert_eq!(mat.shape(), &[3, 3]);
        for i in 0..3 {
            assert_eq!(mat[[i, i]], 1.0);
        }
    }

    #[test]
    fn matches_direct_serial_calls() {
        let traces = vec![
            helix(8, 0.0),
            helix(6, 30.0),
            helix(10, 60.0),
            helix(7, 90.0),
        ];
        let mat = pairwise_tm_scores(&traces).expect("non-empty traces");

        for i in 0..traces.len() {
            for j in (i + 1)..traces.len() {
                let direct = tmalign_pair_serial(&traces[i].coords, &traces[j].coords)
                    .expect("non-empty traces");
                assert_eq!(
                    mat[[i, j]],
                    direct.tm_score_norm1,
                    "mat[{i},{j}] vs tm_score_norm1"
                );
                assert_eq!(
                    mat[[j, i]],
                    direct.tm_score_norm2,
                    "mat[{j},{i}] vs tm_score_norm2"
                );
            }
        }
    }

    #[test]
    fn self_pair_is_near_symmetric_high_score() {
        // Two copies of the same structure: not required to be bit-identical
        // to the fixed 1.0 diagonal convention (that's only for i==j), but
        // both directions should independently land near 1.0.
        let a = helix(8, 0.0);
        let b = helix(8, 0.0);
        let traces = vec![a, b];
        let mat = pairwise_tm_scores(&traces).expect("non-empty traces");
        assert!((mat[[0, 1]] - 1.0).abs() < 1e-3, "got {}", mat[[0, 1]]);
        assert!((mat[[1, 0]] - 1.0).abs() < 1e-3, "got {}", mat[[1, 0]]);
    }

    #[test]
    fn asymmetric_for_different_lengths() {
        let traces = vec![helix(20, 0.0), helix(6, 0.0)];
        let mat = pairwise_tm_scores(&traces).expect("non-empty traces");
        assert_ne!(mat[[0, 1]], mat[[1, 0]]);
    }

    #[test]
    fn single_trace_is_trivial() {
        let traces = vec![helix(8, 0.0)];
        let mat = pairwise_tm_scores(&traces).expect("non-empty traces");
        assert_eq!(mat.shape(), &[1, 1]);
        assert_eq!(mat[[0, 0]], 1.0);
    }

    #[test]
    fn empty_input_yields_empty_matrix() {
        let mat = pairwise_tm_scores(&[]).expect("empty slice is not an error");
        assert_eq!(mat.shape(), &[0, 0]);
    }

    #[test]
    fn any_empty_structure_is_an_error() {
        let traces = vec![helix(8, 0.0), make_trace(vec![])];
        assert!(matches!(
            pairwise_tm_scores(&traces),
            Err(TmAlignError::EmptyStructure)
        ));
    }
}
