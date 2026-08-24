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

use crate::d0::{d0_final, d0_search};
use crate::error::TmAlignError;
use crate::pipeline::tmalign_pair_serial;
use crate::score::get_score_fast;
use crate::seed::AlignmentMap;
use crate::structure::CaTrace;
use nalgebra::Vector3;
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

/// One-vs-many TM-scores under a **fixed residue correspondence**.
///
/// [`pairwise_tm_scores`] is O(n²) and re-derives the residue correspondence from
/// scratch for every pair, via the full five-strategy seeding search. Both are the
/// wrong shape for the motivating case: scoring one query conformation against every
/// frame of an MD trajectory, where all candidates share the query's topology and the
/// correspondence is therefore the identity, already known. Running a seeding search to
/// rediscover residue `i` maps to residue `i` costs orders of magnitude more than the
/// scoring it feeds, and a search that returns anything else on same-topology input
/// would be a bug we would rather not silently absorb.
///
/// So: identity correspondence, then [`get_score_fast`]'s three-iteration Kabsch
/// refinement per candidate, parallel over candidates. O(n) in the candidate count.
///
/// All candidates must have exactly `query.len()` residues — that is what "same
/// topology" means here, and a length mismatch is a caller error rather than something
/// to pad or truncate around. Use [`pairwise_tm_scores`] or
/// [`crate::pipeline::tmalign_pair_serial`] when the correspondence is genuinely
/// unknown.
///
/// Scores are normalized by `query.len()`, so a candidate identical to the query scores
/// `1.0`.
///
/// # Errors
///
/// [`TmAlignError::EmptyStructure`] if `query` is empty, and
/// [`TmAlignError::LengthMismatch`] if any candidate's length differs from the query's.
pub fn tm_scores_fixed_correspondence(
    query: &[Vector3<f32>],
    candidates: &[Vec<Vector3<f32>>],
) -> Result<Vec<f32>, TmAlignError> {
    let n_res = query.len();
    if n_res == 0 {
        return Err(TmAlignError::EmptyStructure);
    }
    // Checked up front, before any alignment work, for the same reason
    // pairwise_tm_scores validates eagerly: one bad candidate should not waste the
    // whole batch.
    if let Some(bad) = candidates.iter().find(|c| c.len() != n_res) {
        return Err(TmAlignError::LengthMismatch {
            expected: n_res,
            found: bad.len(),
        });
    }
    if candidates.is_empty() {
        return Ok(Vec::new());
    }

    let alignment: AlignmentMap = (0..n_res).map(|i| (Some(i), Some(i))).collect();
    let d0 = d0_final(n_res);
    let (_, d0_srch) = d0_search(n_res);

    let par = candidates
        .into_par()
        .map(|cand| get_score_fast(query, cand, &alignment, d0, d0_srch, n_res));
    #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
    let par = par.num_threads(proxide_parallel_rt::num_threads());
    Ok(par.collect())
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

    // ---------------- tm_scores_fixed_correspondence ----------------

    #[test]
    fn fixed_correspondence_self_scores_one() {
        // The invariant that makes the whole donor-selection path trustworthy: a
        // candidate identical to the query must score exactly 1.0. If this drifts, a
        // frame is no longer its own nearest neighbour and every selection below is
        // suspect -- cheap to check, and it catches an inverted d0/l_norm convention.
        let q = helix(30, 0.0);
        let scores =
            tm_scores_fixed_correspondence(&q.coords, &[q.coords.clone()]).expect("valid input");
        assert_eq!(scores.len(), 1);
        assert!(
            (scores[0] - 1.0).abs() < 1e-4,
            "self-score was {} not 1.0",
            scores[0]
        );
    }

    #[test]
    fn fixed_correspondence_is_invariant_to_rigid_motion() {
        // TM-score superposes, so translating a candidate must not change its score.
        let q = helix(24, 0.0);
        let shifted: Vec<Vector3<f32>> = q
            .coords
            .iter()
            .map(|p| p + Vector3::new(17.0, -4.0, 9.5))
            .collect();
        let scores = tm_scores_fixed_correspondence(&q.coords, &[shifted]).expect("valid input");
        assert!(
            (scores[0] - 1.0).abs() < 1e-4,
            "translated copy scored {} not 1.0",
            scores[0]
        );
    }

    /// Bend the tail half of a trace outward by `amplitude` A.
    ///
    /// Note `helix(n, seed)` cannot be used to make conformationally distinct traces:
    /// varying its seed rotates every point by the same angle about z, so the results
    /// are congruent and TM-score is correctly 1.0 for all of them. A real
    /// conformational difference has to be a non-rigid deformation.
    fn bent(base: &CaTrace, amplitude: f32) -> Vec<Vector3<f32>> {
        let n = base.coords.len();
        base.coords
            .iter()
            .enumerate()
            .map(|(i, p)| {
                if i * 2 < n {
                    *p
                } else {
                    let t = (i * 2 - n) as f32 / n as f32;
                    p + Vector3::new(amplitude * t, 0.0, 0.0)
                }
            })
            .collect()
    }

    #[test]
    fn fixed_correspondence_ranks_nearer_conformations_higher() {
        let q = helix(40, 0.0);
        let near = bent(&q, 1.0);
        let far = bent(&q, 12.0);
        let scores = tm_scores_fixed_correspondence(&q.coords, &[near, far]).expect("valid input");
        assert!(
            scores[0] > scores[1],
            "near ({}) should outrank far ({})",
            scores[0],
            scores[1]
        );
    }

    #[test]
    fn fixed_correspondence_score_decreases_monotonically_with_deformation() {
        // Stronger than the ranking test: the metric must be monotone in the
        // deformation it is meant to measure, or "nearest" is not meaningful.
        let q = helix(40, 0.0);
        let cands: Vec<Vec<Vector3<f32>>> = [0.0f32, 1.0, 3.0, 6.0, 12.0]
            .iter()
            .map(|a| bent(&q, *a))
            .collect();
        let scores = tm_scores_fixed_correspondence(&q.coords, &cands).expect("valid input");
        for w in scores.windows(2) {
            assert!(
                w[0] >= w[1],
                "scores not monotonically decreasing: {scores:?}"
            );
        }
        assert!((scores[0] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn fixed_correspondence_rejects_length_mismatch() {
        // Silently scoring a differently-sized candidate would compare residue i to a
        // different residue i and still return a plausible number.
        let q = helix(30, 0.0);
        let wrong = helix(29, 0.0);
        assert!(matches!(
            tm_scores_fixed_correspondence(&q.coords, &[wrong.coords]),
            Err(TmAlignError::LengthMismatch {
                expected: 30,
                found: 29
            })
        ));
    }

    #[test]
    fn fixed_correspondence_rejects_empty_query() {
        assert!(matches!(
            tm_scores_fixed_correspondence(&[], &[helix(8, 0.0).coords]),
            Err(TmAlignError::EmptyStructure)
        ));
    }

    #[test]
    fn fixed_correspondence_empty_candidate_list_is_empty_not_an_error() {
        let q = helix(8, 0.0);
        let scores = tm_scores_fixed_correspondence(&q.coords, &[]).expect("not an error");
        assert!(scores.is_empty());
    }

    #[test]
    fn fixed_correspondence_agrees_with_full_alignment_on_same_topology() {
        // The fast path exists only because the correspondence is already known. If it
        // disagreed with the full seeding search on input where that search should
        // recover the identity anyway, the shortcut would be changing the answer rather
        // than skipping redundant work.
        let q = helix(40, 0.0);
        let c = helix(40, 5.0);
        let fast =
            tm_scores_fixed_correspondence(&q.coords, &[c.coords.clone()]).expect("valid input")[0];
        let full = tmalign_pair_serial(&q.coords, &c.coords).expect("valid input");
        assert!(
            (fast - full.tm_score_norm1).abs() < 0.05,
            "fixed-correspondence {fast} vs full-search {}",
            full.tm_score_norm1
        );
    }
}
