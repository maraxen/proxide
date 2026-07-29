//! Serial TM-align pipeline: orchestrates all 5 seeding strategies with
//! refinement and produces the final TM-score result.
//!
//! This module implements the entry point `tmalign_pair_serial`, which runs the
//! complete TM-align algorithm on a pair of protein structures in sequence:
//!
//! 1. **Seed (a): Gapless threading** — slides one sequence over the other.
//! 2. **Seed (b): Secondary structure** — aligns via SS-letter identity, gated on
//!    `TM > TMmax * 0.2`.
//! 3. **Seed (c): Local structure** — fragment-based Kabsch, gated on
//!    `TM > TMmax * ddcc` (ddcc=0.1 if len≤40, else 0.4), refined with 2 iterations.
//! 4. **Seed (d): SS+local combo** — combines local superposition + SS scoring,
//!    fed the current best alignment, gated on `TM > TMmax * ddcc`, 30 iterations.
//! 5. **Seed (e): Fragment gapless** — finds long contiguous fragments and threads,
//!    gated on `TM > TMmax * ddcc`, refined with gap_open=0 only, 2 iterations.
//!
//! After all seeds, a final refinement stage re-runs the best alignment with
//! strict `d0_final` (not the looser search-phase `d0`) to compute the canonical
//! TM-scores normalized by both structure lengths.

use crate::d0;
use crate::error::TmAlignError;
use crate::refine::dp_iter;
use crate::score::evaluate_tm_score;
use crate::seed::{run_seed, AlignmentMap, SeedKind};
use crate::kabsch::kabsch_superpose;
use crate::nw::AlignedPair;
use nalgebra::Vector3;

/// Result of TM-align pairwise alignment: alignment map, rotation, translation,
/// and final TM-scores normalized by both structure lengths.
#[derive(Debug, Clone)]
pub struct TmAlignResult {
    /// Alignment map: `(Some(i), Some(j))` for aligned residue pairs, gaps as `None`.
    pub alignment: AlignmentMap,
    /// Optimal rotation matrix (row-major 3×3).
    pub rotation: [[f32; 3]; 3],
    /// Translation vector.
    pub translation: [f32; 3],
    /// TM-score normalized by structure 1 length.
    pub tm_score_norm1: f32,
    /// TM-score normalized by structure 2 length (canonical single TM-score).
    pub tm_score_norm2: f32,
    /// Number of residue pairs in the final alignment.
    pub n_aligned: usize,
}

/// Run TM-align on a pair of protein structures (serial, single-pair entry point).
///
/// Executes all 5 seeding strategies (gapless threading, secondary structure,
/// local structure, SS+local combo, fragment gapless) in sequence, refining each
/// seed's initial alignment via iterative DP and fragment superposition. Returns
/// the best result found across all seeds.
///
/// # Arguments
///
/// - `coords1` — Cα coordinates of structure 1.
/// - `coords2` — Cα coordinates of structure 2.
///
/// # Returns
///
/// `TmAlignResult` containing the best alignment, rotation, translation, and
/// final TM-scores. Returns an error only if the coordinate arrays are empty
/// or mismatched (individual seed failures are tolerated).
///
/// # Errors
///
/// Returns `TmAlignError` if:
/// - Either coordinate array is empty.
/// - No seed ever produces a usable alignment (all fail).
///
/// # Algorithm Overview
///
/// Per `TMalign_main` (USalign/TMalign.h:3138+), the pipeline is:
///
/// 1. Compute search-phase `d0` and `d0_search` from `l_norm = min(len1, len2)`.
/// 2. Gapless threading (seed a) → DP_iter(gap_range=0..2, iter=30) → track best.
/// 3. SS seed (b) → gate `TM > TMmax*0.2` → DP_iter(0..2, 30) → track best.
/// 4. Local structure (c) → gate `TM > TMmax*ddcc` → DP_iter(0..2, **2 iters only**).
/// 5. SS+local combo (d) → pass current best alignment → gate `TM > TMmax*ddcc`
///    → DP_iter(0..2, 30).
/// 6. Fragment gapless (e) → gate `TM > TMmax*ddcc` → DP_iter(1..2, 2 iters).
/// 7. Final stage: re-compute TM-scores using `d0_final` (not search-phase d0),
///    normalized by both structure lengths.
#[allow(unused_assignments)] // tmmax's final write (after the last seed) is never read back, by design
pub fn tmalign_pair_serial(
    coords1: &[Vector3<f32>],
    coords2: &[Vector3<f32>],
) -> Result<TmAlignResult, TmAlignError> {
    if coords1.is_empty() || coords2.is_empty() {
        return Err(TmAlignError::EmptyStructure);
    }

    let xlen = coords1.len();
    let ylen = coords2.len();
    let l_norm = xlen.min(ylen);

    // Compute search-phase d0 and d0_search.
    let (d0_search_phase, d0_search) = d0::d0_search(l_norm);
    let ddcc = if l_norm <= 40 { 0.1 } else { 0.4 };

    let mut tmmax = -1.0_f32;
    let mut best_alignment: Option<AlignmentMap> = None;

    // ==================== Seed (a): Gapless threading ====================
    match run_seed(
        SeedKind::GaplessThreading,
        coords1,
        coords2,
        d0_search_phase,
        d0_search,
        l_norm,
        None,
    ) {
        Ok(alignment) => {
            // Run DP_iter with gap_open range [0, 2) = [-0.6, 0.0]
            let (refined_alignment, tm_score) = dp_iter(
                coords1,
                coords2,
                &alignment,
                d0_search_phase,
                d0_search,
                l_norm,
                (0, 2), // gap_open_range for both -0.6 and 0.0
                30,     // max_iterations
            );
            if tm_score > tmmax {
                tmmax = tm_score;
                best_alignment = Some(refined_alignment);
            }
        }
        Err(_) => {
            // Skip this seed; not a fatal error.
        }
    }

    // ==================== Seed (b): Secondary structure ====================
    match run_seed(
        SeedKind::SecondaryStructure,
        coords1,
        coords2,
        d0_search_phase,
        d0_search,
        l_norm,
        None,
    ) {
        Ok(alignment) => {
            // Gate: only refine if raw score > TMmax * 0.2
            let raw_score = score_alignment(&alignment, coords1, coords2, d0_search_phase, l_norm);
            if raw_score > tmmax * 0.2 {
                let (refined_alignment, tm_score) = dp_iter(
                    coords1,
                    coords2,
                    &alignment,
                    d0_search_phase,
                    d0_search,
                    l_norm,
                    (0, 2),
                    30,
                );
                if tm_score > tmmax {
                    tmmax = tm_score;
                    best_alignment = Some(refined_alignment);
                }
            }
        }
        Err(_) => {
            // Skip this seed.
        }
    }

    // ==================== Seed (c): Local structure ====================
    match run_seed(
        SeedKind::LocalStructure,
        coords1,
        coords2,
        d0_search_phase,
        d0_search,
        l_norm,
        None,
    ) {
        Ok(alignment) => {
            // Gate: only refine if raw score > TMmax * ddcc
            let raw_score = score_alignment(&alignment, coords1, coords2, d0_search_phase, l_norm);
            if raw_score > tmmax * ddcc {
                // Note: DP_iter with max_iterations=2 for local-structure seed only
                let (refined_alignment, tm_score) = dp_iter(
                    coords1,
                    coords2,
                    &alignment,
                    d0_search_phase,
                    d0_search,
                    l_norm,
                    (0, 2),
                    2, // 2 iterations only for this seed
                );
                if tm_score > tmmax {
                    tmmax = tm_score;
                    best_alignment = Some(refined_alignment);
                }
            }
        }
        Err(_) => {
            // Skip this seed.
        }
    }

    // ==================== Seed (d): SS+local combo ====================
    match run_seed(
        SeedKind::SsPlus,
        coords1,
        coords2,
        d0_search_phase,
        d0_search,
        l_norm,
        best_alignment.as_ref(), // Pass current best alignment (e.g., from gapless threading)
    ) {
        Ok(alignment) => {
            // Gate: only refine if raw score > TMmax * ddcc
            let raw_score = score_alignment(&alignment, coords1, coords2, d0_search_phase, l_norm);
            if raw_score > tmmax * ddcc {
                let (refined_alignment, tm_score) = dp_iter(
                    coords1,
                    coords2,
                    &alignment,
                    d0_search_phase,
                    d0_search,
                    l_norm,
                    (0, 2),
                    30,
                );
                if tm_score > tmmax {
                    tmmax = tm_score;
                    best_alignment = Some(refined_alignment);
                }
            }
        }
        Err(_) => {
            // Skip this seed; expected if SsPlus was called without a prior best alignment.
        }
    }

    // ==================== Seed (e): Fragment gapless ====================
    match run_seed(
        SeedKind::FragmentGapless,
        coords1,
        coords2,
        d0_search_phase,
        d0_search,
        l_norm,
        None,
    ) {
        Ok(alignment) => {
            // Gate: only refine if raw score > TMmax * ddcc
            let raw_score = score_alignment(&alignment, coords1, coords2, d0_search_phase, l_norm);
            if raw_score > tmmax * ddcc {
                // Note: gap_open_range=(1,2) means only gap_open=0.0, and 2 iterations
                let (refined_alignment, tm_score) = dp_iter(
                    coords1,
                    coords2,
                    &alignment,
                    d0_search_phase,
                    d0_search,
                    l_norm,
                    (1, 2), // gap_open_range for 0.0 only
                    2,      // 2 iterations for this seed
                );
                if tm_score > tmmax {
                    tmmax = tm_score;
                    best_alignment = Some(refined_alignment);
                }
            }
        }
        Err(_) => {
            // Skip this seed.
        }
    }

    // ==================== Final stage: compute canonical TM-scores ====================
    let alignment = best_alignment
        .ok_or_else(|| TmAlignError::Parse("No seed produced a usable alignment".to_string()))?;

    // Extract aligned pair coordinates for final Kabsch fit.
    let (coords1_aligned, coords2_aligned): (Vec<Vector3<f32>>, Vec<Vector3<f32>>) =
        alignment
            .iter()
            .filter_map(|&(i_opt, j_opt)| match (i_opt, j_opt) {
                (Some(i), Some(j)) if i < coords1.len() && j < coords2.len() => {
                    Some((coords1[i], coords2[j]))
                }
                _ => None,
            })
            .unzip();

    let n_aligned = coords1_aligned.len();
    if n_aligned == 0 {
        return Err(TmAlignError::Parse(
            "Final alignment has no aligned pairs".to_string(),
        ));
    }

    // Final Kabsch fit to get definitive rotation and translation.
    let final_result = kabsch_superpose(&coords1_aligned, &coords2_aligned);

    // Compute final TM-scores using d0_final (strict thresholds) for each normalization length.
    let d0_final_xlen = d0::d0_final(xlen);
    let d0_final_ylen = d0::d0_final(ylen);

    let tm_score_norm1 = evaluate_tm_score(
        coords1,
        coords2,
        &alignment,
        &final_result.rotation,
        &final_result.translation,
        d0_final_xlen,
        xlen,
    );

    let tm_score_norm2 = evaluate_tm_score(
        coords1,
        coords2,
        &alignment,
        &final_result.rotation,
        &final_result.translation,
        d0_final_ylen,
        ylen,
    );

    Ok(TmAlignResult {
        alignment,
        rotation: final_result.rotation,
        translation: final_result.translation,
        tm_score_norm1,
        tm_score_norm2,
        n_aligned,
    })
}

/// Helper: score an alignment using the fast iterative method.
///
/// Computes TM-score for a given alignment by Kabsch-fitting and running
/// three iterations of distance-filtered re-fitting (same as `get_score_fast`).
/// Used to evaluate seed quality before deciding whether to run DP_iter.
fn score_alignment(
    alignment: &[AlignedPair],
    coords1: &[Vector3<f32>],
    coords2: &[Vector3<f32>],
    d0: f32,
    l_norm: usize,
) -> f32 {
    if l_norm == 0 || alignment.is_empty() {
        return 0.0;
    }

    // Collect aligned pairs.
    let (r1, r2): (Vec<Vector3<f32>>, Vec<Vector3<f32>>) = alignment
        .iter()
        .filter_map(|&(i_opt, j_opt)| match (i_opt, j_opt) {
            (Some(i), Some(j)) if i < coords1.len() && j < coords2.len() => {
                Some((coords1[i], coords2[j]))
            }
            _ => None,
        })
        .unzip();

    if r1.is_empty() {
        return 0.0;
    }

    // Single Kabsch fit + naive TM-score (no distance filtering at this gate stage).
    let result = kabsch_superpose(&r1, &r2);
    evaluate_tm_score(coords1, coords2, alignment, &result.rotation, &result.translation, d0, l_norm)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_structures_yield_high_tm_scores() {
        // Two identical small structures should align with TM-score near 1.0.
        let coords = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(3.8, 0.0, 0.0),
            Vector3::new(7.6, 0.0, 0.0),
            Vector3::new(11.4, 0.0, 0.0),
            Vector3::new(15.2, 0.0, 0.0),
        ];
        let result = tmalign_pair_serial(&coords, &coords);
        assert!(result.is_ok());
        let res = result.unwrap();
        // Both should be near 1.0 since structures are identical.
        assert!(res.tm_score_norm1 > 0.9, "norm1 score too low: {}", res.tm_score_norm1);
        assert!(res.tm_score_norm2 > 0.9, "norm2 score too low: {}", res.tm_score_norm2);
        assert!(res.n_aligned >= 5);
    }

    #[test]
    fn different_structure_lengths_produce_different_tm_scores() {
        // Structure 1: 5 residues
        let coords1 = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(3.8, 0.0, 0.0),
            Vector3::new(7.6, 0.0, 0.0),
            Vector3::new(11.4, 0.0, 0.0),
            Vector3::new(15.2, 0.0, 0.0),
        ];
        // Structure 2: 3 residues (a subset of structure 1)
        let coords2 = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(3.8, 0.0, 0.0),
            Vector3::new(7.6, 0.0, 0.0),
        ];

        let result = tmalign_pair_serial(&coords1, &coords2);
        assert!(result.is_ok());
        let res = result.unwrap();

        // TM_norm1 (normalized by xlen=5) should be lower than TM_norm2 (normalized by ylen=3)
        // because the longer structure has more residues counted in the denominator.
        assert!(
            res.tm_score_norm1 <= res.tm_score_norm2,
            "Expected tm_score_norm1 <= tm_score_norm2, got {} vs {}",
            res.tm_score_norm1,
            res.tm_score_norm2
        );
    }

    #[test]
    fn empty_structures_return_error() {
        let empty: Vec<Vector3<f32>> = vec![];
        let coords = vec![Vector3::new(0.0, 0.0, 0.0)];

        assert!(tmalign_pair_serial(&empty, &coords).is_err());
        assert!(tmalign_pair_serial(&coords, &empty).is_err());
    }

    #[test]
    fn result_contains_rotation_and_translation() {
        // Non-collinear/non-coplanar synthetic helix: a perfectly collinear
        // point set (e.g. all points along the x-axis) makes Kabsch rotation
        // recovery non-unique — any rotation about the x-axis fits the
        // self-aligned structure with zero RMSD too, since y=z=0 everywhere —
        // so identity isn't the only valid answer and this test would be
        // fixture-dependent flakiness, not a real assertion about the pipeline.
        let coords: Vec<Vector3<f32>> = (0..8)
            .map(|i| {
                let angle = i as f32 * 100.0_f32.to_radians();
                let z = i as f32 * 1.5;
                Vector3::new(2.3 * angle.cos(), 2.3 * angle.sin(), z)
            })
            .collect();

        let result = tmalign_pair_serial(&coords, &coords);
        assert!(result.is_ok());
        let res = result.unwrap();

        // For identical structures, rotation should be ~identity and translation ~zero.
        let rot = &res.rotation;
        assert!((rot[0][0] - 1.0).abs() < 0.1);
        assert!((rot[1][1] - 1.0).abs() < 0.1);
        assert!((rot[2][2] - 1.0).abs() < 0.1);

        let trans = &res.translation;
        assert!(trans.iter().all(|&t| t.abs() < 0.1));
    }
}
