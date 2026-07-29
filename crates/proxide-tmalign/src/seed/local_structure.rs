//! Local-structure superposition seed strategy.
//!
//! Implements USalign's `get_initial5()` (TMalign.h:943-1040), which performs
//! fragment-based Kabsch superposition followed by full-length rotation-aware
//! Needleman-Wunsch dynamic programming alignment.
//!
//! For two fragment lengths (20 and 100 residues, capped at aL/3 and aL/2
//! respectively, where aL = min(xlen, ylen)), the algorithm iterates over
//! all starting positions in both sequences (with specified jump steps),
//! extracts fragments, fits them via Kabsch, then runs rotation-aware DP
//! to produce a full-length alignment. The highest-scoring alignment is
//! returned.

use super::AlignmentMap;
use crate::error::TmAlignError;
use crate::kabsch::kabsch_superpose;
use crate::nwdp_tm::nwdp_tm;
use crate::score::get_score_fast;
use nalgebra::Vector3;

/// Local-structure superposition seed: fragment-based Kabsch + rotation-aware DP.
///
/// For each of two fragment lengths (20 and 100 residues, capped appropriately),
/// iterates over all starting positions in both coordinate sets with specified
/// jump steps. For each start pair:
///
/// 1. Extracts an `n_frag`-length fragment from each coordinate set.
/// 2. Fits the fragment pair via Kabsch superposition.
/// 3. Runs rotation-aware Needleman-Wunsch DP to produce a full-length alignment.
/// 4. Scores the alignment via `get_score_fast()`.
/// 5. Keeps the best-scoring alignment.
///
/// Returns `TmAlignError` if no candidate improves score > 0.
///
/// # Arguments
///
/// - `coords1` — Cα coordinates of structure 1 (query).
/// - `coords2` — Cα coordinates of structure 2 (target).
/// - `d0` — Final TM-score distance threshold.
/// - `d0_search` — Search-phase distance threshold.
/// - `l_norm` — Normalization length for TM-score calculation.
///
/// # Returns
///
/// The best-scoring alignment, or an error if no valid alignment is found.
///
/// # Errors
///
/// Returns `TmAlignError::Parse` if:
/// - Either coordinate set has length < 3.
/// - No candidate alignment ever achieves a score > 0.
pub fn get_initial5(
    coords1: &[Vector3<f32>],
    coords2: &[Vector3<f32>],
    d0: f32,
    d0_search: f32,
    l_norm: usize,
) -> Result<AlignmentMap, TmAlignError> {
    let xlen = coords1.len();
    let ylen = coords2.len();

    if xlen < 3 || ylen < 3 {
        return Err(TmAlignError::Parse(
            "Sequence is too short < 3".to_string(),
        ));
    }

    // Hardcoded parameters (matched to USalign defaults):
    // - fast_opt = false (no jump step multiplication)
    // - D0_MIN = 0.5
    let d0_min = 0.5_f32;

    // Compute d01 and d02 for the DP score computation.
    let d01 = (d0 + 1.5).max(d0_min);
    let d02 = d01 * d01;

    // Compute aL = min(xlen, ylen) for fragment length capping.
    let al = xlen.min(ylen);

    // Fragment lengths: cap 20 at aL/3 and 100 at aL/2.
    let n_frag = [
        20.min(al / 3),
        100.min(al / 2),
    ];

    // Jump steps based on chain length, capped at len/3.
    let n_jump1 = compute_jump_step(xlen);
    let n_jump2 = compute_jump_step(ylen);

    let mut best_score = 0.0_f32;
    let mut best_alignment = Vec::new();

    // Iterate over both fragment lengths.
    for &frag_len in &n_frag {
        if frag_len < 3 {
            // Skip fragments that are too short for meaningful alignment.
            continue;
        }

        // Compute the number of valid starting positions.
        let m1 = xlen.saturating_sub(frag_len - 1);
        let m2 = ylen.saturating_sub(frag_len - 1);

        // Iterate over starting positions in coords1.
        let mut i = 0;
        while i < m1 {
            // Iterate over starting positions in coords2.
            let mut j = 0;
            while j < m2 {
                // Extract fragment from coords1 and coords2.
                let frag1 = &coords1[i..i + frag_len];
                let frag2 = &coords2[j..j + frag_len];

                // Fit the fragment pair via Kabsch.
                let kabsch_result = kabsch_superpose(frag1, frag2);

                // If Kabsch fails, skip this candidate.
                if kabsch_result.rmsd.is_infinite() {
                    j += n_jump2;
                    continue;
                }

                // Extract rotation and translation.
                let rotation = &kabsch_result.rotation;
                let translation = &kabsch_result.translation;

                // Run rotation-aware DP with score computed on the fly.
                // The score closure computes: 1 / (1 + dist / d02)
                let score_fn = |i_idx: usize, j_idx: usize| -> f32 {
                    let p1 = coords1[i_idx];
                    let p2 = coords2[j_idx];

                    // Apply rotation + translation to p1.
                    let rotated = apply_transform(p1, rotation, translation);

                    // Compute squared distance.
                    let diff = rotated - p2;
                    let sq_dist = diff.norm_squared();

                    // Compute TM-score-weighted similarity: 1 / (1 + dist / d02).
                    // Note: dist is the *distance*, not squared distance.
                    let dist = sq_dist.sqrt();
                    1.0 / (1.0 + dist / d02)
                };

                // Run the shared simplified-Gotoh NWDP_TM core with gap_open = 0.0
                // (matches `nw::needleman_wunsch_affine`'s behavior exactly when
                // gap_open=gap_extend=0, since both DP variants become
                // penalty-free in that case — using the shared core here keeps
                // every NWDP_TM call site on one implementation instead of two).
                let alignment = nwdp_tm(xlen, ylen, 0.0, score_fn);

                // Score the alignment using get_score_fast.
                // This function re-fits the rotation and returns the best TM-score.
                let score = get_score_fast(coords1, coords2, &alignment, d0, d0_search, l_norm);

                // Keep the best alignment.
                if score > best_score {
                    best_score = score;
                    best_alignment = alignment;
                }

                j += n_jump2;
            }
            i += n_jump1;
        }
    }

    // Return the best alignment if one was found, otherwise return an error.
    if best_score > 0.0 && !best_alignment.is_empty() {
        Ok(best_alignment)
    } else {
        Err(TmAlignError::Parse(
            "No valid alignment found by local-structure seed".to_string(),
        ))
    }
}

/// Compute the jump step for a given sequence length.
///
/// Per USalign TMalign.h:960-971 (and 973-984 for the second sequence),
/// the jump step is determined by chain length:
///
/// - 45 if length > 250
/// - 35 if length > 200
/// - 25 if length > 150
/// - 15 otherwise
///
/// Capped at length / 3.
fn compute_jump_step(len: usize) -> usize {
    let base_jump = if len > 250 {
        45
    } else if len > 200 {
        35
    } else if len > 150 {
        25
    } else {
        15
    };

    // Cap at len / 3.
    base_jump.min(len / 3)
}

/// Apply rotation and translation to a single point: `R·p + t`.
///
/// Helper for rotating a single point using a row-major rotation matrix.
fn apply_transform(
    p: Vector3<f32>,
    rotation: &[[f32; 3]; 3],
    translation: &[f32; 3],
) -> Vector3<f32> {
    Vector3::new(
        translation[0] + rotation[0][0] * p.x + rotation[0][1] * p.y + rotation[0][2] * p.z,
        translation[1] + rotation[1][0] * p.x + rotation[1][1] * p.y + rotation[1][2] * p.z,
        translation[2] + rotation[2][0] * p.x + rotation[2][1] * p.y + rotation[2][2] * p.z,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test that identical coordinate sets yield a high-scoring alignment.
    #[test]
    fn identical_structures_yield_good_alignment() {
        let coords = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(2.0, 0.0, 0.0),
            Vector3::new(3.0, 0.0, 0.0),
            Vector3::new(4.0, 0.0, 0.0),
            Vector3::new(5.0, 0.0, 0.0),
            Vector3::new(6.0, 0.0, 0.0),
            Vector3::new(7.0, 0.0, 0.0),
            Vector3::new(8.0, 0.0, 0.0),
            Vector3::new(9.0, 0.0, 0.0),
        ];

        let d0 = 3.0;
        let d0_search = 8.0;
        let l_norm = coords.len();

        let result = get_initial5(&coords, &coords, d0, d0_search, l_norm);
        assert!(result.is_ok(), "Should produce an alignment for identical structures");

        let alignment = result.unwrap();
        assert!(!alignment.is_empty(), "Alignment should not be empty");

        // For identical structures, we expect mostly diagonal (i, i) pairs.
        let diagonal_pairs: Vec<_> = alignment
            .iter()
            .filter_map(|(i_opt, j_opt)| match (i_opt, j_opt) {
                (Some(i), Some(j)) if i == j => Some((*i, *j)),
                _ => None,
            })
            .collect();

        assert!(!diagonal_pairs.is_empty(), "Should have some diagonal pairs");
    }

    /// Test that very short sequences return an error.
    #[test]
    fn short_sequence_error() {
        let coords_short = vec![Vector3::new(0.0, 0.0, 0.0)];
        let coords_ok = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(2.0, 0.0, 0.0),
        ];

        let d0 = 2.0;
        let d0_search = 5.0;
        let l_norm = 3;

        let result = get_initial5(&coords_short, &coords_ok, d0, d0_search, l_norm);
        assert!(result.is_err(), "Should error for short sequence");

        let result = get_initial5(&coords_ok, &coords_short, d0, d0_search, l_norm);
        assert!(result.is_err(), "Should error for short sequence");
    }

    /// Test that jump step formula is correctly computed.
    #[test]
    fn jump_step_formula() {
        // 45 for len > 250
        assert_eq!(compute_jump_step(251), 45); // 45

        // 35 for len > 200
        assert_eq!(compute_jump_step(201), 35); // 35

        // 25 for len > 150
        assert_eq!(compute_jump_step(151), 25); // 25

        // 15 for len <= 150
        assert_eq!(compute_jump_step(100), 15); // 15

        // Capping test: for len=45, base_jump=15, cap is 45/3=15, so result is 15.
        assert_eq!(compute_jump_step(45), 15);

        // For len=30, base_jump=15, cap is 30/3=10, so result is 10.
        assert_eq!(compute_jump_step(30), 10);
    }

    /// Test d0_min clamping in d01 calculation.
    #[test]
    fn d01_clamping() {
        // When d0 is very small (e.g., 0.0), d01 = 0.0 + 1.5 = 1.5, which is >= D0_MIN.
        // So no clamping occurs: d01 = 1.5, d02 = 2.25.
        // aL must be >= 9 so n_frag[0] = min(20, aL/3) >= 3 (otherwise both
        // fragment lengths round down below the minimum-3 threshold and the
        // seed correctly finds nothing to try, per `get_initial5`'s `frag_len
        // < 3` skip).
        let coords: Vec<Vector3<f32>> = (0..9).map(|i| Vector3::new(i as f32, 0.0, 0.0)).collect();

        // With a very loose d0_search, we should still get an alignment.
        let result = get_initial5(&coords, &coords, 0.5, 8.0, 9);
        assert!(result.is_ok());
    }

    /// Test that fragment length capping works correctly.
    #[test]
    fn fragment_length_capping() {
        // For a 30-residue structure:
        // aL = min(30, 30) = 30
        // n_frag[0] = min(20, 30/3) = min(20, 10) = 10
        // n_frag[1] = min(100, 30/2) = min(100, 15) = 15
        let coords: Vec<Vector3<f32>> = (0..30).map(|i| Vector3::new(i as f32, 0.0, 0.0)).collect();

        let result = get_initial5(&coords, &coords, 2.0, 8.0, 30);
        assert!(result.is_ok());
    }

    /// Test rotation-aware scoring with translated structures.
    #[test]
    fn rotation_aware_scoring_translated() {
        // Create two identical structures, but shift one along x-axis.
        let coords1 = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(2.0, 0.0, 0.0),
            Vector3::new(3.0, 0.0, 0.0),
            Vector3::new(4.0, 0.0, 0.0),
            Vector3::new(5.0, 0.0, 0.0),
        ];
        let coords2: Vec<_> = coords1.iter().map(|p| p + Vector3::new(10.0, 0.0, 0.0)).collect();

        let d0 = 2.0;
        let d0_search = 8.0;
        let l_norm = coords1.len();

        let result = get_initial5(&coords1, &coords2, d0, d0_search, l_norm);
        assert!(result.is_ok());

        // The algorithm should find the translated structure and achieve a high score.
        let alignment = result.unwrap();
        let matched_pairs: Vec<_> = alignment
            .iter()
            .filter_map(|(i_opt, j_opt)| match (i_opt, j_opt) {
                (Some(i), Some(j)) => Some((*i, *j)),
                _ => None,
            })
            .collect();

        // Should find matches across the translation.
        assert!(!matched_pairs.is_empty());
    }

    /// Test that the algorithm handles slightly misaligned structures.
    #[test]
    fn misaligned_structures() {
        // Create two offset linear structures. aL=9 so n_frag[0]=min(20,9/3=3)=3
        // survives the seed's `frag_len < 3` skip (see `d01_clamping`'s comment).
        let coords1: Vec<Vector3<f32>> = (0..9).map(|i| Vector3::new(i as f32, 0.0, 0.0)).collect();
        let coords2: Vec<Vector3<f32>> = (1..10).map(|i| Vector3::new(i as f32, 0.0, 0.0)).collect();

        let d0 = 2.0;
        let d0_search = 8.0;
        let l_norm = 9;

        let result = get_initial5(&coords1, &coords2, d0, d0_search, l_norm);
        assert!(result.is_ok());

        let alignment = result.unwrap();
        assert!(!alignment.is_empty());
    }

    /// Test that the algorithm fails gracefully when structures are too different.
    #[test]
    fn completely_different_structures() {
        let coords1 = vec![Vector3::new(0.0, 0.0, 0.0); 10];
        let coords2 = vec![Vector3::new(100.0, 100.0, 100.0); 10];

        let d0 = 1.0;
        let d0_search = 2.0;
        let l_norm = 10;

        // With a very tight d0 and vastly separated structures, we may not find a good alignment.
        // The function should either return a low-scoring alignment or an error.
        let result = get_initial5(&coords1, &coords2, d0, d0_search, l_norm);
        // This may either succeed with a low score or fail. Either is acceptable.
        // The important thing is that it doesn't panic.
        let _ = result;
    }
}
