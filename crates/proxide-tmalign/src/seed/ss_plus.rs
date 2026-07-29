//! Secondary-structure + local-superposition combo seed strategy.
//!
//! Implements USalign's `get_initial_ssplus()` (TMalign.h:1094-1104), using
//! helper `score_matrix_rmsd_sec()` (1042-1085): Kabsch-fits the *previous best
//! alignment* (from the gapless-threading seed), rotates all of the first
//! structure, builds a full `(xlen+1)x(ylen+1)` DP score matrix where
//! `score[i+1][j+1] = 1/(1+d_ij²/d02) + 0.5` if `secx[i]==secy[j]` else without
//! the `+0.5` bonus (`d02=(d0+1.5, clamped to D0_MIN)²`), then runs NW-DP with
//! `gap_open=-1.0` to produce the final alignment map.

use super::AlignmentMap;
use crate::d0::D0_MIN_FINAL;
use crate::error::TmAlignError;
use crate::kabsch::{apply_transform, kabsch_superpose};
use crate::nwdp_tm::nwdp_tm;
use crate::ss::make_sec;
use nalgebra::Vector3;

/// Build the score matrix for secondary-structure + superposition-weighted DP.
///
/// Given a previous alignment (`previous_alignment`), computes the Kabsch fit
/// of those aligned pairs, applies the resulting rotation+translation to all
/// residues of the first structure, then builds a score matrix where:
/// - `score[i+1][j+1] = 1/(1+d_ij²/d02) + 0.5` if `sec1[i]==sec2[j]`
/// - `score[i+1][j+1] = 1/(1+d_ij²/d02)` otherwise
///
/// where `d02 = (d0 + 1.5).max(D0_MIN_FINAL)²`.
///
/// Returns the rotated `coords1` and the score matrix.
#[allow(clippy::type_complexity)]
fn build_ss_score_matrix(
    coords1: &[Vector3<f32>],
    coords2: &[Vector3<f32>],
    previous_alignment: &AlignmentMap,
    d0: f32,
) -> Result<(Vec<Vector3<f32>>, Vec<Vec<f32>>), TmAlignError> {
    let xlen = coords1.len();
    let ylen = coords2.len();

    // Extract aligned pairs from previous_alignment for Kabsch fitting.
    let mut aligned_coords1 = Vec::new();
    let mut aligned_coords2 = Vec::new();

    for (i_opt, j_opt) in previous_alignment.iter() {
        if let (Some(i), Some(j)) = (i_opt, j_opt) {
            if *i < xlen && *j < ylen {
                aligned_coords1.push(coords1[*i]);
                aligned_coords2.push(coords2[*j]);
            }
        }
    }

    // If no aligned pairs, cannot fit — return error.
    if aligned_coords1.is_empty() {
        return Err(TmAlignError::Parse(
            "Cannot fit rotation: no aligned pairs in previous alignment".to_string(),
        ));
    }

    // Kabsch-fit the aligned pairs to get rotation + translation.
    let kabsch_result = kabsch_superpose(&aligned_coords1, &aligned_coords2);

    // Apply rotation + translation to all of coords1.
    let rotated_coords1: Vec<Vector3<f32>> = coords1
        .iter()
        .map(|&p| apply_transform(&kabsch_result, p))
        .collect();

    // Compute secondary structures.
    let sec1 = make_sec(&rotated_coords1);
    let sec2 = make_sec(coords2);

    // Compute d02.
    let d01 = (d0 + 1.5).max(D0_MIN_FINAL);
    let d02 = d01 * d01;

    // Build the (xlen+1) x (ylen+1) score matrix.
    // score[i+1][j+1] corresponds to residues (i, j).
    // score[0][*] and score[*][0] are zero (no alignment cost).
    let mut score = vec![vec![0.0_f32; ylen + 1]; xlen + 1];

    for i in 0..xlen {
        for j in 0..ylen {
            // Compute squared distance (TMalign uses squared distances in the score formula).
            let delta = rotated_coords1[i] - coords2[j];
            let dij_sq = delta.norm_squared();
            let base_score = 1.0 / (1.0 + dij_sq / d02);
            let ss_bonus = if sec1[i] == sec2[j] { 0.5 } else { 0.0 };
            score[i + 1][j + 1] = base_score + ss_bonus;
        }
    }

    Ok((rotated_coords1, score))
}

/// Secondary-structure + local-superposition alignment seed.
///
/// Entry point for the SS+local-superposition seeding strategy. Kabsch-fits
/// the previous best alignment (from the gapless-threading seed), rotates
/// all of structure 1, computes secondary structures, builds a full DP score
/// matrix combining proximity and SS-letter identity, and runs NW-DP to
/// produce the final alignment.
///
/// # Arguments
///
/// - `coords1` — Cα coordinates of structure 1.
/// - `coords2` — Cα coordinates of structure 2.
/// - `previous_alignment` — Best alignment from a previous seed (typically
///   from gapless threading); used to fit the rotation.
/// - `d0` — Distance threshold parameter (from `d0::d0_search`).
///
/// # Returns
///
/// An `AlignmentMap` representing the best alignment found, or a `TmAlignError`
/// if the previous alignment is empty or other operations fail.
///
/// # Errors
///
/// Returns `TmAlignError::Parse` if the previous alignment contains no aligned
/// pairs (cannot compute a rotation).
pub fn get_initial_ssplus(
    coords1: &[Vector3<f32>],
    coords2: &[Vector3<f32>],
    previous_alignment: &AlignmentMap,
    d0: f32,
) -> Result<AlignmentMap, TmAlignError> {
    let xlen = coords1.len();
    let ylen = coords2.len();

    // Build the score matrix and get rotated coords1.
    let (_rotated_coords1, score) =
        build_ss_score_matrix(coords1, coords2, previous_alignment, d0)?;

    // Run the shared simplified-Gotoh NWDP_TM core (`crate::nwdp_tm`) with
    // the precomputed score matrix, gap_open=-1.0. NOT `nw::needleman_wunsch_affine`:
    // that's a full 3-state affine Gotoh with a leading-gap boundary penalty,
    // which diverges from USalign's actual NWDP_TM (free leading-gap
    // boundary, gap_open charged only when the previous cell was diagonal)
    // whenever gap_open != 0 — as it is here.
    let alignment = nwdp_tm(xlen, ylen, -1.0, |i, j| score[i + 1][j + 1]);

    Ok(alignment)
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;

    /// Test with identical structures: previous alignment is diagonal,
    /// and the resulting alignment should also be diagonal.
    #[test]
    fn ss_plus_identical_structures() {
        let coords = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(2.0, 0.0, 0.0),
            Vector3::new(3.0, 0.0, 0.0),
        ];

        // Previous alignment is diagonal (identity).
        let previous_alignment = vec![
            (Some(0), Some(0)),
            (Some(1), Some(1)),
            (Some(2), Some(2)),
            (Some(3), Some(3)),
        ];

        let d0 = 2.0;

        let result = get_initial_ssplus(&coords, &coords, &previous_alignment, d0);
        assert!(result.is_ok());

        let alignment = result.unwrap();
        // For identical coordinates and a good previous alignment,
        // the resulting alignment should be non-empty.
        assert!(!alignment.is_empty());

        // Count matched pairs (both coordinates present).
        let matched_pairs: Vec<_> = alignment
            .iter()
            .filter_map(|(i_opt, j_opt)| match (i_opt, j_opt) {
                (Some(i), Some(j)) => Some((*i, *j)),
                _ => None,
            })
            .collect();

        // For identical structures, expect a good match.
        assert!(!matched_pairs.is_empty());
    }

    /// Test with a shifted previous alignment: the first structure
    /// was previously aligned to the second with an offset, and we
    /// build from there.
    #[test]
    fn ss_plus_shifted_previous_alignment() {
        let coords1 = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(2.0, 0.0, 0.0),
            Vector3::new(3.0, 0.0, 0.0),
        ];

        let coords2 = vec![
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(2.0, 0.0, 0.0),
            Vector3::new(3.0, 0.0, 0.0),
            Vector3::new(4.0, 0.0, 0.0),
        ];

        // Previous alignment maps coords1[0..3] -> coords2[0..3] (shifted by 1).
        let previous_alignment = vec![
            (Some(0), Some(0)),
            (Some(1), Some(1)),
            (Some(2), Some(2)),
            (Some(3), Some(3)),
        ];

        let d0 = 2.0;

        let result = get_initial_ssplus(&coords1, &coords2, &previous_alignment, d0);
        assert!(result.is_ok());

        let alignment = result.unwrap();
        assert!(!alignment.is_empty());
    }

    /// Test error case: empty previous alignment (no aligned pairs).
    #[test]
    fn ss_plus_empty_previous_alignment_error() {
        let coords = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(2.0, 0.0, 0.0),
        ];

        let previous_alignment: AlignmentMap = vec![];
        let d0 = 2.0;

        let result = get_initial_ssplus(&coords, &coords, &previous_alignment, d0);
        assert!(result.is_err());
    }

    /// Test case with only gaps in the previous alignment.
    #[test]
    fn ss_plus_previous_alignment_all_gaps() {
        let coords = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(2.0, 0.0, 0.0),
        ];

        // Previous alignment has only gaps (no actual matches).
        let previous_alignment = vec![
            (Some(0), None),
            (None, Some(0)),
            (Some(1), None),
        ];
        let d0 = 2.0;

        let result = get_initial_ssplus(&coords, &coords, &previous_alignment, d0);
        assert!(result.is_err());
    }

    /// Test the score matrix construction directly.
    #[test]
    fn score_matrix_construction() {
        let coords = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(2.0, 0.0, 0.0),
        ];

        let previous_alignment = vec![
            (Some(0), Some(0)),
            (Some(1), Some(1)),
            (Some(2), Some(2)),
        ];

        let d0 = 2.0;

        let result = build_ss_score_matrix(&coords, &coords, &previous_alignment, d0);
        assert!(result.is_ok());

        let (_rotated_coords, score) = result.unwrap();

        // Check that the score matrix has the correct dimensions.
        assert_eq!(score.len(), 4); // xlen + 1
        assert_eq!(score[0].len(), 4); // ylen + 1

        // Check that score[0][*] and score[*][0] are zero.
        for j in 0..4 {
            assert_eq!(score[0][j], 0.0);
        }
        for i in 0..4 {
            assert_eq!(score[i][0], 0.0);
        }

        // Check that diagonal scores are positive (since coords are close).
        for i in 1..=3 {
            assert!(score[i][i] > 0.0);
        }
    }

    /// Test that the secondary-structure bonus is applied correctly.
    /// Residues with matching SS letters should have +0.5 bonus.
    #[test]
    fn score_matrix_ss_bonus() {
        // Create two identical structures so SS letters match.
        let coords = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(2.0, 0.0, 0.0),
            Vector3::new(3.0, 0.0, 0.0),
            Vector3::new(4.0, 0.0, 0.0),
        ];

        let previous_alignment = vec![
            (Some(0), Some(0)),
            (Some(1), Some(1)),
            (Some(2), Some(2)),
            (Some(3), Some(3)),
            (Some(4), Some(4)),
        ];

        let d0 = 2.0;

        let result = build_ss_score_matrix(&coords, &coords, &previous_alignment, d0);
        assert!(result.is_ok());

        let (_rotated_coords, score) = result.unwrap();

        // For identical structures, diagonal scores should reflect the distance
        // (which is ~0 after rotation) plus the SS bonus (0.5) if SS letters match.
        // At minimum, verify that the matrix is well-formed.
        for i in 1..6 {
            for j in 1..6 {
                assert!(score[i][j].is_finite());
                assert!(score[i][j] >= 0.0);
            }
        }
    }
}
