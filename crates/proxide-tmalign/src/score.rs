//! TM-score evaluation and fast iterative refitting.
//!
//! [`get_score_fast`] is the cheap scoring probe used by every gapless-threading-style
//! seed (gapless threading, local-structure superposition, fragment-gapless threading)
//! to rank candidate alignments without a full `DP_iter` refinement loop. Given an
//! alignment map, it performs three iterations of Kabsch fitting + distance-filtered
//! re-fitting, returning the best TM-score found.
//!
//! [`evaluate_tm_score`] is a helper that computes TM-score given an alignment,
//! rotation/translation, and distance threshold — used both internally by
//! `get_score_fast` and later by the full `DP_iter` refinement pipeline.

use crate::d0;
use crate::kabsch::kabsch_superpose;
use crate::nw::AlignedPair;
use nalgebra::Vector3;

/// Evaluate TM-score for a given alignment, rotation, and translation.
///
/// Given two coordinate sets, an alignment map, a rotation/translation, and
/// a distance threshold `d0`, computes the TM-score as:
/// `TM = (1/l_norm) * Σ 1/(1+(d_i/d0)²)` over all aligned pairs.
///
/// # Arguments
///
/// - `coords1` — Cα coordinates of structure 1.
/// - `coords2` — Cα coordinates of structure 2.
/// - `alignment` — Alignment map `(i_in_1, j_in_2)` pairs; `None` denotes gaps.
/// - `rotation` — Row-major 3×3 rotation matrix.
/// - `translation` — Translation vector `[tx, ty, tz]`.
/// - `d0` — TM-score distance threshold.
/// - `l_norm` — Normalization length (typically `min(len1, len2)` or structure length).
///
/// Returns the normalized TM-score. Returns `0.0` if `l_norm == 0` or no aligned pairs exist.
pub fn evaluate_tm_score(
    coords1: &[Vector3<f32>],
    coords2: &[Vector3<f32>],
    alignment: &[AlignedPair],
    rotation: &[[f32; 3]; 3],
    translation: &[f32; 3],
    d0: f32,
    l_norm: usize,
) -> f32 {
    if l_norm == 0 {
        return 0.0;
    }

    let mut squared_distances = Vec::new();
    for (i_opt, j_opt) in alignment {
        if let (Some(i), Some(j)) = (i_opt, j_opt) {
            if *i >= coords1.len() || *j >= coords2.len() {
                continue;
            }
            // Apply rotation + translation to coords1[i]
            let rotated = apply_transform_with_matrix(coords1[*i], rotation, translation);
            let diff = rotated - coords2[*j];
            let sq_dist = diff.norm_squared();
            squared_distances.push(sq_dist);
        }
    }

    if squared_distances.is_empty() {
        return 0.0;
    }

    d0::tm_score(&squared_distances, d0, l_norm)
}

/// Apply rotation and translation: `R·p + t`.
///
/// Helper for rotating a single point using a row-major rotation matrix.
pub(crate) fn apply_transform_with_matrix(
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

/// Fast iterative TM-score refinement for initial seeding.
///
/// Performs up to 3 iterations of Kabsch fitting and TM-score evaluation:
///
/// 1. **Iteration 1**: Gather aligned pairs, fit Kabsch once over all pairs,
///    compute `tmscore` unnormalized.
/// 2. **Iteration 2**: Recompute distance cutoff as `max(d0_search², 3rd-smallest-squared-distance)`,
///    filter pairs ≤ cutoff (escalating by +0.5 if <3 survive), re-fit Kabsch on subset,
///    recompute `tmscore1` over *all* original pairs with the new rotation.
/// 3. **Iteration 3**: Same with cutoff `d0_search² + 1`.
///
/// Returns `max(tmscore, tmscore1, tmscore2)`, normalized by `l_norm`.
///
/// # Arguments
///
/// - `coords1` — Cα coordinates of structure 1.
/// - `coords2` — Cα coordinates of structure 2.
/// - `alignment` — Discrete alignment map from `nw::needleman_wunsch_affine` or seeding strategy.
/// - `d0` — Final TM-score distance threshold (from `d0::d0_final`).
/// - `d0_search` — Search-phase threshold (from `d0::d0_search`).
/// - `l_norm` — Normalization length for final TM-score.
///
/// Returns the best TM-score found across the three iterations, normalized by `l_norm`.
/// Returns `0.0` if alignment is empty or coordinate sets have mismatched/invalid indices.
pub fn get_score_fast(
    coords1: &[Vector3<f32>],
    coords2: &[Vector3<f32>],
    alignment: &[AlignedPair],
    d0: f32,
    d0_search: f32,
    l_norm: usize,
) -> f32 {
    if l_norm == 0 {
        return 0.0;
    }

    // Collect valid aligned pairs (both residues present in both structures).
    let mut r1 = Vec::new();
    let mut r2 = Vec::new();
    for (i_opt, j_opt) in alignment {
        if let (Some(i), Some(j)) = (i_opt, j_opt) {
            if *i < coords1.len() && *j < coords2.len() {
                r1.push(coords1[*i]);
                r2.push(coords2[*j]);
            }
        }
    }

    if r1.is_empty() {
        return 0.0;
    }

    // Iteration 1: Initial Kabsch fit over all aligned pairs.
    let result1 = kabsch_superpose(&r1, &r2);
    let tmscore = evaluate_tm_score(
        coords1,
        coords2,
        alignment,
        &result1.rotation,
        &result1.translation,
        d0,
        l_norm,
    );

    // Iteration 2: Distance-filtered re-fit.
    let mut squared_distances = Vec::new();
    for (&pt1, &pt2) in r1.iter().zip(r2.iter()) {
        let rotated = apply_transform_with_matrix(pt1, &result1.rotation, &result1.translation);
        let sq_dist = (rotated - pt2).norm_squared();
        squared_distances.push(sq_dist);
    }

    let d0_search_sq = d0_search * d0_search;
    let mut d002t = d0_search_sq;

    // Find the 3rd smallest squared distance.
    if squared_distances.len() >= 3 {
        let mut sorted_dists = squared_distances.clone();
        sorted_dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        d002t = d002t.max(sorted_dists[2]);
    }

    // Filter pairs and re-fit, escalating the cutoff by 0.5 repeatedly while
    // fewer than 3 pairs survive (TMalign.h's `while(1){...if(j<3&&n_ali>3)
    // d002t+=0.5; else break;}` — a real loop, not a single-shot retry).
    let mut r1_filtered = Vec::new();
    let mut r2_filtered = Vec::new();
    loop {
        r1_filtered.clear();
        r2_filtered.clear();
        for (i, (&pt1, &pt2)) in r1.iter().zip(r2.iter()).enumerate() {
            if squared_distances[i] <= d002t {
                r1_filtered.push(pt1);
                r2_filtered.push(pt2);
            }
        }
        if r1_filtered.len() < 3 && r1.len() > 3 {
            d002t += 0.5;
        } else {
            break;
        }
    }

    let result2 = if r1_filtered.is_empty() {
        result1.clone()
    } else {
        kabsch_superpose(&r1_filtered, &r2_filtered)
    };
    let tmscore1 = evaluate_tm_score(
        coords1,
        coords2,
        alignment,
        &result2.rotation,
        &result2.translation,
        d0,
        l_norm,
    );

    // Iteration 3: Re-fit with tighter cutoff, same repeated-escalation loop.
    let mut d002t_iter3_esc = d0_search_sq + 1.0;
    let mut r1_filtered3 = Vec::new();
    let mut r2_filtered3 = Vec::new();
    loop {
        r1_filtered3.clear();
        r2_filtered3.clear();
        for (i, (&pt1, &pt2)) in r1.iter().zip(r2.iter()).enumerate() {
            if squared_distances[i] <= d002t_iter3_esc {
                r1_filtered3.push(pt1);
                r2_filtered3.push(pt2);
            }
        }
        if r1_filtered3.len() < 3 && r1.len() > 3 {
            d002t_iter3_esc += 0.5;
        } else {
            break;
        }
    }

    let result3 = if r1_filtered3.is_empty() {
        result2.clone()
    } else {
        kabsch_superpose(&r1_filtered3, &r2_filtered3)
    };
    let tmscore2 = evaluate_tm_score(
        coords1,
        coords2,
        alignment,
        &result3.rotation,
        &result3.translation,
        d0,
        l_norm,
    );

    tmscore.max(tmscore1).max(tmscore2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn evaluate_tm_score_perfect_overlap_is_one() {
        let coords: Vec<Vector3<f32>> = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
        ];
        let alignment = vec![(Some(0), Some(0)), (Some(1), Some(1)), (Some(2), Some(2))];
        let identity_rotation = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let zero_translation = [0.0; 3];

        let d0 = 5.85; // From d0_final(250)
        let tm = evaluate_tm_score(
            &coords,
            &coords,
            &alignment,
            &identity_rotation,
            &zero_translation,
            d0,
            3,
        );
        assert_relative_eq!(tm, 1.0, epsilon = 1e-4);
    }

    #[test]
    fn evaluate_tm_score_with_gaps_is_zero_for_gapped_residues() {
        let coords: Vec<Vector3<f32>> =
            vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0)];
        // Only first residue is aligned.
        let alignment = vec![(Some(0), Some(0)), (None, Some(1))];
        let identity_rotation = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let zero_translation = [0.0; 3];

        let d0 = 5.85;
        let tm = evaluate_tm_score(
            &coords,
            &coords,
            &alignment,
            &identity_rotation,
            &zero_translation,
            d0,
            2,
        );
        // Only 1 pair is aligned, so TM = 1.0 / 2 = 0.5
        assert_relative_eq!(tm, 0.5, epsilon = 1e-4);
    }

    #[test]
    fn get_score_fast_returns_positive_score_for_identical_alignment() {
        let coords: Vec<Vector3<f32>> = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
        ];
        let alignment = vec![
            (Some(0), Some(0)),
            (Some(1), Some(1)),
            (Some(2), Some(2)),
            (Some(3), Some(3)),
        ];
        let d0 = 5.85;
        let d0_search = 8.0;
        let score = get_score_fast(&coords, &coords, &alignment, d0, d0_search, 4);
        assert_relative_eq!(score, 1.0, epsilon = 1e-4);
    }

    #[test]
    fn get_score_fast_empty_alignment_returns_zero() {
        let coords: Vec<Vector3<f32>> = vec![Vector3::new(0.0, 0.0, 0.0)];
        let alignment = vec![];
        let score = get_score_fast(&coords, &coords, &alignment, 5.85, 8.0, 1);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn get_score_fast_with_gaps_returns_positive_score() {
        let coords: Vec<Vector3<f32>> = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
        ];
        let alignment = vec![(Some(0), Some(0)), (None, Some(1)), (Some(2), Some(2))];
        let d0 = 5.85;
        let d0_search = 8.0;
        let score = get_score_fast(&coords, &coords, &alignment, d0, d0_search, 3);
        // At least some of the aligned pairs contribute to the score.
        assert!(score > 0.0);
    }
}
