//! Iterative DP refinement loop (`DP_iter`) for TM-align.
//!
//! After an initial seeding strategy produces a candidate alignment, `DP_iter`
//! refines the rotation/translation by repeatedly:
//! 1. Running NWDP_TM with the current rotation (scoring on rotated-frame proximity).
//! 2. Extracting the aligned pair coordinates and refining via `tmscore8_search`,
//!    which tries multiple fragment sizes and positions to find the best local superposition.
//! 3. Checking convergence by comparing TM-scores across iterations.
//!
//! This loop runs for two gap-open penalty values (from a range like `[-0.6, 0.0]`)
//! and up to a specified maximum iteration count.

use crate::kabsch::{apply_transform, kabsch_superpose, KabschResult};
use crate::nwdp_tm::nwdp_tm;
use crate::seed::AlignmentMap;
use nalgebra::Vector3;

/// Apply rotation and translation: `R·p + t`.
///
/// Helper for rotating a single point using a row-major rotation matrix and translation vector.
fn apply_transform_inline(
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

/// Helper to compute TM-score for a subset of aligned pairs at a given distance cutoff.
///
/// Mimics USalign's `score_fun8`: collects pairs within distance `d`,
/// with a fallback loop that increases `d` by 0.5 if fewer than 3 pairs survive
/// (and the total count > 3). Returns the collected indices and the normalized score.
fn score_fun8(
    coords1_rotated: &[Vector3<f32>],
    coords2: &[Vector3<f32>],
    d0: f32,
    d_cutoff: f32,
    l_norm: usize,
) -> (Vec<usize>, f32) {
    if l_norm == 0 {
        return (Vec::new(), 0.0);
    }

    let d0_sq = d0 * d0;
    let mut d = d_cutoff;

    loop {
        let d_sq = d * d;
        let mut aligned_indices = Vec::new();
        let mut score_sum = 0.0_f32;

        for i in 0..coords1_rotated.len().min(coords2.len()) {
            let diff = coords1_rotated[i] - coords2[i];
            let dist_sq = diff.norm_squared();

            if dist_sq < d_sq {
                aligned_indices.push(i);
            }
            // Always accumulate score for all pairs (score_sum_method=8 in the C code)
            score_sum += 1.0 / (1.0 + dist_sq / d0_sq);
        }

        // If fewer than 3 pairs and we have more than 3 total, increase cutoff.
        if aligned_indices.len() < 3 && coords1_rotated.len() > 3 {
            d += 0.5;
        } else {
            return (aligned_indices, score_sum / (l_norm as f32));
        }
    }
}

/// Refine a rotation+translation via multi-scale fragment superposition.
///
/// Given an alignment of extracted coordinates, iteratively fits fragments
/// of different lengths (full length, half, quarter, etc., down to 4) to find
/// the rotation and translation that maximize TM-score. For each fragment length,
/// slides the fragment across the alignment with a fixed step, extracting and
/// Kabsch-fitting each fragment, then refining iteratively at a tighter distance
/// cutoff.
///
/// Returns `(best_rotation, best_translation, best_tm_score)`.
fn tmscore8_search(
    coords1_aligned: &[Vector3<f32>],
    coords2_aligned: &[Vector3<f32>],
    d0: f32,
    d0_search: f32,
    l_norm: usize,
) -> (KabschResult, f32) {
    if coords1_aligned.is_empty() {
        let dummy = KabschResult {
            rmsd: f32::INFINITY,
            rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            translation: [0.0; 3],
        };
        return (dummy, 0.0);
    }

    let lali = coords1_aligned.len();
    let mut best_result = KabschResult {
        rmsd: f32::INFINITY,
        rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        translation: [0.0; 3],
    };
    let mut best_score = -1.0_f32;

    // Build fragment lengths: lali, lali/2, lali/4, ..., down to L_ini_min (=4 or smaller).
    let l_ini_min = 4.min(lali);
    let mut l_ini = Vec::new();
    let mut n = lali;
    while n > l_ini_min {
        l_ini.push(n);
        n /= 2;
    }
    l_ini.push(l_ini_min);

    const SIMPLIFY_STEP: usize = 40;
    const N_IT_MAX: usize = 20;

    for &l_frag in &l_ini {
        let il_max = lali.saturating_sub(l_frag);

        let mut i = 0;
        loop {
            // Extract the fragment at position i
            if i + l_frag > lali {
                break;
            }
            let fragment1: Vec<_> = coords1_aligned[i..i + l_frag].to_vec();
            let fragment2: Vec<_> = coords2_aligned[i..i + l_frag].to_vec();

            // Kabsch fit on the fragment to get rotation+translation.
            let kabsch_result = kabsch_superpose(&fragment1, &fragment2);

            // Rotate all coords1 using this rotation and score at d = d0_search - 1.
            let rotated: Vec<_> = coords1_aligned
                .iter()
                .map(|&p| apply_transform(&kabsch_result, p))
                .collect();

            let (_aligned_indices, score) = score_fun8(&rotated, coords2_aligned, d0, d0_search - 1.0, l_norm);
            if score > best_score {
                best_score = score;
                best_result = kabsch_result.clone();
            }

            // Iteratively refine at d = d0_search + 1 for up to N_IT_MAX iterations.
            let mut current_indices = _aligned_indices;

            for _ in 0..N_IT_MAX {
                // Re-fit Kabsch on the currently aligned indices.
                let mut pts1_subset = Vec::new();
                let mut pts2_subset = Vec::new();
                for &idx in &current_indices {
                    if idx < coords1_aligned.len() && idx < coords2_aligned.len() {
                        pts1_subset.push(coords1_aligned[idx]);
                        pts2_subset.push(coords2_aligned[idx]);
                    }
                }

                if pts1_subset.is_empty() {
                    break;
                }

                let new_result = kabsch_superpose(&pts1_subset, &pts2_subset);

                // Score with the new rotation at d = d0_search + 1.
                let rotated_new: Vec<_> = coords1_aligned
                    .iter()
                    .map(|&p| apply_transform(&new_result, p))
                    .collect();
                let (new_indices, score_new) = score_fun8(&rotated_new, coords2_aligned, d0, d0_search + 1.0, l_norm);

                if score_new > best_score {
                    best_score = score_new;
                    best_result = new_result.clone();
                }

                // Convergence check: if the index set hasn't changed, we've converged.
                if new_indices.len() == current_indices.len() {
                    let mut converged = true;
                    for (&old_idx, &new_idx) in current_indices.iter().zip(new_indices.iter()) {
                        if old_idx != new_idx {
                            converged = false;
                            break;
                        }
                    }
                    if converged {
                        break;
                    }
                }

                current_indices = new_indices;
            }

            // Move to next fragment position.
            if i < il_max {
                i += SIMPLIFY_STEP;
                if i > il_max {
                    i = il_max;
                }
            } else {
                break;
            }
        }
    }

    (best_result, best_score)
}

/// Iterative DP refinement loop for TM-align.
///
/// Given an initial alignment from a seeding strategy, refines the rotation/translation
/// by alternating between NWDP_TM (proximity-based sequence alignment using the current
/// superposition) and `tmscore8_search` (rotation refinement via multi-scale fragments).
/// Runs for two gap-open penalties and up to a specified maximum iteration count.
///
/// # Arguments
///
/// - `coords1` — Cα coordinates of structure 1 (full length).
/// - `coords2` — Cα coordinates of structure 2 (full length).
/// - `initial_alignment` — Starting alignment from a seeding strategy.
/// - `d0` — Final TM-score distance threshold (from `d0::d0_final`).
/// - `d0_search` — Search-phase threshold (from `d0::d0_search`).
/// - `l_norm` — Normalization length (typically `min(len1, len2)`).
/// - `gap_open_range` — `(start_idx, end_idx)` for indexing into the gap_open array `[-0.6, 0.0]`.
///   Typical values: `(0, 1)` for `-0.6`, `(0, 2)` for both `[-0.6, 0.0]`, `(1, 2)` for `0.0` only.
/// - `max_iterations` — Maximum iterations per gap_open value (typically 30 for normal runs,
///   2 for local-structure seeding).
///
/// # Returns
///
/// `(best_alignment, best_tm_score)` — the best alignment found across all iterations
/// and gap-open values, and its TM-score.
///
/// # Algorithm
///
/// For each gap_open value in range:
///   For each iteration up to max_iterations:
///     1. Run NWDP_TM with current rotation (on-the-fly scoring via squared distance).
///     2. Extract aligned pair coordinates.
///     3. Call `tmscore8_search` to refine rotation+translation.
///     4. Check convergence: if `|tmscore - tmscore_old| < 1e-6`, break early.
///   Track best `(alignment, tmscore)` across all gap_open values.
#[allow(clippy::too_many_arguments)]
pub fn dp_iter(
    coords1: &[Vector3<f32>],
    coords2: &[Vector3<f32>],
    initial_alignment: &AlignmentMap,
    d0: f32,
    d0_search: f32,
    l_norm: usize,
    gap_open_range: (usize, usize),
    max_iterations: usize,
) -> (AlignmentMap, f32) {
    if coords1.is_empty() || coords2.is_empty() || l_norm == 0 {
        return (initial_alignment.clone(), 0.0);
    }

    let mut best_alignment = initial_alignment.clone();
    let mut best_score = 0.0_f32;

    let gap_open_values = [-0.6_f32, 0.0_f32];
    let d0_sq = d0 * d0;

    // Pairs from the seed alignment, used to derive the starting rotation for
    // each gap_open pass (before any refinement has happened).
    let initial_pairs: Vec<(Vector3<f32>, Vector3<f32>)> = initial_alignment
        .iter()
        .filter_map(|&(i_opt, j_opt)| match (i_opt, j_opt) {
            (Some(i), Some(j)) if i < coords1.len() && j < coords2.len() => {
                Some((coords1[i], coords2[j]))
            }
            _ => None,
        })
        .collect();
    let identity_transform = ([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [0.0_f32; 3]);

    let (gap_start, gap_end) = gap_open_range;
    for gap_idx in gap_start..gap_end {
        if gap_idx >= gap_open_values.len() {
            break;
        }
        let gap_open = gap_open_values[gap_idx];
        let mut tmscore_old = 0.0_f32;

        // Rotation/translation state carried forward across iterations within
        // this gap_open pass — updated from `tmscore8_search`'s refined result
        // at the end of every iteration (regardless of whether it improved the
        // running best), matching DP_iter's "t,u carry over to the next
        // NWDP_TM call" behavior. Re-deriving this from `best_alignment` each
        // iteration (the prior bug here) would silently repeat the same
        // NWDP_TM computation on any non-improving iteration instead of
        // continuing to explore from the latest refinement.
        let (mut current_rotation, mut current_translation) = if initial_pairs.is_empty() {
            identity_transform
        } else {
            let (p1, p2): (Vec<_>, Vec<_>) = initial_pairs.iter().cloned().unzip();
            let kabsch_result = kabsch_superpose(&p1, &p2);
            (kabsch_result.rotation, kabsch_result.translation)
        };

        for iteration in 0..max_iterations {
            // Run NWDP_TM with rotation-aware scoring against the carried-forward rotation.
            let alignment = nwdp_tm(coords1.len(), coords2.len(), gap_open, |i, j| {
                let p1_rot = apply_transform_inline(coords1[i], &current_rotation, &current_translation);
                let diff = p1_rot - coords2[j];
                let dist_sq = diff.norm_squared();
                // Score as: 1/(1 + dist²/d0²)
                // We want higher scores for closer points.
                1.0 / (1.0 + dist_sq / d0_sq)
            });

            // Extract aligned coordinates from the new alignment.
            let mut new_aligned_coords1 = Vec::new();
            let mut new_aligned_coords2 = Vec::new();
            for (i_opt, j_opt) in &alignment {
                if let (Some(i), Some(j)) = (i_opt, j_opt) {
                    if *i < coords1.len() && *j < coords2.len() {
                        new_aligned_coords1.push(coords1[*i]);
                        new_aligned_coords2.push(coords2[*j]);
                    }
                }
            }

            if new_aligned_coords1.is_empty() {
                break;
            }

            // Refine rotation via tmscore8_search.
            let (refined_result, refined_score) =
                tmscore8_search(&new_aligned_coords1, &new_aligned_coords2, d0, d0_search, l_norm);

            // Carry the refined rotation/translation into the next iteration's
            // NWDP_TM call — unconditionally, not only on improvement.
            current_rotation = refined_result.rotation;
            current_translation = refined_result.translation;

            // Update the running best if improved.
            if refined_score > best_score {
                best_score = refined_score;
                best_alignment = alignment.clone();
            }

            // Convergence check: break if TM-score change is < 1e-6.
            if iteration > 0 && (tmscore_old - refined_score).abs() < 1e-6 {
                break;
            }
            tmscore_old = refined_score;
        }
    }

    (best_alignment, best_score)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tmscore8_search_on_identical_coordinates_returns_high_score() {
        let coords: Vec<Vector3<f32>> = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.5, 1.0, 0.0),
            Vector3::new(0.5, 0.5, 1.0),
        ];
        let d0 = 5.85;
        let d0_search = 8.0;
        let l_norm = 4;

        let (_rotation, score) = tmscore8_search(&coords, &coords, d0, d0_search, l_norm);
        // For identical coordinates, score should be close to 1.0.
        assert!(score > 0.8, "Score for identical coords should be high, got {}", score);
    }

    #[test]
    fn tmscore8_search_on_translated_coordinates_recovers_well() {
        let coords1: Vec<Vector3<f32>> = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.5, 1.0, 0.0),
            Vector3::new(0.5, 0.5, 1.0),
        ];
        let offset = Vector3::new(5.0, -3.0, 2.0);
        let coords2: Vec<_> = coords1.iter().map(|&p| p + offset).collect();

        let d0 = 5.85;
        let d0_search = 8.0;
        let l_norm = 4;

        let (_rotation, score) = tmscore8_search(&coords1, &coords2, d0, d0_search, l_norm);
        // After recovery of translation, should get high score.
        assert!(score > 0.8, "Score for translated coords should recover, got {}", score);
    }

    #[test]
    fn dp_iter_improves_on_identity_initialization() {
        let coords: Vec<Vector3<f32>> = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.5, 1.0, 0.0),
            Vector3::new(0.5, 0.5, 1.0),
        ];

        // Create a perfect identity alignment.
        let initial_alignment: AlignmentMap = vec![
            (Some(0), Some(0)),
            (Some(1), Some(1)),
            (Some(2), Some(2)),
            (Some(3), Some(3)),
        ];

        let d0 = 5.85;
        let d0_search = 8.0;
        let l_norm = 4;

        let (_refined_alignment, score) =
            dp_iter(&coords, &coords, &initial_alignment, d0, d0_search, l_norm, (0, 2), 5);

        // Score should be very high for identical structures.
        assert!(score > 0.8, "DP_iter score for identical coords should be high, got {}", score);
    }

    #[test]
    fn score_fun8_collects_pairs_within_distance() {
        let coords1: Vec<Vector3<f32>> = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(100.0, 100.0, 100.0),
        ];
        // Pairs 0 and 1 match their coords1 counterpart exactly (distance 0);
        // pair 2 is deliberately displaced far from coords1[2] so it's NOT a
        // close pair (a coords1.clone() here would make every pair distance-0,
        // including the "far" one, defeating the point of this test).
        let coords2 = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(200.0, 200.0, 200.0),
        ];

        let d0 = 5.85;
        let d_cutoff = 2.0;
        let l_norm = 3;

        let (indices, _score) = score_fun8(&coords1, &coords2, d0, d_cutoff, l_norm);

        // Should collect the two close pairs, not the far one.
        assert_eq!(indices.len(), 2, "Expected 2 close pairs, got {}", indices.len());
    }

    #[test]
    fn score_fun8_returns_zero_for_empty_structures() {
        let coords1: Vec<Vector3<f32>> = vec![];
        let coords2: Vec<Vector3<f32>> = vec![];

        let d0 = 5.85;
        let d_cutoff = 2.0;
        let l_norm = 0;

        let (_indices, score) = score_fun8(&coords1, &coords2, d0, d_cutoff, l_norm);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn tmscore8_search_empty_alignment_returns_dummy_result() {
        let coords1: Vec<Vector3<f32>> = vec![];
        let coords2: Vec<Vector3<f32>> = vec![];

        let d0 = 5.85;
        let d0_search = 8.0;
        let l_norm = 0;

        let (result, score) = tmscore8_search(&coords1, &coords2, d0, d0_search, l_norm);
        assert!(result.rmsd.is_infinite());
        assert_eq!(score, 0.0);
    }
}
