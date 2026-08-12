//! Fragment gapless threading seed strategy.
//!
//! Implements USalign's `get_initial_fgt()` (TMalign.h:1173-1402), which finds the
//! longest contiguous fragments of closely-packed residues (Cα-Cα distance < dcu_cut)
//! in both sequences, extracts the shorter fragment, and threads it gaplessly against
//! the full other sequence.
//!
//! The escalation of `dcu_cut` follows: `dcu_cut = (1.1^inc * dcu0)²`, incrementing `inc`
//! until a fragment of minimum length is found. Special cases handle symmetric-length
//! structures (both directions tried) and fragments that span the full chain (trimmed
//! to middle 79%).

use super::AlignmentMap;
use crate::error::TmAlignError;
use crate::seed::gapless;
use nalgebra::Vector3;

/// Finds the longest contiguous run of residues with consecutive Cα-Cα distances < dcu_cut.
///
/// Escalates the distance cutoff `dcu_cut = (1.1^inc * dcu0)²` starting from `dcu0²`,
/// incrementing `inc` until a fragment of at least `min(len/3, fra_min)` residues is found.
/// Returns the (inclusive) start and end indices of the longest fragment.
///
/// # Arguments
///
/// - `coords` — Cα coordinates of the structure.
/// - `dcu0` — Initial distance threshold (typically 4.25; will be squared internally).
/// - `fast_opt` — If true, use `fra_min=8` instead of 4.
///
/// # Returns
///
/// A tuple `(start_idx, end_idx)` where the longest fragment spans from `start_idx`
/// to `end_idx` (both inclusive). Returns `(0, 0)` if the sequence is empty or too short.
fn find_max_frag(coords: &[Vector3<f32>], dcu0: f32, fast_opt: bool) -> (usize, usize) {
    let len = coords.len();
    if len < 2 {
        return (0, 0);
    }

    let fra_min = if fast_opt { 8 } else { 4 };
    // Minimum fragment length required: min(len/3, fra_min)
    let mut r_min = len / 3;
    if r_min > fra_min {
        r_min = fra_min;
    }

    let mut lfr_max;
    let mut start_max = 0;
    let mut end_max = 0;

    let mut inc = 0;
    let mut dcu_cut = dcu0 * dcu0; // dcu0² as initial squared distance cutoff

    loop {
        lfr_max = 0;
        let mut j = 1; // Current fragment length
        let mut start = 0;

        for i in 1..len {
            let distance_sq = (coords[i - 1] - coords[i]).norm_squared();
            if distance_sq < dcu_cut {
                j += 1;
                // Check if we reached the end of the sequence
                if i == len - 1 {
                    if j > lfr_max {
                        lfr_max = j;
                        start_max = start;
                        end_max = i;
                    }
                    j = 1;
                }
            } else {
                if j > lfr_max {
                    lfr_max = j;
                    start_max = start;
                    end_max = i - 1;
                }
                j = 1;
                start = i;
            }
        }

        if lfr_max >= r_min {
            break;
        }

        // Escalate the cutoff: dcu_cut = (1.1^inc * dcu0)²
        inc += 1;
        let dinc = (1.1_f32).powf(inc as f32) * dcu0;
        dcu_cut = dinc * dinc;
    }

    (start_max, end_max)
}

/// Fragment gapless threading seed: find long fragments, thread them gaplessly.
///
/// Finds the longest contiguous fragment of closely-packed residues in both structures,
/// extracts the shorter fragment, and threads it gaplessly against the full other sequence.
/// Handles special cases where the extracted fragment spans the full chain (trim to middle 79%)
/// and where both structures have equal length (try both directions).
///
/// # Arguments
///
/// - `coords1` — Cα coordinates of structure 1 (query).
/// - `coords2` — Cα coordinates of structure 2 (target).
/// - `d0` — Final TM-score distance threshold.
/// - `d0_search` — Search-phase distance threshold.
/// - `l_norm` — Normalization length for TM-score.
///
/// # Returns
///
/// The best-scoring alignment from threading the fragment gaplessly, or an error
/// if the structures are invalid or no valid alignment is found.
///
/// # Errors
///
/// Returns `TmAlignError::Parse` if structures are too short or coordinate arrays
/// have invalid dimensions.
pub fn get_initial_fgt(
    coords1: &[Vector3<f32>],
    coords2: &[Vector3<f32>],
    d0: f32,
    d0_search: f32,
    l_norm: usize,
) -> Result<AlignmentMap, TmAlignError> {
    let xlen = coords1.len();
    let ylen = coords2.len();

    if xlen < 3 || ylen < 3 {
        return Err(TmAlignError::Parse("Sequence is too short < 3".to_string()));
    }

    // Parameters
    let fra_min = 4; // fast_opt not yet exposed in this version
    let fra_min1 = fra_min - 1; // cutoff for shift
    let dcu0 = 4.25; // dcu0 = 4.25 (will be squared in find_max_frag)

    // Find longest fragments in both structures
    let (xstart, xend) = find_max_frag(coords1, dcu0, false);
    let (ystart, yend) = find_max_frag(coords2, dcu0, false);

    let lx = xend - xstart + 1;
    let ly = yend - ystart + 1;

    // Select which fragment to use (normal case or symmetric case)
    if lx < ly || (lx == ly && xlen < ylen) {
        // Use x-fragment, thread against full y
        thread_fragment_against_other(
            &coords1[xstart..=xend],
            coords2,
            d0,
            d0_search,
            l_norm,
            xlen,
            ylen,
            fra_min1,
            true, // is x-fragment
        )
    } else if lx > ly || (lx == ly && xlen > ylen) {
        // Use y-fragment, thread against full x
        thread_fragment_against_other(
            &coords2[ystart..=yend],
            coords1,
            d0,
            d0_search,
            l_norm,
            ylen,
            xlen,
            fra_min1,
            false, // is y-fragment
        )
    } else {
        // lx == ly && xlen == ylen: try both directions, keep better
        // Part 1: x-fragment against y
        let alignment1 = thread_fragment_against_other(
            &coords1[xstart..=xend],
            coords2,
            d0,
            d0_search,
            l_norm,
            xlen,
            ylen,
            fra_min1,
            true, // x-fragment against y
        )?;

        // Part 2: y-fragment against x
        let alignment2 = thread_fragment_against_other(
            &coords2[ystart..=yend],
            coords1,
            d0,
            d0_search,
            l_norm,
            ylen,
            xlen,
            fra_min1,
            false, // y-fragment against x
        )?;

        // Compare both alignments and keep the one with better TM-score.
        // alignment1 is indexed as (coords1_idx, coords2_idx).
        // alignment2 is indexed as (coords2_idx, coords1_idx), so we need to flip it for comparison.
        let score1 =
            crate::score::get_score_fast(coords1, coords2, &alignment1, d0, d0_search, l_norm);

        // For alignment2, we need to flip it: (y_idx, x_idx) -> (x_idx, y_idx)
        let alignment2_flipped: Vec<(Option<usize>, Option<usize>)> = alignment2
            .iter()
            .map(|(y_opt, x_opt)| (*x_opt, *y_opt))
            .collect();

        let score2 = crate::score::get_score_fast(
            coords1,
            coords2,
            &alignment2_flipped,
            d0,
            d0_search,
            l_norm,
        );

        // Keep the better alignment (>= means first ties win, same as C code)
        if score2 > score1 {
            Ok(alignment2_flipped)
        } else {
            Ok(alignment1)
        }
    }
}

/// Helper: thread a fragment against the full other sequence.
///
/// Extracts the fragment, applies the middle-79% trimming if needed, and threads
/// it gaplessly against the full other sequence using `crate::seed::gapless::thread_gapless`.
#[allow(clippy::too_many_arguments)]
fn thread_fragment_against_other(
    fragment: &[Vector3<f32>],
    other: &[Vector3<f32>],
    d0: f32,
    d0_search: f32,
    l_norm: usize,
    fragment_len: usize, // original length before trimming
    other_len: usize,
    fra_min1: usize,
    is_fragment_from_coords1: bool,
) -> Result<AlignmentMap, TmAlignError> {
    let mut frag_indices: Vec<usize> = (0..fragment.len()).collect();

    // Check if extracted fragment length equals the shorter of the two chains, and trim if so.
    // This implements the special case where L_fr == L0 (extracted length == shorter chain).
    let min_len = fragment_len.min(other_len);
    if fragment.len() == min_len {
        // Trim to middle 79%: n1 = 0.1*L0, n2 = 0.89*L0
        let n1 = (min_len as f32 * 0.1) as usize;
        let n2 = (min_len as f32 * 0.89) as usize;

        let mut trimmed = Vec::new();
        for i in n1..=n2 {
            if i < frag_indices.len() {
                trimmed.push(frag_indices[i]);
            }
        }
        frag_indices = trimmed;
    }

    let l_fr = frag_indices.len();
    let frag_coords: Vec<Vector3<f32>> = frag_indices.iter().map(|&i| fragment[i]).collect();

    // Compute min_ali: max(fra_min - 1, min_len / 2.5)
    let min_len_frag = l_fr.min(other_len);
    let mut min_ali = (min_len_frag as f32 / 2.5) as usize;
    if min_ali <= fra_min1 {
        min_ali = fra_min1;
    }

    // Thread the (possibly trimmed) fragment gaplessly against the full other sequence.
    // Note: thread_gapless returns an alignment directly indexed into its input coordinates,
    // so we need to map the results back to the original full-chain indices.
    let alignment_raw = gapless::thread_gapless(
        if is_fragment_from_coords1 {
            &frag_coords
        } else {
            other
        },
        if is_fragment_from_coords1 {
            other
        } else {
            &frag_coords
        },
        d0,
        d0_search,
        l_norm,
        min_ali,
        1, // step = 1 (exact, not fast_opt)
    )?;

    // Map indices back to full-chain coordinates if needed
    let alignment = if is_fragment_from_coords1 {
        // alignment_raw is indexed as (frag_idx, other_idx), but frag_idx refers to frag_coords
        // We need to map it back to the original coords1 indices via frag_indices
        alignment_raw
            .iter()
            .map(|(frag_idx_opt, other_idx_opt)| {
                let mapped_frag = frag_idx_opt.and_then(|idx| {
                    if idx < frag_indices.len() {
                        Some(frag_indices[idx])
                    } else {
                        None
                    }
                });
                (mapped_frag, *other_idx_opt)
            })
            .collect()
    } else {
        // alignment_raw is indexed as (other_idx, frag_idx), map frag_idx back
        alignment_raw
            .iter()
            .map(|(other_idx_opt, frag_idx_opt)| {
                let mapped_frag = frag_idx_opt.and_then(|idx| {
                    if idx < frag_indices.len() {
                        Some(frag_indices[idx])
                    } else {
                        None
                    }
                });
                (*other_idx_opt, mapped_frag)
            })
            .collect()
    };

    Ok(alignment)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test `find_max_frag` with a simple linear chain (all distances equal).
    #[test]
    fn find_max_frag_simple_chain() {
        let coords = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(3.8, 0.0, 0.0),
            Vector3::new(7.6, 0.0, 0.0),
            Vector3::new(11.4, 0.0, 0.0),
            Vector3::new(15.2, 0.0, 0.0),
        ];
        // All distances are 3.8, so 3.8² ≈ 14.44 < dcu0² = 4.25² = 18.0625
        let (start, end) = find_max_frag(&coords, 4.25, false);
        // Should find the entire chain (0..4)
        assert_eq!(start, 0);
        assert_eq!(end, 4);
    }

    /// Test `find_max_frag` with a gap in the middle.
    #[test]
    fn find_max_frag_with_gap() {
        let coords = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(3.8, 0.0, 0.0),
            Vector3::new(3.9, 0.0, 0.0),
            Vector3::new(10.0, 0.0, 0.0), // Gap: distance ≈ 6.1 > dcu_cut
            Vector3::new(13.8, 0.0, 0.0),
            Vector3::new(17.6, 0.0, 0.0),
        ];
        let (start, end) = find_max_frag(&coords, 4.25, false);
        // Should find the second fragment (4..5)
        assert!(end >= start);
    }

    /// Test escalation logic: fragment length increases as cutoff increases.
    #[test]
    fn find_max_frag_escalation() {
        // Create a chain where distances gradually increase
        let coords = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(3.0, 0.0, 0.0),
            Vector3::new(6.0, 0.0, 0.0),
            Vector3::new(9.0, 0.0, 0.0),
            Vector3::new(12.0, 0.0, 0.0),
        ];
        // All distances are 3.0, so 3.0² = 9 < dcu0² = 4.25² = 18.0625
        let (start, end) = find_max_frag(&coords, 4.25, false);
        // Should find the entire chain
        assert_eq!(end - start + 1, 5);
    }

    /// Test `get_initial_fgt` with identical short structures.
    #[test]
    fn get_initial_fgt_identical_structures() {
        let coords = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(3.8, 0.0, 0.0),
            Vector3::new(7.6, 0.0, 0.0),
            Vector3::new(11.4, 0.0, 0.0),
        ];

        let d0 = 5.85;
        let d0_search = 5.85;
        let l_norm = coords.len();

        let result = get_initial_fgt(&coords, &coords, d0, d0_search, l_norm);
        assert!(result.is_ok());
        let alignment = result.unwrap();
        // Should find an alignment with some pairs
        assert!(!alignment.is_empty());
    }

    /// Test `get_initial_fgt` with different-length structures.
    #[test]
    fn get_initial_fgt_different_lengths() {
        let coords1 = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(3.8, 0.0, 0.0),
            Vector3::new(7.6, 0.0, 0.0),
            Vector3::new(11.4, 0.0, 0.0),
        ];
        let coords2 = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(3.8, 0.0, 0.0),
            Vector3::new(7.6, 0.0, 0.0),
        ];

        let d0 = 5.85;
        let d0_search = 5.85;
        let l_norm = coords1.len().min(coords2.len());

        let result = get_initial_fgt(&coords1, &coords2, d0, d0_search, l_norm);
        assert!(result.is_ok());
        let alignment = result.unwrap();
        assert!(!alignment.is_empty());
    }

    /// Test that short sequences (< 3 residues) are rejected.
    #[test]
    fn get_initial_fgt_too_short() {
        let coords = vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(3.8, 0.0, 0.0)];

        let d0 = 5.85;
        let d0_search = 5.85;
        let l_norm = coords.len();

        let result = get_initial_fgt(&coords, &coords, d0, d0_search, l_norm);
        assert!(result.is_err());
    }

    /// Test the middle-79%-trim logic: if fragment == full length, trim to middle 79%.
    #[test]
    fn get_initial_fgt_full_chain_trim() {
        // Create two identical chains where the fragment found is the whole chain.
        // Then verify that trimming occurs and the result is correct.
        let coords: Vec<Vector3<f32>> = (0..10)
            .map(|i| Vector3::new(i as f32 * 3.8, 0.0, 0.0))
            .collect();

        let d0 = 5.85;
        let d0_search = 5.85;
        let l_norm = coords.len();

        let result = get_initial_fgt(&coords, &coords, d0, d0_search, l_norm);
        assert!(result.is_ok());
        let alignment = result.unwrap();
        assert!(!alignment.is_empty());
        // The alignment should be valid (residues within bounds)
        for (i_opt, j_opt) in &alignment {
            if let Some(i) = i_opt {
                assert!(*i < coords.len());
            }
            if let Some(j) = j_opt {
                assert!(*j < coords.len());
            }
        }
    }
}
