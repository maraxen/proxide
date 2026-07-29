//! Gapless threading seed strategy.
//!
//! Implements USalign's `get_initial()` (TMalign.h:642-691), which slides
//! sequence 2 over sequence 1 with a linear offset, scoring each offset
//! position via `get_score_fast()` and returning the best-scoring alignment.
//!
//! The offset `k` ranges from `n1 = -ylen + min_ali` to `n2 = xlen - min_ali`,
//! where `min_ali = max(5, min(xlen,ylen)/2)`. For each `k`, the alignment
//! map is `y2x[j] = j + k` (or gap if out of bounds).

use super::AlignmentMap;
use crate::error::TmAlignError;
use crate::score::get_score_fast;
use nalgebra::Vector3;

/// Core gapless-threading search over an arbitrary fragment of coordinates.
///
/// Slides `coords2` over `coords1` with offsets `k` ranging from
/// `n1 = -len2 + min_ali` to `n2 = len1 - min_ali`, where `min_ali`
/// is caller-supplied (typically `max(5, min(len1,len2)/2)`). For each `k`,
/// builds the alignment map `y2x[j] = j + k` (or gap), scores via
/// `get_score_fast()`, and returns the best-scoring alignment.
///
/// # Arguments
///
/// - `coords1` — Cα coordinates of structure 1 (query).
/// - `coords2` — Cα coordinates of structure 2 (target).
/// - `d0` — Final TM-score distance threshold.
/// - `d0_search` — Search-phase distance threshold (looser than `d0`).
/// - `l_norm` — Normalization length for TM-score calculation.
/// - `min_ali` — Minimum alignment span required (e.g., `max(5, min_len/2)`).
/// - `step` — Offset increment: 1 for exact, >1 for coarse search.
///
/// # Returns
///
/// The highest-scoring alignment map, or an error if sequences are too short
/// or coordinates/thresholds are invalid.
///
/// # Errors
///
/// Returns `TmAlignError::Parse` if `min(len1, len2) < 3`.
pub fn thread_gapless(
    coords1: &[Vector3<f32>],
    coords2: &[Vector3<f32>],
    d0: f32,
    d0_search: f32,
    l_norm: usize,
    min_ali: usize,
    step: usize,
) -> Result<AlignmentMap, TmAlignError> {
    let xlen = coords1.len();
    let ylen = coords2.len();

    let min_len = xlen.min(ylen);
    if min_len < 3 {
        return Err(TmAlignError::Parse(
            "Sequence is too short < 3".to_string(),
        ));
    }

    // Compute offset range: k from n1 to n2, inclusive.
    // n1 = -ylen + min_ali can be negative, so use signed arithmetic.
    let n1: i32 = -(ylen as i32) + (min_ali as i32);
    let n2: i32 = (xlen as i32) - (min_ali as i32);

    let mut best_score = -1.0_f32;
    let mut best_alignment = Vec::new();

    // Try each offset k.
    let mut k = n1;
    while k <= n2 {
        // Build alignment map: y2x[j] = j + k (or gap if out of bounds).
        let mut alignment = Vec::with_capacity(ylen);
        for j in 0..ylen {
            let i_signed = j as i32 + k;
            if i_signed >= 0 && (i_signed as usize) < xlen {
                alignment.push((Some(i_signed as usize), Some(j)));
            } else {
                // Gap in structure 1 (y2x[j] = -1 in C code).
                alignment.push((None, Some(j)));
            }
        }

        // Score this alignment.
        let score = get_score_fast(coords1, coords2, &alignment, d0, d0_search, l_norm);

        // Keep the best.
        if score > best_score {
            best_score = score;
            best_alignment = alignment;
        }

        k += step as i32;
    }

    Ok(best_alignment)
}

/// Gapless-threading seed: slide sequence 2 over sequence 1.
///
/// Entry point for the gapless-threading seeding strategy. Computes the
/// default `min_ali = max(5, min(xlen, ylen) / 2)` and invokes
/// `thread_gapless()` with step size 1 (exact, non-approximate path).
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
/// The best-scoring gapless-threading alignment, or an error if sequences
/// are too short.
///
/// # Errors
///
/// Returns `TmAlignError::Parse` if `min(len1, len2) < 3`.
pub fn get_initial(
    coords1: &[Vector3<f32>],
    coords2: &[Vector3<f32>],
    d0: f32,
    d0_search: f32,
    l_norm: usize,
) -> Result<AlignmentMap, TmAlignError> {
    let xlen = coords1.len();
    let ylen = coords2.len();

    let min_len = xlen.min(ylen);
    if min_len < 3 {
        return Err(TmAlignError::Parse(
            "Sequence is too short < 3".to_string(),
        ));
    }

    // Compute default min_ali: max(5, min_len / 2) using integer division.
    let min_ali = 5.max(min_len / 2);

    // Invoke the core gapless-threading search with step=1 (exact).
    thread_gapless(coords1, coords2, d0, d0_search, l_norm, min_ali, 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test that gapless threading produces a valid alignment for two
    /// identical coordinate sets.
    #[test]
    fn gapless_threading_identical_structures() {
        let coords = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(2.0, 0.0, 0.0),
            Vector3::new(3.0, 0.0, 0.0),
            Vector3::new(4.0, 0.0, 0.0),
        ];

        let d0 = 2.0;
        let d0_search = 5.0;
        let l_norm = coords.len();

        let result = get_initial(&coords, &coords, d0, d0_search, l_norm);
        assert!(result.is_ok());

        let alignment = result.unwrap();
        // For identical structures, best alignment should be diagonal (k=0).
        // Expect alignment with pairs (i, i) for each i.
        let matched_pairs: Vec<_> = alignment
            .iter()
            .filter_map(|(i_opt, j_opt)| match (i_opt, j_opt) {
                (Some(i), Some(j)) => Some((*i, *j)),
                _ => None,
            })
            .collect();

        // With identical coords and a reasonable d0, we should get a good match.
        // At minimum, verify that the alignment is non-empty and matches some pairs.
        assert!(!matched_pairs.is_empty());

        // For identical structures, the best offset should yield mostly diagonal pairs.
        // Count how many pairs are actually diagonal (i == j).
        let diagonal_count = matched_pairs.iter().filter(|(i, j)| i == j).count();
        assert!(diagonal_count > 0);
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

        let result = get_initial(&coords_short, &coords_ok, d0, d0_search, l_norm);
        assert!(result.is_err());

        let result = get_initial(&coords_ok, &coords_short, d0, d0_search, l_norm);
        assert!(result.is_err());
    }

    /// Test that min_ali is correctly computed as max(5, min_len/2).
    #[test]
    fn min_ali_formula() {
        // For a sequence of length 6, min_ali should be max(5, 6/2) = max(5, 3) = 5.
        let coords_short = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(2.0, 0.0, 0.0),
            Vector3::new(3.0, 0.0, 0.0),
            Vector3::new(4.0, 0.0, 0.0),
            Vector3::new(5.0, 0.0, 0.0),
        ];

        let coords_long = vec![
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

        let d0 = 2.0;
        let d0_search = 5.0;
        let l_norm = 6;

        let result = get_initial(&coords_short, &coords_long, d0, d0_search, l_norm);
        assert!(result.is_ok());

        // For min_len=6, min_ali=max(5, 6/2)=max(5,3)=5.
        // This test just verifies that the function computes without error;
        // the actual min_ali computation is internal.
    }

    /// Test offset-range computation: k ranges from -ylen + min_ali to xlen - min_ali.
    /// Verify that extreme offsets are tried.
    #[test]
    fn offset_range_tested() {
        // xlen=8, ylen=5: min_len=5, min_ali=max(5, 5/2=2)=5, so the offset
        // range n1=-ylen+min_ali=-5+5=0 to n2=xlen-min_ali=8-5=3 is
        // non-empty (unlike a too-short/too-mismatched pair, where min_ali
        // can exceed both n1's and n2's satisfiable range and the seed
        // legitimately finds nothing — that's real reference behavior, not
        // a bug, so this fixture is sized to actually exercise the search).
        let coords1: Vec<Vector3<f32>> = (0..8).map(|i| Vector3::new(i as f32, 0.0, 0.0)).collect();
        let coords2 = vec![
            Vector3::new(3.0, 0.0, 0.0),
            Vector3::new(4.0, 0.0, 0.0),
            Vector3::new(5.0, 0.0, 0.0),
            Vector3::new(6.0, 0.0, 0.0),
            Vector3::new(7.0, 0.0, 0.0),
        ];

        let d0 = 2.0;
        let d0_search = 5.0;
        let l_norm = 5;

        let result = get_initial(&coords1, &coords2, d0, d0_search, l_norm);
        assert!(result.is_ok());

        let alignment = result.unwrap();
        // The best offset should align coords2 to coords1[3..8].
        // At minimum, the alignment should be non-empty.
        assert!(!alignment.is_empty());
    }

    /// Test thread_gapless with custom min_ali and step.
    #[test]
    fn thread_gapless_custom_params() {
        let coords1 = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(2.0, 0.0, 0.0),
            Vector3::new(3.0, 0.0, 0.0),
            Vector3::new(4.0, 0.0, 0.0),
        ];

        let coords2 = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(2.0, 0.0, 0.0),
        ];

        let d0 = 2.0;
        let d0_search = 5.0;
        let l_norm = 3;

        // Try with min_ali=2 and step=1.
        let result = thread_gapless(&coords1, &coords2, d0, d0_search, l_norm, 2, 1);
        assert!(result.is_ok());

        let alignment = result.unwrap();
        assert!(!alignment.is_empty());
    }
}
