//! Sequence alignment algorithms
//!
//! Implements Smith-Waterman (local alignment) and Needleman-Wunsch (global alignment)
//! for protein sequence comparison.
//!
//! These are non-differentiable implementations optimized for speed.
//! For gradient-based learning, use the JAX implementations.

#![allow(dead_code)]

use std::cmp::max;

/// BLOSUM62 substitution matrix for amino acid alignment
/// Order: A R N D C Q E G H I L K M F P S T W Y V
const BLOSUM62: [[i32; 20]; 20] = [
    [
        4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0,
    ], // A
    [
        -1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3,
    ], // R
    [
        -2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3,
    ], // N
    [
        -2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3,
    ], // D
    [
        0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1,
    ], // C
    [
        -1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2,
    ], // Q
    [
        -1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2,
    ], // E
    [
        0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3,
    ], // G
    [
        -2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3,
    ], // H
    [
        -1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3,
    ], // I
    [
        -1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1,
    ], // L
    [
        -1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2,
    ], // K
    [
        -1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1,
    ], // M
    [
        -2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1,
    ], // F
    [
        -1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2,
    ], // P
    [
        1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2,
    ], // S
    [
        0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0,
    ], // T
    [
        -3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3,
    ], // W
    [
        -2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1,
    ], // Y
    [
        0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4,
    ], // V
];

/// Map amino acid single letter code to BLOSUM62 index
fn aa_to_index(aa: u8) -> usize {
    match aa {
        b'A' => 0,
        b'R' => 1,
        b'N' => 2,
        b'D' => 3,
        b'C' => 4,
        b'Q' => 5,
        b'E' => 6,
        b'G' => 7,
        b'H' => 8,
        b'I' => 9,
        b'L' => 10,
        b'K' => 11,
        b'M' => 12,
        b'F' => 13,
        b'P' => 14,
        b'S' => 15,
        b'T' => 16,
        b'W' => 17,
        b'Y' => 18,
        b'V' => 19,
        _ => 0, // Unknown maps to A (or could return option)
    }
}

/// Get substitution score from BLOSUM62
pub fn substitution_score(a: u8, b: u8) -> i32 {
    BLOSUM62[aa_to_index(a)][aa_to_index(b)]
}

/// Alignment result with score and traceback
#[derive(Debug, Clone)]
pub struct AlignmentResult {
    /// Alignment score
    pub score: i32,
    /// Aligned positions from sequence A (None = gap)
    pub aligned_a: Vec<Option<usize>>,
    /// Aligned positions from sequence B (None = gap)
    pub aligned_b: Vec<Option<usize>>,
    /// Aligned sequence A (with gaps as '-')
    pub aligned_seq_a: String,
    /// Aligned sequence B (with gaps as '-')
    pub aligned_seq_b: String,
}

/// Smith-Waterman local alignment with affine gap penalties
///
/// # Arguments
/// * `seq_a` - First sequence (amino acid bytes)
/// * `seq_b` - Second sequence (amino acid bytes)
/// * `gap_open` - Gap opening penalty (positive value)
/// * `gap_extend` - Gap extension penalty (positive value)
pub fn smith_waterman_affine(
    seq_a: &[u8],
    seq_b: &[u8],
    gap_open: i32,
    gap_extend: i32,
) -> AlignmentResult {
    let m = seq_a.len();
    let n = seq_b.len();

    if m == 0 || n == 0 {
        return AlignmentResult {
            score: 0,
            aligned_a: vec![],
            aligned_b: vec![],
            aligned_seq_a: String::new(),
            aligned_seq_b: String::new(),
        };
    }

    // Three matrices for affine gaps: M (match), I (insertion), D (deletion)
    let mut h = vec![vec![0i32; n + 1]; m + 1]; // Main scoring matrix
    let mut e = vec![vec![i32::MIN / 2; n + 1]; m + 1]; // Gap in seq_b
    let mut f = vec![vec![i32::MIN / 2; n + 1]; m + 1]; // Gap in seq_a

    let mut max_score = 0;
    let mut max_i = 0;
    let mut max_j = 0;

    // Fill matrices
    for i in 1..=m {
        for j in 1..=n {
            // Match/mismatch from diagonal
            let match_score = h[i - 1][j - 1] + substitution_score(seq_a[i - 1], seq_b[j - 1]);

            // Extension of gap in seq_b (E)
            e[i][j] = max(
                e[i][j - 1] - gap_extend,            // Extend gap
                h[i][j - 1] - gap_open - gap_extend, // Open gap
            );

            // Extension of gap in seq_a (F)
            f[i][j] = max(
                f[i - 1][j] - gap_extend,            // Extend gap
                h[i - 1][j] - gap_open - gap_extend, // Open gap
            );

            // Main matrix
            h[i][j] = max(0, max(match_score, max(e[i][j], f[i][j])));

            if h[i][j] > max_score {
                max_score = h[i][j];
                max_i = i;
                max_j = j;
            }
        }
    }

    // Traceback from max position
    let (aligned_a, aligned_b, aligned_seq_a, aligned_seq_b) =
        traceback_local(&h, &e, &f, seq_a, seq_b, max_i, max_j, gap_open, gap_extend);

    AlignmentResult {
        score: max_score,
        aligned_a,
        aligned_b,
        aligned_seq_a,
        aligned_seq_b,
    }
}

/// Traceback for local alignment
fn traceback_local(
    h: &[Vec<i32>],
    _e: &[Vec<i32>],
    f: &[Vec<i32>],
    seq_a: &[u8],
    seq_b: &[u8],
    mut i: usize,
    mut j: usize,
    _gap_open: i32,
    _gap_extend: i32,
) -> (Vec<Option<usize>>, Vec<Option<usize>>, String, String) {
    let mut aligned_a = Vec::new();
    let mut aligned_b = Vec::new();
    let mut seq_a_aligned = Vec::new();
    let mut seq_b_aligned = Vec::new();

    while i > 0 && j > 0 && h[i][j] > 0 {
        let current = h[i][j];
        let diagonal = h[i - 1][j - 1] + substitution_score(seq_a[i - 1], seq_b[j - 1]);

        if current == diagonal {
            // Match/mismatch
            aligned_a.push(Some(i - 1));
            aligned_b.push(Some(j - 1));
            seq_a_aligned.push(seq_a[i - 1]);
            seq_b_aligned.push(seq_b[j - 1]);
            i -= 1;
            j -= 1;
        } else if current == f[i][j] {
            // Gap in seq_b
            aligned_a.push(Some(i - 1));
            aligned_b.push(None);
            seq_a_aligned.push(seq_a[i - 1]);
            seq_b_aligned.push(b'-');
            i -= 1;
        } else {
            // Gap in seq_a
            aligned_a.push(None);
            aligned_b.push(Some(j - 1));
            seq_a_aligned.push(b'-');
            seq_b_aligned.push(seq_b[j - 1]);
            j -= 1;
        }
    }

    // Reverse since we built backwards
    aligned_a.reverse();
    aligned_b.reverse();
    seq_a_aligned.reverse();
    seq_b_aligned.reverse();

    (
        aligned_a,
        aligned_b,
        String::from_utf8_lossy(&seq_a_aligned).to_string(),
        String::from_utf8_lossy(&seq_b_aligned).to_string(),
    )
}

/// Needleman-Wunsch global alignment with linear gap penalty
///
/// # Arguments
/// * `seq_a` - First sequence (amino acid bytes)
/// * `seq_b` - Second sequence (amino acid bytes)
/// * `gap_penalty` - Gap penalty (positive value)
pub fn needleman_wunsch(seq_a: &[u8], seq_b: &[u8], gap_penalty: i32) -> AlignmentResult {
    let m = seq_a.len();
    let n = seq_b.len();

    if m == 0 && n == 0 {
        return AlignmentResult {
            score: 0,
            aligned_a: vec![],
            aligned_b: vec![],
            aligned_seq_a: String::new(),
            aligned_seq_b: String::new(),
        };
    }

    // Scoring matrix
    let mut h = vec![vec![0i32; n + 1]; m + 1];

    // Initialize first row and column
    for i in 0..=m {
        h[i][0] = -(i as i32 * gap_penalty);
    }
    for j in 0..=n {
        h[0][j] = -(j as i32 * gap_penalty);
    }

    // Fill matrix
    for i in 1..=m {
        for j in 1..=n {
            let match_score = h[i - 1][j - 1] + substitution_score(seq_a[i - 1], seq_b[j - 1]);
            let delete = h[i - 1][j] - gap_penalty;
            let insert = h[i][j - 1] - gap_penalty;

            h[i][j] = max(match_score, max(delete, insert));
        }
    }

    // Traceback from bottom-right
    let (aligned_a, aligned_b, aligned_seq_a, aligned_seq_b) =
        traceback_global(&h, seq_a, seq_b, gap_penalty);

    AlignmentResult {
        score: h[m][n],
        aligned_a,
        aligned_b,
        aligned_seq_a,
        aligned_seq_b,
    }
}

/// Traceback for global alignment
fn traceback_global(
    h: &[Vec<i32>],
    seq_a: &[u8],
    seq_b: &[u8],
    gap_penalty: i32,
) -> (Vec<Option<usize>>, Vec<Option<usize>>, String, String) {
    let mut aligned_a = Vec::new();
    let mut aligned_b = Vec::new();
    let mut seq_a_aligned = Vec::new();
    let mut seq_b_aligned = Vec::new();

    let mut i = seq_a.len();
    let mut j = seq_b.len();

    while i > 0 || j > 0 {
        if i > 0 && j > 0 {
            let diagonal = h[i - 1][j - 1] + substitution_score(seq_a[i - 1], seq_b[j - 1]);
            if h[i][j] == diagonal {
                // Match/mismatch
                aligned_a.push(Some(i - 1));
                aligned_b.push(Some(j - 1));
                seq_a_aligned.push(seq_a[i - 1]);
                seq_b_aligned.push(seq_b[j - 1]);
                i -= 1;
                j -= 1;
                continue;
            }
        }

        if i > 0 && h[i][j] == h[i - 1][j] - gap_penalty {
            // Gap in seq_b
            aligned_a.push(Some(i - 1));
            aligned_b.push(None);
            seq_a_aligned.push(seq_a[i - 1]);
            seq_b_aligned.push(b'-');
            i -= 1;
        } else {
            // Gap in seq_a
            aligned_a.push(None);
            aligned_b.push(Some(j - 1));
            seq_a_aligned.push(b'-');
            seq_b_aligned.push(seq_b[j - 1]);
            j -= 1;
        }
    }

    // Reverse since we built backwards
    aligned_a.reverse();
    aligned_b.reverse();
    seq_a_aligned.reverse();
    seq_b_aligned.reverse();

    (
        aligned_a,
        aligned_b,
        String::from_utf8_lossy(&seq_a_aligned).to_string(),
        String::from_utf8_lossy(&seq_b_aligned).to_string(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_substitution_score() {
        // Identical amino acids should have positive score
        assert!(substitution_score(b'A', b'A') > 0);
        assert!(substitution_score(b'W', b'W') > 0);

        // Different should vary
        let score = substitution_score(b'A', b'W');
        assert!(score < 4); // A-A is 4, A-W should be less
    }

    #[test]
    fn test_smith_waterman_identical() {
        let seq = b"ACDEFGHIKLMNPQRSTVWY";
        let result = smith_waterman_affine(seq, seq, 10, 1);

        // Perfect match should have high score
        assert!(result.score > 50);
        assert_eq!(result.aligned_seq_a.len(), result.aligned_seq_b.len());
    }

    #[test]
    fn test_smith_waterman_with_gap() {
        let seq_a = b"ACDGHIK";
        let seq_b = b"ACDEFGHIK"; // Has EF insertion

        let result = smith_waterman_affine(seq_a, seq_b, 10, 1);

        assert!(result.score > 0);
        // Should find local match around ACD and GHIK
    }

    #[test]
    fn test_needleman_wunsch_identical() {
        let seq = b"ACDEF";
        let result = needleman_wunsch(seq, seq, 5);

        assert!(result.score > 10);
        assert_eq!(result.aligned_seq_a, "ACDEF");
        assert_eq!(result.aligned_seq_b, "ACDEF");
    }

    #[test]
    fn test_needleman_wunsch_with_gap() {
        let seq_a = b"ACDF";
        let seq_b = b"ACDEF"; // Has E insertion

        let result = needleman_wunsch(seq_a, seq_b, 5);

        // Should align with gap
        assert!(result.aligned_seq_a.contains('-') || result.aligned_seq_b.contains('-'));
    }

    #[test]
    fn test_empty_sequences() {
        let result = smith_waterman_affine(b"", b"ACE", 10, 1);
        assert_eq!(result.score, 0);

        let result = needleman_wunsch(b"", b"", 5);
        assert_eq!(result.score, 0);
    }
}
