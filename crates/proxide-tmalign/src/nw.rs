//! Needleman-Wunsch / Gotoh global alignment with affine gap penalties.
//!
//! Used both to seed low-identity/secondary-structure-based initial
//! alignments and inside TM-align's `DP_iter` refinement loop (aligning
//! residues by rotated-frame proximity rather than sequence similarity) —
//! the same DP core serves both roles, only the similarity function
//! passed in differs.

/// One step of a global alignment path. `None` on either side denotes a
/// gap in that sequence.
pub type AlignedPair = (Option<usize>, Option<usize>);

#[derive(Clone, Copy, PartialEq)]
enum State {
    Match,
    GapInB,
    GapInA,
}

/// Global (Needleman-Wunsch/Gotoh) affine-gap alignment over a precomputed
/// similarity function.
///
/// `score(i, j)` is the pairwise similarity between residue `i` of
/// sequence A (`0..len_a`) and residue `j` of sequence B (`0..len_b`) —
/// higher is better (e.g. a TM-score-weighted proximity term inside
/// `DP_iter`, or a secondary-structure/sequence term when seeding).
///
/// `gap_open` is the penalty charged once when a gap begins; `gap_extend`
/// is charged per additional gap column. TM-align's own DP uses
/// `gap_extend = 0.0` — only the opening penalty matters.
pub fn needleman_wunsch_affine<F>(
    len_a: usize,
    len_b: usize,
    score: F,
    gap_open: f32,
    gap_extend: f32,
) -> Vec<AlignedPair>
where
    F: Fn(usize, usize) -> f32,
{
    if len_a == 0 || len_b == 0 {
        return Vec::new();
    }

    let neg_inf = f32::NEG_INFINITY;
    let rows = len_a + 1;
    let cols = len_b + 1;

    // m[i][j]  = best score of an alignment of A[..i]/B[..j] ending in a match/mismatch
    // ix[i][j] = ... ending in a gap in B (A[i-1] consumed, no B residue)
    // iy[i][j] = ... ending in a gap in A (B[j-1] consumed, no A residue)
    let mut m = vec![vec![neg_inf; cols]; rows];
    let mut ix = vec![vec![neg_inf; cols]; rows];
    let mut iy = vec![vec![neg_inf; cols]; rows];

    m[0][0] = 0.0;
    for (i, row) in ix.iter_mut().enumerate().take(rows).skip(1) {
        row[0] = gap_open + gap_extend * (i - 1) as f32;
    }
    for (j, cell) in iy[0].iter_mut().enumerate().take(cols).skip(1) {
        *cell = gap_open + gap_extend * (j - 1) as f32;
    }

    for i in 1..rows {
        for j in 1..cols {
            let s = score(i - 1, j - 1);
            let best_prev = m[i - 1][j - 1].max(ix[i - 1][j - 1]).max(iy[i - 1][j - 1]);
            m[i][j] = best_prev + s;

            ix[i][j] = (m[i - 1][j] + gap_open).max(ix[i - 1][j] + gap_extend);
            iy[i][j] = (m[i][j - 1] + gap_open).max(iy[i][j - 1] + gap_extend);
        }
    }

    let (mut i, mut j) = (rows - 1, cols - 1);
    let mut state = [
        (State::Match, m[i][j]),
        (State::GapInB, ix[i][j]),
        (State::GapInA, iy[i][j]),
    ]
    .into_iter()
    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
    .unwrap()
    .0;

    let mut path = Vec::with_capacity(len_a.max(len_b));
    while i > 0 || j > 0 {
        match state {
            State::Match => {
                path.push((Some(i - 1), Some(j - 1)));
                let s = score(i - 1, j - 1);
                let incoming = m[i][j] - s;
                state = if (m[i - 1][j - 1] - incoming).abs() < 1e-3 {
                    State::Match
                } else if (ix[i - 1][j - 1] - incoming).abs() < 1e-3 {
                    State::GapInB
                } else {
                    State::GapInA
                };
                i -= 1;
                j -= 1;
            }
            State::GapInB => {
                path.push((Some(i - 1), None));
                state = if (m[i - 1][j] + gap_open - ix[i][j]).abs() < 1e-3 {
                    State::Match
                } else {
                    State::GapInB
                };
                i -= 1;
            }
            State::GapInA => {
                path.push((None, Some(j - 1)));
                state = if (m[i][j - 1] + gap_open - iy[i][j]).abs() < 1e-3 {
                    State::Match
                } else {
                    State::GapInA
                };
                j -= 1;
            }
        }
    }
    path.reverse();
    path
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_sequences_align_gaplessly() {
        let path =
            needleman_wunsch_affine(4, 4, |i, j| if i == j { 1.0 } else { -1.0 }, -2.0, 0.0);
        assert_eq!(path.len(), 4);
        for (k, (a, b)) in path.iter().enumerate() {
            assert_eq!(*a, Some(k));
            assert_eq!(*b, Some(k));
        }
    }

    #[test]
    fn insertion_in_b_produces_a_gap_in_a() {
        // A = "AC" (len 2), B = "ABC" (len 3): a match at (0,0) and (1,2),
        // everything else mismatched — the optimal alignment should insert
        // a gap in A opposite B's middle residue.
        let score = |i: usize, j: usize| -> f32 {
            match (i, j) {
                (0, 0) => 1.0,
                (1, 2) => 1.0,
                _ => -1.0,
            }
        };
        let path = needleman_wunsch_affine(2, 3, score, -0.6, 0.0);
        assert!(path.contains(&(Some(0), Some(0))));
        assert!(path.contains(&(Some(1), Some(2))));
        assert!(path.contains(&(None, Some(1))));
    }

    #[test]
    fn empty_inputs_yield_empty_alignment() {
        assert!(needleman_wunsch_affine(0, 5, |_, _| 0.0, -1.0, 0.0).is_empty());
        assert!(needleman_wunsch_affine(5, 0, |_, _| 0.0, -1.0, 0.0).is_empty());
    }

    #[test]
    fn path_covers_full_length_with_gaps() {
        // Every A residue and every B residue appears exactly once across
        // the path (global alignment property), regardless of gaps.
        let path = needleman_wunsch_affine(2, 3, |i, j| if (i, j) == (0, 0) { 1.0 } else { -1.0 }, -1.0, 0.0);
        let a_seen: Vec<_> = path.iter().filter_map(|&(a, _)| a).collect();
        let b_seen: Vec<_> = path.iter().filter_map(|&(_, b)| b).collect();
        assert_eq!(a_seen, vec![0, 1]);
        assert_eq!(b_seen, vec![0, 1, 2]);
    }
}
