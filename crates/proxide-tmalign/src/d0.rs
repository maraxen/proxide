//! TM-score length-normalization (`d0`) formulas, ported verbatim from
//! USalign's `param_set.h` (`parameter_set4final`/`parameter_set4search`).
//!
//! Reference: Zhang Y, Skolnick J. "Scoring function for automated
//! assessment of protein structure template quality." *Proteins.*
//! 2004;57(4):702-10 — origin of the empirical `d0(L)` formula.

/// Minimum d0 floor used by the final, reported TM-score.
pub const D0_MIN_FINAL: f32 = 0.5;

fn base_d0(l_norm: usize) -> f32 {
    if l_norm <= 21 {
        0.5
    } else {
        1.24 * (l_norm as f32 - 15.0).cbrt() - 1.8
    }
}

/// `d0` for the FINAL, reported TM-score at normalization length `l_norm`
/// (residues). Verbatim port of `parameter_set4final`.
pub fn d0_final(l_norm: usize) -> f32 {
    base_d0(l_norm).max(D0_MIN_FINAL)
}

/// `parameter_set4search`'s own small-`Lnorm` branch/floor — NOT the same as
/// [`base_d0`] (`parameter_set4final`'s `Lnorm<=21 -> 0.5`). Verbatim port of
/// `param_set.h:18-19`: `Lnorm<=19 -> d0=0.168`.
fn search_base_d0(l_norm: usize) -> f32 {
    if l_norm <= 19 {
        0.168
    } else {
        1.24 * (l_norm as f32 - 15.0).cbrt() - 1.8
    }
}

/// `(d0, d0_search)` used during the SEARCH-phase (seeding/refinement)
/// scoring — looser than [`d0_final`] so a promising-but-not-yet-optimal
/// alignment isn't discarded early. Verbatim port of
/// `parameter_set4search`: `search_base_d0`, then `D0_MIN = d0 + 0.8`
/// used as `d0`, with `d0_search` further clamped to `[4.5, 8.0]`.
pub fn d0_search(l_norm: usize) -> (f32, f32) {
    let d0 = search_base_d0(l_norm) + 0.8;
    let d0_search = d0.clamp(4.5, 8.0);
    (d0, d0_search)
}

/// The `score_d8` cutoff excluding poorly-superposed pairs from the
/// search-phase TM-score sum: `1.5 * l_norm^0.3 + 3.5`.
pub fn score_d8(l_norm: usize) -> f32 {
    1.5 * (l_norm as f32).powf(0.3) + 3.5
}

/// TM-score of a set of already-superposed aligned Cα pairs, given `d0`
/// and the normalization length `l_norm`: `TM = (1/l_norm) * sum_i
/// 1/(1+(d_i/d0)^2)`.
pub fn tm_score(squared_distances: &[f32], d0: f32, l_norm: usize) -> f32 {
    if l_norm == 0 {
        return 0.0;
    }
    let d0_sq = d0 * d0;
    let sum: f32 = squared_distances
        .iter()
        .map(|&d2| 1.0 / (1.0 + d2 / d0_sq))
        .sum();
    sum / (l_norm as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn d0_final_floor_for_short_structures() {
        assert_relative_eq!(d0_final(10), 0.5);
        assert_relative_eq!(d0_final(21), 0.5);
    }

    #[test]
    fn d0_final_matches_live_usalign_reference_output() {
        // Live smoke test this session: `TMalign PDB1.pdb PDB2.pdb` (USalign
        // commit 177cc8a, version 20240303) reported:
        //   "TM-score= 0.42654 (normalized by length of Structure_1: L=250, d0=5.85)"
        //   "TM-score= 0.61629 (normalized by length of Structure_2: L=166, d0=4.80)"
        assert_relative_eq!(d0_final(250), 5.85, epsilon = 0.01);
        assert_relative_eq!(d0_final(166), 4.80, epsilon = 0.01);
    }

    #[test]
    fn d0_search_is_looser_than_d0_final_and_clamped() {
        let (d0, d0_search) = d0_search(250);
        assert!(d0 > d0_final(250));
        assert!((4.5..=8.0).contains(&d0_search));
    }

    #[test]
    fn tm_score_of_perfect_overlap_is_one() {
        let zeros = vec![0.0_f32; 100];
        assert_relative_eq!(tm_score(&zeros, d0_final(100), 100), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn tm_score_is_zero_for_empty_normalization() {
        assert_eq!(tm_score(&[], 5.0, 0), 0.0);
    }
}
