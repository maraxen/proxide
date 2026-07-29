//! Initial seeding strategies for TM-align.
//!
//! TM-align uses 5 independent initial seeding strategies to generate candidate
//! alignments; each is refined via [`crate::refine::DP_iter`] to produce a final
//! TM-score. The best across all seeds (and their gap-open variants) is selected
//! as the final alignment. This module defines the [`SeedKind`] enum and a
//! dispatch function [`run_seed`] that orchestrates seed selection and execution.
//!
//! Individual seed implementations live in submodules:
//! - [`gapless`] — `get_initial()`: slide sequence 2 over sequence 1 gaplessly.
//! - [`secondary_structure`] — `get_initial_ss()`: NW-DP with secondary-structure scoring.
//! - [`local_structure`] — `get_initial5()`: local Kabsch fit + NW-DP on fragments.
//! - [`ss_plus`] — `get_initial_ssplus()`: combine local superposition + SS scoring.
//! - [`fragment_gapless`] — `get_initial_fgt()`: find and thread long contiguous fragments.

pub mod fragment_gapless;
pub mod gapless;
pub mod local_structure;
pub mod secondary_structure;
pub mod ss_plus;

use crate::error::TmAlignError;
use nalgebra::Vector3;

/// Discrete set of TM-align seeding strategies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeedKind {
    /// Gapless threading: slide sequence 2 over sequence 1 by offset.
    GaplessThreading,
    /// Secondary-structure alignment: NW-DP with SS-letter identity scoring.
    SecondaryStructure,
    /// Local-structure superposition: fragment-based Kabsch + NW-DP.
    LocalStructure,
    /// Local superposition + SS combo: Kabsch rotate, then SS-weighted DP.
    SsPlus,
    /// Fragment gapless threading: find long fragments, thread gaplessly.
    FragmentGapless,
}

/// Alignment map: residue-index pairs from structure 1 to structure 2.
///
/// `(Some(i), Some(j))` means residue `i` in structure 1 aligns to residue `j` in structure 2.
/// `(Some(i), None)` and `(None, Some(j))` represent gaps.
pub type AlignmentMap = Vec<(Option<usize>, Option<usize>)>;

/// Run a single seeding strategy.
///
/// Executes the specified seeding strategy and returns an alignment map.
/// Each strategy generates an initial alignment that is later refined via
/// `DP_iter` (not yet landed — a separate follow-up module).
///
/// # Arguments
///
/// - `kind` — Which of the 5 seeding strategies to run.
/// - `coords1` — Cα coordinates of structure 1.
/// - `coords2` — Cα coordinates of structure 2.
/// - `d0` — Final TM-score distance threshold (from `d0::d0_final`).
/// - `d0_search` — Search-phase threshold (from `d0::d0_search`).
/// - `l_norm` — Normalization length (typically `min(len1, len2)`).
/// - `previous_alignment` — Required only by [`SeedKind::SsPlus`], which
///   Kabsch-fits an earlier seed's (typically gapless threading's) best
///   alignment before building its score matrix. `None` for every other
///   seed kind.
///
/// # Returns
///
/// An `AlignmentMap` if successful, or a `TmAlignError` if the seeding strategy
/// fails (e.g., no valid alignment found, invalid input dimensions, or
/// [`SeedKind::SsPlus`] was requested without a `previous_alignment`).
///
/// # Errors
///
/// Returns `TmAlignError` if:
/// - Coordinate lengths are too short for the requested strategy.
/// - The seeding strategy produces no valid alignment candidates.
/// - `SeedKind::SsPlus` is requested with `previous_alignment = None`, or the
///   supplied alignment has no aligned (non-gap) pairs to Kabsch-fit.
#[allow(clippy::too_many_arguments)]
pub fn run_seed(
    kind: SeedKind,
    coords1: &[Vector3<f32>],
    coords2: &[Vector3<f32>],
    d0: f32,
    d0_search: f32,
    l_norm: usize,
    previous_alignment: Option<&AlignmentMap>,
) -> Result<AlignmentMap, TmAlignError> {
    match kind {
        SeedKind::GaplessThreading => gapless::get_initial(coords1, coords2, d0, d0_search, l_norm),
        SeedKind::SecondaryStructure => secondary_structure::get_initial_ss(coords1, coords2),
        SeedKind::LocalStructure => {
            local_structure::get_initial5(coords1, coords2, d0, d0_search, l_norm)
        }
        SeedKind::SsPlus => {
            let previous = previous_alignment.ok_or_else(|| {
                TmAlignError::Parse(
                    "SsPlus seed requires a previous_alignment (e.g. from GaplessThreading)"
                        .to_string(),
                )
            })?;
            ss_plus::get_initial_ssplus(coords1, coords2, previous, d0)
        }
        SeedKind::FragmentGapless => {
            fragment_gapless::get_initial_fgt(coords1, coords2, d0, d0_search, l_norm)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ss_plus_without_previous_alignment_errors() {
        let coords = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(2.0, 0.0, 0.0),
        ];
        let result = run_seed(SeedKind::SsPlus, &coords, &coords, 2.0, 5.0, 3, None);
        assert!(result.is_err());
    }

    #[test]
    fn gapless_threading_dispatches_correctly() {
        let coords = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(2.0, 0.0, 0.0),
            Vector3::new(3.0, 0.0, 0.0),
            Vector3::new(4.0, 0.0, 0.0),
        ];
        let result = run_seed(
            SeedKind::GaplessThreading,
            &coords,
            &coords,
            2.0,
            5.0,
            coords.len(),
            None,
        );
        assert!(result.is_ok());
    }
}
