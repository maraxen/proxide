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
/// [`crate::refine::DP_iter`].
///
/// # Arguments
///
/// - `kind` — Which of the 5 seeding strategies to run.
/// - `coords1` — Cα coordinates of structure 1.
/// - `coords2` — Cα coordinates of structure 2.
/// - `seq1` — One-letter amino acid codes for structure 1.
/// - `seq2` — One-letter amino acid codes for structure 2.
/// - `d0` — Final TM-score distance threshold (from `d0::d0_final`).
/// - `d0_search` — Search-phase threshold (from `d0::d0_search`).
/// - `l_norm` — Normalization length (typically `min(len1, len2)`).
///
/// # Returns
///
/// An `AlignmentMap` if successful, or a `TmAlignError` if the seeding strategy
/// fails (e.g., no valid alignment found, invalid input dimensions).
///
/// # Errors
///
/// Returns `TmAlignError` if:
/// - Coordinate and sequence lengths are mismatched.
/// - The seeding strategy produces no valid alignment candidates.
/// - Internal Kabsch or DP operations fail.
pub fn run_seed(
    kind: SeedKind,
    _coords1: &[Vector3<f32>],
    _coords2: &[Vector3<f32>],
    _seq1: &[u8],
    _seq2: &[u8],
    _d0: f32,
    _d0_search: f32,
    _l_norm: usize,
) -> Result<AlignmentMap, TmAlignError> {
    match kind {
        SeedKind::GaplessThreading => {
            // To be implemented in gapless.rs::get_initial
            todo!("gapless threading seed not yet implemented")
        }
        SeedKind::SecondaryStructure => {
            // To be implemented in secondary_structure.rs::get_initial_ss
            todo!("secondary structure seed not yet implemented")
        }
        SeedKind::LocalStructure => {
            // To be implemented in local_structure.rs::get_initial5
            todo!("local structure seed not yet implemented")
        }
        SeedKind::SsPlus => {
            // To be implemented in ss_plus.rs::get_initial_ssplus
            todo!("SS+local seed not yet implemented")
        }
        SeedKind::FragmentGapless => {
            // To be implemented in fragment_gapless.rs::get_initial_fgt
            todo!("fragment gapless seed not yet implemented")
        }
    }
}
