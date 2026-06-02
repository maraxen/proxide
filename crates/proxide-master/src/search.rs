//! Fragment search: parallel and serial RMSD search over a [`FragmentDb`].

use crate::db::{FragmentDb, SourceLabel};
use crate::fragment::{Centered, Fragment};
use crate::kabsch::kabsch_rmsd;
use rayon::prelude::*;

// ---------------------------------------------------------------------------
// SearchResult
// ---------------------------------------------------------------------------

/// A single match returned by [`FragmentDb::search`].
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// RMSD between the query and this database fragment (Å).
    pub rmsd: f32,
    /// Structural provenance of the matching fragment.
    pub label: SourceLabel,
    /// Optimal rotation matrix aligning the query onto this fragment (row-major).
    pub rotation: [[f32; 3]; 3],
}

// ---------------------------------------------------------------------------
// FragmentDb::search / search_serial
// ---------------------------------------------------------------------------

impl<const N: usize> FragmentDb<N> {
    /// Search the database in parallel for all fragments within `epsilon` RMSD
    /// of `query`.
    ///
    /// Results are sorted by ascending RMSD.
    pub fn search(&self, query: &Fragment<N, Centered>, epsilon: f32) -> Vec<SearchResult> {
        let query_norm_sq = query.norm_sq();

        // Reconstruct a Fragment<N, Centered> view from each entry for kabsch_rmsd.
        let mut results: Vec<SearchResult> = self
            .entries
            .par_iter()
            .filter_map(|entry| {
                // Re-wrap raw coords slice as a Fragment<N, Centered>.
                let db_frag = wrap_centered(entry.coords);
                let kr = kabsch_rmsd(query, query_norm_sq, &db_frag, entry.norm_sq);
                if kr.rmsd <= epsilon {
                    Some(SearchResult {
                        rmsd: kr.rmsd,
                        label: entry.label.clone(),
                        rotation: kr.rotation,
                    })
                } else {
                    None
                }
            })
            .collect();

        results.sort_unstable_by(|a, b| {
            a.rmsd
                .partial_cmp(&b.rmsd)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results
    }

    /// Serial (single-threaded) version of [`FragmentDb::search`].
    ///
    /// Produces identical results to `search`; useful for testing and for
    /// callers that manage their own thread pools.
    pub fn search_serial(&self, query: &Fragment<N, Centered>, epsilon: f32) -> Vec<SearchResult> {
        let query_norm_sq = query.norm_sq();

        let mut results: Vec<SearchResult> = self
            .entries
            .iter()
            .filter_map(|entry| {
                let db_frag = wrap_centered(entry.coords);
                let kr = kabsch_rmsd(query, query_norm_sq, &db_frag, entry.norm_sq);
                if kr.rmsd <= epsilon {
                    Some(SearchResult {
                        rmsd: kr.rmsd,
                        label: entry.label.clone(),
                        rotation: kr.rotation,
                    })
                } else {
                    None
                }
            })
            .collect();

        results.sort_unstable_by(|a, b| {
            a.rmsd
                .partial_cmp(&b.rmsd)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Zero-copy re-wrap of a raw coord array as a `Fragment<N, Centered>`.
///
/// This is safe because `FragmentDbEntry::coords` are guaranteed centered
/// (they were produced by `fragment.center()`).
fn wrap_centered<const N: usize>(coords: [[[f32; 3]; 4]; N]) -> Fragment<N, Centered> {
    // SAFETY: Fragment<N, Centered> is repr(Rust) with coords as first field
    // and _state: PhantomData as second (zero-sized). We construct it via
    // Fragment::new_centered which exists only within this crate.
    // We cannot use Fragment { coords, _state: PhantomData } from here
    // because _state is private. Use crate::fragment::Fragment::new_centered.
    crate::fragment::Fragment::new_centered(coords)
}
