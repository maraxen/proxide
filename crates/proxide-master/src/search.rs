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

    /// Search the database in parallel for all fragments within `epsilon` RMSD
    /// of `query`, using a Cauchy–Schwarz norm-bound pre-filter to skip
    /// entries that cannot possibly match.
    ///
    /// Returns identical results to [`FragmentDb::search`] but skips expensive
    /// Kabsch SVD for entries failing the pre-filter test.
    ///
    /// **Pre-filter criterion:** If `(||A|| - ||B||)² > ε² · M`, where M = N*4,
    /// then RMSD > ε, so the entry is pruned.
    pub fn search_prefiltered(&self, query: &Fragment<N, Centered>, epsilon: f32) -> Vec<SearchResult> {
        let query_norm_sq = query.norm_sq();
        let query_norm = query_norm_sq.sqrt();
        let epsilon_sq_m = epsilon * epsilon * ((N * 4) as f32);

        let mut results: Vec<SearchResult> = self
            .entries
            .par_iter()
            .filter_map(|entry| {
                let entry_norm = entry.norm_sq.sqrt();
                let norm_diff = query_norm - entry_norm;

                // Pre-filter: if (||A|| - ||B||)² > ε² · M, RMSD > ε, skip Kabsch.
                if norm_diff * norm_diff > epsilon_sq_m {
                    return None;
                }

                // Passed pre-filter: run Kabsch.
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::SourceLabel;
    use crate::fragment::Raw;
    use rand::{Rng, SeedableRng};

    /// Measure prune rates at benchmark epsilon values (informational only).
    #[test]
    #[ignore]
    fn measure_prune_info() {
        eprintln!("\n\n=== PRUNE RATE MEASUREMENTS ===");
        eprintln!("N\t\tEpsilon\tPruned\tTotal\tRate%");

        for size in &[10_000, 100_000, 1_000_000] {
            let mut rng = rand::rngs::StdRng::seed_from_u64(42);
            let mut builder = crate::db::FragmentDbBuilder::<5>::new();

            for i in 0..*size {
                let mut coords: [[[f32; 3]; 4]; 5] = [[[0.0; 3]; 4]; 5];
                for r in 0..5 {
                    for at in 0..4 {
                        coords[r][at][0] = rng.gen_range(-20.0_f32..20.0);
                        coords[r][at][1] = rng.gen_range(-20.0_f32..20.0);
                        coords[r][at][2] = rng.gen_range(-20.0_f32..20.0);
                    }
                }
                let label = SourceLabel::new(&format!("DB{:07}", i), 'A', 1, ' ', 5, ' ');
                let frag = Fragment::<5, Raw>::new(coords);
                let _ = builder.add_fragment(frag, label);
            }
            let db = builder.build();

            // Query
            let mut rng_q = rand::rngs::StdRng::seed_from_u64(99);
            let mut query_coords: [[[f32; 3]; 4]; 5] = [[[0.0; 3]; 4]; 5];
            for r in 0..5 {
                for at in 0..4 {
                    query_coords[r][at][0] = rng_q.gen_range(-20.0_f32..20.0);
                    query_coords[r][at][1] = rng_q.gen_range(-20.0_f32..20.0);
                    query_coords[r][at][2] = rng_q.gen_range(-20.0_f32..20.0);
                }
            }
            let q_raw = Fragment::<5, Raw>::new(query_coords);
            let (query, _) = q_raw.center().unwrap();

            for epsilon in &[1.0_f32, 5.0_f32] {
                let query_norm = query.norm_sq().sqrt();
                let epsilon_sq_m = epsilon * epsilon * 20.0_f32;
                let mut pruned_count = 0;
                for entry in &db.entries {
                    let entry_norm = entry.norm_sq.sqrt();
                    let norm_diff = query_norm - entry_norm;
                    if norm_diff * norm_diff > epsilon_sq_m {
                        pruned_count += 1;
                    }
                }
                let prune_rate = pruned_count as f32 / db.len() as f32 * 100.0;
                eprintln!("{}\t\t{}\t{}\t{}\t{:.1}", size, epsilon, pruned_count, db.len(), prune_rate);
            }
        }
    }

    /// Parity test: search() and search_prefiltered() return identical results
    /// on a small random database.
    #[test]
    fn test_search_prefiltered_parity() {
        // Build a small random database (20 random 5-mers).
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut builder = crate::db::FragmentDbBuilder::<5>::new();

        for i in 0..20 {
            let mut coords: [[[f32; 3]; 4]; 5] = [[[0.0; 3]; 4]; 5];
            for r in 0..5 {
                for at in 0..4 {
                    coords[r][at][0] = rng.gen_range(-20.0_f32..20.0);
                    coords[r][at][1] = rng.gen_range(-20.0_f32..20.0);
                    coords[r][at][2] = rng.gen_range(-20.0_f32..20.0);
                }
            }
            let label = SourceLabel::new(&format!("TEST{:04}", i), 'A', 1, ' ', 5, ' ');
            let frag = Fragment::<5, Raw>::new(coords);
            let _ = builder.add_fragment(frag, label);
        }
        let db = builder.build();

        // Generate a random query.
        let mut query_coords: [[[f32; 3]; 4]; 5] = [[[0.0; 3]; 4]; 5];
        for r in 0..5 {
            for at in 0..4 {
                query_coords[r][at][0] = rng.gen_range(-20.0_f32..20.0);
                query_coords[r][at][1] = rng.gen_range(-20.0_f32..20.0);
                query_coords[r][at][2] = rng.gen_range(-20.0_f32..20.0);
            }
        }
        let query_raw = Fragment::<5, Raw>::new(query_coords);
        let (query, _) = query_raw.center().expect("Failed to center query");

        // Test at multiple epsilon values.
        for epsilon in &[1.0, 2.5, 5.0, 10.0] {
            let results_regular = db.search(&query, *epsilon);
            let results_prefiltered = db.search_prefiltered(&query, *epsilon);

            // Both should return same number of results.
            assert_eq!(
                results_regular.len(),
                results_prefiltered.len(),
                "Result count mismatch at epsilon={}",
                epsilon
            );

            // Results should be in same order and have same RMSD (within 1e-5).
            for (r_reg, r_pre) in results_regular.iter().zip(results_prefiltered.iter()) {
                assert_eq!(r_reg.label, r_pre.label);
                assert!(
                    (r_reg.rmsd - r_pre.rmsd).abs() < 1e-5,
                    "RMSD mismatch: {} vs {} at epsilon={}",
                    r_reg.rmsd,
                    r_pre.rmsd,
                    epsilon
                );
            }
        }
    }
}
