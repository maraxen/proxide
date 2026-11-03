//! Phase B: Parallel pairwise contact degree computation.
//!
//! Citations:
//! - Zheng, F., Zhang, J., & Grigoryan, G. (2015). "Tertiary Structural Propensities Reveal
//!   Fundamental Sequence/Structure Relationships in Proteins." *Structure*, 23(5), 961–971.
//! - Zheng, W., & Grigoryan, G. (2017). "Sequence statistics of tertiary structural motifs
//!   reflect protein stability." *PLoS ONE*, 12(5), e0178272.

use crate::cache::{bboxes_overlap, ResidueCache};
use crate::config::Cutoffs;
use crate::contact_list::ContactList;
use crate::coords::{ProteinBackbone, ResidueIndex};
use crate::error::ConFindError;
use dashmap::DashMap;
use orx_parallel::{IntoParIter, ParIter};
use std::num::NonZeroUsize;
use std::sync::Arc;

/// Apply a configured worker limit. `None` keeps the orx-parallel default.
pub(crate) fn apply_num_threads<I>(iter: I, threads: Option<NonZeroUsize>) -> I
where
    I: ParIter,
{
    match threads {
        Some(n) => iter.num_threads(n.get()),
        None => iter,
    }
}

/// Compute pairwise contact degree and collision masses between two cached residues.
///
/// Evaluates
/// `c(i, j) = (1 / (Omega_i * Omega_j)) * sum P(a) P(b) P(r_a) P(r_b) near(S_cont(r_a), S_cont(r_b); d_cont)`
/// over surviving rotamers, and returns the per-rotamer collision masses
/// `P_coll` contributions (`P(b)/100` fractional probability) for both residues.
///
/// Short-circuits analytically to 0.0 if Omega_i == 0.0 or Omega_j == 0.0 (spec §2.1).
/// The distance test is inclusive and symmetric in its arguments.
pub fn compute_pair_contact(
    cache_i: &ResidueCache,
    cache_j: &ResidueCache,
    d_cont: f64,
) -> (f64, Vec<f64>, Vec<f64>) {
    let n_i = cache_i.surviving_rotamers.len();
    let n_j = cache_j.surviving_rotamers.len();

    let mut p_coll_i = vec![0.0f64; n_i];
    let mut p_coll_j = vec![0.0f64; n_j];

    // Short-circuit if either residue has a zero partition function (crowdedness = 1.0)
    if cache_i.omega_i <= 0.0 || cache_j.omega_i <= 0.0 || n_i == 0 || n_j == 0 {
        return (0.0, p_coll_i, p_coll_j);
    }

    let cont_dist_sq = d_cont * d_cont;
    let mut contact_sum = 0.0f64;

    for (ra_idx, ra) in cache_i.surviving_rotamers.iter().enumerate() {
        if ra.atoms.is_empty() {
            continue;
        }

        for (rb_idx, rb) in cache_j.surviving_rotamers.iter().enumerate() {
            if rb.atoms.is_empty() {
                continue;
            }

            // Bounding box spatial overlap test with contact distance margin
            if !bboxes_overlap(&ra.bbox, &rb.bbox, d_cont) {
                continue;
            }

            // A rotamer pair contributes at most once however many atom pairs touch.
            let mut in_contact = false;
            'atom_check: for &pt_a in &ra.atoms {
                for &pt_b in &rb.atoms {
                    let dx = pt_a[0] - pt_b[0];
                    let dy = pt_a[1] - pt_b[1];
                    let dz = pt_a[2] - pt_b[2];
                    let d2 = dx * dx + dy * dy + dz * dz;
                    if d2 <= cont_dist_sq {
                        in_contact = true;
                        break 'atom_check;
                    }
                }
            }

            if in_contact {
                let pair_weight = ra.weight * rb.weight;
                contact_sum += pair_weight;

                // Collision mass: P(b) is in percent, divided by 100 (spec §2.5).
                p_coll_i[ra_idx] += rb.weight / 100.0;
                p_coll_j[rb_idx] += ra.weight / 100.0;
            }
        }
    }

    let degree = contact_sum / (cache_i.omega_i * cache_j.omega_i);
    (degree, p_coll_i, p_coll_j)
}

/// Collision mass `P_coll(r)` of every surviving rotamer of `cache_i`, accumulated over
/// `neighbours` in the given order (spec §2.5, §3.8.2). The result is an unnormalised
/// mass and may exceed 1.
pub fn collision_masses(
    cache_i: &ResidueCache,
    neighbours: &[Arc<ResidueCache>],
    d_cont: f64,
) -> Vec<f64> {
    let mut total = vec![0.0f64; cache_i.surviving_rotamers.len()];
    for nb in neighbours {
        let (_, p_i, _) = compute_pair_contact(cache_i, nb, d_cont);
        for (t, v) in total.iter_mut().zip(p_i) {
            *t += v;
        }
    }
    total
}

/// Execute Phase B across a query set of residues.
///
/// Candidate pairs satisfy `|Cα_i − Cα_j| <= d_cut` (inclusive). Results are ordered by
/// the query order and independent of the thread count.
///
/// `cutoffs` supplies `d_cut` and `d_cont`. `threads` is the worker limit from
/// [`crate::ConFindConfig`]: `None` is the orx-parallel default, `Some(n)` calls
/// `.num_threads(n)` on the contact loop.
pub fn run_phase_b(
    residues: &[ResidueIndex],
    cd_cut: f64,
    backbone: &ProteinBackbone,
    cache: &DashMap<ResidueIndex, Arc<ResidueCache>>,
    contacts_cache: &DashMap<(ResidueIndex, ResidueIndex), f64>,
    cutoffs: &Cutoffs,
    threads: Option<NonZeroUsize>,
) -> Result<ContactList, ConFindError> {
    let d_cut = cutoffs.d_cut;
    let d_cont = cutoffs.d_cont;
    // 1. Validate that all query residues are cached
    for &ri in residues {
        if !cache.contains_key(&ri) {
            return Err(ConFindError::NotCached(ri));
        }
    }

    // 2. Identify candidate pairs satisfying the Cα–Cα cutoff
    let dcut_sq = d_cut * d_cut;
    let mut candidate_pairs = Vec::new();

    let n_query = residues.len();
    for i in 0..n_query {
        let ri = residues[i];
        let ca_i = match backbone.bb[ri.0 as usize].ca {
            Some(ca) => ca,
            None => continue,
        };

        for &rj in &residues[(i + 1)..n_query] {
            let ca_j = match backbone.bb[rj.0 as usize].ca {
                Some(ca) => ca,
                None => continue,
            };

            let dx = ca_i[0] - ca_j[0];
            let dy = ca_i[1] - ca_j[1];
            let dz = ca_i[2] - ca_j[2];
            let d2 = dx * dx + dy * dy + dz * dz;

            if d2 <= dcut_sq {
                candidate_pairs.push((ri, rj));
            }
        }
    }

    // 3. Parallel pairwise contact evaluation
    let results: Vec<((ResidueIndex, ResidueIndex), f64)> =
        apply_num_threads(candidate_pairs.into_par(), threads)
            .map(|(ri, rj)| {
                // Evaluate with the lower index first so that c(i, j) is bitwise equal to
                // c(j, i) whatever the query order (summation order is fixed).
                let (lo, hi) = if ri <= rj { (ri, rj) } else { (rj, ri) };
                let cache_lo = cache.get(&lo).unwrap();
                let cache_hi = cache.get(&hi).unwrap();
                let (deg, _, _) = compute_pair_contact(&cache_lo, &cache_hi, d_cont);
                ((ri, rj), deg)
            })
            .collect();

    // 4. Collect results
    let mut contact_list = ContactList::default();
    for ((ri, rj), deg) in results {
        contacts_cache.insert((ri, rj), deg);
        contacts_cache.insert((rj, ri), deg);

        if deg > cd_cut {
            contact_list.pairs.push((ri, rj));
            contact_list.degrees.push(deg);
        }
    }

    Ok(contact_list)
}
