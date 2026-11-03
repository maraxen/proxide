//! Phase B: Parallel pairwise contact degree computation.
//!
//! Citations:
//! - Zheng, F., Zhang, J., & Grigoryan, G. (2015). "Tertiary Structural Propensities Reveal
//!   Fundamental Sequence/Structure Relationships in Proteins." *Structure*, 23(5), 961–971.
//! - Zheng, W., & Grigoryan, G. (2017). "Sequence statistics of tertiary structural motifs
//!   reflect protein stability." *PLoS ONE*, 12(5), e0178272.

use crate::cache::{bboxes_overlap, ResidueCache};
use crate::contact_list::ContactList;
use crate::coords::{ProteinBackbone, ResidueIndex};
use crate::error::ConFindError;
use crate::freedom::compute_freedom;
use crate::params::{CONT_DIST, DCUT, HI_COLL_PROB_CUT, LO_COLL_PROB_CUT};
use dashmap::DashMap;
use orx_parallel::{IntoParIter, ParIter};
use std::sync::Arc;

/// Compute pairwise contact degree and collision probabilities between two cached residues.
///
/// For pairs satisfying d(Cα_i, Cα_j) <= 25.0 Å, evaluates:
/// c(i, j) = (1 / (Omega_i * Omega_j)) * sum P(a) P(b) P(r_a) P(r_b) w(r_a, r_b)
///
/// Short-circuits analytically to 0.0 if Omega_i == 0.0 or Omega_j == 0.0 (spec §2.1).
pub fn compute_pair_contact(
    cache_i: &ResidueCache,
    cache_j: &ResidueCache,
) -> (f64, Vec<f64>, Vec<f64>) {
    let n_i = cache_i.surviving_rotamers.len();
    let n_j = cache_j.surviving_rotamers.len();

    let mut p_coll_i = vec![0.0f64; n_i];
    let mut p_coll_j = vec![0.0f64; n_j];

    // Short-circuit if either residue has a zero partition function (crowdedness = 1.0)
    if cache_i.omega_i <= 0.0 || cache_j.omega_i <= 0.0 || n_i == 0 || n_j == 0 {
        return (0.0, p_coll_i, p_coll_j);
    }

    let cont_dist_sq = CONT_DIST * CONT_DIST;
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
            if !bboxes_overlap(&ra.bbox, &rb.bbox, CONT_DIST) {
                continue;
            }

            // Check if any atom pair is within contact threshold
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

                // Accumulate Phase C collision probabilities dividing P(b) by 100.0 (spec §2.5)
                p_coll_i[ra_idx] += rb.weight / 100.0;
                p_coll_j[rb_idx] += ra.weight / 100.0;
            }
        }
    }

    let degree = contact_sum / (cache_i.omega_i * cache_j.omega_i);
    (degree, p_coll_i, p_coll_j)
}

/// Execute Phase B and Phase C across a query set of residues.
pub fn run_phases_b_c(
    residues: &[ResidueIndex],
    cd_cut: f64,
    backbone: &ProteinBackbone,
    cache: &DashMap<ResidueIndex, Arc<ResidueCache>>,
    contacts_cache: &DashMap<(ResidueIndex, ResidueIndex), f64>,
    freedom_cache: &DashMap<ResidueIndex, f64>,
) -> Result<ContactList, ConFindError> {
    // 1. Validate that all query residues are cached
    for &ri in residues {
        if !cache.contains_key(&ri) {
            return Err(ConFindError::NotCached(ri));
        }
    }

    // 2. Identify candidate pairs satisfying Cα–Cα distance <= DCUT (25.0 Å)
    let dcut_sq = DCUT * DCUT;
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
    let collision_map: DashMap<ResidueIndex, Vec<f64>> = DashMap::new();

    // Initialize collision vectors for each residue
    for &ri in residues {
        let c_len = cache.get(&ri).map(|c| c.surviving_rotamers.len()).unwrap_or(0);
        collision_map.insert(ri, vec![0.0f64; c_len]);
    }

    struct PairResult {
        pair: (ResidueIndex, ResidueIndex),
        degree: f64,
        p_coll_i: Vec<f64>,
        p_coll_j: Vec<f64>,
    }

    let results: Vec<PairResult> = candidate_pairs
        .into_par()
        .map(|(ri, rj)| {
            let cache_i = cache.get(&ri).unwrap();
            let cache_j = cache.get(&rj).unwrap();
            let (deg, p_i, p_j) = compute_pair_contact(&cache_i, &cache_j);
            PairResult {
                pair: (ri, rj),
                degree: deg,
                p_coll_i: p_i,
                p_coll_j: p_j,
            }
        })
        .collect();

    // 4. Collect results and accumulate collision probabilities
    let mut contact_list = ContactList::default();

    for res in results {
        let (ri, rj) = res.pair;
        let deg = res.degree;

        // Cache symmetric pairwise contact degree
        contacts_cache.insert((ri, rj), deg);
        contacts_cache.insert((rj, ri), deg);

        if deg > cd_cut {
            contact_list.pairs.push((ri, rj));
            contact_list.degrees.push(deg);
        }

        // Accumulate collisions for residue i
        if let Some(mut coll_i) = collision_map.get_mut(&ri) {
            for (k, &val) in res.p_coll_i.iter().enumerate() {
                if k < coll_i.len() {
                    coll_i[k] += val;
                }
            }
        }

        // Accumulate collisions for residue j
        if let Some(mut coll_j) = collision_map.get_mut(&rj) {
            for (k, &val) in res.p_coll_j.iter().enumerate() {
                if k < coll_j.len() {
                    coll_j[k] += val;
                }
            }
        }
    }

    // 5. Phase C: Compute residual conformational freedom F(i) for all query residues
    for &ri in residues {
        let rc = cache.get(&ri).unwrap();
        let coll_vec = collision_map.get(&ri).map(|v| v.clone()).unwrap_or_default();
        let non_zero_collisions: Vec<f64> = coll_vec.into_iter().filter(|&c| c > 0.0).collect();

        let freedom_val = compute_freedom(
            &non_zero_collisions,
            rc.n_surviving,
            rc.n_library,
            LO_COLL_PROB_CUT,
            HI_COLL_PROB_CUT,
        );

        freedom_cache.insert(ri, freedom_val);
    }

    Ok(contact_list)
}
