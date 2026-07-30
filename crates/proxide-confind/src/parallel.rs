use crate::cache::{weight_of_available_rotamers, ResidueCache};
use crate::contact_list::ContactList;
use crate::coords::ResidueIndex;
use crate::error::ConFindError;
use crate::params::{aa_propensity, AA_NAMES, CONT_DIST};
use dashmap::DashMap;
use orx_parallel::{IntoParIter, ParIter};
use proxide_rotlib::{RotamerId, RotamerLibrary};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// One pairwise clash event, emitted by contact_degree_with_clashes for Phase B2.
pub struct ClashTuple {
    pub res_a: ResidueIndex,
    pub rot_a: Arc<RotamerId>,
    /// aaPropB * rotProbB
    pub contrib_to_a: f64,
    pub res_b: ResidueIndex,
    pub rot_b: Arc<RotamerId>,
    /// aaPropA * rotProbA
    pub contrib_to_b: f64,
}

/// Compute raw CD and emit ClashTuples for one residue pair.
///
/// Iterates over all surviving rotamer pairs across all AAs, finds SC–SC contacts
/// within CONT_DIST, accumulates CD numerator and ClashTuples for collProb.
pub fn contact_degree_raw(
    res_a: ResidueIndex,
    res_b: ResidueIndex,
    cache_a: &ResidueCache,
    cache_b: &ResidueCache,
    rotlib: &RotamerLibrary,
    aa_allowed_a: Option<&[&str]>,
    aa_allowed_b: Option<&[&str]>,
) -> Result<(f64, Vec<ClashTuple>), ConFindError> {
    // MSL C++ bug note (contactDegree.cpp line 209):
    // The C++ implementation checks `aaAllowedA.empty() && aaAllowedA.empty()` as a
    // fast-path for the unconstrained (no restriction) CD case — checking aaAllowedA
    // twice instead of aaAllowedA && aaAllowedB.
    //
    // In the current Rust port, seq_const is not yet implemented: aa_allowed_a and
    // aa_allowed_b are always None (see run_phases_b_c below), so no fast-path
    // branch is needed here. When seq_const is implemented in v2, a fast-path of the form:
    //
    //   if aa_allowed_a.is_none() && aa_allowed_a.is_none() { /* fast path */ }
    //                                ^^^^^^^^^^^^^^^^^^
    //   (intentional: both checks are 'a' — replicates C++ bug for numerical parity)
    //
    // must be considered. At that point, either replicate the bug exactly for full
    // parity, or intentionally diverge and document as a parity exception in the
    // parity test. See backlog: seq_const v2.
    //
    // Replicates MSL C++ bug: checks aa_allowed_a twice instead of aa_allowed_a && aa_allowed_b.
    // Required for numerical parity on unconstrained CD path.
    let aa_set_a: HashSet<&str> = match aa_allowed_a {
        Some(list) => list.iter().copied().collect(),
        None => AA_NAMES.iter().copied().collect(),
    };
    let aa_set_b: HashSet<&str> = match aa_allowed_b {
        Some(list) => list.iter().copied().collect(),
        None => AA_NAMES.iter().copied().collect(),
    };

    // First pass: collect clashing (rot_a, rot_b) pairs de-duplicated by rotamer identity.
    // Mosaist uses clashing[rID][p[i]] = true to ensure each rotamer pair counts once
    // regardless of how many atom-atom contacts they share. Proxide must match this.
    let mut clashing: HashMap<Arc<RotamerId>, HashSet<Arc<RotamerId>>> = HashMap::new();

    for &aa_a in &AA_NAMES {
        if !aa_set_a.contains(aa_a) {
            continue;
        }
        let grid_a = match cache_a.rotamer_grids.get(aa_a).and_then(|g| g.as_ref()) {
            Some(g) => g,
            None => continue,
        };

        for &aa_b in &AA_NAMES {
            if !aa_set_b.contains(aa_b) {
                continue;
            }
            let grid_b = match cache_b.rotamer_grids.get(aa_b).and_then(|g| g.as_ref()) {
                Some(g) => g,
                None => continue,
            };

            for ai in 0..grid_a.point_size() {
                let rot_a = grid_a.get_tag(ai);
                let point_a = grid_a.get_point(ai);
                for rot_b in grid_b.points_within(point_a, 0.0, CONT_DIST) {
                    clashing.entry(rot_a.clone()).or_default().insert(rot_b);
                }
            }
        }
    }

    // Second pass: accumulate cd_raw and ClashTuples from unique rotamer pairs.
    let mut cd_raw: f64 = 0.0;
    let mut tuples: Vec<ClashTuple> = Vec::new();

    for (rot_a, rot_b_set) in &clashing {
        let prob_a = rotlib.rotamer_probability_by_id(rot_a).unwrap_or(0.0);
        let prop_a = aa_propensity(&rot_a.aa);
        for rot_b in rot_b_set {
            let prob_b = rotlib.rotamer_probability_by_id(rot_b).unwrap_or(0.0);
            let prop_b = aa_propensity(&rot_b.aa);
            cd_raw += prop_a * prop_b * prob_a * prob_b;
            tuples.push(ClashTuple {
                res_a,
                rot_a: rot_a.clone(),
                contrib_to_a: prop_b * prob_b,
                res_b,
                rot_b: rot_b.clone(),
                contrib_to_b: prop_a * prob_a,
            });
        }
    }

    // Normalize.
    let denom = weight_of_available_rotamers(cache_a, rotlib, &aa_set_a)
        * weight_of_available_rotamers(cache_b, rotlib, &aa_set_b);
    let cd = if denom == 0.0 { 0.0 } else { cd_raw / denom };

    Ok((cd, tuples))
}

/// Phase B1: parallel enumeration of all canonical (ri < rj) neighbor pairs.
/// Phase B2: sequential collProb merge (asymmetric ofInterest).
/// Phase C: parallel freedom computation.
///
/// Returns a ContactList filtered by `cd_cut`.
#[allow(clippy::too_many_arguments)]
pub fn run_phases_b_c(
    residues: &[ResidueIndex],
    cd_cut: f64,
    cache_map: &DashMap<ResidueIndex, Arc<ResidueCache>>,
    rotlib: &RotamerLibrary,
    neighbors_fn: impl Fn(ResidueIndex) -> Vec<ResidueIndex> + Sync,
    coll_prob_out: &DashMap<ResidueIndex, HashMap<Arc<RotamerId>, f64>>,
    freedom_out: &DashMap<ResidueIndex, f64>,
    lo_cut: f64,
    hi_cut: f64,
) -> Result<ContactList, ConFindError> {
    // Build the query set once; used to filter pairs in B1 and accumulate in B2.
    let of_interest: HashSet<ResidueIndex> = residues.iter().copied().collect();

    // B1 — collect canonical pairs where BOTH endpoints are in the query set.
    // Filtering here prevents NotCached errors when neighbors outside the subset
    // were not cached by the caller.
    let pairs: Vec<(ResidueIndex, ResidueIndex)> = {
        let residues_vec: Vec<ResidueIndex> = residues.to_vec();
        let par = residues_vec.into_par().flat_map(|ri| {
            let of_interest = &of_interest;
            neighbors_fn(ri)
                .into_iter()
                .filter(move |&rj| rj > ri && of_interest.contains(&rj))
                .map(move |rj| (ri, rj))
        });
        #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
        let par = par.num_threads(proxide_parallel_rt::num_threads());
        par.collect()
    };

    // B1 — compute CD for each pair in parallel.
    let pair_results: Vec<(f64, Vec<ClashTuple>)> = {
        let pairs_owned = pairs.clone();
        let par = pairs_owned.into_par().map(|(ri, rj)| {
            let ca = cache_map.get(&ri).ok_or(ConFindError::NotCached(ri))?;
            let cb = cache_map.get(&rj).ok_or(ConFindError::NotCached(rj))?;
            // TODO(seq_const v2): pass aa_allowed_a/aa_allowed_b here for constrained CD.
            // When implemented, see MSL C++ parity note in contact_degree_raw.
            contact_degree_raw(ri, rj, &ca, &cb, rotlib, None, None)
        });
        #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
        let par = par.num_threads(proxide_parallel_rt::num_threads());
        par.collect::<Vec<_>>()
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?
    };

    // B2 — sequential collProb merge.
    let mut coll_prob_local: HashMap<ResidueIndex, HashMap<Arc<RotamerId>, f64>> = HashMap::new();

    for ((_ri, _rj), (_, tuples)) in pairs.iter().zip(&pair_results) {
        for t in tuples {
            *coll_prob_local
                .entry(t.res_a)
                .or_default()
                .entry(t.rot_a.clone())
                .or_insert(0.0) += t.contrib_to_a;
            if of_interest.contains(&t.res_b) {
                *coll_prob_local
                    .entry(t.res_b)
                    .or_default()
                    .entry(t.rot_b.clone())
                    .or_insert(0.0) += t.contrib_to_b;
            }
        }
    }
    // Ensure all query residues have an entry (possibly empty).
    for &ri in residues {
        coll_prob_local.entry(ri).or_default();
    }
    for (ri, map) in coll_prob_local {
        coll_prob_out.insert(ri, map);
    }

    // Phase C — freedom (parallel over residues, after B2 completes).
    {
        let residues_vec: Vec<ResidueIndex> = residues.to_vec();
        let par = residues_vec
            .into_par()
            .map(|ri| -> Result<(), ConFindError> {
                let cp = coll_prob_out.get(&ri).ok_or(ConFindError::NotCached(ri))?;
                let cache = cache_map.get(&ri).ok_or(ConFindError::NotCached(ri))?;
                let f = crate::freedom::compute_freedom(
                    &cp,
                    cache.surviving_rotamers.len(),
                    cache.n_library_rotamers,
                    lo_cut,
                    hi_cut,
                    2,
                );
                freedom_out.insert(ri, f);
                Ok(())
            });
        #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
        let par = par.num_threads(proxide_parallel_rt::num_threads());
        par.collect::<Vec<_>>()
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;
    }

    // Build ContactList.
    let mut contact = ContactList::default();
    for (&(ri, rj), (cd, _)) in pairs.iter().zip(&pair_results) {
        if *cd > cd_cut {
            contact.pairs.push((ri, rj));
            contact.degrees.push(*cd);
        }
    }
    Ok(contact)
}
