use crate::coords::{ProteinBackbone, ResidueIndex};
use crate::error::ConFindError;
use crate::grid::ProximityGrid;
use crate::params::{aa_propensity, AA_NAMES, CLASH_DIST, CONT_DIST};
use proxide_rotlib::{counts_as_sidechain, RotamerId, RotamerLibrary};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Per-residue data computed in Phase A.
pub struct ResidueCache {
    pub surviving_rotamers: Vec<Arc<RotamerId>>,
    /// Per-aa proximity grid of surviving heavy-SC atoms.
    /// None if no surviving rotamers for that aa.
    pub rotamer_grids: HashMap<String, Option<ProximityGrid<Arc<RotamerId>>>>,
    pub fraction_pruned: f64,
    pub n_library_rotamers: usize,
    /// interference[resB_idx][aa] = accumulated aaP * rotP / 100
    pub interference: HashMap<ResidueIndex, HashMap<String, f64>>,
    /// Backbone atoms permanently clashed by any ALA rotamer of self.
    /// Stores raw backbone atom indices for constrained-contacts lookup.
    pub permanent_contacts: Vec<usize>,
}

/// Sum of aaProp[aa] * rotP over surviving rotamers of `res` in `available_aa`.
pub fn weight_of_available_rotamers(
    cache: &ResidueCache,
    rotlib: &RotamerLibrary,
    available_aa: &HashSet<&str>,
) -> f64 {
    if available_aa.is_empty() {
        return 0.0;
    }
    cache
        .surviving_rotamers
        .iter()
        .filter(|r| available_aa.contains(r.aa.as_str()))
        .map(|r| {
            let p = rotlib.rotamer_probability_by_id(r).unwrap_or(0.0);
            aa_propensity(&r.aa) * p
        })
        .sum()
}

/// Sum of aaProp[aa] for each aa in `available_aa`, divided by 100.
pub fn weight_of_available_amino_acids(available_aa: &HashSet<&str>) -> f64 {
    available_aa.iter().map(|aa| aa_propensity(aa)).sum::<f64>() / 100.0
}

/// Build the backbone atom flat vector and a per-atom ResidueIndex lookup.
pub fn build_bb_atoms(backbone: &ProteinBackbone) -> (Vec<[f64; 3]>, Vec<ResidueIndex>) {
    let mut atoms: Vec<[f64; 3]> = Vec::new();
    let mut atom_res: Vec<ResidueIndex> = Vec::new();

    for (i, rb) in backbone.bb.iter().enumerate() {
        let ri = ResidueIndex(i as u32);
        for opt in [rb.n, rb.ca, rb.c, rb.o].into_iter().flatten() {
            atoms.push(opt);
            atom_res.push(ri);
        }
    }
    (atoms, atom_res)
}

/// Cache one residue: place all rotamers, prune backbone clashes, accumulate interference.
///
/// Thread-safe: reads from shared grids; writes only into `cache_out` and `interf_out`
/// which are per-residue slots.
#[allow(clippy::too_many_arguments)]
pub fn cache_residue_impl(
    ri: ResidueIndex,
    backbone: &ProteinBackbone,
    rotlib: &RotamerLibrary,
    bb_grid: &ProximityGrid<usize>,    // tag = index into bb_atoms
    bb_atom_res: &[ResidueIndex],      // maps bb_atom_idx → owning ResidueIndex
) -> Result<ResidueCache, ConFindError> {
    let rb = &backbone.bb[ri.0 as usize];
    let phi = rb.phi;
    let psi = rb.psi;

    let n = rb.n.ok_or(ConFindError::MissingBackbone(ri, "N"))?;
    let ca = rb.ca.ok_or(ConFindError::MissingBackbone(ri, "CA"))?;
    let c = rb.c.ok_or(ConFindError::MissingBackbone(ri, "C"))?;

    let mut surviving: Vec<Arc<RotamerId>> = Vec::new();
    let mut rotamer_grids: HashMap<String, Option<ProximityGrid<Arc<RotamerId>>>> =
        HashMap::new();
    let mut interference: HashMap<ResidueIndex, HashMap<String, f64>> = HashMap::new();
    let mut permanent_contacts_set: HashSet<usize> = HashSet::new();
    let mut total_rotamers: usize = 0;

    for &aa in &AA_NAMES {
        let nr = match rotlib.num_rotamers(aa, phi, psi) {
            Ok(n) => n,
            Err(_) => {
                rotamer_grids.insert(aa.to_string(), None);
                continue;
            }
        };
        total_rotamers += nr;

        let mut aa_sc_points: Vec<[f64; 3]> = Vec::new();
        let mut aa_sc_tags: Vec<Arc<RotamerId>> = Vec::new();

        for rot_index in 0..nr {
            let placed = rotlib
                .place_rotamer(aa, phi, psi, rot_index, n, ca, c)
                .map_err(ConFindError::RotlibError)?;

            let rot_id = Arc::new(placed.id.clone());
            let rot_prob = rotlib
                .rotamer_probability_by_id(&placed.id)
                .unwrap_or(0.0);

            // Accumulate interference for this rotamer across all its SC heavy atoms.
            // Track which resB values we've already accumulated for this rotamer (first-win).
            let mut seen_res_b: HashSet<ResidueIndex> = HashSet::new();
            let mut prune = false;

            'atom_loop: for atom in &placed.atoms {
                if !counts_as_sidechain(&atom.name, aa) {
                    continue;
                }
                let hits = bb_grid.points_within(atom.xyz, 0.0, CLASH_DIST);
                // Accumulate interference for ALL foreign residues in this atom's clash list
                // (matches Mosaist: interference loop runs over full closeOnes before k-loop break).
                let mut clashed_here = false;
                for bb_atom_idx in hits {
                    let res_b = bb_atom_res[bb_atom_idx];
                    if res_b == ri {
                        continue;
                    }
                    clashed_here = true;
                    prune = true;
                    if aa == "ALA" {
                        permanent_contacts_set.insert(bb_atom_idx);
                    }
                    if !seen_res_b.contains(&res_b) {
                        seen_res_b.insert(res_b);
                        *interference
                            .entry(res_b)
                            .or_default()
                            .entry(aa.to_string())
                            .or_insert(0.0) += aa_propensity(aa) * rot_prob / 100.0;
                    }
                }
                // For non-ALA: exit after the first SC atom that clashes (matches Mosaist).
                if clashed_here && aa != "ALA" {
                    break 'atom_loop;
                }
            }

            if prune {
                continue;
            }

            // Surviving rotamer: collect heavy SC atoms for the proximity grid.
            let tag = rot_id.clone();
            for atom in &placed.atoms {
                if counts_as_sidechain(&atom.name, aa) {
                    aa_sc_points.push(atom.xyz);
                    aa_sc_tags.push(tag.clone());
                }
            }
            surviving.push(rot_id);
        }

        let grid = if aa_sc_points.is_empty() {
            None
        } else {
            Some(ProximityGrid::build(&aa_sc_points, aa_sc_tags, CONT_DIST))
        };
        rotamer_grids.insert(aa.to_string(), grid);
    }

    let n_surviving = surviving.len();
    let fraction_pruned = if total_rotamers > 0 {
        (total_rotamers - n_surviving) as f64 / total_rotamers as f64
    } else {
        0.0
    };

    let permanent_contacts: Vec<usize> = {
        let mut v: Vec<usize> = permanent_contacts_set.into_iter().collect();
        v.sort_unstable();
        v
    };

    Ok(ResidueCache {
        surviving_rotamers: surviving,
        rotamer_grids,
        fraction_pruned,
        n_library_rotamers: total_rotamers,
        interference,
        permanent_contacts,
    })
}
