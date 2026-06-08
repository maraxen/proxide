use crate::cache::{build_bb_atoms, cache_residue_impl, ResidueCache};
use crate::contact_list::ContactList;
use crate::coords::{ProteinBackbone, ResidueIndex};
use crate::error::ConFindError;
use crate::grid::ProximityGrid;
use crate::parallel::{contact_degree_raw, run_phases_b_c};
use crate::params::{CLASH_DIST, DCUT, HI_COLL_PROB_CUT, LO_COLL_PROB_CUT};
use dashmap::DashMap;
use proxide_core::processing::residues::ResidueId;
use proxide_rotlib::{RotamerId, RotamerLibrary};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// ConFind session for a single protein structure.
///
/// Holds the rotamer library, backbone geometry, spatial grids, and all
/// per-residue caches accumulated during `contacts()` calls.
pub struct ConFind {
    /// Rotamer library used for sidechain enumeration.
    pub rotlib: Arc<RotamerLibrary>,
    /// If `true`, treat any residue without a complete set of backbone atoms
    /// as an error rather than skipping it silently.
    pub strict: bool,
    /// Full protein backbone extracted from the processed structure.
    pub backbone: Arc<ProteinBackbone>,

    bb_grid: Arc<ProximityGrid<usize>>,   // tag = index into bb_atoms
    ca_grid: Arc<ProximityGrid<ResidueIndex>>,
    bb_atoms: Arc<Vec<[f64; 3]>>,
    bb_atom_res: Arc<Vec<ResidueIndex>>,  // parallel to bb_atoms

    cache: DashMap<ResidueIndex, Arc<ResidueCache>>,
    coll_prob: DashMap<ResidueIndex, HashMap<Arc<RotamerId>, f64>>,
    freedom: DashMap<ResidueIndex, f64>,
}

impl ConFind {
    pub fn new(
        rotlib: Arc<RotamerLibrary>,
        backbone: Arc<ProteinBackbone>,
        strict: bool,
    ) -> Self {
        // Build backbone atom flat vec and atom→residue mapping.
        let (bb_atoms_vec, bb_atom_res_vec) = build_bb_atoms(&backbone);
        // Grid over all backbone atoms; tag = atom index.
        let bb_atom_tags: Vec<usize> = (0..bb_atoms_vec.len()).collect();
        let bb_grid = Arc::new(ProximityGrid::build(&bb_atoms_vec, bb_atom_tags, CLASH_DIST));

        // CA grid for neighbor queries; tag = ResidueIndex.
        let ca_points: Vec<[f64; 3]> = backbone
            .bb
            .iter()
            .filter_map(|rb| rb.ca)
            .collect();
        let ca_tags: Vec<ResidueIndex> = backbone
            .bb
            .iter()
            .enumerate()
            .filter_map(|(i, rb)| rb.ca.map(|_| ResidueIndex(i as u32)))
            .collect();
        let ca_grid = Arc::new(ProximityGrid::build(&ca_points, ca_tags, DCUT));

        Self {
            rotlib,
            strict,
            backbone,
            bb_grid,
            ca_grid,
            bb_atoms: Arc::new(bb_atoms_vec),
            bb_atom_res: Arc::new(bb_atom_res_vec),
            cache: DashMap::new(),
            coll_prob: DashMap::new(),
            freedom: DashMap::new(),
        }
    }

    pub fn n_residues(&self) -> usize {
        self.backbone.bb.len()
    }

    /// Phase A: cache one residue (idempotent, rayon-safe).
    pub fn cache_residue(&self, ri: ResidueIndex) -> Result<(), ConFindError> {
        if self.cache.contains_key(&ri) {
            return Ok(());
        }
        let entry = cache_residue_impl(
            ri,
            &self.backbone,
            &self.rotlib,
            &self.bb_grid,
            &self.bb_atom_res,
        )?;
        self.cache.insert(ri, Arc::new(entry));
        Ok(())
    }

    /// Phase A: cache all residues in parallel.
    pub fn cache_all(&self) -> Result<(), ConFindError> {
        (0..self.backbone.bb.len() as u32)
            .into_par_iter()
            .map(ResidueIndex)
            .try_for_each(|ri| self.cache_residue(ri))
    }

    /// CA–CA neighbors of ri within DCUT.
    pub fn neighbors(&self, ri: ResidueIndex) -> Vec<ResidueIndex> {
        let ca = match self.backbone.bb[ri.0 as usize].ca {
            Some(c) => c,
            None => return Vec::new(),
        };
        self.ca_grid
            .points_within(ca, 0.0, DCUT)
            .into_iter()
            .filter(|&r| r != ri)
            .collect()
    }

    /// Compute CD for one pair; emit ClashTuples for Phase B2 collProb accumulation.
    pub fn contact_degree_with_clashes(
        &self,
        res_a: ResidueIndex,
        res_b: ResidueIndex,
        aa_allowed_a: Option<&[&str]>,
        aa_allowed_b: Option<&[&str]>,
    ) -> Result<(f64, Vec<crate::parallel::ClashTuple>), ConFindError> {
        let ca = self.cache.get(&res_a).ok_or(ConFindError::NotCached(res_a))?;
        let cb = self.cache.get(&res_b).ok_or(ConFindError::NotCached(res_b))?;
        contact_degree_raw(res_a, res_b, &ca, &cb, &self.rotlib, aa_allowed_a, aa_allowed_b)
    }

    /// Full contact + freedom pipeline (Phases A–C).
    ///
    /// `residues` is the set of residues for which freedom will be computed.
    /// All residues for which you want `freedom()` to succeed must appear here.
    pub fn contacts(
        &self,
        residues: &[ResidueIndex],
        cd_cut: f64,
    ) -> Result<ContactList, ConFindError> {
        // Ensure all queried residues are cached.
        residues
            .par_iter()
            .try_for_each(|&ri| self.cache_residue(ri))?;

        run_phases_b_c(
            residues,
            cd_cut,
            &self.cache,
            &self.rotlib,
            |ri| self.neighbors(ri),
            &self.coll_prob,
            &self.freedom,
            LO_COLL_PROB_CUT,
            HI_COLL_PROB_CUT,
        )
    }

    /// Interference: pairs where EITHER endpoint is in `residues` and value >= incut.
    /// Directional (resA sidechain → resB backbone). Call after cache_all() or contacts().
    pub fn interference(
        &self,
        residues: &[ResidueIndex],
        in_cut: f64,
    ) -> Result<ContactList, ConFindError> {
        let of_interest: HashSet<ResidueIndex> = residues.iter().copied().collect();
        let mut contact = ContactList::default();

        for ri in residues {
            let cache = self.cache.get(ri).ok_or(ConFindError::NotCached(*ri))?;
            for (&res_b, aa_map) in &cache.interference {
                if !of_interest.contains(&res_b) {
                    continue;
                }
                // Sum over all AAs (unconstrained interference).
                let all_aa: HashSet<&str> = crate::params::AA_NAMES.iter().copied().collect();
                let val: f64 = aa_map
                    .iter()
                    .filter(|(aa, _)| all_aa.contains(aa.as_str()))
                    .map(|(_, v)| v)
                    .sum();
                let denom = crate::cache::weight_of_available_amino_acids(&all_aa);
                let norm_val = if denom == 0.0 { 0.0 } else { val / denom };
                if norm_val >= in_cut {
                    contact.pairs.push((*ri, res_b));
                    contact.degrees.push(norm_val);
                }
            }
        }
        Ok(contact)
    }

    /// BB–BB minimum atom-pair distance, skipping same-chain adjacents.
    pub fn bb_interaction(
        &self,
        residues: &[ResidueIndex],
        dcut_bb: f64,
        ignore_flanking: usize,
    ) -> Result<ContactList, ConFindError> {
        let bb_atoms = self.bb_atoms.as_ref();

        // Build per-residue atom index ranges in bb_atoms.
        // bb_atoms was built in order of backbone residues.
        // We need atom indices per residue.
        let n_res = self.backbone.bb.len();
        let mut res_atom_ranges: Vec<(usize, usize)> = vec![(0, 0); n_res];
        let mut cur = 0;
        for (i, rb) in self.backbone.bb.iter().enumerate() {
            let start = cur;
            for opt in [rb.n, rb.ca, rb.c, rb.o] {
                if opt.is_some() {
                    cur += 1;
                }
            }
            res_atom_ranges[i] = (start, cur);
        }

        let mut contact = ContactList::default();

        for &ri in residues {
            let (start_a, end_a) = res_atom_ranges[ri.0 as usize];
            for &rj in residues {
                if rj <= ri {
                    continue;
                }
                // Skip same-chain adjacent residues.
                if self.backbone.chain_map[ri.0 as usize]
                    == self.backbone.chain_map[rj.0 as usize]
                {
                    let diff = (rj.0 as usize).saturating_sub(ri.0 as usize);
                    if diff <= ignore_flanking {
                        continue;
                    }
                }
                let (start_b, end_b) = res_atom_ranges[rj.0 as usize];
                let mut min_d = f64::MAX;
                for ai in start_a..end_a {
                    for bi in start_b..end_b {
                        let a = &bb_atoms[ai];
                        let b = &bb_atoms[bi];
                        let d = ((a[0] - b[0]).powi(2)
                            + (a[1] - b[1]).powi(2)
                            + (a[2] - b[2]).powi(2))
                        .sqrt();
                        if d < min_d {
                            min_d = d;
                        }
                    }
                }
                if min_d <= dcut_bb {
                    contact.pairs.push((ri, rj));
                    contact.degrees.push(min_d);
                }
            }
        }
        Ok(contact)
    }

    /// Backbone crowdedness = fraction_pruned.
    pub fn crowdedness(&self, ri: ResidueIndex) -> Result<f64, ConFindError> {
        self.cache
            .get(&ri)
            .map(|c| c.fraction_pruned)
            .ok_or(ConFindError::NotCached(ri))
    }

    /// Rotamer freedom for `ri` (must have been in a `contacts()` query).
    pub fn freedom(&self, ri: ResidueIndex) -> Result<f64, ConFindError> {
        self.freedom
            .get(&ri)
            .map(|f| *f)
            .ok_or(ConFindError::FreedomNotComputed(ri))
    }

    /// Map a `ResidueIndex` back to its chain/sequence `ResidueId` for output.
    pub fn residue_id(&self, ri: ResidueIndex) -> &ResidueId {
        &self.backbone.ids[ri.0 as usize]
    }

    /// Residues whose backbone atoms permanently clash with any ALA rotamer of `res`.
    ///
    /// Returns an empty vec if `res` was not cached yet. Call `cache_residue(res)` first.
    pub fn constrained_contacts(&self, res: ResidueIndex) -> Vec<ResidueIndex> {
        let cache = match self.cache.get(&res) {
            Some(c) => c,
            None => return Vec::new(),
        };
        let mut result: Vec<ResidueIndex> = cache
            .permanent_contacts
            .iter()
            .map(|&atom_idx| self.bb_atom_res[atom_idx])
            .collect();
        result.sort_unstable();
        result.dedup();
        result
    }
}
