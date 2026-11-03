//! ConFind pipeline orchestrator.
//!
//! Citations:
//! - Zheng, F., Zhang, J., & Grigoryan, G. (2015). "Tertiary Structural Propensities Reveal
//!   Fundamental Sequence/Structure Relationships in Proteins." *Structure*, 23(5), 961–971.
//! - Zheng, W., & Grigoryan, G. (2017). "Sequence statistics of tertiary structural motifs
//!   reflect protein stability." *PLoS ONE*, 12(5), e0178272.

use crate::cache::{build_backbone_grid_with, cache_residue, ResidueCache};
use crate::config::{ConFindConfig, ConfigError, ResidueScope};
use crate::contact_list::ContactList;
use crate::coords::{ProteinBackbone, ResidueIndex};
use crate::error::ConFindError;
use crate::freedom::compute_freedom;
use crate::grid::ProximityGrid;
use crate::parallel::{collision_masses, run_phase_b};
use crate::params::AA_NAMES;
use crate::precondition::require_preconditions;
use dashmap::DashMap;
use orx_parallel::{IntoParIter, ParIter};
use proxide_core::processing::residues::ResidueId;
use proxide_rotlib::RotamerLibrary;
use std::sync::Arc;

/// ConFind session state: rotamer library, protein backbone, spatial acceleration, and caches.
pub struct ConFind {
    /// Shared rotamer library.
    pub rotlib: Arc<RotamerLibrary>,
    /// Strict precondition enforcement flag.
    pub strict: bool,
    /// Shared protein backbone.
    pub backbone: Arc<ProteinBackbone>,
    /// Validated configuration in force.
    config: ConFindConfig,
    /// Membership mask of the freedom analysis scope (`None` = every residue).
    scope_mask: Option<Vec<bool>>,
    /// Maximum sidechain reach per amino acid, derived from the loaded library (Å).
    r_max: Vec<(&'static str, f64)>,
    /// Spatial acceleration grid for backbone heavy atoms.
    bb_grid: Arc<ProximityGrid<ResidueIndex>>,
    /// Concurrent cache of Phase A residue states.
    cache: DashMap<ResidueIndex, Arc<ResidueCache>>,
    /// Concurrent cache of Phase B pairwise contact degrees.
    contacts_cache: DashMap<(ResidueIndex, ResidueIndex), f64>,
    /// Concurrent cache of residual structural freedom values.
    freedom_cache: DashMap<ResidueIndex, f64>,
}

impl ConFind {
    /// Construct with [`ConFindConfig::default`]. Equivalent to
    /// [`with_config`](Self::with_config) with the default configuration.
    pub fn new(
        rotlib: Arc<RotamerLibrary>,
        backbone: Arc<ProteinBackbone>,
        strict: bool,
    ) -> Result<Self, ConFindError> {
        Self::with_config(rotlib, backbone, strict, ConFindConfig::default())
    }

    /// Alias of [`new`](Self::new), retained for source compatibility.
    pub fn try_new(
        rotlib: Arc<RotamerLibrary>,
        backbone: Arc<ProteinBackbone>,
        strict: bool,
    ) -> Result<Self, ConFindError> {
        Self::new(rotlib, backbone, strict)
    }

    /// Construct with an explicit configuration.
    ///
    /// Validates the configuration (V1, V2, V3, then V6), enforces structural preconditions
    /// when `strict` is set, computes `R_max(a)` for the 18 amino acids from `rotlib`
    /// (spec §3.8.5), checks `d_cut >= 2 * max R_max + d_cont` (V5) and builds the spatial
    /// index. Nothing is silently corrected.
    pub fn with_config(
        rotlib: Arc<RotamerLibrary>,
        backbone: Arc<ProteinBackbone>,
        strict: bool,
        config: ConFindConfig,
    ) -> Result<Self, ConFindError> {
        config.validate()?;

        let n_residues = backbone.bb.len();
        let scope_mask = match &config.scope {
            ResidueScope::All => None,
            ResidueScope::Subset(list) => {
                let mut mask = vec![false; n_residues];
                for ri in list {
                    let slot =
                        mask.get_mut(ri.0 as usize)
                            .ok_or(ConfigError::ScopeOutOfRange {
                                index: ri.0,
                                n_residues,
                            })?;
                    *slot = true;
                }
                Some(mask)
            }
        };

        let mut r_max = Vec::with_capacity(AA_NAMES.len());
        for &aa in AA_NAMES {
            r_max.push((aa, rotlib.max_sidechain_reach(aa)?));
        }
        let largest = r_max.iter().map(|&(_, r)| r).fold(0.0f64, f64::max);
        let required = 2.0 * largest + config.cutoffs.d_cont;
        if config.cutoffs.d_cut < required {
            return Err(ConfigError::DcutTooSmall {
                d_cut: config.cutoffs.d_cut,
                required,
            }
            .into());
        }

        if strict {
            require_preconditions(&backbone)?;
        }

        let bb_grid = Arc::new(build_backbone_grid_with(
            &backbone,
            config.cutoffs.d_clash,
        ));

        Ok(Self {
            rotlib,
            strict,
            backbone,
            config,
            scope_mask,
            r_max,
            bb_grid,
            cache: DashMap::new(),
            contacts_cache: DashMap::new(),
            freedom_cache: DashMap::new(),
        })
    }

    /// The validated configuration in force.
    pub fn config(&self) -> &ConFindConfig {
        &self.config
    }

    /// Library-derived maximum sidechain reach `R_max(aa)` in Å for one of the 18 amino
    /// acids of [`AA_NAMES`], or `None` for any other name.
    pub fn r_max(&self, aa: &str) -> Option<f64> {
        self.r_max.iter().find(|(a, _)| *a == aa).map(|&(_, r)| r)
    }

    /// Minimum valid neighbour cutoff `2 * max_a R_max(a) + d_cont` for the loaded library.
    pub fn d_cut_min(&self) -> f64 {
        2.0 * self.r_max.iter().map(|&(_, r)| r).fold(0.0f64, f64::max)
            + self.config.cutoffs.d_cont
    }

    /// Whether residue `ri` belongs to the configured analysis scope.
    pub fn in_scope(&self, ri: ResidueIndex) -> bool {
        match &self.scope_mask {
            None => (ri.0 as usize) < self.n_residues(),
            Some(mask) => mask.get(ri.0 as usize).copied().unwrap_or(false),
        }
    }

    /// Number of protein residues in backbone.
    pub fn n_residues(&self) -> usize {
        self.backbone.bb.len()
    }

    /// Query original PDB ResidueId (chain, sequence number, insertion code) from internal ResidueIndex.
    pub fn residue_id(&self, ri: ResidueIndex) -> &ResidueId {
        &self.backbone.ids[ri.0 as usize]
    }

    /// Return sorted list of residue indices that form constrained contacts (contact degree > 0.0) with residue `res`.
    pub fn constrained_contacts(&self, res: ResidueIndex) -> Vec<ResidueIndex> {
        let mut result = Vec::new();
        for entry in self.contacts_cache.iter() {
            let (r1, r2) = *entry.key();
            let deg = *entry.value();
            if deg > 0.0 {
                if r1 == res && r2 != res {
                    result.push(r2);
                } else if r2 == res && r1 != res {
                    result.push(r1);
                }
            }
        }
        result.sort_unstable();
        result.dedup();
        result
    }

    fn compute_cache(&self, ri: ResidueIndex) -> Result<ResidueCache, ConFindError> {
        cache_residue(
            ri,
            &self.backbone,
            &self.rotlib,
            &self.bb_grid,
            self.config.cutoffs.d_clash,
            self.config.cb_clash_policy,
        )
    }

    /// The Phase A state of residue `ri` (counts, masses, surviving rotamers, interference),
    /// computed on first use.
    pub fn residue_cache(&self, ri: ResidueIndex) -> Result<Arc<ResidueCache>, ConFindError> {
        self.cached(ri)
    }

    /// Execute Phase A rotamer placement and backbone clash pruning for residue `ri`.
    pub fn cache_residue(&self, ri: ResidueIndex) -> Result<(), ConFindError> {
        self.cached(ri).map(|_| ())
    }

    fn cached(&self, ri: ResidueIndex) -> Result<Arc<ResidueCache>, ConFindError> {
        if let Some(rc) = self.cache.get(&ri) {
            return Ok(Arc::clone(&rc));
        }
        let rc = Arc::new(self.compute_cache(ri)?);
        self.cache.insert(ri, Arc::clone(&rc));
        Ok(rc)
    }

    /// Concurrently execute Phase A caching for all residues in structure.
    pub fn cache_all(&self) -> Result<(), ConFindError> {
        let n = self.n_residues();
        let indices: Vec<ResidueIndex> = (0..n).map(|i| ResidueIndex(i as u32)).collect();

        let cached_items: Vec<Result<(ResidueIndex, ResidueCache), ConFindError>> = indices
            .into_par()
            .map(|ri| {
                let rc = self.compute_cache(ri)?;
                Ok((ri, rc))
            })
            .collect();

        for item in cached_items {
            let (ri, rc) = item?;
            self.cache.insert(ri, Arc::new(rc));
        }

        Ok(())
    }

    /// Query residues within `config.cutoffs.d_cut` Cα–Cα distance (inclusive).
    pub fn neighbors(&self, ri: ResidueIndex) -> Vec<ResidueIndex> {
        let ca_i = match self.backbone.bb[ri.0 as usize].ca {
            Some(ca) => ca,
            None => return Vec::new(),
        };

        let dcut_sq = self.config.cutoffs.d_cut * self.config.cutoffs.d_cut;
        let mut result = Vec::new();

        for (j, rb) in self.backbone.bb.iter().enumerate() {
            let rj = ResidueIndex(j as u32);
            if rj == ri {
                continue;
            }
            if let Some(ca_j) = rb.ca {
                let dx = ca_i[0] - ca_j[0];
                let dy = ca_i[1] - ca_j[1];
                let dz = ca_i[2] - ca_j[2];
                if dx * dx + dy * dy + dz * dz <= dcut_sq {
                    result.push(rj);
                }
            }
        }

        result.sort_unstable();
        result
    }

    /// Return whether `default_bin` fallback was used when caching residue `ri`
    /// due to an undefined (`None`) `phi` or `psi` backbone angle (Debt #1897).
    pub fn used_default_bin(&self, ri: ResidueIndex) -> Result<bool, ConFindError> {
        Ok(self.cached(ri)?.used_default_bin)
    }

    /// Compute pairwise contact degrees across query residue set.
    /// Returns pairs with contact degree > cd_cut.
    pub fn contacts(
        &self,
        residues: &[ResidueIndex],
        cd_cut: f64,
    ) -> Result<ContactList, ConFindError> {
        for &ri in residues {
            self.cache_residue(ri)?;
        }

        run_phase_b(
            residues,
            cd_cut,
            &self.backbone,
            &self.cache,
            &self.contacts_cache,
            self.config.cutoffs.d_cut,
            self.config.cutoffs.d_cont,
        )
    }

    /// Mutual interference `I(i, j)` (spec §2.6, §3.8.3): pairs where BOTH endpoints are
    /// present in the query set and value >= in_cut.
    ///
    /// Interference is directed: the entry `(i, j)` is the prior mass of the unpruned
    /// library rotamers of `i` whose clash atoms hit the backbone of `j`; `(j, i)` is
    /// reported separately. Pairs are listed in query order of `i`, then of `j`.
    ///
    /// All query residues must be cached prior to calling (e.g. via [`cache_residue`](Self::cache_residue),
    /// [`cache_all`](Self::cache_all), or [`contacts`](Self::contacts)); returns
    /// [`ConFindError::NotCached`] if any query residue is not yet cached.
    pub fn interference(
        &self,
        residues: &[ResidueIndex],
        in_cut: f64,
    ) -> Result<ContactList, ConFindError> {
        for &ri in residues {
            if !self.cache.contains_key(&ri) {
                return Err(ConFindError::NotCached(ri));
            }
        }

        let mut contact_list = ContactList::default();

        for &ri in residues {
            let cache_i = self.cache.get(&ri).unwrap();
            for &rj in residues {
                if rj == ri {
                    continue;
                }
                let value = cache_i.interference.get(&rj).copied().unwrap_or(0.0);
                if value >= in_cut {
                    contact_list.pairs.push((ri, rj));
                    contact_list.degrees.push(value);
                }
            }
        }

        Ok(contact_list)
    }

    /// Minimum heavy-atom distance between backbone residues.
    pub fn bb_interaction(
        &self,
        residues: &[ResidueIndex],
        dcut_bb: f64,
        ignore_flanking: usize,
    ) -> Result<ContactList, ConFindError> {
        let mut contact_list = ContactList::default();
        let dcut_sq = dcut_bb * dcut_bb;
        let n = residues.len();

        for i in 0..n {
            let ri = residues[i];
            let atoms_i = self.backbone_atoms(ri);
            if atoms_i.is_empty() {
                continue;
            }

            for &rj in &residues[(i + 1)..n] {
                let seq_diff = (ri.0 as i64 - rj.0 as i64).unsigned_abs() as usize;
                if self.backbone.chain_map[ri.0 as usize] == self.backbone.chain_map[rj.0 as usize]
                    && seq_diff <= ignore_flanking
                {
                    continue;
                }

                let atoms_j = self.backbone_atoms(rj);
                if atoms_j.is_empty() {
                    continue;
                }

                let mut min_d2 = f64::INFINITY;
                for &ai in &atoms_i {
                    for &aj in &atoms_j {
                        let dx = ai[0] - aj[0];
                        let dy = ai[1] - aj[1];
                        let dz = ai[2] - aj[2];
                        let d2 = dx * dx + dy * dy + dz * dz;
                        if d2 < min_d2 {
                            min_d2 = d2;
                        }
                    }
                }

                if min_d2 <= dcut_sq {
                    contact_list.pairs.push((ri, rj));
                    contact_list.degrees.push(min_d2.sqrt());
                }
            }
        }

        Ok(contact_list)
    }

    /// Helper to get backbone heavy atom coordinates for a residue.
    fn backbone_atoms(&self, ri: ResidueIndex) -> Vec<[f64; 3]> {
        let mut atoms = Vec::with_capacity(4);
        let rb = &self.backbone.bb[ri.0 as usize];
        if let Some(n) = rb.n {
            atoms.push(n);
        }
        if let Some(ca) = rb.ca {
            atoms.push(ca);
        }
        if let Some(c) = rb.c {
            atoms.push(c);
        }
        if let Some(o) = rb.o {
            atoms.push(o);
        }
        atoms
    }

    /// Crowdedness `C(i)` per `config.crowdedness_mode` (spec §2.4): the pruned fraction of
    /// prior mass (`ProbabilityWeighted`) or of rotamer count (`PrunedCountFraction`).
    /// Independent of the analysis scope and of the freedom options.
    pub fn crowdedness(&self, ri: ResidueIndex) -> Result<f64, ConFindError> {
        Ok(self.cached(ri)?.crowdedness(self.config.crowdedness_mode))
    }

    /// Freedom `F(i)` per `config.freedom_mode` (spec §2.5), with collision mass accumulated
    /// over the residues of `config.scope` within `d_cut` Cα–Cα of `ri`.
    ///
    /// Computed on demand; does not require a preceding [`contacts`](Self::contacts) call.
    /// Returns [`ConFindError::OutOfScope`] if `ri` is not in the scope. Errors while
    /// caching `ri` or any neighbour propagate.
    pub fn freedom(&self, ri: ResidueIndex) -> Result<f64, ConFindError> {
        if !self.in_scope(ri) {
            return Err(ConFindError::OutOfScope(ri));
        }
        if let Some(f) = self.freedom_cache.get(&ri) {
            return Ok(*f);
        }

        let cache_i = self.cached(ri)?;
        let mut neighbour_caches = Vec::new();
        for rj in self.neighbors(ri) {
            if self.in_scope(rj) {
                neighbour_caches.push(self.cached(rj)?);
            }
        }

        let masses = collision_masses(&cache_i, &neighbour_caches, self.config.cutoffs.d_cont);
        let contested: Vec<f64> = masses.into_iter().filter(|&m| m > 0.0).collect();
        let cut = &self.config.cutoffs;
        let value = compute_freedom(
            self.config.freedom_mode,
            &contested,
            cache_i.n_surviving,
            cache_i.n_library,
            cut.tau_lo,
            cut.tau_hi,
        );
        self.freedom_cache.insert(ri, value);
        Ok(value)
    }
}
