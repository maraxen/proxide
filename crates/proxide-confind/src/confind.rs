//! ConFind pipeline orchestrator.
//!
//! Citations:
//! - Zheng, F., Zhang, J., & Grigoryan, G. (2015). "Tertiary Structural Propensities Reveal
//!   Fundamental Sequence/Structure Relationships in Proteins." *Structure*, 23(5), 961–971.
//! - Zheng, W., & Grigoryan, G. (2017). "Sequence statistics of tertiary structural motifs
//!   reflect protein stability." *PLoS ONE*, 12(5), e0178272.

use crate::cache::{build_backbone_grid, cache_residue, ResidueCache};
use crate::contact_list::ContactList;
use crate::coords::{ProteinBackbone, ResidueIndex};
use crate::error::ConFindError;
use crate::grid::ProximityGrid;
use crate::parallel::run_phases_b_c;
use crate::params::DCUT;
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
    /// Spatial acceleration grid for backbone heavy atoms.
    bb_grid: Arc<ProximityGrid<ResidueIndex>>,
    /// Concurrent cache of Phase A residue states.
    cache: DashMap<ResidueIndex, Arc<ResidueCache>>,
    /// Concurrent cache of Phase B pairwise contact degrees.
    contacts_cache: DashMap<(ResidueIndex, ResidueIndex), f64>,
    /// Concurrent cache of Phase C residual structural freedom values.
    freedom_cache: DashMap<ResidueIndex, f64>,
}

impl ConFind {
    /// Construct a new ConFind instance and build spatial acceleration structures.
    pub fn new(
        rotlib: Arc<RotamerLibrary>,
        backbone: Arc<ProteinBackbone>,
        strict: bool,
    ) -> Self {
        let bb_grid = Arc::new(build_backbone_grid(&backbone));

        Self {
            rotlib,
            strict,
            backbone,
            bb_grid,
            cache: DashMap::new(),
            contacts_cache: DashMap::new(),
            freedom_cache: DashMap::new(),
        }
    }

    /// Construct a new ConFind instance and validate preconditions if strict is true.
    pub fn try_new(
        rotlib: Arc<RotamerLibrary>,
        backbone: Arc<ProteinBackbone>,
        strict: bool,
    ) -> Result<Self, ConFindError> {
        if strict {
            require_preconditions(&backbone)?;
        }
        Ok(Self::new(rotlib, backbone, strict))
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

    /// Execute Phase A rotamer placement and backbone clash pruning for residue `ri`.
    pub fn cache_residue(&self, ri: ResidueIndex) -> Result<(), ConFindError> {
        if self.cache.contains_key(&ri) {
            return Ok(());
        }

        let rc = cache_residue(ri, &self.backbone, &self.rotlib, &self.bb_grid)?;
        self.cache.insert(ri, Arc::new(rc));
        Ok(())
    }

    /// Concurrently execute Phase A caching for all residues in structure.
    pub fn cache_all(&self) -> Result<(), ConFindError> {
        let n = self.n_residues();
        let indices: Vec<ResidueIndex> = (0..n).map(|i| ResidueIndex(i as u32)).collect();

        let cached_items: Vec<Result<(ResidueIndex, ResidueCache), ConFindError>> = indices
            .into_par()
            .map(|ri| {
                let rc = cache_residue(ri, &self.backbone, &self.rotlib, &self.bb_grid)?;
                Ok((ri, rc))
            })
            .collect();

        for item in cached_items {
            let (ri, rc) = item?;
            self.cache.insert(ri, Arc::new(rc));
        }

        Ok(())
    }

    /// Query residues within 25.0 Å Cα–Cα distance cutoff.
    pub fn neighbors(&self, ri: ResidueIndex) -> Vec<ResidueIndex> {
        let ca_i = match self.backbone.bb[ri.0 as usize].ca {
            Some(ca) => ca,
            None => return Vec::new(),
        };

        let dcut_sq = DCUT * DCUT;
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
        self.cache_residue(ri)?;
        let entry = self
            .cache
            .get(&ri)
            .ok_or(ConFindError::NotCached(ri))?;
        Ok(entry.used_default_bin)
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

        run_phases_b_c(
            residues,
            cd_cut,
            &self.backbone,
            &self.cache,
            &self.contacts_cache,
            &self.freedom_cache,
        )
    }

    /// Compute mutual sidechain-to-backbone interference.
    /// Evaluates pairs where BOTH endpoints are present in `residues` and value >= in_cut.
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
        let n = residues.len();

        for i in 0..n {
            let ri = residues[i];
            let cache_i = self.cache.get(&ri).unwrap();

            for &rj in &residues[(i + 1)..n] {
                let cache_j = self.cache.get(&rj).unwrap();

                let val_ij = cache_i.interference.get(&rj).copied().unwrap_or(0.0);
                let val_ji = cache_j.interference.get(&ri).copied().unwrap_or(0.0);
                let total_interference = val_ij + val_ji;

                if total_interference >= in_cut {
                    contact_list.pairs.push((ri, rj));
                    contact_list.degrees.push(total_interference);
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

    /// Query fraction of rotamers pruned by backbone clash for residue `ri`.
    pub fn crowdedness(&self, ri: ResidueIndex) -> Result<f64, ConFindError> {
        self.cache_residue(ri)?;
        Ok(self.cache.get(&ri).unwrap().crowdedness)
    }

    /// Query residual conformational freedom for residue `ri`.
    /// Residue must have been included in a preceding `contacts()` query set.
    pub fn freedom(&self, ri: ResidueIndex) -> Result<f64, ConFindError> {
        if let Some(f) = self.freedom_cache.get(&ri) {
            Ok(*f)
        } else {
            Err(ConFindError::FreedomNotComputed(ri))
        }
    }
}
