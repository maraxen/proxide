//! Phase A: Sidechain rotamer placement, local backbone clash filtering, and residue caching.
//!
//! Citations:
//! - Zheng, F., Zhang, J., & Grigoryan, G. (2015). "Tertiary Structural Propensities Reveal
//!   Fundamental Sequence/Structure Relationships in Proteins." *Structure*, 23(5), 961–971.
//! - Zheng, W., & Grigoryan, G. (2017). "Sequence statistics of tertiary structural motifs
//!   reflect protein stability." *PLoS ONE*, 12(5), e0178272.

use crate::config::{CbClashPolicy, CrowdednessMode};
use crate::coords::{ProteinBackbone, ResidueBackbone, ResidueIndex};
use crate::error::ConFindError;
use crate::grid::ProximityGrid;
use crate::params::{aa_propensity, AA_NAMES, CLASH_DIST};
use proxide_rotlib::sidechain::counts_as_sidechain;
use proxide_rotlib::RotamerLibrary;
use std::collections::{HashMap, HashSet};

/// Precomputed spatial representation of a surviving rotamer.
#[derive(Clone, Debug)]
pub struct PlacedRotamerData {
    /// Amino acid code (e.g. "ALA", "VAL").
    pub aa: &'static str,
    /// Rotamer index within the Ramachandran bin.
    pub rot_idx: usize,
    /// Empirical background propensity P(a) in percent.
    pub p_a: f64,
    /// Backbone-dependent rotamer probability P(r_a | a, phi, psi).
    pub p_r: f64,
    /// Product weight P(a) * P(r_a).
    pub weight: f64,
    /// World coordinates of contact atoms `S_cont` (sidechain heavy atoms; Cβ only for ALA).
    pub atoms: Vec<[f64; 3]>,
    /// Bounding box [min_x, max_x, min_y, max_y, min_z, max_z] for fast collision pruning.
    pub bbox: [f64; 6],
}

/// Cached Phase A computation for a single protein residue position.
///
/// Holds the counts and masses from which every crowdedness mode is derived, so changing
/// the mode never changes the surviving sets.
#[derive(Clone, Debug)]
pub struct ResidueCache {
    /// Residue index in ProteinBackbone.
    pub index: ResidueIndex,
    /// Residue name in structure.
    pub res_name: String,
    /// Whether `default_bin` was used because `phi` or `psi` was undefined (`None`, Debt #1897).
    pub used_default_bin: bool,
    /// Surviving partition function weight Omega_i = sum P(a) P(r_a) for unpruned rotamers
    /// (`W_surv`, with P(a) in percent).
    pub omega_i: f64,
    /// Library partition function weight Omega_lib = sum P(a) P(r_a) across all rotamers
    /// (`W_lib`, with P(a) in percent).
    pub omega_lib: f64,
    /// Total number of rotamers in the library for this position (`N_lib`).
    pub n_library: usize,
    /// Number of rotamers that survived backbone clash pruning (`N_surv`).
    pub n_surviving: usize,
    /// Surviving rotamers with placed heavy-atom coordinates.
    pub surviving_rotamers: Vec<PlacedRotamerData>,
    /// Directed interference I(i, j) of this residue's library rotamers with the backbone of
    /// residue j (prior mass, with P(a) divided by 100).
    pub interference: HashMap<ResidueIndex, f64>,
}

impl ResidueCache {
    /// Crowdedness C(i) under `mode` (spec §2.4). A zero denominator gives 0.
    pub fn crowdedness(&self, mode: CrowdednessMode) -> f64 {
        let (surv, lib) = match mode {
            CrowdednessMode::ProbabilityWeighted => (self.omega_i, self.omega_lib),
            CrowdednessMode::PrunedCountFraction => {
                (self.n_surviving as f64, self.n_library as f64)
            }
        };
        if lib > 0.0 {
            (1.0 - surv / lib).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
}

/// A library rotamer placed in the lab frame, before backbone pruning.
#[derive(Clone, Debug)]
pub struct RotamerCandidate {
    /// Amino acid code.
    pub aa: &'static str,
    /// Rotamer index within the bin.
    pub rot_idx: usize,
    /// Background propensity P(a) in percent.
    pub p_a: f64,
    /// Rotamer probability P(r | a, phi, psi).
    pub p_r: f64,
    /// Contact atoms `S_cont` (per `counts_as_sidechain`).
    pub cont_atoms: Vec<[f64; 3]>,
    /// Cβ position if the rotamer has one (added to the clash set per [`CbClashPolicy`]).
    pub cb: Option<[f64; 3]>,
}

/// Extract all present backbone heavy atoms (N, CA, C, O) and their corresponding residue indices.
pub fn build_bb_atoms(bb: &ProteinBackbone) -> (Vec<[f64; 3]>, Vec<ResidueIndex>) {
    let mut points = Vec::new();
    let mut tags = Vec::new();

    for (i, rb) in bb.bb.iter().enumerate() {
        let ri = ResidueIndex(i as u32);
        if let Some(n) = rb.n {
            points.push(n);
            tags.push(ri);
        }
        if let Some(ca) = rb.ca {
            points.push(ca);
            tags.push(ri);
        }
        if let Some(c) = rb.c {
            points.push(c);
            tags.push(ri);
        }
        if let Some(o) = rb.o {
            points.push(o);
            tags.push(ri);
        }
    }

    (points, tags)
}

/// Build spatial hash grid indexing all backbone heavy atoms (N, CA, C, O) across the protein.
pub fn build_backbone_grid(bb: &ProteinBackbone) -> ProximityGrid<ResidueIndex> {
    build_backbone_grid_with(bb, CLASH_DIST)
}

/// Build the backbone grid for clash distance `d_clash`.
///
/// The cell size is only a performance parameter (queries are exact for any radius); it is
/// floored at the default clash distance so that a very small `d_clash` cannot make the
/// grid arbitrarily fine.
pub fn build_backbone_grid_with(bb: &ProteinBackbone, d_clash: f64) -> ProximityGrid<ResidueIndex> {
    let (points, tags) = build_bb_atoms(bb);
    ProximityGrid::build(&points, tags, d_clash.max(CLASH_DIST))
}

/// Phase A kernel: backbone-clash pruning, crowdedness masses and interference for the
/// library rotamers of one residue (spec §2.4, §2.6, §3.8.3).
///
/// A rotamer is pruned if any atom of `S_clash` lies within `d_clash` (inclusive) of any
/// backbone atom of a residue other than `ri`. A pruned rotamer contributes its full prior
/// mass `P(a)/100 * P(r)` to the interference of every residue whose backbone it hits,
/// so the result does not depend on atom order within a rotamer.
pub fn evaluate_candidates(
    ri: ResidueIndex,
    res_name: &str,
    used_default_bin: bool,
    candidates: Vec<RotamerCandidate>,
    bb_grid: &ProximityGrid<ResidueIndex>,
    d_clash: f64,
    policy: CbClashPolicy,
) -> ResidueCache {
    let mut n_library = 0usize;
    let mut n_surviving = 0usize;
    let mut omega_lib = 0.0f64;
    let mut omega_i = 0.0f64;
    let mut surviving_rotamers = Vec::new();
    let mut interference: HashMap<ResidueIndex, f64> = HashMap::new();

    for cand in candidates {
        n_library += 1;
        let rot_weight = cand.p_a * cand.p_r;
        omega_lib += rot_weight;

        let extra = match policy {
            CbClashPolicy::AllResidues => cand.cb,
            CbClashPolicy::AlaOnly => None,
        };
        let mut clashing: HashSet<ResidueIndex> = HashSet::new();
        for pt in cand.cont_atoms.iter().copied().chain(extra) {
            for neighbor in bb_grid.points_within(pt, 0.0, d_clash) {
                if neighbor != ri {
                    clashing.insert(neighbor);
                }
            }
        }

        if clashing.is_empty() {
            omega_i += rot_weight;
            n_surviving += 1;
            let bbox = compute_bbox(&cand.cont_atoms);
            surviving_rotamers.push(PlacedRotamerData {
                aa: cand.aa,
                rot_idx: cand.rot_idx,
                p_a: cand.p_a,
                p_r: cand.p_r,
                weight: rot_weight,
                atoms: cand.cont_atoms,
                bbox,
            });
        } else {
            for neighbor in clashing {
                *interference.entry(neighbor).or_insert(0.0) += rot_weight / 100.0;
            }
        }
    }

    ResidueCache {
        index: ri,
        res_name: res_name.to_string(),
        used_default_bin,
        omega_i,
        omega_lib,
        n_library,
        n_surviving,
        surviving_rotamers,
        interference,
    }
}

/// Execute Phase A rotamer placement, backbone clash pruning, and crowdedness masses.
pub fn cache_residue(
    ri: ResidueIndex,
    bb: &ProteinBackbone,
    rotlib: &RotamerLibrary,
    bb_grid: &ProximityGrid<ResidueIndex>,
    d_clash: f64,
    policy: CbClashPolicy,
) -> Result<ResidueCache, ConFindError> {
    let rb: &ResidueBackbone = &bb.bb[ri.0 as usize];

    let n = rb.n.ok_or(ConFindError::MissingBackbone(ri, "N"))?;
    let ca = rb.ca.ok_or(ConFindError::MissingBackbone(ri, "CA"))?;
    let c = rb.c.ok_or(ConFindError::MissingBackbone(ri, "C"))?;

    // Glycine: no sidechain, empty library, crowdedness defined as 0.
    if rb.res_name == "GLY" {
        return Ok(evaluate_candidates(
            ri,
            &rb.res_name,
            rb.phi.is_none() || rb.psi.is_none(),
            Vec::new(),
            bb_grid,
            d_clash,
            policy,
        ));
    }

    let phi = rb.phi;
    let psi = rb.psi;
    let cis_proline = rb.is_cis_peptide;
    let mut used_default_bin = phi.is_none() || psi.is_none();
    let mut candidates = Vec::new();

    for &aa in AA_NAMES {
        let p_a = aa_propensity(aa);
        if p_a <= 0.0 {
            continue;
        }

        let (_, fb) = rotlib.backbone_bin_with_fallback(aa, phi, psi, cis_proline)?;
        used_default_bin |= fb;
        let nr = rotlib.num_rotamers(aa, phi, psi, cis_proline)?;

        for rot_idx in 0..nr {
            let p_r = rotlib.rotamer_probability(aa, phi, psi, cis_proline, rot_idx)?;
            let placed = rotlib.place_rotamer(aa, phi, psi, rot_idx, cis_proline, n, ca, c)?;
            used_default_bin |= placed.used_default_bin;

            let mut cont_atoms = Vec::new();
            let mut cb = None;
            for atom in placed.atoms {
                if atom.name == "CB" {
                    cb = Some(atom.xyz);
                }
                if counts_as_sidechain(&atom.name, aa) {
                    cont_atoms.push(atom.xyz);
                }
            }
            candidates.push(RotamerCandidate {
                aa,
                rot_idx,
                p_a,
                p_r,
                cont_atoms,
                cb,
            });
        }
    }

    Ok(evaluate_candidates(
        ri,
        &rb.res_name,
        used_default_bin,
        candidates,
        bb_grid,
        d_clash,
        policy,
    ))
}

/// Compute axis-aligned bounding box [min_x, max_x, min_y, max_y, min_z, max_z]
pub fn compute_bbox(atoms: &[[f64; 3]]) -> [f64; 6] {
    if atoms.is_empty() {
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    }
    let mut min_x = atoms[0][0];
    let mut max_x = atoms[0][0];
    let mut min_y = atoms[0][1];
    let mut max_y = atoms[0][1];
    let mut min_z = atoms[0][2];
    let mut max_z = atoms[0][2];

    for a in &atoms[1..] {
        if a[0] < min_x {
            min_x = a[0];
        }
        if a[0] > max_x {
            max_x = a[0];
        }
        if a[1] < min_y {
            min_y = a[1];
        }
        if a[1] > max_y {
            max_y = a[1];
        }
        if a[2] < min_z {
            min_z = a[2];
        }
        if a[2] > max_z {
            max_z = a[2];
        }
    }

    [min_x, max_x, min_y, max_y, min_z, max_z]
}

/// Check if two bounding boxes expanded by margin overlap
#[inline]
pub fn bboxes_overlap(a: &[f64; 6], b: &[f64; 6], margin: f64) -> bool {
    !(a[0] > b[1] + margin
        || a[1] < b[0] - margin
        || a[2] > b[3] + margin
        || a[3] < b[2] - margin
        || a[4] > b[5] + margin
        || a[5] < b[4] - margin)
}
