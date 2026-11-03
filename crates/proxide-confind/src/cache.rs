//! Phase A: Sidechain rotamer placement, local backbone clash filtering, and residue caching.
//!
//! Citations:
//! - Zheng, F., Zhang, J., & Grigoryan, G. (2015). "Tertiary Structural Propensities Reveal
//!   Fundamental Sequence/Structure Relationships in Proteins." *Structure*, 23(5), 961–971.
//! - Zheng, W., & Grigoryan, G. (2017). "Sequence statistics of tertiary structural motifs
//!   reflect protein stability." *PLoS ONE*, 12(5), e0178272.

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
    /// World coordinates of sidechain heavy atoms (excluding backbone and non-ALA CB).
    pub atoms: Vec<[f64; 3]>,
    /// Bounding box [min_x, max_x, min_y, max_y, min_z, max_z] for fast collision pruning.
    pub bbox: [f64; 6],
}

/// Cached Phase A computation for a single protein residue position.
#[derive(Clone, Debug)]
pub struct ResidueCache {
    /// Residue index in ProteinBackbone.
    pub index: ResidueIndex,
    /// Residue name in structure.
    pub res_name: String,
    /// Whether `default_bin` was used because `phi` or `psi` was undefined (`None`, Debt #1897).
    pub used_default_bin: bool,
    /// Fraction of rotamers pruned by steric clash with the backbone in [0.0, 1.0].
    pub crowdedness: f64,
    /// Surviving partition function weight Omega_i = sum P(a) P(r_a) for unpruned rotamers.
    pub omega_i: f64,
    /// Unpruned library partition function weight Omega_lib = sum P(a) P(r_a) across all rotamers.
    pub omega_lib: f64,
    /// Total number of rotamers in the library for this position.
    pub n_library: usize,
    /// Number of rotamers that survived backbone clash pruning.
    pub n_surviving: usize,
    /// Surviving rotamers with placed heavy-atom coordinates.
    pub surviving_rotamers: Vec<PlacedRotamerData>,
    /// Mutual sidechain-to-backbone interference I(i, j) with other residues.
    pub interference: HashMap<ResidueIndex, f64>,
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
    let (points, tags) = build_bb_atoms(bb);
    ProximityGrid::build(&points, tags, CLASH_DIST)
}

/// Execute Phase A rotamer placement, backbone clash pruning, and crowdedness evaluation.
pub fn cache_residue(
    ri: ResidueIndex,
    bb: &ProteinBackbone,
    rotlib: &RotamerLibrary,
    bb_grid: &ProximityGrid<ResidueIndex>,
) -> Result<ResidueCache, ConFindError> {
    let rb: &ResidueBackbone = &bb.bb[ri.0 as usize];

    let n = rb
        .n
        .ok_or(ConFindError::MissingBackbone(ri, "N"))?;
    let ca = rb
        .ca
        .ok_or(ConFindError::MissingBackbone(ri, "CA"))?;
    let c = rb
        .c
        .ok_or(ConFindError::MissingBackbone(ri, "C"))?;

    // For Glycine: no sidechain, crowdedness is defined as 0.0 (spec §3.3).
    if rb.res_name == "GLY" {
        return Ok(ResidueCache {
            index: ri,
            res_name: rb.res_name.clone(),
            used_default_bin: rb.phi.is_none() || rb.psi.is_none(),
            crowdedness: 0.0,
            omega_i: 0.0,
            omega_lib: 0.0,
            n_library: 0,
            n_surviving: 0,
            surviving_rotamers: Vec::new(),
            interference: HashMap::new(),
        });
    }

    let phi = rb.phi;
    let psi = rb.psi;
    let cis_proline = rb.is_cis_peptide;

    let mut n_library = 0usize;
    let mut n_surviving = 0usize;
    let mut omega_lib = 0.0f64;
    let mut omega_i = 0.0f64;
    let mut surviving_rotamers = Vec::new();
    let mut interference_map: HashMap<ResidueIndex, f64> = HashMap::new();
    let mut used_default_bin = phi.is_none() || psi.is_none();

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

            n_library += 1;
            let rot_weight = p_a * p_r;
            omega_lib += rot_weight;

            let placed = rotlib.place_rotamer(aa, phi, psi, rot_idx, cis_proline, n, ca, c)?;
            used_default_bin |= placed.used_default_bin;

            // Extract sidechain heavy atoms (spec §4.5)
            let mut sc_atoms = Vec::new();
            for atom in placed.atoms {
                if counts_as_sidechain(&atom.name, aa) {
                    sc_atoms.push(atom.xyz);
                }
            }

            // Check backbone steric clashes (spec §2.4: distance <= 2.0 Å to any other residue)
            let mut has_clash = false;
            let mut clashing_neighbors = HashSet::new();

            for &pt in &sc_atoms {
                let neighbors = bb_grid.points_within(pt, 0.0, CLASH_DIST);
                for neighbor_idx in neighbors {
                    if neighbor_idx != ri {
                        has_clash = true;
                        clashing_neighbors.insert(neighbor_idx);
                    }
                }
            }

            if has_clash {
                // Rotamer clashing with backbone does not survive Phase A
                for neighbor_idx in clashing_neighbors {
                    *interference_map.entry(neighbor_idx).or_insert(0.0) += rot_weight;
                }
            } else {
                // Rotamer survives Phase A
                omega_i += rot_weight;
                n_surviving += 1;
                let bbox = compute_bbox(&sc_atoms);
                surviving_rotamers.push(PlacedRotamerData {
                    aa,
                    rot_idx,
                    p_a,
                    p_r,
                    weight: rot_weight,
                    atoms: sc_atoms,
                    bbox,
                });
            }
        }
    }

    // Crowdedness C(i) calculation (spec §2.4): 1.0 - (Omega_i / Omega_lib)
    let crowdedness = if omega_lib > 0.0 {
        if n_surviving == 0 {
            1.0
        } else {
            (1.0 - (omega_i / omega_lib)).clamp(0.0, 1.0)
        }
    } else {
        0.0
    };

    Ok(ResidueCache {
        index: ri,
        res_name: rb.res_name.clone(),
        used_default_bin,
        crowdedness,
        omega_i,
        omega_lib,
        n_library,
        n_surviving,
        surviving_rotamers,
        interference: interference_map,
    })
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
