use crate::coords::{ProteinBackbone, ResidueIndex};
use crate::error::ConFindError;
use crate::grid::ProximityGrid;
use crate::params::{aa_propensity, AA_NAMES, CLASH_DIST, CONT_DIST};
use proxide_rotlib::{counts_as_sidechain, RotamerId, RotamerLibrary};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// A rotamer kept in the phase-A cache.
///
/// `used_default_bin` is the library's own flag for an undefined backbone
/// angle. `RotamerId` does not carry it, so dropping `PlacedRotamer` down to
/// `placed.id` would hide a default-bin placement.
#[derive(Clone, Debug)]
pub struct CachedRotamer {
    pub id: RotamerId,
    pub used_default_bin: bool,
}

/// Map the in-band undefined dihedral `9999.0` to `None` at the rotlib boundary.
///
/// `ResidueBackbone.phi` / `psi` are still `f64` (PR #91 makes them `Option<f64>`).
/// Passing `Some(9999.0)` wraps to −81° and selects a real Dunbrack bin.
fn defined_backbone_angle(angle: f64) -> Option<f64> {
    if angle == 9999.0 {
        None
    } else {
        Some(angle)
    }
}

/// Per-residue rotamer cache computed in phase A: surviving rotamers, grids, and interference data.
pub struct ResidueCache {
    pub surviving_rotamers: Vec<Arc<CachedRotamer>>,
    /// Per-aa proximity grid of surviving heavy-SC atoms.
    /// `None` means every rotamer of that (library-known) aa was pruned by backbone clash;
    /// it never means the aa was unknown to the library — an unknown aa is a hard error from
    /// `cache_residue_impl` instead (`ConFindError::RotlibError(RotlibError::UnknownAa)`).
    pub rotamer_grids: HashMap<String, Option<ProximityGrid<Arc<RotamerId>>>>,
    pub fraction_pruned: f64,
    pub n_library_rotamers: usize,
    /// `interference[resB_idx][aa]` = accumulated aaP * rotP / 100
    pub interference: HashMap<ResidueIndex, HashMap<String, f64>>,
    /// Backbone atoms permanently clashed by any ALA rotamer of self.
    /// Stores raw backbone atom indices for constrained-contacts lookup.
    pub permanent_contacts: Vec<usize>,
}

/// Return sum of amino-acid propensity times rotamer probability over surviving rotamers in the available set.
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
        .filter(|r| available_aa.contains(r.id.aa.as_str()))
        .map(|r| {
            // RotamerId came from this library's own surviving_rotamers, so its (aa, bin,
            // rot_index) is guaranteed valid here — an error would mean the cache and the
            // library it was built from have gone out of sync.
            let p = rotlib
                .rotamer_probability_by_id(&r.id)
                .expect("RotamerId came from this library");
            aa_propensity(&r.id.aa) * p
        })
        .sum()
}

/// Sum of `aaProp` for each amino acid in `available_aa`, divided by 100.
pub fn weight_of_available_amino_acids(available_aa: &HashSet<&str>) -> f64 {
    available_aa.iter().map(|aa| aa_propensity(aa)).sum::<f64>() / 100.0
}

/// Build flattened backbone atom coordinate vector and residue index mapping.
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
/// # Why Phase A pruning is non-skippable
///
/// Phase A backbone-clash pruning (`CLASH_DIST = 2.0 Å`) is non-skippable: backbone-clashing
/// rotamers must be excluded from the `weight_of_available_rotamers` denominator before contact
/// degree is computed. Including them would inflate the denominator and dilute all CD values.
/// `crowdedness = fraction_pruned` is a free byproduct.
///
/// More precisely, the CD formula is:
///
/// ```text
/// CD(i,j) = Σ aaProp_a * rotP_a * aaProp_b * rotP_b  /  W_a * W_b
/// ```
///
/// where `W_a = Σ aaProp * rotP` over **surviving** rotamers of residue i.
///
/// If backbone-clashing rotamers were retained (Phase A skipped):
/// 1. `W_a` would be inflated — physically impossible states would dilute all
///    CD values, making every position appear more exposed than it is.
/// 2. False sidechain–sidechain contacts would arise from clashing rotamers that
///    can never coexist with the backbone.
///
/// `CLASH_DIST = 2.0 Å` is a hard vdW exclusion threshold, not a tunable approximation.
/// `fraction_pruned` (crowdedness) is a free byproduct with zero additional compute cost.
///
/// # Thread safety
///
/// Thread-safe: reads from shared grids; writes only into `cache_out` and `interf_out`
/// which are per-residue slots.
#[allow(clippy::too_many_arguments)]
pub fn cache_residue_impl(
    ri: ResidueIndex,
    backbone: &ProteinBackbone,
    rotlib: &RotamerLibrary,
    bb_grid: &ProximityGrid<usize>, // tag = index into bb_atoms
    bb_atom_res: &[ResidueIndex],   // maps bb_atom_idx → owning ResidueIndex
) -> Result<ResidueCache, ConFindError> {
    let rb = &backbone.bb[ri.0 as usize];
    let phi = defined_backbone_angle(rb.phi);
    let psi = defined_backbone_angle(rb.psi);

    let n = rb.n.ok_or(ConFindError::MissingBackbone(ri, "N"))?;
    let ca = rb.ca.ok_or(ConFindError::MissingBackbone(ri, "CA"))?;
    let c = rb.c.ok_or(ConFindError::MissingBackbone(ri, "C"))?;

    let mut surviving: Vec<Arc<CachedRotamer>> = Vec::new();
    let mut rotamer_grids: HashMap<String, Option<ProximityGrid<Arc<RotamerId>>>> = HashMap::new();
    let mut interference: HashMap<ResidueIndex, HashMap<String, f64>> = HashMap::new();
    let mut permanent_contacts_set: HashSet<usize> = HashSet::new();
    let mut total_rotamers: usize = 0;

    for &aa in &AA_NAMES {
        let nr = rotlib
            .num_rotamers(aa, phi, psi, rb.is_cis_peptide)
            .map_err(ConFindError::RotlibError)?;
        total_rotamers += nr;

        let mut aa_sc_points: Vec<[f64; 3]> = Vec::new();
        let mut aa_sc_tags: Vec<Arc<RotamerId>> = Vec::new();

        for rot_index in 0..nr {
            let placed = rotlib
                .place_rotamer(aa, phi, psi, rot_index, rb.is_cis_peptide, n, ca, c)
                .map_err(ConFindError::RotlibError)?;

            let rot_id = Arc::new(placed.id.clone());
            let cached = Arc::new(CachedRotamer {
                id: placed.id.clone(),
                used_default_bin: placed.used_default_bin,
            });
            let rot_prob = rotlib.rotamer_probability_by_id(&placed.id)?;

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
            surviving.push(cached);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coords::{ProteinBackbone, ResidueBackbone, ResidueIndex};
    use crate::grid::ProximityGrid;
    use crate::params::{AA_NAMES, CONT_DIST};
    use proxide_core::processing::residues::ResidueId;
    use proxide_rotlib::rotlib_v1::{self, GeometryMode};
    use proxide_rotlib::RotamerLibrary;

    /// 9999° is the in-band "undefined" sentinel on `ResidueBackbone`. It must
    /// not be wrapped into a real Dunbrack bin: `angle_to_standard(9999) = -81`,
    /// whose nearest center in this grid is -80 (bin 0), while `default_bin` is 3.
    #[test]
    fn cache_sentinel_9999_uses_default_bin() {
        let lib = library_with_distinct_default_bin();
        let backbone = ProteinBackbone {
            bb: vec![ResidueBackbone {
                res_name: "ALA".to_string(),
                n: Some([0.0, 0.0, 0.0]),
                ca: Some([1.5, 0.0, 0.0]),
                c: Some([2.0, 1.4, 0.0]),
                o: Some([2.5, 1.4, 1.0]),
                phi: 9999.0,
                psi: 9999.0,
                omega: None,
                is_cis_peptide: false,
            }],
            ids: vec![ResidueId {
                chain_id: "A".to_string(),
                res_id: 1,
                insertion_code: ' ',
            }],
            chain_map: vec![0],
        };
        let bb_grid = ProximityGrid::build(&[], Vec::<usize>::new(), CONT_DIST);
        let cache =
            cache_residue_impl(ResidueIndex(0), &backbone, &lib, &bb_grid, &[]).expect("cache");
        assert!(
            !cache.surviving_rotamers.is_empty(),
            "synthetic library must place rotamers"
        );
        for rot in &cache.surviving_rotamers {
            assert_eq!(
                rot.id.bin_index, 3,
                "{} cached bin {} for phi=psi=9999; undefined angles must use default_bin 3, not the -81° wrap",
                rot.id.aa, rot.id.bin_index
            );
            assert!(
                rot.used_default_bin,
                "{} default-bin flag must stay visible on the cached rotamer",
                rot.id.aa
            );
        }
    }

    fn library_with_distinct_default_bin() -> RotamerLibrary {
        let phi = vec![-80.0_f64, 60.0];
        let psi = vec![-80.0_f64, 60.0];
        let residues = AA_NAMES
            .iter()
            .map(|code| {
                let mut bins = Vec::new();
                for &p in &phi {
                    for &s in &psi {
                        bins.push(rotlib_v1::Bin {
                            phi: p,
                            psi: s,
                            freq: 1.0,
                            rotamers: vec![rotlib_v1::Rotamer {
                                prob: 1.0,
                                chi: vec![],
                                coords: vec![rotlib_v1::Vec3 {
                                    x: 1.5,
                                    y: 0.5,
                                    z: 0.2,
                                }],
                            }],
                        });
                    }
                }
                rotlib_v1::ResidueEntry {
                    code: (*code).to_string(),
                    atom_names: vec!["CB".to_string()],
                    num_chi: 0,
                    phi_centers: phi.clone(),
                    psi_centers: psi.clone(),
                    default_bin: 3,
                    bins,
                }
            })
            .collect();
        RotamerLibrary::from_protobuf(&rotlib_v1::RotamerLibrary {
            version: 1,
            provenance: "synthetic".to_string(),
            attribution: "test".to_string(),
            data_license: "test".to_string(),
            geometry_mode: GeometryMode::Precomputed as i32,
            residues,
            geometry_source: String::new(),
            geometry_license: String::new(),
        })
        .expect("synthetic library")
    }
}
