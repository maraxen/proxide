use crate::error::ConFindError;
use crate::precondition::chain_breaks;
use proxide_core::processing::residues::{ProcessedStructure, ResidueId};
use proxide_geometry::geometry::angles::{compute_backbone_dihedrals_f64, dihedral_angle_f64};
use proxide_io::formats::pdb_fields::parse_pdb_records;
use std::collections::{HashMap, HashSet};
use std::io::BufReader;
use std::path::Path;

/// Flat dense index into ProteinBackbone::bb (0-based, protein residues only).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ResidueIndex(pub u32);

impl std::fmt::Display for ResidueIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ResidueIndex({})", self.0)
    }
}

/// How to handle a geometric chain break (a CA–CA gap larger than
/// [`CHAIN_BREAK_CA_ANGSTROM`]) when filling backbone dihedrals.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChainBreakPolicy {
    /// Compute dihedrals across a geometric chain break as if the break were
    /// not there (if atoms are physically present).
    Bridge,
    /// Treat each geometric chain break as a boundary: the dihedral that would
    /// cross the break (the departing residue's ψ, the arriving residue's φ and ω)
    /// is `None`. Default under clean-room specification.
    #[default]
    Split,
}

/// How to handle a missing backbone atom when filling backbone dihedrals.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MissingAtomPolicy {
    /// Each dihedral is computed independently from its own four named atoms
    /// and sequence neighbour(s). Missing required atom or neighbour results in `None`.
    #[default]
    PerDihedral,
    /// Legacy dense-array bridging algorithm (for backward compatibility).
    LegacyCompact,
}

/// Options controlling how backbone dihedrals are filled.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BackboneOptions {
    pub chain_breaks: ChainBreakPolicy,
    pub missing_atoms: MissingAtomPolicy,
}

impl BackboneOptions {
    /// Set the chain-break policy, returning `self` for chaining.
    #[must_use]
    pub const fn with_chain_breaks(self, policy: ChainBreakPolicy) -> Self {
        Self {
            chain_breaks: policy,
            ..self
        }
    }

    /// Set the missing-atom policy, returning `self` for chaining.
    #[must_use]
    pub const fn with_missing_atoms(self, policy: MissingAtomPolicy) -> Self {
        Self {
            missing_atoms: policy,
            ..self
        }
    }
}

/// Backbone geometry for a single protein residue in f64 precision.
///
/// Under Debt #1897, undefined phi/psi angles (at chain termini or breaks)
/// are represented strictly as `None`. Sentinel floats are prohibited.
#[derive(Debug, Clone, PartialEq)]
pub struct ResidueBackbone {
    /// Three-letter amino acid name (e.g. `"ALA"`).
    pub res_name: String,
    /// N atom position in Å (world frame).
    pub n: Option<[f64; 3]>,
    /// Cα atom position in Å (world frame).
    pub ca: Option<[f64; 3]>,
    /// C atom position in Å (world frame).
    pub c: Option<[f64; 3]>,
    /// O atom position in Å (world frame).
    pub o: Option<[f64; 3]>,
    /// φ dihedral angle in degrees; `None` if undefined (Debt #1897).
    pub phi: Option<f64>,
    /// ψ dihedral angle in degrees; `None` if undefined (Debt #1897).
    pub psi: Option<f64>,
    /// ω dihedral angle in degrees; `None` if undefined (Debt #1897).
    /// Follows standard IUPAC convention torsion(CA_{i-1}, C_{i-1}, N_i, CA_i) (Debt #1903).
    pub omega: Option<f64>,
    /// `true` iff `|omega| < 30.0°` (cis peptide bond); `false` when `omega` is `None`.
    pub is_cis_peptide: bool,
}

/// Backbone geometry for an entire protein, extracted in f64 precision.
#[derive(Debug, Clone, PartialEq)]
pub struct ProteinBackbone {
    /// Per-residue backbone geometry.
    pub bb: Vec<ResidueBackbone>,
    /// Parallel to `bb`; identifies each residue for output.
    pub ids: Vec<ResidueId>,
    /// Parallel to `bb`; maps residue index → chain index within the structure.
    pub chain_map: Vec<usize>,
}

/// Check if a residue name represents a known terminal capping group.
#[inline]
pub fn is_capping_group(res_name: &str) -> bool {
    matches!(res_name, "ACE" | "NME" | "NH2")
}

/// Extract backbone from ProcessedStructure, widening f32 to f64 for full-precision computation.
pub fn extract_f64_backbone(s: &ProcessedStructure) -> Result<ProteinBackbone, ConFindError> {
    extract_f64_backbone_with_options(s, &BackboneOptions::default())
}

/// Extract backbone from ProcessedStructure with explicit dihedral-filling policies.
pub fn extract_f64_backbone_with_options(
    s: &ProcessedStructure,
    opts: &BackboneOptions,
) -> Result<ProteinBackbone, ConFindError> {
    let mut bb: Vec<ResidueBackbone> = Vec::new();
    let mut ids: Vec<ResidueId> = Vec::new();
    let mut chain_map: Vec<usize> = Vec::new();
    let mut ter_before: Vec<bool> = Vec::new();
    let mut skipped_before: Vec<bool> = Vec::new();
    let mut pending_ter = false;
    let mut pending_skip = false;

    for (res_idx, resinfo) in s.residue_info.iter().enumerate() {
        let has_ter = s.ter_positions.contains(&res_idx);
        if s.molecule_type[resinfo.start_atom] != 0 && !is_capping_group(&resinfo.res_name) {
            pending_skip = true;
            pending_ter |= has_ter;
            continue;
        }
        let mut n_pos = None;
        let mut ca_pos = None;
        let mut c_pos = None;
        let mut o_pos = None;

        for atom_idx in resinfo.start_atom..(resinfo.start_atom + resinfo.num_atoms) {
            let name = s.raw_atoms.atom_names[atom_idx].as_str();
            let raw_xyz = [
                s.raw_atoms.coords[3 * atom_idx],
                s.raw_atoms.coords[3 * atom_idx + 1],
                s.raw_atoms.coords[3 * atom_idx + 2],
            ];
            if !raw_xyz[0].is_finite() || !raw_xyz[1].is_finite() || !raw_xyz[2].is_finite() {
                return Err(ConFindError::NonFiniteCoordinate {
                    residue: ResidueIndex(bb.len() as u32),
                    atom_name: name.to_string(),
                });
            }
            let xyz = [raw_xyz[0] as f64, raw_xyz[1] as f64, raw_xyz[2] as f64];
            match name {
                "N" => n_pos = Some(xyz),
                "CA" => ca_pos = Some(xyz),
                "C" => c_pos = Some(xyz),
                "O" => o_pos = Some(xyz),
                _ => {}
            }
        }

        let chain_idx = *s.chain_indices.get(&resinfo.chain_id).unwrap_or(&0);
        let had_ter = !bb.is_empty() && (has_ter || pending_ter);
        bb.push(ResidueBackbone {
            res_name: resinfo.res_name.clone(),
            n: n_pos,
            ca: ca_pos,
            c: c_pos,
            o: o_pos,
            phi: None,
            psi: None,
            omega: None,
            is_cis_peptide: false,
        });
        ids.push(ResidueId {
            chain_id: resinfo.chain_id.clone(),
            res_id: resinfo.res_id,
            insertion_code: resinfo.insertion_code,
        });
        chain_map.push(chain_idx);
        ter_before.push(had_ter);
        skipped_before.push(pending_skip);
        pending_ter = false;
        pending_skip = false;
    }

    fill_dihedrals(&mut bb, &chain_map, &ter_before, &skipped_before, opts)?;
    Ok(ProteinBackbone { bb, ids, chain_map })
}

/// Parse a PDB file reading coordinate columns 30–54 as f64 for full text precision.
/// Preserves file encounter order and detects explicit TER records (Debt #1904).
pub fn load_pdb_f64<P: AsRef<Path>>(path: P) -> Result<ProteinBackbone, ConFindError> {
    load_pdb_f64_with_options(path, &BackboneOptions::default())
}

/// `load_pdb_f64` with explicit dihedral-filling policies.
pub(crate) fn load_pdb_f64_with_options<P: AsRef<Path>>(
    path: P,
    opts: &BackboneOptions,
) -> Result<ProteinBackbone, ConFindError> {
    // 1. Parse records in f64 precision (includes TER boundary flags from parse_pdb_records)
    let file = std::fs::File::open(path.as_ref())?;
    let records = parse_pdb_records(BufReader::new(file))
        .map_err(|e| std::io::Error::other(e.to_string()))?;

    if records.is_empty() {
        return Err(std::io::Error::other("No atoms found in PDB file").into());
    }

    // 2. Group atoms by residue in encounter order (preserving file order, Debt #1904)
    struct ResidueBuilder {
        id: ResidueId,
        res_name: String,
        n: Option<[f64; 3]>,
        ca: Option<[f64; 3]>,
        c: Option<[f64; 3]>,
        o: Option<[f64; 3]>,
        ter_before: bool,
        is_protein: bool,
    }

    let is_solvent = |name: &str| matches!(name, "HOH" | "WAT" | "TIP3" | "SOL" | "DOD");

    let mut residues: Vec<ResidueBuilder> = Vec::new();
    let mut current_builder: Option<ResidueBuilder> = None;

    for rec in records {
        let res_id = ResidueId {
            chain_id: rec.chain_id.clone(),
            res_id: rec.res_seq,
            insertion_code: rec.i_code,
        };

        let matches_current = current_builder
            .as_ref()
            .is_some_and(|b| b.id == res_id && b.res_name == rec.res_name);

        if !matches_current {
            if let Some(b) = current_builder.take() {
                residues.push(b);
            }
            let is_prot = !is_solvent(&rec.res_name);
            current_builder = Some(ResidueBuilder {
                id: res_id,
                res_name: rec.res_name.clone(),
                n: None,
                ca: None,
                c: None,
                o: None,
                ter_before: rec.ter_before,
                is_protein: is_prot,
            });
        }

        let builder = current_builder.as_mut().unwrap();
        builder.ter_before |= rec.ter_before;
        let xyz = [rec.x64, rec.y64, rec.z64];
        match rec.atom_name.as_str() {
            "N" => {
                builder.n.get_or_insert(xyz);
            }
            "CA" => {
                builder.ca.get_or_insert(xyz);
            }
            "C" => {
                builder.c.get_or_insert(xyz);
            }
            "O" => {
                builder.o.get_or_insert(xyz);
            }
            _ => {}
        }
    }
    if let Some(b) = current_builder.take() {
        residues.push(b);
    }

    // 3. Construct ProteinBackbone vectors in preserved file order
    let mut bb: Vec<ResidueBackbone> = Vec::new();
    let mut ids: Vec<ResidueId> = Vec::new();
    let mut chain_map: Vec<usize> = Vec::new();
    let mut ter_before: Vec<bool> = Vec::new();
    let mut skipped_before: Vec<bool> = Vec::new();
    let mut pending_ter = false;
    let mut pending_skip = false;
    let mut chain_indices: HashMap<String, usize> = HashMap::new();

    for r in residues {
        if !r.is_protein {
            pending_skip = true;
            pending_ter |= r.ter_before;
            continue;
        }

        let next_chain_idx = chain_indices.len();
        let chain_idx = *chain_indices
            .entry(r.id.chain_id.clone())
            .or_insert(next_chain_idx);

        let had_ter = !bb.is_empty() && (r.ter_before || pending_ter);

        bb.push(ResidueBackbone {
            res_name: r.res_name,
            n: r.n,
            ca: r.ca,
            c: r.c,
            o: r.o,
            phi: None,
            psi: None,
            omega: None,
            is_cis_peptide: false,
        });
        ids.push(r.id);
        chain_map.push(chain_idx);
        ter_before.push(had_ter);
        skipped_before.push(pending_skip);
        pending_ter = false;
        pending_skip = false;
    }

    fill_dihedrals(&mut bb, &chain_map, &ter_before, &skipped_before, opts)?;
    Ok(ProteinBackbone { bb, ids, chain_map })
}

/// Compute phi/psi/omega per chain segment and fill into `bb`, per `opts`.
fn fill_dihedrals(
    bb: &mut [ResidueBackbone],
    chain_map: &[usize],
    ter_before: &[bool],
    skipped_before: &[bool],
    opts: &BackboneOptions,
) -> Result<(), ConFindError> {
    if opts.missing_atoms == MissingAtomPolicy::LegacyCompact
        && opts.chain_breaks == ChainBreakPolicy::Split
    {
        return Err(ConFindError::InvalidOptions(
            "MissingAtomPolicy::LegacyCompact cannot be combined with \
             ChainBreakPolicy::Split: LegacyCompact's dense-bridging algorithm has no \
             chain-break-aware notion of residue adjacency"
                .to_string(),
        ));
    }

    match opts.missing_atoms {
        MissingAtomPolicy::LegacyCompact => fill_dihedrals_legacy_compact(bb, chain_map),
        MissingAtomPolicy::PerDihedral => fill_dihedrals_per_dihedral(
            bb,
            chain_map,
            ter_before,
            skipped_before,
            opts.chain_breaks,
        ),
    }

    Ok(())
}

/// `MissingAtomPolicy::LegacyCompact`: dense-array bridging algorithm.
fn fill_dihedrals_legacy_compact(bb: &mut [ResidueBackbone], chain_map: &[usize]) {
    let n = bb.len();
    if n == 0 {
        return;
    }

    let mut starts: Vec<usize> = vec![0];
    for i in 1..n {
        if chain_map[i] != chain_map[i - 1] {
            starts.push(i);
        }
    }
    starts.push(n);

    for w in starts.windows(2) {
        let seg_start = w[0];
        let seg_end = w[1];

        let mut dense: Vec<[[f64; 3]; 3]> = Vec::new();
        let mut dense_to_bb: Vec<usize> = Vec::new();

        for (i, rb) in bb[seg_start..seg_end].iter().enumerate() {
            if let (Some(n), Some(ca), Some(c)) = (rb.n, rb.ca, rb.c) {
                dense.push([n, ca, c]);
                dense_to_bb.push(seg_start + i);
            }
        }

        if dense.is_empty() {
            continue;
        }

        let dihedrals = compute_backbone_dihedrals_f64(&dense);
        for (d_pos, &bb_i) in dense_to_bb.iter().enumerate() {
            let d = &dihedrals[d_pos];
            bb[bb_i].phi = d.phi.map(|r| -r.to_degrees());
            bb[bb_i].psi = d.psi.map(|r| -r.to_degrees());
            bb[bb_i].omega = d.omega.map(|r| -r.to_degrees());
            bb[bb_i].is_cis_peptide = bb[bb_i].omega.is_some_and(|w| w.abs() < 30.0);
        }
    }
}

/// `MissingAtomPolicy::PerDihedral`: each dihedral is computed independently
/// from its four named atoms, respecting segmented chain semantics (Debt #1904)
/// and standard IUPAC omega dihedral convention (Debt #1903).
fn fill_dihedrals_per_dihedral(
    bb: &mut [ResidueBackbone],
    chain_map: &[usize],
    ter_before: &[bool],
    skipped_before: &[bool],
    chain_break_policy: ChainBreakPolicy,
) {
    let n = bb.len();
    if n == 0 {
        return;
    }

    // Identify geometric chain breaks (consecutive CA-CA > 4.5 Å)
    let break_indices: HashSet<usize> = if chain_break_policy == ChainBreakPolicy::Split {
        chain_breaks(bb, chain_map)
            .into_iter()
            .map(|b| b.index)
            .collect()
    } else {
        HashSet::new()
    };

    // A chain boundary occurs between i-1 and i if:
    // 1. i == 0 (N-terminus of first residue)
    // 2. chain_map[i] != chain_map[i - 1] (chain transition)
    // 3. ter_before[i] (explicit TER record)
    // 4. skipped_before[i] (skipped non-protein residue)
    // 5. Either residue is a capping group (ACE, NME, NH2)
    // 6. break_indices.contains(&i) (CA-CA distance > 4.5 Å)
    let mut is_boundary = vec![false; n];
    is_boundary[0] = true;
    for i in 1..n {
        if chain_map[i] != chain_map[i - 1]
            || ter_before.get(i).copied().unwrap_or(false)
            || skipped_before.get(i).copied().unwrap_or(false)
            || is_capping_group(&bb[i - 1].res_name)
            || is_capping_group(&bb[i].res_name)
            || break_indices.contains(&i)
        {
            is_boundary[i] = true;
        }
    }

    for i in 0..n {
        let prev_ok = !is_boundary[i];
        let next_ok = i + 1 < n && !is_boundary[i + 1];

        let phi = if prev_ok {
            match (bb[i - 1].c, bb[i].n, bb[i].ca, bb[i].c) {
                (Some(c_prev), Some(n_i), Some(ca_i), Some(c_i)) => {
                    Some(-dihedral_angle_f64(&c_prev, &n_i, &ca_i, &c_i).to_degrees())
                }
                _ => None,
            }
        } else {
            None
        };

        let psi = if next_ok {
            match (bb[i].n, bb[i].ca, bb[i].c, bb[i + 1].n) {
                (Some(n_i), Some(ca_i), Some(c_i), Some(n_next)) => {
                    Some(-dihedral_angle_f64(&n_i, &ca_i, &c_i, &n_next).to_degrees())
                }
                _ => None,
            }
        } else {
            None
        };

        // Standard IUPAC omega torsion: torsion(CA_{i-1}, C_{i-1}, N_i, CA_i) (Debt #1903)
        let omega = if prev_ok {
            match (bb[i - 1].ca, bb[i - 1].c, bb[i].n, bb[i].ca) {
                (Some(ca_prev), Some(c_prev), Some(n_i), Some(ca_i)) => {
                    Some(-dihedral_angle_f64(&ca_prev, &c_prev, &n_i, &ca_i).to_degrees())
                }
                _ => None,
            }
        } else {
            None
        };

        bb[i].phi = phi;
        bb[i].psi = psi;
        bb[i].omega = omega;
        bb[i].is_cis_peptide = omega.is_some_and(|w| w.abs() < 30.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_rb(
        res_name: &str,
        n: Option<[f64; 3]>,
        ca: Option<[f64; 3]>,
        c: Option<[f64; 3]>,
    ) -> ResidueBackbone {
        ResidueBackbone {
            res_name: res_name.to_string(),
            n,
            ca,
            c,
            o: None,
            phi: None,
            psi: None,
            omega: None,
            is_cis_peptide: false,
        }
    }

    #[test]
    fn backbone_options_default_is_split_per_dihedral() {
        assert_eq!(
            BackboneOptions::default(),
            BackboneOptions {
                chain_breaks: ChainBreakPolicy::Split,
                missing_atoms: MissingAtomPolicy::PerDihedral,
            }
        );
    }

    #[test]
    fn backbone_options_builders_chain() {
        let o = BackboneOptions::default()
            .with_chain_breaks(ChainBreakPolicy::Bridge)
            .with_missing_atoms(MissingAtomPolicy::LegacyCompact);
        assert_eq!(o.chain_breaks, ChainBreakPolicy::Bridge);
        assert_eq!(o.missing_atoms, MissingAtomPolicy::LegacyCompact);
    }

    #[test]
    fn legacy_compact_plus_split_is_err() {
        let mut bb = vec![mk_rb(
            "ALA",
            Some([0.0, 0.0, 0.0]),
            Some([1.0, 0.0, 0.0]),
            Some([2.0, 0.0, 0.0]),
        )];
        let chain_map = vec![0usize];
        let ter_before = vec![false];
        let skipped_before = vec![false];
        let opts = BackboneOptions::default()
            .with_missing_atoms(MissingAtomPolicy::LegacyCompact)
            .with_chain_breaks(ChainBreakPolicy::Split);

        let result = fill_dihedrals(&mut bb, &chain_map, &ter_before, &skipped_before, &opts);
        match result {
            Err(ConFindError::InvalidOptions(_)) => {}
            other => panic!("expected Err(InvalidOptions), got {other:?}"),
        }
    }

    #[test]
    fn dihedrals_none_at_chain_termini() {
        let n0 = [0.0, 0.0, 0.0];
        let ca0 = [1.46, 0.0, 0.0];
        let c0 = [2.0, 1.4, 0.0];
        let n1 = [3.3, 1.6, 0.3];
        let ca1 = [4.0, 2.9, 0.5];
        let c1 = [4.8, 4.1, 0.6];

        let mut bb = vec![
            mk_rb("ALA", Some(n0), Some(ca0), Some(c0)),
            mk_rb("GLY", Some(n1), Some(ca1), Some(c1)),
        ];
        let chain_map = vec![0usize, 0];
        let ter_before = vec![false, false];
        let skipped_before = vec![false, false];

        fill_dihedrals(
            &mut bb,
            &chain_map,
            &ter_before,
            &skipped_before,
            &BackboneOptions::default(),
        )
        .expect("fill should succeed");

        // Chain first: phi is None, omega is None, psi is Some
        assert_eq!(bb[0].phi, None);
        assert_eq!(bb[0].omega, None);
        assert!(bb[0].psi.is_some());

        // Chain last: phi is Some, omega is Some, psi is None
        assert!(bb[1].phi.is_some());
        assert!(bb[1].omega.is_some());
        assert_eq!(bb[1].psi, None);
    }

    #[test]
    fn chain_break_causes_dihedrals_to_be_none() {
        // Two residues with CA-CA distance > 4.5 Å
        let n0 = [0.0, 0.0, 0.0];
        let ca0 = [1.0, 0.0, 0.0];
        let c0 = [2.0, 0.0, 0.0];

        let n1 = [10.0, 0.0, 0.0];
        let ca1 = [11.0, 0.0, 0.0];
        let c1 = [12.0, 0.0, 0.0];

        let mut bb = vec![
            mk_rb("ALA", Some(n0), Some(ca0), Some(c0)),
            mk_rb("GLY", Some(n1), Some(ca1), Some(c1)),
        ];
        let chain_map = vec![0usize, 0];
        let ter_before = vec![false, false];
        let skipped_before = vec![false, false];

        fill_dihedrals(
            &mut bb,
            &chain_map,
            &ter_before,
            &skipped_before,
            &BackboneOptions::default(),
        )
        .expect("fill should succeed");

        // Distance break: bb[0].psi must be None, bb[1].phi/omega must be None
        assert_eq!(bb[0].psi, None);
        assert_eq!(bb[1].phi, None);
        assert_eq!(bb[1].omega, None);
        assert!(!bb[1].is_cis_peptide);
    }

    #[test]
    fn ter_record_causes_dihedrals_to_be_none() {
        let n0 = [0.0, 0.0, 0.0];
        let ca0 = [1.46, 0.0, 0.0];
        let c0 = [2.0, 1.4, 0.0];
        let n1 = [3.3, 1.6, 0.3];
        let ca1 = [4.0, 2.9, 0.5];
        let c1 = [4.8, 4.1, 0.6];

        let mut bb = vec![
            mk_rb("ALA", Some(n0), Some(ca0), Some(c0)),
            mk_rb("GLY", Some(n1), Some(ca1), Some(c1)),
        ];
        let chain_map = vec![0usize, 0];
        let ter_before = vec![false, true]; // Explicit TER between 0 and 1
        let skipped_before = vec![false, false];

        fill_dihedrals(
            &mut bb,
            &chain_map,
            &ter_before,
            &skipped_before,
            &BackboneOptions::default(),
        )
        .expect("fill should succeed");

        assert_eq!(bb[0].psi, None);
        assert_eq!(bb[1].phi, None);
        assert_eq!(bb[1].omega, None);
    }

    #[test]
    fn capping_group_severs_dihedrals() {
        let n0 = [0.0, 0.0, 0.0];
        let ca0 = [1.46, 0.0, 0.0];
        let c0 = [2.0, 1.4, 0.0];
        let n1 = [3.3, 1.6, 0.3];
        let ca1 = [4.0, 2.9, 0.5];
        let c1 = [4.8, 4.1, 0.6];

        let mut bb = vec![
            mk_rb("ACE", Some(n0), Some(ca0), Some(c0)), // N-terminal cap
            mk_rb("GLY", Some(n1), Some(ca1), Some(c1)),
        ];
        let chain_map = vec![0usize, 0];
        let ter_before = vec![false, false];
        let skipped_before = vec![false, false];

        fill_dihedrals(
            &mut bb,
            &chain_map,
            &ter_before,
            &skipped_before,
            &BackboneOptions::default(),
        )
        .expect("fill should succeed");

        // ACE should not cross-link to GLY
        assert_eq!(bb[0].psi, None);
        assert_eq!(bb[1].phi, None);
        assert_eq!(bb[1].omega, None);
    }

    #[test]
    fn garbage_x_errors_naming_line_and_field() {
        let content = "\
ATOM      1  N   MET A   1      xx.xxx  29.699   5.276  1.00 49.05           N
ATOM      2  CA  MET A   1      21.154  29.699   5.276  1.00 49.05           C
";
        let mut f = tempfile::NamedTempFile::new().expect("failed to create temp file");
        use std::io::Write as _;
        f.write_all(content.as_bytes()).expect("write failed");

        let err = load_pdb_f64(f.path()).expect_err("garbage coordinate should error");
        let msg = err.to_string();
        assert!(msg.contains("line 1"), "expected line 1 in msg: {msg}");
        assert!(msg.contains("'x'"), "expected 'x' in msg: {msg}");
    }
}
