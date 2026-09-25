use crate::error::ConFindError;
use crate::precondition::chain_breaks;
use proxide_core::processing::residues::{ProcessedStructure, ResidueId};
use proxide_geometry::geometry::angles::{compute_backbone_dihedrals_f64, dihedral_angle_f64};
use std::collections::HashSet;
use std::io::{BufRead, BufReader};
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
/// [`crate::precondition::CHAIN_BREAK_CA_ANGSTROM`], as detected by
/// [`crate::precondition::chain_breaks`]) when filling backbone dihedrals.
///
/// Neither variant fails fast on its own — the loud path is
/// [`crate::precondition::check_preconditions`] /
/// [`crate::precondition::require_preconditions`], which `ConFind` and the CLI
/// do not call automatically. Choosing a policy only changes which sentinel
/// values come out of dihedral computation; it never panics or errors by
/// itself (except the `LegacyCompact` + `Split` combination — see
/// [`MissingAtomPolicy`]).
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChainBreakPolicy {
    /// Compute dihedrals across a geometric chain break exactly as if the
    /// break were not there (today's / Mosaist's default behaviour): as long
    /// as the four atoms a dihedral needs are physically present, the
    /// dihedral is computed from them, however far apart they are in space.
    #[default]
    Bridge,
    /// Treat each geometric chain break as an additional missing neighbour:
    /// the dihedral that would cross the break (the departing residue's ψ,
    /// the arriving residue's φ and ω) is undefined (9999.0 / `None`) even
    /// when its four atoms are all present.
    Split,
}

/// How to handle a missing backbone atom (N, CA, or C absent on some residue
/// in the chain) when filling backbone dihedrals.
///
/// Neither variant fails fast on its own; see [`ChainBreakPolicy`] for the
/// same caveat. `9999.0`/`None` never encode "unknown" as a plausible
/// physical value here — they are the library's long-standing explicit
/// sentinel for "this dihedral could not be computed", consumed downstream
/// by the rotamer library's default-bin fallback (matching Mosaist's
/// non-strict behaviour), not a substituted measurement.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MissingAtomPolicy {
    /// Mosaist semantics (`getPhi`/`getPsi`/`getOmega`): each dihedral is
    /// computed independently from its own four named atoms and its
    /// immediate sequence neighbour(s). A dihedral is undefined only when
    /// one of ITS OWN four atoms is missing, or its required neighbour
    /// residue does not exist (chain terminus), was skipped because it is
    /// not a protein residue, or (under [`ChainBreakPolicy::Split`]) sits on
    /// the far side of a geometric chain break. A missing atom on one
    /// residue never propagates its effect past that residue's own
    /// dihedrals and the one neighbouring dihedral that names it.
    #[default]
    PerDihedral,
    /// The library's original (pre-#1890) behaviour: within each chain
    /// segment, residues missing N/CA/C are dropped from a dense array and
    /// dihedrals are computed between the surviving residues' four named
    /// atoms — silently bridging over any gap, regardless of how many
    /// residues were skipped or how far apart they are in the sequence.
    /// Reproduces `extract_f64_backbone`'s output as of base commit 316d1ab
    /// bit-for-bit (see `tests/data/dihedral_golden_316d1ab.json`).
    ///
    /// This policy computes dihedrals from non-adjacent residues; the
    /// result is not physically meaningful across a gap and exists only for
    /// backward compatibility. It is available only through
    /// [`extract_f64_backbone_with_options`] (the `ProcessedStructure`
    /// path) — not through [`load_pdb_f64`] or the confind CLI, which read
    /// f64 text coordinates directly and carry a separate, tracked gap
    /// (debt #1885).
    ///
    /// Known Mosaist divergences that apply regardless of this policy (both
    /// variants): Mosaist's own ACE/NH2 cap handling, its treatment of a
    /// `TER` record within a single chain ID, and its file-encounter
    /// residue ordering vs. this library's sorted order are not reproduced
    /// here; each is tracked as a separate filed debt.
    LegacyCompact,
}

/// Options controlling how [`extract_f64_backbone_with_options`] fills
/// backbone dihedrals. `BackboneOptions::default()` is `{Bridge, PerDihedral}`
/// — Mosaist's per-dihedral semantics, bridging geometric chain breaks.
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
/// Atom positions are `None` when the atom is absent from the PDB. Dihedral
/// angles use the sentinel `9999.0` (`None` for omega) when the angle cannot
/// be computed: at a chain terminus, when an atom the dihedral itself needs
/// is missing, when the required neighbour residue was skipped because it
/// is not a protein residue, or — under [`ChainBreakPolicy::Split`] — when
/// the neighbour sits on the far side of a geometric chain break. See
/// [`BackboneOptions`] for how these cases are chosen.
#[derive(Debug, Clone)]
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
    /// φ dihedral angle in degrees; `9999.0` if undefined (see struct docs).
    pub phi: f64,
    /// ψ dihedral angle in degrees; `9999.0` if undefined (see struct docs).
    pub psi: f64,
    /// ω dihedral angle in degrees; `None` if undefined (see struct docs) —
    /// including always for the N-terminal residue of each chain segment.
    pub omega: Option<f64>,
    /// `true` iff `|omega| < 30.0°`; `false` when `omega` is `None`.
    pub is_cis_peptide: bool,
}

/// Backbone geometry for an entire protein, extracted in f64 precision.
///
/// All three parallel vectors (`bb`, `ids`, `chain_map`) share the same
/// length and index space, where index `i` corresponds to
/// `ResidueIndex(i as u32)`.
#[derive(Debug)]
pub struct ProteinBackbone {
    /// Per-residue backbone geometry.
    pub bb: Vec<ResidueBackbone>,
    /// Parallel to `bb`; identifies each residue for output.
    pub ids: Vec<ResidueId>,
    /// Parallel to `bb`; maps residue index → chain index within the structure.
    pub chain_map: Vec<usize>,
}

/// Extract backbone from ProcessedStructure, widening f32 to f64 for full-precision computation.
///
/// This is the single f32→f64 boundary; no f32 appears in proxide-confind past this point.
///
/// Uses [`BackboneOptions::default`] (`{Bridge, PerDihedral}`). For other
/// policies, use [`extract_f64_backbone_with_options`].
pub fn extract_f64_backbone(s: &ProcessedStructure) -> Result<ProteinBackbone, ConFindError> {
    extract_f64_backbone_with_options(s, &BackboneOptions::default())
}

/// Extract backbone from ProcessedStructure with explicit dihedral-filling policies.
///
/// See [`BackboneOptions`], [`ChainBreakPolicy`], and [`MissingAtomPolicy`].
pub fn extract_f64_backbone_with_options(
    s: &ProcessedStructure,
    opts: &BackboneOptions,
) -> Result<ProteinBackbone, ConFindError> {
    let mut bb: Vec<ResidueBackbone> = Vec::new();
    let mut ids: Vec<ResidueId> = Vec::new();
    let mut chain_map: Vec<usize> = Vec::new();
    let mut skipped_before: Vec<bool> = Vec::new();
    let mut pending_skip = false;

    for resinfo in &s.residue_info {
        if s.molecule_type[resinfo.start_atom] != 0 {
            pending_skip = true;
            continue;
        }
        let mut n_pos = None;
        let mut ca_pos = None;
        let mut c_pos = None;
        let mut o_pos = None;

        for atom_idx in resinfo.start_atom..(resinfo.start_atom + resinfo.num_atoms) {
            let name = s.raw_atoms.atom_names[atom_idx].as_str();
            let xyz = [
                s.raw_atoms.coords[3 * atom_idx] as f64,
                s.raw_atoms.coords[3 * atom_idx + 1] as f64,
                s.raw_atoms.coords[3 * atom_idx + 2] as f64,
            ];
            match name {
                "N" => n_pos = Some(xyz),
                "CA" => ca_pos = Some(xyz),
                "C" => c_pos = Some(xyz),
                "O" => o_pos = Some(xyz),
                _ => {}
            }
        }

        let chain_idx = *s.chain_indices.get(&resinfo.chain_id).unwrap_or(&0);
        bb.push(ResidueBackbone {
            res_name: resinfo.res_name.clone(),
            n: n_pos,
            ca: ca_pos,
            c: c_pos,
            o: o_pos,
            phi: 9999.0,
            psi: 9999.0,
            omega: None,
            is_cis_peptide: false,
        });
        ids.push(ResidueId {
            chain_id: resinfo.chain_id.clone(),
            res_id: resinfo.res_id,
            insertion_code: resinfo.insertion_code,
        });
        chain_map.push(chain_idx);
        skipped_before.push(pending_skip);
        pending_skip = false;
    }

    fill_dihedrals(&mut bb, &chain_map, &skipped_before, opts)?;
    Ok(ProteinBackbone { bb, ids, chain_map })
}

/// Parse a PDB file reading coordinate columns 30–54 as f64 for full text precision.
///
/// Uses [`BackboneOptions::default`] (`{Bridge, PerDihedral}`). This path
/// does not support [`MissingAtomPolicy::LegacyCompact`] (debt #1885); use
/// [`extract_f64_backbone_with_options`] for that.
pub fn load_pdb_f64<P: AsRef<Path>>(path: P) -> Result<ProteinBackbone, ConFindError> {
    load_pdb_f64_with_options(path, &BackboneOptions::default())
}

/// `load_pdb_f64` with explicit dihedral-filling policies. `pub(crate)`: exercised
/// only by this crate's own tests (spec D4, debt #1890); the public entry point
/// is [`load_pdb_f64`].
pub(crate) fn load_pdb_f64_with_options<P: AsRef<Path>>(
    path: P,
    opts: &BackboneOptions,
) -> Result<ProteinBackbone, ConFindError> {
    use proxide_core::processing::residues::ProcessedStructure;

    // First pass: f64 coords in atom-record order (parallel to raw_atoms).
    let mut f64_coords: Vec<[f64; 3]> = Vec::new();
    let file = std::fs::File::open(path.as_ref())?;
    for line in BufReader::new(file).lines() {
        let line = line?;
        if line.len() < 54 {
            continue;
        }
        let rec = line[0..6].trim();
        if rec != "ATOM" && rec != "HETATM" {
            continue;
        }
        let x: f64 = line[30..38].trim().parse().unwrap_or(0.0);
        let y: f64 = line[38..46].trim().parse().unwrap_or(0.0);
        let z: f64 = line[46..54].trim().parse().unwrap_or(0.0);
        f64_coords.push([x, y, z]);
    }

    // Second pass: use standard parser for residue grouping.
    let (raw, _) = proxide_io::formats::pdb::parse_pdb_file(path.as_ref())
        .map_err(|e| std::io::Error::other(e.to_string()))?;
    let processed = ProcessedStructure::from_raw(raw).map_err(std::io::Error::other)?;

    let mut bb: Vec<ResidueBackbone> = Vec::new();
    let mut ids: Vec<ResidueId> = Vec::new();
    let mut chain_map: Vec<usize> = Vec::new();
    let mut skipped_before: Vec<bool> = Vec::new();
    let mut pending_skip = false;

    for resinfo in &processed.residue_info {
        if processed.molecule_type[resinfo.start_atom] != 0 {
            pending_skip = true;
            continue;
        }
        let mut n_pos = None;
        let mut ca_pos = None;
        let mut c_pos = None;
        let mut o_pos = None;

        for atom_idx in resinfo.start_atom..(resinfo.start_atom + resinfo.num_atoms) {
            let name = processed.raw_atoms.atom_names[atom_idx].as_str();
            let xyz = if atom_idx < f64_coords.len() {
                f64_coords[atom_idx]
            } else {
                [
                    processed.raw_atoms.coords[3 * atom_idx] as f64,
                    processed.raw_atoms.coords[3 * atom_idx + 1] as f64,
                    processed.raw_atoms.coords[3 * atom_idx + 2] as f64,
                ]
            };
            match name {
                "N" => n_pos = Some(xyz),
                "CA" => ca_pos = Some(xyz),
                "C" => c_pos = Some(xyz),
                "O" => o_pos = Some(xyz),
                _ => {}
            }
        }

        let chain_idx = *processed.chain_indices.get(&resinfo.chain_id).unwrap_or(&0);
        bb.push(ResidueBackbone {
            res_name: resinfo.res_name.clone(),
            n: n_pos,
            ca: ca_pos,
            c: c_pos,
            o: o_pos,
            phi: 9999.0,
            psi: 9999.0,
            omega: None,
            is_cis_peptide: false,
        });
        ids.push(ResidueId {
            chain_id: resinfo.chain_id.clone(),
            res_id: resinfo.res_id,
            insertion_code: resinfo.insertion_code,
        });
        chain_map.push(chain_idx);
        skipped_before.push(pending_skip);
        pending_skip = false;
    }

    fill_dihedrals(&mut bb, &chain_map, &skipped_before, opts)?;
    Ok(ProteinBackbone { bb, ids, chain_map })
}

/// Compute phi/psi/omega per chain segment and fill into `bb`, per `opts`.
///
/// `skipped_before[i]` is `true` iff a non-protein residue (`molecule_type != 0`)
/// was skipped between `bb[i - 1]` and `bb[i]` during extraction — it is not a
/// `ProteinBackbone` field because it only matters for this one decision.
///
/// Returns `Err(ConFindError::InvalidOptions)` for the
/// `{LegacyCompact, Split}` combination: `LegacyCompact`'s dense-bridging
/// algorithm has no chain-break-aware notion of adjacency, so pairing it with
/// `Split` would either silently ignore `Split` or silently reinterpret
/// `LegacyCompact` — both are the silent substitution this crate's rules
/// forbid. Callers must pick one or the other.
fn fill_dihedrals(
    bb: &mut [ResidueBackbone],
    chain_map: &[usize],
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
        MissingAtomPolicy::PerDihedral => {
            fill_dihedrals_per_dihedral(bb, chain_map, skipped_before, opts.chain_breaks)
        }
    }

    Ok(())
}

/// `MissingAtomPolicy::LegacyCompact` (only reachable with `ChainBreakPolicy::Bridge`):
/// today's (base 316d1ab) dense-array bridging algorithm, moved verbatim.
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

        // Dense array of residues with complete N/CA/C; track original indices.
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
            // phi=None at chain N-terminus; psi=None at chain C-terminus.
            // Negate to match Mosaist's dihedral sign convention: Mosaist computes
            // dihedral(p1-p2, p3-p2) with reversed first vector vs. atan2 formula.
            bb[bb_i].phi = d.phi.map(|r| -r.to_degrees()).unwrap_or(9999.0);
            bb[bb_i].psi = d.psi.map(|r| -r.to_degrees()).unwrap_or(9999.0);
            // omega: convert radians -> degrees; None for N-terminal residue.
            // dihedral_angle_f64 returns the negative of the IUPAC angle; phi/psi
            // are negated above to match Mosaist's getPhi/getPsi. omega is NOT
            // negated, so it is the negative of Mosaist's getOmega (bits unchanged
            // from base 316d1ab; is_cis_peptide, which only tests |omega|, is
            // unaffected — see debt filed in step 8).
            bb[bb_i].omega = d.omega.map(|r| r.to_degrees());
            bb[bb_i].is_cis_peptide = bb[bb_i].omega.is_some_and(|w| w.abs() < 30.0);
        }
    }
}

/// `MissingAtomPolicy::PerDihedral` (Mosaist semantics, default): each dihedral is
/// computed independently, directly from its own four named atoms via
/// `dihedral_angle_f64` — never re-derived, never bridged past a missing
/// neighbour, a skipped non-protein residue, or (under `ChainBreakPolicy::Split`)
/// a geometric chain break.
fn fill_dihedrals_per_dihedral(
    bb: &mut [ResidueBackbone],
    chain_map: &[usize],
    skipped_before: &[bool],
    chain_break_policy: ChainBreakPolicy,
) {
    let n = bb.len();
    if n == 0 {
        return;
    }

    // Geometric chain breaks (shared predicate with check_preconditions). Only
    // consulted under Split; computed unconditionally since it's cheap and the
    // read-only borrow doesn't conflict with the per-residue writes below.
    let break_indices: HashSet<usize> = if chain_break_policy == ChainBreakPolicy::Split {
        chain_breaks(bb, chain_map)
            .into_iter()
            .map(|b| b.index)
            .collect()
    } else {
        HashSet::new()
    };

    // has_prev[i]: true iff bb[i-1] is a real, dihedral-eligible sequence
    // neighbour of bb[i] — same chain segment, no non-protein residue skipped
    // in between, and (under Split) not severed by a geometric chain break.
    let has_prev = |i: usize| -> bool {
        i > 0
            && chain_map[i] == chain_map[i - 1]
            && !skipped_before[i]
            && !break_indices.contains(&i)
    };

    for i in 0..n {
        let prev_ok = has_prev(i);
        let next_ok = i + 1 < n && has_prev(i + 1);

        let phi = if prev_ok {
            match (bb[i - 1].c, bb[i].n, bb[i].ca, bb[i].c) {
                (Some(c_prev), Some(n_i), Some(ca_i), Some(c_i)) => {
                    -dihedral_angle_f64(&c_prev, &n_i, &ca_i, &c_i).to_degrees()
                }
                _ => 9999.0,
            }
        } else {
            9999.0
        };

        let psi = if next_ok {
            match (bb[i].n, bb[i].ca, bb[i].c, bb[i + 1].n) {
                (Some(n_i), Some(ca_i), Some(c_i), Some(n_next)) => {
                    -dihedral_angle_f64(&n_i, &ca_i, &c_i, &n_next).to_degrees()
                }
                _ => 9999.0,
            }
        } else {
            9999.0
        };

        let omega = if prev_ok {
            match (bb[i - 1].ca, bb[i - 1].c, bb[i].n, bb[i].ca) {
                (Some(ca_prev), Some(c_prev), Some(n_i), Some(ca_i)) => {
                    Some(dihedral_angle_f64(&ca_prev, &c_prev, &n_i, &ca_i).to_degrees())
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
            phi: 9999.0,
            psi: 9999.0,
            omega: None,
            is_cis_peptide: false,
        }
    }

    // ---------------------------------------------------------------
    // BackboneOptions: defaults, builders, LegacyCompact+Split -> Err
    // ---------------------------------------------------------------

    #[test]
    fn backbone_options_default_is_bridge_per_dihedral() {
        assert_eq!(
            BackboneOptions::default(),
            BackboneOptions {
                chain_breaks: ChainBreakPolicy::Bridge,
                missing_atoms: MissingAtomPolicy::PerDihedral,
            }
        );
        assert_eq!(ChainBreakPolicy::default(), ChainBreakPolicy::Bridge);
        assert_eq!(MissingAtomPolicy::default(), MissingAtomPolicy::PerDihedral);
    }

    #[test]
    fn backbone_options_builders_chain() {
        let o = BackboneOptions::default()
            .with_chain_breaks(ChainBreakPolicy::Split)
            .with_missing_atoms(MissingAtomPolicy::LegacyCompact);
        assert_eq!(o.chain_breaks, ChainBreakPolicy::Split);
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
        let skipped_before = vec![false];
        let opts = BackboneOptions::default()
            .with_missing_atoms(MissingAtomPolicy::LegacyCompact)
            .with_chain_breaks(ChainBreakPolicy::Split);

        let result = fill_dihedrals(&mut bb, &chain_map, &skipped_before, &opts);
        match result {
            Err(ConFindError::InvalidOptions(_)) => {}
            other => panic!("expected Err(InvalidOptions), got {other:?}"),
        }
    }

    // ---------------------------------------------------------------
    // PerDihedral atom-level computation vs. LegacyCompact bridging
    // ---------------------------------------------------------------

    #[test]
    fn per_dihedral_atom_level_vs_legacy_bridging_missing_c() {
        // Non-planar synthetic 3-residue backbone; residue 1 (GLY) is missing C
        // (and therefore O), so it cannot appear in LegacyCompact's dense array.
        let n0 = [0.0, 0.0, 0.0];
        let ca0 = [1.46, 0.0, 0.0];
        let c0 = [2.0, 1.4, 0.0];
        let n1 = [3.3, 1.6, 0.3];
        let ca1 = [4.0, 2.9, 0.5];
        let n2 = [6.0, 4.0, 1.0];
        let ca2 = [7.5, 4.0, 1.2];
        let c2 = [8.1, 5.3, 1.5];

        let build = || {
            vec![
                mk_rb("ALA", Some(n0), Some(ca0), Some(c0)),
                mk_rb("GLY", Some(n1), Some(ca1), None),
                mk_rb("SER", Some(n2), Some(ca2), Some(c2)),
            ]
        };
        let chain_map = vec![0usize, 0, 0];
        let skipped_before = vec![false, false, false];

        // PerDihedral (default: Bridge + PerDihedral).
        let mut per = build();
        fill_dihedrals(
            &mut per,
            &chain_map,
            &skipped_before,
            &BackboneOptions::default(),
        )
        .expect("PerDihedral fill should not error");

        let exp_psi0 = -dihedral_angle_f64(&n0, &ca0, &c0, &n1).to_degrees();
        assert_eq!(per[0].phi, 9999.0, "chain-first residue: phi undefined");
        assert_eq!(
            per[0].psi, exp_psi0,
            "psi(res0) must use the immediate neighbour's N (res1), not bridge past it"
        );
        assert!(
            per[0].omega.is_none(),
            "chain-first residue: omega undefined"
        );

        assert_eq!(
            per[1].phi, 9999.0,
            "res1's own C is missing -> phi undefined"
        );
        assert_eq!(
            per[1].psi, 9999.0,
            "res1's own C is missing -> psi undefined"
        );
        let exp_omega1 = dihedral_angle_f64(&ca0, &c0, &n1, &ca1).to_degrees();
        assert_eq!(
            per[1].omega,
            Some(exp_omega1),
            "omega(res1) does not need res1's own C, only res0's CA/C and res1's N/CA"
        );

        assert_eq!(
            per[2].phi, 9999.0,
            "phi(res2) needs predecessor's C (res1.C), which is missing"
        );
        assert_eq!(per[2].psi, 9999.0, "chain-last residue: psi undefined");
        assert!(
            per[2].omega.is_none(),
            "omega(res2) needs predecessor's CA/C (res1), whose C is missing"
        );

        // LegacyCompact reproduces today's bridged values (res1 dropped from the
        // dense array entirely; res0/res2 bridge directly across the gap).
        let mut legacy = build();
        fill_dihedrals(
            &mut legacy,
            &chain_map,
            &skipped_before,
            &BackboneOptions::default().with_missing_atoms(MissingAtomPolicy::LegacyCompact),
        )
        .expect("LegacyCompact fill should not error");

        let exp_psi0_bridged = -dihedral_angle_f64(&n0, &ca0, &c0, &n2).to_degrees();
        assert_eq!(
            legacy[0].psi, exp_psi0_bridged,
            "LegacyCompact bridges res0's psi directly to res2's N, skipping res1"
        );
        let exp_phi2_bridged = -dihedral_angle_f64(&c0, &n2, &ca2, &c2).to_degrees();
        assert_eq!(
            legacy[2].phi, exp_phi2_bridged,
            "LegacyCompact bridges res2's phi directly to res0's C, skipping res1"
        );
        // res1 is excluded from the dense array entirely, so it keeps its
        // never-touched initial sentinel values.
        assert_eq!(legacy[1].phi, 9999.0);
        assert_eq!(legacy[1].psi, 9999.0);
        assert!(legacy[1].omega.is_none());
    }

    // ---------------------------------------------------------------
    // Missing CA only (N and C present): Bridge report == base; Split
    // does not suppress the residue's own missing-atom errors.
    // ---------------------------------------------------------------

    #[test]
    fn missing_ca_only_bridge_report_matches_base_split_does_not_suppress() {
        // A - B(no CA, has N and C) - C, no missing N/C anywhere else, and the
        // A.CA-C.CA carry-over distance is kept under the chain-break threshold
        // so this test isolates missing-atom behaviour from chain-break behaviour.
        let n_a: [f64; 3] = [0.0, 0.0, 0.0];
        let ca_a: [f64; 3] = [1.0, 0.0, 0.0];
        let c_a: [f64; 3] = [1.5, 1.0, 0.0];
        let n_b: [f64; 3] = [2.0, 1.2, 0.2];
        let c_b: [f64; 3] = [2.8, 2.0, 0.3];
        let n_c: [f64; 3] = [3.2, 2.2, 0.4];
        let ca_c: [f64; 3] = [3.5, 2.2, 0.4];
        let c_c: [f64; 3] = [4.0, 3.0, 0.5];

        // Sanity: A.CA-C.CA carry-over distance is within the chain-break threshold.
        let dx = ca_c[0] - ca_a[0];
        let dy = ca_c[1] - ca_a[1];
        let dz = ca_c[2] - ca_a[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        assert!(
            dist <= crate::precondition::CHAIN_BREAK_CA_ANGSTROM,
            "test fixture must not trip the chain-break threshold: dist={dist}"
        );

        let build = || {
            vec![
                mk_rb("ALA", Some(n_a), Some(ca_a), Some(c_a)),
                mk_rb("GLY", Some(n_b), None, Some(c_b)),
                mk_rb("SER", Some(n_c), Some(ca_c), Some(c_c)),
            ]
        };
        let chain_map = vec![0usize, 0, 0];
        let ids = vec![
            ResidueId {
                chain_id: "A".to_string(),
                res_id: 1,
                insertion_code: ' ',
            },
            ResidueId {
                chain_id: "A".to_string(),
                res_id: 2,
                insertion_code: ' ',
            },
            ResidueId {
                chain_id: "A".to_string(),
                res_id: 3,
                insertion_code: ' ',
            },
        ];

        let per_bridge_violations = {
            let mut bb_vec = build();
            fill_dihedrals(
                &mut bb_vec,
                &chain_map,
                &[false; 3],
                &BackboneOptions::default(),
            )
            .unwrap();
            let pb = ProteinBackbone {
                bb: bb_vec,
                ids: ids.clone(),
                chain_map: chain_map.clone(),
            };
            crate::precondition::check_preconditions(&pb).violations
        };

        // No geometric chain break in this fixture, so Split must produce the
        // identical report — it must not suppress B's real missing-atom errors.
        let per_split_violations = {
            let mut bb_vec = build();
            fill_dihedrals(
                &mut bb_vec,
                &chain_map,
                &[false; 3],
                &BackboneOptions::default().with_chain_breaks(ChainBreakPolicy::Split),
            )
            .unwrap();
            let pb = ProteinBackbone {
                bb: bb_vec,
                ids: ids.clone(),
                chain_map: chain_map.clone(),
            };
            crate::precondition::check_preconditions(&pb).violations
        };

        use crate::precondition::ViolationKind;
        let expected_kinds = [
            ViolationKind::MissingBackboneAtom { atom: "CA" },
            ViolationKind::UndefinedPhi,
            ViolationKind::UndefinedPsi,
        ];
        for (label, violations) in [
            ("Bridge", &per_bridge_violations),
            ("Split", &per_split_violations),
        ] {
            let kinds: Vec<_> = violations.iter().map(|v| v.kind.clone()).collect();
            assert_eq!(
                kinds, expected_kinds,
                "{label}: expected exactly B's own missing-CA/UndefinedPhi/UndefinedPsi errors"
            );
            assert!(
                violations.iter().all(|v| v.residue.0 == 1),
                "{label}: all violations should be attributed to residue index 1 (B)"
            );
        }
        assert_eq!(
            per_bridge_violations.len(),
            per_split_violations.len(),
            "Split must not suppress or add to B's own missing-atom errors here"
        );
    }

    // ---------------------------------------------------------------
    // skipped_before: a skipped non-protein residue severs adjacency
    // ---------------------------------------------------------------

    #[test]
    fn skipped_before_severs_per_dihedral_adjacency() {
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
        // A non-protein (e.g. ligand/HETATM) residue was skipped between res0 and res1.
        let skipped_before = vec![false, true];

        fill_dihedrals(
            &mut bb,
            &chain_map,
            &skipped_before,
            &BackboneOptions::default(),
        )
        .expect("fill should not error");

        assert_eq!(
            bb[1].phi, 9999.0,
            "phi(res1) must be undefined: its predecessor is not a real sequence neighbour"
        );
        assert!(
            bb[1].omega.is_none(),
            "omega(res1) must be undefined for the same reason"
        );
        // res0's psi is symmetrically severed too, since res1 is not a real
        // successor of res0 either.
        assert_eq!(
            bb[0].psi, 9999.0,
            "psi(res0) must be undefined: its successor is not a real sequence neighbour"
        );
    }

    // ---------------------------------------------------------------
    // Golden bit-equality against dihedral_golden_316d1ab.json
    // ---------------------------------------------------------------

    fn manifest_dir() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    }

    fn golden() -> serde_json::Value {
        let path = manifest_dir()
            .join("tests")
            .join("data")
            .join("dihedral_golden_316d1ab.json");
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("failed to read golden file {path:?}: {e}"));
        serde_json::from_str(&text)
            .unwrap_or_else(|e| panic!("failed to parse golden file {path:?}: {e}"))
    }

    fn golden_entries<'a>(
        golden: &'a serde_json::Value,
        section: &str,
        fixture: &str,
        loader: &str,
    ) -> &'a Vec<serde_json::Value> {
        golden[section][fixture][loader]
            .as_array()
            .unwrap_or_else(|| {
                panic!("golden file missing {section}.{fixture}.{loader} (or not an array)")
            })
    }

    fn assert_bb_matches_golden(
        entries: &[serde_json::Value],
        bb: &ProteinBackbone,
        context: &str,
    ) {
        assert_eq!(
            entries.len(),
            bb.bb.len(),
            "{context}: residue count mismatch (golden={}, computed={})",
            entries.len(),
            bb.bb.len()
        );
        for (i, (entry, (rb, id))) in entries
            .iter()
            .zip(bb.bb.iter().zip(bb.ids.iter()))
            .enumerate()
        {
            let chain = entry["chain"].as_str().expect("golden entry missing chain");
            let res_id = entry["res_id"]
                .as_i64()
                .expect("golden entry missing res_id") as i32;
            assert_eq!(chain, id.chain_id, "{context}: residue {i} chain mismatch");
            assert_eq!(res_id, id.res_id, "{context}: residue {i} res_id mismatch");

            let phi_bits = entry["phi_bits"]
                .as_u64()
                .expect("golden entry missing phi_bits");
            assert_eq!(
                phi_bits,
                rb.phi.to_bits(),
                "{context}: residue {i} ({chain}{res_id}) phi bits mismatch"
            );

            let psi_bits = entry["psi_bits"]
                .as_u64()
                .expect("golden entry missing psi_bits");
            assert_eq!(
                psi_bits,
                rb.psi.to_bits(),
                "{context}: residue {i} ({chain}{res_id}) psi bits mismatch"
            );

            match &entry["omega_bits"] {
                serde_json::Value::String(s) if s == "NONE" => {
                    assert!(
                        rb.omega.is_none(),
                        "{context}: residue {i} ({chain}{res_id}) expected omega None, got {:?}",
                        rb.omega
                    );
                }
                serde_json::Value::Number(num) => {
                    let bits = num.as_u64().expect("omega_bits number not u64");
                    let omega = rb.omega.unwrap_or_else(|| {
                        panic!("{context}: residue {i} ({chain}{res_id}) expected omega Some, got None")
                    });
                    assert_eq!(
                        bits,
                        omega.to_bits(),
                        "{context}: residue {i} ({chain}{res_id}) omega bits mismatch"
                    );
                }
                other => panic!("{context}: unexpected omega_bits value: {other:?}"),
            }

            let is_cis = entry["is_cis"]
                .as_bool()
                .expect("golden entry missing is_cis");
            assert_eq!(
                is_cis, rb.is_cis_peptide,
                "{context}: residue {i} ({chain}{res_id}) is_cis mismatch"
            );
        }
    }

    fn committed_fixture_path(name: &str) -> std::path::PathBuf {
        let md = manifest_dir();
        match name {
            "1crn.pdb" => md
                .join("..")
                .join("..")
                .join("tests")
                .join("data")
                .join("1crn.pdb"),
            "ubiquitin_1ubq_A.pdb" => md
                .join("..")
                .join("proxide-tmalign")
                .join("tests")
                .join("data")
                .join("ubiquitin_1ubq_A.pdb"),
            "missing_atoms.pdb" => md
                .join("..")
                .join("proxide_fixer")
                .join("tests")
                .join("data")
                .join("missing_atoms.pdb"),
            "chain_break.pdb" => md
                .join("..")
                .join("proxide_fixer")
                .join("tests")
                .join("data")
                .join("chain_break.pdb"),
            other => panic!("unknown committed fixture: {other}"),
        }
    }

    fn load_processed_for(path: &std::path::Path) -> ProcessedStructure {
        let (raw, _models) =
            proxide_io::formats::pdb::parse_pdb_file(path).expect("parse_pdb_file failed");
        ProcessedStructure::from_raw(raw).expect("ProcessedStructure::from_raw failed")
    }

    #[test]
    fn golden_dihedral_bit_equality_committed_fixtures() {
        let golden = golden();
        let fixtures = [
            "1crn.pdb",
            "ubiquitin_1ubq_A.pdb",
            "missing_atoms.pdb",
            "chain_break.pdb",
        ];

        for name in fixtures {
            let path = committed_fixture_path(name);
            assert!(path.exists(), "committed fixture missing: {path:?}");
            let processed = load_processed_for(&path);

            // {Bridge, LegacyCompact} via the extract path must equal golden for
            // ALL fixtures, including ones with missing atoms / chain breaks.
            let legacy_opts =
                BackboneOptions::default().with_missing_atoms(MissingAtomPolicy::LegacyCompact);
            let legacy_bb = extract_f64_backbone_with_options(&processed, &legacy_opts)
                .unwrap_or_else(|e| panic!("LegacyCompact extract failed for {name}: {e}"));
            assert_bb_matches_golden(
                golden_entries(&golden, "fixtures", name, "extract_f64_backbone"),
                &legacy_bb,
                &format!("{name} LegacyCompact/extract_f64_backbone"),
            );

            // Determine cleanliness (zero ChainBreak, zero missing backbone atoms)
            // directly from the structure, and ASSERT it before relying on it, so a
            // future dirty fixture can't silently skip the PerDihedral golden check.
            let default_bb = extract_f64_backbone(&processed)
                .unwrap_or_else(|e| panic!("default extract failed for {name}: {e}"));
            let all_atoms_present = default_bb
                .bb
                .iter()
                .all(|rb| rb.n.is_some() && rb.ca.is_some() && rb.c.is_some());
            let breaks = chain_breaks(&default_bb.bb, &default_bb.chain_map);
            let is_clean = all_atoms_present && breaks.is_empty();

            let expected_clean = matches!(name, "1crn.pdb" | "ubiquitin_1ubq_A.pdb");
            assert_eq!(
                is_clean, expected_clean,
                "{name}: cleanliness (all atoms present, zero chain breaks) changed unexpectedly \
                 (all_atoms_present={all_atoms_present}, breaks={breaks:?})"
            );

            if !is_clean {
                continue;
            }

            assert_bb_matches_golden(
                golden_entries(&golden, "fixtures", name, "extract_f64_backbone"),
                &default_bb,
                &format!("{name} Bridge+PerDihedral/extract_f64_backbone"),
            );

            let split_bb = extract_f64_backbone_with_options(
                &processed,
                &BackboneOptions::default().with_chain_breaks(ChainBreakPolicy::Split),
            )
            .unwrap_or_else(|e| panic!("Split extract failed for {name}: {e}"));
            assert_bb_matches_golden(
                golden_entries(&golden, "fixtures", name, "extract_f64_backbone"),
                &split_bb,
                &format!("{name} Split+PerDihedral/extract_f64_backbone"),
            );

            let default_loadpb = load_pdb_f64_with_options(&path, &BackboneOptions::default())
                .unwrap_or_else(|e| panic!("default load_pdb_f64 failed for {name}: {e}"));
            assert_bb_matches_golden(
                golden_entries(&golden, "fixtures", name, "load_pdb_f64"),
                &default_loadpb,
                &format!("{name} Bridge+PerDihedral/load_pdb_f64"),
            );

            let split_loadpb = load_pdb_f64_with_options(
                &path,
                &BackboneOptions::default().with_chain_breaks(ChainBreakPolicy::Split),
            )
            .unwrap_or_else(|e| panic!("Split load_pdb_f64 failed for {name}: {e}"));
            assert_bb_matches_golden(
                golden_entries(&golden, "fixtures", name, "load_pdb_f64"),
                &split_loadpb,
                &format!("{name} Split+PerDihedral/load_pdb_f64"),
            );
        }
    }

    #[test]
    #[ignore = "needs /home/marielle/repos/mosaist/testfiles"]
    fn golden_dihedral_bit_equality_mosaist_fixtures() {
        let fixtures: &[(&str, &str)] = &[
            (
                "1DC7.pdb",
                "/home/marielle/repos/mosaist/testfiles/1DC7.pdb",
            ),
            (
                "small.pdb",
                "/home/marielle/repos/mosaist/testfiles/small.pdb",
            ),
        ];
        for (name, path_str) in fixtures {
            let path = std::path::PathBuf::from(path_str);
            if !path.exists() {
                panic!(
                    "mosaist fixture not found: {path:?} (needs /home/marielle/repos/mosaist/testfiles)"
                );
            }
            let golden = golden();
            let processed = load_processed_for(&path);

            let default_bb = extract_f64_backbone(&processed)
                .unwrap_or_else(|e| panic!("default extract failed for {name}: {e}"));
            assert_bb_matches_golden(
                golden_entries(&golden, "mosaist_fixtures", name, "extract_f64_backbone"),
                &default_bb,
                &format!("{name} extract_f64_backbone (mosaist)"),
            );

            let default_loadpb = load_pdb_f64(&path)
                .unwrap_or_else(|e| panic!("default load_pdb_f64 failed for {name}: {e}"));
            assert_bb_matches_golden(
                golden_entries(&golden, "mosaist_fixtures", name, "load_pdb_f64"),
                &default_loadpb,
                &format!("{name} load_pdb_f64 (mosaist)"),
            );
        }
    }
}
