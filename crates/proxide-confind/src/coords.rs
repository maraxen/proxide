use crate::error::ConFindError;
use proxide_core::processing::residues::{ProcessedStructure, ResidueId};
use proxide_geometry::geometry::angles::compute_backbone_dihedrals_f64;
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

/// Backbone geometry for a single protein residue in f64 precision.
///
/// Atom positions are `None` when the atom is absent from the PDB.
/// Dihedral angles use the sentinel `9999.0` when the angle cannot be
/// computed (terminal residues or missing predecessors/successors).
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
    /// φ dihedral angle in degrees; `9999.0` if terminal or missing.
    pub phi: f64,
    /// ψ dihedral angle in degrees; `9999.0` if terminal or missing.
    pub psi: f64,
    /// ω dihedral angle in degrees; `None` for the N-terminal residue of each chain or
    /// when the preceding Cα/C atoms are absent.
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

/// Extract backbone from ProcessedStructure, widening f32 -> f64.
/// Single f32->f64 boundary; no f32 appears in proxide-confind past this point.
pub fn extract_f64_backbone(s: &ProcessedStructure) -> Result<ProteinBackbone, ConFindError> {
    let mut bb: Vec<ResidueBackbone> = Vec::new();
    let mut ids: Vec<ResidueId> = Vec::new();
    let mut chain_map: Vec<usize> = Vec::new();

    for resinfo in &s.residue_info {
        if s.molecule_type[resinfo.start_atom] != 0 {
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
    }

    fill_dihedrals(&mut bb, &chain_map);
    Ok(ProteinBackbone { bb, ids, chain_map })
}

/// Parse a PDB re-reading cols 30-54 as f64 to recover full text precision.
pub fn load_pdb_f64<P: AsRef<Path>>(path: P) -> Result<ProteinBackbone, ConFindError> {
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
    let processed = ProcessedStructure::from_raw(raw)
        .map_err(std::io::Error::other)?;

    let mut bb: Vec<ResidueBackbone> = Vec::new();
    let mut ids: Vec<ResidueId> = Vec::new();
    let mut chain_map: Vec<usize> = Vec::new();

    for resinfo in &processed.residue_info {
        if processed.molecule_type[resinfo.start_atom] != 0 {
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
    }

    fill_dihedrals(&mut bb, &chain_map);
    Ok(ProteinBackbone { bb, ids, chain_map })
}

/// Compute phi/psi/omega per chain segment and fill into `bb`.
fn fill_dihedrals(bb: &mut [ResidueBackbone], chain_map: &[usize]) {
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
            // No sign negation: omega is measured on the preceding peptide bond
            // and the Mosaist sign convention does not apply to omega.
            bb[bb_i].omega = d.omega.map(|r| r.to_degrees());
            bb[bb_i].is_cis_peptide = bb[bb_i].omega.is_some_and(|w| w.abs() < 30.0);
        }
    }
}
