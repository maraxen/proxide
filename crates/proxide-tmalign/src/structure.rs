//! Cα trace + sequence extraction for TM-align input structures.
//!
//! Built on [`proxide_core::processing::residues::ProcessedStructure`], not
//! a standalone PDB parser — mirrors the pattern in
//! `proxide-confind/src/coords.rs` (a crate-local backbone wrapper over the
//! shared structure-processing core) rather than consuming
//! `ProcessedStructure::extract_ca_coords` directly: that helper returns a
//! NaN sentinel for unresolved Cα atoms with no paired sequence, whereas
//! TM-align needs residues with no resolved Cα excluded outright (matching
//! the reference implementation's own residue-exclusion behavior), plus a
//! one-letter sequence index-aligned with the coordinates.

use crate::error::TmAlignError;
use crate::seq::three_to_one;
use nalgebra::Vector3;
use proxide_core::processing::residues::{ProcessedStructure, ResidueId};
use std::path::Path;

/// Cα coordinates + one-letter sequence + residue identifiers for a single
/// protein chain (or concatenated set of chains), index-aligned across all
/// three fields. Residues with no resolved Cα atom are excluded at
/// construction time.
#[derive(Debug, Clone)]
pub struct CaTrace {
    /// Cα positions, world frame, Å.
    pub coords: Vec<Vector3<f32>>,
    /// One-letter amino acid codes, ASCII, same length as `coords`.
    pub seq: Vec<u8>,
    /// Residue identifiers, for reporting alignment output against the
    /// original structure's chain/residue numbering.
    pub res_ids: Vec<ResidueId>,
}

impl CaTrace {
    /// Number of residues (with a resolved Cα) in this trace.
    pub fn len(&self) -> usize {
        self.coords.len()
    }

    pub fn is_empty(&self) -> bool {
        self.coords.is_empty()
    }
}

/// Extract a [`CaTrace`] from an already-parsed [`ProcessedStructure`].
///
/// Protein residues only (`molecule_type == 0`, matching
/// `proxide-confind`'s convention); residues without a resolved Cα atom are
/// skipped rather than represented with a sentinel value.
pub fn extract_ca_trace(s: &ProcessedStructure) -> Result<CaTrace, TmAlignError> {
    let mut coords = Vec::new();
    let mut seq = Vec::new();
    let mut res_ids = Vec::new();

    for resinfo in &s.residue_info {
        if s.molecule_type[resinfo.start_atom] != 0 {
            continue;
        }

        let mut ca_pos: Option<[f32; 3]> = None;
        for atom_idx in resinfo.start_atom..(resinfo.start_atom + resinfo.num_atoms) {
            if s.raw_atoms.atom_names[atom_idx] == "CA" {
                ca_pos = Some([
                    s.raw_atoms.coords[3 * atom_idx],
                    s.raw_atoms.coords[3 * atom_idx + 1],
                    s.raw_atoms.coords[3 * atom_idx + 2],
                ]);
                break;
            }
        }

        let Some(ca) = ca_pos else { continue };
        coords.push(Vector3::new(ca[0], ca[1], ca[2]));
        seq.push(three_to_one(&resinfo.res_name) as u8);
        res_ids.push(ResidueId {
            chain_id: resinfo.chain_id.clone(),
            res_id: resinfo.res_id,
            insertion_code: resinfo.insertion_code,
        });
    }

    if coords.is_empty() {
        return Err(TmAlignError::EmptyStructure);
    }

    Ok(CaTrace {
        coords,
        seq,
        res_ids,
    })
}

/// Parse a PDB file and extract its Cα trace directly.
pub fn load_pdb_ca_trace<P: AsRef<Path>>(path: P) -> Result<CaTrace, TmAlignError> {
    let (raw, _) = proxide_io::formats::pdb::parse_pdb_file(path.as_ref())
        .map_err(|e| TmAlignError::Parse(e.to_string()))?;
    let processed = ProcessedStructure::from_raw(raw).map_err(TmAlignError::Parse)?;
    extract_ca_trace(&processed)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Locate the local `~/repos/USalign` reference clone's bundled sample
    /// PDBs, if present. Skips (not fails) when absent — following the
    /// env-gated skip-if-absent convention in
    /// `proxide-confind/tests/test_parity_1dc7.rs`.
    fn usalign_sample(name: &str) -> Option<std::path::PathBuf> {
        let base = std::env::var("USALIGN_REPO")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| {
                let home = std::env::var("HOME").unwrap_or_default();
                std::path::PathBuf::from(home).join("repos/USalign")
            });
        let path = base.join(name);
        path.exists().then_some(path)
    }

    #[test]
    fn load_pdb1_from_usalign_reference_produces_matching_length() {
        let Some(path) = usalign_sample("PDB1.pdb") else {
            return;
        };
        let trace = load_pdb_ca_trace(&path).expect("PDB1.pdb should parse");
        // Reference TMalign output (this session): "Length of Structure_1: 250 residues"
        assert_eq!(trace.len(), 250);
        assert_eq!(trace.seq.len(), 250);
        assert_eq!(trace.res_ids.len(), 250);
    }

    #[test]
    fn load_pdb2_from_usalign_reference_produces_matching_length() {
        let Some(path) = usalign_sample("PDB2.pdb") else {
            return;
        };
        let trace = load_pdb_ca_trace(&path).expect("PDB2.pdb should parse");
        // Reference TMalign output (this session): "Length of Structure_2: 166 residues"
        assert_eq!(trace.len(), 166);
    }
}
