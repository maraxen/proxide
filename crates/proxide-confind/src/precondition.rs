use crate::coords::{ProteinBackbone, ResidueBackbone, ResidueIndex};
use crate::error::ConFindError;
use proxide_core::processing::residues::ResidueId;

/// Precondition violation recorded during structure check.
#[derive(Debug, Clone, PartialEq)]
pub struct PreconditionViolation {
    pub residue: ResidueIndex,
    pub id: ResidueId,
    pub res_name: String,
    pub kind: ViolationKind,
}

/// Categorization of each violation type.
#[derive(Debug, Clone, PartialEq)]
pub enum ViolationKind {
    /// Backbone atom (N, CA, or C) is missing.
    MissingBackboneAtom { atom: &'static str },
    /// φ dihedral is None for a non-terminal residue (Debt #1897).
    UndefinedPhi,
    /// ψ dihedral is None for a non-terminal residue (Debt #1897).
    UndefinedPsi,
    /// Residue name not in canonical set (20 standard AAs + variants).
    UnknownResidueType { res_name: String },
    /// Consecutive same-chain residues with CA atoms separated by more than
    /// [`CHAIN_BREAK_CA_ANGSTROM`].
    ChainBreak { gap_to_next_ca: f64 },
}

/// Severity level for a violation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    /// Error: structural integrity compromised; ConFind cannot run.
    Error,
    /// Warning: minor structural irregularity; ConFind may still run.
    Warning,
}

/// Result of precondition checking on a ProteinBackbone.
#[derive(Debug, Clone)]
pub struct PreconditionReport {
    pub violations: Vec<PreconditionViolation>,
}

impl PreconditionReport {
    /// Returns true if no violations were found.
    pub fn is_clean(&self) -> bool {
        self.violations.is_empty()
    }

    /// Returns an iterator over Error-severity violations.
    pub fn errors(&self) -> impl Iterator<Item = &PreconditionViolation> {
        self.violations
            .iter()
            .filter(|v| PreconditionReport::severity_of(&v.kind) == Severity::Error)
    }

    /// Returns an iterator over Warning-severity violations.
    pub fn warnings(&self) -> impl Iterator<Item = &PreconditionViolation> {
        self.violations
            .iter()
            .filter(|v| PreconditionReport::severity_of(&v.kind) == Severity::Warning)
    }

    /// Classify the severity of a violation kind.
    pub fn severity_of(kind: &ViolationKind) -> Severity {
        match kind {
            ViolationKind::MissingBackboneAtom { .. }
            | ViolationKind::UndefinedPhi
            | ViolationKind::UndefinedPsi => Severity::Error,
            ViolationKind::UnknownResidueType { .. } | ViolationKind::ChainBreak { .. } => {
                Severity::Warning
            }
        }
    }
}

/// Canonical amino acids (20 standard + common variants).
pub const CANONICAL_AA_NAMES: &[&str] = &[
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "HID", "HIE", "HIP", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "CYX",
];

/// CA–CA distance threshold (Å) for chain break detection.
pub const CHAIN_BREAK_CA_ANGSTROM: f64 = 4.5;

/// A geometric chain break detected by [`chain_breaks`]: the CA–CA distance
/// between `bb[prev_with_ca]` and `bb[index]` (the nearest preceding and
/// following residues that both have a CA atom, within the same chain
/// segment) exceeds [`CHAIN_BREAK_CA_ANGSTROM`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChainBreakAt {
    /// Global index of the residue immediately after the break.
    pub index: usize,
    /// Global index of the nearest preceding residue that has a CA atom.
    pub prev_with_ca: usize,
    /// CA–CA distance (Å) between `bb[prev_with_ca]` and `bb[index]`.
    pub gap: f64,
}

/// Detect geometric chain breaks: consecutive same-chain-segment residues
/// whose CA atoms are more than [`CHAIN_BREAK_CA_ANGSTROM`] Å apart.
pub fn chain_breaks(bb: &[ResidueBackbone], chain_map: &[usize]) -> Vec<ChainBreakAt> {
    let mut breaks = Vec::new();
    let n = bb.len();
    if n == 0 {
        return breaks;
    }

    let mut chain_starts: Vec<usize> = vec![0];
    for i in 1..n {
        if chain_map[i] != chain_map[i - 1] {
            chain_starts.push(i);
        }
    }
    chain_starts.push(n);

    for w in chain_starts.windows(2) {
        let seg_start = w[0];
        let seg_end = w[1];

        let mut last_ca: Option<(usize, [f64; 3])> = None;

        for (offset, rb) in bb[seg_start..seg_end].iter().enumerate() {
            let i = seg_start + offset;
            if let Some(ca) = rb.ca {
                if let Some((prev_i, prev_ca)) = last_ca {
                    let dx = ca[0] - prev_ca[0];
                    let dy = ca[1] - prev_ca[1];
                    let dz = ca[2] - prev_ca[2];
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt();

                    if dist > CHAIN_BREAK_CA_ANGSTROM {
                        breaks.push(ChainBreakAt {
                            index: i,
                            prev_with_ca: prev_i,
                            gap: dist,
                        });
                    }
                }
                last_ca = Some((i, ca));
            }
        }
    }

    breaks
}

/// Check preconditions on a ProteinBackbone without modifying it.
///
/// Performs a single pass over all residues, recording violations:
/// - Missing N/CA/C atoms → MissingBackboneAtom (Error)
/// - φ undefined (None) for non-first-in-chain → UndefinedPhi (Error)
/// - ψ undefined (None) for non-last-in-chain → UndefinedPsi (Error)
/// - Residue name not canonical → UnknownResidueType (Warning)
/// - Consecutive same-chain CA atoms > 4.5 Å apart → ChainBreak (Warning)
pub fn check_preconditions(bb: &ProteinBackbone) -> PreconditionReport {
    let mut violations = Vec::new();
    let n = bb.bb.len();

    if n == 0 {
        return PreconditionReport { violations };
    }

    // Identify chain segment boundaries using chain_map.
    let mut chain_starts: Vec<usize> = vec![0];
    for i in 1..n {
        if bb.chain_map[i] != bb.chain_map[i - 1] {
            chain_starts.push(i);
        }
    }
    chain_starts.push(n);

    let breaks = chain_breaks(&bb.bb, &bb.chain_map);
    let break_indices: std::collections::HashSet<usize> = breaks.iter().map(|b| b.index).collect();

    // (a) Check for missing backbone atoms and unknown residue types.
    for (i, rb) in bb.bb.iter().enumerate() {
        let res_idx = ResidueIndex(i as u32);
        let res_id = &bb.ids[i];

        if rb.n.is_none() {
            violations.push(PreconditionViolation {
                residue: res_idx,
                id: res_id.clone(),
                res_name: rb.res_name.clone(),
                kind: ViolationKind::MissingBackboneAtom { atom: "N" },
            });
            log::error!(
                "Residue {}/{}/{}{} missing backbone atom N",
                res_id.chain_id,
                res_idx.0,
                res_id.res_id,
                res_id.insertion_code
            );
        }

        if rb.ca.is_none() {
            violations.push(PreconditionViolation {
                residue: res_idx,
                id: res_id.clone(),
                res_name: rb.res_name.clone(),
                kind: ViolationKind::MissingBackboneAtom { atom: "CA" },
            });
            log::error!(
                "Residue {}/{}/{}{} missing backbone atom CA",
                res_id.chain_id,
                res_idx.0,
                res_id.res_id,
                res_id.insertion_code
            );
        }

        if rb.c.is_none() {
            violations.push(PreconditionViolation {
                residue: res_idx,
                id: res_id.clone(),
                res_name: rb.res_name.clone(),
                kind: ViolationKind::MissingBackboneAtom { atom: "C" },
            });
            log::error!(
                "Residue {}/{}/{}{} missing backbone atom C",
                res_id.chain_id,
                res_idx.0,
                res_id.res_id,
                res_id.insertion_code
            );
        }

        if !CANONICAL_AA_NAMES.contains(&rb.res_name.as_str()) {
            violations.push(PreconditionViolation {
                residue: res_idx,
                id: res_id.clone(),
                res_name: rb.res_name.clone(),
                kind: ViolationKind::UnknownResidueType {
                    res_name: rb.res_name.clone(),
                },
            });
            log::warn!(
                "Residue {}/{}/{}{} has unknown residue type '{}'",
                res_id.chain_id,
                res_idx.0,
                res_id.res_id,
                res_id.insertion_code,
                rb.res_name
            );
        }
    }

    // (b) Check φ/ψ for chain-terminal rules using Option checks.
    for w in chain_starts.windows(2) {
        let seg_start = w[0];
        let seg_end = w[1];

        for (local_idx, global_idx) in (seg_start..seg_end).enumerate() {
            let res_idx = ResidueIndex(global_idx as u32);
            let res_id = &bb.ids[global_idx];
            let rb = &bb.bb[global_idx];

            let is_chain_first = local_idx == 0;
            let phi_atoms_present = global_idx > 0
                && bb.bb[global_idx - 1].c.is_some()
                && rb.n.is_some()
                && rb.ca.is_some()
                && rb.c.is_some();
            let phi_split_exempt = break_indices.contains(&global_idx) && phi_atoms_present;
            if !is_chain_first && !phi_split_exempt && rb.phi.is_none() {
                violations.push(PreconditionViolation {
                    residue: res_idx,
                    id: res_id.clone(),
                    res_name: rb.res_name.clone(),
                    kind: ViolationKind::UndefinedPhi,
                });
                log::error!(
                    "Residue {}/{}/{}{} has undefined φ despite being mid-chain",
                    res_id.chain_id,
                    res_idx.0,
                    res_id.res_id,
                    res_id.insertion_code
                );
            }

            let is_chain_last = local_idx == (seg_end - seg_start - 1);
            let psi_atoms_present = global_idx + 1 < n
                && rb.n.is_some()
                && rb.ca.is_some()
                && rb.c.is_some()
                && bb.bb[global_idx + 1].n.is_some();
            let psi_split_exempt = break_indices.contains(&(global_idx + 1)) && psi_atoms_present;
            if !is_chain_last && !psi_split_exempt && rb.psi.is_none() {
                violations.push(PreconditionViolation {
                    residue: res_idx,
                    id: res_id.clone(),
                    res_name: rb.res_name.clone(),
                    kind: ViolationKind::UndefinedPsi,
                });
                log::error!(
                    "Residue {}/{}/{}{} has undefined ψ despite being mid-chain",
                    res_id.chain_id,
                    res_idx.0,
                    res_id.res_id,
                    res_id.insertion_code
                );
            }
        }
    }

    // (c) Chain breaks: consecutive same-chain residues with CA atoms further apart than 4.5 Å
    for b in &breaks {
        let rb = &bb.bb[b.index];
        let prev_res_idx = ResidueIndex(b.prev_with_ca as u32);
        let prev_res_id = &bb.ids[b.prev_with_ca];
        violations.push(PreconditionViolation {
            residue: ResidueIndex(b.index as u32),
            id: bb.ids[b.index].clone(),
            res_name: rb.res_name.clone(),
            kind: ViolationKind::ChainBreak {
                gap_to_next_ca: b.gap,
            },
        });
        log::warn!(
            "Chain break detected: {}/{}/{}{} to {}/{}/{}{} gap = {:.2} Å (threshold {:.1} Å)",
            prev_res_id.chain_id,
            prev_res_idx.0,
            prev_res_id.res_id,
            prev_res_id.insertion_code,
            bb.ids[b.index].chain_id,
            b.index as u32,
            bb.ids[b.index].res_id,
            bb.ids[b.index].insertion_code,
            b.gap,
            CHAIN_BREAK_CA_ANGSTROM
        );
    }

    PreconditionReport { violations }
}

/// Check preconditions and return an error if any Error-severity violations exist.
pub fn require_preconditions(bb: &ProteinBackbone) -> Result<(), ConFindError> {
    let report = check_preconditions(bb);
    let error_count = report.errors().count();
    if error_count > 0 {
        Err(ConFindError::PreconditionsFailed(error_count))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coords::ResidueBackbone;

    fn make_residue(
        name: &str,
        n: Option<[f64; 3]>,
        ca: Option<[f64; 3]>,
        c: Option<[f64; 3]>,
        phi: Option<f64>,
        psi: Option<f64>,
    ) -> (ResidueBackbone, ResidueId) {
        let rb = ResidueBackbone {
            res_name: name.to_string(),
            n,
            ca,
            c,
            o: None,
            phi,
            psi,
            omega: None,
            is_cis_peptide: false,
        };
        let id = ResidueId {
            chain_id: "A".to_string(),
            res_id: 1,
            insertion_code: ' ',
        };
        (rb, id)
    }

    #[test]
    fn test_clean_backbone() {
        let (rb, id) = make_residue(
            "ALA",
            Some([0.0, 0.0, 0.0]),
            Some([1.0, 0.0, 0.0]),
            Some([2.0, 0.0, 0.0]),
            None, // N-terminal: phi is None
            Some(-100.0),
        );

        let bb = ProteinBackbone {
            bb: vec![rb],
            ids: vec![id],
            chain_map: vec![0],
        };

        let report = check_preconditions(&bb);
        assert!(report.is_clean());
    }

    #[test]
    fn test_missing_backbone_atom_c() {
        let (mut rb, id) = make_residue(
            "ALA",
            Some([0.0, 0.0, 0.0]),
            Some([1.0, 0.0, 0.0]),
            None, // Missing C
            None,
            Some(-100.0),
        );
        rb.phi = None;
        rb.psi = Some(-100.0);

        let bb = ProteinBackbone {
            bb: vec![rb],
            ids: vec![id],
            chain_map: vec![0],
        };

        let report = check_preconditions(&bb);
        assert!(!report.is_clean());

        let errors: Vec<_> = report.errors().collect();
        assert_eq!(errors.len(), 1);
        assert_eq!(
            errors[0].kind,
            ViolationKind::MissingBackboneAtom { atom: "C" }
        );
    }

    #[test]
    fn test_undefined_phi_mid_chain_multi_residue() {
        let (rb1, mut id1) = make_residue(
            "ALA",
            Some([0.0, 0.0, 0.0]),
            Some([1.0, 0.0, 0.0]),
            Some([2.0, 0.0, 0.0]),
            None, // OK for chain first
            Some(-80.0),
        );
        id1.res_id = 1;

        let (rb2, mut id2) = make_residue(
            "GLY",
            Some([3.0, 0.0, 0.0]),
            Some([4.0, 0.0, 0.0]),
            Some([5.0, 0.0, 0.0]),
            None, // ERROR: mid-chain φ is None
            Some(-60.0),
        );
        id2.res_id = 2;

        let bb = ProteinBackbone {
            bb: vec![rb1, rb2],
            ids: vec![id1, id2],
            chain_map: vec![0, 0],
        };

        let report = check_preconditions(&bb);
        let errors: Vec<_> = report.errors().collect();
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ViolationKind::UndefinedPhi);
    }

    #[test]
    fn test_require_preconditions_error() {
        let (rb, id) = make_residue(
            "ALA",
            Some([0.0, 0.0, 0.0]),
            None, // Missing CA
            Some([2.0, 0.0, 0.0]),
            None,
            Some(-100.0),
        );

        let bb = ProteinBackbone {
            bb: vec![rb],
            ids: vec![id],
            chain_map: vec![0],
        };

        let result = require_preconditions(&bb);
        match result {
            Err(ConFindError::PreconditionsFailed(n)) => {
                assert_eq!(n, 1);
            }
            _ => panic!("Expected PreconditionsFailed"),
        }
    }

    #[test]
    fn test_chain_break_warning() {
        let (rb1, mut id1) = make_residue(
            "ALA",
            Some([0.0, 0.0, 0.0]),
            Some([1.0, 0.0, 0.0]),
            Some([2.0, 0.0, 0.0]),
            None,
            Some(-80.0),
        );
        id1.res_id = 1;

        let (rb2, mut id2) = make_residue(
            "GLY",
            Some([8.0, 0.0, 0.0]),
            Some([9.0, 0.0, 0.0]),
            Some([10.0, 0.0, 0.0]),
            Some(-100.0),
            None,
        );
        id2.res_id = 2;

        let bb = ProteinBackbone {
            bb: vec![rb1, rb2],
            ids: vec![id1, id2],
            chain_map: vec![0, 0],
        };

        let report = check_preconditions(&bb);
        let warnings: Vec<_> = report.warnings().collect();
        assert_eq!(warnings.len(), 1);
        assert_eq!(report.errors().count(), 0);
        assert!(require_preconditions(&bb).is_ok());
    }

    #[test]
    fn test_canonical_variants_accepted() {
        for variant in &["HID", "HIE", "HIP", "CYX"] {
            let (rb, id) = make_residue(
                variant,
                Some([0.0, 0.0, 0.0]),
                Some([1.0, 0.0, 0.0]),
                Some([2.0, 0.0, 0.0]),
                None,
                Some(-100.0),
            );

            let bb = ProteinBackbone {
                bb: vec![rb],
                ids: vec![id],
                chain_map: vec![0],
            };

            let report = check_preconditions(&bb);
            assert_eq!(report.warnings().count(), 0, "Variant {} should be canonical", variant);
        }
    }
}
