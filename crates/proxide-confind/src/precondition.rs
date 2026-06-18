use crate::coords::{ProteinBackbone, ResidueIndex};
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
    /// φ dihedral is undefined (sentinel 9999.0) for a non-terminal residue.
    UndefinedPhi,
    /// ψ dihedral is undefined (sentinel 9999.0) for a non-terminal residue.
    UndefinedPsi,
    /// Residue name not in canonical set (20 standard AAs + variants).
    UnknownResidueType { res_name: String },
    /// Consecutive same-chain residues with CA atoms separated by > 4.5 Å.
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
const CANONICAL_AA_NAMES: &[&str] = &[
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "HID", "HIE", "HIP", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "CYX",
];

/// CA–CA distance threshold (Å) for chain break detection.
const CHAIN_BREAK_CA_ANGSTROM: f64 = 4.5;

/// Check preconditions on a ProteinBackbone without modifying it.
///
/// Performs a single pass over all residues, recording violations:
/// - Missing N/CA/C atoms → MissingBackboneAtom (Error)
/// - φ undefined for non-first-in-chain → UndefinedPhi (Error)
/// - ψ undefined for non-last-in-chain → UndefinedPsi (Error)
/// - Residue name not canonical → UnknownResidueType (Warning)
/// - Consecutive same-chain CA atoms > 4.5 Å apart → ChainBreak (Warning)
///
/// Logs each violation via `log::error!` or `log::warn!` with chain/residue ID,
/// insertion code, and a suggested fix.
pub fn check_preconditions(bb: &ProteinBackbone) -> PreconditionReport {
    let mut violations = Vec::new();
    let n = bb.bb.len();

    if n == 0 {
        return PreconditionReport { violations };
    }

    // Identify chain segment boundaries using chain_map (same logic as fill_dihedrals).
    let mut chain_starts: Vec<usize> = vec![0];
    for i in 1..n {
        if bb.chain_map[i] != bb.chain_map[i - 1] {
            chain_starts.push(i);
        }
    }
    chain_starts.push(n);

    // (a) Check for missing backbone atoms and unknown residue types.
    for (i, rb) in bb.bb.iter().enumerate() {
        let res_idx = ResidueIndex(i as u32);
        let res_id = &bb.ids[i];

        // Check for missing N, CA, C atoms.
        if rb.n.is_none() {
            violations.push(PreconditionViolation {
                residue: res_idx,
                id: res_id.clone(),
                res_name: rb.res_name.clone(),
                kind: ViolationKind::MissingBackboneAtom { atom: "N" },
            });
            log::error!(
                "Residue {}/{}/{}{} missing backbone atom N; fix: re-run PDB parsing or check input file",
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
                "Residue {}/{}/{}{} missing backbone atom CA; fix: re-run PDB parsing or check input file",
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
                "Residue {}/{}/{}{} missing backbone atom C; fix: re-run PDB parsing or check input file",
                res_id.chain_id,
                res_idx.0,
                res_id.res_id,
                res_id.insertion_code
            );
        }

        // (c) Check for unknown residue types.
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
                "Residue {}/{}/{}{} has unknown residue type '{}'; fix: verify residue name or exclude this residue",
                res_id.chain_id,
                res_idx.0,
                res_id.res_id,
                res_id.insertion_code,
                rb.res_name
            );
        }
    }

    // (b) Check φ/ψ for chain-terminal rules.
    for w in chain_starts.windows(2) {
        let seg_start = w[0];
        let seg_end = w[1];

        for (local_idx, global_idx) in (seg_start..seg_end).enumerate() {
            let res_idx = ResidueIndex(global_idx as u32);
            let res_id = &bb.ids[global_idx];
            let rb = &bb.bb[global_idx];

            // φ should be undefined (9999.0) ONLY for the chain's first residue.
            let is_chain_first = local_idx == 0;
            if !is_chain_first && (rb.phi - 9999.0).abs() < 1e-6 {
                violations.push(PreconditionViolation {
                    residue: res_idx,
                    id: res_id.clone(),
                    res_name: rb.res_name.clone(),
                    kind: ViolationKind::UndefinedPhi,
                });
                log::error!(
                    "Residue {}/{}/{}{} has undefined φ despite being mid-chain; fix: check if predecessor CA/C are missing or check PDB",
                    res_id.chain_id,
                    res_idx.0,
                    res_id.res_id,
                    res_id.insertion_code
                );
            }

            // ψ should be undefined (9999.0) ONLY for the chain's last residue.
            let is_chain_last = local_idx == (seg_end - seg_start - 1);
            if !is_chain_last && (rb.psi - 9999.0).abs() < 1e-6 {
                violations.push(PreconditionViolation {
                    residue: res_idx,
                    id: res_id.clone(),
                    res_name: rb.res_name.clone(),
                    kind: ViolationKind::UndefinedPsi,
                });
                log::error!(
                    "Residue {}/{}/{}{} has undefined ψ despite being mid-chain; fix: check if successor CA/C are missing or check PDB",
                    res_id.chain_id,
                    res_idx.0,
                    res_id.res_id,
                    res_id.insertion_code
                );
            }
        }
    }

    // (d) Check for chain breaks: consecutive same-chain residues with CA atoms > 4.5 Å apart.
    for w in chain_starts.windows(2) {
        let seg_start = w[0];
        let seg_end = w[1];

        let mut last_ca: Option<([f64; 3], usize, ResidueIndex, ResidueId)> = None;

        for i in seg_start..seg_end {
            let rb = &bb.bb[i];
            if let Some(ca) = rb.ca {
                if let Some((prev_ca, _prev_i, prev_res_idx, prev_res_id)) = last_ca {
                    let dx = ca[0] - prev_ca[0];
                    let dy = ca[1] - prev_ca[1];
                    let dz = ca[2] - prev_ca[2];
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt();

                    if dist > CHAIN_BREAK_CA_ANGSTROM {
                        violations.push(PreconditionViolation {
                            residue: ResidueIndex(i as u32),
                            id: bb.ids[i].clone(),
                            res_name: rb.res_name.clone(),
                            kind: ViolationKind::ChainBreak {
                                gap_to_next_ca: dist,
                            },
                        });
                        log::warn!(
                            "Chain break detected: {}/{}/{}{} to {}/{}/{}{} gap = {:.2} Å (threshold {:.1} Å); fix: verify residues are consecutive in PDB or check for missing residues",
                            prev_res_id.chain_id,
                            prev_res_idx.0,
                            prev_res_id.res_id,
                            prev_res_id.insertion_code,
                            bb.ids[i].chain_id,
                            i as u32,
                            bb.ids[i].res_id,
                            bb.ids[i].insertion_code,
                            dist,
                            CHAIN_BREAK_CA_ANGSTROM
                        );
                    }
                }
                last_ca = Some((ca, i, ResidueIndex(i as u32), bb.ids[i].clone()));
            }
        }
    }

    PreconditionReport { violations }
}

/// Check preconditions and return an error if any Error-severity violations exist.
///
/// Returns `Ok(())` if the structure is clean or only has warnings.
/// Returns `Err(ConFindError::PreconditionsFailed(n))` if there are n Error-severity violations.
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

    /// Helper to construct a minimal ResidueBackbone.
    fn make_residue(
        name: &str,
        n: Option<[f64; 3]>,
        ca: Option<[f64; 3]>,
        c: Option<[f64; 3]>,
        phi: f64,
        psi: f64,
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
            9999.0, // Sentinel for N-terminal
            -100.0,
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
            9999.0,
            -100.0,
        );
        rb.phi = 9999.0; // First residue: φ should be 9999
        rb.psi = -100.0; // Non-terminal: ψ should be defined

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
    fn test_undefined_phi_mid_chain() {
        let (mut rb, id) = make_residue(
            "ALA",
            Some([0.0, 0.0, 0.0]),
            Some([1.0, 0.0, 0.0]),
            Some([2.0, 0.0, 0.0]),
            9999.0, // Undefined φ for mid-chain residue
            -100.0,
        );

        let bb = ProteinBackbone {
            bb: vec![rb],
            ids: vec![id],
            chain_map: vec![0],
        };

        let report = check_preconditions(&bb);
        // Single residue is both first and last in chain, so φ=9999 is OK
        assert!(report.is_clean());
    }

    #[test]
    fn test_undefined_phi_mid_chain_multi_residue() {
        let (mut rb1, mut id1) = make_residue(
            "ALA",
            Some([0.0, 0.0, 0.0]),
            Some([1.0, 0.0, 0.0]),
            Some([2.0, 0.0, 0.0]),
            9999.0, // OK for chain first
            -80.0,
        );
        id1.res_id = 1;

        let (mut rb2, mut id2) = make_residue(
            "GLY",
            Some([3.0, 0.0, 0.0]),
            Some([4.0, 0.0, 0.0]),
            Some([5.0, 0.0, 0.0]),
            9999.0, // ERROR: mid-chain φ undefined
            -60.0,
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
        let (mut rb, id) = make_residue(
            "ALA",
            Some([0.0, 0.0, 0.0]),
            None, // Missing CA
            Some([2.0, 0.0, 0.0]),
            9999.0,
            -100.0,
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
        let (mut rb1, mut id1) = make_residue(
            "ALA",
            Some([0.0, 0.0, 0.0]),
            Some([1.0, 0.0, 0.0]),
            Some([2.0, 0.0, 0.0]),
            9999.0,
            -80.0,
        );
        id1.res_id = 1;

        let (mut rb2, mut id2) = make_residue(
            "GLY",
            Some([8.0, 0.0, 0.0]), // 8 Å away from previous CA at [1, 0, 0]
            Some([9.0, 0.0, 0.0]),
            Some([10.0, 0.0, 0.0]),
            -100.0,
            -60.0,
        );
        id2.res_id = 2;

        let bb = ProteinBackbone {
            bb: vec![rb1, rb2],
            ids: vec![id1, id2],
            chain_map: vec![0, 0],
        };

        let report = check_preconditions(&bb);

        // Should have one warning for chain break
        let warnings: Vec<_> = report.warnings().collect();
        assert_eq!(warnings.len(), 1);

        // Errors should be empty
        let errors: Vec<_> = report.errors().collect();
        assert_eq!(errors.len(), 0);

        // require_preconditions should still pass (warning only)
        let result = require_preconditions(&bb);
        assert!(result.is_ok());
    }

    #[test]
    fn test_unknown_residue_type_warning() {
        let (mut rb, id) = make_residue(
            "XYZ", // Unknown residue type
            Some([0.0, 0.0, 0.0]),
            Some([1.0, 0.0, 0.0]),
            Some([2.0, 0.0, 0.0]),
            9999.0,
            -100.0,
        );

        let bb = ProteinBackbone {
            bb: vec![rb],
            ids: vec![id],
            chain_map: vec![0],
        };

        let report = check_preconditions(&bb);

        let warnings: Vec<_> = report.warnings().collect();
        assert_eq!(warnings.len(), 1);
        match &warnings[0].kind {
            ViolationKind::UnknownResidueType { res_name } => {
                assert_eq!(res_name, "XYZ");
            }
            _ => panic!("Expected UnknownResidueType"),
        }

        // require_preconditions should still pass
        let result = require_preconditions(&bb);
        assert!(result.is_ok());
    }

    #[test]
    fn test_canonical_variants_accepted() {
        for variant in &["HID", "HIE", "HIP", "CYX"] {
            let (rb, id) = make_residue(
                variant,
                Some([0.0, 0.0, 0.0]),
                Some([1.0, 0.0, 0.0]),
                Some([2.0, 0.0, 0.0]),
                9999.0,
                -100.0,
            );

            let bb = ProteinBackbone {
                bb: vec![rb],
                ids: vec![id],
                chain_map: vec![0],
            };

            let report = check_preconditions(&bb);
            let warnings: Vec<_> = report.warnings().collect();
            // Should have 0 warnings (variant is canonical)
            assert_eq!(warnings.len(), 0, "Variant {} should be canonical", variant);
        }
    }
}
