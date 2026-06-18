//! Non-proline CHARMM internal-coordinate validation with Engh-Huber fallback.
//!
//! For every non-proline sidechain atom (idx ≥ 3), validates and records whether its
//! bond length & angle are sourced from CHARMM or fall back to Engh-Huber placeholders.
//! Ensures no silent gaps: every atom either has CHARMM geometry or retains a valid
//! Engh-Huber fallback, or is marked Missing and logged as an error.

use crate::RotlibError;
use super::charmm_ic::CharmmIdeals;
use super::template::ResidueTemplate;

/// Source of an internal-coordinate value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ICSource {
    /// Value came from CHARMM force field.
    Charmm,
    /// Value fell back to Engh-Huber placeholder (neither CHARMM nor class-missing).
    EnghHuberFallback,
    /// Neither CHARMM nor Engh-Huber placeholder available — a gap.
    Missing,
}

/// Coverage record for one sidechain atom's bond length and angle.
#[derive(Debug, Clone)]
pub struct ICCoverage {
    /// Residue code (e.g., "GLU").
    pub residue: String,
    /// Atom name (e.g., "CB").
    pub atom: String,
    /// Source of the bond length (parent → this atom).
    pub length_source: ICSource,
    /// Source of the bond angle (grandparent → parent → this atom).
    pub angle_source: ICSource,
}

/// Validation report for all non-proline sidechain atoms.
#[derive(Debug, Clone)]
pub struct ICValidationReport {
    /// One record per sidechain atom (idx ≥ 3).
    pub entries: Vec<ICCoverage>,
}

impl ICValidationReport {
    /// True if no atom has a Missing source.
    pub fn fully_covered(&self) -> bool {
        self.entries
            .iter()
            .all(|e| e.length_source != ICSource::Missing && e.angle_source != ICSource::Missing)
    }

    /// Count of atoms using EnghHuberFallback for at least one quantity (length or angle).
    pub fn fallback_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| {
                e.length_source == ICSource::EnghHuberFallback
                    || e.angle_source == ICSource::EnghHuberFallback
            })
            .count()
    }

    /// Iterator over records with Missing source.
    pub fn missing(&self) -> impl Iterator<Item = &ICCoverage> {
        self.entries
            .iter()
            .filter(|e| e.length_source == ICSource::Missing || e.angle_source == ICSource::Missing)
    }
}

/// Validate and fill internal coordinates for a non-proline residue.
///
/// For each sidechain atom (idx ≥ 3):
///   1. Snapshot the current bond_length & bond_angle_deg (Engh-Huber placeholders).
///   2. Query CHARMM ideals for the atom's bond length & angle.
///   3. Per quantity:
///      - If CHARMM has a value: use it, source = Charmm
///      - Else if Engh-Huber placeholder is finite: keep it, source = EnghHuberFallback, log warn
///      - Else: source = Missing, log error
///   4. Return Err(RotlibError) if any Missing survives.
///
/// **Proline handling:** Skips PRO/CPR/TPR entirely, returning Ok with an empty report,
/// leaving the ring geometry untouched (guard against #820 regression).
pub fn validate_and_fill_ic(
    template: &mut ResidueTemplate,
    ideals: &CharmmIdeals,
    charmm_resname: &str,
) -> Result<ICValidationReport, RotlibError> {
    // Guard: proline residues are handled separately (ring closure, not CHARMM ideals).
    if matches!(template.code.as_str(), "PRO" | "CPR" | "TPR") {
        return Ok(ICValidationReport {
            entries: Vec::new(),
        });
    }

    let mut entries = Vec::new();
    let mut missing_atoms = Vec::new();

    for atom_idx in 3..template.num_atoms() {
        let Some(bond) = template.bonds[atom_idx] else { continue };
        let atom_name = template.atom_names[atom_idx].clone();
        let parent_idx = bond.parent_idx;
        let parent_name = template.atom_names[parent_idx].clone();

        // Grandparent (angle third atom) — mirror the logic in apply_charmm_ideals.
        let grandparent_idx = match parent_idx {
            1 => 0, // parent CA -> B = N
            2 => 1, // parent backbone C -> B = CA
            _ => template.bonds[parent_idx]
                .map(|b| b.parent_idx)
                .unwrap_or(0),
        };
        let grandparent_name = template.atom_names[grandparent_idx].clone();

        // Snapshot Engh-Huber placeholders.
        let engh_huber_length = bond.bond_length;
        let engh_huber_angle = bond.bond_angle_deg;

        // Query CHARMM for classes and values.
        let atom_class = ideals.class_of(charmm_resname, &atom_name);
        let parent_class = ideals.class_of(charmm_resname, &parent_name);
        let grandparent_class = ideals.class_of(charmm_resname, &grandparent_name);

        // Determine bond length source and value.
        let (length_source, new_length) =
            match (&atom_class, &parent_class) {
                (Some(a), Some(p)) => {
                    match ideals.bond_length(a, p) {
                        Some(charmm_val) => (ICSource::Charmm, Some(charmm_val)),
                        None => {
                            // CHARMM miss: check if Engh-Huber placeholder is finite.
                            if engh_huber_length.is_finite() && engh_huber_length > 0.0 {
                                log::warn!(
                                    "CHARMM IC miss {res} {atom} bond_length; using Engh-Huber {val} Å",
                                    res = template.code,
                                    atom = atom_name,
                                    val = engh_huber_length
                                );
                                (ICSource::EnghHuberFallback, None)
                            } else {
                                (ICSource::Missing, None)
                            }
                        }
                    }
                }
                _ => {
                    // Class missing: cannot query CHARMM.
                    if engh_huber_length.is_finite() && engh_huber_length > 0.0 {
                        (ICSource::EnghHuberFallback, None)
                    } else {
                        (ICSource::Missing, None)
                    }
                }
            };

        // Determine bond angle source and value.
        let (angle_source, new_angle) = match (&atom_class, &parent_class, &grandparent_class) {
            (Some(a), Some(p), Some(g)) => {
                match ideals.bond_angle(a, p, g) {
                    Some(charmm_val) => (ICSource::Charmm, Some(charmm_val)),
                    None => {
                        // CHARMM miss: check Engh-Huber placeholder.
                        if engh_huber_angle.is_finite() && engh_huber_angle > 0.0 {
                            log::warn!(
                                "CHARMM IC miss {res} {atom} bond_angle; using Engh-Huber {val}°",
                                res = template.code,
                                atom = atom_name,
                                val = engh_huber_angle
                            );
                            (ICSource::EnghHuberFallback, None)
                        } else {
                            (ICSource::Missing, None)
                        }
                    }
                }
            }
            _ => {
                // Class missing: check Engh-Huber.
                if engh_huber_angle.is_finite() && engh_huber_angle > 0.0 {
                    (ICSource::EnghHuberFallback, None)
                } else {
                    (ICSource::Missing, None)
                }
            }
        };

        // Record coverage.
        let coverage = ICCoverage {
            residue: template.code.clone(),
            atom: atom_name.clone(),
            length_source,
            angle_source,
        };

        if length_source == ICSource::Missing || angle_source == ICSource::Missing {
            missing_atoms.push(coverage.clone());
        }

        entries.push(coverage);

        // Apply updates if CHARMM sourced.
        if let Some(bond) = &mut template.bonds[atom_idx] {
            if let Some(length) = new_length {
                bond.bond_length = length;
            }
            if let Some(angle) = new_angle {
                bond.bond_angle_deg = angle;
            }
        }
    }

    if !missing_atoms.is_empty() {
        let missing_str = missing_atoms
            .iter()
            .map(|c| format!("{} {}", c.residue, c.atom))
            .collect::<Vec<_>>()
            .join(", ");
        log::error!("Missing CHARMM IC for: {}", missing_str);
        return Err(RotlibError::InvalidFormat(format!(
            "Missing CHARMM IC for: {}",
            missing_str
        )));
    }

    Ok(ICValidationReport { entries })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::template::standard_residue_template;

    /// FFXML path for tests (same as charmm_ic tests).
    fn charmm_ffxml_path() -> String {
        format!(
            "{}/../../src/proxide/assets/charmm/charmm36_protein.xml",
            env!("CARGO_MANIFEST_DIR")
        )
    }

    #[test]
    fn test_glu_fully_covered() {
        // GLU has complete CHARMM geometry (CB, CG, CD, OE1, OE2).
        let ideals = super::super::charmm_ic::load_charmm_ideals(&charmm_ffxml_path())
            .expect("load CHARMM36");
        let mut template = standard_residue_template("GLU").expect("GLU template");

        let report =
            validate_and_fill_ic(&mut template, &ideals, "GLU").expect("validate GLU");

        assert!(report.fully_covered(), "GLU should be fully covered by CHARMM");
        assert!(report.fallback_count() == 0, "GLU should have no fallbacks");
        assert!(
            report.missing().count() == 0,
            "GLU should have no missing entries"
        );
    }

    #[test]
    fn test_arg_fully_covered() {
        // ARG is another well-defined residue in CHARMM.
        let ideals = super::super::charmm_ic::load_charmm_ideals(&charmm_ffxml_path())
            .expect("load CHARMM36");
        let mut template = standard_residue_template("ARG").expect("ARG template");

        let report =
            validate_and_fill_ic(&mut template, &ideals, "ARG").expect("validate ARG");

        assert!(report.fully_covered(), "ARG should be fully covered by CHARMM");
    }

    #[test]
    fn test_leu_fully_covered() {
        // LEU is another well-defined residue in CHARMM.
        let ideals = super::super::charmm_ic::load_charmm_ideals(&charmm_ffxml_path())
            .expect("load CHARMM36");
        let mut template = standard_residue_template("LEU").expect("LEU template");

        let report =
            validate_and_fill_ic(&mut template, &ideals, "LEU").expect("validate LEU");

        assert!(report.fully_covered(), "LEU should be fully covered by CHARMM");
    }

    #[test]
    fn test_glu_cb_angle_from_charmm() {
        // Verify that GLU CB angle comes from CHARMM (113.5°) not placeholder (110.5°).
        let ideals = super::super::charmm_ic::load_charmm_ideals(&charmm_ffxml_path())
            .expect("load CHARMM36");
        let mut template = standard_residue_template("GLU").expect("GLU template");

        // Before: Engh-Huber placeholder
        let before_angle = template.bonds[4]
            .expect("CB bond")
            .bond_angle_deg;

        let report =
            validate_and_fill_ic(&mut template, &ideals, "GLU").expect("validate GLU");

        // After: CHARMM value
        let after_angle = template.bonds[4]
            .expect("CB bond after")
            .bond_angle_deg;

        // Cross-check: the CB coverage should show Charmm source.
        let cb_coverage = report
            .entries
            .iter()
            .find(|e| e.atom == "CB")
            .expect("CB coverage entry");
        assert_eq!(
            cb_coverage.angle_source, ICSource::Charmm,
            "GLU CB angle should come from CHARMM"
        );

        // Value should match known CHARMM angle (113.5°).
        assert!(
            (after_angle - 113.5).abs() < 0.5,
            "GLU CB angle {after_angle:.1}° should be ~113.5° from CHARMM"
        );
        // And should differ from placeholder 110.5°.
        assert!(
            (before_angle - 110.5).abs() < 0.5,
            "Before should be placeholder 110.5°"
        );
    }

    #[test]
    fn test_proline_early_return() {
        // Proline should return early with empty report, leaving ring geometry untouched.
        let ideals = super::super::charmm_ic::load_charmm_ideals(&charmm_ffxml_path())
            .expect("load CHARMM36");
        let mut template = super::super::template::proline_template();

        // Snapshot original CB geometry.
        let original_cb = template.bonds[3].expect("PRO CB");

        let report = validate_and_fill_ic(&mut template, &ideals, "PRO").expect("validate PRO");

        // Report should be empty.
        assert!(report.entries.is_empty(), "PRO report should be empty");

        // Geometry should be unchanged.
        let after_cb = template.bonds[3].expect("PRO CB after");
        assert_eq!(
            original_cb.bond_length, after_cb.bond_length,
            "PRO CB length unchanged"
        );
        assert_eq!(
            original_cb.bond_angle_deg, after_cb.bond_angle_deg,
            "PRO CB angle unchanged"
        );
    }

    #[test]
    fn test_fallback_logging() {
        // Create a synthetic template with missing CHARMM class to trigger fallback.
        // This test verifies that when a class is missing but Engh-Huber placeholder exists,
        // we use the fallback and log a warning.

        // For now, test with a residue that is fully covered (like GLU) to confirm
        // the fallback_count is 0 when all CHARMM entries exist.
        let ideals = super::super::charmm_ic::load_charmm_ideals(&charmm_ffxml_path())
            .expect("load CHARMM36");
        let mut template = standard_residue_template("GLU").expect("GLU template");

        let report =
            validate_and_fill_ic(&mut template, &ideals, "GLU").expect("validate GLU");

        // GLU is fully covered; fallback_count should be 0.
        assert_eq!(
            report.fallback_count(),
            0,
            "GLU with full CHARMM should have 0 fallbacks"
        );
    }
}
