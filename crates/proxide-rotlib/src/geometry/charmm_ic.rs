//! Load CHARMM ideal internal coordinates from FFXML and apply to residue templates.
//!
//! Replaces the Engh-Huber / CCD placeholder bond lengths and bond angles in
//! proxide-rotlib's residue templates with CHARMM ideal values, sourced from the
//! bundled CHARMM36 protein force field (`src/proxide/assets/charmm/charmm36_protein.xml`).
//! This matches the geometry MASTER/MSL used to bake `rotlib.bin` and is the path to
//! reducing the confind `load_pb()` contact-degree drift (research #820 ->
//! `.praxia/docs/research/260603_master-rotlib-cartesian-derivation.md`).
//!
//! The FFXML is parsed **once** in [`load_charmm_ideals`]; [`apply_charmm_ideals`] then
//! mutates a template in place using the cached force field — no per-residue re-parse.

use std::collections::HashMap;
use crate::RotlibError;
use super::template::ResidueTemplate;
use proxide_core::forcefield::{ForceField, parse_forcefield_xml};

/// CHARMM ideal internal coordinates, parsed once from a force-field XML.
///
/// Holds the parsed [`ForceField`] (for residue atom -> class resolution) plus fast
/// lookup maps for bond lengths (Å) and bond angles (degrees) keyed by atom class.
pub struct CharmmIdeals {
    ff: ForceField,
    /// Bond length (Å), keyed by the class pair sorted lexicographically.
    bond_lengths: HashMap<(String, String), f32>,
    /// Bond angle (degrees), keyed by (class1, center_class, class3). Stored as parsed;
    /// [`Self::bond_angle`] also tries the reversed terminal order.
    bond_angles: HashMap<(String, String, String), f32>,
}

impl CharmmIdeals {
    /// Bond length (Å) for the unordered class pair, if defined.
    fn bond_length(&self, class_a: &str, class_b: &str) -> Option<f32> {
        let key = if class_a <= class_b {
            (class_a.to_string(), class_b.to_string())
        } else {
            (class_b.to_string(), class_a.to_string())
        };
        self.bond_lengths.get(&key).copied()
    }

    /// Bond angle (degrees) for (terminal, center, terminal); tries both terminal orders.
    fn bond_angle(&self, class_a: &str, center: &str, class_c: &str) -> Option<f32> {
        self.bond_angles
            .get(&(class_a.to_string(), center.to_string(), class_c.to_string()))
            .or_else(|| {
                self.bond_angles
                    .get(&(class_c.to_string(), center.to_string(), class_a.to_string()))
            })
            .copied()
    }

    /// Resolve a residue atom name to its CHARMM atom class via the cached force field.
    fn class_of(&self, charmm_resname: &str, atom_name: &str) -> Option<String> {
        let res = self.ff.get_residue(charmm_resname)?;
        let atom = res.atoms.iter().find(|a| a.name == atom_name)?;
        self.ff
            .get_atom_type(&atom.atom_type)
            .map(|t| t.class.clone())
    }

    /// CHARMM ideal N–CD bond length (Å) for a proline-family residue — the ring-closure
    /// target for [`super::proline::ProlineBuilder`]. Returns `None` if the residue or
    /// the N/CD classes are absent.
    pub fn proline_cd_n_length(&self, charmm_resname: &str) -> Option<f32> {
        let n_class = self.class_of(charmm_resname, "N")?;
        let cd_class = self.class_of(charmm_resname, "CD")?;
        self.bond_length(&n_class, &cd_class)
    }
}

/// Parse a force-field XML once and build the CHARMM ideal lookup tables.
pub fn load_charmm_ideals(ffxml_path: &str) -> Result<CharmmIdeals, RotlibError> {
    let ff = parse_forcefield_xml(ffxml_path)
        .map_err(|e| RotlibError::InvalidFormat(format!("Failed to parse FFXML {ffxml_path}: {e}")))?;

    let mut bond_lengths = HashMap::new();
    for bond in &ff.harmonic_bonds {
        let length_a = bond.length * 10.0; // nm -> Å
        let key = if bond.class1 <= bond.class2 {
            (bond.class1.clone(), bond.class2.clone())
        } else {
            (bond.class2.clone(), bond.class1.clone())
        };
        bond_lengths.insert(key, length_a);
    }

    let mut bond_angles = HashMap::new();
    for angle in &ff.harmonic_angles {
        bond_angles.insert(
            (angle.class1.clone(), angle.class2.clone(), angle.class3.clone()),
            angle.angle.to_degrees(),
        );
    }

    Ok(CharmmIdeals { ff, bond_lengths, bond_angles })
}

/// Map a proxide template residue code to its CHARMM36 FFXML residue name.
///
/// CHARMM36 splits histidine into tautomers (HSD/HSE/HSP) and has a single neutral
/// cysteine (`CYS`); the proxide CYS/CYH/CYD variants share heavy-atom ring geometry,
/// so they map to `CYS`, and HIS maps to the delta-protonated `HSD` (its ring bond
/// lengths/angles match the other tautomers for IC purposes).
pub fn map_template_to_charmm_name(template_code: &str) -> &'static str {
    match template_code {
        "CYS" | "CYH" | "CYD" => "CYS",
        "HIS" => "HSD",
        "ALA" => "ALA",
        "ARG" => "ARG",
        "ASN" => "ASN",
        "ASP" => "ASP",
        "GLN" => "GLN",
        "GLU" => "GLU",
        "ILE" => "ILE",
        "LEU" => "LEU",
        "LYS" => "LYS",
        "MET" => "MET",
        "PHE" => "PHE",
        // Proline and its cis/trans peptide-bond variants share the CHARMM PRO topology.
        "PRO" | "CPR" | "TPR" => "PRO",
        "SER" => "SER",
        "THR" => "THR",
        "TRP" => "TRP",
        "TYR" => "TYR",
        "VAL" => "VAL",
        // Unknown code: apply_charmm_ideals will find no FFXML residue and warn + no-op.
        _ => "UNK",
    }
}

/// Override a template's bond lengths and bond angles with CHARMM ideals, in place.
///
/// For each sidechain atom (index ≥ 3) with a `BondDef`, looks up the CHARMM bond
/// length `(atom, parent)` and bond angle `(atom, parent, grandparent)`, where the
/// grandparent (the angle's third atom `B`) is resolved with the SAME convention as
/// [`super::build_standard_sidechain`] / `resolve_parent_chain`:
///   * parent == CA (idx 1) → grandparent = N (idx 0)
///   * parent == backbone C (idx 2) → grandparent = CA (idx 1)
///   * parent is a sidechain atom → grandparent = `bonds[parent].parent_idx`
///
/// `torsion_deg` and `relative_chi` are left untouched (improper/branch torsions are not
/// sourced from harmonic bond/angle terms). Misses are logged via `tracing::warn!` so
/// IC coverage is auditable.
pub fn apply_charmm_ideals(
    template: &mut ResidueTemplate,
    ideals: &CharmmIdeals,
    charmm_resname: &str,
) -> Result<usize, RotlibError> {
    if ideals.ff.get_residue(charmm_resname).is_none() {
        tracing::warn!("residue {charmm_resname} not found in CHARMM FFXML; template left unchanged");
        return Ok(0);
    }

    let mut updates: Vec<(usize, Option<f32>, Option<f32>)> = Vec::new();

    for atom_idx in 3..template.num_atoms() {
        let Some(bond) = template.bonds[atom_idx] else { continue };
        let atom_name = template.atom_names[atom_idx].clone();
        let parent_idx = bond.parent_idx;
        let parent_name = template.atom_names[parent_idx].clone();

        // Grandparent (angle third atom B) — mirror resolve_parent_chain in mod.rs.
        let grandparent_idx = match parent_idx {
            1 => 0, // parent CA -> B = N
            2 => 1, // parent backbone C -> B = CA
            _ => template.bonds[parent_idx].map(|b| b.parent_idx).unwrap_or(0),
        };
        let grandparent_name = template.atom_names[grandparent_idx].clone();

        let atom_class = ideals.class_of(charmm_resname, &atom_name);
        let parent_class = ideals.class_of(charmm_resname, &parent_name);
        let grandparent_class = ideals.class_of(charmm_resname, &grandparent_name);

        let new_length = match (&atom_class, &parent_class) {
            (Some(a), Some(p)) => ideals.bond_length(a, p).or_else(|| {
                tracing::warn!("no CHARMM bond {atom_name}-{parent_name} ({a}-{p}) in {charmm_resname}");
                None
            }),
            _ => None,
        };

        let new_angle = match (&atom_class, &parent_class, &grandparent_class) {
            (Some(a), Some(p), Some(g)) => ideals.bond_angle(a, p, g).or_else(|| {
                tracing::warn!(
                    "no CHARMM angle {atom_name}-{parent_name}-{grandparent_name} ({a}-{p}-{g}) in {charmm_resname}"
                );
                None
            }),
            _ => None,
        };

        updates.push((atom_idx, new_length, new_angle));
    }

    let mut applied = 0usize;
    for (atom_idx, new_length, new_angle) in updates {
        if let Some(bond) = &mut template.bonds[atom_idx] {
            if let Some(length) = new_length {
                bond.bond_length = length;
                applied += 1;
            }
            if let Some(angle) = new_angle {
                bond.bond_angle_deg = angle;
                applied += 1;
            }
        }
    }

    Ok(applied)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FFXML path anchored to the crate so tests pass regardless of CWD.
    pub(crate) fn charmm_ffxml_path() -> String {
        format!(
            "{}/../../src/proxide/assets/charmm/charmm36_protein.xml",
            env!("CARGO_MANIFEST_DIR")
        )
    }

    #[test]
    fn test_map_template_to_charmm_name() {
        assert_eq!(map_template_to_charmm_name("CYS"), "CYS");
        assert_eq!(map_template_to_charmm_name("CYH"), "CYS");
        assert_eq!(map_template_to_charmm_name("CYD"), "CYS");
        assert_eq!(map_template_to_charmm_name("HIS"), "HSD");
        assert_eq!(map_template_to_charmm_name("ALA"), "ALA");
    }

    #[test]
    fn test_load_and_proline_cd_n() {
        // Sanity-check the parser/lookup against the verified CHARMM proline ideals:
        // N-CD = 1.455 Å (research #820). Confirms nm->Å conversion + class resolution.
        let ideals = load_charmm_ideals(&charmm_ffxml_path()).expect("parse CHARMM36 FFXML");
        let cd_n = ideals.proline_cd_n_length("PRO").expect("PRO N-CD bond");
        assert!((cd_n - 1.455).abs() < 0.01, "PRO N-CD {cd_n:.3} Å != CHARMM 1.455");
    }

    #[test]
    fn test_glu_cb_angle_after_charmm_ideals() {
        // Verify apply_charmm_ideals sets the correct bond_angle_deg for GLU CB (idx 4).
        // Expected: NH1-CT1-CT2A = 113.5° from CHARMM36 FFXML.
        use crate::geometry::template::standard_residue_template;
        let ideals = load_charmm_ideals(&charmm_ffxml_path()).expect("parse CHARMM36 FFXML");
        let mut template = standard_residue_template("GLU").expect("GLU template");

        // Before: Engh-Huber placeholder (110.5°)
        let before = template.bonds[4].expect("CB bond").bond_angle_deg;

        let applied = apply_charmm_ideals(&mut template, &ideals, "GLU").expect("apply ideals");

        let after = template.bonds[4].expect("CB bond after").bond_angle_deg;
        eprintln!("GLU CB bond_angle_deg: before={before:.3} after={after:.3} applied={applied}");

        // CHARMM NH1-CT1-CT2A = 113.5°; must differ from placeholder 110.5°
        assert!(
            (after - 113.5).abs() < 0.5,
            "GLU CB angle after CHARMM = {after:.3}°, expected ~113.5°"
        );
    }

    #[test]
    fn test_charmm_angle_lookup_direct() {
        // Directly verify the class-lookup chain for GLU CB works end-to-end.
        let ideals = load_charmm_ideals(&charmm_ffxml_path()).expect("parse CHARMM36 FFXML");

        let cb_class = ideals.class_of("GLU", "CB");
        let ca_class = ideals.class_of("GLU", "CA");
        let n_class  = ideals.class_of("GLU", "N");
        eprintln!("GLU classes: CB={cb_class:?} CA={ca_class:?} N={n_class:?}");

        assert_eq!(cb_class.as_deref(), Some("CT2A"), "CB class");
        assert_eq!(ca_class.as_deref(), Some("CT1"),  "CA class");
        assert_eq!(n_class.as_deref(),  Some("NH1"),  "N class");

        let angle = ideals.bond_angle("CT2A", "CT1", "NH1");
        eprintln!("CT2A-CT1-NH1 angle = {angle:?}°");
        assert!(
            angle.map(|a| (a - 113.5).abs() < 0.5).unwrap_or(false),
            "angle not found or wrong: {angle:?}"
        );
    }
}
