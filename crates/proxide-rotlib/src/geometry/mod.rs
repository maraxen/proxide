//! Χ → Cartesian geometry engine for sidechain building.
//!
//! Builds sidechain coordinates from dihedral angles using internal coordinates (nerf).
//! Proline ring closure keeps χ exact and relaxes the CB-CG-CD angle (P3 v2).

pub mod ccd_parser;
pub mod charmm_ic;
pub mod ic_validate;
pub mod proline;
pub mod rtf_parser;
pub mod template;

pub use ccd_parser::parse_ccd_ic_table;
pub use charmm_ic::{
    apply_charmm_ideals, load_charmm_ideals, map_template_to_charmm_name, CharmmIdeals,
};
pub use ic_validate::{validate_and_fill_ic, ICCoverage, ICSource, ICValidationReport};
pub use proline::{ProlineBuilder, ProlineCoords};
pub use rtf_parser::parse_rtf_ic_table;
pub use template::{proline_template, standard_residue_template, ResidueTemplate};

use crate::pb::proxide::rotlib::v1::ResidueGeometryTable;
use proxide_geometry::geometry::nerf::Nerf;
use std::collections::HashMap;

/// Apply IC geometry from a ResidueGeometryTable to a ResidueTemplate.
///
/// For each atom in the template (index >= 3), looks up the corresponding IC records
/// in the geometry table and updates the template's BondDef with:
/// - bond_length: extracted from IC records
/// - bond_angle_deg: extracted from IC records
/// - torsion_deg and relative_chi: preserved unchanged
///
/// # Arguments
/// * `template` - mutable template to update
/// * `table` - ResidueGeometryTable (pre-parsed) containing IC records
///
/// # Returns
/// () on success, or warning/skip if IC records not found for specific atoms
pub fn apply_ic_table(template: &mut ResidueTemplate, table: &ResidueGeometryTable) {
    // Find the ResidueGeometry matching this template's code.
    // Try exact match first, then CHARMM name alias (HIS->HSD, CYS/CYH/CYD->CYS, etc.).
    let residue_geom = table
        .residues
        .iter()
        .find(|r| r.name == template.code)
        .or_else(|| {
            let charmm = charmm_ic::map_template_to_charmm_name(&template.code);
            table.residues.iter().find(|r| r.name == charmm)
        });

    let residue_geom = match residue_geom {
        Some(rg) => rg,
        None => {
            tracing::warn!("No geometry for residue {} in IC table", template.code);
            return;
        }
    };

    // RTF IC convention: `IC i j k l | b_ij θ_ijk φ θ_jkl b_kl`
    //   The record PLACES atom l given anchors i, j, k:
    //     bond length  parent(k) → atom(l) = b_kl
    //     angle at parent k      (j-k-l)   = theta_jkl
    //   An intermediate record with atom_k = X (no asterisk) also encodes:
    //     angle at parent-of-X (i-j-k)     = theta_ijk  (angle from grandparent through parent to X)
    //
    // Hybrid lookup:
    //   bond_length : atom_l == this_atom → b_kl         (always correct)
    //   bond_angle  : atom_k == this_atom → theta_ijk    (correct for most atoms)
    //                 fallback atom_l == this_atom → theta_jkl (terminal atoms like CE)

    type IcRef<'a> = &'a crate::pb::proxide::rotlib::v1::IcRecord;

    // by_atom_l: last atom → record  (for bond lengths)
    let by_atom_l: HashMap<&str, IcRef<'_>> = residue_geom
        .ic
        .iter()
        .map(|rec| (rec.atom_l.as_str(), rec))
        .collect();

    // by_atom_k: (parent, atom) → record  (for bond angles via theta_ijk)
    let by_atom_k: HashMap<(&str, &str), IcRef<'_>> = residue_geom
        .ic
        .iter()
        .map(|rec| ((rec.atom_j.as_str(), rec.atom_k.as_str()), rec))
        .collect();

    for atom_idx in 3..template.num_atoms() {
        let atom_name = &template.atom_names[atom_idx].clone();
        let Some(bond_def) = template.bonds[atom_idx] else {
            continue;
        };
        let parent_name = &template.atom_names[bond_def.parent_idx].clone();

        // Bond length: from record placing this atom as atom_l; verify parent matches atom_k.
        let bond_length = by_atom_l
            .get(atom_name.as_str())
            .filter(|r| r.atom_k == *parent_name)
            .map(|r| r.b_kl);

        // Bond angle: theta_ijk from record where atom_k == this_atom, atom_j == parent.
        // Fallback: theta_jkl from atom_l record (works for terminal atoms with no atom_k record).
        let bond_angle = by_atom_k
            .get(&(parent_name.as_str(), atom_name.as_str()))
            .map(|r| r.theta_ijk)
            .or_else(|| {
                by_atom_l
                    .get(atom_name.as_str())
                    .filter(|r| r.atom_k == *parent_name)
                    .map(|r| r.theta_jkl)
            });

        match (bond_length, bond_angle) {
            (Some(bl), Some(ba)) => {
                if let Some(bond) = &mut template.bonds[atom_idx] {
                    bond.bond_length = bl;
                    bond.bond_angle_deg = ba;
                }
            }
            _ => {
                tracing::debug!(
                    "{}.{}: IC table miss (bond_length={}, bond_angle={}); geometry unchanged",
                    template.code,
                    atom_name,
                    bond_length.is_some(),
                    bond_angle.is_some()
                );
            }
        }
    }
}

/// Build sidechain coordinates for a standard (non-proline) residue.
///
/// # Arguments
/// * `template` - The residue template defining atom connectivity and ideal geometry
/// * `chi_values` - Slice of χ angles in degrees, in chi order (chi1, chi2, chi3, chi4)
/// * `backbone_n` - N atom position in canonical backbone frame
/// * `backbone_ca` - CA atom position in canonical backbone frame
/// * `backbone_c` - C atom position in canonical backbone frame
///
/// # Returns
/// Vector of [f32;3] coordinates for each atom in template.atom_names order
/// (backbone atoms N/CA/C/O first, then sidechain atoms).
pub fn build_standard_sidechain(
    template: &ResidueTemplate,
    chi_values: &[f32],
    backbone_n: [f32; 3],
    backbone_ca: [f32; 3],
    backbone_c: [f32; 3],
) -> Vec<[f32; 3]> {
    let mut coords = vec![backbone_n, backbone_ca, backbone_c];

    // O atom (idx 3, parent=backbone_C=idx 2): dihedral N-CA-C-O
    let o_bond = template.bonds[3].expect("O bond missing");
    let o = Nerf::place_atom(
        &[backbone_n, backbone_ca, backbone_c], // A=N, B=CA, C=C
        o_bond.bond_length,
        o_bond.bond_angle_deg,
        o_bond.torsion_deg,
    );
    coords.push(o);

    // Sidechain atoms starting from CB (idx 4).
    // Grandparent/great-grandparent must be resolved by walking the template's parent
    // chain — NOT by subtracting 1/2 from parent_idx (breaks for branched residues and CB).
    for atom_idx in 4..template.num_atoms() {
        if let Some(bond) = template.bonds[atom_idx] {
            let c_idx = bond.parent_idx;
            let (b_idx, a_idx) = resolve_parent_chain(template, c_idx);
            let torsion = determine_torsion(template, atom_idx, chi_values);
            let atom = Nerf::place_atom(
                &[coords[a_idx], coords[b_idx], coords[c_idx]],
                bond.bond_length,
                bond.bond_angle_deg,
                torsion,
            );
            coords.push(atom);
        }
    }

    coords
}

/// Resolve grandparent (B) and great-grandparent (A) indices for NeRF placement.
///
/// Returns `(b_idx, a_idx)` where A-B-C-D defines the dihedral (D is the atom being
/// placed, C = `c_idx` is its parent, B is grandparent, A is great-grandparent).
///
/// CB uses the backbone improper C-N-CA-CB, so parent=CA maps to B=N, A=backbone_C.
fn resolve_parent_chain(template: &ResidueTemplate, c_idx: usize) -> (usize, usize) {
    match c_idx {
        1 => (0, 2), // parent=CA: B=N, A=backbone_C  (CB improper: C-N-CA-CB)
        2 => (1, 0), // parent=backbone_C: B=CA, A=N  (O placement: N-CA-C-O)
        _ => {
            // Parent is a sidechain atom — walk up via template.bonds.
            let b_bond = template.bonds[c_idx].expect("parent bond missing");
            let b_idx = b_bond.parent_idx;
            let a_idx = match b_idx {
                0 => 2, // B=N: A=backbone_C (rare; treat as improper)
                1 => 0, // B=CA: A=N  (e.g. CG: N-CA-CB-CG)
                2 => 1, // B=backbone_C: A=CA
                _ => template.bonds[b_idx].map(|bb| bb.parent_idx).unwrap_or(0),
            };
            (b_idx, a_idx)
        }
    }
}

/// Determine the torsion angle for a sidechain atom.
///
/// Priority order:
/// 1. Chi-defining atom: return chi_values[chi_idx].
/// 2. Branch atom with relative_chi: return chi_values[chi_idx] + torsion_deg.
/// 3. Otherwise: return the template's fixed torsion_deg.
fn determine_torsion(template: &ResidueTemplate, atom_idx: usize, chi_values: &[f32]) -> f32 {
    // Check if this atom defines a chi dihedral (highest priority).
    for (chi_idx, dihedral) in template.dihedrals.iter().enumerate() {
        if dihedral.atom_indices[3] == atom_idx && chi_idx < chi_values.len() {
            return chi_values[chi_idx];
        }
    }

    if let Some(bond) = template.bonds[atom_idx] {
        // Branch atom whose torsion is chi-relative.
        if let Some(chi_idx) = bond.relative_chi {
            if chi_idx < chi_values.len() {
                return chi_values[chi_idx] + bond.torsion_deg;
            }
        }
        bond.torsion_deg
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::template::standard_residue_template;

    fn dist(a: [f32; 3], b: [f32; 3]) -> f32 {
        let d: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
        d.sqrt()
    }

    // Canonical build frame (same as ProlineBuilder)
    const N: [f32; 3] = [0.0, 0.0, 0.0];
    const CA: [f32; 3] = [1.458, 0.0, 0.0];
    const C: [f32; 3] = [2.009, 1.420, 0.0];

    #[test]
    fn test_val_cg1_cg2_distinct() {
        // VAL at chi1=60°: CG1 at chi1, CG2 at 120° (fixed). They must differ.
        let tmpl = standard_residue_template("VAL").unwrap();
        let coords = build_standard_sidechain(&tmpl, &[60.0], N, CA, C);
        // idx 5 = CG1, idx 6 = CG2
        let cg1 = coords[5];
        let cg2 = coords[6];
        assert!(
            dist(cg1, cg2) > 1.5,
            "CG1 and CG2 must be distinct (got {:.3} Å)",
            dist(cg1, cg2)
        );
        // Both should be ~1.524 Å from CB (idx 4)
        let cb = coords[4];
        assert!(
            (dist(cb, cg1) - 1.524).abs() < 0.01,
            "CB-CG1 bond: {:.3}",
            dist(cb, cg1)
        );
        assert!(
            (dist(cb, cg2) - 1.524).abs() < 0.01,
            "CB-CG2 bond: {:.3}",
            dist(cb, cg2)
        );
    }

    #[test]
    fn test_cys_sg_bond_length() {
        let tmpl = standard_residue_template("CYS").unwrap();
        let coords = build_standard_sidechain(&tmpl, &[60.0], N, CA, C);
        // idx 4 = CB, idx 5 = SG
        let cb = coords[4];
        let sg = coords[5];
        assert!(
            (dist(cb, sg) - 1.808).abs() < 0.01,
            "CB-SG bond: {:.3}",
            dist(cb, sg)
        );
    }

    #[test]
    fn test_o_bonded_to_c_not_n() {
        // O (idx 3) must be ~1.231 Å from backbone C (idx 2), not from N (idx 0).
        let tmpl = standard_residue_template("SER").unwrap();
        let coords = build_standard_sidechain(&tmpl, &[60.0], N, CA, C);
        let o = coords[3];
        let d_to_c = dist(o, C);
        let d_to_n = dist(o, N);
        assert!(
            d_to_c < 1.3,
            "O must be ~1.231 Å from backbone C, got {:.3}",
            d_to_c
        );
        assert!(d_to_n > 2.0, "O must not be bonded to N, got {:.3}", d_to_n);
    }

    #[test]
    fn test_apply_ic_table_updates_bond_lengths() {
        use crate::pb::proxide::rotlib::v1::{IcRecord, ResidueGeometry, ResidueGeometryTable};

        // Create a minimal ResidueGeometryTable for testing
        let table = ResidueGeometryTable {
            source: "test".to_string(),
            version: "test_v1".to_string(),
            license: "test".to_string(),
            citation: "test".to_string(),
            residues: vec![ResidueGeometry {
                name: "SER".to_string(),
                ic: vec![
                    // O placement: N-CA-C-O
                    IcRecord {
                        atom_i: "N".to_string(),
                        atom_j: "CA".to_string(),
                        atom_k: "C".to_string(),
                        atom_l: "O".to_string(),
                        branch: false,
                        b_ij: 1.458,      // N-CA bond (prerequisite)
                        theta_ijk: 108.5, // N-CA-C angle
                        phi_ijkl: 180.0,  // N-CA-C-O dihedral
                        theta_jkl: 120.8, // CA-C-O angle
                        b_kl: 1.231,      // C-O bond (consequent)
                    },
                    // CB placement: N-CA-CB-OG (parent=CA)
                    IcRecord {
                        atom_i: "N".to_string(),
                        atom_j: "CA".to_string(),
                        atom_k: "CB".to_string(),
                        atom_l: "OG".to_string(),
                        branch: false,
                        b_ij: 1.458,      // N-CA bond
                        theta_ijk: 110.5, // N-CA-CB angle
                        phi_ijkl: -119.7, // dihedral
                        theta_jkl: 111.1, // CA-CB-OG angle
                        b_kl: 1.417,      // CB-OG bond
                    },
                    // OG placement: CA-CB-OG-HG (parent=CB, child would be HG)
                    IcRecord {
                        atom_i: "CA".to_string(),
                        atom_j: "CB".to_string(),
                        atom_k: "OG".to_string(),
                        atom_l: "HG".to_string(),
                        branch: false,
                        b_ij: 1.540,      // CA-CB bond
                        theta_ijk: 111.1, // CA-CB-OG angle
                        phi_ijkl: 0.0,    // dihedral
                        theta_jkl: 109.5, // CB-OG-HG angle
                        b_kl: 0.96,       // OG-HG bond
                    },
                    // HG placement: CB-OG-HG-* (parent=OG, for successor bond lookup)
                    IcRecord {
                        atom_i: "CB".to_string(),
                        atom_j: "OG".to_string(),
                        atom_k: "HG".to_string(),
                        atom_l: "O".to_string(), // dummy reference
                        branch: false,
                        b_ij: 1.417,      // CB-OG bond (this is what OG needs!)
                        theta_ijk: 109.5, // CB-OG-HG angle
                        phi_ijkl: 180.0,  // dihedral
                        theta_jkl: 104.5, // OG-HG-* angle
                        b_kl: 0.96,       // HG-* bond
                    },
                ],
            }],
        };

        // Get SER template and apply IC
        let mut tmpl = standard_residue_template("SER").unwrap();

        apply_ic_table(&mut tmpl, &table);

        // CB should be updated with IC values
        assert!(tmpl.bonds[4].is_some());
        let cb_bond = tmpl.bonds[4].unwrap();
        // bond_angle_deg should be theta_ijk from N-CA-CB-OG record = 110.5
        assert!(
            (cb_bond.bond_angle_deg - 110.5).abs() < 0.01,
            "CB angle updated: expected ~110.5, got {}",
            cb_bond.bond_angle_deg
        );

        // OG should be updated with IC values from CA-CB-OG-HG record (for angle)
        // and CB-OG-HG-* record (for bond length)
        assert!(tmpl.bonds[5].is_some());
        let og_bond = tmpl.bonds[5].unwrap();
        // bond_angle_deg should be theta_ijk from CA-CB-OG-HG = 111.1
        assert!(
            (og_bond.bond_angle_deg - 111.1).abs() < 0.01,
            "OG angle updated: expected ~111.1, got {}",
            og_bond.bond_angle_deg
        );
        // bond_length should be b_ij from CB-OG-HG-* record = 1.417
        assert!(
            (og_bond.bond_length - 1.417).abs() < 0.01,
            "OG bond updated: expected ~1.417, got {}",
            og_bond.bond_length
        );
    }
}
