//! Χ → Cartesian geometry engine for sidechain building.
//!
//! Builds sidechain coordinates from dihedral angles using internal coordinates (nerf).
//! Proline ring closure keeps χ exact and relaxes the CB-CG-CD angle (P3 v2).

pub mod template;
pub mod proline;
pub mod charmm_ic;

pub use template::{ResidueTemplate, proline_template, standard_residue_template};
pub use proline::{ProlineBuilder, ProlineCoords};
pub use charmm_ic::{CharmmIdeals, load_charmm_ideals, apply_charmm_ideals, map_template_to_charmm_name};

use proxide_geometry::geometry::nerf::Nerf;

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
        &[backbone_n, backbone_ca, backbone_c],  // A=N, B=CA, C=C
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
                0 => 2,  // B=N: A=backbone_C (rare; treat as improper)
                1 => 0,  // B=CA: A=N  (e.g. CG: N-CA-CB-CG)
                2 => 1,  // B=backbone_C: A=CA
                _ => template.bonds[b_idx]
                    .map(|bb| bb.parent_idx)
                    .unwrap_or(0),
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
        assert!(dist(cg1, cg2) > 1.5, "CG1 and CG2 must be distinct (got {:.3} Å)", dist(cg1, cg2));
        // Both should be ~1.524 Å from CB (idx 4)
        let cb = coords[4];
        assert!((dist(cb, cg1) - 1.524).abs() < 0.01, "CB-CG1 bond: {:.3}", dist(cb, cg1));
        assert!((dist(cb, cg2) - 1.524).abs() < 0.01, "CB-CG2 bond: {:.3}", dist(cb, cg2));
    }

    #[test]
    fn test_cys_sg_bond_length() {
        let tmpl = standard_residue_template("CYS").unwrap();
        let coords = build_standard_sidechain(&tmpl, &[60.0], N, CA, C);
        // idx 4 = CB, idx 5 = SG
        let cb = coords[4];
        let sg = coords[5];
        assert!((dist(cb, sg) - 1.808).abs() < 0.01, "CB-SG bond: {:.3}", dist(cb, sg));
    }

    #[test]
    fn test_o_bonded_to_c_not_n() {
        // O (idx 3) must be ~1.231 Å from backbone C (idx 2), not from N (idx 0).
        let tmpl = standard_residue_template("SER").unwrap();
        let coords = build_standard_sidechain(&tmpl, &[60.0], N, CA, C);
        let o = coords[3];
        let d_to_c = dist(o, C);
        let d_to_n = dist(o, N);
        assert!(d_to_c < 1.3, "O must be ~1.231 Å from backbone C, got {:.3}", d_to_c);
        assert!(d_to_n > 2.0, "O must not be bonded to N, got {:.3}", d_to_n);
    }
}
