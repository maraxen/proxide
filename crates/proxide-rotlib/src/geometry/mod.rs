//! Χ → Cartesian geometry engine for sidechain building.
//!
//! Builds sidechain coordinates from dihedral angles using internal coordinates (nerf).
//! Proline ring closure keeps χ exact and relaxes the CB-CG-CD angle (P3 v2).

pub mod template;
pub mod proline;

pub use template::{ResidueTemplate, proline_template, standard_residue_template};
pub use proline::{ProlineBuilder, ProlineCoords};

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

    // O atom: parent=C (idx 2)
    let o_bond = template.bonds[3].expect("O bond missing");
    let o = Nerf::place_atom(
        &[backbone_ca, backbone_c, backbone_n],
        o_bond.bond_length,
        o_bond.bond_angle_deg,
        o_bond.torsion_deg,
    );
    coords.push(o);

    // Sidechain atoms starting from CB (idx 4)
    for atom_idx in 4..template.num_atoms() {
        if let Some(bond) = template.bonds[atom_idx] {
            let parent_idx = bond.parent_idx;
            let parent = coords[parent_idx];

            // Determine grandparent and great-grandparent
            let (grandparent, great_grandparent) = if parent_idx >= 2 {
                (coords[parent_idx - 1], coords[parent_idx - 2])
            } else {
                // Shouldn't happen for sidechains, but fallback
                (coords[0], coords[1])
            };

            // Determine torsion angle
            let torsion = determine_torsion(template, atom_idx, chi_values);

            let atom = Nerf::place_atom(
                &[great_grandparent, grandparent, parent],
                bond.bond_length,
                bond.bond_angle_deg,
                torsion,
            );
            coords.push(atom);
        }
    }

    coords
}

/// Determine the torsion angle for a sidechain atom.
///
/// If the atom defines a χ dihedral (is the terminal atom of a dihedral definition),
/// use the χ value from chi_values. Otherwise, use the template's torsion_deg.
fn determine_torsion(template: &ResidueTemplate, atom_idx: usize, chi_values: &[f32]) -> f32 {
    for (chi_idx, dihedral) in template.dihedrals.iter().enumerate() {
        if dihedral.atom_indices[3] == atom_idx {
            // This atom is the terminal atom of a dihedral
            if chi_idx < chi_values.len() {
                return chi_values[chi_idx];
            }
        }
    }

    // Not a chi-defining atom; use template value
    template.bonds[atom_idx]
        .map(|b| b.torsion_deg)
        .unwrap_or(0.0)
}
