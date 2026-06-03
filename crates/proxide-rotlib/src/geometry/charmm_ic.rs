//! Load CHARMM ideal internal coordinates from FFXML and apply to residue templates.
//!
//! Replaces Engh-Huber / CCD placeholder coordinates with CHARMM ideal bond lengths
//! and bond angles, sourced from CHARMM36 force field parameters.

use std::collections::HashMap;
use crate::RotlibError;
use super::template::ResidueTemplate;

/// CHARMM ideal internal coordinates for a residue.
///
/// Contains lookup maps for bond lengths and bond angles keyed by atom class pairs.
pub struct CharmmIdeals {
    /// Bond length in Å: key = unordered pair of atom classes (sorted alphabetically)
    bond_lengths: HashMap<(String, String), f32>,
    /// Bond angle in degrees: key = (class1, class2_center, class3)
    bond_angles: HashMap<(String, String, String), f32>,
}

impl CharmmIdeals {
    /// Create empty CharmmIdeals.
    fn new() -> Self {
        Self {
            bond_lengths: HashMap::new(),
            bond_angles: HashMap::new(),
        }
    }

    /// Look up a bond length by atom classes (in Å).
    fn get_bond_length(&self, class1: &str, class2: &str) -> Option<f32> {
        // Try both orderings (normalized by sorting)
        let (c1, c2) = if class1 <= class2 {
            (class1.to_string(), class2.to_string())
        } else {
            (class2.to_string(), class1.to_string())
        };
        self.bond_lengths.get(&(c1, c2)).copied()
    }

    /// Look up a bond angle by atom classes (in degrees).
    /// The second class is the central atom.
    fn get_bond_angle(&self, class1: &str, class2_center: &str, class3: &str) -> Option<f32> {
        // Try both orderings of the terminal atoms
        if let Some(angle) = self.bond_angles.get(&(class1.to_string(), class2_center.to_string(), class3.to_string())) {
            return Some(*angle);
        }
        // Try reversed
        self.bond_angles.get(&(class3.to_string(), class2_center.to_string(), class1.to_string())).copied()
    }
}

/// Load CHARMM ideals from a force field XML file.
pub fn load_charmm_ideals(ffxml_path: &str) -> Result<CharmmIdeals, RotlibError> {
    let ff = proxide_core::forcefield::parse_forcefield_xml(ffxml_path)
        .map_err(|e| RotlibError::InvalidFormat(format!("Failed to parse FFXML: {}", e)))?;

    let mut ideals = CharmmIdeals::new();

    // Build bond length map (class1-class2 -> length in Å)
    for bond in &ff.harmonic_bonds {
        let length_angstrom = bond.length * 10.0; // Convert nm to Å
        let (c1, c2) = if bond.class1 <= bond.class2 {
            (bond.class1.clone(), bond.class2.clone())
        } else {
            (bond.class2.clone(), bond.class1.clone())
        };
        ideals.bond_lengths.insert((c1, c2), length_angstrom);
    }

    // Build bond angle map (class1-class2_center-class3 -> angle in degrees)
    for angle in &ff.harmonic_angles {
        let angle_deg = angle.angle.to_degrees();
        ideals.bond_angles.insert(
            (angle.class1.clone(), angle.class2.clone(), angle.class3.clone()),
            angle_deg,
        );
    }

    Ok(ideals)
}

/// Map template residue code to CHARMM residue name.
/// Used by converter to look up residue parameters in the FFXML.
pub fn map_template_to_charmm_name(template_code: &str) -> String {
    match template_code {
        "CYS" | "CYH" | "CYD" => "CYS".to_string(),
        "HIS" => "HSD".to_string(),
        _ => template_code.to_string(),
    }
}

/// Apply CHARMM ideals to a residue template.
///
/// For each atom with a BondDef, looks up the bond length and angle from CHARMM
/// parameters using atom classes. Overrides the template's ideal values.
/// Leaves torsion_deg and relative_chi unchanged.
///
/// # Arguments
/// * `template` - The residue template to modify
/// * `ideals` - CHARMM ideal internal coordinates
/// * `charmm_resname` - CHARMM residue name (e.g., "ALA", "PRO", "HSD" for HIS)
pub fn apply_charmm_ideals(
    template: &mut ResidueTemplate,
    ideals: &CharmmIdeals,
    charmm_resname: &str,
) -> Result<(), RotlibError> {
    // Get residue template from ForceField to look up atom types
    // Try multiple paths since this may be called from different contexts
    let ff_path = "src/proxide/assets/charmm/charmm36_protein.xml";
    let ff = proxide_core::forcefield::parse_forcefield_xml(ff_path)
        .or_else(|_| {
            proxide_core::forcefield::parse_forcefield_xml("../../../src/proxide/assets/charmm/charmm36_protein.xml")
        })
        .or_else(|_| {
            proxide_core::forcefield::parse_forcefield_xml("../../../../src/proxide/assets/charmm/charmm36_protein.xml")
        })
        .map_err(|e| RotlibError::InvalidFormat(format!("Failed to parse FFXML: {}", e)))?;

    // Get the residue template from the forcefield
    let ff_residue = ff.get_residue(charmm_resname)
        .ok_or_else(|| RotlibError::InvalidFormat(format!("Residue {} not found in FFXML", charmm_resname)))?;

    // Build atom name -> atom type map
    let mut atom_type_map: HashMap<String, String> = HashMap::new();
    for ff_atom in &ff_residue.atoms {
        atom_type_map.insert(ff_atom.name.clone(), ff_atom.atom_type.clone());
    }

    // Build atom type name -> class map
    let mut type_class_map: HashMap<String, String> = HashMap::new();
    for atom_type in &ff.atom_types {
        type_class_map.insert(atom_type.name.clone(), atom_type.class.clone());
    }

    // First pass: collect parent indices and grandparent lookups to avoid borrow issues
    let mut updates: Vec<(usize, Option<f32>, Option<f32>)> = Vec::new();

    for atom_idx in 3..template.num_atoms() {
        if let Some(bond) = template.bonds[atom_idx] {
            let atom_name = &template.atom_names[atom_idx];
            let parent_idx = bond.parent_idx;
            let parent_name = &template.atom_names[parent_idx];

            // Get atom classes
            let atom_class = atom_type_map.get(atom_name)
                .and_then(|type_name| type_class_map.get(type_name).cloned());
            let parent_class = atom_type_map.get(parent_name)
                .and_then(|type_name| type_class_map.get(type_name).cloned());

            // Look up bond length
            let new_length = if let (Some(atom_cl), Some(parent_cl)) = (&atom_class, &parent_class) {
                if let Some(length) = ideals.get_bond_length(atom_cl, parent_cl) {
                    Some(length)
                } else {
                    tracing::warn!(
                        "No CHARMM bond length found for {}-{} ({}-{})",
                        atom_name, parent_name, atom_cl, parent_cl
                    );
                    None
                }
            } else {
                None
            };

            // Look up bond angle by resolving grandparent
            let new_angle = if parent_idx == 1 {
                // parent=CA: B=N, A=backbone_C
                let grandparent_idx = 0;
                let grandparent_name = &template.atom_names[grandparent_idx];
                let grandparent_class = atom_type_map.get(grandparent_name)
                    .and_then(|type_name| type_class_map.get(type_name).cloned());

                if let (Some(gp_cl), Some(p_cl), Some(atom_cl)) = (&grandparent_class, &parent_class, &atom_class) {
                    if let Some(angle) = ideals.get_bond_angle(gp_cl, p_cl, atom_cl) {
                        Some(angle)
                    } else {
                        tracing::warn!(
                            "No CHARMM angle found for {}-{}-{} ({}-{}-{})",
                            grandparent_name, parent_name, atom_name, gp_cl, p_cl, atom_cl
                        );
                        None
                    }
                } else {
                    None
                }
            } else if parent_idx == 2 {
                // parent=backbone_C: B=CA, A=N (O placement)
                let grandparent_idx = 1;
                let grandparent_name = &template.atom_names[grandparent_idx];
                let grandparent_class = atom_type_map.get(grandparent_name)
                    .and_then(|type_name| type_class_map.get(type_name).cloned());

                if let (Some(gp_cl), Some(p_cl), Some(atom_cl)) = (&grandparent_class, &parent_class, &atom_class) {
                    if let Some(angle) = ideals.get_bond_angle(gp_cl, p_cl, atom_cl) {
                        Some(angle)
                    } else {
                        tracing::warn!(
                            "No CHARMM angle found for {}-{}-{} ({}-{}-{})",
                            grandparent_name, parent_name, atom_name, gp_cl, p_cl, atom_cl
                        );
                        None
                    }
                } else {
                    None
                }
            } else {
                // Parent is sidechain; walk up via bonds
                if let Some(b_bond) = template.bonds[parent_idx] {
                    let grandparent_idx = b_bond.parent_idx;
                    let grandparent_name = &template.atom_names[grandparent_idx];
                    let grandparent_class = atom_type_map.get(grandparent_name)
                        .and_then(|type_name| type_class_map.get(type_name).cloned());

                    if let (Some(gp_cl), Some(p_cl), Some(atom_cl)) = (&grandparent_class, &parent_class, &atom_class) {
                        if let Some(angle) = ideals.get_bond_angle(gp_cl, p_cl, atom_cl) {
                            Some(angle)
                        } else {
                            tracing::warn!(
                                "No CHARMM angle found for {}-{}-{} ({}-{}-{})",
                                grandparent_name, parent_name, atom_name, gp_cl, p_cl, atom_cl
                            );
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            };

            updates.push((atom_idx, new_length, new_angle));
        }
    }

    // Second pass: apply updates
    for (atom_idx, new_length, new_angle) in updates {
        if let Some(bond) = &mut template.bonds[atom_idx] {
            if let Some(length) = new_length {
                bond.bond_length = length;
            }
            if let Some(angle) = new_angle {
                bond.bond_angle_deg = angle;
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_map_template_to_charmm_name_only() {
        // Unit test the mapping function (doesn't require FFXML file)
        assert_eq!(map_template_to_charmm_name("CYS"), "CYS");
        assert_eq!(map_template_to_charmm_name("CYH"), "CYS");
        assert_eq!(map_template_to_charmm_name("HIS"), "HSD");
    }

    #[test]
    fn test_map_template_to_charmm_name() {
        assert_eq!(map_template_to_charmm_name("CYS"), "CYS");
        assert_eq!(map_template_to_charmm_name("CYH"), "CYS");
        assert_eq!(map_template_to_charmm_name("CYD"), "CYS");
        assert_eq!(map_template_to_charmm_name("HIS"), "HSD");
        assert_eq!(map_template_to_charmm_name("ALA"), "ALA");
    }
}
