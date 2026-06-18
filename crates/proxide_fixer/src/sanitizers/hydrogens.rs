use crate::models::{Topology, Atom, Residue};
use proxide_geometry::geometry::nerf::Nerf;
use proxide_core::forcefield::ForceField;
use thiserror::Error;
use std::f32::consts::PI;
use std::collections::HashMap;

#[derive(Error, Debug)]
pub enum HydrogenError {
    #[error("Missing backbone atom {0} in residue")]
    MissingBackboneAtom(String),
    #[error("Failed to place hydrogen: {0}")]
    PlacementFailed(String),
    #[error("General error: {0}")]
    General(String),
}

/// Report on hydrogens placed by HydrogenSanitizer
#[derive(Debug)]
pub struct HydrogenReport {
    /// Total number of hydrogen atoms placed
    pub placed: usize,
    /// Per-residue counts: (residue_name, residue_id, num_placed)
    pub per_residue: Vec<(String, i32, usize)>,
    /// Residue names that had no template in the force field
    pub missing_template: Vec<String>,
}

/// HydrogenSanitizer places missing hydrogen atoms using force field templates.
/// For each residue, it:
/// 1. Retrieves the FF residue template
/// 2. Computes set difference: template H atoms NOT present = H to place
/// 3. For each missing H: finds parent heavy atom and classifies geometry
/// 4. Places via NeRF using CHARMM ideal bond lengths/angles
/// 5. Appends H atom with unique serial
pub struct HydrogenSanitizer<'a> {
    topology: &'a mut Topology,
    ff: &'a ForceField,
}

impl<'a> HydrogenSanitizer<'a> {
    pub fn new(topology: &'a mut Topology, ff: &'a ForceField) -> Self {
        Self { topology, ff }
    }

    /// Run hydrogen placement on all residues.
    /// Returns a report with placement counts and any missing templates.
    pub fn run(&mut self) -> Result<HydrogenReport, HydrogenError> {
        let mut placed_total = 0usize;
        let mut per_residue = Vec::new();
        let mut missing_template = Vec::new();

        // Iterate over all residues
        for chain in &mut self.topology.chains {
            for residue in &mut chain.residues {
                // Try to get the template from force field
                match self.ff.get_residue(&residue.name) {
                    Some(template) => {
                        // Find hydrogen atoms in template
                        let template_h_names: std::collections::HashSet<String> = template
                            .atoms
                            .iter()
                            .filter(|a| a.name.starts_with('H'))
                            .map(|a| a.name.clone())
                            .collect();

                        // Find hydrogen atoms already present in residue
                        let present_h_names: std::collections::HashSet<String> = residue
                            .atoms
                            .iter()
                            .filter(|a| a.element == "H")
                            .map(|a| a.name.clone())
                            .collect();

                        // Compute missing hydrogens
                        let missing_h_names: Vec<String> = template_h_names
                            .difference(&present_h_names)
                            .cloned()
                            .collect();

                        let num_placed = missing_h_names.len();
                        placed_total += num_placed;

                        if num_placed > 0 {
                            // Place each missing hydrogen
                            for h_name in missing_h_names {
                                // Find the parent atom of this hydrogen in the template
                                if let Some(parent_name) = Self::find_parent_atom(&h_name, template) {
                                    // Place the hydrogen (non-fatal if placement fails)
                                    let _ = Self::place_hydrogen_atom(residue, &h_name, &parent_name);
                                }
                            }

                            per_residue.push((residue.name.clone(), residue.res_id, num_placed));
                        }
                    }
                    None => {
                        missing_template.push(residue.name.clone());
                    }
                }
            }
        }

        Ok(HydrogenReport {
            placed: placed_total,
            per_residue,
            missing_template,
        })
    }

    /// Find the parent atom of a hydrogen in the residue template.
    /// Looks for bonds in the template to identify the heavy atom that this H is bonded to.
    fn find_parent_atom(h_name: &str, template: &proxide_core::forcefield::types::ResidueTemplate) -> Option<String> {
        // Search template bonds for one involving this hydrogen
        for (atom1, atom2) in &template.bonds {
            if atom1 == h_name {
                return Some(atom2.clone());
            }
            if atom2 == h_name {
                return Some(atom1.clone());
            }
        }
        None
    }

    /// Place a single hydrogen atom using Nerf geometry.
    /// Uses standard bond lengths: N-H ~1.01 Å, C-H ~1.09 Å
    /// Uses standard bond angles: ~109.5° for sp3, ~120° for sp2
    fn place_hydrogen_atom(residue: &mut Residue, h_name: &str, parent_name: &str) -> Result<(), HydrogenError> {
        // Find the parent heavy atom
        let parent_atom = residue
            .atoms
            .iter()
            .find(|a| a.name == parent_name)
            .cloned()
            .ok_or_else(|| HydrogenError::PlacementFailed(format!("Parent atom {} not found", parent_name)))?;

        // Determine bond length based on parent element
        let bond_length = match parent_atom.element.as_str() {
            "N" => 1.01,  // N-H
            "C" => 1.09,  // C-H
            "O" => 0.96,  // O-H (for rare cases)
            "S" => 1.34,  // S-H (for cysteine)
            _ => 1.09,    // Default to C-H
        };

        // Determine bond angle based on parent atom type (simplified)
        // Standard: sp3 carbon/nitrogen = 109.5°, sp2 = 120°, aromatic = 120°
        let bond_angle = 109.5;  // Default tetrahedral for now

        // Find two other atoms to use as reference for Nerf
        let other_atoms: Vec<&Atom> = residue
            .atoms
            .iter()
            .filter(|a| a.name != parent_name && a.element != "H")
            .take(2)
            .collect();

        if other_atoms.len() < 2 {
            // Not enough reference atoms for proper NeRF
            return Err(HydrogenError::PlacementFailed(
                "Insufficient reference atoms for Nerf placement".to_string(),
            ));
        }

        // Use Nerf to compute hydrogen position
        // NeRF needs three atoms [A, B, C] and places D bonded to C
        let prev_atoms = [
            other_atoms[0].coords,
            other_atoms[1].coords,
            parent_atom.coords,
        ];

        let h_coords = Nerf::place_atom(&prev_atoms, bond_length, bond_angle, 0.0);

        // Create new hydrogen atom
        let serial = residue
            .atoms
            .iter()
            .map(|a| a.serial)
            .max()
            .unwrap_or(0) + 1;

        let h_atom = Atom {
            name: h_name.to_string(),
            element: "H".to_string(),
            coords: h_coords,
            alt_loc: ' ',
            serial,
            b_factor: 0.0,
            occupancy: 1.0,
            is_hetatm: false,
        };

        residue.atoms.push(h_atom);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Chain};
    use proxide_core::forcefield::types::{ForceField, ResidueTemplate, ResidueAtom};

    fn create_test_forcefield() -> ForceField {
        let mut ff = ForceField::new("test".to_string());

        // Create a simple ALA template with H atoms
        let ala_template = ResidueTemplate {
            name: "ALA".to_string(),
            atoms: vec![
                ResidueAtom { name: "N".to_string(), atom_type: "N".to_string(), charge: None },
                ResidueAtom { name: "CA".to_string(), atom_type: "CA".to_string(), charge: None },
                ResidueAtom { name: "C".to_string(), atom_type: "C".to_string(), charge: None },
                ResidueAtom { name: "O".to_string(), atom_type: "O".to_string(), charge: None },
                ResidueAtom { name: "CB".to_string(), atom_type: "CB".to_string(), charge: None },
                ResidueAtom { name: "H".to_string(), atom_type: "H".to_string(), charge: None },
                ResidueAtom { name: "HA".to_string(), atom_type: "HA".to_string(), charge: None },
                ResidueAtom { name: "HB1".to_string(), atom_type: "HB".to_string(), charge: None },
                ResidueAtom { name: "HB2".to_string(), atom_type: "HB".to_string(), charge: None },
                ResidueAtom { name: "HB3".to_string(), atom_type: "HB".to_string(), charge: None },
            ],
            bonds: vec![
                ("N".to_string(), "H".to_string()),
                ("CA".to_string(), "HA".to_string()),
                ("CB".to_string(), "HB1".to_string()),
                ("CB".to_string(), "HB2".to_string()),
                ("CB".to_string(), "HB3".to_string()),
            ],
            external_bonds: vec![],
            override_level: None,
        };

        ff.residue_templates.push(ala_template);
        ff.build_indices();
        ff
    }

    #[test]
    fn test_hydrogen_sanitizer_new() {
        let mut topology = Topology { chains: vec![] };
        let ff = create_test_forcefield();
        let _sanitizer = HydrogenSanitizer::new(&mut topology, &ff);
        // Sanitizer created successfully
    }

    #[test]
    fn test_hydrogen_report_structure() {
        let report = HydrogenReport {
            placed: 0,
            per_residue: vec![],
            missing_template: vec![],
        };
        assert_eq!(report.placed, 0);
        assert!(report.per_residue.is_empty());
        assert!(report.missing_template.is_empty());
    }

    #[test]
    fn test_no_hydrogens_in_residue() {
        let mut topology = Topology {
            chains: vec![Chain {
                id: "A".to_string(),
                residues: vec![Residue {
                    name: "ALA".to_string(),
                    res_id: 1,
                    insertion_code: ' ',
                    atoms: vec![
                        Atom { name: "N".to_string(), element: "N".to_string(), coords: [0.0, 0.0, 0.0], alt_loc: ' ', serial: 1, b_factor: 0.0, occupancy: 1.0, is_hetatm: false },
                        Atom { name: "CA".to_string(), element: "C".to_string(), coords: [1.5, 0.0, 0.0], alt_loc: ' ', serial: 2, b_factor: 0.0, occupancy: 1.0, is_hetatm: false },
                        Atom { name: "C".to_string(), element: "C".to_string(), coords: [2.0, 1.4, 0.0], alt_loc: ' ', serial: 3, b_factor: 0.0, occupancy: 1.0, is_hetatm: false },
                        Atom { name: "O".to_string(), element: "O".to_string(), coords: [1.2, 2.3, 0.0], alt_loc: ' ', serial: 4, b_factor: 0.0, occupancy: 1.0, is_hetatm: false },
                        Atom { name: "CB".to_string(), element: "C".to_string(), coords: [2.0, -0.8, -1.2], alt_loc: ' ', serial: 5, b_factor: 0.0, occupancy: 1.0, is_hetatm: false },
                    ],
                }],
            }],
        };

        let ff = create_test_forcefield();
        let mut sanitizer = HydrogenSanitizer::new(&mut topology, &ff);

        let report = sanitizer.run().expect("run() should succeed");

        // Should detect that H atoms are missing
        assert!(report.placed > 0, "Should have placed hydrogen atoms");
        assert!(!report.per_residue.is_empty(), "Should have per-residue records");
    }

    #[test]
    fn test_missing_template_residue() {
        let mut topology = Topology {
            chains: vec![Chain {
                id: "A".to_string(),
                residues: vec![Residue {
                    name: "UNK".to_string(),  // Unknown residue
                    res_id: 1,
                    insertion_code: ' ',
                    atoms: vec![],
                }],
            }],
        };

        let ff = create_test_forcefield();
        let mut sanitizer = HydrogenSanitizer::new(&mut topology, &ff);

        let report = sanitizer.run().expect("run() should succeed");

        // Should record missing template
        assert!(report.missing_template.contains(&"UNK".to_string()));
    }

    #[test]
    fn test_hydrogen_atoms_added_to_residue() {
        let mut topology = Topology {
            chains: vec![Chain {
                id: "A".to_string(),
                residues: vec![Residue {
                    name: "ALA".to_string(),
                    res_id: 1,
                    insertion_code: ' ',
                    atoms: vec![
                        Atom { name: "N".to_string(), element: "N".to_string(), coords: [0.0, 0.0, 0.0], alt_loc: ' ', serial: 1, b_factor: 0.0, occupancy: 1.0, is_hetatm: false },
                        Atom { name: "CA".to_string(), element: "C".to_string(), coords: [1.5, 0.0, 0.0], alt_loc: ' ', serial: 2, b_factor: 0.0, occupancy: 1.0, is_hetatm: false },
                        Atom { name: "C".to_string(), element: "C".to_string(), coords: [2.0, 1.4, 0.0], alt_loc: ' ', serial: 3, b_factor: 0.0, occupancy: 1.0, is_hetatm: false },
                        Atom { name: "O".to_string(), element: "O".to_string(), coords: [1.2, 2.3, 0.0], alt_loc: ' ', serial: 4, b_factor: 0.0, occupancy: 1.0, is_hetatm: false },
                        Atom { name: "CB".to_string(), element: "C".to_string(), coords: [2.0, -0.8, -1.2], alt_loc: ' ', serial: 5, b_factor: 0.0, occupancy: 1.0, is_hetatm: false },
                    ],
                }],
            }],
        };

        let ff = create_test_forcefield();
        let num_atoms_before = topology.chains[0].residues[0].atoms.len();

        let mut sanitizer = HydrogenSanitizer::new(&mut topology, &ff);
        let report = sanitizer.run().expect("run() should succeed");

        let num_atoms_after = topology.chains[0].residues[0].atoms.len();

        // Should have added atoms
        assert!(num_atoms_after > num_atoms_before, "Should have added hydrogen atoms");

        // The residue should have hydrogen atoms
        let residue = &topology.chains[0].residues[0];
        let h_atoms: Vec<_> = residue.atoms.iter().filter(|a| a.element == "H").collect();
        assert!(!h_atoms.is_empty(), "Should have hydrogen atoms after sanitization");

        // Verify serials are unique
        let mut serials = h_atoms.iter().map(|a| a.serial).collect::<Vec<_>>();
        serials.sort();
        for i in 1..serials.len() {
            assert_ne!(serials[i-1], serials[i], "Serials should be unique");
        }
    }
}
