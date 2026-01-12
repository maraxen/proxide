//! GAFF (General Amber Force Field) for non-standard residues
//!
//! Provides atom typing and parameter assignment for ligands and
//! non-standard residues that aren't covered by protein force fields.
//!
//! GAFF uses element, hybridization, and connectivity to assign atom types.

#![allow(dead_code)]

use crate::forcefield::topology::Topology;
use std::collections::HashMap;

/// GAFF atom type
#[derive(Debug, Clone)]
pub struct GaffAtomType {
    pub name: String,
    pub element: String,
    pub description: String,
    pub mass: f32,
    /// LJ sigma (nm)
    pub sigma: f32,
    /// LJ epsilon (kJ/mol)
    pub epsilon: f32,
}

/// GAFF parameter set
#[derive(Debug, Clone)]
pub struct GaffParameters {
    /// Atom types mapped by name
    pub atom_types: HashMap<String, GaffAtomType>,
    /// Bond parameters: (type1, type2) -> (k, r0)
    pub bonds: HashMap<(String, String), (f32, f32)>,
    /// Angle parameters: (type1, type2, type3) -> (k, theta0)
    pub angles: HashMap<(String, String, String), (f32, f32)>,
}

impl Default for GaffParameters {
    fn default() -> Self {
        Self::new()
    }
}

impl GaffParameters {
    /// Create GAFF parameter set with standard atom types
    pub fn new() -> Self {
        let mut atom_types = HashMap::new();

        // Carbon types
        atom_types.insert(
            "c".to_string(),
            GaffAtomType {
                name: "c".to_string(),
                element: "C".to_string(),
                description: "Sp2 C carbonyl group".to_string(),
                mass: 12.01,
                sigma: 0.339967,
                epsilon: 0.359824,
            },
        );
        atom_types.insert(
            "c1".to_string(),
            GaffAtomType {
                name: "c1".to_string(),
                element: "C".to_string(),
                description: "Sp C".to_string(),
                mass: 12.01,
                sigma: 0.339967,
                epsilon: 0.359824,
            },
        );
        atom_types.insert(
            "c2".to_string(),
            GaffAtomType {
                name: "c2".to_string(),
                element: "C".to_string(),
                description: "Sp2 C".to_string(),
                mass: 12.01,
                sigma: 0.339967,
                epsilon: 0.359824,
            },
        );
        atom_types.insert(
            "c3".to_string(),
            GaffAtomType {
                name: "c3".to_string(),
                element: "C".to_string(),
                description: "Sp3 C".to_string(),
                mass: 12.01,
                sigma: 0.339967,
                epsilon: 0.457730,
            },
        );
        atom_types.insert(
            "ca".to_string(),
            GaffAtomType {
                name: "ca".to_string(),
                element: "C".to_string(),
                description: "Sp2 C in aromatic ring".to_string(),
                mass: 12.01,
                sigma: 0.339967,
                epsilon: 0.359824,
            },
        );

        // Nitrogen types
        atom_types.insert(
            "n".to_string(),
            GaffAtomType {
                name: "n".to_string(),
                element: "N".to_string(),
                description: "Sp2 N amide".to_string(),
                mass: 14.01,
                sigma: 0.325000,
                epsilon: 0.711280,
            },
        );
        atom_types.insert(
            "n3".to_string(),
            GaffAtomType {
                name: "n3".to_string(),
                element: "N".to_string(),
                description: "Sp3 N".to_string(),
                mass: 14.01,
                sigma: 0.325000,
                epsilon: 0.711280,
            },
        );
        atom_types.insert(
            "nh".to_string(),
            GaffAtomType {
                name: "nh".to_string(),
                element: "N".to_string(),
                description: "Amine N connected to aromatic".to_string(),
                mass: 14.01,
                sigma: 0.325000,
                epsilon: 0.711280,
            },
        );

        // Oxygen types
        atom_types.insert(
            "o".to_string(),
            GaffAtomType {
                name: "o".to_string(),
                element: "O".to_string(),
                description: "Sp2 O carbonyl".to_string(),
                mass: 16.00,
                sigma: 0.295992,
                epsilon: 0.878640,
            },
        );
        atom_types.insert(
            "oh".to_string(),
            GaffAtomType {
                name: "oh".to_string(),
                element: "O".to_string(),
                description: "O in hydroxyl".to_string(),
                mass: 16.00,
                sigma: 0.306647,
                epsilon: 0.880314,
            },
        );
        atom_types.insert(
            "os".to_string(),
            GaffAtomType {
                name: "os".to_string(),
                element: "O".to_string(),
                description: "Ether O".to_string(),
                mass: 16.00,
                sigma: 0.300001,
                epsilon: 0.711280,
            },
        );

        // Hydrogen types
        atom_types.insert(
            "h1".to_string(),
            GaffAtomType {
                name: "h1".to_string(),
                element: "H".to_string(),
                description: "H on C with 1 electron-withdrawing".to_string(),
                mass: 1.008,
                sigma: 0.247135,
                epsilon: 0.065689,
            },
        );
        atom_types.insert(
            "hc".to_string(),
            GaffAtomType {
                name: "hc".to_string(),
                element: "H".to_string(),
                description: "H on C".to_string(),
                mass: 1.008,
                sigma: 0.264953,
                epsilon: 0.065689,
            },
        );
        atom_types.insert(
            "hn".to_string(),
            GaffAtomType {
                name: "hn".to_string(),
                element: "H".to_string(),
                description: "H on N".to_string(),
                mass: 1.008,
                sigma: 0.106908,
                epsilon: 0.065689,
            },
        );
        atom_types.insert(
            "ho".to_string(),
            GaffAtomType {
                name: "ho".to_string(),
                element: "H".to_string(),
                description: "H in hydroxyl".to_string(),
                mass: 1.008,
                sigma: 0.000000,
                epsilon: 0.000000,
            },
        );
        atom_types.insert(
            "ha".to_string(),
            GaffAtomType {
                name: "ha".to_string(),
                element: "H".to_string(),
                description: "H on aromatic C".to_string(),
                mass: 1.008,
                sigma: 0.259964,
                epsilon: 0.062760,
            },
        );

        // Sulfur
        atom_types.insert(
            "s".to_string(),
            GaffAtomType {
                name: "s".to_string(),
                element: "S".to_string(),
                description: "S in thioether".to_string(),
                mass: 32.06,
                sigma: 0.356359,
                epsilon: 1.046,
            },
        );
        atom_types.insert(
            "sh".to_string(),
            GaffAtomType {
                name: "sh".to_string(),
                element: "S".to_string(),
                description: "S in thiol".to_string(),
                mass: 32.06,
                sigma: 0.356359,
                epsilon: 1.046,
            },
        );

        // Halogens
        atom_types.insert(
            "f".to_string(),
            GaffAtomType {
                name: "f".to_string(),
                element: "F".to_string(),
                description: "Fluorine".to_string(),
                mass: 19.00,
                sigma: 0.311815,
                epsilon: 0.255224,
            },
        );
        atom_types.insert(
            "cl".to_string(),
            GaffAtomType {
                name: "cl".to_string(),
                element: "Cl".to_string(),
                description: "Chlorine".to_string(),
                mass: 35.45,
                sigma: 0.347094,
                epsilon: 1.108_78,
            },
        );
        atom_types.insert(
            "br".to_string(),
            GaffAtomType {
                name: "br".to_string(),
                element: "Br".to_string(),
                description: "Bromine".to_string(),
                mass: 79.90,
                sigma: 0.390180,
                epsilon: 1.108_78,
            },
        );

        // Phosphorus
        atom_types.insert(
            "p5".to_string(),
            GaffAtomType {
                name: "p5".to_string(),
                element: "P".to_string(),
                description: "P in phosphate".to_string(),
                mass: 30.97,
                sigma: 0.374180,
                epsilon: 0.836800,
            },
        );

        Self {
            atom_types,
            bonds: HashMap::new(), // Would be populated from GAFF parameter file
            angles: HashMap::new(),
        }
    }

    /// Get atom type for an element based on connectivity
    pub fn assign_atom_type(
        &self,
        element: &str,
        num_neighbors: usize,
        is_aromatic: bool,
    ) -> Option<String> {
        let elem_upper = element.to_uppercase();

        match elem_upper.as_str() {
            "C" => {
                if is_aromatic {
                    Some("ca".to_string())
                } else {
                    match num_neighbors {
                        2 => Some("c1".to_string()), // Sp
                        3 => Some("c2".to_string()), // Sp2
                        4 => Some("c3".to_string()), // Sp3
                        _ => Some("c3".to_string()),
                    }
                }
            }
            "N" => {
                match num_neighbors {
                    1 | 2 => Some("n".to_string()),  // Sp2
                    3 | 4 => Some("n3".to_string()), // Sp3
                    _ => Some("n3".to_string()),
                }
            }
            "O" => {
                match num_neighbors {
                    1 => Some("o".to_string()),  // Carbonyl
                    2 => Some("os".to_string()), // Ether (could be oh)
                    _ => Some("os".to_string()),
                }
            }
            "H" => Some("hc".to_string()), // Generic, refine based on parent
            "S" => Some("s".to_string()),
            "F" => Some("f".to_string()),
            "CL" => Some("cl".to_string()),
            "BR" => Some("br".to_string()),
            "P" => Some("p5".to_string()),
            _ => None,
        }
    }
}

/// Assign GAFF atom types to a molecule
pub fn assign_gaff_types(
    elements: &[String],
    topology: &Topology,
    gaff: &GaffParameters,
) -> Vec<Option<String>> {
    let mut types = Vec::with_capacity(elements.len());

    // Compute aromaticity based on topology
    let is_aromatic_map = topology.compute_aromaticity(elements);

    // Initial pass
    for (i, element) in elements.iter().enumerate() {
        let num_neighbors = topology.adjacency.get(&i).map(|v| v.len()).unwrap_or(0);
        let is_aromatic = is_aromatic_map[i];

        types.push(gaff.assign_atom_type(element, num_neighbors, is_aromatic));
    }

    // Refinement pass (Hydrogens and Nitrogens depending on neighbors)
    // We clone types to look up neighbors while mutating
    let initial_types = types.clone();

    for (i, element) in elements.iter().enumerate() {
        let elem_upper = element.to_uppercase();

        if elem_upper == "H" {
            if let Some(neighbors) = topology.adjacency.get(&i) {
                if let Some(&neighbor_idx) = neighbors.first() {
                    if let Some(Some(neighbor_type)) = initial_types.get(neighbor_idx) {
                        let refined = match neighbor_type.as_str() {
                            "ca" | "cp" | "cq" => Some("ha".to_string()),
                            "n" | "na" | "nb" | "nc" | "nd" | "ne" | "nf" | "nh" | "no" => {
                                Some("hn".to_string())
                            }
                            "o" | "oh" | "os" => Some("ho".to_string()),
                            _ => None,
                        };
                        if refined.is_some() {
                            types[i] = refined;
                        }
                    }
                }
            }
        } else if elem_upper == "N" {
            // Check if connected to aromatic ring -> nh
            // But only if it is an amine (n3 or n2?)
            // If currently 'n3' (sp3) but attached to 'ca', often becomes planar 'nh'
            if let Some(Some(current_type)) = initial_types.get(i) {
                if current_type == "n3" || current_type == "n" {
                    if let Some(neighbors) = topology.adjacency.get(&i) {
                        for &neighbor_idx in neighbors {
                            if let Some(Some(neighbor_type)) = initial_types.get(neighbor_idx) {
                                if neighbor_type == "ca" {
                                    // Amine on aromatic ring
                                    types[i] = Some("nh".to_string());
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    types
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaff_parameters() {
        let gaff = GaffParameters::new();

        assert!(gaff.atom_types.contains_key("c3"));
        assert!(gaff.atom_types.contains_key("n3"));
        assert!(gaff.atom_types.contains_key("oh"));

        let c3 = &gaff.atom_types["c3"];
        assert_eq!(c3.element, "C");
        assert!((c3.mass - 12.01).abs() < 0.1);
    }

    #[test]
    fn test_atom_type_assignment() {
        let gaff = GaffParameters::new();

        // Sp3 carbon with 4 neighbors
        let t = gaff.assign_atom_type("C", 4, false);
        assert_eq!(t, Some("c3".to_string()));

        // Aromatic carbon
        let t = gaff.assign_atom_type("C", 3, true);
        assert_eq!(t, Some("ca".to_string()));

        // Sp3 nitrogen
        let t = gaff.assign_atom_type("N", 3, false);
        assert_eq!(t, Some("n3".to_string()));
    }
}
