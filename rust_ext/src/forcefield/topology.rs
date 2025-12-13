//! Molecular topology generation
//!
//! Generates bonds, angles, and dihedrals from atom coordinates and connectivity.
//! Used for MD simulation setup.

#![allow(dead_code)]

use crate::geometry::topology::infer_bonds;
use std::collections::{HashMap, HashSet};

/// Bond between two atoms
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Bond {
    pub i: usize,
    pub j: usize,
}

impl Bond {
    pub fn new(i: usize, j: usize) -> Self {
        // Canonical order
        if i < j {
            Self { i, j }
        } else {
            Self { i: j, j: i }
        }
    }
}

/// Angle between three atoms (i-j-k, with j as the central atom)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Angle {
    pub i: usize,
    pub j: usize, // Central atom
    pub k: usize,
}

impl Angle {
    pub fn new(i: usize, j: usize, k: usize) -> Self {
        // Canonical order: i < k
        if i < k {
            Self { i, j, k }
        } else {
            Self { i: k, j, k: i }
        }
    }
}

/// Dihedral/torsion between four atoms (i-j-k-l)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Dihedral {
    pub i: usize,
    pub j: usize,
    pub k: usize,
    pub l: usize,
    pub is_improper: bool,
}

impl Dihedral {
    pub fn new_proper(i: usize, j: usize, k: usize, l: usize) -> Self {
        // Canonical order: middle bond j-k should have j < k for proper torsions
        if j < k {
            Self {
                i,
                j,
                k,
                l,
                is_improper: false,
            }
        } else {
            Self {
                i: l,
                j: k,
                k: j,
                l: i,
                is_improper: false,
            }
        }
    }

    pub fn new_improper(central: usize, a: usize, b: usize, c: usize) -> Self {
        // Improper: central atom is j, others sorted
        let mut others = [a, b, c];
        others.sort();
        Self {
            i: others[0],
            j: central,
            k: others[1],
            l: others[2],
            is_improper: true,
        }
    }
}

/// Complete molecular topology
#[derive(Debug, Clone)]
pub struct Topology {
    pub bonds: Vec<Bond>,
    pub angles: Vec<Angle>,
    pub proper_dihedrals: Vec<Dihedral>,
    pub improper_dihedrals: Vec<Dihedral>,
    /// Maps atom index to list of bonded neighbors
    pub adjacency: HashMap<usize, Vec<usize>>,
}

impl Topology {
    /// Build topology from coordinates and elements
    pub fn from_coords(coords: &[[f32; 3]], elements: &[String], tolerance: f32) -> Self {
        // Infer bonds
        let bond_pairs = infer_bonds(coords, elements, tolerance);
        let bonds: Vec<Bond> = bond_pairs.iter().map(|&[i, j]| Bond::new(i, j)).collect();

        // Build adjacency list
        let mut adjacency: HashMap<usize, Vec<usize>> = HashMap::new();
        for bond in &bonds {
            adjacency
                .entry(bond.i)
                .or_insert_with(Vec::new)
                .push(bond.j);
            adjacency
                .entry(bond.j)
                .or_insert_with(Vec::new)
                .push(bond.i);
        }

        // Generate angles (i-j-k where i-j and j-k are bonds)
        let angles = Self::generate_angles(&adjacency);

        // Generate proper dihedrals (i-j-k-l where i-j, j-k, k-l are bonds)
        let proper_dihedrals = Self::generate_proper_dihedrals(&adjacency);

        // Generate improper dihedrals (for planar groups like sp2 carbons)
        let improper_dihedrals = Self::generate_improper_dihedrals(&adjacency, elements);

        Self {
            bonds,
            angles,
            proper_dihedrals,
            improper_dihedrals,
            adjacency,
        }
    }

    /// Identify aromatic atoms based on connectivity (heuristic)
    pub fn compute_aromaticity(&self, elements: &[String]) -> Vec<bool> {
        let mut is_aromatic = vec![false; elements.len()];

        for (i, element) in elements.iter().enumerate() {
            let neighbors = self.adjacency.get(&i).map(|v| v.len()).unwrap_or(0);

            // Heuristic: sp2 hybridized atoms (C, N) in 5 or 6 membered rings
            let possible_aromatic = match element.to_uppercase().as_str() {
                "C" => neighbors == 3,
                "N" => neighbors == 2 || neighbors == 3,
                _ => false,
            };

            if possible_aromatic {
                if self.is_in_ring(i, 6) {
                    is_aromatic[i] = true;
                }
            }
        }
        is_aromatic
    }

    /// Check if an atom is part of a ring of size <= max_size
    pub fn is_in_ring(&self, atom_idx: usize, max_size: usize) -> bool {
        let mut visited = HashSet::new();
        visited.insert(atom_idx);

        if let Some(neighbors) = self.adjacency.get(&atom_idx) {
            for &next in neighbors {
                visited.insert(next);
                // Start DFS from neighbor, looking for path back to atom_idx
                if self.dfs_ring_search(atom_idx, next, atom_idx, 1, max_size, &mut visited) {
                    return true;
                }
                visited.remove(&next);
            }
        }
        false
    }

    /// Generate all unique angles from adjacency list
    pub fn generate_angles(adjacency: &HashMap<usize, Vec<usize>>) -> Vec<Angle> {
        let mut angles = HashSet::new();

        // For each central atom j
        for (&j, neighbors) in adjacency {
            if neighbors.len() < 2 {
                continue;
            }

            // For each pair of neighbors (i, k)
            for (idx, &i) in neighbors.iter().enumerate() {
                for &k in neighbors.iter().skip(idx + 1) {
                    angles.insert(Angle::new(i, j, k));
                }
            }
        }

        angles.into_iter().collect()
    }

    /// Generate all unique proper dihedrals from adjacency list
    pub fn generate_proper_dihedrals(adjacency: &HashMap<usize, Vec<usize>>) -> Vec<Dihedral> {
        let mut dihedrals = HashSet::new();

        // For each central bond j-k
        for (&j, j_neighbors) in adjacency {
            for &k in j_neighbors {
                if j >= k {
                    continue; // Avoid duplicates
                }

                let k_neighbors = adjacency.get(&k).cloned().unwrap_or_default();

                // For each i bonded to j (and not k)
                for &i in j_neighbors {
                    if i == k {
                        continue;
                    }

                    // For each l bonded to k (and not j, not i)
                    for &l in &k_neighbors {
                        if l == j || l == i {
                            continue;
                        }

                        dihedrals.insert(Dihedral::new_proper(i, j, k, l));
                    }
                }
            }
        }

        dihedrals.into_iter().collect()
    }

    /// Generate improper dihedrals for planar groups (sp2 centers)
    pub fn generate_improper_dihedrals(
        adjacency: &HashMap<usize, Vec<usize>>,
        elements: &[String],
    ) -> Vec<Dihedral> {
        let mut impropers = Vec::new();

        // Impropers are typically on sp2 carbons (3 neighbors) and amide nitrogens
        for (&center, neighbors) in adjacency {
            if neighbors.len() != 3 {
                continue;
            }

            // Check if likely sp2 (C with 3 bonds, or N with 3 bonds)
            let elem = elements.get(center).map(|s| s.as_str()).unwrap_or("");
            if elem == "C" || elem == "N" {
                impropers.push(Dihedral::new_improper(
                    center,
                    neighbors[0],
                    neighbors[1],
                    neighbors[2],
                ));
            }
        }

        impropers
    }

    fn dfs_ring_search(
        &self,
        target: usize,
        current: usize,
        prev: usize,
        depth: usize,
        max_size: usize,
        visited: &mut HashSet<usize>,
    ) -> bool {
        if depth >= max_size {
            return false;
        }

        if let Some(neighbors) = self.adjacency.get(&current) {
            for &next in neighbors {
                if next == prev {
                    continue;
                }
                if next == target {
                    return depth >= 2;
                }

                if !visited.contains(&next) {
                    visited.insert(next);
                    if self.dfs_ring_search(target, next, current, depth + 1, max_size, visited) {
                        return true;
                    }
                    visited.remove(&next);
                }
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bond_canonical() {
        let b1 = Bond::new(5, 3);
        let b2 = Bond::new(3, 5);
        assert_eq!(b1, b2);
        assert_eq!(b1.i, 3);
        assert_eq!(b1.j, 5);
    }

    #[test]
    fn test_angle_canonical() {
        let a1 = Angle::new(5, 2, 3);
        let a2 = Angle::new(3, 2, 5);
        assert_eq!(a1, a2);
    }

    #[test]
    fn test_methane_topology() {
        // Methane: C at origin, 4 H around it
        let coords = [
            [0.0, 0.0, 0.0],       // C
            [1.09, 0.0, 0.0],      // H
            [-0.36, 1.03, 0.0],    // H
            [-0.36, -0.51, 0.89],  // H
            [-0.36, -0.51, -0.89], // H
        ];
        let elements = vec![
            "C".to_string(),
            "H".to_string(),
            "H".to_string(),
            "H".to_string(),
            "H".to_string(),
        ];

        let topo = Topology::from_coords(&coords, &elements, 1.3);

        // 4 C-H bonds
        assert_eq!(topo.bonds.len(), 4);

        // 6 angles (H-C-H, choose 2 from 4 hydrogens = 6)
        assert_eq!(topo.angles.len(), 6);

        // No proper dihedrals (need 4-atom chain)
        assert_eq!(topo.proper_dihedrals.len(), 0);
    }

    #[test]
    fn test_ethane_topology() {
        // Simplified ethane: C-C with 3 H each
        // Just checking angle/dihedral generation logic
        let coords = [
            [0.0, 0.0, 0.0],   // C1
            [1.54, 0.0, 0.0],  // C2 (C-C bond ~1.54 A)
            [-0.5, 0.9, 0.0],  // H1 on C1
            [-0.5, -0.9, 0.0], // H2 on C1
            [2.04, 0.9, 0.0],  // H3 on C2
            [2.04, -0.9, 0.0], // H4 on C2
        ];
        let elements = vec![
            "C".to_string(),
            "C".to_string(),
            "H".to_string(),
            "H".to_string(),
            "H".to_string(),
            "H".to_string(),
        ];

        let topo = Topology::from_coords(&coords, &elements, 1.7);

        // Should have C-C bond and C-H bonds
        assert!(topo.bonds.len() >= 5);

        // Should have angles
        assert!(topo.angles.len() > 0);
    }
}
