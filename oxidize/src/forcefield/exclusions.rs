//! Nonbonded exclusion list generation
//!
//! Computes 1-2 (bonded), 1-3 (angle), and 1-4 (dihedral) exclusion pairs
//! for nonbonded force calculations.

#![allow(dead_code)]

use crate::forcefield::topology::Topology;
use std::collections::HashSet;

/// Exclusion pairs for nonbonded interactions
#[derive(Debug, Clone)]
pub struct Exclusions {
    /// 1-2 pairs (directly bonded atoms) - full exclusion
    pub exclusions_12: HashSet<(usize, usize)>,
    /// 1-3 pairs (atoms separated by 2 bonds) - full exclusion
    pub exclusions_13: HashSet<(usize, usize)>,
    /// 1-4 pairs (atoms separated by 3 bonds) - often scaled
    pub exclusions_14: HashSet<(usize, usize)>,
}

impl Exclusions {
    /// Build exclusion lists from topology
    pub fn from_topology(topo: &Topology) -> Self {
        let mut exclusions_12 = HashSet::new();
        let mut exclusions_13 = HashSet::new();
        let mut exclusions_14 = HashSet::new();

        // 1-2: directly from bonds
        for bond in &topo.bonds {
            let pair = if bond.i < bond.j {
                (bond.i, bond.j)
            } else {
                (bond.j, bond.i)
            };
            exclusions_12.insert(pair);
        }

        // 1-3: from angles (i and k in i-j-k)
        for angle in &topo.angles {
            let pair = if angle.i < angle.k {
                (angle.i, angle.k)
            } else {
                (angle.k, angle.i)
            };
            exclusions_13.insert(pair);
        }

        // 1-4: from proper dihedrals (i and l in i-j-k-l)
        for dihedral in &topo.proper_dihedrals {
            let pair = if dihedral.i < dihedral.l {
                (dihedral.i, dihedral.l)
            } else {
                (dihedral.l, dihedral.i)
            };
            exclusions_14.insert(pair);
        }

        Self {
            exclusions_12,
            exclusions_13,
            exclusions_14,
        }
    }

    /// Check if a pair should be fully excluded (1-2 or 1-3)
    pub fn is_excluded(&self, i: usize, j: usize) -> bool {
        let pair = if i < j { (i, j) } else { (j, i) };
        self.exclusions_12.contains(&pair) || self.exclusions_13.contains(&pair)
    }

    /// Check if a pair is a 1-4 pair (for scaling)
    pub fn is_14_pair(&self, i: usize, j: usize) -> bool {
        let pair = if i < j { (i, j) } else { (j, i) };
        self.exclusions_14.contains(&pair)
    }

    /// Get all pairs that should be fully excluded from nonbonded
    pub fn all_excluded_pairs(&self) -> HashSet<(usize, usize)> {
        let mut all = self.exclusions_12.clone();
        all.extend(&self.exclusions_13);
        all
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forcefield::topology::Topology;

    #[test]
    fn test_methane_exclusions() {
        // Methane: C at center, 4 H
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
        let excl = Exclusions::from_topology(&topo);

        // All C-H pairs should be 1-2 excluded
        assert!(excl.is_excluded(0, 1)); // C-H1
        assert!(excl.is_excluded(0, 2)); // C-H2

        // All H-H pairs should be 1-3 excluded (through C)
        assert!(excl.is_excluded(1, 2)); // H1-H2
        assert!(excl.is_excluded(1, 3)); // H1-H3
    }
}
