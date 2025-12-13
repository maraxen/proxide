//! Generalized Born Surface Area (GBSA) support for implicit solvent.
//!
//! Provides functions for assigning intrinsic radii and scaling factors
//! for GB calculations.

use std::collections::HashMap;

/// Assign intrinsic radii using the MBondi2 scheme.
///
/// Reference:
///   Onufriev, Bashford, Case, "Exploring native states and large-scale dynamics
///   with the generalized born model", Proteins 55, 383-394 (2004).
///
/// Rules (MBondi2):
///   - C: 1.70 Å
///   - N: 1.55 Å
///   - O: 1.50 Å
///   - S: 1.80 Å
///   - H (generic): 1.20 Å
///   - H (bound to N): 1.30 Å
///   - P: 1.85 Å
///   - F: 1.50 Å
///   - Cl: 1.70 Å
pub fn assign_mbondi2_radii(atom_names: &[String], bonds: &[[usize; 2]]) -> Vec<f32> {
    let n_atoms = atom_names.len();
    let mut radii = vec![0.0f32; n_atoms];

    // Build adjacency list for H-bonding check
    let mut adj: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..n_atoms {
        adj.insert(i, Vec::new());
    }
    for bond in bonds {
        adj.get_mut(&bond[0]).unwrap().push(bond[1]);
        adj.get_mut(&bond[1]).unwrap().push(bond[0]);
    }

    for (i, name) in atom_names.iter().enumerate() {
        let element = name.chars().next().unwrap_or('X');

        match element {
            'H' => {
                // Check if bonded to Nitrogen
                let is_bound_to_nitrogen = adj
                    .get(&i)
                    .map(|neighbors| {
                        neighbors.iter().any(|&neighbor| {
                            atom_names
                                .get(neighbor)
                                .map(|n| n.starts_with('N'))
                                .unwrap_or(false)
                        })
                    })
                    .unwrap_or(false);

                radii[i] = if is_bound_to_nitrogen { 1.30 } else { 1.20 };
            }
            'C' => radii[i] = 1.70,
            'N' => radii[i] = 1.55,
            'O' => radii[i] = 1.50,
            'S' => radii[i] = 1.80,
            'P' => radii[i] = 1.85,
            'F' => radii[i] = 1.50,
            _ => {
                // Check for Cl (two-letter element)
                if name.starts_with("Cl") || name.starts_with("CL") {
                    radii[i] = 1.70;
                } else {
                    // Default fallback
                    radii[i] = 1.50;
                }
            }
        }
    }

    radii
}

/// Assign scaling factors for OBC2 GBSA calculation.
///
/// Reference:
///   Onufriev, Bashford, Case, Proteins 55, 383-394 (2004).
///
/// Factors:
///   - H: 0.85
///   - C: 0.72
///   - N: 0.79
///   - O: 0.85
///   - F: 0.88
///   - P: 0.86
///   - S: 0.96
///   - Other: 0.80
pub fn assign_obc2_scaling_factors(atom_names: &[String]) -> Vec<f32> {
    atom_names
        .iter()
        .map(|name| {
            let element = name.chars().next().unwrap_or('X');
            match element {
                'H' => 0.85,
                'C' => 0.72,
                'N' => 0.79,
                'O' => 0.85,
                'F' => 0.88,
                'P' => 0.86,
                'S' => 0.96,
                _ => 0.80,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mbondi2_radii_basic() {
        let atom_names = vec![
            "N".to_string(),
            "CA".to_string(),
            "C".to_string(),
            "O".to_string(),
            "H".to_string(),
            "HA".to_string(),
        ];
        // N-H bond, CA-HA bond
        let bonds = vec![[0, 4], [1, 5]];

        let radii = assign_mbondi2_radii(&atom_names, &bonds);

        assert!((radii[0] - 1.55).abs() < 0.01); // N
        assert!((radii[1] - 1.70).abs() < 0.01); // CA (carbon)
        assert!((radii[2] - 1.70).abs() < 0.01); // C
        assert!((radii[3] - 1.50).abs() < 0.01); // O
        assert!((radii[4] - 1.30).abs() < 0.01); // H bound to N
        assert!((radii[5] - 1.20).abs() < 0.01); // HA bound to C
    }

    #[test]
    fn test_obc2_scaling() {
        let atom_names = vec![
            "N".to_string(),
            "CA".to_string(),
            "H".to_string(),
            "O".to_string(),
            "S".to_string(),
        ];

        let factors = assign_obc2_scaling_factors(&atom_names);

        assert!((factors[0] - 0.79).abs() < 0.01); // N
        assert!((factors[1] - 0.72).abs() < 0.01); // C
        assert!((factors[2] - 0.85).abs() < 0.01); // H
        assert!((factors[3] - 0.85).abs() < 0.01); // O
        assert!((factors[4] - 0.96).abs() < 0.01); // S
    }
}
