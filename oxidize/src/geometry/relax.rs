//! Hydrogen position relaxation via energy minimization.
//!
//! This module implements energy relaxation for hydrogen atom positions using
//! a hill-climbing algorithm based on electrostatic and Lennard-Jones potentials.
//!
//! # Algorithm
//!
//! The relaxation finds optimal hydrogen positions by:
//! 1. Identifying rotatable bonds (heavy atom → hydrogen connections)
//! 2. Rotating hydrogen atoms about terminal bonds
//! 3. Accepting rotations that decrease total energy
//!
//! # Energy Function
//!
//! V = V_el + V_nb
//!
//! - Electrostatic: V_el = 332.067 * q_i * q_j / D_ij
//! - Lennard-Jones: V_nb = ε * (r^12/D^12 - 2*r^6/D^6)
//!
//! # References
//!
//! - Rappé et al. (1992). UFF, a Full Periodic Table Force Field. JACS 114, 10024-10035.
//! - hydride library: https://github.com/biotite-dev/hydride

#![allow(dead_code)]

use crate::physics::constants::COULOMB_CONSTANT;
use crate::physics::vdw::{combine_lj_parameters, compute_lj_energy};
use std::f32::consts::PI;

/// UFF nonbonded parameters: (van der Waals radius Å, well depth kcal/mol)
/// From Rappé et al. JACS 1992
pub fn get_uff_params(element: &str) -> Option<(f32, f32)> {
    let element_upper = element.to_uppercase();
    match element_upper.as_str() {
        "H" => Some((2.886, 0.044)),
        "HE" => Some((2.362, 0.056)),
        "LI" => Some((2.451, 0.025)),
        "BE" => Some((2.745, 0.085)),
        "B" => Some((4.083, 0.180)),
        "C" => Some((3.851, 0.105)),
        "N" => Some((3.660, 0.069)),
        "O" => Some((3.500, 0.060)),
        "F" => Some((3.364, 0.050)),
        "NE" => Some((3.243, 0.042)),
        "NA" => Some((2.983, 0.030)),
        "MG" => Some((3.021, 0.111)),
        "AL" => Some((4.499, 0.505)),
        "SI" => Some((4.295, 0.402)),
        "P" => Some((4.147, 0.305)),
        "S" => Some((4.035, 0.274)),
        "CL" => Some((3.947, 0.227)),
        "AR" => Some((3.868, 0.185)),
        "K" => Some((3.812, 0.035)),
        "CA" => Some((3.399, 0.238)),
        "FE" => Some((2.912, 0.013)),
        "ZN" => Some((2.763, 0.124)),
        "BR" => Some((4.189, 0.251)),
        "I" => Some((4.500, 0.339)),
        _ => None,
    }
}

/// Elements that can participate in hydrogen bonds
const HBOND_ELEMENTS: &[&str] = &["N", "O", "F", "S", "CL"];

/// Hydrogen bond distance correction factor
const HBOND_FACTOR: f32 = 0.79;

/// A rotatable bond group
#[derive(Debug, Clone)]
pub struct RotatableGroup {
    /// Index of central heavy atom
    pub heavy_atom_idx: usize,
    /// Index of bonded heavy atom (rotation axis partner)
    pub bonded_heavy_idx: usize,
    /// Indices of hydrogen atoms in this group
    pub hydrogen_indices: Vec<usize>,
    /// Whether rotation is free (true) or restricted to 180° (false)
    pub is_free: bool,
}

/// Parameters for energy relaxation
#[derive(Debug, Clone)]
pub struct RelaxOptions {
    /// Maximum number of iterations (None = until convergence)
    pub max_iterations: Option<usize>,
    /// Rotation angle increment in radians
    pub angle_increment: f32,
    /// Force cutoff distance in Angstroms
    pub force_cutoff: f32,
}

impl Default for RelaxOptions {
    fn default() -> Self {
        Self {
            max_iterations: None,
            angle_increment: 10.0_f32.to_radians(),
            force_cutoff: 10.0,
        }
    }
}

/// Precomputed interaction parameters for a pair of atoms
#[derive(Debug, Clone)]
struct InteractionPair {
    /// Index of first atom (hydrogen in movable group)
    atom_i: usize,
    /// Index of second atom
    atom_j: usize,
    /// Group index of the hydrogen atom
    group_idx: usize,
    /// Electrostatic parameter: COULOMB_CONSTANT * q_i * q_j
    elec_param: f32,
    /// Combined LJ sigma (with H-bond correction applied)
    sigma: f32,
    /// Combined LJ epsilon
    epsilon: f32,
}

impl InteractionPair {
    /// Compute pairwise energy at given distance.
    ///
    /// Uses our existing physics implementations for consistency.
    #[inline]
    fn energy(&self, distance: f32) -> f32 {
        // Electrostatic term: COULOMB_CONSTANT * q_i * q_j / r
        let e_elec = if self.elec_param.abs() > 1e-10 {
            self.elec_param / distance.max(0.1)
        } else {
            0.0
        };

        // LJ term using our existing function
        let e_lj = compute_lj_energy(distance, self.sigma, self.epsilon);

        e_elec + e_lj
    }
}

/// Energy minimizer for hydrogen relaxation
pub struct EnergyMinimizer {
    /// Interaction pairs to evaluate
    pairs: Vec<InteractionPair>,
    /// Number of rotatable groups
    n_groups: usize,
    /// Group assignment for each atom (-1 = not rotatable)
    groups: Vec<i32>,
    /// Previous group energies for comparison
    prev_group_energies: Option<Vec<f32>>,
    /// Previous coordinates
    prev_coords: Vec<[f32; 3]>,
    /// Mask for deduplicating H-H interactions
    dedup_mask: Vec<bool>,
}

impl EnergyMinimizer {
    /// Create a new energy minimizer.
    ///
    /// # Arguments
    /// * `coords` - Initial atom coordinates
    /// * `elements` - Element symbols for each atom
    /// * `charges` - Partial charges for each atom (will use 0 if None)
    /// * `bonds` - Bond pairs [atom_i, atom_j]
    /// * `groups` - Group assignment for each atom (-1 = fixed)
    /// * `force_cutoff` - Distance cutoff for interactions
    pub fn new(
        coords: &[[f32; 3]],
        elements: &[String],
        charges: Option<&[f32]>,
        bonds: &[[usize; 2]],
        groups: &[i32],
        force_cutoff: f32,
    ) -> Self {
        let n_atoms = coords.len();
        let n_groups = groups
            .iter()
            .filter(|&&g| g >= 0)
            .max()
            .map(|m| m + 1)
            .unwrap_or(0) as usize;

        // Build adjacency list
        let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n_atoms];
        for bond in bonds {
            if bond[0] < n_atoms && bond[1] < n_atoms {
                adjacency[bond[0]].push(bond[1]);
                adjacency[bond[1]].push(bond[0]);
            }
        }

        // Get default charges if not provided
        let default_charges: Vec<f32> = vec![0.0; n_atoms];
        let charges = charges.unwrap_or(&default_charges);

        // Get UFF parameters for all atoms
        let uff_params: Vec<Option<(f32, f32)>> =
            elements.iter().map(|e| get_uff_params(e)).collect();

        // Check which atoms are H-bond elements
        let hbond_mask: Vec<bool> = elements
            .iter()
            .map(|e| HBOND_ELEMENTS.contains(&e.to_uppercase().as_str()))
            .collect();

        // Build interaction pairs
        let mut pairs = Vec::new();
        let force_cutoff_sq = force_cutoff * force_cutoff;

        for i in 0..n_atoms {
            if groups[i] < 0 {
                continue; // Skip non-rotatable atoms
            }

            let Some((sigma_i, eps_i)) = uff_params[i] else {
                continue; // Skip unknown elements
            };

            for j in 0..n_atoms {
                if i == j {
                    continue;
                }
                if groups[i] == groups[j] && groups[j] >= 0 {
                    continue; // Skip atoms in same group
                }

                // Check if directly bonded
                if adjacency[i].contains(&j) {
                    continue;
                }

                // Distance check
                let dx = coords[i][0] - coords[j][0];
                let dy = coords[i][1] - coords[j][1];
                let dz = coords[i][2] - coords[j][2];
                let dist_sq = dx * dx + dy * dy + dz * dz;

                if dist_sq > force_cutoff_sq {
                    continue;
                }

                let Some((sigma_j, eps_j)) = uff_params[j] else {
                    continue;
                };

                // Calculate electrostatic parameter
                let elec_param = COULOMB_CONSTANT * charges[i] * charges[j];

                // Check for hydrogen bonding factor
                let bonded_to_donor = adjacency[i].iter().any(|&b| hbond_mask[b]);
                let is_acceptor = hbond_mask[j];
                let hbond_factor = if bonded_to_donor && is_acceptor {
                    HBOND_FACTOR
                } else {
                    1.0
                };

                // Use Lorentz-Berthelot mixing rules with H-bond correction
                let (sigma_ij, epsilon_ij) = combine_lj_parameters(sigma_i, sigma_j, eps_i, eps_j);
                let sigma = hbond_factor * sigma_ij;

                pairs.push(InteractionPair {
                    atom_i: i,
                    atom_j: j,
                    group_idx: groups[i] as usize,
                    elec_param,
                    sigma,
                    epsilon: epsilon_ij,
                });
            }
        }

        // Build deduplication mask for H-H interactions
        let dedup_mask: Vec<bool> = pairs
            .iter()
            .map(|p| p.atom_j > p.atom_i || groups[p.atom_j] < 0)
            .collect();

        Self {
            pairs,
            n_groups,
            groups: groups.to_vec(),
            prev_group_energies: None,
            prev_coords: coords.to_vec(),
            dedup_mask,
        }
    }

    /// Calculate all pairwise energies for given coordinates
    fn calculate_energies(&self, coords: &[[f32; 3]]) -> Vec<f32> {
        self.pairs
            .iter()
            .map(|pair| {
                let dx = coords[pair.atom_i][0] - coords[pair.atom_j][0];
                let dy = coords[pair.atom_i][1] - coords[pair.atom_j][1];
                let dz = coords[pair.atom_i][2] - coords[pair.atom_j][2];
                let distance = (dx * dx + dy * dy + dz * dz).sqrt();
                pair.energy(distance)
            })
            .collect()
    }

    /// Sum energies by group
    fn sum_by_group(&self, energies: &[f32]) -> Vec<f32> {
        let mut group_energies = vec![0.0; self.n_groups];
        for (pair, &energy) in self.pairs.iter().zip(energies.iter()) {
            if pair.group_idx < self.n_groups {
                group_energies[pair.group_idx] += energy;
            }
        }
        group_energies
    }

    /// Calculate global energy (with deduplication)
    pub fn global_energy(&self, coords: &[[f32; 3]]) -> f32 {
        let energies = self.calculate_energies(coords);
        energies
            .iter()
            .zip(self.dedup_mask.iter())
            .filter(|(_, &mask)| mask)
            .map(|(e, _)| e)
            .sum()
    }

    /// Select minimum energy coordinates for each group.
    ///
    /// Returns (accepted_coords, global_energy, accept_mask)
    pub fn select_minimum(&mut self, next_coords: &[[f32; 3]]) -> (Vec<[f32; 3]>, f32, Vec<bool>) {
        // Calculate previous group energies if not cached
        if self.prev_group_energies.is_none() {
            let prev_energies = self.calculate_energies(&self.prev_coords);
            self.prev_group_energies = Some(self.sum_by_group(&prev_energies));
        }

        // Calculate energies for proposed coordinates
        let next_energies = self.calculate_energies(next_coords);
        let next_group_energies = self.sum_by_group(&next_energies);

        // Decide which groups to accept
        let prev_group_energies = self.prev_group_energies.as_ref().unwrap();
        let accept_mask: Vec<bool> = (0..self.n_groups)
            .map(|i| next_group_energies[i] < prev_group_energies[i])
            .collect();

        // Update coordinates for accepted groups
        for i in 0..self.prev_coords.len() {
            let group = self.groups[i];
            if group >= 0 && accept_mask[group as usize] {
                self.prev_coords[i] = next_coords[i];
            }
        }

        // Recalculate energies for new state
        let final_energies = self.calculate_energies(&self.prev_coords);
        self.prev_group_energies = Some(self.sum_by_group(&final_energies));

        let global_energy: f32 = final_energies
            .iter()
            .zip(self.dedup_mask.iter())
            .filter(|(_, &mask)| mask)
            .map(|(e, _)| e)
            .sum();

        (self.prev_coords.clone(), global_energy, accept_mask)
    }
}

/// Find rotatable bonds in a structure.
///
/// A rotatable bond is a single bond between two heavy atoms where one
/// has only one heavy neighbor and one or more hydrogen neighbors.
pub fn find_rotatable_groups(elements: &[String], bonds: &[[usize; 2]]) -> Vec<RotatableGroup> {
    let n_atoms = elements.len();

    // Build adjacency list
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n_atoms];
    for bond in bonds {
        if bond[0] < n_atoms && bond[1] < n_atoms {
            adjacency[bond[0]].push(bond[1]);
            adjacency[bond[1]].push(bond[0]);
        }
    }

    let is_hydrogen: Vec<bool> = elements.iter().map(|e| e.to_uppercase() == "H").collect();
    let is_nitrogen: Vec<bool> = elements.iter().map(|e| e.to_uppercase() == "N").collect();

    let mut groups = Vec::new();

    for i in 0..n_atoms {
        if is_hydrogen[i] {
            continue;
        }

        let mut hydrogen_indices = Vec::new();
        let mut bonded_heavy_idx: Option<usize> = None;
        let mut is_rotatable = true;

        for &j in &adjacency[i] {
            if is_hydrogen[j] {
                hydrogen_indices.push(j);
            } else if bonded_heavy_idx.is_none() {
                bonded_heavy_idx = Some(j);
            } else {
                // More than one heavy neighbor - not rotatable
                is_rotatable = false;
                break;
            }
        }

        // Must have at least one hydrogen
        if hydrogen_indices.is_empty() {
            is_rotatable = false;
        }

        // Must have exactly one heavy neighbor
        let Some(bonded_idx) = bonded_heavy_idx else {
            continue;
        };

        if !is_rotatable {
            continue;
        }

        // Check rotation freedom
        // Nitrogen attached to double-bonded carbon is restricted to 180°
        let mut is_free = true;
        if is_nitrogen[i] {
            // In a more complete implementation, we'd check bond order
            // For now, assume free rotation for simplicity
        }

        // 180° rotation only makes sense with single hydrogen
        if !is_free && hydrogen_indices.len() > 1 {
            continue;
        }

        groups.push(RotatableGroup {
            heavy_atom_idx: i,
            bonded_heavy_idx: bonded_idx,
            hydrogen_indices,
            is_free,
        });
    }

    groups
}

/// Rotate a point around an axis using Rodrigues' rotation formula.
fn rotate_around_axis(
    point: [f32; 3],
    axis_origin: [f32; 3],
    axis_dir: [f32; 3],
    angle: f32,
) -> [f32; 3] {
    // Translate to origin
    let p = [
        point[0] - axis_origin[0],
        point[1] - axis_origin[1],
        point[2] - axis_origin[2],
    ];

    // Normalize axis
    let len = (axis_dir[0].powi(2) + axis_dir[1].powi(2) + axis_dir[2].powi(2)).sqrt();
    if len < 1e-10 {
        return point; // Degenerate axis
    }
    let k = [axis_dir[0] / len, axis_dir[1] / len, axis_dir[2] / len];

    let cos_a = angle.cos();
    let sin_a = angle.sin();

    // k × p
    let k_cross_p = [
        k[1] * p[2] - k[2] * p[1],
        k[2] * p[0] - k[0] * p[2],
        k[0] * p[1] - k[1] * p[0],
    ];

    // k · p
    let k_dot_p = k[0] * p[0] + k[1] * p[1] + k[2] * p[2];

    // Rodrigues: p' = p*cos(θ) + (k×p)*sin(θ) + k*(k·p)*(1-cos(θ))
    let rotated = [
        p[0] * cos_a + k_cross_p[0] * sin_a + k[0] * k_dot_p * (1.0 - cos_a),
        p[1] * cos_a + k_cross_p[1] * sin_a + k[1] * k_dot_p * (1.0 - cos_a),
        p[2] * cos_a + k_cross_p[2] * sin_a + k[2] * k_dot_p * (1.0 - cos_a),
    ];

    // Translate back
    [
        rotated[0] + axis_origin[0],
        rotated[1] + axis_origin[1],
        rotated[2] + axis_origin[2],
    ]
}

/// Relax hydrogen positions by energy minimization.
///
/// # Arguments
/// * `coords` - Mutable coordinates array
/// * `elements` - Element symbols
/// * `charges` - Optional partial charges
/// * `bonds` - Bond pairs
/// * `options` - Relaxation options
///
/// # Returns
/// * Total number of iterations performed
/// * Final global energy
pub fn relax_hydrogens(
    coords: &mut [[f32; 3]],
    elements: &[String],
    charges: Option<&[f32]>,
    bonds: &[[usize; 2]],
    options: &RelaxOptions,
) -> (usize, f32) {
    // Find rotatable groups
    let rotatable_groups = find_rotatable_groups(elements, bonds);
    eprintln!(
        "[OXIDIZE DEBUG] relax_hydrogens: found {} rotatable groups",
        rotatable_groups.len()
    );
    if rotatable_groups.is_empty() {
        return (0, 0.0);
    }

    // Build group assignment array
    let mut groups = vec![-1i32; coords.len()];
    for (idx, group) in rotatable_groups.iter().enumerate() {
        for &h_idx in &group.hydrogen_indices {
            groups[h_idx] = idx as i32;
        }
    }

    // Initialize minimizer
    let mut minimizer = EnergyMinimizer::new(
        coords,
        elements,
        charges,
        bonds,
        &groups,
        options.force_cutoff,
    );

    // Rotation angles for each group
    let mut angles: Vec<f32> = rotatable_groups
        .iter()
        .map(|g| {
            if g.is_free {
                options.angle_increment
            } else {
                PI
            }
        })
        .collect();

    let mut prev_accepted = true;
    let mut prev_energy = f32::NAN;
    let mut iteration = 0;

    loop {
        // Check iteration limit
        if let Some(max) = options.max_iterations {
            if iteration >= max {
                break;
            }
        }

        // Generate next coordinates by rotating each group
        let mut next_coords = coords.to_vec();
        for (group_idx, group) in rotatable_groups.iter().enumerate() {
            let axis_origin = coords[group.heavy_atom_idx];
            let bonded = coords[group.bonded_heavy_idx];
            let axis_dir = [
                bonded[0] - axis_origin[0],
                bonded[1] - axis_origin[1],
                bonded[2] - axis_origin[2],
            ];

            for &h_idx in &group.hydrogen_indices {
                next_coords[h_idx] =
                    rotate_around_axis(coords[h_idx], axis_origin, axis_dir, angles[group_idx]);
            }
        }

        // Select minimum energy configuration
        let (accepted_coords, curr_energy, accepted_mask) = minimizer.select_minimum(&next_coords);

        // Invert angles for rejected groups
        for (i, &accepted) in accepted_mask.iter().enumerate() {
            if !accepted {
                angles[i] = -angles[i];
            }
        }

        let curr_accepted = accepted_mask.iter().any(|&a| a);

        // Convergence check
        if !curr_accepted && !prev_accepted {
            break;
        }
        if !prev_energy.is_nan() && curr_energy > prev_energy {
            break;
        }

        // Update coordinates
        for (i, coord) in accepted_coords.iter().enumerate() {
            coords[i] = *coord;
        }

        prev_energy = curr_energy;
        prev_accepted = curr_accepted;
        iteration += 1;
    }

    (iteration, prev_energy)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uff_params() {
        assert!(get_uff_params("C").is_some());
        assert!(get_uff_params("H").is_some());
        assert!(get_uff_params("N").is_some());
        assert!(get_uff_params("XX").is_none());

        let (r, eps) = get_uff_params("C").unwrap();
        assert!((r - 3.851).abs() < 0.01);
        assert!((eps - 0.105).abs() < 0.01);
    }

    #[test]
    fn test_find_rotatable_groups() {
        // Simple methane-like: C with 4 H
        let elements: Vec<String> = vec![
            "C".to_string(),
            "H".to_string(),
            "H".to_string(),
            "H".to_string(),
            "H".to_string(),
        ];
        let bonds: Vec<[usize; 2]> = vec![[0, 1], [0, 2], [0, 3], [0, 4]];

        // C has no heavy neighbors, won't have a rotatable group
        let groups = find_rotatable_groups(&elements, &bonds);
        assert!(groups.is_empty(), "Methane has no rotatable bonds");

        // Ethane-like: C-C with H's
        let elements: Vec<String> = vec![
            "C".to_string(), // 0
            "C".to_string(), // 1
            "H".to_string(), // 2
            "H".to_string(), // 3
            "H".to_string(), // 4
            "H".to_string(), // 5
            "H".to_string(), // 6
            "H".to_string(), // 7
        ];
        let bonds: Vec<[usize; 2]> = vec![[0, 1], [0, 2], [0, 3], [0, 4], [1, 5], [1, 6], [1, 7]];

        let groups = find_rotatable_groups(&elements, &bonds);
        assert_eq!(groups.len(), 2, "Ethane has 2 rotatable methyl groups");
    }

    #[test]
    fn test_rotate_around_axis() {
        // Rotate [1, 0, 0] around Z axis by 90 degrees
        let point = [1.0, 0.0, 0.0];
        let origin = [0.0, 0.0, 0.0];
        let axis = [0.0, 0.0, 1.0];

        let rotated = rotate_around_axis(point, origin, axis, PI / 2.0);

        assert!((rotated[0] - 0.0).abs() < 0.001);
        assert!((rotated[1] - 1.0).abs() < 0.001);
        assert!((rotated[2] - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_interaction_pair_energy() {
        let pair = InteractionPair {
            atom_i: 0,
            atom_j: 1,
            group_idx: 0,
            elec_param: 0.0, // No electrostatic
            sigma: 3.0,
            epsilon: 0.1,
        };

        // At equilibrium distance (r = 2^(1/6) * σ), energy should be -ε
        let r_eq = 3.0 * 2.0_f32.powf(1.0 / 6.0);
        let energy = pair.energy(r_eq);
        assert!(
            (energy - (-0.1)).abs() < 0.01,
            "Energy at equilibrium: {}",
            energy
        );

        // At short distance, energy should be very positive (repulsive)
        let energy_short = pair.energy(2.0);
        assert!(energy_short > 0.0, "Energy should be repulsive at r < σ");
    }
}
