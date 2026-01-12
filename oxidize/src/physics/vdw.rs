//! Van der Waals (Lennard-Jones) force calculations
//!
//! Port of proxide/physics/vdw.py
//!
//! Note: These utilities will be exposed to Python in a future phase.

#![allow(dead_code)]

use crate::physics::constants::{DEFAULT_EPSILON, DEFAULT_SIGMA, MAX_FORCE, MIN_DISTANCE};

/// Combine LJ parameters using Lorentz-Berthelot rules
///
/// σ_ij = (σ_i + σ_j) / 2
/// ε_ij = sqrt(ε_i * ε_j)
#[inline]
pub fn combine_lj_parameters(
    sigma_i: f32,
    sigma_j: f32,
    epsilon_i: f32,
    epsilon_j: f32,
) -> (f32, f32) {
    let sigma_ij = (sigma_i + sigma_j) / 2.0;
    let epsilon_ij = (epsilon_i * epsilon_j).sqrt();
    (sigma_ij, epsilon_ij)
}

/// Compute LJ potential energy: U = 4ε[(σ/r)^12 - (σ/r)^6]
#[inline]
pub fn compute_lj_energy(distance: f32, sigma: f32, epsilon: f32) -> f32 {
    let dist_safe = distance.max(MIN_DISTANCE);
    let sigma_over_r = sigma / dist_safe;
    let sigma_6 = sigma_over_r.powi(6);
    let sigma_12 = sigma_6 * sigma_6;
    4.0 * epsilon * (sigma_12 - sigma_6)
}

/// Compute LJ force magnitude: F = 24ε/r [2(σ/r)^12 - (σ/r)^6]
///
/// Positive = repulsive, negative = attractive
#[inline]
pub fn compute_lj_force_magnitude(distance: f32, sigma: f32, epsilon: f32) -> f32 {
    let dist_safe = distance.max(MIN_DISTANCE);
    let sigma_over_r = sigma / dist_safe;
    let sigma_6 = sigma_over_r.powi(6);
    let sigma_12 = sigma_6 * sigma_6;
    24.0 * epsilon * (2.0 * sigma_12 - sigma_6) / dist_safe
}

/// Compute LJ forces at target positions from source atoms
///
/// # Arguments
/// * `target_positions` - Target atom positions
/// * `source_positions` - Source atom positions
/// * `target_sigmas` - LJ sigma for targets
/// * `target_epsilons` - LJ epsilon for targets
/// * `source_sigmas` - LJ sigma for sources
/// * `source_epsilons` - LJ epsilon for sources
///
/// # Returns
/// Force vectors at each target position
pub fn compute_lj_forces(
    target_positions: &[[f32; 3]],
    source_positions: &[[f32; 3]],
    target_sigmas: &[f32],
    target_epsilons: &[f32],
    source_sigmas: &[f32],
    source_epsilons: &[f32],
    exclude_self: bool,
) -> Vec<[f32; 3]> {
    let n = target_positions.len();
    let mut forces = vec![[0.0f32; 3]; n];

    for (((pos_i, sigma_i), epsilon_i), force) in target_positions
        .iter()
        .zip(target_sigmas.iter())
        .zip(target_epsilons.iter())
        .zip(forces.iter_mut())
    {
        let mut fx = 0.0f32;
        let mut fy = 0.0f32;
        let mut fz = 0.0f32;

        for ((pos_j, sigma_j), epsilon_j) in source_positions
            .iter()
            .zip(source_sigmas.iter())
            .zip(source_epsilons.iter())
        {
            let dx = pos_j[0] - pos_i[0];
            let dy = pos_j[1] - pos_i[1];
            let dz = pos_j[2] - pos_i[2];
            let dist_sq = dx * dx + dy * dy + dz * dz;
            let dist = dist_sq.sqrt();

            // Skip self-interactions
            if exclude_self && dist < MIN_DISTANCE / 10.0 {
                continue;
            }

            // Combine LJ parameters
            let (sigma_ij, epsilon_ij) =
                combine_lj_parameters(*sigma_i, *sigma_j, *epsilon_i, *epsilon_j);

            // Skip if no interaction
            if epsilon_ij < 1e-10 {
                continue;
            }

            // Compute force magnitude
            let force_mag = compute_lj_force_magnitude(dist, sigma_ij, epsilon_ij);
            let force_mag = force_mag.clamp(-MAX_FORCE, MAX_FORCE);

            // For LJ: positive force_mag means repulsion
            // dx points from target to source, so for repulsion we negate
            // (force pushes target away from source)
            let dist_safe = dist.max(MIN_DISTANCE);
            let inv_dist = 1.0 / dist_safe;
            fx -= force_mag * dx * inv_dist;
            fy -= force_mag * dy * inv_dist;
            fz -= force_mag * dz * inv_dist;
        }

        *force = [fx, fy, fz];
    }

    forces
}

/// Compute LJ forces at backbone positions from all atoms
pub fn compute_lj_forces_at_backbone(
    backbone_positions: &[[[f32; 3]; 5]], // (n_res, 5, 3)
    all_atom_positions: &[[f32; 3]],      // (n_atoms, 3)
    backbone_sigmas: &[f32],              // (n_res * 5)
    backbone_epsilons: &[f32],            // (n_res * 5)
    all_atom_sigmas: &[f32],              // (n_atoms)
    all_atom_epsilons: &[f32],            // (n_atoms)
) -> Vec<[f32; 3]> {
    // Flatten backbone to (n_res * 5, 3)
    let backbone_flat: Vec<[f32; 3]> = backbone_positions
        .iter()
        .flat_map(|res| res.iter().cloned())
        .collect();

    compute_lj_forces(
        &backbone_flat,
        all_atom_positions,
        backbone_sigmas,
        backbone_epsilons,
        all_atom_sigmas,
        all_atom_epsilons,
        true, // exclude_self
    )
}

/// Compute SE(3)-invariant vdW features for each residue
///
/// Returns force magnitude at each backbone atom (n_res, 5)
pub fn compute_vdw_features(
    backbone_positions: &[[[f32; 3]; 5]],
    all_positions: &[[f32; 3]],
    all_sigmas: &[f32],
    all_epsilons: &[f32],
) -> Vec<f32> {
    let mut features = Vec::with_capacity(backbone_positions.len() * 5);

    for res_pos in backbone_positions {
        for target in res_pos {
            // Sum LJ potential at this point (as SE(3) invariant feature)
            let mut energy = 0.0f32;

            for (source, (sigma, epsilon)) in all_positions
                .iter()
                .zip(all_sigmas.iter().zip(all_epsilons.iter()))
            {
                let dx = target[0] - source[0];
                let dy = target[1] - source[1];
                let dz = target[2] - source[2];
                let dist_sq = dx * dx + dy * dy + dz * dz;

                if dist_sq < MIN_DISTANCE * MIN_DISTANCE {
                    continue;
                }

                let dist = dist_sq.sqrt();

                // Use default backbone sigma/epsilon combined with source params
                let (sigma_ij, epsilon_ij) =
                    combine_lj_parameters(DEFAULT_SIGMA, *sigma, DEFAULT_EPSILON, *epsilon);

                energy += compute_lj_energy(dist, sigma_ij, epsilon_ij);
            }

            features.push(energy);
        }
    }

    features
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_combine_lj_parameters() {
        let (sigma, epsilon) = combine_lj_parameters(3.0, 4.0, 0.1, 0.4);
        assert!((sigma - 3.5).abs() < 1e-6);
        assert!((epsilon - 0.2).abs() < 1e-6); // sqrt(0.04) = 0.2
    }

    #[test]
    fn test_lj_energy_at_equilibrium() {
        // At r = 2^(1/6) * σ, energy should be at minimum (-ε)
        let sigma = 3.5;
        let epsilon = 0.1;
        let r_min = sigma * 2.0f32.powf(1.0 / 6.0);

        let energy = compute_lj_energy(r_min, sigma, epsilon);
        assert!((energy - (-epsilon)).abs() < 0.001);
    }

    #[test]
    fn test_lj_force_sign() {
        let sigma = 3.5;
        let epsilon = 0.1;

        // At distance < σ, force should be repulsive (positive)
        let force_close = compute_lj_force_magnitude(2.0, sigma, epsilon);
        assert!(
            force_close > 0.0,
            "Force should be repulsive at short distance"
        );

        // At distance = equilibrium, force should be ~0
        let r_min = sigma * 2.0f32.powf(1.0 / 6.0);
        let force_eq = compute_lj_force_magnitude(r_min, sigma, epsilon);
        assert!(
            force_eq.abs() < 0.01,
            "Force should be near zero at equilibrium"
        );

        // At distance > equilibrium but < cutoff, force should be attractive (negative)
        let force_far = compute_lj_force_magnitude(5.0, sigma, epsilon);
        assert!(
            force_far < 0.0,
            "Force should be attractive at medium distance"
        );
    }

    #[test]
    fn test_lj_forces_repulsion() {
        // Two atoms close together should repel
        let target = [[0.0, 0.0, 0.0]];
        let source = [[2.0, 0.0, 0.0]]; // Close, should repel
        let sigmas = [3.5];
        let epsilons = [0.1];

        let forces = compute_lj_forces(
            &target, &source, &sigmas, &epsilons, &sigmas, &epsilons, false,
        );

        // Force should push target away from source (negative x)
        assert!(forces[0][0] < 0.0, "Close atoms should repel");
    }

    #[test]
    fn test_vdw_features_shape() {
        let backbone = [
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [2.0, 1.0, 0.0],
            ],
            [
                [5.0, 0.0, 0.0],
                [6.0, 0.0, 0.0],
                [7.0, 0.0, 0.0],
                [6.0, 1.0, 0.0],
                [7.0, 1.0, 0.0],
            ],
        ];
        let all_pos = [[0.0, 0.0, 5.0], [10.0, 0.0, 0.0]];
        let sigmas = [3.0, 3.5];
        let epsilons = [0.1, 0.15];

        let features = compute_vdw_features(&backbone, &all_pos, &sigmas, &epsilons);

        assert_eq!(features.len(), 2 * 5); // n_res * 5 backbone atoms
    }
}
