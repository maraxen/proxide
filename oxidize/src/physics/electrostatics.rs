//! Electrostatic force calculations using Coulomb's law
//!
//! Port of priox/physics/electrostatics.py
//!
//! Note: These utilities will be exposed to Python in a future phase.

#![allow(dead_code)]

use crate::physics::constants::{COULOMB_CONSTANT, MIN_DISTANCE, MAX_FORCE};

/// Compute Coulomb potential energy: U = sum_ij (k * q_i * q_j / r_ij)
pub fn compute_coulomb_potential(
    target_positions: &[[f32; 3]],
    source_positions: &[[f32; 3]],
    target_charges: &[f32],
    source_charges: &[f32],
    exclude_self: bool,
) -> f32 {
    let n = target_positions.len();
    let m = source_positions.len();
    let mut total = 0.0f32;
    
    for i in 0..n {
        for j in 0..m {
            let dx = target_positions[i][0] - source_positions[j][0];
            let dy = target_positions[i][1] - source_positions[j][1];
            let dz = target_positions[i][2] - source_positions[j][2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            
            if exclude_self && dist < MIN_DISTANCE / 10.0 {
                continue;
            }
            
            let dist_safe = dist.max(MIN_DISTANCE);
            total += COULOMB_CONSTANT * target_charges[i] * source_charges[j] / dist_safe;
        }
    }
    
    total
}

/// Compute Coulomb force vectors at target positions
///
/// F_i = sum_j (k * q_i * q_j / r_ij^2) * r̂_ij
pub fn compute_coulomb_forces(
    target_positions: &[[f32; 3]],
    source_positions: &[[f32; 3]],
    target_charges: &[f32],
    source_charges: &[f32],
    exclude_self: bool,
) -> Vec<[f32; 3]> {
    let n = target_positions.len();
    let m = source_positions.len();
    let mut forces = vec![[0.0f32; 3]; n];
    
    for i in 0..n {
        let mut fx = 0.0f32;
        let mut fy = 0.0f32;
        let mut fz = 0.0f32;
        
        for j in 0..m {
            let dx = source_positions[j][0] - target_positions[i][0];
            let dy = source_positions[j][1] - target_positions[i][1];
            let dz = source_positions[j][2] - target_positions[i][2];
            let dist_sq = dx * dx + dy * dy + dz * dz;
            let dist = dist_sq.sqrt();
            
            // Skip self-interactions
            if exclude_self && dist < MIN_DISTANCE / 10.0 {
                continue;
            }
            
            let dist_safe = dist.max(MIN_DISTANCE);
            
            // Force magnitude: k * q_i * q_j / r^2
            let force_mag = COULOMB_CONSTANT * target_charges[i] * source_charges[j] 
                / (dist_safe * dist_safe);
            
            // Clamp force magnitude
            let force_mag = force_mag.clamp(-MAX_FORCE, MAX_FORCE);
            
            // For Coulomb: positive force_mag means repulsion
            // dx points from target to source, so for repulsion we negate
            // (force pushes target away from source)
            let inv_dist = 1.0 / dist_safe;
            fx -= force_mag * dx * inv_dist;
            fy -= force_mag * dy * inv_dist;
            fz -= force_mag * dz * inv_dist;
        }
        
        forces[i] = [fx, fy, fz];
    }
    
    forces
}

/// Compute Coulomb forces at backbone positions from all atoms
///
/// # Arguments
/// * `backbone_positions` - (n_residues, 5, 3) backbone coordinates [N, CA, C, O, CB]
/// * `all_atom_positions` - (n_atoms, 3) all atom positions
/// * `backbone_charges` - (n_residues * 5) backbone charges flattened
/// * `all_atom_charges` - (n_atoms) all atom charges
///
/// # Returns
/// Force vectors at backbone atoms (n_residues * 5, 3)
pub fn compute_coulomb_forces_at_backbone(
    backbone_positions: &[[[f32; 3]; 5]],  // (n_res, 5, 3)
    all_atom_positions: &[[f32; 3]],        // (n_atoms, 3)
    backbone_charges: &[f32],               // (n_res * 5)
    all_atom_charges: &[f32],               // (n_atoms)
) -> Vec<[f32; 3]> {
    // Flatten backbone to (n_res * 5, 3)
    let backbone_flat: Vec<[f32; 3]> = backbone_positions
        .iter()
        .flat_map(|res| res.iter().cloned())
        .collect();
    
    compute_coulomb_forces(
        &backbone_flat,
        all_atom_positions,
        backbone_charges,
        all_atom_charges,
        true,  // exclude_self
    )
}

/// Project forces onto local backbone frame for SE(3) invariance
///
/// For each residue, project the force vector at each backbone atom onto
/// a local coordinate frame defined by the backbone geometry.
pub fn project_forces_to_backbone_frame(
    forces: &[[f32; 3]],        // (n_res * 5, 3)
    backbone_positions: &[[[f32; 3]; 5]],  // (n_res, 5, 3)
) -> Vec<f32> {
    let n_res = backbone_positions.len();
    let mut projections = Vec::with_capacity(n_res * 5);
    
    for i in 0..n_res {
        let n_pos = &backbone_positions[i][0];
        let ca_pos = &backbone_positions[i][1];
        let _c_pos = &backbone_positions[i][2];
        
        // Local frame: CA-N direction as x-axis
        let x_vec = [
            n_pos[0] - ca_pos[0],
            n_pos[1] - ca_pos[1],
            n_pos[2] - ca_pos[2],
        ];
        let x_mag = (x_vec[0] * x_vec[0] + x_vec[1] * x_vec[1] + x_vec[2] * x_vec[2]).sqrt();
        
        if x_mag < 1e-6 {
            // Degenerate case: just use force magnitude
            for j in 0..5 {
                let f = &forces[i * 5 + j];
                let mag = (f[0] * f[0] + f[1] * f[1] + f[2] * f[2]).sqrt();
                projections.push(mag);
            }
            continue;
        }
        
        let x_unit = [x_vec[0] / x_mag, x_vec[1] / x_mag, x_vec[2] / x_mag];
        
        // Project each force onto this axis (simplified to 1D projection for invariance)
        for j in 0..5 {
            let f = &forces[i * 5 + j];
            let projection = f[0] * x_unit[0] + f[1] * x_unit[1] + f[2] * x_unit[2];
            projections.push(projection);
        }
    }
    
    projections
}

/// Compute SE(3)-invariant electrostatic features for each residue
///
/// # Arguments
/// * `backbone_positions` - (n_res, 5, 3) backbone coordinates
/// * `all_positions` - (n_atoms, 3) all atom positions
/// * `all_charges` - (n_atoms) partial charges
///
/// # Returns
/// Electrostatic features (n_res, 5) - force projections at each backbone atom
pub fn compute_electrostatic_features(
    backbone_positions: &[[[f32; 3]; 5]],
    all_positions: &[[f32; 3]],
    all_charges: &[f32],
) -> Vec<f32> {
    let n_res = backbone_positions.len();
    
    // Use backbone charges as 0.0 (we're computing field at backbone positions)
    let _backbone_charges = vec![0.0f32; n_res * 5];
    
    // Actually, for SE(3) invariant features, we compute field strength, not forces
    // This simplifies to just potential gradient magnitude at each point
    
    let mut features = Vec::with_capacity(n_res * 5);
    
    for i in 0..n_res {
        for j in 0..5 {
            let target = &backbone_positions[i][j];
            
            // Compute electric field magnitude at this point
            let mut ex = 0.0f32;
            let mut ey = 0.0f32;
            let mut ez = 0.0f32;
            
            for (k, source) in all_positions.iter().enumerate() {
                let dx = target[0] - source[0];
                let dy = target[1] - source[1];
                let dz = target[2] - source[2];
                let dist_sq = dx * dx + dy * dy + dz * dz;
                let dist = dist_sq.sqrt();
                
                if dist < MIN_DISTANCE {
                    continue;
                }
                
                // E = k * q / r^2 * r̂
                let field_mag = COULOMB_CONSTANT * all_charges[k] / (dist_sq);
                let field_mag = field_mag.clamp(-MAX_FORCE, MAX_FORCE);
                
                ex += field_mag * dx / dist;
                ey += field_mag * dy / dist;
                ez += field_mag * dz / dist;
            }
            
            // Return field magnitude (SE(3) invariant)
            let field_mag = (ex * ex + ey * ey + ez * ez).sqrt();
            features.push(field_mag);
        }
    }
    
    features
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_coulomb_potential_two_charges() {
        // Two +1 charges at distance 5 Å
        let pos1 = [[0.0, 0.0, 0.0]];
        let pos2 = [[5.0, 0.0, 0.0]];
        let charges1 = [1.0];
        let charges2 = [1.0];
        
        let potential = compute_coulomb_potential(&pos1, &pos2, &charges1, &charges2, false);
        
        // Expected: 332.0636 * 1 * 1 / 5 ≈ 66.4
        assert!((potential - 66.4).abs() < 0.5);
    }
    
    #[test]
    fn test_coulomb_forces_repulsion() {
        // Two +1 charges should repel
        let target = [[0.0, 0.0, 0.0]];
        let source = [[5.0, 0.0, 0.0]];
        let charges = [1.0];
        
        let forces = compute_coulomb_forces(&target, &source, &charges, &charges, false);
        
        // Force on target should point away from source (negative x)
        assert!(forces[0][0] < 0.0, "Like charges should repel: force should be negative x");
    }
    
    #[test]
    fn test_coulomb_forces_attraction() {
        // +1 and -1 charges should attract
        let target = [[0.0, 0.0, 0.0]];
        let source = [[5.0, 0.0, 0.0]];
        let charges_target = [1.0];
        let charges_source = [-1.0];
        
        let forces = compute_coulomb_forces(&target, &source, &charges_target, &charges_source, false);
        
        // Force on target should point toward source (positive x)
        assert!(forces[0][0] > 0.0, "Opposite charges should attract: force should be positive x");
    }
    
    #[test]
    fn test_coulomb_self_exclusion() {
        // Same position should not contribute if exclude_self=true
        let pos = [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]];
        let charges = [1.0, 1.0];
        
        let pot_with_self = compute_coulomb_potential(&pos, &pos, &charges, &charges, false);
        let pot_no_self = compute_coulomb_potential(&pos, &pos, &charges, &charges, true);
        
        // Without self-exclusion, potential is much higher (includes 1/0)
        assert!(pot_no_self < pot_with_self);
    }
    
    #[test]
    fn test_electrostatic_features_shape() {
        let backbone = [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0]],
            [[5.0, 0.0, 0.0], [6.0, 0.0, 0.0], [7.0, 0.0, 0.0], [6.0, 1.0, 0.0], [7.0, 1.0, 0.0]],
        ];
        let all_pos = [[0.0, 0.0, 5.0], [10.0, 0.0, 0.0]];
        let charges = [1.0, -1.0];
        
        let features = compute_electrostatic_features(&backbone, &all_pos, &charges);
        
        assert_eq!(features.len(), 2 * 5);  // n_res * 5 backbone atoms
    }
}
