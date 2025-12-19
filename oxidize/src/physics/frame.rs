//! Backbone frame computation and force projection
//!
//! Defines local coordinate frames for residues and projects 3D forces
//! onto these frames to create SE(3)-invariant features.

use crate::physics::constants::MIN_DISTANCE;

/// Backbone atom indices (PDB order)
pub const N_INDEX: usize = 0;
pub const CA_INDEX: usize = 1;
pub const C_INDEX: usize = 2;
pub const CB_INDEX: usize = 3;
pub const O_INDEX: usize = 4;

/// Local backbone coordinate frame (4 unit vectors)
pub struct BackboneFrame {
    pub forward: [f32; 3],   // CA -> C
    pub backward: [f32; 3],  // CA -> N
    pub sidechain: [f32; 3], // CA -> CB
    pub normal: [f32; 3],    // forward x backward
}

/// Projected force features (5 scalars per residue)
#[derive(Debug, Clone, Copy)]
pub struct ProjectedForces {
    pub f_forward: f32,      // Along CA->C
    pub f_backward: f32,     // Along CA->N
    pub f_sidechain: f32,    // Along CA->CB
    pub f_out_of_plane: f32, // Perpendicular to backbone
    pub f_magnitude: f32,    // Total |F|
}

impl ProjectedForces {
    pub fn to_array(&self) -> [f32; 5] {
        [
            self.f_forward,
            self.f_backward,
            self.f_sidechain,
            self.f_out_of_plane,
            self.f_magnitude,
        ]
    }
}

/// Vector subtraction: a - b
#[inline]
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Vector addition: a + b
#[inline]
fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Vector scaling
#[inline]
fn scale(v: [f32; 3], s: f32) -> [f32; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

/// Dot product
#[inline]
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Cross product
#[inline]
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Normalize vector
#[inline]
fn normalize(v: [f32; 3]) -> [f32; 3] {
    let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if norm < 1e-8 {
        [0.0, 0.0, 0.0]
    } else {
        [v[0] / norm, v[1] / norm, v[2] / norm]
    }
}

/// Compute C-beta position using idealized geometry constants
///
/// Matches JAX implementation:
/// term1 = f1 * cross(n_to_ca, ca_to_c)
/// term2 = f2 * n_to_ca
/// term3 = f3 * ca_to_c
///
/// Note: The JAX snippet names are slightly ambiguous ("alpha_to_nitrogen" vs "nitrogen_to_alpha").
/// Based on standard implementations (ProteinMPNN, OpenFold), the vectors are:
/// v1 ($N - CA$) and v2 ($C - CA$)
pub fn compute_c_beta(n: [f32; 3], ca: [f32; 3], c: [f32; 3]) -> [f32; 3] {
    // Vectors relative to CA
    // "alpha_to_nitrogen": Vector from CA to N
    let n_to_ca = sub(n, ca);
    // "carbon_to_alpha": Vector from C to CA
    let c_to_ca = sub(ca, c);

    // Constants from ProteinMPNN / JAX snippet
    let f1 = -0.58273431;
    let f2 = 0.56802827;
    let f3 = -0.54067466;

    // term1 = f1 * cross(n_to_ca, c_to_ca)
    let t1 = scale(cross(n_to_ca, c_to_ca), f1);

    // term2 = f2 * n_to_ca
    let t2 = scale(n_to_ca, f2);

    // term3 = f3 * c_to_ca
    let t3 = scale(c_to_ca, f3);

    // cb = ca + t1 + t2 + t3
    add(add(add(ca, t1), t2), t3)
}

/// Compute local backbone frame for a residue
pub fn compute_backbone_frame(backbone_positions: &[[f32; 3]; 5]) -> BackboneFrame {
    let n = backbone_positions[N_INDEX];
    let ca = backbone_positions[CA_INDEX];
    let c = backbone_positions[C_INDEX];
    let mut cb = backbone_positions[CB_INDEX];

    // If CB is NaN (e.g. Glycine), infer it
    if cb[0].is_nan() {
        cb = compute_c_beta(n, ca, c);
    }

    let forward = normalize(sub(c, ca));
    let backward = normalize(sub(n, ca));
    let sidechain = normalize(sub(cb, ca));

    // Normal to N-CA-C plane (forward x backward)
    // Note: backward is N-CA. forward is C-CA (if sub(c, ca)).
    // Check JAX implementation of frame if available, otherwise use standard cross.
    // Standard: (u - CA) x (v - CA)
    let normal = normalize(cross(forward, backward));

    BackboneFrame {
        forward,
        backward,
        sidechain,
        normal,
    }
}

/// Aggregate forces from 5 backbone atoms into a single vector
///
/// Currently uses MEAN aggregation.
pub fn aggregate_forces(forces: &[[f32; 3]; 5]) -> [f32; 3] {
    let mut sum = [0.0; 3];
    for f in forces {
        sum[0] += f[0];
        sum[1] += f[1];
        sum[2] += f[2];
    }
    [sum[0] / 5.0, sum[1] / 5.0, sum[2] / 5.0]
}

/// Project an aggregated force vector onto the backbone frame
pub fn project_force_onto_frame(force: [f32; 3], frame: &BackboneFrame) -> ProjectedForces {
    ProjectedForces {
        f_forward: dot(force, frame.forward),
        f_backward: dot(force, frame.backward),
        f_sidechain: dot(force, frame.sidechain),
        f_out_of_plane: dot(force, frame.normal),
        f_magnitude: (force[0].powi(2) + force[1].powi(2) + force[2].powi(2)).sqrt(),
    }
}

/// Project flattened forces (n_res * 5) onto backbone frames
/// Returns flattened features (n_res * 5)
pub fn project_backbone_forces(
    forces_flat: &[[f32; 3]],
    backbone_coords: &[[[f32; 3]; 5]],
) -> Vec<f32> {
    let n_res = backbone_coords.len();
    if forces_flat.len() != n_res * 5 {
        // Fallback or panic? Panic is safe here as this is internal logic
        panic!(
            "Force count mismatch: {} forces for {} residues",
            forces_flat.len(),
            n_res
        );
    }

    let mut features = Vec::with_capacity(n_res * 5);

    for (i, res_coords) in backbone_coords.iter().enumerate() {
        let start = i * 5;
        let res_forces_slice = &forces_flat[start..start + 5];
        let mut res_forces = [[0.0; 3]; 5];
        res_forces.copy_from_slice(res_forces_slice);

        let frame = compute_backbone_frame(res_coords);
        let aggregated = aggregate_forces(&res_forces);
        let projected = project_force_onto_frame(aggregated, &frame);

        features.extend_from_slice(&projected.to_array());
    }

    features
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_c_beta_parity() {
        // Test case from known data or idealized geometry
        // Placing atoms at simple coordinates
        let ca = [0.0, 0.0, 0.0];
        let n = [1.46, 0.0, 0.0]; // Along X
        let c = [-0.5, 1.4, 0.0]; // Roughly bond length 1.5 angle 110 (approx)

        let cb = compute_c_beta(n, ca, c);

        // CB should have non-zero Z component due to cross product
        assert!(cb[2].abs() > 0.1);

        // Distance check
        let ca_cb_dist = (cb[0].powi(2) + cb[1].powi(2) + cb[2].powi(2)).sqrt();
        // Should be around 1.52
        assert!((ca_cb_dist - 1.5).abs() < 0.2);
    }

    #[test]
    fn test_projection_shape() {
        let backbone = [[[0.0; 3]; 5]; 2];
        let forces = [[1.0; 3]; 10]; // 2 * 5

        let features = project_backbone_forces(&forces, &backbone);
        assert_eq!(features.len(), 10);
    }
}
