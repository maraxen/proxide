//! Angle and dihedral calculations for protein structures
//!
//! Implements backbone dihedral angles (phi, psi, omega) and chi angles.
//!
//! Note: These utilities will be exposed to Python in a future phase.

#![allow(dead_code)]
#![allow(unused_imports)]

use std::f32::consts::PI;

/// Compute the dihedral angle between four points (A, B, C, D)
/// The dihedral is the angle between planes ABC and BCD
/// Returns angle in radians in range [-π, π]
pub fn dihedral_angle(a: &[f32; 3], b: &[f32; 3], c: &[f32; 3], d: &[f32; 3]) -> f32 {
    // Vectors
    let b1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let b2 = [c[0] - b[0], c[1] - b[1], c[2] - b[2]];
    let b3 = [d[0] - c[0], d[1] - c[1], d[2] - c[2]];

    // Cross products
    let n1 = cross(&b1, &b2);
    let n2 = cross(&b2, &b3);

    // Normalize b2 for computing the angle
    let b2_norm = normalize(&b2);

    // m1 = n1 x b2_norm
    let m1 = cross(&n1, &b2_norm);

    // x = n1 · n2
    let x = dot(&n1, &n2);

    // y = m1 · n2
    let y = dot(&m1, &n2);

    y.atan2(x)
}

/// Compute the angle between three points (A, B, C)
/// Returns angle at B in radians
pub fn bond_angle(a: &[f32; 3], b: &[f32; 3], c: &[f32; 3]) -> f32 {
    let ba = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let bc = [c[0] - b[0], c[1] - b[1], c[2] - b[2]];

    let dot_product = dot(&ba, &bc);
    let mag_ba = magnitude(&ba);
    let mag_bc = magnitude(&bc);

    if mag_ba < 1e-6 || mag_bc < 1e-6 {
        return 0.0;
    }

    let cos_angle = (dot_product / (mag_ba * mag_bc)).clamp(-1.0, 1.0);
    cos_angle.acos()
}

/// Compute backbone dihedral angles for a residue
/// Returns (phi, psi, omega) in radians, or None for terminal residues
#[derive(Debug, Clone, Copy)]
pub struct BackboneDihedrals {
    pub phi: Option<f32>,   // C(i-1)-N-CA-C
    pub psi: Option<f32>,   // N-CA-C-N(i+1)
    pub omega: Option<f32>, // CA(i-1)-C(i-1)-N-CA
}

/// Compute backbone dihedrals for a chain of residues
/// backbone_coords should be organized as [[N, CA, C], ...] for each residue
pub fn compute_backbone_dihedrals(backbone_coords: &[[[f32; 3]; 3]]) -> Vec<BackboneDihedrals> {
    let n_residues = backbone_coords.len();
    let mut dihedrals = Vec::with_capacity(n_residues);

    log::debug!("Computing backbone dihedrals for {} residues", n_residues);

    for i in 0..n_residues {
        let n_i = &backbone_coords[i][0];
        let ca_i = &backbone_coords[i][1];
        let c_i = &backbone_coords[i][2];

        // Phi: C(i-1)-N(i)-CA(i)-C(i)
        let phi = if i > 0 {
            let c_prev = &backbone_coords[i - 1][2];
            Some(dihedral_angle(c_prev, n_i, ca_i, c_i))
        } else {
            None
        };

        // Psi: N(i)-CA(i)-C(i)-N(i+1)
        let psi = if i < n_residues - 1 {
            let n_next = &backbone_coords[i + 1][0];
            Some(dihedral_angle(n_i, ca_i, c_i, n_next))
        } else {
            None
        };

        // Omega: CA(i-1)-C(i-1)-N(i)-CA(i)
        let omega = if i > 0 {
            let ca_prev = &backbone_coords[i - 1][1];
            let c_prev = &backbone_coords[i - 1][2];
            Some(dihedral_angle(ca_prev, c_prev, n_i, ca_i))
        } else {
            None
        };

        dihedrals.push(BackboneDihedrals { phi, psi, omega });
    }

    dihedrals
}

// Helper functions

#[inline]
fn cross(a: &[f32; 3], b: &[f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn dot(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn magnitude(v: &[f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

#[inline]
fn normalize(v: &[f32; 3]) -> [f32; 3] {
    let mag = magnitude(v);
    if mag < 1e-6 {
        return [0.0, 0.0, 0.0];
    }
    [v[0] / mag, v[1] / mag, v[2] / mag]
}

// =============================================================================
// f64 versions for high-precision dihedral calculations (MDTraj parity)
// =============================================================================

use std::f64::consts::PI as PI_F64;

/// Compute dihedral angle using f64 precision (for MDTraj parity)
pub fn dihedral_angle_f64(a: &[f64; 3], b: &[f64; 3], c: &[f64; 3], d: &[f64; 3]) -> f64 {
    let b1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let b2 = [c[0] - b[0], c[1] - b[1], c[2] - b[2]];
    let b3 = [d[0] - c[0], d[1] - c[1], d[2] - c[2]];

    let n1 = cross_f64(&b1, &b2);
    let n2 = cross_f64(&b2, &b3);
    let b2_norm = normalize_f64(&b2);
    let m1 = cross_f64(&n1, &b2_norm);

    let x = dot_f64(&n1, &n2);
    let y = dot_f64(&m1, &n2);

    y.atan2(x)
}

/// High-precision backbone dihedrals
#[derive(Debug, Clone, Copy)]
pub struct BackboneDihedrals64 {
    pub phi: Option<f64>,
    pub psi: Option<f64>,
    pub omega: Option<f64>,
}

/// Compute backbone dihedrals with f64 precision
pub fn compute_backbone_dihedrals_f64(
    backbone_coords: &[[[f64; 3]; 3]],
) -> Vec<BackboneDihedrals64> {
    let n_residues = backbone_coords.len();
    let mut dihedrals = Vec::with_capacity(n_residues);

    for i in 0..n_residues {
        let n_i = &backbone_coords[i][0];
        let ca_i = &backbone_coords[i][1];
        let c_i = &backbone_coords[i][2];

        let phi = if i > 0 {
            let c_prev = &backbone_coords[i - 1][2];
            Some(dihedral_angle_f64(c_prev, n_i, ca_i, c_i))
        } else {
            None
        };

        let psi = if i < n_residues - 1 {
            let n_next = &backbone_coords[i + 1][0];
            Some(dihedral_angle_f64(n_i, ca_i, c_i, n_next))
        } else {
            None
        };

        let omega = if i > 0 {
            let ca_prev = &backbone_coords[i - 1][1];
            let c_prev = &backbone_coords[i - 1][2];
            Some(dihedral_angle_f64(ca_prev, c_prev, n_i, ca_i))
        } else {
            None
        };

        dihedrals.push(BackboneDihedrals64 { phi, psi, omega });
    }

    dihedrals
}

// f64 helper functions

#[inline]
fn cross_f64(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn dot_f64(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn magnitude_f64(v: &[f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

#[inline]
fn normalize_f64(v: &[f64; 3]) -> [f64; 3] {
    let mag = magnitude_f64(v);
    if mag < 1e-12 {
        return [0.0, 0.0, 0.0];
    }
    [v[0] / mag, v[1] / mag, v[2] / mag]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dihedral_angle_planar() {
        // Four points in a plane (dihedral should be 0 or π)
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let c = [2.0, 0.0, 0.0];
        let d = [3.0, 0.0, 0.0];

        let angle = dihedral_angle(&a, &b, &c, &d);
        // Collinear points, degenerate case
        assert!(angle.is_finite());
    }

    #[test]
    fn test_dihedral_angle_perpendicular() {
        // 90 degree dihedral
        let a = [0.0, 1.0, 0.0];
        let b = [0.0, 0.0, 0.0];
        let c = [1.0, 0.0, 0.0];
        let d = [1.0, 0.0, 1.0];

        let angle = dihedral_angle(&a, &b, &c, &d);
        assert!((angle - PI / 2.0).abs() < 0.1 || (angle + PI / 2.0).abs() < 0.1);
    }

    #[test]
    fn test_bond_angle() {
        // 90 degree angle
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 0.0, 0.0];
        let c = [0.0, 1.0, 0.0];

        let angle = bond_angle(&a, &b, &c);
        assert!((angle - PI / 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_backbone_dihedrals() {
        // Simple 3-residue backbone
        let backbone = [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], // Residue 0
            [[3.0, 0.0, 0.0], [4.0, 0.0, 0.0], [5.0, 0.0, 0.0]], // Residue 1
            [[6.0, 0.0, 0.0], [7.0, 0.0, 0.0], [8.0, 0.0, 0.0]], // Residue 2
        ];

        let dihedrals = compute_backbone_dihedrals(&backbone);

        assert_eq!(dihedrals.len(), 3);

        // First residue: no phi or omega
        assert!(dihedrals[0].phi.is_none());
        assert!(dihedrals[0].omega.is_none());
        assert!(dihedrals[0].psi.is_some());

        // Last residue: no psi
        assert!(dihedrals[2].psi.is_none());
        assert!(dihedrals[2].phi.is_some());
        assert!(dihedrals[2].omega.is_some());
    }
}
