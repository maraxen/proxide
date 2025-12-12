//! Coordinate transformations and geometric utilities
//!
//! Provides functions for:
//! - Rotation and translation of coordinates
//! - Centering and normalization
//! - Coordinate frame conversions

#![allow(dead_code)]

/// Apply a 3x3 rotation matrix to coordinates
///
/// Rotates all coordinates in-place.
pub fn apply_rotation(coords: &mut [[f32; 3]], rotation: &[[f32; 3]; 3]) {
    for coord in coords.iter_mut() {
        let x = coord[0];
        let y = coord[1];
        let z = coord[2];

        coord[0] = rotation[0][0] * x + rotation[0][1] * y + rotation[0][2] * z;
        coord[1] = rotation[1][0] * x + rotation[1][1] * y + rotation[1][2] * z;
        coord[2] = rotation[2][0] * x + rotation[2][1] * y + rotation[2][2] * z;
    }
}

/// Apply a translation vector to coordinates
///
/// Translates all coordinates in-place.
pub fn apply_translation(coords: &mut [[f32; 3]], translation: &[f32; 3]) {
    for coord in coords.iter_mut() {
        coord[0] += translation[0];
        coord[1] += translation[1];
        coord[2] += translation[2];
    }
}

/// Compute the centroid (center of mass) of coordinates
///
/// Returns the mean position of all points.
pub fn compute_centroid(coords: &[[f32; 3]]) -> [f32; 3] {
    if coords.is_empty() {
        return [0.0, 0.0, 0.0];
    }

    let n = coords.len() as f32;
    let mut centroid = [0.0f32; 3];

    for coord in coords {
        centroid[0] += coord[0];
        centroid[1] += coord[1];
        centroid[2] += coord[2];
    }

    centroid[0] /= n;
    centroid[1] /= n;
    centroid[2] /= n;

    centroid
}

/// Center coordinates on the origin (subtract centroid)
///
/// Modifies coordinates in-place.
pub fn center_coordinates(coords: &mut [[f32; 3]]) {
    let centroid = compute_centroid(coords);
    apply_translation(coords, &[-centroid[0], -centroid[1], -centroid[2]]);
}

/// Compute weighted centroid using masses or other weights
pub fn compute_weighted_centroid(coords: &[[f32; 3]], weights: &[f32]) -> [f32; 3] {
    if coords.is_empty() || weights.is_empty() {
        return [0.0, 0.0, 0.0];
    }

    let mut centroid = [0.0f32; 3];
    let mut total_weight = 0.0f32;

    for (coord, &weight) in coords.iter().zip(weights.iter()) {
        centroid[0] += coord[0] * weight;
        centroid[1] += coord[1] * weight;
        centroid[2] += coord[2] * weight;
        total_weight += weight;
    }

    if total_weight > 0.0 {
        centroid[0] /= total_weight;
        centroid[1] /= total_weight;
        centroid[2] /= total_weight;
    }

    centroid
}

/// Compute rotation matrix from Euler angles (ZYX convention)
///
/// Returns a 3x3 rotation matrix.
pub fn euler_to_rotation_matrix(roll: f32, pitch: f32, yaw: f32) -> [[f32; 3]; 3] {
    let (sr, cr) = roll.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    let (sy, cy) = yaw.sin_cos();

    [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
}

/// Compute rotation matrix for alignment between two vectors
///
/// Returns a matrix that rotates `from` to align with `to`.
pub fn rotation_between_vectors(from: &[f32; 3], to: &[f32; 3]) -> [[f32; 3]; 3] {
    // Normalize vectors
    let from_norm = normalize_vector(from);
    let to_norm = normalize_vector(to);

    // Cross product gives rotation axis
    let cross = cross_product(&from_norm, &to_norm);
    let cos_angle = dot_product(&from_norm, &to_norm);

    // Handle parallel vectors
    if (1.0 - cos_angle.abs()) < 1e-6 {
        if cos_angle > 0.0 {
            // Same direction - identity
            return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        } else {
            // Opposite direction - 180 degree rotation around any perpendicular axis
            return [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]];
        }
    }

    // Rodrigues' rotation formula
    let sin_angle = vector_magnitude(&cross);
    let k = normalize_vector(&cross);

    // K matrix (skew-symmetric)
    let kx = [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]];

    // K^2
    let k2 = matrix_multiply(&kx, &kx);

    // R = I + sin(θ)K + (1-cos(θ))K²
    let mut r = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let identity = if i == j { 1.0 } else { 0.0 };
            r[i][j] = identity + sin_angle * kx[i][j] + (1.0 - cos_angle) * k2[i][j];
        }
    }

    r
}

/// Helper: Normalize a 3D vector
fn normalize_vector(v: &[f32; 3]) -> [f32; 3] {
    let mag = vector_magnitude(v);
    if mag > 0.0 {
        [v[0] / mag, v[1] / mag, v[2] / mag]
    } else {
        [0.0, 0.0, 0.0]
    }
}

/// Helper: Vector magnitude
fn vector_magnitude(v: &[f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// Helper: Dot product
fn dot_product(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Helper: Cross product
fn cross_product(a: &[f32; 3], b: &[f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Helper: 3x3 matrix multiplication
fn matrix_multiply(a: &[[f32; 3]; 3], b: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut result = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_centroid() {
        let coords = [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
        ];

        let centroid = compute_centroid(&coords);
        assert!((centroid[0] - 0.5).abs() < 1e-6);
        assert!((centroid[1] - 0.5).abs() < 1e-6);
        assert!((centroid[2] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_center_coordinates() {
        let mut coords = [[1.0, 1.0, 1.0], [3.0, 1.0, 1.0]];

        center_coordinates(&mut coords);

        // Centroid was (2, 1, 1), so centered should be:
        assert!((coords[0][0] - (-1.0)).abs() < 1e-6);
        assert!((coords[1][0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_apply_translation() {
        let mut coords = [[1.0, 2.0, 3.0]];
        apply_translation(&mut coords, &[1.0, -1.0, 0.5]);

        assert!((coords[0][0] - 2.0).abs() < 1e-6);
        assert!((coords[0][1] - 1.0).abs() < 1e-6);
        assert!((coords[0][2] - 3.5).abs() < 1e-6);
    }

    #[test]
    fn test_identity_rotation() {
        let rotation = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let mut coords = [[1.0, 2.0, 3.0]];

        apply_rotation(&mut coords, &rotation);

        assert!((coords[0][0] - 1.0).abs() < 1e-6);
        assert!((coords[0][1] - 2.0).abs() < 1e-6);
        assert!((coords[0][2] - 3.0).abs() < 1e-6);
    }
}
