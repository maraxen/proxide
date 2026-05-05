//! Distance calculations for protein structures
//!
//! Note: These utilities will be exposed to Python in a future phase.

#![allow(dead_code)]

/// Compute the Euclidean distance between two 3D points
#[inline]
pub fn euclidean_distance(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Compute squared Euclidean distance (faster when only comparing distances)
#[inline]
pub fn euclidean_distance_squared(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

/// Compute pairwise distance matrix for a set of points
/// Returns a flattened upper triangular matrix
pub fn pairwise_distances(coords: &[[f32; 3]]) -> Vec<f32> {
    let n = coords.len();
    let mut distances = Vec::with_capacity(n * (n - 1) / 2);

    for i in 0..n {
        for j in (i + 1)..n {
            distances.push(euclidean_distance(&coords[i], &coords[j]));
        }
    }

    distances
}

/// Compute CA-CA distance matrix for backbone analysis
pub fn ca_distance_matrix(ca_coords: &[[f32; 3]]) -> Vec<Vec<f32>> {
    let n = ca_coords.len();
    let mut matrix = vec![vec![0.0f32; n]; n];

    for i in 0..n {
        for j in (i + 1)..n {
            let dist = euclidean_distance(&ca_coords[i], &ca_coords[j]);
            matrix[i][j] = dist;
            matrix[j][i] = dist;
        }
    }

    matrix
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euclidean_distance() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        assert!((euclidean_distance(&a, &b) - 1.0).abs() < 1e-6);

        let c = [1.0, 1.0, 1.0];
        let d = [2.0, 2.0, 2.0];
        let expected = 3.0f32.sqrt();
        assert!((euclidean_distance(&c, &d) - expected).abs() < 1e-6);
    }

    #[test]
    fn test_pairwise_distances() {
        let coords = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];

        let dists = pairwise_distances(&coords);
        assert_eq!(dists.len(), 3); // 3 choose 2 = 3
        assert!((dists[0] - 1.0).abs() < 1e-6); // (0,1)
        assert!((dists[1] - 1.0).abs() < 1e-6); // (0,2)
        assert!((dists[2] - 2.0f32.sqrt()).abs() < 1e-6); // (1,2)
    }
}
