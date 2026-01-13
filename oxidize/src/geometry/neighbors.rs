//! K-nearest neighbor search for protein structures
//!
//! Implements efficient neighbor finding for RBF computation.
//!
//! Note: These utilities will be exposed to Python in a future phase.

#![allow(dead_code)]

use crate::geometry::distances::euclidean_distance_squared;

/// Find K nearest neighbors for each CA atom
///
/// Returns indices of K nearest neighbors for each residue.
/// Self is excluded from neighbors.
pub fn find_k_nearest_neighbors(ca_coords: &[[f32; 3]], k: usize) -> Vec<Vec<usize>> {
    let n = ca_coords.len();

    if n == 0 {
        return Vec::new();
    }

    // Clamp k to valid range
    let k = k.min(n - 1);

    let mut result = Vec::with_capacity(n);

    for i in 0..n {
        // Compute distances to all other points
        let mut distances: Vec<(usize, f32)> = (0..n)
            .filter(|&j| j != i) // Exclude self
            .map(|j| {
                let dist_sq = euclidean_distance_squared(&ca_coords[i], &ca_coords[j]);
                (j, dist_sq)
            })
            .collect();

        // Partial sort to get K smallest
        if k < distances.len() {
            distances.select_nth_unstable_by(k, |a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            distances.truncate(k);
        }

        // Sort final K by distance (for deterministic ordering)
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        result.push(distances.into_iter().map(|(idx, _)| idx).collect());
    }

    result
}

/// Find neighbors within a distance cutoff
///
/// Returns indices of all neighbors within `cutoff` distance.
pub fn find_neighbors_within_cutoff(ca_coords: &[[f32; 3]], cutoff: f32) -> Vec<Vec<usize>> {
    let n = ca_coords.len();
    let cutoff_sq = cutoff * cutoff;

    let mut result = Vec::with_capacity(n);

    for i in 0..n {
        let neighbors: Vec<usize> = (0..n)
            .filter(|&j| j != i)
            .filter(|&j| euclidean_distance_squared(&ca_coords[i], &ca_coords[j]) < cutoff_sq)
            .collect();

        result.push(neighbors);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knn_simple() {
        // 4 points: origin, and along x, y, z axes
        let coords = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ];

        let neighbors = find_k_nearest_neighbors(&coords, 2);

        assert_eq!(neighbors.len(), 4);

        // Origin's closest neighbors should be (1, 0, 0) and (0, 2, 0)
        assert_eq!(neighbors[0].len(), 2);
        assert_eq!(neighbors[0][0], 1); // Closest: x=1
        assert_eq!(neighbors[0][1], 2); // Second: y=2
    }

    #[test]
    fn test_knn_excludes_self() {
        let coords = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];

        let neighbors = find_k_nearest_neighbors(&coords, 5);

        // Each point should only have 1 neighbor (the other point)
        assert_eq!(neighbors[0].len(), 1);
        assert_eq!(neighbors[0][0], 1);
        assert_eq!(neighbors[1].len(), 1);
        assert_eq!(neighbors[1][0], 0);
    }

    #[test]
    fn test_knn_empty() {
        let coords: [[f32; 3]; 0] = [];
        let neighbors = find_k_nearest_neighbors(&coords, 5);
        assert!(neighbors.is_empty());
    }

    #[test]
    fn test_neighbors_within_cutoff() {
        let coords = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0], // Distance 1
            [0.0, 2.0, 0.0], // Distance 2
            [0.0, 0.0, 5.0], // Distance 5
        ];

        let neighbors = find_neighbors_within_cutoff(&coords, 2.5);

        // Origin should have 2 neighbors within 2.5
        assert_eq!(neighbors[0].len(), 2);
        assert!(neighbors[0].contains(&1));
        assert!(neighbors[0].contains(&2));
        assert!(!neighbors[0].contains(&3)); // Distance 5 > 2.5
    }
}
