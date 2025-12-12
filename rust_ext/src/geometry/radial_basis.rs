//! Radial basis functions for distance encoding
//!
//! Computes RBF features for backbone atom pairs, used in protein structure learning.
//! Port of priox/geometry/radial_basis.py
//!
//! Note: These utilities will be exposed to Python in a future phase.

#![allow(dead_code)]

/// Number of radial basis functions
pub const RADIAL_BASES: usize = 16;

/// Distance range for RBF centers (Angstroms)
pub const RBF_MIN: f32 = 2.0;
pub const RBF_MAX: f32 = 22.0;

/// RBF width (sigma)
pub const RBF_SIGMA: f32 = (RBF_MAX - RBF_MIN) / RADIAL_BASES as f32;

/// Small epsilon for numerical stability
pub const DISTANCE_EPSILON: f32 = 1e-6;

/// Backbone atom indices: N=0, CA=1, C=2, CB=3, O=4
/// All 25 backbone atom pairs for RBF computation
pub const BACKBONE_PAIRS: [[usize; 2]; 25] = [
    [1, 1], // CA-CA
    [0, 0], // N-N
    [2, 2], // C-C
    [3, 3], // CB-CB
    [4, 4], // O-O
    [1, 0], // CA-N
    [1, 2], // CA-C
    [1, 3], // CA-CB
    [1, 4], // CA-O
    [0, 2], // N-C
    [0, 3], // N-CB
    [0, 4], // N-O
    [4, 2], // O-C
    [4, 3], // O-CB
    [3, 2], // CB-C
    [0, 1], // N-CA
    [2, 1], // C-CA
    [3, 1], // CB-CA
    [4, 1], // O-CA
    [2, 0], // C-N
    [3, 0], // CB-N
    [4, 0], // O-N
    [2, 4], // C-O
    [3, 4], // CB-O
    [2, 3], // C-CB
];

/// Compute RBF centers (lazily computed)
fn rbf_centers() -> [f32; RADIAL_BASES] {
    let mut centers = [0.0f32; RADIAL_BASES];
    for i in 0..RADIAL_BASES {
        centers[i] = RBF_MIN + (RBF_MAX - RBF_MIN) * (i as f32) / (RADIAL_BASES as f32 - 1.0);
    }
    centers
}

/// Apply Gaussian RBF to a distance
///
/// Returns exp(-(d - center)^2 / sigma^2) for each center
#[inline]
fn apply_rbf(distance: f32, centers: &[f32; RADIAL_BASES]) -> [f32; RADIAL_BASES] {
    let mut result = [0.0f32; RADIAL_BASES];
    let sigma_sq = RBF_SIGMA * RBF_SIGMA;

    for (i, &center) in centers.iter().enumerate() {
        let diff = distance - center;
        result[i] = (-diff * diff / sigma_sq).exp();
    }

    result
}

/// Compute radial basis functions for backbone coordinates
///
/// # Arguments
/// * `backbone_coords` - (N_res, 5, 3) backbone coordinates [N, CA, C, CB, O]
/// * `neighbor_indices` - (N_res, K) neighbor indices for each residue
///
/// # Returns
/// Flattened RBF features (N_res, K, 400) where 400 = 25 pairs × 16 bases
pub fn compute_radial_basis(
    backbone_coords: &[[[f32; 3]; 5]], // (N_res, 5, 3)
    neighbor_indices: &[Vec<usize>],   // (N_res, K)
) -> Vec<f32> {
    let n_res = backbone_coords.len();

    if n_res == 0 || neighbor_indices.is_empty() {
        return Vec::new();
    }

    let k = neighbor_indices.get(0).map(|v| v.len()).unwrap_or(0);
    if k == 0 {
        return Vec::new();
    }

    let centers = rbf_centers();
    let features_per_neighbor = BACKBONE_PAIRS.len() * RADIAL_BASES; // 25 × 16 = 400

    // Output shape: (N_res, K, 400)
    let mut result = vec![0.0f32; n_res * k * features_per_neighbor];

    for i in 0..n_res {
        let neighbors = &neighbor_indices[i];

        for (k_idx, &j) in neighbors.iter().enumerate() {
            if j >= n_res {
                continue; // Skip invalid neighbor
            }

            // Compute RBF for each of the 25 backbone pairs
            for (pair_idx, &[atom_a, atom_b]) in BACKBONE_PAIRS.iter().enumerate() {
                // Distance between atom_a of residue i and atom_b of residue j
                let coord_a = &backbone_coords[i][atom_a];
                let coord_b = &backbone_coords[j][atom_b];

                // Skip if any coordinate is NaN (e.g. missing atom like CB in Glycine)
                if coord_a[0].is_nan() || coord_b[0].is_nan() {
                    continue;
                }

                let dist_sq = (coord_a[0] - coord_b[0]).powi(2)
                    + (coord_a[1] - coord_b[1]).powi(2)
                    + (coord_a[2] - coord_b[2]).powi(2);
                let distance = (DISTANCE_EPSILON + dist_sq).sqrt();

                // Apply RBF
                let rbf_values = apply_rbf(distance, &centers);

                // Store in output: result[i, k_idx, pair_idx * 16 + basis_idx]
                let base_idx = i * k * features_per_neighbor
                    + k_idx * features_per_neighbor
                    + pair_idx * RADIAL_BASES;

                for (basis_idx, &rbf_val) in rbf_values.iter().enumerate() {
                    result[base_idx + basis_idx] = rbf_val;
                }
            }
        }
    }

    result
}

/// Compute RBF features with shape information
pub struct RBFResult {
    pub features: Vec<f32>,
    pub shape: (usize, usize, usize), // (N_res, K, 400)
}

/// Compute radial basis functions with shape metadata
pub fn compute_radial_basis_with_shape(
    backbone_coords: &[[[f32; 3]; 5]],
    neighbor_indices: &[Vec<usize>],
) -> RBFResult {
    let n_res = backbone_coords.len();
    let k = neighbor_indices.get(0).map(|v| v.len()).unwrap_or(0);
    let features_dim = BACKBONE_PAIRS.len() * RADIAL_BASES;

    let features = compute_radial_basis(backbone_coords, neighbor_indices);

    RBFResult {
        features,
        shape: (n_res, k, features_dim),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rbf_centers() {
        let centers = rbf_centers();
        assert_eq!(centers.len(), RADIAL_BASES);
        assert!((centers[0] - RBF_MIN).abs() < 1e-6);
        assert!((centers[RADIAL_BASES - 1] - RBF_MAX).abs() < 1e-6);
    }

    #[test]
    fn test_apply_rbf() {
        let centers = rbf_centers();

        // Distance at first center should have high RBF value there
        let rbf = apply_rbf(RBF_MIN, &centers);
        assert!(rbf[0] > 0.9); // Should be close to 1.0
        assert!(rbf[RADIAL_BASES - 1] < 0.1); // Far center should be low

        // Distance at last center
        let rbf = apply_rbf(RBF_MAX, &centers);
        assert!(rbf[RADIAL_BASES - 1] > 0.9);
        assert!(rbf[0] < 0.1);
    }

    #[test]
    fn test_rbf_output_shape() {
        // 3 residues, each with 2 neighbors
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
            [
                [10.0, 0.0, 0.0],
                [11.0, 0.0, 0.0],
                [12.0, 0.0, 0.0],
                [11.0, 1.0, 0.0],
                [12.0, 1.0, 0.0],
            ],
        ];

        let neighbors = vec![vec![1, 2], vec![0, 2], vec![0, 1]];

        let result = compute_radial_basis_with_shape(&backbone, &neighbors);

        assert_eq!(result.shape.0, 3); // N_res
        assert_eq!(result.shape.1, 2); // K
        assert_eq!(result.shape.2, 400); // 25 pairs × 16 bases
        assert_eq!(result.features.len(), 3 * 2 * 400);
    }

    #[test]
    fn test_rbf_empty() {
        let backbone: [[[f32; 3]; 5]; 0] = [];
        let neighbors: Vec<Vec<usize>> = vec![];

        let result = compute_radial_basis(&backbone, &neighbors);
        assert!(result.is_empty());
    }

    #[test]
    fn test_rbf_known_distance() {
        // Two residues at known distance
        let backbone = [
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [2.0, 1.0, 0.0],
            ],
            [
                [10.0, 0.0, 0.0],
                [11.0, 0.0, 0.0],
                [12.0, 0.0, 0.0],
                [11.0, 1.0, 0.0],
                [12.0, 1.0, 0.0],
            ],
        ];

        // CA-CA distance is 10.0 Å
        let neighbors = vec![vec![1], vec![0]];

        let result = compute_radial_basis(&backbone, &neighbors);

        // Check that CA-CA (pair 0) has expected RBF at distance ~10
        // First 16 values are for CA-CA pair of residue 0 -> neighbor 0
        let ca_ca_rbf = &result[0..16];

        // Center closest to 10 should have highest value
        let centers = rbf_centers();
        let expected_center_idx = centers
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| (10.0 - *a).abs().partial_cmp(&(10.0 - *b).abs()).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        // This center should have a high RBF value
        assert!(ca_ca_rbf[expected_center_idx] > 0.5);
    }
}
