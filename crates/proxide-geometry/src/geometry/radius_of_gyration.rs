//! Radius of gyration
//!
//! Radius of gyration (Rg) measures the compactness of a set of coordinates
//! around their center of mass -- the root-mean-square distance of all
//! points from the centroid, optionally mass-weighted. Used as an MD
//! trajectory frame-quality signal (unfolded/extended conformations show
//! elevated Rg relative to the native/crystal reference).

use super::transforms::{compute_centroid, compute_weighted_centroid};

/// Compute the (unweighted) radius of gyration of a set of coordinates.
///
/// Rg = sqrt(mean(|r_i - r_centroid|^2))
pub fn radius_of_gyration(coords: &[[f32; 3]]) -> f32 {
    if coords.is_empty() {
        return 0.0;
    }

    let centroid = compute_centroid(coords);
    let sum_sq: f32 = coords
        .iter()
        .map(|c| {
            let dx = c[0] - centroid[0];
            let dy = c[1] - centroid[1];
            let dz = c[2] - centroid[2];
            dx * dx + dy * dy + dz * dz
        })
        .sum();

    (sum_sq / coords.len() as f32).sqrt()
}

/// Compute the mass-weighted radius of gyration.
///
/// Rg = sqrt(sum(w_i * |r_i - r_com|^2) / sum(w_i))
pub fn weighted_radius_of_gyration(coords: &[[f32; 3]], weights: &[f32]) -> f32 {
    if coords.is_empty() || weights.is_empty() {
        return 0.0;
    }

    let centroid = compute_weighted_centroid(coords, weights);
    let mut weighted_sum_sq = 0.0f32;
    let mut total_weight = 0.0f32;

    for (c, &w) in coords.iter().zip(weights.iter()) {
        let dx = c[0] - centroid[0];
        let dy = c[1] - centroid[1];
        let dz = c[2] - centroid[2];
        weighted_sum_sq += w * (dx * dx + dy * dy + dz * dz);
        total_weight += w;
    }

    if total_weight > 0.0 {
        (weighted_sum_sq / total_weight).sqrt()
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_radius_of_gyration_unit_cross() {
        // Four points at unit distance from the origin on the x/y axes.
        // Centroid is exactly the origin, so Rg = sqrt(mean(1^2,1^2,1^2,1^2)) = 1.0.
        let coords = [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]];
        let rg = radius_of_gyration(&coords);
        assert!((rg - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_radius_of_gyration_single_point_is_zero() {
        let coords = [[5.0, -3.0, 2.0]];
        assert!(radius_of_gyration(&coords).abs() < 1e-6);
    }

    #[test]
    fn test_radius_of_gyration_empty_is_zero() {
        let coords: [[f32; 3]; 0] = [];
        assert_eq!(radius_of_gyration(&coords), 0.0);
    }

    #[test]
    fn test_radius_of_gyration_scales_with_distance() {
        // Same shape, scaled 2x -> Rg should exactly double.
        let coords_a = [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]];
        let coords_b = [[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]];
        let rg_a = radius_of_gyration(&coords_a);
        let rg_b = radius_of_gyration(&coords_b);
        assert!((rg_b - 2.0 * rg_a).abs() < 1e-6);
    }

    #[test]
    fn test_weighted_radius_of_gyration_matches_unweighted_when_uniform() {
        let coords = [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]];
        let weights = [1.0, 1.0, 1.0, 1.0];
        let rg = radius_of_gyration(&coords);
        let rg_w = weighted_radius_of_gyration(&coords, &weights);
        assert!((rg - rg_w).abs() < 1e-6);
    }

    #[test]
    fn test_weighted_radius_of_gyration_heavier_point_dominates() {
        // A very heavy point at the origin and a light point far away: Rg should
        // be pulled toward the heavy point's position (near zero contribution)
        // rather than the unweighted midpoint.
        let coords = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]];
        let weights = [1000.0, 1.0];
        let rg_w = weighted_radius_of_gyration(&coords, &weights);
        let rg_unweighted = radius_of_gyration(&coords);
        assert!(rg_w < rg_unweighted);
    }
}
