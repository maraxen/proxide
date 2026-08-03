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

    // NOTE: intentionally `== 0.0`, not `> 0.0`. The prior `> 0.0` guard
    // silently returned 0.0 for a NaN total_weight too (`NaN > 0.0` is false
    // in IEEE-754), which is the worst possible failure direction for an MD
    // frame-quality filter: a NaN mass (e.g. a bad lookup) would read as a
    // maximally-compact frame and be silently KEPT by a >-threshold reject
    // filter instead of surfacing as bad input. `== 0.0` is also false for
    // NaN, so NaN (and any pathological negative-sum weights) now correctly
    // propagates through the division into a NaN result instead.
    if total_weight == 0.0 {
        0.0
    } else {
        (weighted_sum_sq / total_weight).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_radius_of_gyration_unit_cross() {
        // Four points at unit distance from the origin on the x/y axes.
        // Centroid is exactly the origin, so Rg = sqrt(mean(1^2,1^2,1^2,1^2)) = 1.0.
        let coords = [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ];
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
        let coords = [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ];
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

    #[test]
    fn test_weighted_radius_of_gyration_nan_weight_propagates_nan_not_zero() {
        // Regression test: a NaN weight (e.g. a bad mass lookup) must NOT
        // silently read as 0.0 -- for an MD frame-quality filter, a silent
        // 0.0 reads as "maximally compact" and would be KEPT by a
        // >-threshold reject filter instead of surfacing as bad input.
        let coords = [[10.0, 0.0, 0.0], [-10.0, 0.0, 0.0]];
        let weights = [f32::NAN, 1.0];
        let rg = weighted_radius_of_gyration(&coords, &weights);
        assert!(rg.is_nan(), "expected NaN to propagate, got {}", rg);
    }

    #[test]
    fn test_weighted_radius_of_gyration_all_zero_weights_is_zero_not_nan() {
        // The true degenerate case (no mass at all) should still cleanly
        // return 0.0, not NaN -- only NaN/negative-sum inputs should NaN out.
        let coords = [[10.0, 0.0, 0.0], [-10.0, 0.0, 0.0]];
        let weights = [0.0, 0.0];
        let rg = weighted_radius_of_gyration(&coords, &weights);
        assert_eq!(rg, 0.0);
    }

    #[test]
    fn test_radius_of_gyration_all_identical_points_is_zero() {
        let coords = [[3.0, 3.0, 3.0]; 5];
        assert_eq!(radius_of_gyration(&coords), 0.0);
    }

    #[test]
    fn test_radius_of_gyration_nan_coordinate_propagates_nan() {
        let coords = [[f32::NAN, 0.0, 0.0], [1.0, 0.0, 0.0]];
        assert!(radius_of_gyration(&coords).is_nan());
    }
}
