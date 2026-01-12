//! Coordinate noising functions for data augmentation
//!
//! Provides Rust-native noising implementations for backbone coordinates.
//! These functions are designed to operate on coordinates before RBF computation.

use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

/// Gaussian backbone noise
///
/// Adds i.i.d. Gaussian noise to each coordinate.
///
/// # Arguments
/// * `coords` - Mutable slice of (N, 3) coordinates to modify in-place
/// * `std` - Standard deviation of the Gaussian noise
/// * `seed` - Random seed for reproducibility
#[allow(dead_code)]
pub fn gaussian_backbone_noise(coords: &mut [[f32; 3]], std: f32, seed: u64) {
    if std <= 0.0 {
        return;
    }

    let mut rng = StdRng::seed_from_u64(seed);

    for coord in coords.iter_mut() {
        for c in coord.iter_mut() {
            // Box-Muller transform for Gaussian samples
            let u1: f32 = rng.gen();
            let u2: f32 = rng.gen();
            let z: f32 = (-2.0_f32 * u1.ln()).sqrt() * (2.0_f32 * std::f32::consts::PI * u2).cos();
            *c += std * z;
        }
    }
}

/// Apply Gaussian noise to backbone coordinates array
///
/// Operates on (N_res, 5, 3) backbone format [N, CA, C, CB, O]
///
/// # Arguments
/// * `backbone` - Mutable (N_res, 5, 3) backbone coordinates
/// * `std` - Standard deviation of Gaussian noise
/// * `seed` - Random seed
pub fn gaussian_backbone_noise_5atom(backbone: &mut [[[f32; 3]; 5]], std: f32, seed: u64) {
    if std <= 0.0 {
        return;
    }

    let mut rng = StdRng::seed_from_u64(seed);

    for residue in backbone.iter_mut() {
        for atom in residue.iter_mut() {
            // Skip NaN atoms (missing, e.g., CB in Glycine)
            if atom[0].is_nan() {
                continue;
            }
            for c in atom.iter_mut() {
                let u1: f32 = rng.gen();
                let u2: f32 = rng.gen();
                let z: f32 =
                    (-2.0_f32 * u1.ln()).sqrt() * (2.0_f32 * std::f32::consts::PI * u2).cos();
                *c += std * z;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_noise_zero_std() {
        let mut coords = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let original = coords;
        gaussian_backbone_noise(&mut coords, 0.0, 42);
        assert_eq!(coords, original);
    }

    #[test]
    fn test_gaussian_noise_modifies_coords() {
        let mut coords = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let original = coords;
        gaussian_backbone_noise(&mut coords, 1.0, 42);
        assert_ne!(coords, original);
    }

    #[test]
    fn test_gaussian_noise_reproducible() {
        let mut coords1 = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let mut coords2 = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        gaussian_backbone_noise(&mut coords1, 1.0, 42);
        gaussian_backbone_noise(&mut coords2, 1.0, 42);
        assert_eq!(coords1, coords2);
    }

    #[test]
    fn test_5atom_skips_nan() {
        let mut backbone = [[
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [f32::NAN, f32::NAN, f32::NAN],
            [4.0, 0.0, 0.0],
        ]];
        gaussian_backbone_noise_5atom(&mut backbone, 1.0, 42);
        // NaN atoms should remain NaN
        assert!(backbone[0][3][0].is_nan());
        // Other atoms should be modified
        assert!(backbone[0][0][0] != 1.0 || backbone[0][0][1] != 0.0 || backbone[0][0][2] != 0.0);
    }
}
