//! Fragment library for hydrogen position estimation.
//!
//! This module provides hydrogen addition functionality by using a precomputed
//! library of molecular fragments from the RCSB Chemical Component Dictionary.
//!
//! # Algorithm
//!
//! 1. For each heavy atom, identify its chemical environment (element, charge,
//!    stereo, bond types)
//! 2. Look up matching fragment in library
//! 3. Use Kabsch superimposition to align reference heavy atoms to target
//! 4. Apply same rotation to hydrogen positions
//!
//! # References
//!
//! - Kabsch, W. (1976). A solution for the best rotation to relate two sets
//!   of vectors. Acta Cryst. 32, 922-923.
//! - hydride library: https://github.com/biotite-dev/hydride

#![allow(dead_code)]

use std::collections::HashMap;

/// A fragment key identifying a unique chemical environment.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct FragmentKey {
    /// Central atom element (uppercase)
    pub element: String,
    /// Formal charge (-2 to +2)
    pub charge: i8,
    /// Stereochemistry indicator (-1, 0, +1)
    pub stereo: i8,
    /// Bond types to heavy atoms (sorted)
    pub bond_types: Vec<u8>,
}

impl FragmentKey {
    pub fn new(element: &str, charge: i8, stereo: i8, bond_types: Vec<u8>) -> Self {
        let mut sorted_bond_types = bond_types;
        sorted_bond_types.sort();
        Self {
            element: element.to_uppercase(),
            charge,
            stereo,
            bond_types: sorted_bond_types,
        }
    }
}

/// A fragment containing reference coordinates for hydrogen placement.
#[derive(Debug, Clone)]
pub struct Fragment {
    /// Residue name (for debugging)
    pub residue_name: String,
    /// Atom name (for debugging)
    pub atom_name: String,
    /// Reference heavy atom coordinates (3 atoms, centered at origin)
    /// Shape: [3][3] = 3 atoms × (x, y, z)
    pub heavy_coords: [[f32; 3]; 3],
    /// Hydrogen atom coordinates relative to central atom
    pub hydrogen_coords: Vec<[f32; 3]>,
}

/// Fragment library for hydrogen position estimation.
pub struct FragmentLibrary {
    fragments: HashMap<FragmentKey, Fragment>,
}

impl FragmentLibrary {
    /// Create an empty fragment library.
    pub fn new() -> Self {
        Self {
            fragments: HashMap::new(),
        }
    }

    /// Load fragment library from binary data.
    ///
    /// Binary format (little-endian):
    /// - Header: b"FRAG" (4 bytes)
    /// - Version: u32
    /// - Num entries: u32
    /// - For each entry:
    ///   - element: 2 bytes (padded)
    ///   - charge: i8
    ///   - stereo: i8
    ///   - num_bond_types: u8
    ///   - bond_types: [u8; num_bond_types]
    ///   - num_hydrogens: u8
    ///   - heavy_coords: [f32; 9]
    ///   - hydrogen_coords: [f32; num_hydrogens * 3]
    pub fn from_binary(data: &[u8]) -> Result<Self, String> {
        let mut pos = 0;

        // Check header
        if data.len() < 12 {
            return Err("Data too short for header".to_string());
        }
        if &data[0..4] != b"FRAG" {
            return Err("Invalid header magic".to_string());
        }
        pos += 4;

        // Version
        let version = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
        if version != 1 {
            return Err(format!("Unsupported version: {}", version));
        }
        pos += 4;

        // Num entries
        let num_entries =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;

        let mut fragments = HashMap::with_capacity(num_entries);

        for _ in 0..num_entries {
            // Element (2 bytes)
            let element = String::from_utf8_lossy(&data[pos..pos + 2])
                .trim_end_matches('\0')
                .to_string();
            pos += 2;

            // Charge and stereo
            let charge = data[pos] as i8;
            pos += 1;
            let stereo = data[pos] as i8;
            pos += 1;

            // Bond types
            let num_bond_types = data[pos] as usize;
            pos += 1;
            let bond_types: Vec<u8> = data[pos..pos + num_bond_types].to_vec();
            pos += num_bond_types;

            // Num hydrogens
            let num_hydrogens = data[pos] as usize;
            pos += 1;

            // Heavy coords (9 floats = 36 bytes)
            let mut heavy_coords = [[0.0f32; 3]; 3];
            for row in &mut heavy_coords {
                for element in row {
                    let bytes = [data[pos], data[pos + 1], data[pos + 2], data[pos + 3]];
                    *element = f32::from_le_bytes(bytes);
                    pos += 4;
                }
            }

            // Hydrogen coords
            let mut hydrogen_coords = Vec::with_capacity(num_hydrogens);
            for _ in 0..num_hydrogens {
                let mut coord = [0.0f32; 3];
                for item in coord.iter_mut() {
                    let bytes = [data[pos], data[pos + 1], data[pos + 2], data[pos + 3]];
                    *item = f32::from_le_bytes(bytes);
                    pos += 4;
                }
                hydrogen_coords.push(coord);
            }

            let key = FragmentKey::new(&element, charge, stereo, bond_types);
            let fragment = Fragment {
                residue_name: String::new(), // Not stored in binary format
                atom_name: String::new(),
                heavy_coords,
                hydrogen_coords,
            };

            fragments.insert(key, fragment);
        }

        Ok(Self { fragments })
    }

    /// Look up a fragment by its key.
    pub fn get(&self, key: &FragmentKey) -> Option<&Fragment> {
        self.fragments.get(key)
    }

    /// Number of fragments in the library.
    pub fn len(&self) -> usize {
        self.fragments.len()
    }

    /// Check if the library is empty.
    pub fn is_empty(&self) -> bool {
        self.fragments.is_empty()
    }
}

impl Default for FragmentLibrary {
    fn default() -> Self {
        Self::new()
    }
}

/// Kabsch superimposition algorithm.
///
/// Computes the optimal rotation matrix to superimpose mobile coordinates
/// onto fixed coordinates, minimizing RMSD.
///
/// # Arguments
/// * `fixed` - Target coordinates (3 points × 3D)
/// * `mobile` - Reference coordinates to be rotated (3 points × 3D)
///
/// # Returns
/// 3×3 rotation matrix
pub fn kabsch_rotation(fixed: &[[f32; 3]; 3], mobile: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    use nalgebra::{Matrix3, SVD};

    // Build 3x3 matrices from the input arrays
    // Each row is a point, columns are x, y, z
    let fixed_mat = Matrix3::from_rows(&[
        nalgebra::RowVector3::new(fixed[0][0], fixed[0][1], fixed[0][2]),
        nalgebra::RowVector3::new(fixed[1][0], fixed[1][1], fixed[1][2]),
        nalgebra::RowVector3::new(fixed[2][0], fixed[2][1], fixed[2][2]),
    ]);

    let mobile_mat = Matrix3::from_rows(&[
        nalgebra::RowVector3::new(mobile[0][0], mobile[0][1], mobile[0][2]),
        nalgebra::RowVector3::new(mobile[1][0], mobile[1][1], mobile[1][2]),
        nalgebra::RowVector3::new(mobile[2][0], mobile[2][1], mobile[2][2]),
    ]);

    // Cross-covariance matrix H = fixed^T × mobile
    // This matches hydride's convention
    let h = fixed_mat.transpose() * mobile_mat;

    // SVD decomposition: H = U × S × V^T
    let svd = SVD::new(h, true, true);
    let u = svd.u.expect("SVD failed");
    let vt = svd.v_t.expect("SVD failed");

    // Rotation matrix R = U × V^T
    // This rotates mobile coordinates onto fixed coordinates
    let mut r = u * vt;

    // Check for reflection (det(R) should be +1, not -1)
    if r.determinant() < 0.0 {
        // Fix by negating the last column of U
        let mut u_fixed = u;
        for i in 0..3 {
            u_fixed[(i, 2)] = -u_fixed[(i, 2)];
        }
        r = u_fixed * vt;
    }

    // Convert back to [[f32; 3]; 3]
    [
        [r[(0, 0)], r[(0, 1)], r[(0, 2)]],
        [r[(1, 0)], r[(1, 1)], r[(1, 2)]],
        [r[(2, 0)], r[(2, 1)], r[(2, 2)]],
    ]
}

/// Apply rotation matrix to a coordinate.
pub fn rotate_point(point: &[f32; 3], rotation: &[[f32; 3]; 3]) -> [f32; 3] {
    [
        rotation[0][0] * point[0] + rotation[0][1] * point[1] + rotation[0][2] * point[2],
        rotation[1][0] * point[0] + rotation[1][1] * point[1] + rotation[1][2] * point[2],
        rotation[2][0] * point[0] + rotation[2][1] * point[1] + rotation[2][2] * point[2],
    ]
}

/// Calculate hydrogen positions for a single atom.
///
/// # Arguments
/// * `library` - Fragment library
/// * `element` - Central atom element
/// * `charge` - Formal charge
/// * `stereo` - Stereochemistry indicator
/// * `bond_types` - Bond types to heavy atoms
/// * `center_coord` - Position of central atom
/// * `heavy_coords` - Positions of bonded heavy atoms (up to 3)
///
/// # Returns
/// Vector of hydrogen positions, or None if no matching fragment
pub fn calculate_hydrogen_positions(
    library: &FragmentLibrary,
    element: &str,
    charge: i8,
    stereo: i8,
    bond_types: Vec<u8>,
    center_coord: [f32; 3],
    heavy_coords: &[[f32; 3]; 3],
) -> Option<Vec<[f32; 3]>> {
    let key = FragmentKey::new(element, charge, stereo, bond_types);
    let fragment = library.get(&key)?;

    if fragment.hydrogen_coords.is_empty() {
        return Some(Vec::new());
    }

    // Center the target heavy coords (translate so center is at origin)
    let mut centered_heavy: [[f32; 3]; 3] = [[0.0; 3]; 3];
    for (target_row, ref_row) in centered_heavy.iter_mut().zip(heavy_coords.iter()) {
        for (target_val, (ref_val, &center_val)) in target_row
            .iter_mut()
            .zip(ref_row.iter().zip(center_coord.iter()))
        {
            *target_val = *ref_val - center_val;
        }
    }

    // Compute rotation matrix using Kabsch algorithm
    let rotation = kabsch_rotation(&centered_heavy, &fragment.heavy_coords);

    // Rotate reference hydrogen positions and translate to center
    let mut hydrogen_positions = Vec::with_capacity(fragment.hydrogen_coords.len());
    for h_coord in &fragment.hydrogen_coords {
        let rotated = rotate_point(h_coord, &rotation);
        hydrogen_positions.push([
            rotated[0] + center_coord[0],
            rotated[1] + center_coord[1],
            rotated[2] + center_coord[2],
        ]);
    }

    Some(hydrogen_positions)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fragment_key() {
        let key1 = FragmentKey::new("C", 0, 0, vec![1, 2, 1]);
        let key2 = FragmentKey::new("C", 0, 0, vec![1, 1, 2]);
        assert_eq!(key1, key2);
    }

    #[test]
    fn test_kabsch_identity() {
        let coords: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let rotation = kabsch_rotation(&coords, &coords);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((rotation[i][j] - expected).abs() < 0.01);
            }
        }
    }

    #[test]
    fn test_rotate_point() {
        let rotation = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let point = [1.0, 0.0, 0.0];
        let rotated = rotate_point(&point, &rotation);
        assert!((rotated[0] - 0.0).abs() < 0.01);
        assert!((rotated[1] - 1.0).abs() < 0.01);
        assert!((rotated[2] - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_from_binary_invalid_header() {
        let data = b"BADH1234567890123";
        assert!(FragmentLibrary::from_binary(data).is_err());
    }

    #[test]
    fn test_from_binary_empty_library() {
        // FRAG + version 1 + 0 entries = 12 bytes
        let mut data = vec![];
        data.extend_from_slice(b"FRAG");
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes());

        let lib = FragmentLibrary::from_binary(&data).unwrap();
        assert!(lib.is_empty());
    }

    #[test]
    fn test_calculate_hydrogen_positions_rotation() {
        let mut lib = FragmentLibrary::new();
        let key = FragmentKey::new("C", 0, 0, vec![1]);
        // Centered coordinates:
        let heavy_coords = [[1.0, 0.0, 0.0], [-0.5, 0.866, 0.0], [-0.5, -0.866, 0.0]];
        lib.fragments.insert(key.clone(), Fragment {
            residue_name: "test".to_string(),
            atom_name: "C".to_string(),
            heavy_coords,
            hydrogen_coords: vec![[0.0, 0.0, 1.0]],
        });

        // Rotate target heavy coords by 90 deg around Z
        let rotated_heavy = [[0.0, 1.0, 0.0], [-0.866, -0.5, 0.0], [0.866, -0.5, 0.0]];

        let center = [0.0, 0.0, 0.0];
        let h_pos = calculate_hydrogen_positions(&lib, "C", 0, 0, vec![1], center, &rotated_heavy).unwrap();

        assert_eq!(h_pos.len(), 1);
        // Original H was at [0, 0, 1], rotating by 90 deg around Z leaves it at [0, 0, 1]
        // Wait, rotating heavy_coords by 90 around Z, rotates [0,0,1] around Z?
        // Z is the axis of rotation, so H should remain at [0,0,1].
        assert!((h_pos[0][0] - 0.0).abs() < 0.01);
        assert!((h_pos[0][1] - 0.0).abs() < 0.01);
        assert!((h_pos[0][2] - 1.0).abs() < 0.01);
    }
}
