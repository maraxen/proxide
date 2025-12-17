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
            for i in 0..3 {
                for j in 0..3 {
                    let bytes = [data[pos], data[pos + 1], data[pos + 2], data[pos + 3]];
                    heavy_coords[i][j] = f32::from_le_bytes(bytes);
                    pos += 4;
                }
            }

            // Hydrogen coords
            let mut hydrogen_coords = Vec::with_capacity(num_hydrogens);
            for _ in 0..num_hydrogens {
                let mut coord = [0.0f32; 3];
                for j in 0..3 {
                    let bytes = [data[pos], data[pos + 1], data[pos + 2], data[pos + 3]];
                    coord[j] = f32::from_le_bytes(bytes);
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
    let u = svd.u.unwrap();
    let vt = svd.v_t.unwrap();

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
    for i in 0..3 {
        for j in 0..3 {
            centered_heavy[i][j] = heavy_coords[i][j] - center_coord[j];
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

// ============================================================================
// Helper functions for 3×3 matrix operations
// ============================================================================

fn determinant_3x3(m: &[[f32; 3]; 3]) -> f32 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// Simple SVD for 3×3 matrices using Jacobi rotations.
/// Returns (U, S, V^T)
fn svd_3x3(a: &[[f32; 3]; 3]) -> ([[f32; 3]; 3], [f32; 3], [[f32; 3]; 3]) {
    // For a production implementation, consider using nalgebra crate
    // This is a simplified version using power iteration / Jacobi method

    // A^T × A to get V
    let mut ata = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                ata[i][j] += a[k][i] * a[k][j];
            }
        }
    }

    // Eigendecomposition of A^T A using Jacobi rotations
    let (eigenvalues, v) = jacobi_eigendecomposition(&ata);

    // Singular values are sqrt of eigenvalues
    let s = [
        eigenvalues[0].sqrt(),
        eigenvalues[1].sqrt(),
        eigenvalues[2].sqrt(),
    ];

    // U = A × V × S^-1
    let mut u = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut sum = 0.0f32;
            for k in 0..3 {
                sum += a[i][k] * v[k][j];
            }
            u[i][j] = if s[j] > 1e-10 { sum / s[j] } else { 0.0 };
        }
    }

    // Fix rank deficiency in U to ensure it's a valid rotation base
    let mut zero_cols = Vec::new();
    for j in 0..3 {
        if s[j] <= 1e-10 {
            zero_cols.push(j);
        }
    }

    if zero_cols.len() == 1 {
        // Rank 2: The missing column is cross product of other two
        let missing = zero_cols[0];
        let c1 = (missing + 1) % 3;
        let c2 = (missing + 2) % 3;

        let v1 = [u[0][c1], u[1][c1], u[2][c1]];
        let v2 = [u[0][c2], u[1][c2], u[2][c2]];
        let cross = cross_product(&v1, &v2);
        // Normalize checking is good practice though theoretically v1,v2 are unit orthogonal
        let cross = normalize(&cross);
        u[0][missing] = cross[0];
        u[1][missing] = cross[1];
        u[2][missing] = cross[2];
    } else if zero_cols.len() == 2 {
        // Rank 1: One valid column
        // Build valid orthonormal basis around the one valid column
        let valid = (0..3).find(|&j| s[j] > 1e-10).unwrap_or(0);
        let v_valid = [u[0][valid], u[1][valid], u[2][valid]];
        let v_valid = normalize(&v_valid);

        // Find arbitrary perp
        let mut arb = [1.0, 0.0, 0.0];
        if (v_valid[0].abs() - 1.0).abs() < 0.1 {
            arb = [0.0, 1.0, 0.0];
        }

        let v_perp1 = cross_product(&v_valid, &arb);
        let v_perp1 = normalize(&v_perp1);

        let v_perp2 = cross_product(&v_valid, &v_perp1);
        let v_perp2 = normalize(&v_perp2);

        let m1 = zero_cols[0];
        let m2 = zero_cols[1];

        for i in 0..3 {
            u[i][m1] = v_perp1[i];
        }
        for i in 0..3 {
            u[i][m2] = v_perp2[i];
        }
    } else if zero_cols.len() == 3 {
        // Rank 0: Identity
        u = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }

    // V^T
    let mut vt = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            vt[i][j] = v[j][i];
        }
    }

    (u, s, vt)
}

/// Jacobi eigendecomposition for symmetric 3×3 matrix.
fn jacobi_eigendecomposition(a: &[[f32; 3]; 3]) -> ([f32; 3], [[f32; 3]; 3]) {
    let mut d = *a; // Working copy
    let mut v = [[0.0f32; 3]; 3]; // Eigenvectors

    // Initialize V as identity
    for i in 0..3 {
        v[i][i] = 1.0;
    }

    // Jacobi iterations
    for _ in 0..50 {
        // Find largest off-diagonal element
        let mut max_val = 0.0f32;
        let mut p = 0;
        let mut q = 1;
        for i in 0..3 {
            for j in (i + 1)..3 {
                if d[i][j].abs() > max_val {
                    max_val = d[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }

        if max_val < 1e-10 {
            break;
        }

        // Compute rotation angle
        let theta = if (d[q][q] - d[p][p]).abs() < 1e-10 {
            std::f32::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * d[p][q] / (d[q][q] - d[p][p])).atan()
        };

        let c = theta.cos();
        let s = theta.sin();

        // Apply rotation to D
        let dpp = c * c * d[p][p] - 2.0 * s * c * d[p][q] + s * s * d[q][q];
        let dqq = s * s * d[p][p] + 2.0 * s * c * d[p][q] + c * c * d[q][q];

        d[p][q] = 0.0;
        d[q][p] = 0.0;
        d[p][p] = dpp;
        d[q][q] = dqq;

        for r in 0..3 {
            if r != p && r != q {
                let dpr = c * d[p][r] - s * d[q][r];
                let dqr = s * d[p][r] + c * d[q][r];
                d[p][r] = dpr;
                d[r][p] = dpr;
                d[q][r] = dqr;
                d[r][q] = dqr;
            }
        }

        // Apply rotation to V
        for r in 0..3 {
            let vpr = c * v[r][p] - s * v[r][q];
            let vqr = s * v[r][p] + c * v[r][q];
            v[r][p] = vpr;
            v[r][q] = vqr;
        }
    }

    ([d[0][0], d[1][1], d[2][2]], v)
}

fn cross_product(a: &[f32; 3], b: &[f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(a: &[f32; 3]) -> [f32; 3] {
    let norm = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    if norm > 1e-10 {
        [a[0] / norm, a[1] / norm, a[2] / norm]
    } else {
        [0.0, 0.0, 0.0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fragment_key() {
        let key1 = FragmentKey::new("C", 0, 0, vec![1, 2, 1]);
        let key2 = FragmentKey::new("C", 0, 0, vec![1, 1, 2]);

        // Bond types should be sorted, so these should be equal
        assert_eq!(key1, key2);
    }

    #[test]
    fn test_kabsch_identity() {
        let coords: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        let rotation = kabsch_rotation(&coords, &coords);

        // Should be approximately identity
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (rotation[i][j] - expected).abs() < 0.01,
                    "rotation[{}][{}] = {} (expected {})",
                    i,
                    j,
                    rotation[i][j],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_rotate_point() {
        // 90 degree rotation around Z axis
        let rotation = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let point = [1.0, 0.0, 0.0];
        let rotated = rotate_point(&point, &rotation);

        assert!((rotated[0] - 0.0).abs() < 0.01);
        assert!((rotated[1] - 1.0).abs() < 0.01);
        assert!((rotated[2] - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_determinant() {
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert!((determinant_3x3(&identity) - 1.0).abs() < 0.001);
    }
}
