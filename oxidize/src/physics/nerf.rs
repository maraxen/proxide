use std::f32::consts::PI;

pub struct Nerf;

// Constants from C++ headers
pub const N_TO_CA_DIST: f32 = 1.4581;
pub const PRO_N_TO_CA_DIST: f32 = 1.353;
pub const CA_TO_C_DIST: f32 = 1.5281;
pub const C_TO_N_DIST: f32 = 1.3311;

impl Nerf {
    /// Calculate the position of the next atom using the NeRF algorithm.
    ///
    /// # Arguments
    ///
    /// * `prev_atoms` - Coordinates of the 3 previous atoms [A, B, C]
    /// * `bond_length` - Length of the C-D bond
    /// * `bond_angle_deg` - Angle B-C-D in degrees
    /// * `torsion_angle_deg` - Dihedral angle A-B-C-D in degrees
    ///
    /// # Returns
    ///
    /// * `[f32; 3]` - Coordinates of atom D
    pub fn place_atom(
        prev_atoms: &[[f32; 3]; 3],
        bond_length: f32,
        bond_angle_deg: f32,
        torsion_angle_deg: f32,
    ) -> [f32; 3] {
        let a = prev_atoms[0];
        let b = prev_atoms[1];
        let c = prev_atoms[2];

        // 1. Obtain vectors
        let ba = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let ab = ba; // label matching C++ "ab" variable name which is b-a

        let bc = [c[0] - b[0], c[1] - b[1], c[2] - b[2]];
        let bc_norm = (bc[0] * bc[0] + bc[1] * bc[1] + bc[2] * bc[2]).sqrt();

        // Unit vector of BC
        let bcn = [bc[0] / bc_norm, bc[1] / bc_norm, bc[2] / bc_norm];

        // Convert angles to radians
        let bond_angle_rad = bond_angle_deg * PI / 180.0;
        let torsion_angle_rad = torsion_angle_deg * PI / 180.0;

        // Current atom D coordinates in the local frame
        let r = bond_length;
        let theta = bond_angle_rad;
        let phi = torsion_angle_rad;

        let d_local = [
            -r * theta.cos(),
            r * phi.cos() * theta.sin(),
            r * phi.sin() * theta.sin(),
        ];

        // 2. Calculate cross product n = AB x BC_norm
        let n_x = ab[1] * bcn[2] - ab[2] * bcn[1];
        let n_y = ab[2] * bcn[0] - ab[0] * bcn[2];
        let n_z = ab[0] * bcn[1] - ab[1] * bcn[0];

        let n_norm = (n_x * n_x + n_y * n_y + n_z * n_z).sqrt();

        // Handle collinear case if needed, but for protein backbone it's rarely perfectly collinear
        let n = if n_norm < 1e-6 {
            // Fallback for collinear (rare in folded proteins, but possible in synthetic data)
            // Pick an arbitrary normal
            if bcn[0].abs() < 0.9 {
                [1.0, 0.0, 0.0]
            } else {
                [0.0, 1.0, 0.0]
            }
        } else {
            [n_x / n_norm, n_y / n_norm, n_z / n_norm]
        };

        // nbc = n x bcn
        let nbc_x = n[1] * bcn[2] - n[2] * bcn[1];
        let nbc_y = n[2] * bcn[0] - n[0] * bcn[2];
        let nbc_z = n[0] * bcn[1] - n[1] * bcn[0];
        let nbc = [nbc_x, nbc_y, nbc_z];

        // Rotation matrix M = [bcn, nbc, n] (columns)
        let m = [
            [bcn[0], nbc[0], n[0]],
            [bcn[1], nbc[1], n[1]],
            [bcn[2], nbc[2], n[2]],
        ];

        // Rotate d_local
        let mut d_rel = [0.0; 3];
        d_rel[0] = m[0][0] * d_local[0] + m[0][1] * d_local[1] + m[0][2] * d_local[2];
        d_rel[1] = m[1][0] * d_local[0] + m[1][1] * d_local[1] + m[1][2] * d_local[2];
        d_rel[2] = m[2][0] * d_local[0] + m[2][1] * d_local[1] + m[2][2] * d_local[2];

        // Add C coordinates
        [d_rel[0] + c[0], d_rel[1] + c[1], d_rel[2] + c[2]]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_place_atom_simple() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let c = [2.0, 0.0, 0.0];
        let prev = [a, b, c];

        let d = Nerf::place_atom(&prev, 1.0, 180.0, 0.0);

        assert!((d[0] - 3.0).abs() < 1e-5);
        assert!(d[1].abs() < 1e-5);
        assert!(d[2].abs() < 1e-5);
    }

    #[test]
    fn test_place_atom_right_angle() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let c = [1.0, 1.0, 0.0];
        let prev = [a, b, c];

        let d = Nerf::place_atom(&prev, 1.0, 90.0, 0.0);

        assert!((d[0] - 0.0).abs() < 1e-5);
        assert!((d[1] - 1.0).abs() < 1e-5);
        assert!(d[2].abs() < 1e-5);
    }
}
