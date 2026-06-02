//! Proline ring closure algorithm using NeRF + cyclic coordinate descent (CCD).
//!
//! Builds proline sidechain coordinates from χ angles, closing the pyrrolidine ring
//! N-CA-CB-CG-CD-N via CCD if necessary.

use crate::RotlibError;
use super::template::{ResidueTemplate, BondDef};
use proxide_geometry::geometry::nerf::Nerf;

#[allow(dead_code)]
const CCD_MAX_ITERATIONS: usize = 100;
#[allow(dead_code)]
const CCD_TOLERANCE_CD_N: f32 = 0.02; // Å
#[allow(dead_code)]
const CCD_IDEAL_CD_N: f32 = 1.47; // Å
#[allow(dead_code)]
const CCD_CHI_TOLERANCE: f32 = 5.0; // degrees

/// Result of proline sidechain building.
#[derive(Clone, Debug)]
pub struct ProlineCoords {
    /// Coordinates of sidechain atoms in the canonical backbone frame.
    /// Order: [CB, CG, CD] (N, CA, C are backbone).
    pub sidechain: Vec<[f32; 3]>,
    /// Whether ring closure converged (true) or failed (false).
    pub converged: bool,
    /// Number of CCD iterations performed.
    pub ccd_iterations: usize,
    /// Recovered χ1, χ2, χ3 angles from the built coordinates (degrees).
    pub recovered_chi: [f32; 3],
}

/// Builder for proline sidechain coordinates.
pub struct ProlineBuilder {
    #[allow(dead_code)]
    template: ResidueTemplate,
}

impl ProlineBuilder {
    /// Create a new proline builder.
    pub fn new(template: ResidueTemplate) -> Self {
        Self { template }
    }

    /// Build proline sidechain from χ angles using NeRF + CCD ring closure.
    ///
    /// # Arguments
    /// * `backbone_frame` - Coordinates of N, CA, C in the canonical backbone frame.
    ///   Typically, N=[0,0,0], CA=[1.458,0,0], C=[2.0,1.42,0] (built by `frame::backbone_frame`).
    /// * `chi_angles` - [χ1, χ2, χ3] in degrees (from Dunbrack).
    ///
    /// # Returns
    /// ProlineCoords with sidechain atoms (CB, CG, CD) and ring closure status.
    ///
    /// # Failures
    /// - Ring cannot close within tolerance after CCD_MAX_ITERATIONS.
    /// - CCD rotations drift more than ±5° from input χ values.
    pub fn build(&self, backbone_frame: &[[f32; 3]; 3], chi_angles: [f32; 3]) -> Result<ProlineCoords, RotlibError> {
        // Extract backbone atoms in f32.
        let n_f32 = backbone_frame[0];
        let ca_f32 = backbone_frame[1];
        let c_f32 = backbone_frame[2];

        // Get bond definitions.
        let cb_bond = self.template.bonds[4].ok_or_else(|| {
            RotlibError::InvalidFormat("proline template missing CB bond definition".to_string())
        })?;
        let cg_bond = self.template.bonds[5].ok_or_else(|| {
            RotlibError::InvalidFormat("proline template missing CG bond definition".to_string())
        })?;
        let cd_bond = self.template.bonds[6].ok_or_else(|| {
            RotlibError::InvalidFormat("proline template missing CD bond definition".to_string())
        })?;

        // Build initial coordinates using χ angles.
        // CB: χ1 = N-CA-CB-CG
        let cb_f32 = Nerf::place_atom(&[n_f32, ca_f32, c_f32], cb_bond.bond_length, cb_bond.bond_angle_deg, chi_angles[0]);

        // CG: χ2 = CA-CB-CG-CD
        let cg_f32 = Nerf::place_atom(&[ca_f32, cb_f32, n_f32], cg_bond.bond_length, cg_bond.bond_angle_deg, chi_angles[1]);

        // CD: χ3 = CB-CG-CD-N
        let mut cd_f32 = Nerf::place_atom(&[cb_f32, cg_f32, n_f32], cd_bond.bond_length, cd_bond.bond_angle_deg, chi_angles[2]);

        // Run ring closure if needed.
        let mut ccd_iterations = 0;
        let mut recovered_chi = chi_angles;
        let mut cb_final = cb_f32;
        let mut cg_final = cg_f32;

        // Check if ring is closed within tolerance.
        let cd_n_dist = distance_3d(cd_f32, n_f32);
        if (cd_n_dist - CCD_IDEAL_CD_N).abs() > CCD_TOLERANCE_CD_N {
            // Run CCD to close the ring.
            let mut chi2 = chi_angles[1];
            let mut chi3 = chi_angles[2];

            for iter in 0..CCD_MAX_ITERATIONS {
                ccd_iterations = iter + 1;

                // Rotate χ2 (moves CG, CD).
                chi2 += 0.1;
                let cg_new = Nerf::place_atom(&[ca_f32, cb_f32, n_f32], cg_bond.bond_length, cg_bond.bond_angle_deg, chi2);
                let cd_new = Nerf::place_atom(&[cb_f32, cg_new, n_f32], cd_bond.bond_length, cd_bond.bond_angle_deg, chi3);

                let cd_n_dist_new = distance_3d(cd_new, n_f32);
                if (cd_n_dist_new - CCD_IDEAL_CD_N).abs() < (cd_n_dist - CCD_IDEAL_CD_N).abs() {
                    // Improved, keep this step.
                    cg_final = cg_new;
                    cd_f32 = cd_new;
                } else {
                    // Revert and try negative step.
                    chi2 -= 0.2;
                    let cg_new2 = Nerf::place_atom(&[ca_f32, cb_f32, n_f32], cg_bond.bond_length, cg_bond.bond_angle_deg, chi2);
                    let cd_new2 = Nerf::place_atom(&[cb_f32, cg_new2, n_f32], cd_bond.bond_length, cd_bond.bond_angle_deg, chi3);
                    cg_final = cg_new2;
                    cd_f32 = cd_new2;
                }

                // Rotate χ3 (moves CD).
                chi3 += 0.1;
                cd_f32 = Nerf::place_atom(&[cb_f32, cg_final, n_f32], cd_bond.bond_length, cd_bond.bond_angle_deg, chi3);

                let cd_n_dist_final = distance_3d(cd_f32, n_f32);
                if (cd_n_dist_final - CCD_IDEAL_CD_N).abs() <= CCD_TOLERANCE_CD_N {
                    // Check χ drift.
                    if (chi2 - chi_angles[1]).abs() <= CCD_CHI_TOLERANCE && (chi3 - chi_angles[2]).abs() <= CCD_CHI_TOLERANCE {
                        // Converged!
                        recovered_chi[1] = chi2;
                        recovered_chi[2] = chi3;
                        return Ok(ProlineCoords {
                            sidechain: vec![cb_f32, cg_final, cd_f32],
                            converged: true,
                            ccd_iterations,
                            recovered_chi,
                        });
                    }
                }
            }

            // CCD failed to converge.
            return Err(RotlibError::InvalidFormat(format!(
                "proline ring closure failed: CD-N = {:.3} Å after {} iterations",
                distance_3d(cd_f32, n_f32),
                ccd_iterations
            )));
        }

        // Ring is already closed; compute recovered χ values.
        recovered_chi[0] = compute_dihedral(n_f32, ca_f32, cb_f32, cg_f32);
        recovered_chi[1] = compute_dihedral(ca_f32, cb_f32, cg_f32, cd_f32);
        recovered_chi[2] = compute_dihedral(cb_f32, cg_f32, cd_f32, n_f32);

        Ok(ProlineCoords {
            sidechain: vec![cb_f32, cg_f32, cd_f32],
            converged: true,
            ccd_iterations,
            recovered_chi,
        })
    }

}

/// Compute Euclidean distance between two points.
fn distance_3d(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Compute dihedral angle (degrees) from four points.
fn compute_dihedral(p0: [f32; 3], p1: [f32; 3], p2: [f32; 3], p3: [f32; 3]) -> f32 {
    let v1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
    let v2 = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];
    let v3 = [p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2]];

    let n1 = cross(v1, v2);
    let n2 = cross(v2, v3);

    let cross_mag = magnitude(cross(n1, n2));
    let dot_prod = dot(n1, n2);
    let dihedral_rad = dot_prod.atan2(cross_mag);
    dihedral_rad * 180.0 / std::f32::consts::PI
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn magnitude(a: [f32; 3]) -> f32 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::template::proline_template;

    #[test]
    fn test_proline_template_exists() {
        let tmpl = proline_template();
        assert_eq!(tmpl.code, "PRO");
        assert_eq!(tmpl.num_atoms(), 7); // N, CA, C, O, CB, CG, CD
        assert!(tmpl.atom_index("CB").is_some());
        assert!(tmpl.dihedral("χ1").is_some());
    }
}
