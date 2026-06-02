//! Proline ring closure via angle-relaxation (P3 v2).
//!
//! Builds proline sidechain coordinates from exact χ angles (from Dunbrack).
//! Ring closure is achieved by solving the CB-CG-CD bond angle via 1-D root-finding
//! to satisfy |CD−N| = 1.487 Å (ideal bond length). χ1 and χ2 are preserved exactly;
//! the ring angle is the only degree of freedom relaxed.

use crate::RotlibError;
use super::template::ResidueTemplate;
use proxide_geometry::geometry::nerf::Nerf;

/// Ideal CD-N bond length from CCD PRO.cif (Å).
const CD_N_IDEAL: f32 = 1.487;
/// Tolerance for ring closure |CD−N − ideal| (Å).
const CD_N_TOLERANCE: f32 = 0.001;
/// Max iterations for bisection root-find of CB-CG-CD angle.
const ANGLE_SOLVE_MAX_ITER: usize = 100;
/// Tolerance for bisection convergence (degrees).
const ANGLE_SOLVE_TOL_DEG: f32 = 1e-4;

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
    /// Solved CB-CG-CD ring angle θ (degrees) — the dependent DOF relaxed for closure.
    pub solved_theta: f32,
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

    /// Build proline sidechain from exact χ angles with ring closure via angle relaxation.
    ///
    /// # Algorithm
    /// 1. Place CB: Nerf::place_atom([C, N, CA], CA-CB, N-CA-CB, improper C-N-CA-CB)
    /// 2. Place CG: Nerf::place_atom([N, CA, CB], CB-CG, CA-CB-CG, **χ1 exact**)
    /// 3. Solve CB-CG-CD angle θ via bisection to achieve |CD−N| = 1.487 Å
    ///    - χ2 remains **exact** throughout
    /// 4. Compute recovered χ angles from final coordinates.
    ///
    /// # Arguments
    /// * `backbone_frame` - Coordinates of [N, CA, C] in the canonical backbone frame.
    ///   Typically, N=[0,0,0], CA≈[1.458,0,0], C≈[2.0,1.42,0].
    /// * `chi_angles` - [χ1, χ2, χ3] in degrees (from Dunbrack).
    ///   χ3 is ignored; it is computed as a ring-closure constraint.
    ///
    /// # Returns
    /// ProlineCoords with sidechain atoms (CB, CG, CD) and recovered χ angles.
    /// If ring closure fails: error message includes achieved CD-N distance.
    pub fn build(&self, backbone_frame: &[[f32; 3]; 3], chi_angles: [f32; 3]) -> Result<ProlineCoords, RotlibError> {
        let n_f32 = backbone_frame[0];
        let ca_f32 = backbone_frame[1];
        let c_f32 = backbone_frame[2];

        let cb_bond = self.template.bonds[4].ok_or_else(|| {
            RotlibError::InvalidFormat("proline template missing CB bond definition".to_string())
        })?;
        let cg_bond = self.template.bonds[5].ok_or_else(|| {
            RotlibError::InvalidFormat("proline template missing CG bond definition".to_string())
        })?;
        let cd_bond = self.template.bonds[6].ok_or_else(|| {
            RotlibError::InvalidFormat("proline template missing CD bond definition".to_string())
        })?;

        // Build CB with fixed improper dihedral (C-N-CA-CB).
        let cb_f32 = Nerf::place_atom(
            &[c_f32, n_f32, ca_f32],
            cb_bond.bond_length,
            cb_bond.bond_angle_deg,
            cb_bond.torsion_deg,
        );

        // Build CG with χ1 (exact).
        let cg_f32 = Nerf::place_atom(
            &[n_f32, ca_f32, cb_f32],
            cg_bond.bond_length,
            cg_bond.bond_angle_deg,
            chi_angles[0], // χ1 exact
        );

        // Solve CB-CG-CD angle θ to close the ring.
        // We want to find θ such that |CD(θ) − N| = CD_N_IDEAL.
        let theta_result = solve_cg_cd_angle(
            ca_f32, cb_f32, cg_f32, n_f32,
            cd_bond.bond_length,
            chi_angles[1], // χ2 exact
        )?;

        let (theta_deg, cd_f32) = theta_result;

        // Compute recovered χ angles from final coordinates.
        let chi1_recovered = compute_dihedral(n_f32, ca_f32, cb_f32, cg_f32);
        let chi2_recovered = compute_dihedral(ca_f32, cb_f32, cg_f32, cd_f32);
        let chi3_recovered = compute_dihedral(cb_f32, cg_f32, cd_f32, n_f32);

        let recovered_chi = [chi1_recovered, chi2_recovered, chi3_recovered];

        Ok(ProlineCoords {
            sidechain: vec![cb_f32, cg_f32, cd_f32],
            converged: true,
            ccd_iterations: 0, // Not used with angle-relaxation
            recovered_chi,
            solved_theta: theta_deg,
        })
    }
}

/// Solve the CB-CG-CD bond angle θ to close the proline ring.
///
/// Uses bisection to find θ ∈ [90°, 130°] such that |CD(θ) − N| ≈ CD_N_IDEAL.
/// χ2 (CA-CB-CG-CD dihedral) is held exactly constant.
///
/// # Returns
/// (θ in degrees, CD position in f32) or RotlibError if bisection fails.
fn solve_cg_cd_angle(
    ca: [f32; 3], cb: [f32; 3], cg: [f32; 3], n: [f32; 3],
    cd_bond_len: f32, chi2_deg: f32,
) -> Result<(f32, [f32; 3]), RotlibError> {
    // Bisection bounds: angle must be in physically reasonable range.
    let mut theta_low = 90.0_f32;
    let mut theta_high = 130.0_f32;

    // Evaluate at bounds.
    let cd_low = Nerf::place_atom(&[ca, cb, cg], cd_bond_len, theta_low, chi2_deg);
    let dist_low = distance_3d(cd_low, n);

    let cd_high = Nerf::place_atom(&[ca, cb, cg], cd_bond_len, theta_high, chi2_deg);
    let dist_high = distance_3d(cd_high, n);

    // Check if a solution exists in this interval.
    let error_low = dist_low - CD_N_IDEAL;
    let error_high = dist_high - CD_N_IDEAL;

    if (error_low * error_high) > 0.0 {
        // Errors same sign — no root in [90°, 130°].
        // Return best guess; actual distance will be reported in error.
        let (best_theta, best_cd) = if error_low.abs() < error_high.abs() {
            (theta_low, cd_low)
        } else {
            (theta_high, cd_high)
        };
        return Err(RotlibError::InvalidFormat(format!(
            "proline ring closure: no solution for CB-CG-CD angle in [90,130]°; best θ={:.1}° gives CD-N = {:.3} Å (ideal {:.3})",
            best_theta, distance_3d(best_cd, n), CD_N_IDEAL
        )));
    }

    // Bisection loop.
    for _iter in 0..ANGLE_SOLVE_MAX_ITER {
        let theta_mid = (theta_low + theta_high) / 2.0;

        // Check convergence.
        if (theta_high - theta_low) < ANGLE_SOLVE_TOL_DEG {
            let cd_final = Nerf::place_atom(&[ca, cb, cg], cd_bond_len, theta_mid, chi2_deg);
            let dist_final = distance_3d(cd_final, n);
            if (dist_final - CD_N_IDEAL).abs() <= CD_N_TOLERANCE {
                return Ok((theta_mid, cd_final));
            } else {
                return Err(RotlibError::InvalidFormat(format!(
                    "proline ring closure: bisection converged but tolerance not met. CD-N = {:.3} Å (ideal {:.3})",
                    dist_final, CD_N_IDEAL
                )));
            }
        }

        // Evaluate at midpoint.
        let cd_mid = Nerf::place_atom(&[ca, cb, cg], cd_bond_len, theta_mid, chi2_deg);
        let dist_mid = distance_3d(cd_mid, n);
        let error_mid = dist_mid - CD_N_IDEAL;

        // Update interval.
        if (error_low * error_mid) < 0.0 {
            // Root in [theta_low, theta_mid]
            theta_high = theta_mid;
        } else {
            // Root in [theta_mid, theta_high]
            theta_low = theta_mid;
        }
    }

    // Failed to converge.
    let theta_final = (theta_low + theta_high) / 2.0;
    let cd_final = Nerf::place_atom(&[ca, cb, cg], cd_bond_len, theta_final, chi2_deg);
    Err(RotlibError::InvalidFormat(format!(
        "proline ring closure: bisection max iterations. CD-N = {:.3} Å (ideal {:.3})",
        distance_3d(cd_final, n), CD_N_IDEAL
    )))
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
