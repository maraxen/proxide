//! Empirical Adversarial Verification Suite for `proxide-rotlib`
//!
//! Author: challenger_m3_rotlib (teamwork_challenger)
//! Milestone: M3 (Task ID: 261102_proxide-mosaist-cleanroom)

#[path = "helpers.rs"]
mod helpers;
use helpers::{write_minimal_lib, BinSpec, RotSpec};
use proxide_rotlib::frame::backbone_frame;
use proxide_rotlib::geometry::ccd_parser::{
    parse_ccd_ic_table_raw, FallbackKind,
};
use proxide_rotlib::{RotamerLibrary, RotlibError};

/// Build a 3x3 grid library for testing dihedral bin routing.
/// Grid centers: phi in [-120, 0, 120], psi in [-120, 0, 120].
/// default_bin is at (0, 0) which is index 1 * 3 + 1 = 4.
fn make_test_3x3_lib() -> (tempfile::NamedTempFile, RotamerLibrary) {
    let phi_centers = [-120.0f32, 0.0, 120.0];
    let psi_centers = [-120.0f32, 0.0, 120.0];
    let mut bins = Vec::new();
    for &phi in &phi_centers {
        for &psi in &psi_centers {
            let freq = if phi == 0.0 && psi == 0.0 { 100.0 } else { 1.0 };
            bins.push(BinSpec { phi, psi, freq });
        }
    }
    let rotamers = vec![RotSpec {
        prob: 1.0,
        coords: vec![[1.0, 0.0, 0.0]],
    }];
    let tmp = write_minimal_lib("TST", &["CB"], &bins, &rotamers);
    let lib = RotamerLibrary::load_pb(tmp.path()).unwrap();
    (tmp, lib)
}

// ============================================================================
// TASK 1: DIHEDRAL QUERY API ADVERSARIAL CHALLENGES
// ============================================================================

#[test]
fn test_challenger_dihedral_none_none_routes_to_default_bin() {
    let (_t, lib) = make_test_3x3_lib();
    let bin = lib.backbone_bin("TST", None, None, false).unwrap();
    assert_eq!(bin, 4, "None, None must route to default_bin (index 4)");
}

#[test]
fn test_challenger_dihedral_some_zero_zero_routes_to_center_bin() {
    let (_t, lib) = make_test_3x3_lib();
    // (0.0, 0.0) matches phi_ind=1, psi_ind=1 -> linear index 1*3 + 1 = 4
    let bin = lib
        .backbone_bin("TST", Some(0.0), Some(0.0), false)
        .unwrap();
    assert_eq!(bin, 4, "Some(0.0), Some(0.0) must route to center bin 4");
}

#[test]
fn test_challenger_dihedral_boundary_180_wraps_correctly() {
    let (_t, lib) = make_test_3x3_lib();
    // Dihedrals are periodic: -180.0 and +180.0 normalize to -180.0.
    // In our [-120, 0, 120] grid, -180.0 is equidistant between -120 and +120 (diff 60 deg each).
    // Let's verify that Some(-180.0) and Some(180.0) route to the exact same bin:
    let bin_neg180_pos180 = lib
        .backbone_bin("TST", Some(-180.0), Some(180.0), false)
        .unwrap();
    let bin_pos180_neg180 = lib
        .backbone_bin("TST", Some(180.0), Some(-180.0), false)
        .unwrap();
    let bin_neg180_neg180 = lib
        .backbone_bin("TST", Some(-180.0), Some(-180.0), false)
        .unwrap();
    let bin_pos180_pos180 = lib
        .backbone_bin("TST", Some(180.0), Some(180.0), false)
        .unwrap();

    assert_eq!(
        bin_neg180_pos180, bin_pos180_neg180,
        "-180/+180 must produce identical bin index due to modular periodicity"
    );
    assert_eq!(
        bin_neg180_neg180, bin_pos180_pos180,
        "(-180,-180) and (+180,+180) must produce identical bin index"
    );
    assert_eq!(
        bin_neg180_pos180, bin_neg180_neg180,
        "All +/-180 combinations must map identically"
    );
}

#[test]
fn test_challenger_dihedral_large_angles_periodicity_vs_sentinel_hack() {
    let (_t, lib) = make_test_3x3_lib();
    // 360.0 degrees is equivalent to 0.0 degrees.
    // 360.0 < 999.0, so angle_to_standard(360.0) = 0.0 -> matches bin 4.
    let bin_360 = lib
        .backbone_bin("TST", Some(360.0), Some(360.0), false)
        .unwrap();
    assert_eq!(bin_360, 4, "360.0 must wrap to 0.0 and route to bin 4");

    // What about 1080.0 degrees? 1080 = 3 * 360.0 = 0.0 degrees.
    // BUT because rotlib.rs has `if p < 999.0 && s < 999.0`,
    // 1080.0 >= 999.0 triggers the sentinel fallback `_ => default_bin`!
    // Since default_bin is also 4, let's test an angle that should NOT route to default_bin!
    // For example: phi = -120.0 (bin 0). With period: -120.0 + 360.0 = 240.0 (bin 0).
    // With period: -120.0 + 3 * 360.0 = 960.0 (bin 0).
    // With period: -120.0 + 4 * 360.0 = 1320.0 (bin 0).
    let bin_neg120 = lib
        .backbone_bin("TST", Some(-120.0), Some(0.0), false)
        .unwrap();
    assert_eq!(bin_neg120, 1, "-120, 0 must route to phi_ind=0, psi_ind=1 -> bin 1");

    let bin_240 = lib
        .backbone_bin("TST", Some(240.0), Some(0.0), false)
        .unwrap();
    assert_eq!(bin_240, 1, "240.0 (-120 mod 360) must route to bin 1");

    // Now test 1320.0 (-120.0 + 4*360.0):
    // Under Debt #1897 fix, 1320.0 is normalized via angle_to_standard to -120.0,
    // correctly selecting bin 1 (phi_ind=0, psi_ind=1 -> 0*3 + 1 = 1).
    let bin_1320 = lib
        .backbone_bin("TST", Some(1320.0), Some(0.0), false)
        .unwrap();
    assert_eq!(
        bin_1320, 1,
        "1320.0 (-120 deg mod 360) must correctly route to bin 1 via angle_to_standard"
    );
}

#[test]
fn test_challenger_dihedral_nan_handling() {
    let (_t, lib) = make_test_3x3_lib();
    // Under Debt #1897 and AGENTS.md, NaN dihedrals must raise an error instead of routing to default_bin.
    let res_phi_nan = lib.backbone_bin("TST", Some(f64::NAN), Some(0.0), false);
    assert!(
        matches!(res_phi_nan, Err(RotlibError::InvalidFormat(_))),
        "Some(f64::NAN) for phi must return Err(RotlibError::InvalidFormat), got {:?}",
        res_phi_nan
    );

    let res_psi_nan = lib.backbone_bin("TST", Some(0.0), Some(f64::NAN), false);
    assert!(
        matches!(res_psi_nan, Err(RotlibError::InvalidFormat(_))),
        "Some(f64::NAN) for psi must return Err(RotlibError::InvalidFormat), got {:?}",
        res_psi_nan
    );

    let res_both_nan = lib.backbone_bin("TST", Some(f64::NAN), Some(f64::NAN), false);
    assert!(
        matches!(res_both_nan, Err(RotlibError::InvalidFormat(_))),
        "Some(f64::NAN) for both must return Err(RotlibError::InvalidFormat), got {:?}",
        res_both_nan
    );

    let res_none_nan = lib.backbone_bin("TST", None, Some(f64::NAN), false);
    assert!(
        matches!(res_none_nan, Err(RotlibError::InvalidFormat(_))),
        "None, Some(f64::NAN) must return Err(RotlibError::InvalidFormat), got {:?}",
        res_none_nan
    );

    // Also verify INFINITY and NEG_INFINITY are rejected with InvalidFormat
    let res_inf = lib.backbone_bin("TST", Some(f64::INFINITY), Some(0.0), false);
    assert!(
        matches!(res_inf, Err(RotlibError::InvalidFormat(_))),
        "Some(f64::INFINITY) must return Err(RotlibError::InvalidFormat), got {:?}",
        res_inf
    );

    let res_neg_inf = lib.backbone_bin("TST", Some(0.0), Some(f64::NEG_INFINITY), false);
    assert!(
        matches!(res_neg_inf, Err(RotlibError::InvalidFormat(_))),
        "Some(f64::NEG_INFINITY) must return Err(RotlibError::InvalidFormat), got {:?}",
        res_neg_inf
    );
}

#[test]
fn test_challenger_query_apis_consistent_nan_and_sentinel_free() {
    let (_t, lib) = make_test_3x3_lib();

    // 1. num_rotamers rejects NaN and non-finite
    let nr_nan = lib.num_rotamers("TST", Some(f64::NAN), Some(0.0), false);
    assert!(matches!(nr_nan, Err(RotlibError::InvalidFormat(_))));

    // 2. rotamer_probability rejects NaN and non-finite
    let prob_nan = lib.rotamer_probability("TST", Some(0.0), Some(f64::NAN), false, 0);
    assert!(matches!(prob_nan, Err(RotlibError::InvalidFormat(_))));

    // 3. place_rotamer rejects NaN and non-finite
    let n = [1.0, 0.0, 0.0];
    let ca = [0.0, 0.0, 0.0];
    let c = [0.0, 1.0, 0.0];
    let place_nan = lib.place_rotamer(
        "TST",
        Some(f64::NAN),
        Some(0.0),
        0,
        false,
        n,
        ca,
        c,
    );
    assert!(matches!(place_nan, Err(RotlibError::InvalidFormat(_))));

    // 4. num_rotamers, rotamer_probability, and place_rotamer succeed with None (default bin)
    assert_eq!(lib.num_rotamers("TST", None, None, false).unwrap(), 1);
    assert_eq!(lib.rotamer_probability("TST", None, None, false, 0).unwrap(), 1.0);
    let placed = lib.place_rotamer("TST", None, None, 0, false, n, ca, c).unwrap();
    assert_eq!(placed.id.bin_index, 4); // default_bin is 4

    // 5. num_rotamers, rotamer_probability, and place_rotamer succeed with 1320.0 (bin 1)
    let placed_1320 = lib.place_rotamer(
        "TST",
        Some(1320.0),
        Some(0.0),
        0,
        false,
        n,
        ca,
        c,
    ).unwrap();
    assert_eq!(placed_1320.id.bin_index, 1); // 1320.0 deg -> bin 1

    // 6. -1320.0 (-1320 + 4*360 = 120 deg -> phi_ind=2, psi_ind=1 -> bin 2*3 + 1 = 7)
    let bin_neg1320 = lib.backbone_bin("TST", Some(-1320.0), Some(0.0), false).unwrap();
    assert_eq!(bin_neg1320, 7);

    // 7. GLY place_rotamer always succeeds and returns empty atoms, regardless of dihedrals
    let placed_gly = lib.place_rotamer("GLY", Some(f64::NAN), None, 0, false, n, ca, c).unwrap();
    assert_eq!(placed_gly.id.aa, "GLY");
    assert!(placed_gly.atoms.is_empty());
}


// ============================================================================
// TASK 2: GRAM-SCHMIDT FRAME VERIFICATION
// ============================================================================

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

fn det3(ux: [f64; 3], uy: [f64; 3], uz: [f64; 3]) -> f64 {
    ux[0] * (uy[1] * uz[2] - uy[2] * uz[1])
        - ux[1] * (uy[0] * uz[2] - uy[2] * uz[0])
        + ux[2] * (uy[0] * uz[1] - uy[1] * uz[0])
}

#[test]
fn test_challenger_frame_random_backbone_configurations() {
    // Stress test 10,000 random non-degenerate backbone configurations
    // using a deterministic LCG for reproducibility.
    let mut state = 123456789u64;
    let mut next_f64 = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((state >> 11) as f64) / ((1u64 << 53) as f64) * 20.0 - 10.0
    };

    let mut max_dot_err = 0.0f64;
    let mut max_norm_err = 0.0f64;
    let mut max_det_err = 0.0f64;

    for _ in 0..10_000 {
        let n = [next_f64(), next_f64(), next_f64()];
        let ca = [next_f64(), next_f64(), next_f64()];
        let c = [next_f64(), next_f64(), next_f64()];

        let frame = backbone_frame(n, ca, c);

        // Check origin
        assert_eq!(frame.origin, ca);

        // Check unit lengths
        let len_x = norm(frame.x);
        let len_y = norm(frame.y);
        let len_z = norm(frame.z);
        max_norm_err = max_norm_err
            .max((len_x - 1.0).abs())
            .max((len_y - 1.0).abs())
            .max((len_z - 1.0).abs());

        // Check orthogonality (dot products)
        let dot_xy = dot(frame.x, frame.y).abs();
        let dot_xz = dot(frame.x, frame.z).abs();
        let dot_yz = dot(frame.y, frame.z).abs();
        max_dot_err = max_dot_err.max(dot_xy).max(dot_xz).max(dot_yz);

        // Check right-handedness: det([ux, uy, uz]) must be +1.0
        let d = det3(frame.x, frame.y, frame.z);
        max_det_err = max_det_err.max((d - 1.0).abs());
    }

    eprintln!(
        "[EMPIRICAL RESULTS] Random Frame Orthonormality (10k trials): max_norm_err={:.2e}, max_dot_err={:.2e}, max_det_err={:.2e}",
        max_norm_err, max_dot_err, max_det_err
    );

    assert!(
        max_norm_err < 1e-12,
        "Axes must have unit length within float tolerance, got {:.2e}",
        max_norm_err
    );
    assert!(
        max_dot_err < 1e-12,
        "Axes must be mutually orthogonal within float tolerance, got {:.2e}",
        max_dot_err
    );
    assert!(
        max_det_err < 1e-12,
        "Axes must form a right-handed triad with det=+1, got {:.2e}",
        max_det_err
    );
}

#[test]
fn test_challenger_frame_degenerate_backbones() {
    // Degeneracy 1: Collinear N, CA, C
    let n = [0.0, 0.0, 0.0];
    let ca = [1.0, 0.0, 0.0];
    let c = [2.0, 0.0, 0.0];

    let frame_collinear = backbone_frame(n, ca, c);
    eprintln!(
        "[EMPIRICAL FINDING] Degenerate collinear backbone: x={:?}, y={:?}, z={:?}",
        frame_collinear.x, frame_collinear.y, frame_collinear.z
    );
    assert!(
        frame_collinear.z[0].is_nan() && frame_collinear.y[0].is_nan(),
        "Collinear backbone produces NaN in y and z axes"
    );

    // Degeneracy 2: Coincident N and CA
    let n_coincident = [1.0, 2.0, 3.0];
    let ca_coincident = [1.0, 2.0, 3.0];
    let c_valid = [1.0, 3.0, 3.0];
    let frame_n_ca = backbone_frame(n_coincident, ca_coincident, c_valid);
    eprintln!(
        "[EMPIRICAL FINDING] Degenerate N==CA: x={:?}",
        frame_n_ca.x
    );
    assert!(
        frame_n_ca.x[0].is_nan(),
        "Coincident N==CA produces NaN in x axis"
    );

    // Degeneracy 3: Coincident C and CA
    let frame_c_ca = backbone_frame([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
    eprintln!(
        "[EMPIRICAL FINDING] Degenerate C==CA: z={:?}",
        frame_c_ca.z
    );
    assert!(
        frame_c_ca.z[0].is_nan(),
        "Coincident C==CA produces NaN in z axis"
    );
}

// ============================================================================
// TASK 4: CCD PARSER FALLBACK VERIFICATION (DEBT #1933)
// ============================================================================

#[test]
fn test_challenger_ccd_fallback_on_missing_values() {
    let dir = tempfile::tempdir().unwrap();
    // CIF for ALA missing bond length (marked as '?') and angle (omitted)
    let cif_content = "\
data_ALA
#
loop_
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_dist_ideal
N CA 1.458
CA CB ?
#
loop_
_chem_comp_angle.atom_id_1
_chem_comp_angle.atom_id_2
_chem_comp_angle.atom_id_3
_chem_comp_angle.value_angle_ideal
N CA C 111.0
";
    let cif_file = dir.path().join("ALA.cif");
    std::fs::write(&cif_file, cif_content).unwrap();

    let (records, fallbacks) =
        parse_ccd_ic_table_raw(dir.path().to_str().unwrap()).unwrap();
    assert!(!records.is_empty(), "ALA records must be generated");

    // Verify bond length fallback
    let bond_fb = fallbacks
        .iter()
        .find(|f| f.parameter_kind == FallbackKind::BondLength);
    assert!(
        bond_fb.is_some(),
        "Missing bond length must generate a BondLength fallback"
    );
    let b = bond_fb.unwrap();
    assert_eq!(b.citation, "Engh & Huber 1991");
    assert_eq!(b.fallback_value, 1.50);

    // Verify angle fallback
    let angle_fb = fallbacks
        .iter()
        .find(|f| f.parameter_kind == FallbackKind::ValenceAngle);
    assert!(
        angle_fb.is_some(),
        "Missing angle must generate a ValenceAngle fallback"
    );
    let a = angle_fb.unwrap();
    assert_eq!(
        a.citation,
        "Ideal tetrahedral sp3 valence angle arccos(-1/3)"
    );
    assert_eq!(a.fallback_value, 109.47);
}

#[test]
fn test_challenger_ccd_behavior_on_garbled_values() {
    let dir = tempfile::tempdir().unwrap();
    // CIF for ALA with garbled bond distance string
    let cif_content = "\
data_ALA
#
loop_
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_dist_ideal
N CA GARBLED_NON_FLOAT
CA C 1.525
#
loop_
_chem_comp_angle.atom_id_1
_chem_comp_angle.atom_id_2
_chem_comp_angle.atom_id_3
_chem_comp_angle.value_angle_ideal
N CA C 111.0
";
    let cif_file = dir.path().join("ALA.cif");
    std::fs::write(&cif_file, cif_content).unwrap();

    // When a CIF value is garbled, does parse_ccd_ic_table_raw record a fallback or drop the residue?
    let (records, fallbacks) =
        parse_ccd_ic_table_raw(dir.path().to_str().unwrap()).unwrap();
    eprintln!(
        "[EMPIRICAL FINDING] CCD parser on garbled value: records.len()={}, fallbacks.len()={}",
        records.len(),
        fallbacks.len()
    );

    // In current implementation, parse_cif_file returns Err on garbled floats,
    // which causes parse_ccd_ic_table_raw to log a warning and DROP THE RESIDUE entirely!
    assert!(
        records.is_empty(),
        "Confirmed: when encountering a garbled float value, the entire residue is dropped rather than recording a fallback"
    );
    assert!(
        fallbacks.is_empty(),
        "Confirmed: no fallbacks are recorded when a parse error drops the residue"
    );
}
