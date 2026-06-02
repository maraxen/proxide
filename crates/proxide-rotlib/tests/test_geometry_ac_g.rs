//! AC-G test suite for proline ring closure geometry.
//!
//! Per spec §11 AC-G (strengthened):
//! - (a) all 3 Dunbrack χ recovered within ±2°
//! - (b) both puckers (r1=1,2) build with distinct CG (≥0.5 Å apart)
//! - (c) all five endocyclic angles within ±3° of ideal pyrrolidine
//! - (d) rebuilt PRO ring heavy-atom RMSD vs CCD PRO.cif ≤ 0.05 Å
//! - (e) round-trip identity: place_rotamer onto equal backbone returns stored coords within ≤1e-2 Å

use proxide_rotlib::geometry::template::proline_template;

/// RCSB CCD ideal coordinates for proline (from PRO.cif).
/// Heavy atoms in sidechain order: N, CA, C, O, CB, CG, CD.
const CCD_PRO_IDEAL: &[(&str, [f64; 3])] = &[
    ("N", [-0.816, 1.108, 0.254]),
    ("CA", [0.001, -0.107, 0.509]),
    ("C", [1.408, 0.091, 0.005]),
    ("O", [1.650, 0.980, -0.777]),
    ("CB", [-0.703, -1.227, -0.286]),
    ("CG", [-2.163, -0.753, -0.439]),
    ("CD", [-2.218, 0.614, 0.276]),
];

/// Reference ideal bond lengths (Engh-Huber, from template).
const IDEAL_BOND_LENGTHS: &[(&str, f32)] = &[
    ("N-CA", 1.458),
    ("CA-C", 1.520),
    ("C-O", 1.231),
    ("CA-CB", 1.530),
    ("CB-CG", 1.503),
    ("CG-CD", 1.503),
    ("CD-N", 1.470),
];

/// Reference ideal bond angles (Engh-Huber, from template, ±3° is acceptable).
/// Listed as (atom1, atom2, atom3) -> angle_deg.
const IDEAL_ANGLES: &[(&str, f32)] = &[
    ("N-CA-CB", 110.0),   // endocyclic angle for χ1
    ("CA-CB-CG", 104.0),  // endocyclic angle for χ2
    ("CB-CG-CD", 105.0),  // endocyclic angle for χ3
    ("CG-CD-N", 110.0),   // endocyclic angle
    ("CD-N-CA", 110.0),   // endocyclic angle
];

#[test]
fn test_ac_g_proline_template_structure() {
    let template = proline_template();
    assert_eq!(template.code, "PRO");
    assert_eq!(template.num_atoms(), 7);
    assert!(template.dihedral("χ1").is_some());
    assert!(template.dihedral("χ2").is_some());
    assert!(template.dihedral("χ3").is_some());
}

// Note: ProlineBuilder::build is currently a stub implementation.
// Once NeRF integration is complete, comprehensive AC-G tests will be added
// to validate χ recovery, pucker distinctness, bond angles, RMSD vs CCD PRO.cif, and round-trip identity.

/// Compute distance between two points.
fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Compute bond angle (degrees) from three atoms.
fn angle_deg(p0: [f64; 3], p1: [f64; 3], p2: [f64; 3]) -> f64 {
    let v1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
    let v2 = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];

    let dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
    let mag1 = (v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]).sqrt();
    let mag2 = (v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2]).sqrt();

    let cos_angle = dot / (mag1 * mag2);
    cos_angle.clamp(-1.0, 1.0).acos() * 180.0 / std::f64::consts::PI
}

/// Compute RMSD between two sets of atoms.
fn rmsd(atoms1: &[[f64; 3]], atoms2: &[[f64; 3]]) -> f64 {
    assert_eq!(atoms1.len(), atoms2.len());
    let sum_sq: f64 = atoms1.iter().zip(atoms2.iter())
        .map(|(a, b)| distance(*a, *b).powi(2))
        .sum();
    (sum_sq / atoms1.len() as f64).sqrt()
}

#[test]
fn test_ac_g_ccd_pro_cif_reference() {
    // Verify the CCD reference is valid.
    assert_eq!(CCD_PRO_IDEAL.len(), 7);

    // Check N-CA distance in CCD PRO.cif.
    let n = CCD_PRO_IDEAL[0].1;
    let ca = CCD_PRO_IDEAL[1].1;
    let n_ca_dist = distance(n, ca);
    println!("CCD PRO N-CA distance: {:.3} Å", n_ca_dist);
    assert!((n_ca_dist - 1.458).abs() < 0.1, "CCD PRO N-CA should be ~1.458 Å, got {:.3}", n_ca_dist);

    // Check CD-N distance (ring closure).
    let cd = CCD_PRO_IDEAL[6].1;
    let cd_n_dist = distance(cd, n);
    println!("CCD PRO CD-N distance: {:.3} Å", cd_n_dist);
    assert!((cd_n_dist - 1.470).abs() < 0.1, "CCD PRO CD-N should be ~1.470 Å, got {:.3}", cd_n_dist);
}

#[test]
fn test_ac_g_placeholder_note() {
    // Placeholder: once ProlineBuilder::build is fully integrated with NeRF,
    // these tests will verify:
    // (a) χ recovery within ±2°
    // (b) endo/exo pucker distinctness (CG separation ≥0.5 Å)
    // (c) bond angles within ±3° of ideal
    // (d) RMSD vs CCD PRO.cif ≤ 0.05 Å
    // (e) round-trip identity ≤ 1e-2 Å
    //
    // For now, we validate the reference data structure.
    assert_eq!(CCD_PRO_IDEAL.len(), 7);
}
