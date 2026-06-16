//! AC-R: Cis-PRO routing acceptance tests for load_pb + resolve_entry_bin.
//!
//! Tests verify that `num_rotamers` and `place_rotamer` correctly route
//! "PRO" + cis_proline=true to the "CPR" (cis-proline) entry when it exists,
//! and fall back gracefully to "PRO" when no "CPR" entry is present.
//!
//! Test nomenclature:
//! - AC-R(1): place_rotamer routing — cis=true uses CPR coords
//! - AC-R(2): num_rotamers routing — cis=true reports CPR count
//! - AC-R fallback: no CPR entry → warns + falls back to PRO, no panic
//! - AC-R smoke: load_pb round-trip, contains_aa, known coord survives load
//! - AC-R(3): real data (artifact-gated, #[ignore])

use prost::Message;
use proxide_rotlib::pb::rotlib_v1::{
    RotamerLibrary as PbLib, ResidueEntry, Bin, Rotamer, Vec3, GeometryMode,
};
use proxide_rotlib::RotamerLibrary;

/// Standard attribution string used by the Dunbrack library.
const ATTRIBUTION: &str =
    "Contains information from the 2010 Backbone-Dependent Rotamer Library \
     (http://dunbrack.fccc.edu/bbdep2010), made available under the ODC Attribution License.";

/// Write a pb RotamerLibrary to a temp .pb.zst file, returning the path.
/// The file is named using process ID + a suffix to avoid per-test collisions.
fn write_pb_lib(lib: &PbLib, suffix: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!(
        "acr_{}_{}.pb.zst",
        std::process::id(),
        suffix
    ));
    let encoded = lib.encode_to_vec();
    let compressed = zstd::encode_all(encoded.as_slice(), 3).expect("zstd compress failed");
    std::fs::write(&path, &compressed).expect("write temp pb.zst failed");
    path
}

/// Build a synthetic library with both PRO (3 rotamers) and CPR (2 rotamers),
/// sharing the same grid (single bin at phi=-180, psi=-180) but distinguishable
/// by marker coords:
///   - PRO rotamer 0: CB at x=10.0 (marker)
///   - CPR rotamer 0: CB at x=20.0 (marker)
fn build_pro_cpr_library() -> PbLib {
    // PRO entry: 3 rotamers, CB x-marker = 10.0
    let pro_entry = ResidueEntry {
        code: "PRO".to_string(),
        atom_names: vec!["CB".to_string(), "CG".to_string(), "CD".to_string()],
        num_chi: 3,
        phi_centers: vec![-180.0],
        psi_centers: vec![-180.0],
        default_bin: 0,
        bins: vec![Bin {
            phi: -180.0,
            psi: -180.0,
            freq: 1.0,
            rotamers: vec![
                Rotamer {
                    prob: 0.5,
                    chi: vec![],
                    coords: vec![
                        Vec3 { x: 10.0, y: 1.0, z: 0.5 },
                        Vec3 { x: 11.0, y: 1.5, z: 0.3 },
                        Vec3 { x: 12.0, y: 0.8, z: 0.7 },
                    ],
                },
                Rotamer {
                    prob: 0.3,
                    chi: vec![],
                    coords: vec![
                        Vec3 { x: 10.0, y: -1.0, z: 0.5 },
                        Vec3 { x: 11.0, y: -1.5, z: 0.3 },
                        Vec3 { x: 12.0, y: -0.8, z: 0.7 },
                    ],
                },
                Rotamer {
                    prob: 0.2,
                    chi: vec![],
                    coords: vec![
                        Vec3 { x: 10.0, y: 0.0, z: 1.0 },
                        Vec3 { x: 11.0, y: 0.5, z: 1.3 },
                        Vec3 { x: 12.0, y: 0.2, z: 1.5 },
                    ],
                },
            ],
        }],
    };

    // CPR entry: 2 rotamers, CB x-marker = 20.0 (distinguishable from PRO)
    let cpr_entry = ResidueEntry {
        code: "CPR".to_string(),
        atom_names: vec!["CB".to_string(), "CG".to_string(), "CD".to_string()],
        num_chi: 3,
        phi_centers: vec![-180.0],
        psi_centers: vec![-180.0],
        default_bin: 0,
        bins: vec![Bin {
            phi: -180.0,
            psi: -180.0,
            freq: 1.0,
            rotamers: vec![
                Rotamer {
                    prob: 0.6,
                    chi: vec![],
                    coords: vec![
                        Vec3 { x: 20.0, y: 2.0, z: 0.5 },
                        Vec3 { x: 21.0, y: 2.5, z: 0.3 },
                        Vec3 { x: 22.0, y: 1.8, z: 0.7 },
                    ],
                },
                Rotamer {
                    prob: 0.4,
                    chi: vec![],
                    coords: vec![
                        Vec3 { x: 20.0, y: -2.0, z: 0.5 },
                        Vec3 { x: 21.0, y: -2.5, z: 0.3 },
                        Vec3 { x: 22.0, y: -1.8, z: 0.7 },
                    ],
                },
            ],
        }],
    };

    PbLib {
        version: 1,
        provenance: "test".to_string(),
        attribution: ATTRIBUTION.to_string(),
        data_license: "ODC-BY-1.0".to_string(),
        geometry_mode: GeometryMode::Precomputed as i32,
        residues: vec![pro_entry, cpr_entry],
        geometry_source: String::new(),
        geometry_license: String::new(),
    }
}

/// Build a synthetic library with only PRO (no CPR entry) for fallback tests.
fn build_pro_only_library() -> PbLib {
    let pro_entry = ResidueEntry {
        code: "PRO".to_string(),
        atom_names: vec!["CB".to_string(), "CG".to_string(), "CD".to_string()],
        num_chi: 3,
        phi_centers: vec![-180.0],
        psi_centers: vec![-180.0],
        default_bin: 0,
        bins: vec![Bin {
            phi: -180.0,
            psi: -180.0,
            freq: 1.0,
            rotamers: vec![
                Rotamer {
                    prob: 0.7,
                    chi: vec![],
                    coords: vec![
                        Vec3 { x: 10.0, y: 1.0, z: 0.5 },
                        Vec3 { x: 11.0, y: 1.5, z: 0.3 },
                        Vec3 { x: 12.0, y: 0.8, z: 0.7 },
                    ],
                },
                Rotamer {
                    prob: 0.3,
                    chi: vec![],
                    coords: vec![
                        Vec3 { x: 10.0, y: -1.0, z: 0.5 },
                        Vec3 { x: 11.0, y: -1.5, z: 0.3 },
                        Vec3 { x: 12.0, y: -0.8, z: 0.7 },
                    ],
                },
            ],
        }],
    };

    PbLib {
        version: 1,
        provenance: "test".to_string(),
        attribution: ATTRIBUTION.to_string(),
        data_license: "ODC-BY-1.0".to_string(),
        geometry_mode: GeometryMode::Precomputed as i32,
        residues: vec![pro_entry],
        geometry_source: String::new(),
        geometry_license: String::new(),
    }
}

// ─── AC-R smoke: load_pb round-trip ──────────────────────────────────────────

/// Smoke test: load_pb produces a library that contains both PRO and CPR,
/// and a known stored coord survives the load.
///
/// Backbone chosen for identity transform: N=[-1,0,0], CA=[0,0,0], C=[0,1,0].
/// With this backbone, backbone_frame axes align with lab axes (identity), so
/// place_rotamer returns stored coords unchanged (within float precision).
#[test]
fn test_ac_r_smoke_load_pb_round_trip() {
    let pb_lib = build_pro_cpr_library();
    let path = write_pb_lib(&pb_lib, "smoke");

    let lib = RotamerLibrary::load_pb(&path).expect("load_pb failed");
    std::fs::remove_file(&path).ok();

    // Contains both entries
    assert!(lib.contains_aa("PRO"), "expected PRO entry in library");
    assert!(lib.contains_aa("CPR"), "expected CPR entry in library");

    // Identity backbone: backbone_frame(N, CA, C) == lab identity frame
    // N=[-1,0,0], CA=[0,0,0], C=[0,1,0]
    // x_raw = CA - N = [1,0,0]; z_raw = x × (C-CA) = [1,0,0]×[0,1,0]=[0,0,1]
    // y_raw = z × x = [0,0,1]×[1,0,0]=[0,1,0]  → frame = identity ✓
    let n  = [-1.0_f64, 0.0, 0.0];
    let ca = [ 0.0_f64, 0.0, 0.0];
    let c  = [ 0.0_f64, 1.0, 0.0];

    // CPR rotamer 0, CB stored at (20.0, 2.0, 0.5) — should survive round-trip
    let placed = lib.place_rotamer("CPR", -180.0, -180.0, 0, false, n, ca, c)
        .expect("place_rotamer CPR failed");
    let cb_xyz = placed.atoms[0].xyz;

    // Stored CB = (20.0, 2.0, 0.5); identity frame → coords unchanged
    let expected = [20.0_f64, 2.0, 0.5];
    for i in 0..3 {
        assert!(
            (cb_xyz[i] - expected[i]).abs() < 1e-6,
            "CPR CB coord[{i}]: stored {:.6}, got {:.6}, diff {:.2e}",
            expected[i], cb_xyz[i], (cb_xyz[i] - expected[i]).abs()
        );
    }
}

// ─── AC-R(2): num_rotamers routing ───────────────────────────────────────────

/// AC-R(2): num_rotamers routes by cis flag (identity semantics, not merely inequality).
///
/// PRO has 3 rotamers; CPR has 2 rotamers. All three call sites must match exactly.
#[test]
fn test_ac_r2_num_rotamers_routing() {
    let pb_lib = build_pro_cpr_library();
    let path = write_pb_lib(&pb_lib, "numrot");

    let lib = RotamerLibrary::load_pb(&path).expect("load_pb failed");
    std::fs::remove_file(&path).ok();

    // cis=true on PRO must route to CPR → 2 rotamers
    let cis_count = lib.num_rotamers("PRO", -180.0, -180.0, true)
        .expect("num_rotamers PRO cis=true failed");
    assert_eq!(
        cis_count, 2,
        "PRO+cis=true must route to CPR (2 rotamers), got {}",
        cis_count
    );

    // cis=false on PRO stays in PRO → 3 rotamers
    let trans_count = lib.num_rotamers("PRO", -180.0, -180.0, false)
        .expect("num_rotamers PRO cis=false failed");
    assert_eq!(
        trans_count, 3,
        "PRO+cis=false must stay in PRO (3 rotamers), got {}",
        trans_count
    );

    // Direct CPR lookup → 2 rotamers
    let direct_cpr_count = lib.num_rotamers("CPR", -180.0, -180.0, false)
        .expect("num_rotamers CPR direct failed");
    assert_eq!(
        direct_cpr_count, 2,
        "direct CPR lookup must give 2 rotamers, got {}",
        direct_cpr_count
    );
}

// ─── AC-R(1): place_rotamer routing identity ─────────────────────────────────

/// AC-R(1): place_rotamer with cis=true on "PRO" returns CPR coords, not PRO coords.
///
/// Asserts:
/// - Every atom xyz of p_cis matches p_cpr within ≤1e-6 Å (max abs component diff)
/// - p_cis differs from p_pro by > 0.1 on at least one atom component
///   (proves routing actually switched entries, not a no-op)
/// - p_cis.id.aa == "PRO" (original code preserved) while coords came from CPR
#[test]
fn test_ac_r1_place_rotamer_routing_identity() {
    let pb_lib = build_pro_cpr_library();
    let path = write_pb_lib(&pb_lib, "place");

    let lib = RotamerLibrary::load_pb(&path).expect("load_pb failed");
    std::fs::remove_file(&path).ok();

    // Identity backbone (see smoke test comment for derivation):
    // N=[-1,0,0], CA=[0,0,0], C=[0,1,0] → backbone_frame = lab identity
    let n  = [-1.0_f64, 0.0, 0.0];
    let ca = [ 0.0_f64, 0.0, 0.0];
    let c  = [ 0.0_f64, 1.0, 0.0];

    // p_cis: PRO + cis=true → should use CPR coords
    let p_cis = lib.place_rotamer("PRO", -180.0, -180.0, 0, true, n, ca, c)
        .expect("place_rotamer PRO cis=true failed");

    // p_cpr: direct CPR lookup → CPR coords
    let p_cpr = lib.place_rotamer("CPR", -180.0, -180.0, 0, false, n, ca, c)
        .expect("place_rotamer CPR direct failed");

    // p_pro: PRO + cis=false → PRO coords
    let p_pro = lib.place_rotamer("PRO", -180.0, -180.0, 0, false, n, ca, c)
        .expect("place_rotamer PRO cis=false failed");

    // Check: p_cis.id.aa must be "PRO" (the queried aa code is preserved even when routed)
    assert_eq!(
        p_cis.id.aa, "PRO",
        "place_rotamer with aa='PRO' must preserve id.aa='PRO' even when routed to CPR"
    );

    // Check: p_cis coords match p_cpr coords within ≤1e-6 (identity routing)
    assert_eq!(
        p_cis.atoms.len(),
        p_cpr.atoms.len(),
        "atom count mismatch between p_cis and p_cpr"
    );
    for (atom_cis, atom_cpr) in p_cis.atoms.iter().zip(p_cpr.atoms.iter()) {
        for i in 0..3 {
            assert!(
                (atom_cis.xyz[i] - atom_cpr.xyz[i]).abs() < 1e-6,
                "AC-R(1) atom '{}' coord[{i}]: p_cis={:.9}, p_cpr={:.9}, diff={:.2e} > 1e-6 Å",
                atom_cis.name,
                atom_cis.xyz[i],
                atom_cpr.xyz[i],
                (atom_cis.xyz[i] - atom_cpr.xyz[i]).abs()
            );
        }
    }

    // Check: p_cis differs from p_pro by >0.1 on at least one atom component
    // (proves routing actually switched entries, not a no-op)
    // PRO CB x-marker=10.0, CPR CB x-marker=20.0 → difference ≥ 10.0
    let differs_enough = p_cis.atoms.iter().zip(p_pro.atoms.iter()).any(|(atom_cis, atom_pro)| {
        (0..3).any(|i| (atom_cis.xyz[i] - atom_pro.xyz[i]).abs() > 0.1)
    });
    assert!(
        differs_enough,
        "AC-R(1): p_cis (routed PRO+cis) must differ from p_pro (trans PRO) by >0.1 Å on at \
         least one atom component — routing did not switch entries"
    );
}

// ─── AC-R fallback: no CPR entry ─────────────────────────────────────────────

/// AC-R fallback: when a library has PRO but no CPR, num_rotamers("PRO",...,cis=true)
/// falls back to the PRO count without panic, matching the documented warn+fallback behavior.
#[test]
fn test_ac_r_fallback_no_cpr_entry() {
    let pb_lib = build_pro_only_library();
    let path = write_pb_lib(&pb_lib, "fallback");

    let lib = RotamerLibrary::load_pb(&path).expect("load_pb failed");
    std::fs::remove_file(&path).ok();

    assert!(!lib.contains_aa("CPR"), "precondition: no CPR entry");

    // Must not panic; must return the PRO count (2 rotamers in pro_only)
    let fallback_count = lib.num_rotamers("PRO", -180.0, -180.0, true)
        .expect("num_rotamers PRO cis=true with no CPR must not error");
    let pro_count = lib.num_rotamers("PRO", -180.0, -180.0, false)
        .expect("num_rotamers PRO cis=false failed");

    assert_eq!(
        fallback_count, pro_count,
        "fallback: cis=true with no CPR must return same count as cis=false PRO count ({pro_count}), \
         got {fallback_count}"
    );
}

/// AC-R fallback place_rotamer: when a library has PRO but no CPR,
/// place_rotamer("PRO",...,cis=true) falls back to PRO coords without panic,
/// returning identical coords to place_rotamer("PRO",...,cis=false).
#[test]
fn test_cpr_fallback_when_cpro_absent() {
    let pb_lib = build_pro_only_library();
    let path = write_pb_lib(&pb_lib, "cpr_fallback");

    let lib = RotamerLibrary::load_pb(&path).expect("load_pb failed");
    std::fs::remove_file(&path).ok();

    assert!(!lib.contains_aa("CPR"), "precondition: no CPR entry");

    // Identity backbone: N=[-1,0,0], CA=[0,0,0], C=[0,1,0]
    let n  = [-1.0_f64, 0.0, 0.0];
    let ca = [ 0.0_f64, 0.0, 0.0];
    let c  = [ 0.0_f64, 1.0, 0.0];

    // place_rotamer with cis=false (trans PRO)
    let p_normal = lib.place_rotamer("PRO", -180.0, -180.0, 0, false, n, ca, c)
        .expect("place_rotamer PRO cis=false failed");

    // place_rotamer with cis=true (fallback: no CPR, so uses PRO)
    let p_cis = lib.place_rotamer("PRO", -180.0, -180.0, 0, true, n, ca, c)
        .expect("place_rotamer PRO cis=true with no CPR must not panic");

    // Verify atom counts match
    assert_eq!(
        p_normal.atoms.len(),
        p_cis.atoms.len(),
        "atom count must match in fallback: normal={}, cis={}",
        p_normal.atoms.len(),
        p_cis.atoms.len()
    );

    // Verify coords match within tolerance (1e-6 Å)
    for (atom_normal, atom_cis) in p_normal.atoms.iter().zip(p_cis.atoms.iter()) {
        for i in 0..3 {
            assert!(
                (atom_normal.xyz[i] - atom_cis.xyz[i]).abs() < 1e-6,
                "fallback PRO(cis) must return same coords as PRO(trans) when CPRO absent: \
                 atom '{}' coord[{i}]: normal={:.9}, cis={:.9}, diff={:.2e} >= 1e-6",
                atom_normal.name,
                atom_normal.xyz[i],
                atom_cis.xyz[i],
                (atom_normal.xyz[i] - atom_cis.xyz[i]).abs()
            );
        }
    }
}

// ─── AC-R(3): real data, artifact-gated #[ignore] ─────────────────────────────

/// AC-R(3): Real Dunbrack 2010 BBdep library — cis-proline χ1 check.
///
/// Artifact-gated: the pb.zst file is generated by `convert_rotlib` and is not
/// committed to the repository. This test is marked #[ignore] so CI passes without
/// the artifact. Run manually with:
///   cargo test -p proxide-rotlib --test test_ac_r_routing test_ac_r3_real_data_cis_chi1 -- --ignored
///
/// AC-R(3) criterion (spec §P5):
/// Place cis-PRO rotamer 0 at phi=-180, psi=-180 onto the canonical identity backbone,
/// compute χ1 = N-CA-CB-CG dihedral, and assert it is closer to 32.5° than to 27.3°.
/// (32.5° = Dunbrack CPR endo pucker representative; 27.3° = trans-PRO representative)
#[test]
#[ignore = "artifact-gated: requires data/rotlibs/proxide-rotlib-bbdep2010.pb.zst"]
fn test_ac_r3_real_data_cis_chi1() {
    // Locate the artifact relative to the crate's CARGO_MANIFEST_DIR.
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR not set");
    let pb_path = std::path::PathBuf::from(&manifest_dir)
        .join("data")
        .join("rotlibs")
        .join("proxide-rotlib-bbdep2010.pb.zst");

    // Graceful skip if artifact is absent (gitignored, not present in CI).
    if !pb_path.exists() {
        eprintln!(
            "AC-R(3) skipped: artifact not found at {}",
            pb_path.display()
        );
        return;
    }

    let lib = RotamerLibrary::load_pb(&pb_path).expect("load_pb real library failed");

    // Verify the library has both PRO and CPR entries.
    assert!(lib.contains_aa("PRO"), "real library must contain PRO");
    assert!(lib.contains_aa("CPR"), "real library must contain CPR (cis-proline)");

    // Identity backbone (N-CA-CB-CG dihedral is χ1, must survive frame transform intact):
    // N=[-1,0,0], CA=[0,0,0], C=[0,1,0]
    let n  = [-1.0_f64, 0.0, 0.0];
    let ca = [ 0.0_f64, 0.0, 0.0];
    let c  = [ 0.0_f64, 1.0, 0.0];

    // Place cis-PRO rotamer 0 at phi=-180, psi=-180 onto the identity backbone.
    // cis=true → routes to CPR entry.
    let placed = lib.place_rotamer("PRO", -180.0, -180.0, 0, true, n, ca, c)
        .expect("place_rotamer cis-PRO rot=0 failed");

    // Find the CB and CG atoms in the placed result.
    let cb = placed.atoms.iter().find(|a| a.name == "CB")
        .expect("placed cis-PRO must have a CB atom");
    let cg = placed.atoms.iter().find(|a| a.name == "CG")
        .expect("placed cis-PRO must have a CG atom");

    // Compute χ1 = N-CA-CB-CG dihedral.
    let chi1_rad = dihedral_angle_rad(n, ca, cb.xyz, cg.xyz);
    let chi1_deg = chi1_rad.to_degrees();

    // AC-R(3) criterion: χ1 must be closer to 32.5° than to 27.3°.
    // 32.5° = Dunbrack CPR endo pucker representative (endo: χ1 ≥ 0).
    // 27.3° = Dunbrack trans-PRO representative (would indicate wrong routing).
    let dist_to_32_5 = (chi1_deg - 32.5_f64).abs();
    let dist_to_27_3 = (chi1_deg - 27.3_f64).abs();

    assert!(
        dist_to_32_5 < dist_to_27_3,
        "AC-R(3): cis-PRO χ1 {:.2}° must be closer to 32.5° (endo CPR) than to 27.3° \
         (trans PRO). dist_to_32.5={:.2}°, dist_to_27.3={:.2}°. \
         If this fails, cis routing may not be using the CPR entry.",
        chi1_deg, dist_to_32_5, dist_to_27_3
    );
}

/// Compute the signed dihedral angle (radians) for atoms p1-p2-p3-p4.
fn dihedral_angle_rad(p1: [f64; 3], p2: [f64; 3], p3: [f64; 3], p4: [f64; 3]) -> f64 {
    let sub  = |a: [f64; 3], b: [f64; 3]| [a[0]-b[0], a[1]-b[1], a[2]-b[2]];
    let cross = |a: [f64; 3], b: [f64; 3]| -> [f64; 3] {
        [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
    };
    let dot  = |a: [f64; 3], b: [f64; 3]| a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
    let norm = |a: [f64; 3]| dot(a, a).sqrt();
    let b1 = sub(p2, p1);
    let b2 = sub(p3, p2);
    let b3 = sub(p4, p3);
    let n1 = cross(b1, b2);
    let n2 = cross(b2, b3);
    let b2n = { let s = 1.0 / norm(b2); [b2[0]*s, b2[1]*s, b2[2]*s] };
    let m1 = cross(n1, b2n);
    f64::atan2(dot(m1, n2), dot(n1, n2))
}
