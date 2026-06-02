//! Integration tests for proxide-master — acceptance criteria AC-1 through AC-9.
//!
//! AC-1: phantom type enforcement (Raw cannot be passed where Centered is required)
//! AC-2: double-centering guard (AlreadyCenteredError if centroid ≈ 0)
//! AC-3: RMSD golden values against 1UBQ fixtures
//! AC-4: no false negatives (self-match always found)
//! AC-5: no false positives (all returned results satisfy RMSD ≤ epsilon)
//! AC-6: serial/parallel parity
//! AC-7: empty database search returns empty results
//! AC-9: norm_sq consistency

use approx::assert_relative_eq;
use proxide_master::{
    AlreadyCenteredError, Fragment, FragmentDb, FragmentDbBuilder, Raw, SourceLabel,
    kabsch_rmsd,
};
use rand::{Rng, SeedableRng};

// ---------------------------------------------------------------------------
// 1UBQ coordinate fixtures
// ---------------------------------------------------------------------------

/// 1UBQ residues 1-5 backbone atoms [residue][atom][xyz] (N, CA, C, O).
/// Source: PDB 1UBQ.
const UBIQUITIN_1_5: [[[f32; 3]; 4]; 5] = [
    // MET 1
    [[27.340, 24.430, 2.614], [26.266, 25.413, 2.842], [26.913, 26.639, 3.531], [27.886, 26.463, 4.263]],
    // GLN 2
    [[26.335, 27.783, 3.258], [26.850, 29.024, 3.898], [26.100, 29.200, 5.202], [24.865, 29.378, 5.230]],
    // ILE 3
    [[26.842, 29.241, 6.271], [26.155, 29.362, 7.552], [26.633, 28.288, 8.497], [26.882, 27.140, 8.054]],
    // PHE 4
    [[26.850, 28.665, 9.756], [27.303, 27.728, 10.726], [26.290, 26.630, 11.041], [25.148, 26.680, 10.583]],
    // VAL 5
    [[26.720, 25.674, 11.780], [25.855, 24.568, 12.116], [24.528, 24.935, 12.768], [23.985, 26.049, 12.563]],
];

/// 1UBQ residues 6-10 backbone atoms (LYS, THR, LEU, THR, GLY).
const UBIQUITIN_6_10: [[[f32; 3]; 4]; 5] = [
    // LYS 6
    [[24.021, 24.049, 13.465], [22.773, 24.262, 14.155], [21.836, 23.047, 14.055], [21.876, 22.222, 14.979]],
    // THR 7
    [[20.988, 22.974, 13.014], [20.085, 21.822, 12.849], [19.231, 21.603, 14.080], [18.031, 21.940, 14.051]],
    // LEU 8
    [[19.849, 21.046, 15.115], [19.074, 20.756, 16.318], [19.700, 19.613, 17.072], [20.489, 18.855, 16.488]],
    // THR 9
    [[19.270, 19.440, 18.290], [19.748, 18.333, 19.097], [20.001, 17.137, 18.268], [20.815, 17.209, 17.325]],
    // GLY 10
    [[19.367, 16.101, 18.620], [19.606, 14.907, 17.831], [20.356, 13.895, 18.618], [20.079, 13.717, 19.795]],
];

/// Golden reference RMSD for fixture B: residues 1-5 vs 6-10 of 1UBQ.
/// Computed by scripts/analysis/compute_reference_rmsd.py.
const RMSD_FIXTURE_B: f32 = 1.251516;

// ---------------------------------------------------------------------------
// Helper: apply a 30-degree rotation around z to a set of coords.
// ---------------------------------------------------------------------------

fn rotate_z_30(coords: [[[f32; 3]; 4]; 5]) -> [[[f32; 3]; 4]; 5] {
    use std::f32::consts::PI;
    let theta = 30.0_f32 * PI / 180.0;
    let c = theta.cos();
    let s = theta.sin();
    let mut out = coords;
    for r in 0..5 {
        for at in 0..4 {
            let x = coords[r][at][0];
            let y = coords[r][at][1];
            let z = coords[r][at][2];
            out[r][at][0] = c * x - s * y;
            out[r][at][1] = s * x + c * y;
            out[r][at][2] = z;
        }
    }
    out
}

/// Make a SourceLabel for testing.
fn label(tag: &str) -> SourceLabel {
    SourceLabel::new(tag, 'A', 1, ' ', 5, ' ')
}

/// Build N random 5-mer fragments with a fixed RNG seed.
fn random_fragments(n: usize, seed: u64) -> Vec<Fragment<5, Raw>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let mut coords: [[[f32; 3]; 4]; 5] = [[[0.0; 3]; 4]; 5];
            for r in 0..5 {
                for at in 0..4 {
                    coords[r][at][0] = rng.gen_range(-20.0_f32..20.0);
                    coords[r][at][1] = rng.gen_range(-20.0_f32..20.0);
                    coords[r][at][2] = rng.gen_range(-20.0_f32..20.0);
                }
            }
            Fragment::<5, Raw>::new(coords)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// AC-1: Phantom type enforcement
// ---------------------------------------------------------------------------

/// AC-1 — A Fragment<5, Raw> cannot be passed to kabsch_rmsd (which requires
/// Fragment<5, Centered>). This is enforced at compile time; the test below
/// demonstrates that centering produces the correct state.
#[test]
fn ac1_center_produces_centered_state() {
    let (centered, _centroid) = Fragment::<5, Raw>::new(UBIQUITIN_1_5)
        .center()
        .expect("centering should succeed");
    // The only way to call kabsch_rmsd is with Fragment<N, Centered>.
    // If we could pass a Raw fragment it wouldn't compile.
    let (centered_b, _) = Fragment::<5, Raw>::new(UBIQUITIN_6_10)
        .center()
        .expect("centering b should succeed");
    let ns_a = centered.norm_sq();
    let ns_b = centered_b.norm_sq();
    let kr = kabsch_rmsd(&centered, ns_a, &centered_b, ns_b);
    assert!(kr.rmsd.is_finite(), "RMSD must be finite");
}

// ---------------------------------------------------------------------------
// AC-2: Double-centering guard
// ---------------------------------------------------------------------------

/// AC-2 — Fragment whose centroid is exactly at the origin returns
/// AlreadyCenteredError.
#[test]
fn ac2_already_centered_guard() {
    // All-zero coords → centroid is (0, 0, 0), norm = 0 < 1e-6.
    let zero_coords: [[[f32; 3]; 4]; 5] = [[[0.0; 3]; 4]; 5];
    let result = Fragment::<5, Raw>::new(zero_coords).center();
    assert!(
        matches!(result, Err(AlreadyCenteredError { norm }) if norm < 1e-6),
        "Expected AlreadyCenteredError for already-centered fragment, got: {result:?}"
    );
}

/// AC-2b — Symmetric arrangement: 4 atoms at ±1 on each axis give centroid = 0.
#[test]
fn ac2_symmetric_already_centered() {
    let mut coords: [[[f32; 3]; 4]; 5] = [[[0.0; 3]; 4]; 5];
    // Place atoms symmetrically so centroid = 0: pairs at +d and -d.
    // With 20 atoms total, place 10 at +1 and 10 at -1 on x.
    for r in 0..5 {
        for at in 0..4 {
            let sign = if (r * 4 + at) % 2 == 0 { 1.0 } else { -1.0 };
            coords[r][at][0] = sign;
            coords[r][at][1] = 0.0;
            coords[r][at][2] = 0.0;
        }
    }
    let result = Fragment::<5, Raw>::new(coords).center();
    assert!(result.is_err(), "Symmetric coords with centroid=0 should trigger guard");
}

// ---------------------------------------------------------------------------
// AC-3: RMSD golden values
// ---------------------------------------------------------------------------

/// AC-3 Fixture A — Apply known rotation (30° around z) to 1UBQ residues 1-5.
/// Kabsch should recover the rotation and return RMSD ≈ 0.
#[test]
fn ac3_fixture_a_known_rotation() {
    let rotated = rotate_z_30(UBIQUITIN_1_5);
    let (a_c, _) = Fragment::<5, Raw>::new(UBIQUITIN_1_5).center().unwrap();
    let (b_c, _) = Fragment::<5, Raw>::new(rotated).center().unwrap();
    let ns_a = a_c.norm_sq();
    let ns_b = b_c.norm_sq();
    let kr = kabsch_rmsd(&a_c, ns_a, &b_c, ns_b);
    assert!(
        kr.rmsd < 5e-3,
        "Fixture A: expected RMSD < 5e-3 for known rotation, got {}",
        kr.rmsd
    );
}

/// AC-3 Fixture B — 1UBQ residues 1-5 vs 6-10, reference value 1.251516 Å.
#[test]
fn ac3_fixture_b_ubiquitin_cross_fragment() {
    let (a_c, _) = Fragment::<5, Raw>::new(UBIQUITIN_1_5).center().unwrap();
    let (b_c, _) = Fragment::<5, Raw>::new(UBIQUITIN_6_10).center().unwrap();
    let ns_a = a_c.norm_sq();
    let ns_b = b_c.norm_sq();
    let kr = kabsch_rmsd(&a_c, ns_a, &b_c, ns_b);
    assert_relative_eq!(kr.rmsd, RMSD_FIXTURE_B, epsilon = 0.001_f32);
}

// ---------------------------------------------------------------------------
// AC-4: No false negatives (self-match always found)
// ---------------------------------------------------------------------------

/// AC-4 — Build a database with 100 random fragments + the query itself.
/// The query must appear in the results with epsilon = 0.001.
#[test]
fn ac4_no_false_negatives() {
    let query_raw = Fragment::<5, Raw>::new(UBIQUITIN_1_5);
    // Get a centered copy for use as query.
    let (query_centered, _) = Fragment::<5, Raw>::new(UBIQUITIN_1_5).center().unwrap();

    let mut builder = FragmentDbBuilder::<5>::new();

    // Insert 100 random fragments.
    for (i, frag) in random_fragments(100, 42).into_iter().enumerate() {
        let lbl = label(&format!("{:04}", i));
        let _ = builder.add_fragment(frag, lbl); // some may fail (already centered), that's OK
    }
    // Insert the query fragment itself.
    builder
        .add_fragment(query_raw, label("SELF"))
        .expect("query fragment should not be already centered");

    let db = builder.build();
    let results = db.search(&query_centered, 0.001);
    assert!(
        !results.is_empty(),
        "Self-match must be found with epsilon=0.001"
    );
    let self_found = results.iter().any(|r| r.rmsd < 0.001);
    assert!(self_found, "Self-entry must have RMSD < 0.001; results: {results:?}");
}

// ---------------------------------------------------------------------------
// AC-5: No false positives
// ---------------------------------------------------------------------------

/// AC-5 — All results returned by search satisfy RMSD ≤ epsilon.
#[test]
fn ac5_no_false_positives() {
    let (query_centered, _) = Fragment::<5, Raw>::new(UBIQUITIN_1_5).center().unwrap();
    let epsilon = 2.0_f32; // generous epsilon to get some hits from random frags

    let mut builder = FragmentDbBuilder::<5>::new();
    for (i, frag) in random_fragments(200, 99).into_iter().enumerate() {
        let lbl = label(&format!("{:04}", i));
        let _ = builder.add_fragment(frag, lbl);
    }
    let db = builder.build();
    let results = db.search(&query_centered, epsilon);

    for r in &results {
        assert!(
            r.rmsd <= epsilon + 1e-5, // small float tolerance
            "Result RMSD {} exceeds epsilon {}",
            r.rmsd,
            epsilon
        );
    }
}

// ---------------------------------------------------------------------------
// AC-6: Serial / parallel parity
// ---------------------------------------------------------------------------

/// AC-6 — search() and search_serial() return identical results (same length,
/// same RMSD values, same labels) for the same query and epsilon.
#[test]
fn ac6_serial_parallel_parity() {
    let (query_centered, _) = Fragment::<5, Raw>::new(UBIQUITIN_1_5).center().unwrap();
    let (query_centered_2, _) = Fragment::<5, Raw>::new(UBIQUITIN_1_5).center().unwrap();
    let epsilon = 2.0_f32;

    let mut builder = FragmentDbBuilder::<5>::new();
    for (i, frag) in random_fragments(150, 7).into_iter().enumerate() {
        let lbl = label(&format!("{:04}", i));
        let _ = builder.add_fragment(frag, lbl);
    }
    let db = builder.build();

    let par = db.search(&query_centered, epsilon);
    let ser = db.search_serial(&query_centered_2, epsilon);

    assert_eq!(par.len(), ser.len(), "Parallel and serial must return same number of results");

    for (p, s) in par.iter().zip(ser.iter()) {
        assert_relative_eq!(p.rmsd, s.rmsd, epsilon = 1e-5_f32);
        assert_eq!(p.label, s.label, "Labels must match between parallel and serial");
    }
}

// ---------------------------------------------------------------------------
// AC-7: Empty database
// ---------------------------------------------------------------------------

/// AC-7 — Searching an empty database returns an empty result set.
#[test]
fn ac7_empty_database() {
    let db: FragmentDb<5> = FragmentDbBuilder::<5>::new().build();
    assert!(db.is_empty());
    let (query_centered, _) = Fragment::<5, Raw>::new(UBIQUITIN_1_5).center().unwrap();
    let results = db.search(&query_centered, 100.0);
    assert!(results.is_empty(), "Empty database must return empty results");
}

// ---------------------------------------------------------------------------
// AC-9: norm_sq consistency
// ---------------------------------------------------------------------------

/// AC-9 — norm_sq() equals the manual sum-of-squares over all coords.
#[test]
fn ac9_norm_sq_consistency() {
    let (centered, _) = Fragment::<5, Raw>::new(UBIQUITIN_1_5).center().unwrap();
    let ns = centered.norm_sq();

    let mut manual = 0.0_f32;
    for r in 0..5 {
        for at in 0..4 {
            let [x, y, z] = centered.coords[r][at];
            manual += x * x + y * y + z * z;
        }
    }
    assert_relative_eq!(ns, manual, epsilon = 1e-5_f32);
}
