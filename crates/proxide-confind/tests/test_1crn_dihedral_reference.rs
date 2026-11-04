//! proxide-confind's backbone dihedrals on a real structure, against an external
//! implementation (debt #3224).
//!
//! The confind parity gate's omega check compares the reference binary (Mosaist) with
//! an omega the harness computes itself, so it never exercises proxide. This test
//! closes that gap: it runs both `MissingAtomPolicy` paths and the `load_pdb_f64`
//! entry point on wwPDB 1CRN and compares phi/psi/omega, residue by residue, with a
//! biotite-computed IUPAC table checked in at `tests/data/` (regenerate with
//! `scripts/validation/gen_1crn_biotite_dihedrals.py`; never hand-edit it).
//!
//! Angles are compared wrap-aware (`|Δ|` on the circle), so the ±180° branch of a
//! trans omega cannot produce a spurious failure. Because a sign flip of a trans
//! omega moves it by only `2·(180° − |ω|)`, the test also runs a negative control:
//! the sign-flipped proxide angles must disagree with the reference, so the
//! comparison demonstrably detects the −IUPAC defect class (#3216, #3229, #3230).

mod common;

use common::crambin_path;
use proxide_confind::{
    extract_f64_backbone_with_options, load_pdb_f64, BackboneOptions, MissingAtomPolicy,
    ProteinBackbone,
};
use proxide_core::processing::residues::ProcessedStructure;
use std::path::PathBuf;

/// biotite reads coordinates as f32; the disagreement this allows is ~1e-4°.
const AGREE_TOL_DEG: f64 = 1e-3;
/// A flipped angle counts as detected when it misses the reference by more than this.
const DETECT_DEG: f64 = 1.0;
/// Each angle kind's negative control must be detected on at least this many residues.
const MIN_DETECTED: usize = 10;

struct RefRow {
    res_seq: i32,
    res_name: String,
    phi: Option<f64>,
    psi: Option<f64>,
    omega: Option<f64>,
}

fn reference_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/1crn_biotite_backbone_dihedrals.tsv")
}

fn parse_angle(field: &str, line: &str) -> Option<f64> {
    if field == "NA" {
        return None;
    }
    let v: f64 = field
        .parse()
        .unwrap_or_else(|e| panic!("bad angle {field:?} in {line:?}: {e}"));
    assert!(v.is_finite(), "non-finite angle in {line:?}");
    Some(v)
}

fn load_reference() -> Vec<RefRow> {
    let path = reference_path();
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("reference table {} must exist: {e}", path.display()));
    let mut lines = text.lines().filter(|l| !l.starts_with('#'));
    assert_eq!(
        lines.next(),
        Some("res_seq\tres_name\tphi_deg\tpsi_deg\tomega_deg"),
        "reference table header changed"
    );
    lines
        .map(|line| {
            let f: Vec<&str> = line.split('\t').collect();
            assert_eq!(f.len(), 5, "malformed reference row {line:?}");
            RefRow {
                res_seq: f[0].parse().expect("res_seq"),
                res_name: f[1].to_string(),
                phi: parse_angle(f[2], line),
                psi: parse_angle(f[3], line),
                omega: parse_angle(f[4], line),
            }
        })
        .collect()
}

/// Smallest separation of two angles on the circle, in degrees.
fn circ_delta(a: f64, b: f64) -> f64 {
    let d = (a - b).rem_euclid(360.0);
    d.min(360.0 - d)
}

fn processed_crambin() -> ProcessedStructure {
    let (raw, _) =
        proxide_io::formats::pdb::parse_pdb_file(crambin_path()).expect("parse 1CRN.pdb");
    ProcessedStructure::from_raw(raw).expect("ProcessedStructure from 1CRN")
}

type Getter = fn(&proxide_confind::ResidueBackbone) -> Option<f64>;
type RefGetter = fn(&RefRow) -> Option<f64>;

const KINDS: [(&str, Getter, RefGetter); 3] = [
    ("phi", |r| r.phi, |r| r.phi),
    ("psi", |r| r.psi, |r| r.psi),
    ("omega", |r| r.omega, |r| r.omega),
];

/// Assert `bb` matches the reference on every residue; return, per angle kind, how many
/// residues the sign-flipped value would miss by more than `DETECT_DEG`.
fn assert_matches_reference(label: &str, bb: &ProteinBackbone, reference: &[RefRow]) -> [usize; 3] {
    assert_eq!(bb.bb.len(), reference.len(), "{label}: residue count");
    let mut detected = [0usize; 3];
    for (i, (got, want)) in bb.bb.iter().zip(reference).enumerate() {
        assert_eq!(
            bb.ids[i].res_id, want.res_seq,
            "{label}: residue {i} number"
        );
        assert_eq!(got.res_name, want.res_name, "{label}: residue {i} name");
        for (k, (kind, get, get_ref)) in KINDS.iter().enumerate() {
            match (get(got), get_ref(want)) {
                (None, None) => {}
                (Some(g), Some(w)) => {
                    let d = circ_delta(g, w);
                    assert!(
                        d < AGREE_TOL_DEG,
                        "{label}: {kind} of {} {}: proxide {g:.6}°, biotite {w:.6}° (|Δ| {d:.6}°)",
                        want.res_name,
                        want.res_seq
                    );
                    if circ_delta(-g, w) > DETECT_DEG {
                        detected[k] += 1;
                    }
                }
                (g, w) => panic!(
                    "{label}: {kind} of {} {} defined-ness differs: proxide {g:?}, biotite {w:?}",
                    want.res_name, want.res_seq
                ),
            }
        }
    }
    detected
}

fn assert_negative_control(label: &str, detected: [usize; 3]) {
    for (k, (kind, _, _)) in KINDS.iter().enumerate() {
        assert!(
            detected[k] >= MIN_DETECTED,
            "{label}: a sign flip of {kind} is detected on only {} residues (< {MIN_DETECTED}); \
             the comparison cannot catch a -IUPAC {kind}",
            detected[k]
        );
    }
}

#[test]
fn per_dihedral_matches_biotite_iupac_on_1crn() {
    let reference = load_reference();
    let opts = BackboneOptions::default().with_missing_atoms(MissingAtomPolicy::PerDihedral);
    let bb = extract_f64_backbone_with_options(&processed_crambin(), &opts).expect("extract");
    let detected = assert_matches_reference("PerDihedral", &bb, &reference);
    assert_negative_control("PerDihedral", detected);
}

#[test]
fn legacy_compact_matches_biotite_iupac_on_1crn() {
    let reference = load_reference();
    let opts = BackboneOptions::default()
        .with_missing_atoms(MissingAtomPolicy::LegacyCompact)
        .with_chain_breaks(proxide_confind::ChainBreakPolicy::Bridge);
    let bb = extract_f64_backbone_with_options(&processed_crambin(), &opts).expect("extract");
    let detected = assert_matches_reference("LegacyCompact", &bb, &reference);
    assert_negative_control("LegacyCompact", detected);
}

#[test]
fn load_pdb_f64_matches_biotite_iupac_on_1crn() {
    let reference = load_reference();
    let bb = load_pdb_f64(crambin_path()).expect("load_pdb_f64 1CRN");
    let detected = assert_matches_reference("load_pdb_f64", &bb, &reference);
    assert_negative_control("load_pdb_f64", detected);
}

#[test]
fn circ_delta_is_wrap_aware() {
    assert!((circ_delta(179.5, -179.5) - 1.0).abs() < 1e-12);
    assert!((circ_delta(-180.0, 180.0)).abs() < 1e-12);
    assert!((circ_delta(10.0, -10.0) - 20.0).abs() < 1e-12);
}
