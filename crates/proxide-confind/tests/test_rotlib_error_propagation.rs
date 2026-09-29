// debt #1898: ConFind must propagate rotamer-library errors (e.g. an amino acid missing
// from the library) instead of silently dropping that amino acid from rotamer_grids.
//
// This test builds a synthetic MSL-format rotamer library in-process (no external files,
// no env gate) with all 18 AA_NAMES except TRP, and asserts that caching any residue on a
// synthetic backbone fails with `ConFindError::RotlibError(RotlibError::UnknownAa("TRP"))` —
// `cache_residue_impl` tries every AA_NAMES entry for every backbone position (to score all
// possible substitutions there), so a single missing amino acid always surfaces.
//
// The MSL binary writer below is a copy of proxide-rotlib's own
// `tests/helpers.rs::write_minimal_lib` — confind's integration tests cannot import a
// sibling crate's private test helper module, so the minimal format is duplicated here
// rather than gated behind an env var or `#[ignore]`.

mod common;

use common::make_synthetic_backbone;
use proxide_confind::{ConFind, ConFindError, ResidueIndex, AA_NAMES};
use proxide_rotlib::{RotamerLibrary, RotlibError};
use std::io::Write;
use std::sync::Arc;

/// Minimal single-bin, single-rotamer, single-atom (CB) MSL entry for `aa`.
/// Mirrors proxide-rotlib/tests/helpers.rs::write_minimal_lib exactly (na=1, nc=0, one bin
/// at phi=0/psi=0 covering every query via `default_bin`, one rotamer).
fn write_minimal_entry(f: &mut impl Write, aa: &str) {
    let na: i32 = 1;
    let nc: i32 = 0;
    let nb: i32 = 1;
    let nr: i32 = 1;

    // AA name (C string)
    f.write_all(aa.as_bytes()).unwrap();
    f.write_all(&[0u8]).unwrap();

    // nc, na, nb
    f.write_all(&nc.to_le_bytes()).unwrap();
    f.write_all(&na.to_le_bytes()).unwrap();
    f.write_all(&nb.to_le_bytes()).unwrap();

    // sidechain atom names
    f.write_all(b"CB").unwrap();
    f.write_all(&[0u8]).unwrap();

    // one bin descriptor: phi=0.0, psi=0.0, freq=1.0
    f.write_all(&0.0f32.to_le_bytes()).unwrap();
    f.write_all(&0.0f32.to_le_bytes()).unwrap();
    f.write_all(&1.0f32.to_le_bytes()).unwrap();

    // rotamer data for the one bin: nr, then one rotamer (prob, no chi since nc=0, one atom xyz)
    f.write_all(&nr.to_le_bytes()).unwrap();
    f.write_all(&0.9f32.to_le_bytes()).unwrap(); // prob
    for v in [1.0f32, 0.0, 0.0] {
        f.write_all(&v.to_le_bytes()).unwrap();
    }
}

/// Build a synthetic library containing `aas` (each with one CB rotamer) and load it.
fn build_library(aas: &[&str]) -> (tempfile::NamedTempFile, RotamerLibrary) {
    let mut f = tempfile::NamedTempFile::new().unwrap();
    for &aa in aas {
        write_minimal_entry(&mut f, aa);
    }
    f.flush().unwrap();
    let lib = RotamerLibrary::load(f.path()).expect("synthetic library must parse");
    (f, lib)
}

#[test]
fn cache_residue_propagates_unknown_aa_error() {
    // AA_NAMES minus TRP — derived from the real constant so this test can't drift from it.
    let aas_minus_trp: Vec<&str> = AA_NAMES.iter().copied().filter(|&aa| aa != "TRP").collect();
    assert_eq!(
        aas_minus_trp.len(),
        AA_NAMES.len() - 1,
        "expected exactly TRP to be excluded"
    );

    let (_tmp, lib) = build_library(&aas_minus_trp);
    let rotlib = Arc::new(lib);

    let backbone = make_synthetic_backbone(1, 3.8);
    let confind = ConFind::new(rotlib, backbone, false);

    let err = confind
        .cache_residue(ResidueIndex(0))
        .expect_err("caching must fail: TRP is not in the synthetic library");

    assert!(
        matches!(&err, ConFindError::RotlibError(RotlibError::UnknownAa(a)) if a == "TRP"),
        "expected RotlibError::UnknownAa(\"TRP\"), got {err:?}"
    );
}

#[test]
fn cache_residue_positive_control_with_trp_present_succeeds() {
    // Full AA_NAMES, TRP included — positive control for the negative test above.
    let (_tmp, lib) = build_library(&AA_NAMES);
    let rotlib = Arc::new(lib);

    let backbone = make_synthetic_backbone(1, 3.8);
    let confind = ConFind::new(rotlib, backbone, false);

    confind
        .cache_residue(ResidueIndex(0))
        .expect("caching must succeed: every AA_NAMES entry is present in the library");
}
