// debt #1898: ConFind must propagate rotamer-library errors (e.g. an amino acid missing
// from the library) instead of silently dropping that amino acid from rotamer_grids.
//
// This test builds a synthetic protobuf rotamer library in-process (no external files,
// no env gate) with all 18 AA_NAMES except TRP, and asserts that caching any residue on a
// synthetic backbone fails with `ConFindError::RotlibError(RotlibError::UnknownAa("TRP"))` —
// `cache_residue` tries every AA_NAMES entry for every backbone position (to score all
// possible substitutions there), so a single missing amino acid always surfaces.

mod common;

use common::make_synthetic_backbone;
use proxide_confind::{ConFind, ConFindError, ResidueIndex, AA_NAMES};
use proxide_rotlib::{rotlib_v1, RotamerLibrary, RotlibError};
use std::sync::Arc;

/// Build a synthetic library containing `aas` (each with one CB rotamer) in memory.
fn build_library(aas: &[&str]) -> RotamerLibrary {
    let mut residues = Vec::new();
    for &aa in aas {
        let proto_rotamers = vec![rotlib_v1::Rotamer {
            prob: 0.9,
            chi: vec![],
            coords: vec![rotlib_v1::Vec3 {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            }],
        }];
        let proto_bins = vec![rotlib_v1::Bin {
            phi: 0.0,
            psi: 0.0,
            freq: 1.0,
            rotamers: proto_rotamers,
        }];
        residues.push(rotlib_v1::ResidueEntry {
            code: aa.to_string(),
            atom_names: vec!["CB".to_string()],
            num_chi: 0,
            phi_centers: vec![0.0],
            psi_centers: vec![0.0],
            default_bin: 0,
            bins: proto_bins,
        });
    }
    let pb = rotlib_v1::RotamerLibrary {
        version: 1,
        provenance: "synthetic".to_string(),
        attribution: "synthetic test library".to_string(),
        data_license: "MIT".to_string(),
        geometry_mode: rotlib_v1::GeometryMode::Precomputed as i32,
        geometry_source: "".to_string(),
        geometry_license: "".to_string(),
        residues,
    };
    RotamerLibrary::from_protobuf(&pb).expect("synthetic library must parse")
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

    let lib = build_library(&aas_minus_trp);
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
    let lib = build_library(AA_NAMES);
    let rotlib = Arc::new(lib);

    let backbone = make_synthetic_backbone(1, 3.8);
    let confind = ConFind::new(rotlib, backbone, false);

    confind
        .cache_residue(ResidueIndex(0))
        .expect("caching must succeed: every AA_NAMES entry is present in the library");
}
