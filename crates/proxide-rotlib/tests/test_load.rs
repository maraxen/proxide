#[path = "helpers.rs"]
mod helpers;
use helpers::{real_rotlib_path, write_minimal_lib, BinSpec, RotSpec};
use proxide_rotlib::{RotamerLibrary, RotlibError};

/// Probe Mosaist's own ALA shape via proxide's `RotamerLibrary::load()` (the MSL binary
/// loader), so backlog #5244's synthetic 1x1/p=1/num_chi=0 ALA entry can be judged against
/// what Mosaist itself actually stores for ALA -- WITHOUT reading or copying rotlib.bin's
/// raw bytes (CC-BY-NC-SA; see CLAUDE.md). Only aggregate statistics are printed.
///
/// Decision rule (spec-challenger review 260923_loop_sprint23_coherence, objection #2):
///   1 rotamer/bin, p == 1, CB spread < 0.05 A across bins => the 1x1 model this sprint
///   adds is faithful to Mosaist's own representation.
///   Otherwise: the divergence is recorded here (and as a debt) without changing the
///   synthetic model this sprint -- decision b only covers the CB *constant*, not the
///   bin/rotamer topology, and re-deriving ALA's grid shape from Mosaist is out of scope
///   for #5244 (see fixer_prompt step 2).
#[test]
#[ignore] // Run only when Mosaist is available; prints aggregates only, no raw data.
fn test_mosaist_ala_shape_probe() {
    let rotlib_path = real_rotlib_path();
    if !rotlib_path.exists() {
        eprintln!(
            "SKIP: Mosaist rotlib.bin not available at {}",
            rotlib_path.display()
        );
        return;
    }
    let lib = RotamerLibrary::load(&rotlib_path).expect("failed to load Mosaist rotlib.bin");
    assert!(
        lib.contains_aa("ALA"),
        "Mosaist rotlib.bin has no ALA entry at all -- decision rule cannot be evaluated"
    );

    let summary = lib
        .aa_grid_summary("ALA")
        .expect("aa_grid_summary(\"ALA\") failed after contains_aa(\"ALA\") returned true");

    println!("Mosaist ALA grid summary (aggregates only, no raw coordinates/probabilities):");
    println!("  bin_count:              {}", summary.bin_count);
    println!(
        "  rotamers per bin:       min={} max={}",
        summary.min_rotamers_per_bin, summary.max_rotamers_per_bin
    );
    println!(
        "  probability range:      min={:.6} max={:.6}",
        summary.min_probability, summary.max_probability
    );
    println!(
        "  max CB spread (A):      {:?}",
        summary.max_cb_spread_angstrom
    );

    let is_single_rotamer_per_bin =
        summary.min_rotamers_per_bin == 1 && summary.max_rotamers_per_bin == 1;
    let is_p_one =
        (summary.min_probability - 1.0).abs() < 1e-6 && (summary.max_probability - 1.0).abs() < 1e-6;
    let cb_spread_ok = summary
        .max_cb_spread_angstrom
        .map(|d| d < 0.05)
        .unwrap_or(true); // no CB or <2 bins: nothing to disagree about

    let faithful = is_single_rotamer_per_bin && is_p_one && cb_spread_ok;

    if faithful {
        println!(
            "DECISION RULE: FAITHFUL -- Mosaist's ALA is 1 rotamer/bin, p=1, CB spread < 0.05 A. \
             The synthetic 1x1/p=1/num_chi=0 ALA entry this sprint adds matches Mosaist's own model."
        );
    } else {
        println!(
            "DECISION RULE: DIVERGENCE -- Mosaist's ALA does NOT match the 1x1/p=1 assumption \
             (single_rotamer_per_bin={is_single_rotamer_per_bin}, p_one={is_p_one}, \
             cb_spread_ok={cb_spread_ok}). Per fixer_prompt step 2, the synthetic model is NOT \
             changed this sprint; record this divergence as a debt (backlog #5244) for future \
             research rather than silently accepting or silently ignoring it."
        );
    }
}

const AA_NAMES: &[&str] = &[
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET",
    "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
];

#[test]
#[ignore] // Run only when Mosaist is available
fn test_load_all_aa_names_present() {
    let rotlib_path = real_rotlib_path();
    if !rotlib_path.exists() {
        eprintln!(
            "Skipping: ROTLIB_PATH not available at {}",
            rotlib_path.display()
        );
        return;
    }
    let lib = RotamerLibrary::load(&rotlib_path).unwrap();
    for &aa in AA_NAMES {
        assert!(lib.contains_aa(aa), "missing AA: {aa}");
    }
}

#[test]
#[ignore] // Run only when Mosaist is available
fn test_num_rotamers_sentinel_positive() {
    let rotlib_path = real_rotlib_path();
    if !rotlib_path.exists() {
        eprintln!(
            "Skipping: ROTLIB_PATH not available at {}",
            rotlib_path.display()
        );
        return;
    }
    let lib = RotamerLibrary::load(&rotlib_path).unwrap();
    for &aa in AA_NAMES {
        let n = lib.num_rotamers(aa, 9999.0, 9999.0, false).unwrap();
        assert!(n > 0, "{aa}: sentinel bin has 0 rotamers");
    }
}

#[test]
fn test_load_truncated_file() {
    // Write a file that starts reading a record but is truncated mid-i32
    // "TST\0" (4 bytes) + partial i32 (2 bytes instead of 4) = 6 bytes
    let tmp = write_minimal_lib(
        "TST",
        &["CB"],
        &[BinSpec {
            phi: 0.0,
            psi: 0.0,
            freq: 1.0,
        }],
        &[RotSpec {
            prob: 0.9,
            coords: vec![[1.0, 0.0, 0.0]],
        }],
    );
    let path = tmp.path();
    // Truncate after AA name + 2 bytes of nc field (need 4 bytes total)
    std::fs::write(path, b"TST\x00\x00\x00").unwrap();
    let result = RotamerLibrary::load(path);
    assert!(
        matches!(result, Err(RotlibError::Io(_))),
        "expected Io error, got: {:?}",
        result
    );
}

#[test]
fn test_load_non_rectangular_grid() {
    // 3 bins but unique_phi=2, unique_psi=2 → product=4 ≠ 3
    let tmp = write_minimal_lib(
        "TST",
        &["CB"],
        &[
            BinSpec {
                phi: -60.0,
                psi: -60.0,
                freq: 1.0,
            },
            BinSpec {
                phi: -60.0,
                psi: 60.0,
                freq: 1.0,
            },
            BinSpec {
                phi: 60.0,
                psi: -60.0,
                freq: 1.0,
            },
        ],
        &[RotSpec {
            prob: 0.9,
            coords: vec![[1.0, 0.0, 0.0]],
        }],
    );
    let result = RotamerLibrary::load(tmp.path());
    assert!(
        matches!(result, Err(RotlibError::InvalidFormat(_))),
        "expected InvalidFormat, got: {:?}",
        result
    );
}

#[test]
fn test_load_duplicate_phi_psi() {
    // 2 bins with identical (phi, psi)
    let tmp = write_minimal_lib(
        "TST",
        &["CB"],
        &[
            BinSpec {
                phi: -60.0,
                psi: -40.0,
                freq: 1.0,
            },
            BinSpec {
                phi: -60.0,
                psi: -40.0,
                freq: 2.0,
            },
        ],
        &[RotSpec {
            prob: 0.9,
            coords: vec![[1.0, 0.0, 0.0]],
        }],
    );
    let result = RotamerLibrary::load(tmp.path());
    assert!(
        matches!(result, Err(RotlibError::InvalidFormat(_))),
        "expected InvalidFormat, got: {:?}",
        result
    );
}
