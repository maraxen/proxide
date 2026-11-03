#[path = "helpers.rs"]
mod helpers;
use helpers::{real_rotlib_path, write_minimal_lib, BinSpec, RotSpec};
use proxide_rotlib::RotlibError;
use proxide_rotlib_mosaist::{validate_grid, RotamerLibrary};

#[test]
#[ignore] // Run only when Mosaist is available
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
    assert!(lib.contains_aa("ALA"));

    let summary = lib
        .aa_grid_summary("ALA")
        .expect("aa_grid_summary(\"ALA\") failed after contains_aa(\"ALA\") returned true");

    println!("Mosaist ALA grid summary:");
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
        let n = lib.num_rotamers(aa, None, None, false).unwrap();
        assert!(n > 0, "{aa}: sentinel bin has 0 rotamers");
    }
}

#[test]
fn test_load_truncated_file() {
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

#[test]
fn test_validate_grid_phi_major_sorting() {
    // 2x2 grid in correct phi-major order:
    // bin 0: phi=-60, psi=-40
    // bin 1: phi=-60, psi=40
    // bin 2: phi=60, psi=-40
    // bin 3: phi=60, psi=40
    let valid_bins = vec![
        (-60.0, -40.0, 1.0),
        (-60.0, 40.0, 2.0),
        (60.0, -40.0, 1.5),
        (60.0, 40.0, 2.5),
    ];
    let res = validate_grid(&valid_bins);
    assert!(res.is_ok(), "Expected valid phi-major grid to pass");
    let (phis, psis) = res.unwrap();
    assert_eq!(phis, vec![-60.0, 60.0]);
    assert_eq!(psis, vec![-40.0, 40.0]);

    // 2x2 grid in psi-major order (out of phi-major order):
    // bin 0: phi=-60, psi=-40
    // bin 1: phi=60, psi=-40 (should be phi=-60, psi=40)
    // bin 2: phi=-60, psi=40
    // bin 3: phi=60, psi=40
    let out_of_order_bins = vec![
        (-60.0, -40.0, 1.0),
        (60.0, -40.0, 1.5),
        (-60.0, 40.0, 2.0),
        (60.0, 40.0, 2.5),
    ];
    let err = validate_grid(&out_of_order_bins);
    assert!(
        matches!(err, Err(RotlibError::InvalidFormat(ref msg)) if msg.contains("out of phi-major order")),
        "Expected InvalidFormat mentioning phi-major order, got {:?}",
        err
    );
}
