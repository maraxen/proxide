//! Empirical Adversarial Verification Suite for `proxide-rotlib-mosaist`
//!
//! Author: challenger_m3_rotlib (teamwork_challenger)
//! Milestone: M3 (Task ID: 261102_proxide-mosaist-cleanroom)

#[path = "helpers.rs"]
mod helpers;
use helpers::{write_minimal_lib, BinSpec, RotSpec};
use proxide_rotlib::RotlibError;
use proxide_rotlib_mosaist::{validate_grid, MosaistRotamerLibrary};

// ============================================================================
// TASK 3: SEGREGATED READER PHI-MAJOR SORTING VERIFICATION
// ============================================================================

#[test]
fn test_challenger_validate_grid_strictly_rejects_psi_major_order() {
    // 3x3 grid where inner loop iterates over phi (psi-major order):
    // Outer: psi in [-60, 0, 60]
    // Inner: phi in [-60, 0, 60]
    let mut psi_major_bins = Vec::new();
    let angles = [-60.0, 0.0, 60.0];
    for &psi in &angles {
        for &phi in &angles {
            psi_major_bins.push((phi, psi, 1.0));
        }
    }

    let result = validate_grid(&psi_major_bins);
    eprintln!(
        "[EMPIRICAL RESULT] validate_grid on psi-major 3x3 grid: {:?}",
        result
    );
    assert!(
        matches!(result, Err(RotlibError::InvalidFormat(ref msg)) if msg.contains("out of phi-major order")),
        "validate_grid must strictly reject psi-major order with InvalidFormat"
    );
}

#[test]
fn test_challenger_validate_grid_rejects_reversed_phi_major_order() {
    // 2x2 grid in reverse phi-major order
    let reversed_bins = vec![
        (60.0, 60.0, 1.0),
        (60.0, -60.0, 1.0),
        (-60.0, 60.0, 1.0),
        (-60.0, -60.0, 1.0),
    ];
    let result = validate_grid(&reversed_bins);
    assert!(
        matches!(result, Err(RotlibError::InvalidFormat(ref msg)) if msg.contains("out of phi-major order")),
        "validate_grid must reject reverse phi-major ordering"
    );
}

#[test]
fn test_challenger_validate_grid_rejects_single_adjacent_bin_swap() {
    // 2x2 grid where only bins 1 and 2 are swapped
    let swapped_bins = vec![
        (-60.0, -60.0, 1.0),
        (60.0, -60.0, 1.0), // Swapped! Should be (-60, 60)
        (-60.0, 60.0, 1.0), // Swapped! Should be (60, -60)
        (60.0, 60.0, 1.0),
    ];
    let result = validate_grid(&swapped_bins);
    assert!(
        matches!(result, Err(RotlibError::InvalidFormat(ref msg)) if msg.contains("out of phi-major order")),
        "validate_grid must reject adjacent bin swaps"
    );
}

#[test]
fn test_challenger_load_binary_strictly_rejects_psi_major_file() {
    // Write an actual binary rotlib.bin file with psi-major order
    let angles = [-60.0f32, 0.0, 60.0];
    let mut psi_major_bins = Vec::new();
    for &psi in &angles {
        for &phi in &angles {
            psi_major_bins.push(BinSpec { phi, psi, freq: 1.0 });
        }
    }

    let rotamers = vec![RotSpec {
        prob: 1.0,
        coords: vec![[1.0, 0.0, 0.0]],
    }];

    let tmp = write_minimal_lib("TST", &["CB"], &psi_major_bins, &rotamers);
    let result = MosaistRotamerLibrary::load(tmp.path());
    eprintln!(
        "[EMPIRICAL RESULT] MosaistRotamerLibrary::load on psi-major binary file: {:?}",
        result
    );

    assert!(
        matches!(result, Err(RotlibError::InvalidFormat(ref msg)) if msg.contains("out of phi-major order")),
        "MosaistRotamerLibrary::load must strictly reject a binary grid file that is not sorted phi-major"
    );
}

#[test]
fn test_challenger_load_binary_accepts_valid_phi_major_file() {
    // Write an actual binary rotlib.bin file with phi-major order
    let angles = [-60.0f32, 0.0, 60.0];
    let mut phi_major_bins = Vec::new();
    for &phi in &angles {
        for &psi in &angles {
            phi_major_bins.push(BinSpec { phi, psi, freq: 1.0 });
        }
    }

    let rotamers = vec![RotSpec {
        prob: 1.0,
        coords: vec![[1.0, 0.0, 0.0]],
    }];

    let tmp = write_minimal_lib("TST", &["CB"], &phi_major_bins, &rotamers);
    let result = MosaistRotamerLibrary::load(tmp.path());
    assert!(
        result.is_ok(),
        "MosaistRotamerLibrary::load must successfully load a correctly sorted phi-major file"
    );
}
