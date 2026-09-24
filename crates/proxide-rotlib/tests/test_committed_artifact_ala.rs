/// Non-ignored test on the actually-committed artifact (backlog #5244): confirms the
/// synthetic ALA entry is present in data/rotlibs/proxide-rotlib-dunbrack2010-ccd.pb.zst
/// exactly as documented (1 bin, 1 rotamer, p=1, num_chi=0, correct CB torsion), and that
/// the library still loads via the public `load_pb` path (not just decodes as raw protobuf).
use proxide_rotlib::RotamerLibrary;

fn committed_artifact_path() -> std::path::PathBuf {
    std::path::PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../data/rotlibs/proxide-rotlib-dunbrack2010-ccd.pb.zst"
    ))
}

#[test]
fn test_committed_artifact_has_synthetic_ala() {
    let path = committed_artifact_path();
    let lib = RotamerLibrary::load_pb(&path)
        .unwrap_or_else(|e| panic!("failed to load committed artifact at {}: {}", path.display(), e));

    assert!(
        lib.contains_aa("ALA"),
        "committed artifact must contain an ALA entry (backlog #5244)"
    );

    let summary = lib
        .aa_grid_summary("ALA")
        .expect("aa_grid_summary(\"ALA\") failed on committed artifact");
    assert_eq!(summary.bin_count, 1, "ALA must have exactly 1 bin");
    assert_eq!(
        summary.min_rotamers_per_bin, 1,
        "ALA bin must have exactly 1 rotamer (min)"
    );
    assert_eq!(
        summary.max_rotamers_per_bin, 1,
        "ALA bin must have exactly 1 rotamer (max)"
    );
    assert!(
        (summary.min_probability - 1.0).abs() < 1e-6,
        "ALA rotamer probability must be 1.0, got min={}",
        summary.min_probability
    );
    assert!(
        (summary.max_probability - 1.0).abs() < 1e-6,
        "ALA rotamer probability must be 1.0, got max={}",
        summary.max_probability
    );

    // num_chi isn't exposed by aa_grid_summary (it's a load_pb-internal AaEntry field),
    // but it's exercised end-to-end: rotamer_probability_by_id / place_rotamer would fail
    // to build coordinates if the CB were still collapsed onto the backbone C (the FATAL
    // bug this sprint fixed). Confirm placement succeeds and lands at a physically
    // reasonable CB position (not the ~0.03 A collapse the bug produced).
    let placed = lib
        .place_rotamer(
            "ALA",
            -60.0,
            -45.0,
            0,
            false,
            [-1.458, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.551, 1.420, 0.0],
        )
        .expect("place_rotamer(\"ALA\", ...) failed on committed artifact");
    assert_eq!(placed.atoms.len(), 1, "ALA should place exactly 1 sidechain atom (CB)");
    let cb = placed.atoms[0].xyz;
    let ca = [0.0_f64, 0.0, 0.0];
    let d_ca_cb = ((cb[0] - ca[0]).powi(2) + (cb[1] - ca[1]).powi(2) + (cb[2] - ca[2]).powi(2)).sqrt();
    assert!(
        (d_ca_cb - 1.540).abs() < 0.01,
        "CA-CB bond length should be ~1.540 A, got {:.4}",
        d_ca_cb
    );
    let c = [0.551_f64, 1.420, 0.0];
    let d_c_cb = ((cb[0] - c[0]).powi(2) + (cb[1] - c[1]).powi(2) + (cb[2] - c[2]).powi(2)).sqrt();
    assert!(
        d_c_cb > 2.3,
        "CB must not collapse onto backbone C (backlog #5244 FATAL bug): |CB-C| = {:.4} (must be > 2.3 A)",
        d_c_cb
    );
}
