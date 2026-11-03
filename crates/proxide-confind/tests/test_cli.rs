//! CLI acceptance tests (spec §3.9.6 items 11–12) for the `confind` binary.

mod common;

use common::crambin_path;
use std::path::PathBuf;
use std::process::{Command, Output};

fn rlib() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../data/rotlibs/proxide-rotlib-dunbrack2010-ccd.pb.zst")
}

/// Run `confind` on crambin restricted to a few residues (keeps debug-build runs short).
fn confind(extra: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_confind"))
        .args(["--p", crambin_path().to_str().unwrap()])
        .args(["--rLib", rlib().to_str().unwrap()])
        .args(["--sel", "A:1-6"])
        .args(extra)
        .output()
        .expect("failed to spawn confind")
}

fn stdout(extra: &[&str]) -> String {
    let out = confind(extra);
    assert!(
        out.status.success(),
        "confind {extra:?} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap()
}

fn header(text: &str, key: &str) -> String {
    let prefix = format!("# {key}\t");
    text.lines()
        .find_map(|l| l.strip_prefix(&prefix))
        .unwrap_or_else(|| panic!("header line for {key} missing"))
        .to_string()
}

fn expect_failure(extra: &[&str], needles: &[&str]) {
    let out = confind(extra);
    assert!(!out.status.success(), "confind {extra:?} must fail");
    let err = String::from_utf8_lossy(&out.stderr).to_string();
    for n in needles {
        assert!(err.contains(n), "stderr for {extra:?} lacks '{n}': {err}");
    }
}

#[test]
fn header_records_defaults_and_no_flags_equals_explicit_defaults() {
    let plain = stdout(&[]);
    assert_eq!(header(&plain, "cbclash"), "all-residues");
    assert_eq!(header(&plain, "crowdmode"), "probability-weighted");
    assert_eq!(header(&plain, "freedommode"), "rms");
    assert_eq!(header(&plain, "scope"), "all");
    assert_eq!(header(&plain, "dclash"), "2");
    assert_eq!(header(&plain, "dcont"), "3");
    assert_eq!(header(&plain, "dcut"), "25");
    assert_eq!(header(&plain, "taulo"), "0.5");
    assert_eq!(header(&plain, "tauhi"), "2");

    let explicit = stdout(&[
        "--cbclash", "all-residues", "--crowdmode", "probability-weighted", "--freedommode",
        "rms", "--scope", "all", "--dclash", "2.0", "--dcont", "3.0", "--dcut", "25.0",
        "--taulo", "0.5", "--tauhi", "2.0",
    ]);
    assert_eq!(plain, explicit);
}

#[test]
fn every_enum_combination_runs_and_is_echoed() {
    for cb in ["ala-only", "all-residues"] {
        for crowd in ["probability-weighted", "pruned-count-fraction"] {
            for fm in ["rms", "single-threshold"] {
                // Flags in an order different from the table, to check order independence.
                let out = stdout(&["--freedommode", fm, "--cbclash", cb, "--crowdmode", crowd]);
                assert_eq!(header(&out, "cbclash"), cb);
                assert_eq!(header(&out, "crowdmode"), crowd);
                assert_eq!(header(&out, "freedommode"), fm);
                assert!(out.lines().any(|l| l.starts_with("freedom\t")));
                assert!(out.lines().any(|l| l.starts_with("crowdedness\t")));
            }
        }
    }
}

#[test]
fn crowdmode_changes_only_crowdedness_lines() {
    let a = stdout(&["--crowdmode", "probability-weighted"]);
    let b = stdout(&["--crowdmode", "pruned-count-fraction"]);
    let body = |t: &str, keep: bool| -> Vec<String> {
        t.lines()
            .filter(|l| !l.starts_with('#') && l.starts_with("crowdedness") == keep)
            .map(str::to_string)
            .collect()
    };
    assert_eq!(body(&a, false), body(&b, false));
    assert_ne!(body(&a, true), body(&b, true));
}

#[test]
fn numeric_flags_are_accepted_and_echoed() {
    let out = stdout(&[
        "--dclash", "1.8", "--dcont", "3.5", "--dcut", "26", "--taulo", "0.25", "--tauhi", "1.5",
    ]);
    assert_eq!(header(&out, "dclash"), "1.8");
    assert_eq!(header(&out, "dcont"), "3.5");
    assert_eq!(header(&out, "dcut"), "26");
    assert_eq!(header(&out, "taulo"), "0.25");
    assert_eq!(header(&out, "tauhi"), "1.5");
}

#[test]
fn scope_list_runs_and_limits_freedom_lines() {
    let out = stdout(&["--scope", "A:1-3"]);
    assert!(header(&out, "scope").starts_with("A:1-3"));
    let freedom_lines = out.lines().filter(|l| l.starts_with("freedom\t")).count();
    let crowd_lines = out.lines().filter(|l| l.starts_with("crowdedness\t")).count();
    assert_eq!(freedom_lines, 3, "freedom is reported only for in-scope residues");
    assert_eq!(crowd_lines, 6);
}

#[test]
fn scope_file_is_read() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("scope.txt");
    std::fs::write(&path, "# residues\nA:1-2\nA:4\n").unwrap();
    let out = stdout(&["--scope", path.to_str().unwrap()]);
    assert_eq!(out.lines().filter(|l| l.starts_with("freedom\t")).count(), 3);
}

#[test]
fn invalid_values_fail_naming_option_and_value() {
    expect_failure(&["--cbclash", "bogus"], &["--cbclash", "bogus", "ala-only, all-residues"]);
    expect_failure(
        &["--crowdmode", "median"],
        &["--crowdmode", "median", "probability-weighted, pruned-count-fraction"],
    );
    expect_failure(&["--freedommode", "x"], &["--freedommode", "x", "rms, single-threshold"]);
    expect_failure(&["--dclash", "abc"], &["--dclash", "abc"]);
    expect_failure(&["--dclash", "0"], &["--dclash", "d_clash"]);
    expect_failure(&["--dcont", "-1"], &["--dcont"]);
    expect_failure(&["--dcut", "nan"], &["--dcut"]);
    expect_failure(&["--taulo", "inf"], &["--taulo"]);
    expect_failure(&["--tauhi", "0"], &["--tauhi"]);
    expect_failure(&["--taulo", "3"], &["--taulo", "--tauhi"]);
    expect_failure(&["--dcut", "10"], &["--dcut", "required"]);
    expect_failure(&["--scope", "Z:1-3"], &["--scope", "Z:1-3"]);
    expect_failure(&["--scope", "A:x-3"], &["--scope", "A:x-3"]);
    expect_failure(&["--cbclash"], &["--cbclash", "missing value"]);
}

#[test]
fn tau_order_is_only_enforced_for_rms() {
    let out = stdout(&["--freedommode", "single-threshold", "--taulo", "3"]);
    assert_eq!(header(&out, "taulo"), "3");
}

#[test]
fn output_is_identical_across_thread_counts() {
    let one = stdout(&["--threads", "1"]);
    let many = stdout(&["--threads", "4"]);
    assert_eq!(one, many);
}
