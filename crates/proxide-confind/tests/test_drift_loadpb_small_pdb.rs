mod common;

use common::load_real_backbone;
use proxide_confind::ConFind;
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::Write;
use std::sync::Arc;

// Reference contact degrees from Mosaist testConFind on small.pdb + rotlib.bin
// Format: (chain_a, res_a, chain_b, res_b, contact_degree)
const REF_CONTACTS: &[(&str, i32, &str, i32, f64)] = &[
    ("A", 1, "A", 2, 0.003188),
    ("A", 1, "A", 3, 0.077085),
    ("A", 1, "A", 4, 0.033422),
    ("A", 1, "A", 5, 0.000001),
    ("A", 1, "A", 7, 0.000050),
    ("A", 1, "B", 2, 0.000023),
    ("A", 2, "A", 3, 0.003045),
    ("A", 2, "A", 4, 0.000000),
    ("A", 2, "A", 5, 0.001970),
    ("A", 2, "A", 6, 0.091159),
    ("A", 2, "A", 7, 0.000000),
    ("A", 2, "B", 1, 0.002328),
    ("A", 2, "B", 2, 0.024452),
    ("A", 2, "B", 3, 0.000000),
    ("A", 2, "B", 4, 0.000003),
    ("A", 2, "B", 5, 0.074874),
    ("A", 2, "B", 6, 0.000000),
    ("A", 3, "A", 4, 0.002455),
    ("A", 3, "A", 5, 0.000000),
    ("A", 3, "A", 6, 0.005062),
    ("A", 3, "A", 7, 0.072369),
    ("A", 4, "A", 5, 0.003012),
    ("A", 4, "A", 7, 0.010282),
    ("A", 4, "B", 2, 0.000018),
    ("A", 4, "B", 5, 0.000000),
    ("A", 4, "B", 6, 0.000023),
    ("A", 5, "A", 6, 0.000306),
    ("A", 5, "A", 7, 0.000000),
    ("A", 5, "B", 2, 0.070867),
    ("A", 5, "B", 3, 0.000013),
    ("A", 5, "B", 4, 0.000000),
    ("A", 5, "B", 5, 0.258400),
    ("A", 5, "B", 6, 0.080239),
    ("A", 6, "A", 7, 0.001685),
    ("A", 6, "B", 2, 0.000000),
    ("A", 6, "B", 4, 0.000002),
    ("A", 6, "B", 5, 0.043550),
    ("B", 1, "B", 2, 0.018478),
    ("B", 1, "B", 3, 0.012019),
    ("B", 1, "B", 4, 0.011080),
    ("B", 1, "B", 5, 0.000008),
    ("B", 1, "B", 7, 0.000000),
    ("B", 2, "B", 3, 0.027230),
    ("B", 2, "B", 4, 0.000000),
    ("B", 2, "B", 5, 0.002379),
    ("B", 2, "B", 6, 0.088785),
    ("B", 3, "B", 4, 0.011912),
    ("B", 3, "B", 5, 0.000000),
    ("B", 3, "B", 6, 0.017732),
    ("B", 3, "B", 7, 0.011423),
    ("B", 4, "B", 5, 0.000545),
    ("B", 4, "B", 6, 0.000000),
    ("B", 4, "B", 7, 0.017669),
    ("B", 5, "B", 6, 0.000386),
    ("B", 5, "B", 7, 0.000000),
    ("B", 6, "B", 7, 0.001298),
];

// Per-pair "notable drift" report threshold. This is NOT the pass/fail gate (see
// BASELINE_MAX_DELTA_BEFORE below) -- it only controls which pairs get listed in the
// "pairs above threshold" section of the printed report.
const REPORT_TOLERANCE: f64 = 1e-4;

/// BEFORE baseline (backlog #5244, decision c): re-measured 2026-09-23 on the SAME
/// fixture (small.pdb + the pre-#5244 proxide-rotlib-dunbrack2010-ccd.pb.zst, sha256
/// 13264f972b782970141032617e1395932714abece81d27980b5749e15268579b, ALA silently
/// omitted from contact calculations) and the SAME metric (max|delta| across matched
/// REF_CONTACTS pairs), using code checked out at 47c397d (the parent of a12de2b, i.e.
/// before ConFind started propagating rotamer-library errors instead of silently
/// dropping unknown amino acids). Obtained via `git worktree add target/before-47c397d
/// 47c397d` from this worktree and running this same test there with
/// PROXIDE_ROTLIB_PB pointed at that worktree's own (pre-#5244) copy of the artifact.
/// Full report: target/logs/trackA23_step6_BEFORE_full_report.log.
///
/// This reproduces the max|delta| = 0.043081 previously cited in the #869 record
/// (spec-challenger review 260923_loop_sprint23_coherence, objection #6) to 6 decimal
/// places, confirming that citation was measuring the same fixture/metric.
const BASELINE_MAX_DELTA_BEFORE: f64 = 0.043081;
const BASELINE_MEAN_DELTA_BEFORE: f64 = 0.004591;
const BASELINE_MEDIAN_DELTA_BEFORE: f64 = 0.000830;
/// BEFORE matched 54 of 56 REF_CONTACTS pairs (2 missing, 2 unexpected-in-actual).
const BASELINE_MATCHED_COUNT_BEFORE: usize = 54;
const BASELINE_MISSING_BEFORE: usize = 2;

fn all_res(cf: &ConFind) -> Vec<proxide_confind::ResidueIndex> {
    (0..cf.n_residues() as u32)
        .map(proxide_confind::ResidueIndex)
        .collect()
}

/// Default path to the committed rotamer-library artifact, resolved relative to this
/// crate's manifest dir (not the workspace root or an external main-checkout path) so
/// the test is reproducible from a fresh clone of this worktree without depending on
/// any path outside it. Override with PROXIDE_ROTLIB_PB for ad hoc comparisons against
/// a different build (e.g. an A/B regeneration check).
fn default_pb_path() -> std::path::PathBuf {
    std::path::PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../data/rotlibs/proxide-rotlib-dunbrack2010-ccd.pb.zst"
    ))
}

#[test]
#[ignore]
fn measure_loadpb_drift_vs_master() {
    let pb_path = std::env::var("PROXIDE_ROTLIB_PB")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| default_pb_path());

    // Fail-fast (CLAUDE.md): a missing or unparseable input panics. This test's default
    // library path is the artifact committed to this worktree (backlog #5244), so
    // "missing" now means something is actually broken, not "Mosaist/the artifact
    // wasn't regenerated yet" -- there is nothing left to silently skip past.
    assert!(
        pb_path.exists(),
        "rotamer library not found at {} -- this is a fail-loud test, not a skip. If \
         PROXIDE_ROTLIB_PB is unset, the default is this worktree's own committed \
         artifact; see crates/proxide-rotlib/README.md for the regeneration recipe.",
        pb_path.display()
    );

    let bb = load_real_backbone().unwrap_or_else(|| {
        panic!(
            "small.pdb fixture not found at the configured PDB_PATH (see \
             crates/proxide-confind/tests/common/mod.rs::real_pdb_path) -- this is a \
             fail-loud test, not a skip"
        )
    });

    let rlib = proxide_rotlib::RotamerLibrary::load_pb(&pb_path).unwrap_or_else(|e| {
        panic!(
            "failed to load protobuf library from {}: {}",
            pb_path.display(),
            e
        )
    });
    let rlib = Arc::new(rlib);

    let cf = ConFind::new(rlib, bb.clone(), false);
    let contact_list = cf.contacts(&all_res(&cf), 0.0).unwrap_or_else(|e| {
        panic!(
            "contacts() failed: {:?} -- if this is RotlibError::UnknownAa(\"ALA\"), the \
             library at {} is missing the synthetic ALA entry from backlog #5244; \
             rebuild it via the recipe in crates/proxide-rotlib/README.md",
            e,
            pb_path.display()
        )
    });

    assert!(
        !contact_list.pairs.is_empty(),
        "contact list must not be empty"
    );

    run_drift_comparison(&cf, &contact_list);
}

fn run_drift_comparison(cf: &ConFind, contact_list: &proxide_confind::ContactList) {
    // Build expected map: canonical key (chain_a, res_a, chain_b, res_b) → cd
    let expected: HashMap<(String, i32, String, i32), f64> = REF_CONTACTS
        .iter()
        .map(|&(ca, ra, cb, rb, cd)| ((ca.to_string(), ra, cb.to_string(), rb), cd))
        .collect();

    // Collect deltas. Missing reference pairs (present in REF_CONTACTS but absent from
    // actual) count as delta = |reference| -- a fully-dropped contact is not "no
    // evidence of drift", it is the largest possible drift for that pair (B3/B4-class
    // failure mode this project's CLAUDE.md warns about: a detector whose silence on a
    // missing case reads as success). This also means max_delta can never silently be
    // -inf: REF_CONTACTS is non-empty, so deltas is never empty once missing pairs are
    // folded in, even if contact_list.pairs were somehow empty (guarded separately above).
    let mut deltas: Vec<f64> = Vec::new();
    let mut matched_count = 0;
    let mut missing_from_actual = 0;
    let mut unexpected_in_actual = 0;
    let mut exceeding_tolerance = 0;
    let mut largest_drift: Vec<(String, i32, String, i32, f64, f64, f64)> = Vec::new();
    let mut missing_pairs: Vec<(String, i32, String, i32, f64)> = Vec::new();

    // Check all actual pairs against expected
    for (&(ri_a, ri_b), &actual) in contact_list.pairs.iter().zip(&contact_list.degrees) {
        let id_a = cf.residue_id(ri_a);
        let id_b = cf.residue_id(ri_b);
        let key = (
            id_a.chain_id.clone(),
            id_a.res_id,
            id_b.chain_id.clone(),
            id_b.res_id,
        );

        if let Some(&reference) = expected.get(&key) {
            let delta = (actual - reference).abs();
            deltas.push(delta);
            matched_count += 1;

            if delta >= REPORT_TOLERANCE {
                exceeding_tolerance += 1;
            }

            largest_drift.push((
                id_a.chain_id.clone(),
                id_a.res_id,
                id_b.chain_id.clone(),
                id_b.res_id,
                actual,
                reference,
                delta,
            ));
        } else {
            unexpected_in_actual += 1;
        }
    }

    // Check for missing pairs -- and fold each into deltas as delta = |reference|.
    let actual_pairs: std::collections::HashSet<(String, i32, String, i32)> = contact_list
        .pairs
        .iter()
        .map(|&(ri_a, ri_b)| {
            let id_a = cf.residue_id(ri_a);
            let id_b = cf.residue_id(ri_b);
            (
                id_a.chain_id.clone(),
                id_a.res_id,
                id_b.chain_id.clone(),
                id_b.res_id,
            )
        })
        .collect();

    for &(ca, ra, cb, rb, reference) in REF_CONTACTS {
        let key = (ca.to_string(), ra, cb.to_string(), rb);
        if !actual_pairs.contains(&key) {
            missing_from_actual += 1;
            deltas.push(reference.abs());
            missing_pairs.push((ca.to_string(), ra, cb.to_string(), rb, reference));
        }
    }

    // Sort largest_drift by delta descending
    largest_drift.sort_by(|a, b| b.6.partial_cmp(&a.6).unwrap_or(std::cmp::Ordering::Equal));

    // Compute statistics. deltas is guaranteed non-empty here: REF_CONTACTS is a
    // non-empty const, and every one of its pairs contributes exactly one delta (either
    // matched or folded-in-as-missing above) -- so max_delta can never default to -inf.
    assert!(
        !deltas.is_empty(),
        "internal invariant violated: deltas must be non-empty whenever REF_CONTACTS is \
         non-empty (every reference pair is either matched or counted as missing)"
    );
    let max_delta = deltas.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mean_delta = deltas.iter().sum::<f64>() / deltas.len() as f64;
    let mut sorted_deltas = deltas.clone();
    sorted_deltas.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_delta = if sorted_deltas.len() % 2 == 0 {
        (sorted_deltas[sorted_deltas.len() / 2 - 1] + sorted_deltas[sorted_deltas.len() / 2]) / 2.0
    } else {
        sorted_deltas[sorted_deltas.len() / 2]
    };

    // Build drift report BEFORE any assertion, so a failing gate still leaves a full
    // report on stderr/disk to diagnose from (spec-challenger review objection #5: the
    // old code asserted before building the report).
    let mut report = String::new();
    report.push('\n');
    report.push_str(
        "================================================================================\n",
    );
    report.push_str("DRIFT REPORT: load_pb vs. MASTER rotlib.bin\n");
    report.push_str(
        "================================================================================\n",
    );
    report.push('\n');
    report.push_str(&format!("Matched pairs:              {}\n", matched_count));
    report.push_str(&format!(
        "Missing from actual (PB):   {}  (folded into delta stats as delta=|reference|)\n",
        missing_from_actual
    ));
    report.push_str(&format!(
        "Unexpected in actual (PB):  {}\n",
        unexpected_in_actual
    ));
    report.push('\n');
    report.push_str(&format!(
        "Drift statistics (across {} deltas: {} matched + {} missing-as-worst-case):\n",
        deltas.len(),
        matched_count,
        missing_from_actual
    ));
    report.push_str(&format!(
        "  Max |\u{394}|:                  {:.6}\n",
        max_delta
    ));
    report.push_str(&format!(
        "  Mean |\u{394}|:                 {:.6}\n",
        mean_delta
    ));
    report.push_str(&format!(
        "  Median |\u{394}|:               {:.6}\n",
        median_delta
    ));
    report.push_str(&format!(
        "  Count exceeding {:.0e} (matched only): {} ({}%)\n",
        REPORT_TOLERANCE,
        exceeding_tolerance,
        if matched_count > 0 {
            (exceeding_tolerance * 100) / matched_count
        } else {
            0
        }
    ));
    report.push('\n');
    report.push_str(&format!(
        "BASELINE (BEFORE, ALA silently omitted, code at 47c397d): max|\u{394}|={:.6} \
         mean|\u{394}|={:.6} median|\u{394}|={:.6} matched={} missing={}\n",
        BASELINE_MAX_DELTA_BEFORE,
        BASELINE_MEAN_DELTA_BEFORE,
        BASELINE_MEDIAN_DELTA_BEFORE,
        BASELINE_MATCHED_COUNT_BEFORE,
        BASELINE_MISSING_BEFORE
    ));
    report.push('\n');

    if !missing_pairs.is_empty() {
        report.push_str("Missing pairs (reference present, actual absent):\n");
        for (ca, ra, cb, rb, reference) in &missing_pairs {
            report.push_str(&format!(
                "  {},{} -> {},{}  reference={:.6} (counted as delta={:.6})\n",
                ca, ra, cb, rb, reference, reference
            ));
        }
        report.push('\n');
    }

    // Classify terminal residues: residue 1 (N-terminal) or max res_id per chain (C-terminal).
    // From REF_CONTACTS the chains are A and B, each with res_ids 1..7.
    let terminal_res_ids: std::collections::HashSet<(String, i32)> = {
        let mut m = std::collections::HashMap::<String, (i32, i32)>::new();
        for &(ca, ra, cb, rb, _) in REF_CONTACTS {
            let e = m.entry(ca.to_string()).or_insert((i32::MAX, i32::MIN));
            e.0 = e.0.min(ra);
            e.1 = e.1.max(ra);
            let e = m.entry(cb.to_string()).or_insert((i32::MAX, i32::MIN));
            e.0 = e.0.min(rb);
            e.1 = e.1.max(rb);
        }
        m.iter()
            .flat_map(|(ch, &(mn, mx))| [ch.clone(), ch.clone()].into_iter().zip([mn, mx]))
            .collect()
    };

    let is_terminal = |chain: &str, res: i32| terminal_res_ids.contains(&(chain.to_string(), res));

    let above_threshold: Vec<_> = largest_drift
        .iter()
        .filter(|(_, _, _, _, _, _, d)| *d >= REPORT_TOLERANCE)
        .collect();

    let terminal_involved = above_threshold
        .iter()
        .filter(|(ca, ra, cb, rb, _, _, _)| is_terminal(ca, *ra) || is_terminal(cb, *rb))
        .count();
    let interior_only = above_threshold.len() - terminal_involved;

    report.push_str(&format!(
        "All {} matched pairs above threshold ({:.0e}):\n",
        above_threshold.len(),
        REPORT_TOLERANCE
    ));
    report.push_str(&format!(
        "  Terminal-residue involved: {}\n",
        terminal_involved
    ));
    report.push_str(&format!(
        "  Interior-only:             {}\n\n",
        interior_only
    ));

    for (i, (ca, ra, cb, rb, actual, reference, delta)) in above_threshold.iter().enumerate() {
        let flag = if is_terminal(ca, *ra) || is_terminal(cb, *rb) {
            " [TERMINAL]"
        } else {
            ""
        };
        report.push_str(&format!(
            "  {}. {},{} \u{2192} {},{}{}\n",
            i + 1,
            ca,
            ra,
            cb,
            rb,
            flag
        ));
        report.push_str(&format!(
            "     Actual:    {:.6}  Reference: {:.6}  \u{394}: {:.6}\n",
            actual, reference, delta
        ));
    }
    report.push('\n');
    report.push_str("Top 10 largest-drift matched contacts:\n");
    report.push('\n');

    for (i, (ca, ra, cb, rb, actual, reference, delta)) in largest_drift.iter().take(10).enumerate()
    {
        report.push_str(&format!(
            "  {}. {},{} \u{2192} {},{}\n",
            i + 1,
            ca,
            ra,
            cb,
            rb
        ));
        report.push_str(&format!("     Actual:    {:.6}\n", actual));
        report.push_str(&format!("     Reference: {:.6}\n", reference));
        report.push_str(&format!("     Delta:     {:.6} ({:.2e})\n", delta, delta));
        report.push('\n');
    }
    report.push_str(
        "================================================================================\n",
    );
    report.push('\n');

    // Print to stderr and to a file
    eprint!("{}", report);
    if let Ok(mut file) = OpenOptions::new()
        .create(true)
        .write(true)
        .open("/tmp/drift_report.txt")
    {
        let _ = file.write_all(report.as_bytes());
    }

    // Gate (decision c): after <= before, on the same metric (max|delta|), or STOP.
    // This is a regression gate against the re-measured BEFORE baseline above, not a
    // fixed tolerance -- a fixed 1e-4 tolerance was the old code's bug (objection #6:
    // the tolerance was never validated against a real measurement of the "acceptable"
    // starting point, which was actually 0.043).
    assert!(
        max_delta <= BASELINE_MAX_DELTA_BEFORE,
        "max|delta| {:.6} exceeds the BEFORE baseline {:.6} -- backlog #5244 decision c: \
         STOP and escalate with these numbers. Do not commit the artifact or drift change \
         in this state.",
        max_delta,
        BASELINE_MAX_DELTA_BEFORE
    );

    // Coverage must not regress either: adding ALA should not cause previously-matched
    // reference pairs to silently disappear, and the missing set must not grow.
    assert!(
        matched_count >= BASELINE_MATCHED_COUNT_BEFORE,
        "matched_count {} is below the BEFORE baseline {} -- coverage regressed",
        matched_count,
        BASELINE_MATCHED_COUNT_BEFORE
    );
    assert!(
        missing_from_actual <= BASELINE_MISSING_BEFORE,
        "missing_from_actual {} exceeds the BEFORE baseline {} -- more reference pairs \
         are silently absent from actual than before backlog #5244's fix",
        missing_from_actual,
        BASELINE_MISSING_BEFORE
    );
}
