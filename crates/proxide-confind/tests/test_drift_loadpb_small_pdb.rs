mod common;

use common::load_real_backbone;
use proxide_confind::ConFind;
use std::collections::HashMap;
use std::sync::Arc;
use std::fs::OpenOptions;
use std::io::Write;

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

const TOLERANCE: f64 = 1e-4;

fn all_res(cf: &ConFind) -> Vec<proxide_confind::ResidueIndex> {
    (0..cf.n_residues() as u32).map(proxide_confind::ResidueIndex).collect()
}

#[test]
#[ignore]
fn measure_loadpb_drift_vs_master() {
    let pb_path = std::env::var("PROXIDE_ROTLIB_PB")
        .unwrap_or_else(|_| "/home/marielle/projects/proxide/data/rotlibs/proxide-rotlib-bbdep2010.pb.zst".to_string());

    let pb_path_obj = std::path::Path::new(&pb_path);
    if !pb_path_obj.exists() {
        eprintln!("SKIP: protobuf library not found at {}", pb_path);
        return;
    }

    let bb = match load_real_backbone() {
        Some(v) => v,
        None => {
            eprintln!("SKIP: small.pdb backbone not found");
            return;
        }
    };

    let rlib = match proxide_rotlib::RotamerLibrary::load_pb(pb_path_obj) {
        Ok(v) => Arc::new(v),
        Err(e) => {
            panic!("Failed to load protobuf library from {}: {}", pb_path, e);
        }
    };

    let cf = ConFind::new(rlib, bb.clone(), false);
    let contact_list = cf.contacts(&all_res(&cf), 0.0).expect("contacts should succeed");

    assert!(!contact_list.pairs.is_empty(), "contact list must not be empty");

    // Build expected map: canonical key (chain_a, res_a, chain_b, res_b) → cd
    let expected: HashMap<(String, i32, String, i32), f64> = REF_CONTACTS.iter()
        .map(|&(ca, ra, cb, rb, cd)| ((ca.to_string(), ra, cb.to_string(), rb), cd))
        .collect();

    // Collect deltas
    let mut deltas: Vec<f64> = Vec::new();
    let mut matched_count = 0;
    let mut missing_from_actual = 0;
    let mut unexpected_in_actual = 0;
    let mut exceeding_tolerance = 0;
    let mut largest_drift: Vec<(String, i32, String, i32, f64, f64, f64)> = Vec::new();

    // Check all actual pairs against expected
    for (&(ri_a, ri_b), &actual) in contact_list.pairs.iter().zip(&contact_list.degrees) {
        let id_a = cf.residue_id(ri_a);
        let id_b = cf.residue_id(ri_b);
        let key = (id_a.chain_id.clone(), id_a.res_id, id_b.chain_id.clone(), id_b.res_id);

        if let Some(&reference) = expected.get(&key) {
            let delta = (actual - reference).abs();
            deltas.push(delta);
            matched_count += 1;

            if delta >= TOLERANCE {
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

    // Check for missing pairs
    let actual_pairs: std::collections::HashSet<(String, i32, String, i32)> = contact_list.pairs
        .iter()
        .map(|&(ri_a, ri_b)| {
            let id_a = cf.residue_id(ri_a);
            let id_b = cf.residue_id(ri_b);
            (id_a.chain_id.clone(), id_a.res_id, id_b.chain_id.clone(), id_b.res_id)
        })
        .collect();

    for &(ca, ra, cb, rb, _) in REF_CONTACTS {
        let key = (ca.to_string(), ra, cb.to_string(), rb);
        if !actual_pairs.contains(&key) {
            missing_from_actual += 1;
        }
    }

    // Sort largest_drift by delta descending
    largest_drift.sort_by(|a, b| b.6.partial_cmp(&a.6).unwrap_or(std::cmp::Ordering::Equal));

    // Compute statistics
    let max_delta = deltas.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mean_delta = if !deltas.is_empty() {
        deltas.iter().sum::<f64>() / deltas.len() as f64
    } else {
        0.0
    };
    let mut sorted_deltas = deltas.clone();
    sorted_deltas.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_delta = if sorted_deltas.is_empty() {
        0.0
    } else if sorted_deltas.len() % 2 == 0 {
        (sorted_deltas[sorted_deltas.len() / 2 - 1] + sorted_deltas[sorted_deltas.len() / 2]) / 2.0
    } else {
        sorted_deltas[sorted_deltas.len() / 2]
    };

    // Enforce drift threshold: max_delta must not exceed TOLERANCE
    assert!(
        max_delta <= TOLERANCE,
        "max|delta| {:.6} exceeds threshold {:.0e}",
        max_delta,
        TOLERANCE
    );

    // Build drift report
    let mut report = String::new();
    report.push_str("\n");
    report.push_str("================================================================================\n");
    report.push_str("DRIFT REPORT: load_pb vs. MASTER rotlib.bin\n");
    report.push_str("================================================================================\n");
    report.push_str("\n");
    report.push_str(&format!("Matched pairs:              {}\n", matched_count));
    report.push_str(&format!("Missing from actual (PB):   {}\n", missing_from_actual));
    report.push_str(&format!("Unexpected in actual (PB):  {}\n", unexpected_in_actual));
    report.push_str("\n");
    report.push_str(&format!("Drift statistics (across {} matched pairs):\n", matched_count));
    report.push_str(&format!("  Max |Δ|:                  {:.6}\n", max_delta));
    report.push_str(&format!("  Mean |Δ|:                 {:.6}\n", mean_delta));
    report.push_str(&format!("  Median |Δ|:               {:.6}\n", median_delta));
    report.push_str(&format!("  Count exceeding 5e-4:     {} ({}%)\n",
        exceeding_tolerance,
        if matched_count > 0 { (exceeding_tolerance * 100) / matched_count } else { 0 }
    ));
    report.push_str("\n");
    // Classify terminal residues: residue 1 (N-terminal) or max res_id per chain (C-terminal).
    // From REF_CONTACTS the chains are A and B, each with res_ids 1..7.
    let terminal_res_ids: std::collections::HashSet<(String, i32)> = {
        let mut m = std::collections::HashMap::<String, (i32, i32)>::new();
        for &(ca, ra, cb, rb, _) in REF_CONTACTS {
            let e = m.entry(ca.to_string()).or_insert((i32::MAX, i32::MIN));
            e.0 = e.0.min(ra); e.1 = e.1.max(ra);
            let e = m.entry(cb.to_string()).or_insert((i32::MAX, i32::MIN));
            e.0 = e.0.min(rb); e.1 = e.1.max(rb);
        }
        m.iter().flat_map(|(ch, &(mn, mx))| {
            [ch.clone(), ch.clone()].into_iter().zip([mn, mx])
        }).collect()
    };

    let is_terminal = |chain: &str, res: i32| terminal_res_ids.contains(&(chain.to_string(), res));

    let above_threshold: Vec<_> = largest_drift.iter()
        .filter(|(_, _, _, _, _, _, d)| *d >= TOLERANCE)
        .collect();

    let terminal_involved = above_threshold.iter()
        .filter(|(ca, ra, cb, rb, _, _, _)| is_terminal(ca, *ra) || is_terminal(cb, *rb))
        .count();
    let interior_only = above_threshold.len() - terminal_involved;

    report.push_str(&format!("All {} pairs above threshold ({:.0e}):\n", above_threshold.len(), TOLERANCE));
    report.push_str(&format!("  Terminal-residue involved: {}\n", terminal_involved));
    report.push_str(&format!("  Interior-only:             {}\n\n", interior_only));

    for (i, (ca, ra, cb, rb, actual, reference, delta)) in above_threshold.iter().enumerate() {
        let flag = if is_terminal(ca, *ra) || is_terminal(cb, *rb) { " [TERMINAL]" } else { "" };
        report.push_str(&format!("  {}. {},{} → {},{}{}\n", i + 1, ca, ra, cb, rb, flag));
        report.push_str(&format!("     Actual:    {:.6}  Reference: {:.6}  Δ: {:.6}\n", actual, reference, delta));
    }
    report.push_str("\n");
    report.push_str("Top 10 largest-drift contacts:\n");
    report.push_str("\n");

    for (i, (ca, ra, cb, rb, actual, reference, delta)) in largest_drift.iter().take(10).enumerate() {
        report.push_str(&format!("  {}. {},{} → {},{}\n", i + 1, ca, ra, cb, rb));
        report.push_str(&format!("     Actual:    {:.6}\n", actual));
        report.push_str(&format!("     Reference: {:.6}\n", reference));
        report.push_str(&format!("     Delta:     {:.6} ({:.2e})\n", delta, delta));
        report.push_str("\n");
    }
    report.push_str("================================================================================\n");
    report.push_str("\n");

    // Print to stderr and to a file
    eprint!("{}", report);
    if let Ok(mut file) = OpenOptions::new().create(true).write(true).open("/tmp/drift_report.txt") {
        let _ = file.write_all(report.as_bytes());
    }
}
