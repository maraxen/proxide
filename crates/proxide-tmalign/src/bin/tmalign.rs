//! Minimal CLI driver: `tmalign <pdb1> <pdb2>` prints JSON TM-align results.
//!
//! Deliberately lean — a full-featured CLI (matching more of the reference
//! `TMalign` binary's flags/output formats) is Phase 3b work per
//! `.praxia/docs/specs/260729_proxide-tmalign-phases-2-5.md`. This exists
//! now only so the Phase 2 bathos parity experiment
//! (`scripts/analysis/tmalign_reference_parity.py`) has something to shell
//! out to without waiting on PyO3 bindings (Phase 4).

use proxide_tmalign::{load_pdb_ca_trace, pipeline::tmalign_pair_serial};
use std::env;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: tmalign <pdb1> <pdb2>");
        return ExitCode::FAILURE;
    }

    let trace1 = match load_pdb_ca_trace(&args[1]) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("failed to load {}: {e}", args[1]);
            return ExitCode::FAILURE;
        }
    };
    let trace2 = match load_pdb_ca_trace(&args[2]) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("failed to load {}: {e}", args[2]);
            return ExitCode::FAILURE;
        }
    };

    let result = match tmalign_pair_serial(&trace1.coords, &trace2.coords) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("tmalign_pair_serial failed: {e}");
            return ExitCode::FAILURE;
        }
    };

    println!(
        "{{\"tm_score_norm1\": {}, \"tm_score_norm2\": {}, \"n_aligned\": {}, \"xlen\": {}, \"ylen\": {}}}",
        result.tm_score_norm1,
        result.tm_score_norm2,
        result.n_aligned,
        trace1.len(),
        trace2.len(),
    );
    ExitCode::SUCCESS
}
