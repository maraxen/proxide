#!/usr/bin/env python3
"""tmalign_reference_parity: compare proxide-tmalign's tmalign_pair_serial
against the real USalign TMalign reference binary on a curated fixture set.

Phase 2 first-cut ran this at 2-pair scale (task 260729_tmalign_scaffold).
This is the Phase 5 full-scale run: 6 pairs spanning easy (same protein,
high identity), hard (same fold, low identity), different-length, and
unrelated-fold-negative-control categories, per
`.praxia/docs/specs/260729_proxide-tmalign-phases-2-5.md`'s Phase 5 section.

Shells out to two binaries rather than calling into the Rust crate directly,
matching the Phase 2 script's approach (kept for consistency/reuse rather
than switched to the now-available PyO3 binding, since this needs the
reference `TMalign` binary shelled out to regardless):

- The reference `TMalign` binary (built from ~/repos/USalign, must be at
  $USALIGN_REPO/TMalign or ~/repos/USalign/TMalign).
- proxide-tmalign's own minimal `tmalign` CLI
  (crates/proxide-tmalign/src/bin/tmalign.rs, built via
  `cargo build -p proxide-tmalign --bin tmalign`), which prints
  {"tm_score_norm1", "tm_score_norm2", "n_aligned", "xlen", "ylen"} as JSON.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer()

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_USALIGN_REPO = Path.home() / "repos" / "USalign"
CURATED_FIXTURES_DIR = REPO_ROOT / "crates" / "proxide-tmalign" / "tests" / "data"

# Standard tolerance for well-behaved (non-near-floor) pairs — matches
# TOLERANCE in crates/proxide-tmalign/tests/test_parity_*.rs.
STANDARD_TOLERANCE = 0.01
# Looser tolerance applied only to pairs tagged "hard" below — matches
# LOW_HOMOLOGY_TOLERANCE in test_parity_curated_set.rs. See that file's
# module doc for why: at near-random-floor TM-scores the DP has many
# near-equally-scoring alignments, so seed-selection order can legitimately
# differ from the reference without being a numerics bug.
HARD_TOLERANCE = 0.03

# name1, name2, base ("usalign_repo" | "curated"), category, is_hard
FIXTURE_PAIRS = [
    ("PDB1.pdb", "PDB2.pdb", "usalign_repo", "phase2_baseline", False),
    ("PDB1.pdb", "PDB1.pdb", "usalign_repo", "self_alignment", False),
    ("lysozyme_6lyz_A.pdb", "lysozyme_1lyz_A.pdb", "curated", "easy", False),
    ("myoglobin_1mbn_A.pdb", "hemoglobin_beta_2dhb_B.pdb", "curated", "hard", False),
    (
        "triosephosphate_1ypi_A.pdb",
        "myoglobin_1mbn_A.pdb",
        "curated",
        "different_length_negative_control",
        False,
    ),
    (
        "ubiquitin_1ubq_A.pdb",
        "lysozyme_6lyz_A.pdb",
        "curated",
        "different_length_negative_control",
        True,
    ),
]


def run_reference_tmalign(reference_binary: Path, pdb1: Path, pdb2: Path) -> dict:
    proc = subprocess.run(
        [str(reference_binary), str(pdb1), str(pdb2), "-outfmt", "2"],
        capture_output=True, text=True, check=True,
    )
    lines = [line for line in proc.stdout.splitlines() if line and not line.startswith("#")]
    fields = lines[0].split("\t")
    # header: PDBchain1 PDBchain2 TM1 TM2 RMSD ID1 ID2 IDali L1 L2 Lali
    return {
        "tm1": float(fields[2]),
        "tm2": float(fields[3]),
        "rmsd": float(fields[4]),
        "lali": int(fields[10]),
    }


def run_rust_tmalign(rust_binary: Path, pdb1: Path, pdb2: Path) -> dict:
    proc = subprocess.run(
        [str(rust_binary), str(pdb1), str(pdb2)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(proc.stdout)


@app.command()
def main(
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    usalign_repo: Annotated[Path, typer.Option("--usalign-repo")] = DEFAULT_USALIGN_REPO,
    out: Annotated[
        Path | None,
        typer.Option(
            "--out",
            help="Path to write the result JSON — must match the path passed to `bth run --out`, "
            "which is how bathos reads back the result to evaluate the sidecar's outcome "
            "conditions. Required unless --dry-run.",
        ),
    ] = None,
):
    rust_binary = REPO_ROOT / "target" / "debug" / "tmalign"
    reference_binary = usalign_repo / "TMalign"

    if dry_run:
        usalign_found = reference_binary.exists()
        rust_found = rust_binary.exists()
        fixtures_found = CURATED_FIXTURES_DIR.is_dir()
        print(json.dumps({
            "dry_run": True,
            "usalign_binary_found": usalign_found,
            "rust_binary_found": rust_found,
            "curated_fixtures_found": fixtures_found,
            "ready": usalign_found and rust_found and fixtures_found,
        }))
        return

    if out is None:
        raise SystemExit("--out is required unless --dry-run")
    if not reference_binary.exists():
        raise SystemExit(f"reference TMalign binary not found at {reference_binary}")
    if not rust_binary.exists():
        raise SystemExit(
            f"rust tmalign binary not found at {rust_binary} "
            "-- run `cargo build -p proxide-tmalign --bin tmalign` first"
        )

    pair_results = []
    max_abs_tm_diff_standard = 0.0
    max_abs_tm_diff_hard = 0.0
    max_abs_n_aligned_diff_standard = 0
    any_standard_pair_exceeds_tolerance = False
    any_hard_pair_exceeds_tolerance = False

    for name1, name2, base, category, is_hard in FIXTURE_PAIRS:
        base_dir = usalign_repo if base == "usalign_repo" else CURATED_FIXTURES_DIR
        pdb1 = base_dir / name1
        pdb2 = base_dir / name2
        ref = run_reference_tmalign(reference_binary, pdb1, pdb2)
        rust = run_rust_tmalign(rust_binary, pdb1, pdb2)

        diff_tm1 = abs(rust["tm_score_norm1"] - ref["tm1"])
        diff_tm2 = abs(rust["tm_score_norm2"] - ref["tm2"])
        n_aligned_diff = rust["n_aligned"] - ref["lali"]
        tolerance = HARD_TOLERANCE if is_hard else STANDARD_TOLERANCE
        exceeds_tolerance = diff_tm1 >= tolerance or diff_tm2 >= tolerance

        if is_hard:
            max_abs_tm_diff_hard = max(max_abs_tm_diff_hard, diff_tm1, diff_tm2)
            any_hard_pair_exceeds_tolerance = any_hard_pair_exceeds_tolerance or exceeds_tolerance
        else:
            max_abs_tm_diff_standard = max(max_abs_tm_diff_standard, diff_tm1, diff_tm2)
            max_abs_n_aligned_diff_standard = max(
                max_abs_n_aligned_diff_standard, abs(n_aligned_diff)
            )
            any_standard_pair_exceeds_tolerance = (
                any_standard_pair_exceeds_tolerance or exceeds_tolerance
            )

        pair_results.append({
            "pair": f"{name1}_vs_{name2}",
            "category": category,
            "is_hard": is_hard,
            "tolerance_used": tolerance,
            "ref_tm1": ref["tm1"],
            "ref_tm2": ref["tm2"],
            "ref_lali": ref["lali"],
            "rust_tm_score_norm1": rust["tm_score_norm1"],
            "rust_tm_score_norm2": rust["tm_score_norm2"],
            "rust_n_aligned": rust["n_aligned"],
            "diff_tm1": diff_tm1,
            "diff_tm2": diff_tm2,
            "n_aligned_diff": n_aligned_diff,
            "exceeds_tolerance": exceeds_tolerance,
        })

    # Outcome classification per the phase spec's pre-registered taxonomy:
    # pass = all pairs within tolerance; marginal = only "hard" pairs exceed
    # tolerance; fail = a non-hard ("standard") pair exceeds tolerance,
    # pointing to a systematic d0/DP-scoring/Kabsch-sign bug rather than a
    # seed-selection edge case.
    if any_standard_pair_exceeds_tolerance:
        outcome = "fail"
    elif any_hard_pair_exceeds_tolerance:
        outcome = "marginal"
    else:
        outcome = "pass"

    usalign_git_sha = subprocess.run(
        ["git", "-C", str(usalign_repo), "rev-parse", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip()

    results = {
        "usalign_repo_git_sha": usalign_git_sha,
        "outcome": outcome,
        "max_abs_tm_diff_standard": max_abs_tm_diff_standard,
        "max_abs_tm_diff_hard": max_abs_tm_diff_hard,
        "max_abs_n_aligned_diff_standard": max_abs_n_aligned_diff_standard,
        "pairs": pair_results,
    }
    print(json.dumps(results, indent=2))

    # Written to the path registered via `bth run --out` — this is how bathos
    # reads the result back to evaluate the sidecar's outcome conditions
    # (max_abs_tm_diff_standard / max_abs_n_aligned_diff_standard /
    # any_hard_pair_exceeds_tolerance), not just stdout.
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    if out_dir := os.environ.get("BTH_OUTPUT_DIR"):
        Path(out_dir, "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    app()
