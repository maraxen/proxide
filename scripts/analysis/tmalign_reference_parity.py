#!/usr/bin/env python3
"""tmalign_reference_parity: compare proxide-tmalign's tmalign_pair_serial
against the real USalign TMalign reference binary on a curated fixture set.

Phase 2 first-cut parity check (crate: proxide-tmalign, task 260729_tmalign_scaffold).
Shells out to two binaries rather than calling into the Rust crate directly,
since neither PyO3 bindings (Phase 4) nor a full CLI (Phase 3b) exist yet:

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

import typer

app = typer.Typer()

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_USALIGN_REPO = Path.home() / "repos" / "USalign"

# Curated fixture set (Phase 2 first-cut — full 6-8 pair benchmark is Phase 5).
FIXTURE_PAIRS = [
    ("PDB1.pdb", "PDB2.pdb"),  # different structures, 250 vs 166 residues
    ("PDB1.pdb", "PDB1.pdb"),  # self-alignment sanity check, TM=1.0 exactly
]


def run_reference_tmalign(usalign_repo: Path, pdb1: Path, pdb2: Path) -> dict:
    binary = usalign_repo / "TMalign"
    proc = subprocess.run(
        [str(binary), str(pdb1), str(pdb2), "-outfmt", "2"],
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
    dry_run: bool = typer.Option(False, "--dry-run"),
    usalign_repo: Path = typer.Option(DEFAULT_USALIGN_REPO, "--usalign-repo"),
    out: Path | None = typer.Option(
        None,
        "--out",
        help="Path to write the result JSON — must match the path passed to `bth run --out`, "
        "which is how bathos reads back the result to evaluate the sidecar's outcome conditions. "
        "Required unless --dry-run.",
    ),
):
    rust_binary = REPO_ROOT / "target" / "debug" / "tmalign"
    reference_binary = usalign_repo / "TMalign"

    if dry_run:
        print(json.dumps({
            "dry_run": True,
            "usalign_binary_found": reference_binary.exists(),
            "rust_binary_found": rust_binary.exists(),
            "ready": reference_binary.exists() and rust_binary.exists(),
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
    max_abs_tm_diff = 0.0
    max_abs_n_aligned_diff = 0

    for name1, name2 in FIXTURE_PAIRS:
        pdb1 = usalign_repo / name1
        pdb2 = usalign_repo / name2
        ref = run_reference_tmalign(usalign_repo, pdb1, pdb2)
        rust = run_rust_tmalign(rust_binary, pdb1, pdb2)

        diff_tm1 = abs(rust["tm_score_norm1"] - ref["tm1"])
        diff_tm2 = abs(rust["tm_score_norm2"] - ref["tm2"])
        n_aligned_diff = rust["n_aligned"] - ref["lali"]

        max_abs_tm_diff = max(max_abs_tm_diff, diff_tm1, diff_tm2)
        max_abs_n_aligned_diff = max(max_abs_n_aligned_diff, abs(n_aligned_diff))

        pair_results.append({
            "pair": f"{name1}_vs_{name2}",
            "ref_tm1": ref["tm1"],
            "ref_tm2": ref["tm2"],
            "ref_lali": ref["lali"],
            "rust_tm_score_norm1": rust["tm_score_norm1"],
            "rust_tm_score_norm2": rust["tm_score_norm2"],
            "rust_n_aligned": rust["n_aligned"],
            "diff_tm1": diff_tm1,
            "diff_tm2": diff_tm2,
            "n_aligned_diff": n_aligned_diff,
        })

    usalign_git_sha = subprocess.run(
        ["git", "-C", str(usalign_repo), "rev-parse", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip()

    results = {
        "usalign_repo_git_sha": usalign_git_sha,
        "max_abs_tm_diff": max_abs_tm_diff,
        "max_abs_n_aligned_diff": max_abs_n_aligned_diff,
        "pairs": pair_results,
    }
    print(json.dumps(results, indent=2))

    # Written to the path registered via `bth run --out` — this is how bathos
    # reads the result back to evaluate the sidecar's outcome conditions
    # (max_abs_tm_diff / max_abs_n_aligned_diff), not just stdout.
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    if out_dir := os.environ.get("BTH_OUTPUT_DIR"):
        Path(out_dir, "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    app()
