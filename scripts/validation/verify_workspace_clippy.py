#!/usr/bin/env python3
"""Verify workspace clippy on all targets across non-cleanroom crates.

Pre-registered verification run for Task ID: 261102_proxide-followups (debt #2917).
Positive control: all 15 non-cleanroom crates compile on --all-targets with 0 warnings.
Negative control: clean-room crate proxide-confind produces > 0 warnings
(instrument sensitivity check).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path


def run_clippy(args: list[str], cwd: Path) -> tuple[int, str]:
  env = os.environ.copy()
  env["RUSTC_WRAPPER"] = ""
  proc = subprocess.run(
    ["cargo", "clippy", *args],
    cwd=cwd,
    env=env,
    capture_output=True,
    text=True,
  )
  output = proc.stdout + "\n" + proc.stderr
  return proc.returncode, output


def count_warnings(output: str) -> int:
  matches = re.findall(r"^(?:warning|error):", output, flags=re.MULTILINE)
  return len(matches)


def main() -> int:
  parser = argparse.ArgumentParser(description="Verify workspace clippy on all targets")
  parser.add_argument(
    "--out",
    type=Path,
    default=Path("outputs/validation/workspace_clippy_result.json"),
    help="Path to write result JSON",
  )
  cli_args = parser.parse_args()

  # Find workspace root (parent of scripts/)
  workspace_root = Path(__file__).resolve().parent.parent.parent

  # 1. Positive control: Run clippy on non-cleanroom crates with -D warnings
  cmd_args = [
    "--workspace",
    "--all-targets",
    "--exclude",
    "proxide-confind",
    "--exclude",
    "proxide-rotlib",
    "--exclude",
    "proxide_fixer",
    "--",
    "-D",
    "warnings",
  ]
  ret_code, output = run_clippy(cmd_args, workspace_root)
  non_cleanroom_warnings = count_warnings(output)
  non_cleanroom_passed = ret_code == 0 and non_cleanroom_warnings == 0

  # 2. Negative control: Run clippy on proxide-confind to ensure linter catches defects
  neg_cmd_args = ["-p", "proxide-confind", "--all-targets"]
  neg_ret_code, neg_output = run_clippy(neg_cmd_args, workspace_root)
  negative_control_warnings = count_warnings(neg_output)
  negative_control_detected_lints = negative_control_warnings > 0

  all_pass = non_cleanroom_passed and negative_control_detected_lints
  flat = {
    "all_pass": all_pass,
    "non_cleanroom_passed": non_cleanroom_passed,
    "non_cleanroom_warnings": non_cleanroom_warnings,
    "negative_control_detected_lints": negative_control_detected_lints,
    "negative_control_warnings": negative_control_warnings,
  }

  print(json.dumps(flat, indent=2))

  for path in filter(None, [os.environ.get("BTH_RESULTS_PATH"), cli_args.out]):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(flat, indent=2))

  non_cleanroom_status = "PASS" if non_cleanroom_passed else "FAIL"
  print(f"Non-cleanroom clippy: {non_cleanroom_status} (warnings: {non_cleanroom_warnings})")
  negative_control_status = "PASS" if negative_control_detected_lints else "FAIL"
  print(
    f"Negative control clippy: {negative_control_status} (warnings: {negative_control_warnings})"
  )

  if all_pass:
    print(f"Verification SUCCEEDED. Wrote result to {cli_args.out}")
    return 0
  else:
    print("Verification FAILED.")
    return 1


if __name__ == "__main__":
  raise SystemExit(main())
