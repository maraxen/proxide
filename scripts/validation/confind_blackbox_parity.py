#!/usr/bin/env python3
"""
Black-box parity testing harness comparing an upstream reference confind binary
(executed in an external scratch directory) against clean-room proxide-confind
on standard benchmark structures (1CRN, 1UBQ, a multi-chain complex, and a
segmented chain break).

Pre-registered under bathos (scripts/validation/confind_blackbox_parity.bth.toml).
Task ID: 261102_proxide-mosaist-cleanroom

Required locations have no built-in default path. Pass a flag or set the matching
environment variable. An explicit flag wins when both are present:

  --reference-bin     CONFIND_PARITY_REFERENCE_BIN      upstream reference binary
  --reference-rotlib  CONFIND_PARITY_REFERENCE_ROTLIB   upstream rotamer library
  --clean-bin         CONFIND_PARITY_CLEAN_BIN          clean-room confind binary
  --clean-rotlib      CONFIND_PARITY_CLEAN_ROTLIB       clean-room rotamer library
  --pdb-dir           CONFIND_PARITY_PDB_DIR            directory of input PDBs
  --scratch           CONFIND_PARITY_SCRATCH            scratch directory

--pdb-dir must contain 1CRN.pdb, ubiquitin_1ubq_A.pdb, f2c006r13.pdb, and
chain_break.pdb. A missing required location is an error that names the argument.
"""

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ENV_REFERENCE_BIN = "CONFIND_PARITY_REFERENCE_BIN"
ENV_REFERENCE_ROTLIB = "CONFIND_PARITY_REFERENCE_ROTLIB"
ENV_CLEAN_BIN = "CONFIND_PARITY_CLEAN_BIN"
ENV_CLEAN_ROTLIB = "CONFIND_PARITY_CLEAN_ROTLIB"
ENV_PDB_DIR = "CONFIND_PARITY_PDB_DIR"
ENV_SCRATCH = "CONFIND_PARITY_SCRATCH"

PDB_CRN_NAME = "1CRN.pdb"
PDB_UBQ_NAME = "ubiquitin_1ubq_A.pdb"
PDB_MULTICHAIN_NAME = "f2c006r13.pdb"
PDB_BREAK_NAME = "chain_break.pdb"

# Filled by main() from flags or the environment variables above. No default path.
SCRATCH = None
UPSTREAM_BIN = None
UPSTREAM_RLIB = None
CLEAN_BIN = None
CLEAN_RLIB = None
CRN_PDB = None
UBQ_PDB = None
MULTICHAIN_PDB = None
BREAK_PDB = None


def _require_finite(value, label):
  if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
    raise ValueError(f"non-finite value for {label}: {value!r}")
  return float(value)


def _parse_finite(text, label):
  try:
    value = float(text)
  except ValueError as exc:
    raise ValueError(f"non-numeric value for {label}: {text!r}") from exc
  if not math.isfinite(value):
    raise ValueError(f"non-finite value for {label}: {text!r}")
  return value


def _require_file(path, flag, env_var):
  if path is None or not Path(path).is_file():
    shown = "(unset)" if path is None else path
    raise FileNotFoundError(f"missing required location for {flag} (or {env_var}): {shown}")
  return Path(path)


def _require_dir(path, flag, env_var):
  if path is None or not Path(path).is_dir():
    shown = "(unset)" if path is None else path
    raise FileNotFoundError(f"missing required location for {flag} (or {env_var}): {shown}")
  return Path(path)


def parse_confind_output(path: Path, is_upstream: bool = False):
  """
  Parse confind output format:
  contact\\t<chain_a>,<res_a>\\t<chain_b>,<res_b>\\t<cd>\\t<name_a>\\t<name_b>
  crwdnes/crowdedness\\t<chain>,<res>\\t<cr>\\t...
  freedom\\t<chain>,<res>\\t<fr>\\t...
  """
  contacts = {}
  crowdedness = {}
  freedom = {}
  dihedrals = {}

  with open(path, encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      parts = line.split("\t")
      tag = parts[0]
      if tag == "contact":
        # parts: contact, id_a, id_b, val, name_a, name_b
        p_a, p_b = parts[1], parts[2]
        val = _parse_finite(parts[3], f"contact {p_a} {p_b}")
        pair = tuple(sorted([p_a, p_b]))
        contacts[pair] = val
      elif tag in ("crwdnes", "crowdedness"):
        res = parts[1]
        val = _parse_finite(parts[2], f"crowdedness {res}")
        crowdedness[res] = val
        if is_upstream and len(parts) >= 6:
          # crwdnes, res, val, phi, psi, omega, resname
          phi = _parse_finite(parts[3], f"phi {res}")
          psi = _parse_finite(parts[4], f"psi {res}")
          omg = _parse_finite(parts[5], f"omega {res}")
          dihedrals[res] = (phi, psi, omg)
      elif tag == "freedom":
        res = parts[1]
        val = _parse_finite(parts[2], f"freedom {res}")
        freedom[res] = val
        if is_upstream and len(parts) >= 6:
          phi = _parse_finite(parts[3], f"phi {res}")
          psi = _parse_finite(parts[4], f"psi {res}")
          omg = _parse_finite(parts[5], f"omega {res}")
          dihedrals[res] = (phi, psi, omg)

  return contacts, crowdedness, freedom, dihedrals


def _upstream_cache_reusable(path: Path, binary: Path) -> bool:
  """Reuse a scratch output only when it is non-empty, parseable, and newer than the binary."""
  if not path.is_file() or path.stat().st_size == 0:
    return False
  if path.stat().st_mtime <= binary.stat().st_mtime:
    return False
  try:
    contacts, crowdedness, freedom, _dihedrals = parse_confind_output(path, is_upstream=True)
  except (OSError, ValueError, IndexError, UnicodeError):
    return False
  return bool(contacts or crowdedness or freedom)


def run_benchmark(pdb_path: Path, tag: str, extra_flags=None):
  """Run upstream and clean confind, returning parsed outputs."""
  scratch = _require_dir(SCRATCH, "--scratch", ENV_SCRATCH)
  upstream_bin = _require_file(UPSTREAM_BIN, "--reference-bin", ENV_REFERENCE_BIN)
  upstream_rlib = _require_file(UPSTREAM_RLIB, "--reference-rotlib", ENV_REFERENCE_ROTLIB)
  clean_bin = _require_file(CLEAN_BIN, "--clean-bin", ENV_CLEAN_BIN)
  clean_rlib = _require_file(CLEAN_RLIB, "--clean-rotlib", ENV_CLEAN_ROTLIB)
  pdb_path = _require_file(pdb_path, "--pdb-dir", ENV_PDB_DIR)

  up_out = scratch / f"{tag}_upstream.out"
  suffix = "_" + "_".join(flag.lstrip("-") for flag in extra_flags) if extra_flags else ""
  clean_out = scratch / f"{tag}_clean{suffix}.out"

  if not _upstream_cache_reusable(up_out, upstream_bin):
    if up_out.exists():
      up_out.unlink()
    subprocess.run(
      [
        str(upstream_bin),
        "--p",
        str(pdb_path),
        "--rLib",
        str(upstream_rlib),
        "--o",
        str(up_out),
        "--pp",
        "--omg",
      ],
      check=True,
      cwd=str(scratch),
      stdout=subprocess.DEVNULL,
      stderr=subprocess.PIPE,
    )

  cmd = [
    str(clean_bin),
    "--p",
    str(pdb_path),
    "--rLib",
    str(clean_rlib),
    "--o",
    str(clean_out),
    "--cdcut",
    "0.0",
    "--incut",
    "0.0",
  ]
  if extra_flags:
    cmd.extend(extra_flags)
  subprocess.run(
    cmd,
    check=True,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.PIPE,
  )

  up_c, up_cr, up_fr, up_dih = parse_confind_output(up_out, is_upstream=True)
  cl_c, cl_cr, cl_fr, _ = parse_confind_output(clean_out, is_upstream=False)
  return up_c, up_cr, up_fr, up_dih, cl_c, cl_cr, cl_fr


def compute_contact_deltas(up_c, cl_c):
  if len(up_c) == 0 or len(cl_c) == 0:
    raise ValueError(f"empty contact set: reference={len(up_c)} clean={len(cl_c)}")
  only_ref = set(up_c) - set(cl_c)
  only_clean = set(cl_c) - set(up_c)
  if only_ref or only_clean:
    raise ValueError(
      "contact pair present on one side only: "
      f"reference-only={sorted(only_ref)} clean-only={sorted(only_clean)}"
    )
  for pair, value in up_c.items():
    _require_finite(value, f"reference contact {pair}")
  for pair, value in cl_c.items():
    _require_finite(value, f"clean contact {pair}")
  all_pairs = set(up_c.keys()) | set(cl_c.keys())
  deltas = [abs(up_c[pair] - cl_c[pair]) for pair in all_pairs]
  deltas.sort()
  n = len(deltas)
  median_val = deltas[n // 2]
  mean_val = sum(deltas) / n
  max_val = max(deltas)
  pct_1e4 = sum(1 for delta in deltas if delta <= 1e-4) / n
  pct_005 = sum(1 for delta in deltas if delta <= 0.05) / n
  common = set(up_c.keys()) & set(cl_c.keys())
  return median_val, mean_val, max_val, pct_1e4, pct_005, len(all_pairs), len(common)


def compute_metric_deltas(up_m, cl_m):
  up_keys = set(up_m)
  cl_keys = set(cl_m)
  if up_keys != cl_keys:
    raise ValueError(
      "unequal residue-key sets between clean and reference: "
      f"reference-only={sorted(up_keys - cl_keys)} clean-only={sorted(cl_keys - up_keys)}"
    )
  if not up_keys:
    raise ValueError("empty residue-key intersection")
  for key in up_keys:
    _require_finite(up_m[key], f"reference metric {key}")
    _require_finite(cl_m[key], f"clean metric {key}")
  deltas = sorted(abs(up_m[key] - cl_m[key]) for key in up_keys)
  n = len(deltas)
  return deltas[n // 2], sum(deltas) / n, max(deltas)


def assess_omega_sign(pdb_path: Path, up_dih):
  """
  Compute clean IUPAC omega and compare to upstream omega (Debt #1903).
  Standard IUPAC convention: torsion(CA_{i-1}, C_{i-1}, N_i, CA_i).
  """
  atoms = {}
  res_list = []
  with open(pdb_path, encoding="utf-8") as f:
    for line in f:
      if line.startswith("ATOM  "):
        name = line[12:16].strip()
        res_name = line[17:20].strip()
        chain = line[21]
        res_id = int(line[22:26].strip())
        xyz = (
          _parse_finite(line[30:38], "pdb x"),
          _parse_finite(line[38:46], "pdb y"),
          _parse_finite(line[46:54], "pdb z"),
        )
        key = (chain, res_id)
        if key not in atoms:
          atoms[key] = {}
          res_list.append((chain, res_id, res_name))
        if name in ("N", "CA", "C", "O") and name not in atoms[key]:
          atoms[key][name] = xyz

  def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

  def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

  def cross(a, b):
    return (
      a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0],
    )

  def dihedral(p1, p2, p3, p4):
    b1 = sub(p2, p1)
    b2 = sub(p3, p2)
    b3 = sub(p4, p3)
    n1 = cross(b1, b2)
    n2 = cross(b2, b3)
    b2_len = math.sqrt(dot(b2, b2))
    b2_norm = (b2[0] / b2_len, b2[1] / b2_len, b2[2] / b2_len)
    m1 = cross(n1, b2_norm)
    return math.atan2(dot(m1, n2), dot(n1, n2))

  same_sign = 0
  opp_sign = 0
  max_delta = 0.0

  for i in range(1, len(res_list)):
    prev_k = res_list[i - 1][:2]
    curr_k = res_list[i][:2]
    if (
      "CA" in atoms[prev_k]
      and "C" in atoms[prev_k]
      and "N" in atoms[curr_k]
      and "CA" in atoms[curr_k]
    ):
      # Cleanroom IUPAC omega: -dihedral angle in degrees
      cl_omg = -math.degrees(
        dihedral(
          atoms[prev_k]["CA"],
          atoms[prev_k]["C"],
          atoms[curr_k]["N"],
          atoms[curr_k]["CA"],
        )
      )
      up_str_k = f"{curr_k[0]},{curr_k[1]}"
      if up_str_k in up_dih:
        up_omg = up_dih[up_str_k][2]
        if not math.isfinite(up_omg) or not math.isfinite(cl_omg):
          raise ValueError(
            f"non-finite omega for {up_str_k}: clean={cl_omg!r} reference={up_omg!r}"
          )
        if abs(up_omg - 999.0) > 1.0:
          diff = abs(cl_omg - up_omg)
          max_delta = max(max_delta, diff)
          if (cl_omg * up_omg) >= 0:
            same_sign += 1
          else:
            opp_sign += 1

  if same_sign == 0:
    raise ValueError(
      "omega_same_sign_count == 0 "
      f"(omega_opposite_sign_count={opp_sign}); refusing an empty omega comparison"
    )
  return same_sign, opp_sign, max_delta


def assess_chain_semantics():
  """
  Evaluate Debt #1904 (Chain Semantics):
  Tests chain_break.pdb where a 10 A CA gap exists between residue 2 and 3.
  Upstream erroneously connects across the break, emitting a contact of ~0.000436.
  Cleanroom segments the chain, correctly eliminating this false contact.
  """
  up_c, _, _, _, cl_c, _, _ = run_benchmark(BREAK_PDB, "break")
  pair_2_3 = ("A,2", "A,3")
  up_val = up_c[pair_2_3] if pair_2_3 in up_c else None
  cl_val = cl_c[pair_2_3] if pair_2_3 in cl_c else None
  if up_val is not None:
    _require_finite(up_val, "reference chain-break contact")
  if cl_val is not None:
    _require_finite(cl_val, "clean chain-break contact")
  # Absence is a missing pair, reported as elimination. It is not a numeric zero
  # substituted into a contact delta. The boolean matches an explicit measured 0.
  prevented = (up_val is not None and up_val > 1e-4) and (cl_val is None or cl_val == 0.0)
  return prevented, up_val, cl_val


def _parser():
  parser = argparse.ArgumentParser(description="ConFind black-box parity validation")
  parser.add_argument(
    "--out",
    default="outputs/parity/confind_parity_results.json",
    help="JSON result path",
  )
  parser.add_argument("--reference-bin", default=None, help=f"env: {ENV_REFERENCE_BIN}")
  parser.add_argument("--reference-rotlib", default=None, help=f"env: {ENV_REFERENCE_ROTLIB}")
  parser.add_argument("--clean-bin", default=None, help=f"env: {ENV_CLEAN_BIN}")
  parser.add_argument("--clean-rotlib", default=None, help=f"env: {ENV_CLEAN_ROTLIB}")
  parser.add_argument("--pdb-dir", default=None, help=f"env: {ENV_PDB_DIR}")
  parser.add_argument("--scratch", default=None, help=f"env: {ENV_SCRATCH}")
  return parser


def _resolve(parser, flag_value, flag, env_var, kind):
  raw = flag_value if flag_value else os.environ.get(env_var)
  if raw is None or str(raw).strip() == "":
    parser.error(f"missing required location: {flag} (or {env_var})")
  path = Path(raw)
  if kind == "file" and not path.is_file():
    parser.error(f"missing required location for {flag} (or {env_var}): {path}")
  if kind == "dir" and not path.is_dir():
    parser.error(f"missing required location for {flag} (or {env_var}): {path}")
  return path


def _pdb_file(parser, pdb_dir, filename):
  path = pdb_dir / filename
  if not path.is_file():
    parser.error(f"missing required location for --pdb-dir (or {ENV_PDB_DIR}): {path}")
  return path


def exit_code_for_outcome(outcome):
  if outcome == "pass":
    return 0
  return 1


def main():
  global SCRATCH, UPSTREAM_BIN, UPSTREAM_RLIB, CLEAN_BIN, CLEAN_RLIB
  global CRN_PDB, UBQ_PDB, MULTICHAIN_PDB, BREAK_PDB

  parser = _parser()
  args = parser.parse_args()
  UPSTREAM_BIN = _resolve(parser, args.reference_bin, "--reference-bin", ENV_REFERENCE_BIN, "file")
  UPSTREAM_RLIB = _resolve(
    parser, args.reference_rotlib, "--reference-rotlib", ENV_REFERENCE_ROTLIB, "file"
  )
  CLEAN_BIN = _resolve(parser, args.clean_bin, "--clean-bin", ENV_CLEAN_BIN, "file")
  CLEAN_RLIB = _resolve(parser, args.clean_rotlib, "--clean-rotlib", ENV_CLEAN_ROTLIB, "file")
  pdb_dir = _resolve(parser, args.pdb_dir, "--pdb-dir", ENV_PDB_DIR, "dir")
  SCRATCH = _resolve(parser, args.scratch, "--scratch", ENV_SCRATCH, "dir")
  CRN_PDB = _pdb_file(parser, pdb_dir, PDB_CRN_NAME)
  UBQ_PDB = _pdb_file(parser, pdb_dir, PDB_UBQ_NAME)
  MULTICHAIN_PDB = _pdb_file(parser, pdb_dir, PDB_MULTICHAIN_NAME)
  BREAK_PDB = _pdb_file(parser, pdb_dir, PDB_BREAK_NAME)

  parity_flags = ["--cbclash", "ala-only", "--crowdmode", "pruned-count-fraction"]

  print("Running Benchmark 1 (parity flags: ala-only + pruned-count-fraction): 1CRN...")
  up_c_crn, up_cr_crn, up_fr_crn, up_dih_crn, cl_c_crn, cl_cr_crn, _cl_fr_crn = run_benchmark(
    CRN_PDB, "1crn", parity_flags
  )
  med_cd_crn, mean_cd_crn, max_cd_crn, pct_1e4_crn, pct_005_crn, _, _ = compute_contact_deltas(
    up_c_crn, cl_c_crn
  )
  med_cr_crn, _, _ = compute_metric_deltas(up_cr_crn, cl_cr_crn)
  med_fr_crn, _, _ = compute_metric_deltas(up_fr_crn, _cl_fr_crn)

  print("Running Benchmark 1b (default flags: all-residues + probability-weighted): 1CRN...")
  _, _, _, _, cl_c_crn_def, cl_cr_crn_def, _ = run_benchmark(CRN_PDB, "1crn", None)
  med_cd_crn_def, _, _, _, _, _, _ = compute_contact_deltas(up_c_crn, cl_c_crn_def)
  med_cr_crn_def, _, _ = compute_metric_deltas(up_cr_crn, cl_cr_crn_def)

  print("Running Benchmark 2 (parity flags: ala-only + pruned-count-fraction): 1UBQ...")
  up_c_ubq, up_cr_ubq, up_fr_ubq, _up_dih_ubq, cl_c_ubq, cl_cr_ubq, cl_fr_ubq = run_benchmark(
    UBQ_PDB, "1ubq", parity_flags
  )
  med_cd_ubq, mean_cd_ubq, max_cd_ubq, pct_1e4_ubq, pct_005_ubq, _, _ = compute_contact_deltas(
    up_c_ubq, cl_c_ubq
  )
  med_cr_ubq, _, _ = compute_metric_deltas(up_cr_ubq, cl_cr_ubq)
  med_fr_ubq, _, _ = compute_metric_deltas(up_fr_ubq, cl_fr_ubq)

  print("Running Benchmark 2b (default flags: all-residues + probability-weighted): 1UBQ...")
  _, _, _, _, cl_c_ubq_def, cl_cr_ubq_def, _ = run_benchmark(UBQ_PDB, "1ubq", None)
  med_cd_ubq_def, _, _, _, _, _, _ = compute_contact_deltas(up_c_ubq, cl_c_ubq_def)
  med_cr_ubq_def, _, _ = compute_metric_deltas(up_cr_ubq, cl_cr_ubq_def)

  print("Running Benchmark 3: Multi-chain complex (f2c006r13.pdb)...")
  up_c_mc, _, _, _, cl_c_mc, _, _ = run_benchmark(MULTICHAIN_PDB, "multichain", parity_flags)
  med_cd_mc, _mean_cd_mc, _max_cd_mc, pct_1e4_mc, pct_005_mc, n_pairs_mc, com_pairs_mc = (
    compute_contact_deltas(up_c_mc, cl_c_mc)
  )

  print("Assessing Debt #1903 (Omega Dihedral Sign)...")
  same_sign, opp_sign, max_omg_delta = assess_omega_sign(CRN_PDB, up_dih_crn)

  print("Assessing Debt #1904 (Chain Semantics & Breaks)...")
  prevented, _, _ = assess_chain_semantics()

  # Check that both parity-policy and default-policy runs produce valid outputs
  default_options_valid = (
    med_cd_crn_def <= 0.0001
    and med_cd_ubq_def <= 0.0001
    and len(cl_cr_crn_def) == len(cl_cr_crn)
    and len(cl_cr_ubq_def) == len(cl_cr_ubq)
  )

  passed = (
    med_cd_crn <= 0.0001
    and med_cd_ubq <= 0.0001
    and opp_sign == 0
    and prevented
    and default_options_valid
  )
  if passed:
    outcome_label = "pass"
  elif med_cd_crn <= 0.0002 and med_cd_ubq <= 0.0002:
    outcome_label = "marginal"
  else:
    outcome_label = "fail"

  result = {
    "median_contact_delta_1crn": round(float(med_cd_crn), 6),
    "mean_contact_delta_1crn": round(float(mean_cd_crn), 6),
    "max_contact_delta_1crn": round(float(max_cd_crn), 6),
    "pct_within_1e4_1crn": round(float(pct_1e4_crn), 4),
    "pct_within_005_1crn": round(float(pct_005_crn), 4),
    "median_crowdedness_delta_1crn": round(float(med_cr_crn), 6),
    "median_freedom_delta_1crn": round(float(med_fr_crn), 6),
    "median_contact_delta_1crn_default": round(float(med_cd_crn_def), 6),
    "median_crowdedness_delta_1crn_default": round(float(med_cr_crn_def), 6),
    "median_contact_delta_1ubq": round(float(med_cd_ubq), 6),
    "mean_contact_delta_1ubq": round(float(mean_cd_ubq), 6),
    "max_contact_delta_1ubq": round(float(max_cd_ubq), 6),
    "pct_within_1e4_1ubq": round(float(pct_1e4_ubq), 4),
    "pct_within_005_1ubq": round(float(pct_005_ubq), 4),
    "median_crowdedness_delta_1ubq": round(float(med_cr_ubq), 6),
    "median_freedom_delta_1ubq": round(float(med_fr_ubq), 6),
    "median_contact_delta_1ubq_default": round(float(med_cd_ubq_def), 6),
    "median_crowdedness_delta_1ubq_default": round(float(med_cr_ubq_def), 6),
    "multichain_contacts_count": int(n_pairs_mc),
    "multichain_common_pairs": int(com_pairs_mc),
    "median_contact_delta_multichain": round(float(med_cd_mc), 6),
    "pct_within_1e4_multichain": round(float(pct_1e4_mc), 4),
    "pct_within_005_multichain": round(float(pct_005_mc), 4),
    "omega_same_sign_count": int(same_sign),
    "omega_opposite_sign_count": int(opp_sign),
    "max_omega_delta_deg": round(float(max_omg_delta), 6),
    "chain_break_prevented_spurious_contacts": bool(prevented),
    "default_options_valid": bool(default_options_valid),
    "outcome": outcome_label,
  }

  out_path = Path(args.out)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  with open(out_path, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)

  bth_results_path = os.environ.get("BTH_RESULTS_PATH")
  if bth_results_path:
    with open(bth_results_path, "w", encoding="utf-8") as f:
      json.dump(result, f, indent=2)

  print(f"\nVerification Complete! Verdict: {outcome_label.upper()}")
  print(f"Summary: {json.dumps(result, indent=2)}")
  return exit_code_for_outcome(outcome_label)


if __name__ == "__main__":
  sys.exit(main())
