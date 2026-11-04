#!/usr/bin/env python3
"""
Black-box parity testing harness comparing an upstream reference confind binary
(executed in an external scratch directory) against clean-room proxide-confind
on standard benchmark structures (1CRN, 1UBQ, a multi-chain complex, and a
segmented chain break).

Pre-registered under bathos:
- v1: scripts/validation/confind_blackbox_parity.bth.toml (run ea7e91ff)
- v2: scripts/validation/confind_blackbox_parity_v2.bth.toml (debt #3217)

Required locations have no built-in default path. Pass a flag or set the matching
environment variable. An explicit flag wins when both are present:

  --out               CONFIND_PARITY_OUT                output result JSON path (not /tmp)
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
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ENV_OUT = "CONFIND_PARITY_OUT"
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

PARITY_ARM_LABEL = "mosaist_parity_ala_only_pruned_count_fraction"
DEFAULT_ARM_LABEL = "spec_2_4_all_residues_probability_weighted"

PARITY_FLAGS = ("--cbclash", "ala-only", "--crowdmode", "pruned-count-fraction")
# Factorial completion of the 2x2 (cbclash x crowdmode) design; clean-only arms.
ALL_COUNT_FLAGS = ("--cbclash", "all-residues", "--crowdmode", "pruned-count-fraction")
ALA_WEIGHTED_FLAGS = ("--cbclash", "ala-only", "--crowdmode", "probability-weighted")
# What a run with no flags must report in its header (spec §2.4 default; params.rs:11).
EXPECTED_DEFAULT_HEADER = {
  "cbclash": "all-residues",
  "crowdmode": "probability-weighted",
  "dclash": "2.0",
}
# Pre-registered d_clash sweep (Å), chosen from backbone geometry only: CB(i) to
# N(i+1) is typically ~3.3-3.4 Å, so an all-residues CB clash is expected to become
# reachable within this range. No crowdedness output was consulted.
DCLASH_SWEEP = ("2.0", "2.5", "3.0", "3.5")
SWEEP_POLICIES = ("all-residues", "ala-only")
# parity (1crn, 1ubq, multichain) + default (1crn, 1ubq) + chain_break (default)
# + factorial (2 arms x 1crn, 1ubq) + sweep (2 policies x 4 d_clash on 1crn)
N_CLEAN_RUNS = 3 + 2 + 1 + 4 + len(SWEEP_POLICIES) * len(DCLASH_SWEEP)

# Every binary invocation is its own process with this limit, so a hung binary fails
# loudly. An interrupted output has no completion stamp and is regenerated on resume.
UNIT_TIMEOUT_S = 900

# What each output was, for the <out>.outputs.json ledger: fresh or reused, and from what.
_OUTPUT_LEDGER: dict = {}

# Clean outputs produced this process, for the header check: (path, extra_flags).
_CLEAN_RUNS: list = []


def _dclash_tag(x: str) -> str:
  return x.replace(".", "p")


# ---------------------------------------------------------------------------
# Pre-registered gate terms. The v2 sidecar's [outcomes] conditions are rendered
# from these lists (render_conditions) and a test asserts they match the sidecar
# text exactly, so evaluate_outcome restates the sidecar term for term.
# Each term is (field, op, value) with op in {"=", "<=", ">="}.
# ---------------------------------------------------------------------------

_STRUCTS = ("1crn", "1ubq")

INVARIANT_TERMS = [
  ("reference_omega_opposite_sign_count", "=", 0),
  ("chain_break_prevented_spurious_contacts", "=", True),
  ("arm_header_mismatch_count", "=", 0),
  ("n_headers_checked", "=", N_CLEAN_RUNS),
]
for _s in _STRUCTS:
  for _m in ("count", "weighted"):
    INVARIANT_TERMS += [
      (f"cbclash_mono_violations_{_s}_{_m}", "=", 0),
      (f"cbclash_mono_n_compared_{_s}_{_m}", ">=", 1),
    ]
for _s in _STRUCTS:
  for _p in ("all", "ala"):
    INVARIANT_TERMS += [
      (f"crowdmode_contact_max_micro_diff_{_s}_{_p}", "<=", 1),
      (f"crowdmode_contact_n_compared_{_s}_{_p}", ">=", 1),
      (f"crowdmode_freedom_max_micro_diff_{_s}_{_p}", "<=", 1),
      (f"crowdmode_freedom_n_compared_{_s}_{_p}", ">=", 1),
    ]
for _s in _STRUCTS:
  INVARIANT_TERMS += [
    (f"crowdmode_effect_residues_{_s}", ">=", 1),
    (f"endpoint_violations_{_s}", "=", 0),
    (f"endpoint_n_eligible_{_s}", ">=", 1),
  ]
INVARIANT_TERMS += [
  ("dclash_mono_violations_1crn", "=", 0),
  ("dclash_mono_n_compared_1crn", ">=", 1),
  ("cbclash_effect_max_over_sweep_1crn", ">=", 1),
]


_PARITY_STRUCTS = ("1crn", "1ubq", "multichain")


def _parity_terms(contact_med, contact_p95, contact_max, cr, fr, one_sided):
  """cr/fr = (median, p95, max, max_multichain); contact_max/one_sided = (single, multichain)."""
  terms = []
  for s in _PARITY_STRUCTS:
    mc = s == "multichain"
    terms += [
      (f"median_contact_delta_{s}", "<=", contact_med),
      (f"p95_contact_delta_{s}", "<=", contact_p95),
      (f"max_contact_delta_{s}", "<=", contact_max[1] if mc else contact_max[0]),
      (f"max_one_sided_contact_degree_{s}", "<=", one_sided[1] if mc else one_sided[0]),
    ]
  for name, (med, p95, mx, mx_mc) in (("crowdedness", cr), ("freedom", fr)):
    for s in _PARITY_STRUCTS:
      terms += [
        (f"median_{name}_delta_{s}", "<=", med),
        (f"p95_{name}_delta_{s}", "<=", p95),
        (f"max_{name}_delta_{s}", "<=", mx_mc if s == "multichain" else mx),
      ]
  return terms


TIGHT_TERMS = _parity_terms(
  contact_med=0.0001,
  contact_p95=0.05,
  contact_max=(0.25, 0.35),
  cr=(0.035, 0.15, 0.35, 0.50),
  fr=(0.05, 0.15, 0.35, 0.50),
  one_sided=(0.25, 0.35),
) + [
  (f"median_{m}_delta_{s}_default", "<=", b)
  for m, b in (("contact", 0.0001), ("freedom", 0.05))
  for s in _STRUCTS
]

MARGINAL_TERMS = _parity_terms(
  contact_med=0.0002,
  contact_p95=0.10,
  contact_max=(0.40, 0.50),
  cr=(0.06, 0.25, 0.50, 0.70),
  fr=(0.10, 0.25, 0.50, 0.70),
  one_sided=(0.40, 0.50),
) + [
  (f"median_{m}_delta_{s}_default", "<=", b)
  for m, b in (("contact", 0.0002), ("freedom", 0.10))
  for s in _STRUCTS
]


def _term_holds(result: dict, term) -> bool:
  field, op, value = term
  actual = result[field]
  if isinstance(value, bool):
    if not isinstance(actual, bool):
      raise TypeError(f"{field} must be bool, got {actual!r}")
    return actual is value
  if isinstance(actual, bool) or not isinstance(actual, (int, float)):
    raise TypeError(f"{field} must be numeric, got {actual!r}")
  _require_finite(actual, field)
  if op == "=":
    return actual == value
  if op == "<=":
    return actual <= value
  if op == ">=":
    return actual >= value
  raise ValueError(f"unknown op {op!r}")


def _render_term(term) -> str:
  field, op, value = term
  if isinstance(value, bool):
    rendered = "true" if value else "false"
  else:
    rendered = repr(value)
  return f"{field} {op} {rendered}"


def _render_group(terms) -> str:
  return "(" + " AND ".join(_render_term(t) for t in terms) + ")"


def render_conditions() -> dict:
  """The exact [outcomes] condition strings the v2 sidecar must carry."""
  inv = _render_group(INVARIANT_TERMS)
  tight = _render_group(TIGHT_TERMS)
  marg = _render_group(MARGINAL_TERMS)
  return {
    "pass": f"{inv} AND {tight}",
    "marginal": f"{inv} AND {marg} AND NOT {tight}",
    "fail": f"NOT {inv} OR NOT {marg}",
  }


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


def _file_sha256(path: Path) -> str:
  h = hashlib.sha256()
  with open(path, "rb") as f:
    while chunk := f.read(65536):
      h.update(chunk)
  return h.hexdigest()


def _stamp_path(output_path: Path) -> Path:
  return output_path.with_name(output_path.name + ".stamp.json")


def _write_completion_stamp(
  output_path: Path, pdb_path: Path, binary: Path, rotlib: Path, extra_flags=None
) -> dict:
  stamp = {
    "completed": True,
    "input_sha256": _file_sha256(pdb_path),
    "output_sha256": _file_sha256(output_path),
    "binary_sha256": _file_sha256(binary),
    "rotlib_sha256": _file_sha256(rotlib),
    "extra_flags": list(extra_flags) if extra_flags else [],
  }
  stamp_file = _stamp_path(output_path)
  tmp_stamp = stamp_file.with_name(stamp_file.name + ".tmp")
  with open(tmp_stamp, "w", encoding="utf-8") as f:
    json.dump(stamp, f, indent=2)
  tmp_stamp.replace(stamp_file)
  return stamp


def parse_confind_output(path: Path, is_upstream: bool = False):
  """
  Parse confind output format:
  contact\t<chain_a>,<res_a>\t<chain_b>,<res_b>\t<cd>\t<name_a>\t<name_b>
  crwdnes/crowdedness\t<chain>,<res>\t<cr>\t...
  freedom\t<chain>,<res>\t<fr>\t...
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


def _cache_reusable(
  path: Path,
  binary: Path,
  rotlib: Path,
  pdb_path: Path | None = None,
  *,
  is_upstream: bool = True,
  extra_flags=None,
) -> bool:
  """
  Reuse a cached output only when it is non-empty, parseable, newer than the binary,
  and carries a valid completion stamp matching this input, binary, rotamer library and
  flags. Filenames do not encode the library, so swapping one must not reuse old output.
  """
  if not path.is_file() or path.stat().st_size == 0:
    return False
  if path.stat().st_mtime <= binary.stat().st_mtime:
    return False
  stamp_file = _stamp_path(path)
  if not stamp_file.is_file():
    return False
  try:
    stamp = json.loads(stamp_file.read_text(encoding="utf-8"))
  except (OSError, ValueError, UnicodeError):
    return False
  if not isinstance(stamp, dict) or stamp.get("completed") is not True:
    return False
  expected_flags = list(extra_flags) if extra_flags else []
  if stamp.get("extra_flags", []) != expected_flags:
    return False
  if pdb_path is not None:
    if not Path(pdb_path).is_file():
      return False
    if stamp.get("input_sha256") != _file_sha256(Path(pdb_path)):
      return False
  if stamp.get("output_sha256") != _file_sha256(path):
    return False
  if stamp.get("binary_sha256") != _file_sha256(binary):
    return False
  if stamp.get("rotlib_sha256") != _file_sha256(rotlib):
    return False
  try:
    contacts, crowdedness, freedom, _dihedrals = parse_confind_output(path, is_upstream=is_upstream)
  except (OSError, ValueError, IndexError, UnicodeError):
    return False
  return bool(contacts or crowdedness or freedom)


def _upstream_cache_reusable(
  path: Path, binary: Path, rotlib: Path, pdb_path: Path | None = None
) -> bool:
  return _cache_reusable(path, binary, rotlib, pdb_path, is_upstream=True, extra_flags=None)


def _clean_cache_reusable(
  path: Path, binary: Path, rotlib: Path, pdb_path: Path | None = None, extra_flags=None
) -> bool:
  return _cache_reusable(path, binary, rotlib, pdb_path, is_upstream=False, extra_flags=extra_flags)


def _record_output(path: Path, *, reused: bool) -> None:
  """Note whether an output was computed or reused; the first use in this process wins."""
  if str(path) in _OUTPUT_LEDGER:
    return
  stamp = json.loads(_stamp_path(path).read_text(encoding="utf-8"))
  _OUTPUT_LEDGER[str(path)] = {"reused_from_cache": reused, **stamp}


def _clean_out_path(scratch: Path, tag: str, extra_flags) -> Path:
  suffix = "_" + "_".join(flag.lstrip("-") for flag in extra_flags) if extra_flags else ""
  return scratch / f"{tag}_clean{suffix}.out"


def _ensure_clean_output(clean_out, clean_bin, clean_rlib, pdb_path, extra_flags) -> Path:
  """Run clean confind unless a stamped, sha256-matching cached output exists.

  Every clean output is registered in _CLEAN_RUNS for the arm-header check.
  """
  reused = _clean_cache_reusable(
    clean_out, clean_bin, clean_rlib, pdb_path, extra_flags=extra_flags
  )
  if not reused:
    if clean_out.exists():
      clean_out.unlink()
    if _stamp_path(clean_out).exists():
      _stamp_path(clean_out).unlink()
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
      timeout=UNIT_TIMEOUT_S,
    )
    _write_completion_stamp(clean_out, pdb_path, clean_bin, clean_rlib, extra_flags=extra_flags)
  _record_output(clean_out, reused=reused)
  key = str(clean_out)
  if key not in {p for p, _ in _CLEAN_RUNS}:
    _CLEAN_RUNS.append((key, list(extra_flags) if extra_flags else []))
  return clean_out


def run_clean(pdb_path: Path, tag: str, extra_flags=None):
  """Clean-only run (no reference) for the factorial arms and the d_clash sweep.

  Returns (contacts, crowdedness, freedom).
  """
  scratch = _require_dir(SCRATCH, "--scratch", ENV_SCRATCH)
  clean_bin = _require_file(CLEAN_BIN, "--clean-bin", ENV_CLEAN_BIN)
  clean_rlib = _require_file(CLEAN_RLIB, "--clean-rotlib", ENV_CLEAN_ROTLIB)
  pdb_path = _require_file(pdb_path, "--pdb-dir", ENV_PDB_DIR)
  clean_out = _ensure_clean_output(
    _clean_out_path(scratch, tag, extra_flags), clean_bin, clean_rlib, pdb_path, extra_flags
  )
  cl_c, cl_cr, cl_fr, _ = parse_confind_output(clean_out, is_upstream=False)
  return cl_c, cl_cr, cl_fr


def run_benchmark(pdb_path: Path, tag: str, extra_flags=None):
  """
  Run upstream and clean confind with preemption-safe completion stamps,
  reusing cached outputs only when the completion stamp and input PDB sha256 match.
  """
  scratch = _require_dir(SCRATCH, "--scratch", ENV_SCRATCH)
  upstream_bin = _require_file(UPSTREAM_BIN, "--reference-bin", ENV_REFERENCE_BIN)
  upstream_rlib = _require_file(UPSTREAM_RLIB, "--reference-rotlib", ENV_REFERENCE_ROTLIB)
  clean_bin = _require_file(CLEAN_BIN, "--clean-bin", ENV_CLEAN_BIN)
  clean_rlib = _require_file(CLEAN_RLIB, "--clean-rotlib", ENV_CLEAN_ROTLIB)
  pdb_path = _require_file(pdb_path, "--pdb-dir", ENV_PDB_DIR)

  up_out = scratch / f"{tag}_upstream.out"
  suffix = "_" + "_".join(flag.lstrip("-") for flag in extra_flags) if extra_flags else ""
  clean_out = _clean_out_path(scratch, tag, extra_flags)

  up_reused = _upstream_cache_reusable(up_out, upstream_bin, upstream_rlib, pdb_path)
  if not up_reused:
    if up_out.exists():
      up_out.unlink()
    if _stamp_path(up_out).exists():
      _stamp_path(up_out).unlink()
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
      timeout=UNIT_TIMEOUT_S,
    )
    _write_completion_stamp(up_out, pdb_path, upstream_bin, upstream_rlib, extra_flags=None)
  _record_output(up_out, reused=up_reused)

  _ensure_clean_output(clean_out, clean_bin, clean_rlib, pdb_path, extra_flags)

  up_c, up_cr, up_fr, up_dih = parse_confind_output(up_out, is_upstream=True)
  cl_c, cl_cr, cl_fr, _ = parse_confind_output(clean_out, is_upstream=False)

  # Persist per-structure completion record in scratch for preemption inspection
  struct_record_path = scratch / f"{tag}{suffix}_structure_result.json"
  struct_record = {
    "completed": True,
    "tag": tag,
    "input_pdb": str(pdb_path),
    "input_sha256": _file_sha256(pdb_path),
    "upstream_output_sha256": _file_sha256(up_out),
    "clean_output_sha256": _file_sha256(clean_out),
    "extra_flags": list(extra_flags) if extra_flags else [],
    "n_upstream_contacts": len(up_c),
    "n_clean_contacts": len(cl_c),
    "n_upstream_residues": len(up_cr),
    "n_clean_residues": len(cl_cr),
  }
  with open(struct_record_path, "w", encoding="utf-8") as f:
    json.dump(struct_record, f, indent=2)

  return up_c, up_cr, up_fr, up_dih, cl_c, cl_cr, cl_fr


def _quantile_from_sorted(sorted_deltas: list[float], q: float = 0.95) -> float:
  if not sorted_deltas:
    raise ValueError("empty delta sequence for quantile")
  if not (0.0 <= q <= 1.0):
    raise ValueError(f"quantile must be in [0, 1], got {q!r}")
  n = len(sorted_deltas)
  idx = min(n - 1, max(0, math.ceil(q * n) - 1))
  return float(sorted_deltas[idx])


def _common_contact_deltas(up_c, cl_c) -> list[float]:
  """Sorted |reference - clean| over pairs present in BOTH contact sets.

  One-sided pairs (a pair only one program considered) are never filled with a
  stand-in degree; they are counted and bounded separately by
  compute_one_sided_contacts (v2 sidecar, owner decision 2026-11-03).
  """
  if len(up_c) == 0 or len(cl_c) == 0:
    raise ValueError(f"empty contact set: reference={len(up_c)} clean={len(cl_c)}")
  for pair, value in up_c.items():
    _require_finite(value, f"reference contact {pair}")
  for pair, value in cl_c.items():
    _require_finite(value, f"clean contact {pair}")
  common = set(up_c) & set(cl_c)
  if not common:
    raise ValueError(
      f"no contact pair is present in both sets: reference={len(up_c)} clean={len(cl_c)}"
    )
  return sorted(abs(up_c[pair] - cl_c[pair]) for pair in common)


def compute_contact_deltas(up_c, cl_c):
  """Delta statistics over common pairs.

  Returns (median, mean, max, pct<=1e-4, pct<=0.05, n_union_pairs, n_common_pairs).
  """
  deltas = _common_contact_deltas(up_c, cl_c)
  n = len(deltas)
  median_val = deltas[n // 2]
  mean_val = sum(deltas) / n
  max_val = deltas[-1]
  pct_1e4 = sum(1 for delta in deltas if delta <= 1e-4) / n
  pct_005 = sum(1 for delta in deltas if delta <= 0.05) / n
  n_union = len(set(up_c) | set(cl_c))
  return median_val, mean_val, max_val, pct_1e4, pct_005, n_union, n


def compute_contact_p95_delta(up_c, cl_c) -> float:
  return _quantile_from_sorted(_common_contact_deltas(up_c, cl_c), 0.95)


def compute_one_sided_contacts(up_c, cl_c) -> dict:
  """Count and characterise contact pairs present on one side only.

  max_*_degree is the largest degree the PRESENT side reports for a one-sided pair,
  i.e. the delta that pair would carry if the absent side's degree were 0. When a side
  has no one-sided pairs its count is 0 and its max is 0.0 (supremum of an empty set
  of non-negative degrees); the count field disambiguates. A one-sided pair with a
  large present-side degree is a real contact the other program never considered
  (debt #3223).
  """
  _common_contact_deltas(up_c, cl_c)  # same emptiness/finiteness validation
  ref_only = [up_c[p] for p in set(up_c) - set(cl_c)]
  clean_only = [cl_c[p] for p in set(cl_c) - set(up_c)]
  n_union = len(set(up_c) | set(cl_c))
  max_ref = max(ref_only, default=0.0)
  max_clean = max(clean_only, default=0.0)
  return {
    "ref_only_count": len(ref_only),
    "clean_only_count": len(clean_only),
    "one_sided_fraction": (len(ref_only) + len(clean_only)) / n_union,
    "max_ref_only_degree": float(max_ref),
    "max_clean_only_degree": float(max_clean),
    "max_one_sided_degree": float(max(max_ref, max_clean)),
  }


# ---------------------------------------------------------------------------
# Clean-only invariants (v2, oracle review 2026-11-04). Each function compares
# two proxide outputs whose relationship holds BY CONSTRUCTION, so every gate
# below can fail on a real defect and none is satisfiable by assertion (B6).
# Every comparison requires identical key sets and raises otherwise, so a
# reported n_compared is the full residue/pair count, never an intersection.
# ---------------------------------------------------------------------------


def _require_same_keys(a: dict, b: dict, label: str) -> list:
  if set(a) != set(b):
    raise ValueError(
      f"{label}: key sets differ: only_first={sorted(set(a) - set(b))} "
      f"only_second={sorted(set(b) - set(a))}"
    )
  if not a:
    raise ValueError(f"{label}: empty comparison set")
  for key in a:
    _require_finite(a[key], f"{label} first {key}")
    _require_finite(b[key], f"{label} second {key}")
  return sorted(a)


def cbclash_monotonicity(c_all: dict, c_ala: dict) -> tuple[int, int]:
  """Count residues where crowdedness(ala-only) > crowdedness(all-residues).

  AllResidues adds CB to the clash set (cache.rs:171-176), so its surviving set is a
  subset of AlaOnly's and C(all) >= C(ala) per residue, exactly, at fixed crowdmode.
  Returns (violations, n_compared).
  """
  keys = _require_same_keys(c_all, c_ala, "cbclash monotonicity")
  return sum(1 for k in keys if c_ala[k] > c_all[k]), len(keys)


def cbclash_effect(c_all: dict, c_ala: dict) -> int:
  """Residues where all-residues crowdedness strictly exceeds ala-only (flag reaches kernel)."""
  keys = _require_same_keys(c_all, c_ala, "cbclash effect")
  return sum(1 for k in keys if c_all[k] > c_ala[k])


def max_micro_diff(a: dict, b: dict, label: str) -> tuple[int, int]:
  """Max |round(x*1e6) - round(y*1e6)| over identical keys, in integer micro-units.

  Outputs are printed to 6 decimals (bin/confind.rs:140); integer micro-units avoid a
  float `<= 1e-6` check failing on a legitimate one-unit rounding flip.
  Returns (max_diff, n_compared).
  """
  keys = _require_same_keys(a, b, label)
  return max(abs(round(a[k] * 1e6) - round(b[k] * 1e6)) for k in keys), len(keys)


def crowdmode_effect(c_weighted: dict, c_count: dict) -> int:
  """Residues whose crowdedness differs between crowdmodes at fixed cbclash."""
  keys = _require_same_keys(c_weighted, c_count, "crowdmode effect")
  return sum(1 for k in keys if c_weighted[k] != c_count[k])


def endpoint_agreement(c_count: dict, c_weighted: dict) -> tuple[int, int]:
  """Count-mode endpoints must coincide with weighted-mode endpoints.

  N_surv and W_surv update in the same branch (cache.rs:184-186): N_surv = N_lib
  implies W_surv = W_lib (C = 0 in both) and N_surv = 0 implies W_surv = 0 (C = 1 in
  both). Returns (violations, n_eligible) over residues with count-mode C in {0, 1}.
  """
  keys = _require_same_keys(c_count, c_weighted, "endpoint agreement")
  eligible = [k for k in keys if c_count[k] in (0.0, 1.0)]
  return sum(1 for k in eligible if c_weighted[k] != c_count[k]), len(eligible)


def dclash_monotonicity(series: list[dict]) -> tuple[int, int]:
  """Crowdedness must be non-decreasing in d_clash (inclusive radius filter, grid.rs:110).

  `series` is ordered by increasing d_clash at fixed policy and crowdmode.
  Returns (violations, n_compared) over consecutive pairs.
  """
  if len(series) < 2:
    raise ValueError("dclash monotonicity needs at least two sweep points")
  violations = 0
  n = 0
  for lo, hi in zip(series, series[1:], strict=False):
    keys = _require_same_keys(lo, hi, "dclash monotonicity")
    violations += sum(1 for k in keys if lo[k] > hi[k])
    n += len(keys)
  return violations, n


def _expected_header(extra_flags) -> dict:
  flags = list(extra_flags) if extra_flags else []
  expected = dict(EXPECTED_DEFAULT_HEADER)
  if len(flags) % 2:
    raise ValueError(f"extra_flags must be flag/value pairs: {flags!r}")
  for flag, value in zip(flags[::2], flags[1::2], strict=True):
    key = flag.lstrip("-")
    if key in expected:
      expected[key] = value
  return expected


def read_output_header(path: Path) -> dict:
  header = {}
  with open(path, encoding="utf-8") as f:
    for line in f:
      if not line.startswith("# "):
        continue
      parts = line[2:].rstrip("\n").split("\t")
      if len(parts) >= 2:
        header[parts[0]] = parts[1]
  return header


def compute_clean_invariants(factorial: dict, sweep: dict) -> dict:
  """Result fields for the clean-only invariants.

  factorial[s][arm] = (contacts, crowdedness, freedom) for s in {1crn, 1ubq} and
  arm A = ala-only/count (parity), B = all-residues/weighted (default),
  C = all-residues/count, D = ala-only/weighted.
  sweep[policy] = [crowdedness at each DCLASH_SWEEP value, ascending] for 1crn, count mode.
  """
  out: dict = {}
  for s, arms in factorial.items():
    c_a, c_b, c_c, c_d = (arms[k][1] for k in ("A", "B", "C", "D"))
    out[f"n_residues_{s}"] = len(_require_same_keys(c_a, c_b, f"{s} residue set"))
    v, n = cbclash_monotonicity(c_c, c_a)
    out[f"cbclash_mono_violations_{s}_count"] = v
    out[f"cbclash_mono_n_compared_{s}_count"] = n
    v, n = cbclash_monotonicity(c_b, c_d)
    out[f"cbclash_mono_violations_{s}_weighted"] = v
    out[f"cbclash_mono_n_compared_{s}_weighted"] = n
    for policy, (w, c) in (("all", ("B", "C")), ("ala", ("D", "A"))):
      d, n = max_micro_diff(arms[w][0], arms[c][0], f"{s} {policy} crowdmode contact")
      out[f"crowdmode_contact_max_micro_diff_{s}_{policy}"] = d
      out[f"crowdmode_contact_n_compared_{s}_{policy}"] = n
      d, n = max_micro_diff(arms[w][2], arms[c][2], f"{s} {policy} crowdmode freedom")
      out[f"crowdmode_freedom_max_micro_diff_{s}_{policy}"] = d
      out[f"crowdmode_freedom_n_compared_{s}_{policy}"] = n
    out[f"crowdmode_effect_residues_{s}"] = crowdmode_effect(c_b, c_c)
    v1, n1 = endpoint_agreement(c_c, c_b)
    v2, n2 = endpoint_agreement(c_a, c_d)
    out[f"endpoint_violations_{s}"] = v1 + v2
    out[f"endpoint_n_eligible_{s}"] = n1 + n2
    # Descriptive only (not gated): signed default-minus-parity shift and its factors.
    keys = sorted(c_a)
    for name, (hi, lo) in (
      ("default_minus_parity", (c_b, c_a)),
      ("cbclash_part", (c_c, c_a)),
      ("crowdmode_part", (c_b, c_c)),
    ):
      shifts = sorted(hi[k] - lo[k] for k in keys)
      out[f"crowdedness_shift_{name}_median_signed_{s}"] = round(shifts[len(shifts) // 2], 6)

  violations = 0
  n_compared = 0
  for policy in SWEEP_POLICIES:
    v, n = dclash_monotonicity(sweep[policy])
    violations += v
    n_compared += n
  out["dclash_mono_violations_1crn"] = violations
  out["dclash_mono_n_compared_1crn"] = n_compared
  effects = []
  for i, x in enumerate(DCLASH_SWEEP):
    c_all = sweep["all-residues"][i]
    c_ala = sweep["ala-only"][i]
    effect = cbclash_effect(c_all, c_ala)
    effects.append(effect)
    out[f"cbclash_effect_residues_1crn_dclash_{_dclash_tag(x)}"] = effect
    out[f"ala_only_saturated_residues_1crn_dclash_{_dclash_tag(x)}"] = sum(
      1 for v in c_ala.values() if v == 1.0
    )
  out["cbclash_effect_max_over_sweep_1crn"] = max(effects)
  return out


def check_arm_headers(runs) -> tuple[int, int]:
  """Compare each clean output's header with the arm that was requested.

  Catches a changed Default (which would silently turn the default arm into the parity
  arm) or a mis-parsed flag. Returns (mismatch_count, n_headers_checked).
  """
  mismatches = 0
  for path, extra_flags in runs:
    header = read_output_header(Path(path))
    expected = _expected_header(extra_flags)
    ok = (
      header.get("cbclash") == expected["cbclash"]
      and header.get("crowdmode") == expected["crowdmode"]
      and "dclash" in header
      and float(header["dclash"]) == float(expected["dclash"])
    )
    if not ok:
      mismatches += 1
  return mismatches, len(runs)


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


def compute_metric_p95_delta(up_m, cl_m) -> float:
  compute_metric_deltas(up_m, cl_m)
  deltas = sorted(abs(up_m[key] - cl_m[key]) for key in up_m)
  return _quantile_from_sorted(deltas, 0.95)


def assess_omega_sign(pdb_path: Path, up_dih):
  """
  Compare the reference binary's omega with an IUPAC omega computed HERE from the PDB
  (Debt #1903). This does NOT exercise proxide-confind: its output carries no dihedrals
  (bin/confind.rs). It checks the reference binary and this harness only (debt #3224).
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
    default=None,
    help=f"JSON result path (env: {ENV_OUT}); no default path",
  )
  parser.add_argument("--reference-bin", default=None, help=f"env: {ENV_REFERENCE_BIN}")
  parser.add_argument("--reference-rotlib", default=None, help=f"env: {ENV_REFERENCE_ROTLIB}")
  parser.add_argument("--clean-bin", default=None, help=f"env: {ENV_CLEAN_BIN}")
  parser.add_argument("--clean-rotlib", default=None, help=f"env: {ENV_CLEAN_ROTLIB}")
  parser.add_argument("--pdb-dir", default=None, help=f"env: {ENV_PDB_DIR}")
  parser.add_argument("--scratch", default=None, help=f"env: {ENV_SCRATCH}")
  return parser


def _is_disallowed_tmp_out(path: Path) -> bool:
  """Reject ephemeral /tmp or /var/tmp output paths, while allowing pytest tmp_path fixtures."""
  try:
    resolved = path.resolve()
  except OSError:
    resolved = path
  parts = resolved.parts
  if any(part.startswith("pytest-") for part in parts):
    return False
  if len(parts) >= 2 and parts[0] == "/" and parts[1] == "tmp":
    return True
  if len(parts) >= 3 and parts[0] == "/" and parts[1] == "var" and parts[2] == "tmp":
    return True
  return False


def _resolve(parser, flag_value, flag, env_var, kind):
  raw = flag_value if flag_value else os.environ.get(env_var)
  if raw is None or str(raw).strip() == "":
    parser.error(f"missing required location: {flag} (or {env_var})")
  path = Path(raw)
  if kind == "out":
    if _is_disallowed_tmp_out(path):
      parser.error(
        f"refusing ephemeral /tmp result location for {flag} (or {env_var}): {path}; "
        "use a tracked/configured project path instead"
      )
    return path
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


def evaluate_outcome(result: dict) -> str:
  """
  Evaluate the pre-registered v2 outcome partition
  (scripts/validation/confind_blackbox_parity_v2.bth.toml), built from the same
  INVARIANT_TERMS / TIGHT_TERMS / MARGINAL_TERMS the sidecar text is rendered from.
  pass = INV and TIGHT; marginal = INV and MARG and not TIGHT; fail = not INV or not
  MARG. Every TIGHT bound is at least as strict as its MARG bound, so the three are
  mutually exclusive and exhaustive and any invariant failure is 'fail'.
  The authoritative verdict is the bathos run record, not this function.
  """
  inv = all(_term_holds(result, t) for t in INVARIANT_TERMS)
  tight = all(_term_holds(result, t) for t in TIGHT_TERMS)
  marg = all(_term_holds(result, t) for t in MARGINAL_TERMS)
  if not inv or not marg:
    return "fail"
  if tight:
    return "pass"
  return "marginal"


def exit_code_for_outcome(outcome):
  if outcome == "pass":
    return 0
  return 1


def main():
  global SCRATCH, UPSTREAM_BIN, UPSTREAM_RLIB, CLEAN_BIN, CLEAN_RLIB
  global CRN_PDB, UBQ_PDB, MULTICHAIN_PDB, BREAK_PDB

  parser = _parser()
  args = parser.parse_args()
  out_path = _resolve(parser, args.out, "--out", ENV_OUT, "out")
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

  parity_flags = list(PARITY_FLAGS)
  _CLEAN_RUNS.clear()
  _OUTPUT_LEDGER.clear()

  print("Running Benchmark 1 (parity arm: ala-only + pruned-count-fraction): 1CRN...")
  up_c_crn, up_cr_crn, up_fr_crn, up_dih_crn, cl_c_crn, cl_cr_crn, cl_fr_crn = run_benchmark(
    CRN_PDB, "1crn", parity_flags
  )
  med_cd_crn, mean_cd_crn, max_cd_crn, pct_1e4_crn, pct_005_crn, _, _ = compute_contact_deltas(
    up_c_crn, cl_c_crn
  )
  p95_cd_crn = compute_contact_p95_delta(up_c_crn, cl_c_crn)
  med_cr_crn, _, max_cr_crn = compute_metric_deltas(up_cr_crn, cl_cr_crn)
  p95_cr_crn = compute_metric_p95_delta(up_cr_crn, cl_cr_crn)
  med_fr_crn, _, max_fr_crn = compute_metric_deltas(up_fr_crn, cl_fr_crn)
  p95_fr_crn = compute_metric_p95_delta(up_fr_crn, cl_fr_crn)

  print("Running Benchmark 1b (default arm: all-residues + probability-weighted): 1CRN...")
  _, _, _, _, cl_c_crn_def, cl_cr_crn_def, cl_fr_crn_def = run_benchmark(CRN_PDB, "1crn", None)
  med_cd_crn_def, _, max_cd_crn_def, _, _, _, _ = compute_contact_deltas(up_c_crn, cl_c_crn_def)
  p95_cd_crn_def = compute_contact_p95_delta(up_c_crn, cl_c_crn_def)
  med_cr_crn_def, _, max_cr_crn_def = compute_metric_deltas(up_cr_crn, cl_cr_crn_def)
  p95_cr_crn_def = compute_metric_p95_delta(up_cr_crn, cl_cr_crn_def)
  med_fr_crn_def, _, max_fr_crn_def = compute_metric_deltas(up_fr_crn, cl_fr_crn_def)
  p95_fr_crn_def = compute_metric_p95_delta(up_fr_crn, cl_fr_crn_def)

  print("Running Benchmark 2 (parity arm: ala-only + pruned-count-fraction): 1UBQ...")
  up_c_ubq, up_cr_ubq, up_fr_ubq, _up_dih_ubq, cl_c_ubq, cl_cr_ubq, cl_fr_ubq = run_benchmark(
    UBQ_PDB, "1ubq", parity_flags
  )
  med_cd_ubq, mean_cd_ubq, max_cd_ubq, pct_1e4_ubq, pct_005_ubq, _, _ = compute_contact_deltas(
    up_c_ubq, cl_c_ubq
  )
  p95_cd_ubq = compute_contact_p95_delta(up_c_ubq, cl_c_ubq)
  med_cr_ubq, _, max_cr_ubq = compute_metric_deltas(up_cr_ubq, cl_cr_ubq)
  p95_cr_ubq = compute_metric_p95_delta(up_cr_ubq, cl_cr_ubq)
  med_fr_ubq, _, max_fr_ubq = compute_metric_deltas(up_fr_ubq, cl_fr_ubq)
  p95_fr_ubq = compute_metric_p95_delta(up_fr_ubq, cl_fr_ubq)

  print("Running Benchmark 2b (default arm: all-residues + probability-weighted): 1UBQ...")
  _, _, _, _, cl_c_ubq_def, cl_cr_ubq_def, cl_fr_ubq_def = run_benchmark(UBQ_PDB, "1ubq", None)
  med_cd_ubq_def, _, max_cd_ubq_def, _, _, _, _ = compute_contact_deltas(up_c_ubq, cl_c_ubq_def)
  p95_cd_ubq_def = compute_contact_p95_delta(up_c_ubq, cl_c_ubq_def)
  med_cr_ubq_def, _, max_cr_ubq_def = compute_metric_deltas(up_cr_ubq, cl_cr_ubq_def)
  p95_cr_ubq_def = compute_metric_p95_delta(up_cr_ubq, cl_cr_ubq_def)
  med_fr_ubq_def, _, max_fr_ubq_def = compute_metric_deltas(up_fr_ubq, cl_fr_ubq_def)
  p95_fr_ubq_def = compute_metric_p95_delta(up_fr_ubq, cl_fr_ubq_def)

  print("Running Benchmark 3 (parity arm): Multi-chain complex (f2c006r13.pdb)...")
  up_c_mc, up_cr_mc, up_fr_mc, _, cl_c_mc, cl_cr_mc, cl_fr_mc = run_benchmark(
    MULTICHAIN_PDB, "multichain", parity_flags
  )
  med_cd_mc, _mean_cd_mc, max_cd_mc, pct_1e4_mc, pct_005_mc, n_pairs_mc, com_pairs_mc = (
    compute_contact_deltas(up_c_mc, cl_c_mc)
  )
  p95_cd_mc = compute_contact_p95_delta(up_c_mc, cl_c_mc)
  med_cr_mc, _, max_cr_mc = compute_metric_deltas(up_cr_mc, cl_cr_mc)
  p95_cr_mc = compute_metric_p95_delta(up_cr_mc, cl_cr_mc)
  med_fr_mc, _, max_fr_mc = compute_metric_deltas(up_fr_mc, cl_fr_mc)
  p95_fr_mc = compute_metric_p95_delta(up_fr_mc, cl_fr_mc)

  print("Assessing Debt #1903 (reference omega vs harness IUPAC omega; not proxide, #3224)...")
  same_sign, opp_sign, max_omg_delta = assess_omega_sign(CRN_PDB, up_dih_crn)

  print("Assessing Debt #1904 (Chain Semantics & Breaks)...")
  prevented, _, _ = assess_chain_semantics()

  print("Running clean-only factorial arms (all-residues/count, ala-only/weighted)...")
  factorial = {}
  for s, pdb in (("1crn", CRN_PDB), ("1ubq", UBQ_PDB)):
    arm_a = (cl_c_crn, cl_cr_crn, cl_fr_crn) if s == "1crn" else (cl_c_ubq, cl_cr_ubq, cl_fr_ubq)
    arm_b = (
      (cl_c_crn_def, cl_cr_crn_def, cl_fr_crn_def)
      if s == "1crn"
      else (cl_c_ubq_def, cl_cr_ubq_def, cl_fr_ubq_def)
    )
    factorial[s] = {
      "A": arm_a,
      "B": arm_b,
      "C": run_clean(pdb, s, ALL_COUNT_FLAGS),
      "D": run_clean(pdb, s, ALA_WEIGHTED_FLAGS),
    }

  print(f"Running d_clash sweep on 1CRN: {DCLASH_SWEEP} x {SWEEP_POLICIES}...")
  sweep = {
    policy: [
      run_clean(
        CRN_PDB,
        "1crn",
        ("--cbclash", policy, "--crowdmode", "pruned-count-fraction", "--dclash", x),
      )[1]
      for x in DCLASH_SWEEP
    ]
    for policy in SWEEP_POLICIES
  }

  invariant_fields = compute_clean_invariants(factorial, sweep)
  header_mismatch, n_headers = check_arm_headers(list(_CLEAN_RUNS))
  invariant_fields["arm_header_mismatch_count"] = int(header_mismatch)
  invariant_fields["n_headers_checked"] = int(n_headers)

  one_sided = {
    "1crn": compute_one_sided_contacts(up_c_crn, cl_c_crn),
    "1ubq": compute_one_sided_contacts(up_c_ubq, cl_c_ubq),
    "multichain": compute_one_sided_contacts(up_c_mc, cl_c_mc),
    "1crn_default": compute_one_sided_contacts(up_c_crn, cl_c_crn_def),
    "1ubq_default": compute_one_sided_contacts(up_c_ubq, cl_c_ubq_def),
  }
  one_sided_fields = {}
  for tag, stats in one_sided.items():
    one_sided_fields[f"ref_only_contact_count_{tag}"] = int(stats["ref_only_count"])
    one_sided_fields[f"clean_only_contact_count_{tag}"] = int(stats["clean_only_count"])
    one_sided_fields[f"one_sided_contact_fraction_{tag}"] = round(
      float(stats["one_sided_fraction"]), 6
    )
    one_sided_fields[f"max_ref_only_contact_degree_{tag}"] = round(
      float(stats["max_ref_only_degree"]), 6
    )
    one_sided_fields[f"max_clean_only_contact_degree_{tag}"] = round(
      float(stats["max_clean_only_degree"]), 6
    )
    one_sided_fields[f"max_one_sided_contact_degree_{tag}"] = round(
      float(stats["max_one_sided_degree"]), 6
    )

  result = {
    **one_sided_fields,
    **invariant_fields,
    "parity_arm_label": PARITY_ARM_LABEL,
    "median_contact_delta_1crn": round(float(med_cd_crn), 6),
    "mean_contact_delta_1crn": round(float(mean_cd_crn), 6),
    "p95_contact_delta_1crn": round(float(p95_cd_crn), 6),
    "max_contact_delta_1crn": round(float(max_cd_crn), 6),
    "pct_within_1e4_1crn": round(float(pct_1e4_crn), 4),
    "pct_within_005_1crn": round(float(pct_005_crn), 4),
    "median_crowdedness_delta_1crn": round(float(med_cr_crn), 6),
    "p95_crowdedness_delta_1crn": round(float(p95_cr_crn), 6),
    "max_crowdedness_delta_1crn": round(float(max_cr_crn), 6),
    "median_freedom_delta_1crn": round(float(med_fr_crn), 6),
    "p95_freedom_delta_1crn": round(float(p95_fr_crn), 6),
    "max_freedom_delta_1crn": round(float(max_fr_crn), 6),
    "median_contact_delta_1ubq": round(float(med_cd_ubq), 6),
    "mean_contact_delta_1ubq": round(float(mean_cd_ubq), 6),
    "p95_contact_delta_1ubq": round(float(p95_cd_ubq), 6),
    "max_contact_delta_1ubq": round(float(max_cd_ubq), 6),
    "pct_within_1e4_1ubq": round(float(pct_1e4_ubq), 4),
    "pct_within_005_1ubq": round(float(pct_005_ubq), 4),
    "median_crowdedness_delta_1ubq": round(float(med_cr_ubq), 6),
    "p95_crowdedness_delta_1ubq": round(float(p95_cr_ubq), 6),
    "max_crowdedness_delta_1ubq": round(float(max_cr_ubq), 6),
    "median_freedom_delta_1ubq": round(float(med_fr_ubq), 6),
    "p95_freedom_delta_1ubq": round(float(p95_fr_ubq), 6),
    "max_freedom_delta_1ubq": round(float(max_fr_ubq), 6),
    "multichain_contacts_count": int(n_pairs_mc),
    "multichain_common_pairs": int(com_pairs_mc),
    "median_contact_delta_multichain": round(float(med_cd_mc), 6),
    "p95_contact_delta_multichain": round(float(p95_cd_mc), 6),
    "max_contact_delta_multichain": round(float(max_cd_mc), 6),
    "pct_within_1e4_multichain": round(float(pct_1e4_mc), 4),
    "pct_within_005_multichain": round(float(pct_005_mc), 4),
    "median_crowdedness_delta_multichain": round(float(med_cr_mc), 6),
    "p95_crowdedness_delta_multichain": round(float(p95_cr_mc), 6),
    "max_crowdedness_delta_multichain": round(float(max_cr_mc), 6),
    "median_freedom_delta_multichain": round(float(med_fr_mc), 6),
    "p95_freedom_delta_multichain": round(float(p95_fr_mc), 6),
    "max_freedom_delta_multichain": round(float(max_fr_mc), 6),
    "default_arm_label": DEFAULT_ARM_LABEL,
    "median_contact_delta_1crn_default": round(float(med_cd_crn_def), 6),
    "p95_contact_delta_1crn_default": round(float(p95_cd_crn_def), 6),
    "max_contact_delta_1crn_default": round(float(max_cd_crn_def), 6),
    "median_crowdedness_delta_1crn_default": round(float(med_cr_crn_def), 6),
    "p95_crowdedness_delta_1crn_default": round(float(p95_cr_crn_def), 6),
    "max_crowdedness_delta_1crn_default": round(float(max_cr_crn_def), 6),
    "median_freedom_delta_1crn_default": round(float(med_fr_crn_def), 6),
    "p95_freedom_delta_1crn_default": round(float(p95_fr_crn_def), 6),
    "max_freedom_delta_1crn_default": round(float(max_fr_crn_def), 6),
    "median_contact_delta_1ubq_default": round(float(med_cd_ubq_def), 6),
    "p95_contact_delta_1ubq_default": round(float(p95_cd_ubq_def), 6),
    "max_contact_delta_1ubq_default": round(float(max_cd_ubq_def), 6),
    "median_crowdedness_delta_1ubq_default": round(float(med_cr_ubq_def), 6),
    "p95_crowdedness_delta_1ubq_default": round(float(p95_cr_ubq_def), 6),
    "max_crowdedness_delta_1ubq_default": round(float(max_cr_ubq_def), 6),
    "median_freedom_delta_1ubq_default": round(float(med_fr_ubq_def), 6),
    "p95_freedom_delta_1ubq_default": round(float(p95_fr_ubq_def), 6),
    "max_freedom_delta_1ubq_default": round(float(max_fr_ubq_def), 6),
    "reference_omega_same_sign_count": int(same_sign),
    "reference_omega_opposite_sign_count": int(opp_sign),
    "max_reference_omega_delta_deg": round(float(max_omg_delta), 6),
    "chain_break_prevented_spurious_contacts": bool(prevented),
  }
  outcome_label = evaluate_outcome(result)
  result["outcome"] = outcome_label

  out_path.parent.mkdir(parents=True, exist_ok=True)
  with open(out_path, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)
  ledger_path = out_path.with_name(out_path.name + ".outputs.json")
  with open(ledger_path, "w", encoding="utf-8") as f:
    json.dump(_OUTPUT_LEDGER, f, indent=2)

  bth_results_path = os.environ.get("BTH_RESULTS_PATH")
  if bth_results_path:
    with open(bth_results_path, "w", encoding="utf-8") as f:
      json.dump(result, f, indent=2)

  print(f"\nVerification Complete! Verdict: {outcome_label.upper()}")
  print(f"Summary: {json.dumps(result, indent=2)}")
  return exit_code_for_outcome(outcome_label)


if __name__ == "__main__":
  if os.environ.get("BTH_RESULTS_PATH"):
    # bth binds the sidecar next to the invoked script: here, v1, whose schema this
    # harness no longer emits. Refuse instead of recording a meaningless outcome.
    sys.exit(
      "refusing: a tracked run of this script binds the v1 sidecar; "
      "run scripts/validation/confind_blackbox_parity_v2.py instead"
    )
  sys.exit(main())
