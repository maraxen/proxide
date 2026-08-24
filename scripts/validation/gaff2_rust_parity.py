"""Diff proxide's new Rust GAFF2 engine against the Python reference, on the
real geostd ligand corpus.

This is the Rust-port counterpart to `gaff2_geostd_sample.py`. It reuses that
script's `discover_candidates`/`fetch_mol2` (geostd discovery + caching) and
`Molecule.from_mol2` parsing verbatim -- see that script's docstring for why
RDKit's native mol2 parser can't be used directly on geostd's files.

**The comparison is Rust output vs PYTHON output, not vs geostd's ground
truth.** `proxide.chem.gaff2.assign_gaff2_atom_types` was itself validated
against geostd (see `.praxia/docs/audits/260820_gaff2-parity-verdict.md`,
verdict PARITY) and against a real external reference (antechamber). This
port's job is to reproduce THAT already-validated Python behavior bit-for-bit
-- including any of its known quirks/bugs -- not to independently re-derive
correctness against geostd. So the primary metric here is Rust-vs-Python
exact atom-type match, atom by atom, on the same parsed molecule.

As a secondary cross-check, this script ALSO computes each ligand's
Python-vs-geostd and Rust-vs-geostd mismatch signatures (using the same
(reference_type, candidate_type) pairing geostd_sample.py uses) and, at the
end of the run, compares the two SIGNATURE SETS gathered across the whole
sample. If Rust reproduces Python exactly, these two sets are identical by
construction (transitivity: Rust == Python on every atom implies Rust and
Python diverge from geostd in exactly the same places). If the sets differ
-- a new signature Python never produced, or a known Python signature that
stops appearing -- that is real evidence the port diverged from Python's
actual behavior, even if the raw Rust-vs-Python match rate looks high (e.g.
a rare rule path that happens to fire on very few sampled ligands). This is
reported as `signature_set_identical_to_python` and is treated as a hard
gate, independent of match rate.

Building the Rust engine (must be done once before running this script):

    cargo build -p proxide-gaff2 --features python-validation
    cp target/debug/libproxide_gaff2.so <repo_root>/proxide_gaff2.so
    # macOS: cp target/debug/libproxide_gaff2.dylib <repo_root>/proxide_gaff2.so

`python-validation` is a throwaway feature gate (see
`crates/proxide-gaff2/src/py_validation.rs`'s module doc) -- it is not part
of the real `crates/proxide_py` Cutover integration, which is untouched.
This script looks for `proxide_gaff2.so` at the repo root by default; pass
`--rust-lib-dir` to point elsewhere (e.g. if you built it into a scratch
directory instead of copying it into the tree).

Usage:
    uv run python scripts/validation/gaff2_rust_parity.py
    uv run python scripts/validation/gaff2_rust_parity.py --sample-size 3000 --seed 42
    uv run python scripts/validation/gaff2_rust_parity.py --json-out report.json
    uv run python scripts/validation/gaff2_rust_parity.py --full --workers 24 --json-out full.json
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib
import json
import logging
import random
import sys
import urllib.error
from collections import defaultdict
from pathlib import Path

from rdkit import Chem

from proxide.chem.gaff2 import assign_gaff2_atom_types
from proxide.io.parsing.molecule import Molecule

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gaff2_geostd_sample import (  # noqa: E402 -- path must be set up first
  discover_candidates,
  fetch_mol2,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("gaff2_rust_parity")

_DEFAULT_REF = "eaf8906a478e0fee4e6e6d3d3b2854d0fa13eabb"  # master, captured 2026-08-20
_DEFAULT_SAMPLE_SIZE = 3000
_DEFAULT_SEED = 42
_DEFAULT_CACHE_DIR = Path(".cache/geostd")

_BUILD_HELP = (
  "Could not import the Rust GAFF2 validation extension 'proxide_gaff2'.\n"
  "Build it first:\n"
  "  cargo build -p proxide-gaff2 --features python-validation\n"
  "  cp target/debug/libproxide_gaff2.so <repo_root>/proxide_gaff2.so\n"
  "  (macOS: cp target/debug/libproxide_gaff2.dylib <repo_root>/proxide_gaff2.so)\n"
  "then re-run, or pass --rust-lib-dir pointing at the directory containing it.\n"
)


def load_rust_engine(rust_lib_dir: Path):
  """Import the pyo3 `proxide_gaff2` extension module from `rust_lib_dir`.

  Raises SystemExit with build instructions on failure -- this is a hard
  prerequisite, not a gracefully-degradable one (there is nothing useful
  this script can report without the Rust engine).
  """
  sys.path.insert(0, str(rust_lib_dir))
  try:
    return importlib.import_module("proxide_gaff2")
  except ImportError as e:
    raise SystemExit(f"{_BUILD_HELP}\nLooked in: {rust_lib_dir}\nOriginal error: {e}") from e


def mol_to_rust_inputs(
  mol: Chem.Mol,
) -> tuple[list[str], list[tuple[int, int, int, bool]], list[int], list[list[int]]]:
  """Build the plain (elements, bonds, formal_charges, rings) tuple that
  `proxide_gaff2.assign_gaff2_atom_types_rs` expects, mirroring EXACTLY
  what `assign_gaff2_atom_types` (Python) does to its own internal copy of
  `mol` before rule matching -- see gaff2.py's `mol_for_matching` Kekulize
  comment. Never mutates the caller's `mol`.
  """
  mol_for_matching = Chem.Mol(mol)
  Chem.Kekulize(mol_for_matching, clearAromaticFlags=False)

  elements = [atom.GetSymbol() for atom in mol_for_matching.GetAtoms()]
  formal_charges = [atom.GetFormalCharge() for atom in mol_for_matching.GetAtoms()]
  bonds = [
    (
      bond.GetBeginAtomIdx(),
      bond.GetEndAtomIdx(),
      int(round(bond.GetBondTypeAsDouble())),
      bond.GetIsAromatic(),
    )
    for bond in mol_for_matching.GetBonds()
  ]
  rings = [list(ring) for ring in mol_for_matching.GetRingInfo().AtomRings()]
  return elements, bonds, formal_charges, rings


def compare_one(bucket: str, code: str, ref: str, cache_dir: Path, rust_engine) -> dict:
  """Fetch, parse, type with both engines, and diff one ligand. Never
  raises -- errors are returned as a status string."""
  try:
    path = fetch_mol2(bucket, code, ref, cache_dir)
  except urllib.error.HTTPError as e:
    return {"code": code, "status": "fetch_error", "detail": str(e)}

  try:
    molecule = Molecule.from_mol2(path)
    rdmol = molecule._to_rdkit()  # noqa: SLF001 -- same intended internal use as geostd_sample.py
    python_types = assign_gaff2_atom_types(rdmol)
  except Exception as e:  # noqa: BLE001 -- bulk external-corpus sample: log and skip, don't crash the run
    return {"code": code, "status": "python_error", "detail": f"{type(e).__name__}: {e}"}

  try:
    elements, bonds, formal_charges, rings = mol_to_rust_inputs(rdmol)
    rust_types = rust_engine.assign_gaff2_atom_types_rs(elements, bonds, formal_charges, rings)
  except Exception as e:  # noqa: BLE001 -- same rationale as python_error above
    return {"code": code, "status": "rust_error", "detail": f"{type(e).__name__}: {e}"}

  if len(python_types) != len(rust_types):
    return {
      "code": code,
      "status": "length_mismatch",
      "detail": f"{len(python_types)} python vs {len(rust_types)} rust",
    }

  diffs = [
    {"atom_idx": i, "element": molecule.elements[i], "python": p, "rust": r}
    for i, (p, r) in enumerate(zip(python_types, rust_types, strict=True))
    if p != r
  ]

  # Secondary cross-check evidence (aggregated across the sample in main()):
  # this ligand's Python-vs-geostd and Rust-vs-geostd mismatch signatures,
  # using geostd_sample.py's own (reference, candidate) pairing convention.
  geostd_types = molecule.atom_types
  py_geostd_sig: list[tuple[str, str]] = []
  rust_geostd_sig: list[tuple[str, str]] = []
  if len(geostd_types) == len(python_types):
    py_geostd_sig = [
      (g, p) for g, p in zip(geostd_types, python_types, strict=True) if g and g != p
    ]
  if len(geostd_types) == len(rust_types):
    rust_geostd_sig = [
      (g, r) for g, r in zip(geostd_types, rust_types, strict=True) if g and g != r
    ]

  return {
    "code": code,
    "status": "match" if not diffs else "mismatch",
    "diffs": diffs,
    "py_geostd_signatures": py_geostd_sig,
    "rust_geostd_signatures": rust_geostd_sig,
  }


def main() -> int:
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  parser.add_argument("--sample-size", type=int, default=_DEFAULT_SAMPLE_SIZE)
  parser.add_argument(
    "--full",
    action="store_true",
    help="Sample every discovered candidate instead of --sample-size (for a full-corpus run).",
  )
  parser.add_argument("--seed", type=int, default=_DEFAULT_SEED)
  parser.add_argument("--geostd-ref", type=str, default=_DEFAULT_REF)
  parser.add_argument("--cache-dir", type=Path, default=_DEFAULT_CACHE_DIR)
  parser.add_argument("--json-out", type=Path, default=None)
  parser.add_argument(
    "--rust-lib-dir",
    type=Path,
    default=Path(__file__).resolve().parents[2],
    help="Directory containing the built proxide_gaff2 extension module (default: repo root).",
  )
  parser.add_argument(
    "--workers",
    type=int,
    default=16,
    help="Concurrent fetch+type workers -- this is I/O-bound (network fetch per ligand), "
    "so threads (not processes) are used; a full 37k-ligand run is impractical serially.",
  )
  args = parser.parse_args()

  rust_engine = load_rust_engine(args.rust_lib_dir)

  logger.info("Discovering geostd candidates at ref %s ...", args.geostd_ref)
  candidates = discover_candidates(args.geostd_ref)
  logger.info("Found %d candidate ligands with a real .mol2.", len(candidates))

  if args.full:
    sample = candidates
    logger.info("Sampling all %d candidates (--full).", len(sample))
  else:
    sample = random.Random(args.seed).sample(candidates, k=min(args.sample_size, len(candidates)))
    logger.info("Sampling %d (seed=%d).", len(sample), args.seed)

  results: list[dict] = [{}] * len(sample)
  completed = 0
  with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
    future_to_idx = {
      executor.submit(compare_one, bucket, code, args.geostd_ref, args.cache_dir, rust_engine): i
      for i, (bucket, code) in enumerate(sample)
    }
    for future in concurrent.futures.as_completed(future_to_idx):
      idx = future_to_idx[future]
      results[idx] = future.result()
      completed += 1
      if completed % 500 == 0:
        logger.info("  ... %d/%d", completed, len(sample))

  counts: dict[str, int] = defaultdict(int)
  for r in results:
    counts[r["status"]] += 1

  logger.info("\n=== Summary (Rust vs Python) ===")
  for status in (
    "match",
    "mismatch",
    "python_error",
    "rust_error",
    "fetch_error",
    "length_mismatch",
  ):
    if counts[status]:
      logger.info("%-16s %d", status, counts[status])
  total_typed = counts["match"] + counts["mismatch"]
  match_pct = None
  if total_typed:
    match_pct = 100 * counts["match"] / total_typed
    logger.info("match rate: %.2f%% (%d/%d)", match_pct, counts["match"], total_typed)

  # Deduplicate Rust-vs-Python mismatches by (python, rust) signature --
  # a single root cause can recur across many ligands; investigate signatures.
  signatures: dict[tuple[str, str], list[str]] = defaultdict(list)
  for r in results:
    if r["status"] == "mismatch":
      for d in r["diffs"]:
        signatures[(d["python"], d["rust"])].append(r["code"])

  if signatures:
    logger.info("\n=== Mismatch signatures (python -> rust), most common first ===")
    for (py_type, rust_type), codes in sorted(signatures.items(), key=lambda kv: -len(kv[1])):
      examples = ", ".join(codes[:5])
      more = f" (+{len(codes) - 5} more)" if len(codes) > 5 else ""
      logger.info("%-8s -> %-8s  x%-4d  e.g. %s%s", py_type, rust_type, len(codes), examples, more)

  # Secondary cross-check: does Rust diverge from geostd ground truth in
  # exactly the same places Python does? Aggregate the per-ligand signature
  # lists gathered in compare_one into two sets and compare them.
  py_geostd_signature_set: set[tuple[str, str]] = set()
  rust_geostd_signature_set: set[tuple[str, str]] = set()
  for r in results:
    if r["status"] in ("match", "mismatch"):
      py_geostd_signature_set.update(tuple(p) for p in r.get("py_geostd_signatures", []))
      rust_geostd_signature_set.update(tuple(p) for p in r.get("rust_geostd_signatures", []))

  signature_set_identical = py_geostd_signature_set == rust_geostd_signature_set
  only_in_python = py_geostd_signature_set - rust_geostd_signature_set
  only_in_rust = rust_geostd_signature_set - py_geostd_signature_set

  logger.info(
    "\n=== Cross-check: Python-vs-geostd signature set == Rust-vs-geostd signature set? ==="
  )
  logger.info("python-vs-geostd signatures: %d distinct", len(py_geostd_signature_set))
  logger.info("rust-vs-geostd signatures:   %d distinct", len(rust_geostd_signature_set))
  logger.info("IDENTICAL: %s", signature_set_identical)
  if only_in_python:
    logger.info(
      "  only in python-vs-geostd (Rust FAILED to reproduce these known mismatches): %s",
      sorted(only_in_python),
    )
  if only_in_rust:
    logger.info(
      "  only in rust-vs-geostd (Rust introduces NEW mismatches Python doesn't have): %s",
      sorted(only_in_rust),
    )

  if args.json_out:
    report = {
      "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
      "match_rate_pct": match_pct,
      "counts": dict(counts),
      "signature_set_identical_to_python": signature_set_identical,
      "py_geostd_signature_set": sorted(py_geostd_signature_set),
      "rust_geostd_signature_set": sorted(rust_geostd_signature_set),
      "only_in_python": sorted(only_in_python),
      "only_in_rust": sorted(only_in_rust),
      "results": results,
    }
    args.json_out.write_text(json.dumps(report, indent=2, default=str))
    logger.info("\nFull results written to %s", args.json_out)

  return 0


if __name__ == "__main__":
  sys.exit(main())
