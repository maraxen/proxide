"""Bulk-sample proxide's GAFF2 atom typing against AmberTools' real geostd ligand corpus.

`phenix-project/geostd` on GitHub is AmberTools' own ligand-restraint
database: 37,469 ligand codes each ship a real, antechamber-generated
GAFF2 `.mol2` (verified 260820 by walking all 36 alphanumeric bucket
subtrees via the GitHub API -- not estimated; see
`.praxia/docs/audits/260820_gaff2-parity-verdict.md`). Unlike
`gaff2_external_reference.py`'s 21 hand-authored molecules (each picked to
exercise a *specific* rule) and `TestFullRuleCoverage`'s 47 hand-authored
molecules (one per previously-untested atom type), this script draws a
random sample of *real* ligands and diffs proxide's output against geostd's
embedded ground-truth types -- catching rule *combinations* that no one
would think to hand-author a test for.

This is a discovery tool, not a locked-in regression gate: it does not
exit non-zero on a mismatch (unlike `gaff2_external_reference.py`). It
reports a match rate and a deduplicated list of mismatch *signatures*
(distinct (reference_type, proxide_type) pairs, each with a few example
ligand codes) for manual triage -- a single root cause can appear in dozens
of sampled ligands, and investigating signatures instead of raw atoms
keeps a triage pass tractable.

Mol2 parsing note: real geostd files put GAFF2 type tokens (e.g. "c3",
"ns") in the SYBYL-atom-type column instead of real SYBYL types (e.g.
"C.3"), so RDKit's native `Chem.MolFromMol2File` cannot parse them --
confirmed via a live spike (fails with "Element 'c3' not found"). This
script instead uses proxide's own `Molecule.from_mol2` (which already
extracts GAFF2 types as ground truth AND builds the RDKit structure via
`_to_rdkit()`), which required a precursor fix (this same PR) to correctly
carry TRIPOS "ar" bonds through as real RDKit aromaticity instead of
flattening them to a Kekule-alternation-free single-bonded ring.

Usage:
    uv run python scripts/validation/gaff2_geostd_sample.py
    uv run python scripts/validation/gaff2_geostd_sample.py --sample-size 1000 --seed 7
    uv run python scripts/validation/gaff2_geostd_sample.py --json-out report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

from proxide.chem.gaff2 import assign_gaff2_atom_types
from proxide.io.parsing.molecule import Molecule

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("gaff2_geostd_sample")

_REPO = "phenix-project/geostd"
_DEFAULT_REF = "eaf8906a478e0fee4e6e6d3d3b2854d0fa13eabb"  # master, captured 2026-08-20
_DEFAULT_SAMPLE_SIZE = 500
_DEFAULT_SEED = 42
_DEFAULT_CACHE_DIR = Path(".cache/geostd")
# Non-ligand infrastructure directories in the geostd tree (restraint
# dictionaries/scripts/docs, not per-ligand entries).
_NON_LIGAND_DIRS = {"contrib", "list", "rna_dna", "validate"}


def _api_request(url: str) -> dict:
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read())


def discover_candidates(ref: str) -> list[tuple[str, str]]:
    """Return [(bucket, ligand_code), ...] for every real (non-.min) .mol2 in geostd."""
    top = _api_request(f"https://api.github.com/repos/{_REPO}/git/trees/{ref}")
    buckets = [
        (entry["path"], entry["sha"])
        for entry in top["tree"]
        if entry["type"] == "tree" and entry["path"] not in _NON_LIGAND_DIRS
    ]

    candidates: list[tuple[str, str]] = []
    for bucket, sha in buckets:
        subtree = _api_request(f"https://api.github.com/repos/{_REPO}/git/trees/{sha}")
        for entry in subtree["tree"]:
            path = entry["path"]
            if path.endswith(".mol2") and not path.endswith(".min.mol2"):
                candidates.append((bucket, path[: -len(".mol2")]))
    return candidates


def fetch_mol2(bucket: str, code: str, ref: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"{code}.mol2"
    if target.exists():
        return target
    url = f"https://raw.githubusercontent.com/{_REPO}/{ref}/{bucket}/{code}.mol2"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response, open(target, "wb") as out_file:
        out_file.write(response.read())
    return target


def compare_one(bucket: str, code: str, ref: str, cache_dir: Path) -> dict:
    """Fetch, parse, type, and compare one ligand. Never raises -- errors are returned."""
    try:
        path = fetch_mol2(bucket, code, ref, cache_dir)
    except urllib.error.HTTPError as e:
        return {"code": code, "status": "fetch_error", "detail": str(e)}

    try:
        molecule = Molecule.from_mol2(path)
        rdmol = molecule._to_rdkit()  # noqa: SLF001 -- this script IS the intended internal caller
        proxide_types = assign_gaff2_atom_types(rdmol)
    except Exception as e:  # noqa: BLE001 -- bulk external-corpus sample: log and skip, don't crash the run
        return {"code": code, "status": "parse_error", "detail": f"{type(e).__name__}: {e}"}

    reference_types = molecule.atom_types
    if len(reference_types) != len(proxide_types):
        return {
            "code": code,
            "status": "length_mismatch",
            "detail": f"{len(reference_types)} reference vs {len(proxide_types)} proxide",
        }

    diffs = [
        {"atom_idx": i, "element": molecule.elements[i], "reference": r, "proxide": p}
        for i, (r, p) in enumerate(zip(reference_types, proxide_types, strict=True))
        if r != p
    ]
    return {"code": code, "status": "match" if not diffs else "mismatch", "diffs": diffs}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=_DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED)
    parser.add_argument("--geostd-ref", type=str, default=_DEFAULT_REF)
    parser.add_argument("--cache-dir", type=Path, default=_DEFAULT_CACHE_DIR)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    logger.info("Discovering geostd candidates at ref %s ...", args.geostd_ref)
    candidates = discover_candidates(args.geostd_ref)
    logger.info("Found %d candidate ligands with a real .mol2.", len(candidates))

    sample = random.Random(args.seed).sample(candidates, k=min(args.sample_size, len(candidates)))
    logger.info("Sampling %d (seed=%d).", len(sample), args.seed)

    results = []
    for i, (bucket, code) in enumerate(sample, start=1):
        results.append(compare_one(bucket, code, args.geostd_ref, args.cache_dir))
        if i % 50 == 0:
            logger.info("  ... %d/%d", i, len(sample))

    counts: dict[str, int] = defaultdict(int)
    for r in results:
        counts[r["status"]] += 1

    logger.info("\n=== Summary ===")
    for status in ("match", "mismatch", "parse_error", "fetch_error", "length_mismatch"):
        if counts[status]:
            logger.info("%-16s %d", status, counts[status])
    total_typed = counts["match"] + counts["mismatch"]
    if total_typed:
        match_pct = 100 * counts["match"] / total_typed
        logger.info("match rate: %.2f%% (%d/%d)", match_pct, counts["match"], total_typed)

    # Deduplicate mismatches by (reference, proxide) signature -- a single
    # root cause can recur across many ligands; investigate signatures.
    signatures: dict[tuple[str, str], list[str]] = defaultdict(list)
    for r in results:
        if r["status"] == "mismatch":
            for d in r["diffs"]:
                signatures[(d["reference"], d["proxide"])].append(r["code"])

    if signatures:
        logger.info("\n=== Mismatch signatures (reference -> proxide), most common first ===")
        for (ref_type, prox_type), codes in sorted(signatures.items(), key=lambda kv: -len(kv[1])):
            examples = ", ".join(codes[:5])
            more = f" (+{len(codes) - 5} more)" if len(codes) > 5 else ""
            logger.info(
                "%-8s -> %-8s  x%-4d  e.g. %s%s", ref_type, prox_type, len(codes), examples, more
            )

    if args.json_out:
        report = {
            "args": vars(args) | {"cache_dir": str(args.cache_dir), "json_out": str(args.json_out)},
            "results": results,
        }
        args.json_out.write_text(json.dumps(report, indent=2, default=str))
        logger.info("\nFull results written to %s", args.json_out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
