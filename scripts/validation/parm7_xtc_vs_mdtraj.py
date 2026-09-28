"""Cross-check proxide.parse_amber_trajectory against mdtraj on a real AMBER parm7 + XTC.

mdtraj is an independent reader of both formats, so agreement on residue sequence,
chain breaks and CA coordinates validates the parm7 reader, the topology/frame
assembly and the frame selection together. mdtraj is only used here, as a reference;
it is not a proxide dependency.

Checks (all must hold):
  * protein residue count and one-letter sequence equal mdtraj's protein residues;
  * proxide chain boundaries equal the backbone breaks mdtraj's coordinates imply
    (C(i)-N(i+1) > 2 A), an independent chain definition;
  * per selected frame, CA coordinates match mdtraj's to < --ca-tol A;
  * NEGATIVE CONTROL: proxide frame 0 vs mdtraj's LAST frame must exceed
    --negative-min A, i.e. the coordinate check can fail;
  * the selected frames differ from each other (frame selection is not a no-op).

Usage:
  python scripts/validation/parm7_xtc_vs_mdtraj.py --topology X.parm7 --trajectory X.xtc \
      --frames 0 -1 --out result.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

log = logging.getLogger("parm7_xtc_vs_mdtraj")

CA = 1  # atom37 index of CA
PEPTIDE_BREAK_A = 2.0


def mdtraj_reference(topology: str, trajectory: str, frames: list[int]) -> dict:
  import mdtraj as md  # noqa: PLC0415

  n_frames = md.formats.XTCTrajectoryFile(trajectory).__len__()
  resolved = [f if f >= 0 else n_frames + f for f in frames]
  last = n_frames - 1
  loaded = {i: md.load_frame(trajectory, i, top=topology) for i in sorted({*resolved, last})}
  top = loaded[resolved[0]].topology
  residues = [r for r in top.residues if r.is_protein]
  seq = "".join(r.code or "X" for r in residues)

  def atom_index(res, name):
    matches = [a.index for a in res.atoms if a.name == name]
    return matches[0] if matches else None

  ca_idx = [atom_index(r, "CA") for r in residues]
  c_idx = [atom_index(r, "C") for r in residues]
  n_idx = [atom_index(r, "N") for r in residues]
  xyz = {i: t.xyz[0] * 10.0 for i, t in loaded.items()}  # nm -> A
  first = xyz[resolved[0]]
  breaks = [
    k + 1
    for k in range(len(residues) - 1)
    if np.linalg.norm(first[c_idx[k]] - first[n_idx[k + 1]]) > PEPTIDE_BREAK_A
  ]
  return {
    "n_frames": n_frames,
    "frames": resolved,
    "sequence": seq,
    "chain_starts": [0, *breaks],
    "ca": {i: xyz[i][ca_idx] for i in xyz},
    "last": last,
  }


def proxide_result(topology: str, trajectory: str, frames: list[int]) -> dict:
  import proxide  # noqa: PLC0415

  proteins = proxide.parse_amber_trajectory(topology, trajectory, frames=list(frames), use_jax=False)
  coords = np.stack([np.asarray(p.coordinates) for p in proteins])  # (F, R, 37, 3)
  protein = proteins[0]
  aatype = np.asarray(protein.aatype)
  chain = np.asarray(protein.chain_index)
  # Rust aatype indices follow proxide-core RESTYPES (AlphaFold order); 20 = unknown.
  alphabet = "ARNDCQEGHILKMFPSTWYV"
  seq = "".join(alphabet[a] if a < len(alphabet) else "X" for a in aatype)
  starts = [0, *[k for k in range(1, len(chain)) if chain[k] != chain[k - 1]]]
  return {"sequence": seq, "chain_starts": starts, "ca": coords[:, :, CA, :], "module": proxide.__file__}


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument("--topology", required=True)
  ap.add_argument("--trajectory", required=True)
  ap.add_argument("--frames", type=int, nargs="+", default=[0, -1])
  ap.add_argument("--ca-tol", type=float, default=0.02)
  ap.add_argument("--negative-min", type=float, default=0.5)
  ap.add_argument("--out", type=Path, required=True)
  args = ap.parse_args()
  logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

  ref = mdtraj_reference(args.topology, args.trajectory, args.frames)
  got = proxide_result(args.topology, args.trajectory, args.frames)
  log.info("proxide module: %s", got["module"])

  checks: dict[str, dict] = {}
  checks["sequence_equal"] = {
    "pass": got["sequence"] == ref["sequence"],
    "n_res_proxide": len(got["sequence"]),
    "n_res_mdtraj": len(ref["sequence"]),
    "first_mismatch": next(
      (i for i, (a, b) in enumerate(zip(got["sequence"], ref["sequence"], strict=False)) if a != b), None
    ),
  }
  checks["chain_starts_equal"] = {
    "pass": got["chain_starts"] == ref["chain_starts"],
    "proxide": got["chain_starts"],
    "mdtraj_backbone_breaks": ref["chain_starts"],
  }
  per_frame = []
  for pos, frame in enumerate(ref["frames"]):
    if got["ca"].shape[1] != ref["ca"][frame].shape[0]:
      per_frame.append({"frame": frame, "max_abs_diff": None})
      continue
    diff = float(np.nanmax(np.abs(got["ca"][pos] - ref["ca"][frame])))
    per_frame.append({"frame": frame, "max_abs_diff": diff})
  checks["ca_match"] = {
    "pass": all(f["max_abs_diff"] is not None and f["max_abs_diff"] < args.ca_tol for f in per_frame),
    "tol": args.ca_tol,
    "frames": per_frame,
  }
  neg = None
  if got["ca"].shape[1] == ref["ca"][ref["last"]].shape[0] and ref["frames"][0] != ref["last"]:
    neg = float(np.nanmax(np.abs(got["ca"][0] - ref["ca"][ref["last"]])))
  checks["negative_control_fires"] = {
    "pass": neg is not None and neg > args.negative_min,
    "max_abs_diff_frame0_vs_last": neg,
    "min_required": args.negative_min,
  }
  distinct = None
  if got["ca"].shape[0] > 1:
    distinct = float(np.sqrt(np.nanmean(np.sum((got["ca"][0] - got["ca"][-1]) ** 2, axis=-1))))
  checks["selected_frames_distinct"] = {"pass": distinct is not None and distinct > 0.1, "ca_rmsd": distinct}

  overall = all(c["pass"] for c in checks.values())
  result = {"pass": overall, "n_frames_in_file": ref["n_frames"], "frames": ref["frames"], "checks": checks}
  args.out.write_text(json.dumps(result, indent=2))
  diffs = [f["max_abs_diff"] for f in per_frame]
  flat = {
    "all_pass": overall,
    "sequence_equal": checks["sequence_equal"]["pass"],
    "chain_starts_equal": checks["chain_starts_equal"]["pass"],
    "n_res_proxide": len(got["sequence"]),
    "n_res_mdtraj": len(ref["sequence"]),
    "n_chains_proxide": len(got["chain_starts"]),
    # Missing comparisons count as infinitely wrong, never as a pass.
    "ca_max_abs_diff": max(d if d is not None else float("inf") for d in diffs),
    "negative_max_abs_diff": neg if neg is not None else -1.0,
    "frames_ca_rmsd": distinct if distinct is not None else -1.0,
  }
  results_path = os.environ.get("BTH_RESULTS_PATH")
  if results_path:
    Path(results_path).write_text(json.dumps(flat))
  for name, c in checks.items():
    log.info("%-26s %s", name, "PASS" if c["pass"] else "FAIL")
  log.info("overall: %s -> %s", "PASS" if overall else "FAIL", args.out)
  return 0 if overall else 1


if __name__ == "__main__":
  sys.exit(main())
