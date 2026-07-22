"""Benchmark XTC trajectory decode: mdtraj vs proxide's lazy/parallel readers.

Compares wall-clock decode time across three paths on a synthetic,
production-scale XTC trajectory:

  (a) mdtraj.load(path, stride=..., atom_indices=...)      — today's baseline
  (b) proxide.read_xtc_lazy(...)                            — single-threaded,
      offset-cache-backed XtcReader cursor
  (c) proxide.read_xtc_parallel(...)                        — read_frames_parallel,
      per-worker file handles

Cold (no `<path>.offsets` sidecar) and warm (sidecar already exists) timings
are reported *separately* for (b) and (c) — the offset cache's entire value
proposition is repeat-open speed, so collapsing cold/warm into one number
would hide the finding.

The synthetic fixture is generated fresh into a temp directory on every run
and is never checked into git (a 20k-frame x 5k-atom trajectory is far too
large for that — see `scripts/generate_trajectory_test_data.py`'s
`generate_large_xtc_fixture`, which checks in a much smaller 5000x4 fixture).

Per this repo's local-compute-limits convention, this is a single narrow
script (not a full suite) safe to run locally; thread env vars are capped
below so the mdtraj baseline is a fair single-threaded comparison.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any


def _build_synthetic_xtc(path: Path, n_frames: int, n_atoms: int, seed: int = 1337):
  """Write a synthetic n_frames x n_atoms XTC trajectory to `path`.

  Same construction as tests/validation/test_xtc_reader_parity.py's fixture
  (fixed ALA-chain base topology + seeded jitter noise, constant box),
  scaled to benchmark size. Returns the mdtraj.Topology used.
  """
  import mdtraj
  import numpy as np

  topology = mdtraj.Topology()
  chain = topology.add_chain()
  n_residues = max(1, n_atoms // 4)
  prev_c = None
  for _ in range(n_residues):
    residue = topology.add_residue("ALA", chain)
    n_atom = topology.add_atom("N", mdtraj.element.nitrogen, residue)
    ca_atom = topology.add_atom("CA", mdtraj.element.carbon, residue)
    c_atom = topology.add_atom("C", mdtraj.element.carbon, residue)
    o_atom = topology.add_atom("O", mdtraj.element.oxygen, residue)
    topology.add_bond(n_atom, ca_atom)
    topology.add_bond(ca_atom, c_atom)
    topology.add_bond(c_atom, o_atom)
    if prev_c is not None:
      topology.add_bond(prev_c, n_atom)
    prev_c = c_atom

  actual_n_atoms = topology.n_atoms
  base_coords = np.zeros((actual_n_atoms, 3), dtype=np.float32)
  within_residue = np.array(
    [[0.00, 0.00, 0.00], [0.15, 0.00, 0.00], [0.25, 0.10, 0.00], [0.25, 0.20, 0.00]],
    dtype=np.float32,
  )
  for i in range(actual_n_atoms):
    residue_idx, offset = divmod(i, 4)
    base_coords[i] = within_residue[offset] + np.array(
      [residue_idx * 0.4, 0.0, 0.0], dtype=np.float32
    )

  rng = np.random.default_rng(seed)
  coords = (
    base_coords[None, :, :]
    + rng.standard_normal((n_frames, actual_n_atoms, 3)).astype(np.float32) * 0.01
  )

  unitcell_lengths = np.array([[10.0, 10.0, 10.0]] * n_frames, dtype=np.float32)
  unitcell_angles = np.array([[90.0, 90.0, 90.0]] * n_frames, dtype=np.float32)

  traj = mdtraj.Trajectory(
    coords, topology, unitcell_lengths=unitcell_lengths, unitcell_angles=unitcell_angles
  )
  traj.save_xtc(str(path))
  return topology, actual_n_atoms


def _time_once(fn) -> float:
  start = time.perf_counter()
  fn()
  return time.perf_counter() - start


def _offsets_sidecar(path: Path) -> Path:
  return path.with_name(path.name + ".offsets")


def _bench_mdtraj(path: Path, topology, stride: int, repeats: int) -> list[float]:
  import mdtraj

  return [
    _time_once(lambda: mdtraj.load(str(path), top=topology, stride=stride)) for _ in range(repeats)
  ]


def _bench_proxide(fn_name: str, path: Path, stride: int, repeats: int) -> dict[str, list[float]]:
  import proxide

  fn = getattr(proxide, fn_name)
  sidecar = _offsets_sidecar(path)

  cold_times = []
  for _ in range(repeats):
    sidecar.unlink(missing_ok=True)
    cold_times.append(_time_once(lambda: fn(str(path), stride=stride)))

  # sidecar now exists (written by the last cold call) — measure warm re-opens.
  warm_times = [_time_once(lambda: fn(str(path), stride=stride)) for _ in range(repeats)]

  return {"cold_s": cold_times, "warm_s": warm_times}


def _summary(values: list[float]) -> dict[str, float]:
  return {"median_s": statistics.median(values), "min_s": min(values), "max_s": max(values)}


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--n-frames", type=int, default=20_000)
  parser.add_argument("--n-atoms", type=int, default=5_000)
  parser.add_argument("--stride", type=int, default=1)
  parser.add_argument("--repeats", type=int, default=3)
  parser.add_argument("--json-out", type=Path, default=Path("bench_xtc_decode.json"))
  parser.add_argument(
    "--fixture-dir",
    type=Path,
    default=None,
    help="Directory to generate the synthetic XTC into (default: a fresh tempdir, "
    "cleaned up after the run — never checked into git).",
  )
  args = parser.parse_args()

  with tempfile.TemporaryDirectory(prefix="proxide-xtc-bench-") as tmp:
    fixture_dir = args.fixture_dir or Path(tmp)
    fixture_dir.mkdir(parents=True, exist_ok=True)
    xtc_path = fixture_dir / "bench.xtc"

    print(f"Generating synthetic XTC: {args.n_frames} frames x {args.n_atoms} atoms ...")
    gen_start = time.perf_counter()
    topology, actual_n_atoms = _build_synthetic_xtc(xtc_path, args.n_frames, args.n_atoms)
    print(
      f"  wrote {xtc_path} ({xtc_path.stat().st_size / 1e6:.1f} MB, "
      f"{actual_n_atoms} atoms) in {time.perf_counter() - gen_start:.1f}s"
    )

    print("Benchmarking mdtraj.load (baseline) ...")
    mdtraj_times = _bench_mdtraj(xtc_path, topology, args.stride, args.repeats)

    print("Benchmarking proxide.read_xtc_lazy (cold + warm) ...")
    lazy_times = _bench_proxide("read_xtc_lazy", xtc_path, args.stride, args.repeats)

    print("Benchmarking proxide.read_xtc_parallel (cold + warm) ...")
    parallel_times = _bench_proxide("read_xtc_parallel", xtc_path, args.stride, args.repeats)

  payload: dict[str, Any] = {
    "n_frames": args.n_frames,
    "n_atoms": actual_n_atoms,
    "stride": args.stride,
    "repeats": args.repeats,
    "mdtraj": {"seconds": mdtraj_times, "summary": _summary(mdtraj_times)},
    "read_xtc_lazy": {
      "cold": {"seconds": lazy_times["cold_s"], "summary": _summary(lazy_times["cold_s"])},
      "warm": {"seconds": lazy_times["warm_s"], "summary": _summary(lazy_times["warm_s"])},
    },
    "read_xtc_parallel": {
      "cold": {
        "seconds": parallel_times["cold_s"],
        "summary": _summary(parallel_times["cold_s"]),
      },
      "warm": {
        "seconds": parallel_times["warm_s"],
        "summary": _summary(parallel_times["warm_s"]),
      },
    },
  }
  args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

  print()
  print(f"{'path':<28}{'median_s':>12}{'min_s':>12}{'max_s':>12}")
  print(
    f"{'mdtraj (baseline)':<28}{_summary(mdtraj_times)['median_s']:>12.4f}"
    f"{_summary(mdtraj_times)['min_s']:>12.4f}{_summary(mdtraj_times)['max_s']:>12.4f}"
  )
  for name, times in (("read_xtc_lazy", lazy_times), ("read_xtc_parallel", parallel_times)):
    for phase in ("cold_s", "warm_s"):
      s = _summary(times[phase])
      label = f"{name} ({phase[:-2]})"
      print(f"{label:<28}{s['median_s']:>12.4f}{s['min_s']:>12.4f}{s['max_s']:>12.4f}")
  print()
  print(f"wrote {args.json_out}")


if __name__ == "__main__":
  main()
