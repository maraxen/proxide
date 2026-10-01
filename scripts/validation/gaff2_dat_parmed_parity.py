"""Full-table parity: proxide's GAFF2 .dat loader vs ParmEd's AmberParameterSet.

Pre-registered in gaff2_dat_parmed_parity.bth.toml (task
261001_proxide-debt-sweep-2, debt #2368) before any run of this script.
Compares every mass, bond, angle, torsion term, improper and van der Waals
entry that proxide.chem.gaff2.load_gaff2_parameters reads from the pinned
gaff-2.2.20.dat against ParmEd -- an independent reader of the AMBER parm.dat
format -- on the identical file.

Negative control: the same comparison on a copy of the .dat with exactly one
angle force constant changed (c3-c-c3 59.15 -> 60.15) must report >= 1 angle
mismatch, or the instrument cannot detect a wrong value.

Requires ParmEd in the interpreter (not a project dependency): run with an
environment that has it, e.g. `uv pip install parmed` into a scratch venv.

Usage:
  bth run --project-slug proxide -- uv run --no-sync python3 \
      scripts/validation/gaff2_dat_parmed_parity.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from pathlib import Path

log = logging.getLogger("gaff2_dat_parmed_parity")

_REPO = Path(__file__).resolve().parents[2]
_DAT = _REPO / "src" / "proxide" / "assets" / "gaff" / "dat" / "gaff-2.2.20.dat"
_TOL = 1e-6


def _canon_improper(key: tuple[str, ...]) -> tuple[str, ...]:
  """Improper key with the central (third) atom fixed and the others sorted."""
  outer = sorted((key[0], key[1], key[3]))
  return (outer[0], outer[1], key[2], outer[2])


def _close(a: float, b: float) -> bool:
  return abs(a - b) <= _TOL


def compare(dat: Path) -> dict:
  import parmed

  from proxide.chem.gaff2 import load_gaff2_parameters

  ps = parmed.amber.AmberParameterSet(str(dat))
  px = load_gaff2_parameters(dat)
  out: dict = {}

  # Masses and vdW (per atom type).
  pm_mass = {t: a.mass for t, a in ps.atom_types.items()}
  out["mass_mismatch"] = sum(
    1 for t, m in pm_mass.items() if t not in px["masses"] or not _close(px["masses"][t], m)
  ) + len(set(px["masses"]) - set(pm_mass))
  pm_lj = {t: (a.rmin, a.epsilon) for t, a in ps.atom_types.items() if a.rmin is not None}
  out["vdw_mismatch"] = sum(
    1
    for t, (r, e) in pm_lj.items()
    if t not in px["vdw"] or not (_close(px["vdw"][t][0], r) and _close(px["vdw"][t][1], e))
  ) + len(set(px["vdw"]) - set(pm_lj))

  def table(pm: dict, mine: dict, n: int, values) -> tuple[int, int]:
    pm_c = {min(k, k[::-1]): values(v) for k, v in pm.items()}
    mine_c: dict = {}
    for k, v in mine.items():
      mine_c[min(k, k[::-1])] = v
    bad = sum(
      1
      for k, ref in pm_c.items()
      if k not in mine_c or not all(_close(x, y) for x, y in zip(mine_c[k], ref, strict=True))
    ) + len(set(mine_c) - set(pm_c))
    return bad, len(pm_c)

  out["bond_mismatch"], out["n_bonds"] = table(
    ps.bond_types, px["bonds"], 2, lambda b: (b.k, b.req)
  )
  out["angle_mismatch"], out["n_angles"] = table(
    ps.angle_types, px["angles"], 3, lambda a: (a.k, a.theteq)
  )

  # Torsions: compare the sorted list of (periodicity, barrier, phase) terms.
  pm_t = {
    min(k, k[::-1]): sorted((int(t.per), t.phi_k, t.phase) for t in v)
    for k, v in ps.dihedral_types.items()
  }
  px_t = {min(k, k[::-1]): sorted(v) for k, v in px["torsions"].items()}

  def same_terms(a: list, b: list) -> bool:
    return len(a) == len(b) and all(
      x[0] == y[0] and _close(x[1], y[1]) and _close(x[2], y[2])
      for x, y in zip(a, b, strict=True)
    )

  out["torsion_mismatch"] = sum(
    1 for k, ref in pm_t.items() if k not in px_t or not same_terms(px_t[k], ref)
  ) + len(set(px_t) - set(pm_t))
  out["n_torsion_keys"] = len(pm_t)

  pm_i = {_canon_improper(k): (t.phi_k, t.phase) for k, t in ps.improper_periodic_types.items()}
  px_i = {_canon_improper(k): v for k, v in px["impropers"].items()}
  out["improper_mismatch"] = sum(
    1
    for k, (kk, ph) in pm_i.items()
    if k not in px_i or not (_close(px_i[k][0], kk) and _close(px_i[k][1], ph))
  ) + len(set(px_i) - set(pm_i))
  out["n_impropers"] = len(pm_i)
  out["parmed_version"] = parmed.__version__
  return out


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument("--dat", type=Path, default=_DAT)
  ap.add_argument("--out", type=Path, default=None, help="also write results JSON here")
  args = ap.parse_args()
  logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

  real = compare(args.dat)
  log.info("real: %s", real)

  # Negative control: one angle constant changed must be detected.
  lines = args.dat.read_text().split("\n")
  idx = next(i for i, ln in enumerate(lines) if ln.startswith("c3-c -c3"))
  assert "59.15" in lines[idx], lines[idx]
  lines[idx] = lines[idx].replace("59.15", "60.15", 1)
  with tempfile.TemporaryDirectory() as d:
    tampered = Path(d) / "gaff-2.2.20-tampered.dat"
    tampered.write_text("\n".join(lines))
    import parmed

    # Compare proxide-on-tampered against ParmEd-on-original.
    from proxide.chem.gaff2 import load_gaff2_parameters

    pm = parmed.amber.AmberParameterSet(str(args.dat))
    px = load_gaff2_parameters(tampered)
    neg = sum(
      1
      for k, a in pm.angle_types.items()
      if (k in px["angles"] and not _close(px["angles"][k][0], a.k))
    )
  log.info("negative control angle mismatches (expect >= 1): %d", neg)

  mismatch_keys = [k for k in real if k.endswith("_mismatch")]
  results = {
    **real,
    "negative_control_angle_mismatch": neg,
    "all_pass": all(real[k] == 0 for k in mismatch_keys) and neg >= 1,
  }
  print(json.dumps(results, indent=2))
  for path in filter(None, [os.environ.get("BTH_RESULTS_PATH"), args.out]):
    Path(path).write_text(json.dumps(results, indent=2))
  return 0 if results["all_pass"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
