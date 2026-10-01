"""Full-table parity: proxide's GAFF2 .dat loader vs ParmEd's AmberParameterSet.

Pre-registered in gaff2_dat_parmed_parity.bth.toml (task
261001_proxide-debt-sweep-2, debt #2368) before any run of this version.
Compares every mass, bond, angle, torsion term, improper and van der Waals
entry that proxide.chem.gaff2.load_gaff2_parameters reads from the pinned
gaff-2.2.20.dat against ParmEd -- an independent reader of the AMBER parm.dat
format -- on the identical file.

Negative controls (v2, after review #7 of this sprint): the SAME compare()
is run with proxide reading a tampered copy -- one value changed in each
table (mass, vdW, bond, angle, torsion term, improper) plus one angle row
deleted -- against ParmEd on the original. Each tamper must produce >= 1
mismatch in its own table, so a comparison that silently passes a table
cannot pass the run. (v1's control re-implemented the angle check inline
and exercised nothing else.)

Requires ParmEd in the interpreter (not a project dependency).

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

# One tamper per table: (table, line prefix, old text, new text). Rows and
# values verified against gaff-2.2.20.dat; None as new text deletes the row.
_TAMPERS = [
  ("mass", "c3 12.01", "12.01", "12.11"),
  ("vdw", "  c3          1.9069", "0.1078", "0.2078"),
  ("bond", "c3-c3  228.89", "228.89", "238.89"),
  ("angle", "c3-c -c3   59.15", "59.15", "60.15"),
  ("angle_row_deleted", "c3-c -o    76.45", None, None),
  ("torsion", "X -c -c -X    4    1.200", "1.200", "1.300"),
  ("improper", "X -X -c -o          10.5", "10.5", "11.5"),
]
_TABLE_KEY = {
  "mass": "mass_mismatch", "vdw": "vdw_mismatch", "bond": "bond_mismatch",
  "angle": "angle_mismatch", "angle_row_deleted": "angle_mismatch",
  "torsion": "torsion_mismatch", "improper": "improper_mismatch",
}


def _canon_improper(key: tuple[str, ...]) -> tuple[str, ...]:
  """Improper key with the central (third) atom fixed and the others sorted."""
  outer = sorted((key[0], key[1], key[3]))
  return (outer[0], outer[1], key[2], outer[2])


def _close(a: float, b: float) -> bool:
  return abs(a - b) <= _TOL


def compare(ps, px: dict) -> dict:
  """Mismatch counts between a ParmEd parameter set and proxide's tables."""
  out: dict = {}

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

  def table(pm: dict, mine: dict, values) -> tuple[int, int]:
    pm_c = {min(k, k[::-1]): values(v) for k, v in pm.items()}
    mine_c = {min(k, k[::-1]): v for k, v in mine.items()}
    bad = sum(
      1
      for k, ref in pm_c.items()
      if k not in mine_c or not all(_close(x, y) for x, y in zip(mine_c[k], ref, strict=True))
    ) + len(set(mine_c) - set(pm_c))
    return bad, len(pm_c)

  out["bond_mismatch"], out["n_bonds"] = table(ps.bond_types, px["bonds"], lambda b: (b.k, b.req))
  out["angle_mismatch"], out["n_angles"] = table(
    ps.angle_types, px["angles"], lambda a: (a.k, a.theteq)
  )

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
  return out


def _tampered(lines: list[str], prefix: str, old: str | None, new: str | None) -> list[str]:
  idx = [i for i, ln in enumerate(lines) if ln.startswith(prefix)]
  if len(idx) != 1:
    raise SystemExit(f"tamper target {prefix!r} matched {len(idx)} rows, expected exactly 1")
  out = list(lines)
  if old is None:
    del out[idx[0]]
  else:
    if old not in out[idx[0]]:
      raise SystemExit(f"tamper value {old!r} not in row {out[idx[0]]!r}")
    out[idx[0]] = out[idx[0]].replace(old, new, 1)
  return out


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument("--dat", type=Path, default=_DAT)
  ap.add_argument("--out", type=Path, default=None, help="also write results JSON here")
  args = ap.parse_args()
  logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

  import parmed

  from proxide.chem.gaff2 import load_gaff2_parameters

  ps = parmed.amber.AmberParameterSet(str(args.dat))
  real = compare(ps, load_gaff2_parameters(args.dat))
  log.info("real: %s", real)

  lines = args.dat.read_text().split("\n")
  controls: dict[str, int] = {}
  with tempfile.TemporaryDirectory() as d:
    for name, prefix, old, new in _TAMPERS:
      path = Path(d) / f"tampered_{name}.dat"
      path.write_text("\n".join(_tampered(lines, prefix, old, new)))
      got = compare(ps, load_gaff2_parameters(path))
      controls[name] = got[_TABLE_KEY[name]]
      log.info("negative control %s: %s = %d (expect >= 1)", name, _TABLE_KEY[name], controls[name])

  mismatch_keys = [k for k in real if k.endswith("_mismatch")]
  results = {
    **real,
    **{f"neg_{k}": v for k, v in controls.items()},
    "negative_controls_min": min(controls.values()),
    "parmed_version": parmed.__version__,
    "all_pass": all(real[k] == 0 for k in mismatch_keys) and min(controls.values()) >= 1,
  }
  print(json.dumps(results, indent=2))
  for path in filter(None, [os.environ.get("BTH_RESULTS_PATH"), args.out]):
    Path(path).write_text(json.dumps(results, indent=2))
  return 0 if results["all_pass"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
