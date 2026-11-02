"""Validation script for Workstream C: Unknown Mass Handling (debt #3179).

Pre-registered in unknown_mass_raises.bth.toml before any run.
Verifies:
1. Tabulated elements receive correct physical masses.
2. Untabulated elements and uninferable names raise ValueError.
3. No untabulated element silently returns default mass 12.0 or 12.011.
4. GBSA parameters record ParameterSource::Fallback = 3 for unknown elements.
5. Negative control: a simulated silent-12.0 fallback is detected by the validator.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT / "src"))

log = logging.getLogger("unknown_mass_raises")

# Tabulated elements and expected masses (IUPAC conventional standard atomic weights)
TABULATED_EXPECTED = {
  "H": 1.008,
  "CA": 12.011,
  "C": 12.011,
  "N": 14.007,
  "O": 15.999,
  "S": 32.060,
  "P": 30.970,
  "F": 18.998,
  "CL": 35.450,
  "BR": 79.904,
  "I": 126.900,
  "NA": 22.990,
  "K": 39.098,
  "MG": 24.305,
  "ZN": 65.380,
  "FE": 55.845,
  "CU": 63.546,
  "MN": 54.938,
  "SE": 78.971,
  "SI": 28.085,
}

UNTABULATED_CANDIDATES = [
  "AU", "LI", "U", "AR", "XE", "RN", "AL", "BA", "BE", "B",
  "TI", "V", "GA", "GE", "AS", "QZ", "XX", "EPW", "D1", "", "1", "   ",
]


def run_checks(assign_fn, is_mock: bool = False) -> tuple[bool, int, bool, int, int]:
  """Runs tabulated and untabulated element checks against an assign_masses function."""
  # 1. Tabulated checks
  tabulated_ok = True
  names = list(TABULATED_EXPECTED.keys())
  try:
    masses = assign_fn(names)
    for name, mass in zip(names, masses, strict=True):
      if abs(mass - TABULATED_EXPECTED[name]) >= 1e-3:
        tabulated_ok = False
        log.error("Mismatch for %s: %s vs %s", name, mass, TABULATED_EXPECTED[name])
  except Exception as e:
    tabulated_ok = False
    log.error("Tabulated elements raised exception: %s", e)

  # 2. Untabulated checks
  untabulated_refused = True
  silent_12_count = 0

  for cand in UNTABULATED_CANDIDATES:
    try:
      res = assign_fn(["CA", cand])
      # If it did not raise, it's a refusal failure
      untabulated_refused = False
      if len(res) >= 2 and (abs(res[1] - 12.0) < 1e-3 or abs(res[1] - 12.011) < 1e-3):
        silent_12_count += 1
    except ValueError:
      # Expected: ValueError raised
      pass
    except Exception as e:
      log.warning("Candidate %s raised unexpected exception: %s", cand, e)
      untabulated_refused = False

  return tabulated_ok, len(names), untabulated_refused, len(UNTABULATED_CANDIDATES), silent_12_count


def check_gbsa_fallback(proxide_mod) -> bool:
  """Verifies GBSA records ParameterSource::Fallback = 3 for unknown elements."""
  try:
    r = proxide_mod.assign_mbondi2_radii_with_provenance(["QZ", "CA"], [])
    s = proxide_mod.assign_obc2_scaling_factors_with_provenance(["QZ", "CA"])

    r_sources = list(r["sources"])
    s_sources = list(s["sources"])

    # Fallback = 3
    if r_sources[0] != 3 or s_sources[0] != 3:
      log.error("GBSA sources mismatch: radius=%d, screen=%d (expected 3)", r_sources[0], s_sources[0])
      return False

    if r["all_licensed"] or 0 not in list(r["unlicensed_atoms"]):
      log.error("GBSA unlicensed flags incorrect: %s", r)
      return False

    return True
  except Exception as e:
    log.error("check_gbsa_fallback failed: %s", e)
    return False


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument("--out", type=Path, default=None, help="write results JSON here")
  args = ap.parse_args()
  logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

  import proxide

  # Run real check
  tab_ok, n_tab, untab_ok, n_untab, silent_12 = run_checks(proxide.assign_masses)
  gbsa_ok = check_gbsa_fallback(proxide)

  # Negative control: mock function simulating the pre-fix bug (silently returning 12.0)
  def buggy_mock_assign(atom_names: list[str]) -> list[float]:
    out = []
    for name in atom_names:
      if name in TABULATED_EXPECTED:
        out.append(TABULATED_EXPECTED[name])
      else:
        out.append(12.0)  # DEFAULT_MASS silent fallback
    return out

  neg_tab_ok, _, neg_untab_ok, _, neg_silent_12 = run_checks(buggy_mock_assign, is_mock=True)
  # Negative control passes if the validator DETECTS the bug (neg_untab_ok is False and neg_silent_12 > 0)
  neg_controls_passed = 1 if (not neg_untab_ok and neg_silent_12 > 0) else 0

  all_pass = tab_ok and untab_ok and (silent_12 == 0) and gbsa_ok and (neg_controls_passed >= 1)

  results = {
    "all_pass": all_pass,
    "tabulated_all_pass": tab_ok,
    "n_tabulated_tested": n_tab,
    "untabulated_all_refused": untab_ok,
    "n_untabulated_tested": n_untab,
    "silent_12_count": silent_12,
    "gbsa_fallback_correct": gbsa_ok,
    "neg_controls_passed": neg_controls_passed,
  }

  print(json.dumps(results, indent=2))

  for path in filter(None, [os.environ.get("BTH_RESULTS_PATH"), args.out]):
    Path(path).write_text(json.dumps(results, indent=2))

  return 0 if all_pass else 1


if __name__ == "__main__":
  raise SystemExit(main())
