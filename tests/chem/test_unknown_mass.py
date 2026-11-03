"""Tests verifying that unknown and untabulated elements do not silently return default mass 12.0.

Debt #3179 (Workstream C):
- get_mass returns Option<f32>, eliminating DEFAULT_MASS = 12.0
- assign_masses raises ValueError for names with no inferable element or untabulated mass
- OpenMM projection and GAFF2 parameterization refuse unknown elements/types with ValueError
- GBSA records ParameterSource::Fallback = 3 for unknown elements
"""

from __future__ import annotations

import proxide
import pytest


def test_assign_masses_tabulated_elements():
  """Tabulated elements receive their accurate physical atomic weights."""
  names = ["H", "CA", "C", "N", "O", "S", "P", "F", "CL", "BR", "I", "NA", "K", "MG", "ZN", "FE", "CU", "MN", "SE", "SI"]
  masses = proxide.assign_masses(names)
  assert len(masses) == len(names)

  # Check standard expected masses
  expected = {
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
  for name, mass in zip(names, masses, strict=True):
    assert abs(mass - expected[name]) < 1e-3, f"Mismatch for {name}: {mass} vs {expected[name]}"


@pytest.mark.parametrize(
  "unknown_name",
  ["", "   ", "1", "QZ", "XX", "D1", "EPW", "AU", "LI", "U", "AR", "XE", "RN", "AL", "BA", "BE", "B", "TI", "V", "GA", "GE", "AS"],
)
def test_assign_masses_refuses_unknown_or_untabulated_names(unknown_name: str):
  """Untabulated element names or uninferable names MUST raise ValueError, never silently returning 12.0."""
  with pytest.raises(ValueError) as excinfo:
    proxide.assign_masses(["CA", unknown_name])

  err = str(excinfo.value)
  # Offending name and index must be identified in the error message
  assert "index 1" in err, f"Error should report index 1: {err}"


def test_assign_masses_never_silently_defaults_to_12():
  """Explicit verification that untabulated elements do not yield carbon mass 12.0 or 12.011."""
  test_candidates = ["AU", "LI", "U", "AR", "XE", "RN", "AL", "BA", "BE", "B", "TI", "V", "GA", "GE", "AS", "QZ", "XX", "EPW", "D1"]
  for cand in test_candidates:
    with pytest.raises(ValueError):
      res = proxide.assign_masses([cand])
      # Even if an exception wasn't caught (should not happen), explicitly guard against 12.0
      assert res[0] not in (12.0, 12.011), f"{cand} silently returned default mass {res[0]}"


def test_gbsa_records_fallback_for_unknown_element():
  """GBSA assign_mbondi2_radii_with_provenance records ParameterSource::Fallback = 3."""
  res = proxide.assign_mbondi2_radii_with_provenance(["QZ", "CA"], [])
  sources = list(res["sources"])
  assert sources[0] == 3, f"Expected ParameterSource::Fallback (3), got {sources[0]}"
  assert 0 in list(res["unlicensed_atoms"])
  assert not res["all_licensed"]

  screen = proxide.assign_obc2_scaling_factors_with_provenance(["QZ", "CA"])
  screen_sources = list(screen["sources"])
  assert screen_sources[0] == 3, f"Expected ParameterSource::Fallback (3), got {screen_sources[0]}"
