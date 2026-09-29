"""Python surface of mbondi2/OBC2 parameter provenance (spec D16).

The values arrays are unchanged from the legacy list-returning functions; the
provenance variants add a per-atom uint8 ``sources`` channel so a caller can
tell a tabulated radius from a stand-in without re-deriving element inference.
"""

import numpy as np
import pytest

proxide = pytest.importorskip("proxide")

# Carbon, amide H (bonded to N), nitrogen, selenium, sodium, chlorine.
NAMES = ["CA", "H", "N", "SE", "NA", "CL"]
BONDS = [(1, 2)]


def _codes(result: dict) -> dict[str, int]:
  return {meaning: code for code, meaning in result["source_code_meanings"].items()}


def test_radii_values_match_legacy_function() -> None:
  result = proxide.assign_mbondi2_radii_with_provenance(NAMES, BONDS)
  legacy = proxide.assign_mbondi2_radii(NAMES, BONDS)
  assert result["values"].dtype == np.float32
  np.testing.assert_array_equal(result["values"], np.asarray(legacy, dtype=np.float32))


def test_screen_values_match_legacy_function() -> None:
  result = proxide.assign_obc2_scaling_factors_with_provenance(NAMES)
  legacy = proxide.assign_obc2_scaling_factors(NAMES)
  np.testing.assert_array_equal(result["values"], np.asarray(legacy, dtype=np.float32))


def test_elements_outside_mbondi2_are_flagged_unlicensed() -> None:
  result = proxide.assign_mbondi2_radii_with_provenance(NAMES, BONDS)
  assert result["sources"].dtype == np.uint8
  assert len(result["sources"]) == len(NAMES)
  # SE and NA are not defined by mbondi2; everything else is.
  assert result["unlicensed_atoms"].tolist() == [3, 4]
  assert result["num_unlicensed"] == 2
  assert result["all_licensed"] is False


def test_source_codes_are_self_describing() -> None:
  result = proxide.assign_mbondi2_radii_with_provenance(NAMES, BONDS)
  sources = result["sources"].tolist()
  meanings = result["source_code_meanings"]
  assert set(sources) <= set(meanings)
  # The hydrogen bonded to nitrogen must carry a distinct code from carbon's.
  assert sources[0] != sources[1]


def test_fully_tabulated_input_is_all_licensed() -> None:
  result = proxide.assign_obc2_scaling_factors_with_provenance(["CA", "N", "O", "S"])
  assert result["all_licensed"] is True
  assert result["num_unlicensed"] == 0
  assert result["unlicensed_atoms"].size == 0


def test_provenance_names_citation_and_reference() -> None:
  result = proxide.assign_mbondi2_radii_with_provenance(NAMES, BONDS)
  assert result["schema_version"] == 1
  prov = result["provenance"]
  assert "Onufriev" in prov["citation"]
  assert prov["reference_project"] == "OpenMM"


def test_table_info_does_not_define_selenium() -> None:
  info = proxide.mbondi2_table_info()
  # Exact sets: adding an element must fail as loudly as removing one.
  assert info["radius_elements"] == ["C", "Cl", "F", "N", "O", "P", "S", "Si"]
  assert info["screen_elements"] == ["C", "F", "H", "N", "O", "P", "S"]
  assert info["fallback_radius"] == pytest.approx(1.5)
  assert info["fallback_screen"] == pytest.approx(0.8)
