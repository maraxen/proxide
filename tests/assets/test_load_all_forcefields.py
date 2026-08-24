"""Tests for loading all force field XML files from assets.

This module validates that all force field XML files can be parsed
successfully by the Rust parser. `gaff/` stays git-vendored and is read
directly. `amber/`, `water/`, `implicit/`, and most of `openmm_bundled/`
are fetched-and-cached on first use (proxide.assets._fetch) rather than
vendored -- see .praxia/docs/specs/260824_license-provenance-verification.md.

Parametrization over the fetchable/restricted set uses the static
`_asset_index.ASSET_INDEX` dict (a plain committed Python module, no I/O) so
collection doesn't require any files -- or network -- to be present. Actual
fetches happen lazily inside each test body, and are skipped (not failed) if
the network is unavailable, matching this file's existing skip-on-missing
convention for gracefully degrading environments.
"""

from pathlib import Path

import pytest

from proxide import _proxider
from proxide.assets import _asset_index
from proxide.assets._fetch import CharmmLicenseRequiredError, is_charmm_restricted, resolve_asset

# Get the assets directory (still used for the git-vendored gaff/ subtree)
ASSETS_DIR = Path(__file__).parent.parent.parent / "src" / "proxide" / "assets"

# Names whose content our Rust parser doesn't fully support yet (unrelated to
# fetch-vs-vendor -- these were already xfail/skip before this conversion).
_KNOWN_UNSUPPORTED = {"amoeba2009", "amoeba2013", "amoeba2018", "iamoeba"}

# Not force field parameter files at all (naming-convention/support data
# bundled alongside them) -- the pre-conversion test excluded these by name
# for the same reason; pdbNames.xml genuinely fails our parser
# ("Missing required attribute: type"), confirmed by actually running this
# suite against the fetch mechanism rather than assuming the old exclusion
# list still applied verbatim.
_NOT_FORCE_FIELDS = {"pdbNames"}

_FETCHABLE_ENTRIES = sorted(
  (stem, rel) for stem, rel in _asset_index.ASSET_INDEX.items() if not is_charmm_restricted(rel)
)


def _resolve_or_skip(relative_path: str) -> Path:
  try:
    return resolve_asset(relative_path)
  except (OSError, ValueError) as e:
    pytest.skip(f"Could not fetch {relative_path} (network unavailable?): {e}")


class TestLoadAllForceFields:
  """Tests for loading all fetchable + still-vendored force field XML files."""

  def test_asset_index_is_non_empty(self) -> None:
    assert len(_asset_index.ASSET_INDEX) > 0, "Asset index is empty"

  @pytest.mark.parametrize(
    ("stem", "relative_path"),
    _FETCHABLE_ENTRIES,
    ids=[rel for _, rel in _FETCHABLE_ENTRIES],
  )
  def test_load_fetchable_xml_file(self, stem: str, relative_path: str) -> None:
    """Each fetchable (non-CHARMM-restricted) asset can be fetched and parsed."""
    if stem in _NOT_FORCE_FIELDS:
      pytest.skip(f"Not a force field file: {stem}")
    if stem in _KNOWN_UNSUPPORTED:
      pytest.skip(f"Known unsupported format: {stem}")

    xml_file = _resolve_or_skip(relative_path)
    try:
      result = _proxider.load_forcefield(str(xml_file))
      assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    except (ValueError, RuntimeError) as e:
      if "AMOEBA" in str(e) or "Amoeba" in str(e):
        pytest.skip(f"Known unsupported format (AMOEBA): {e}")
      raise

  def test_charmm_restricted_names_require_configuration(self) -> None:
    """CHARMM-restricted names are excluded from the fetchable set and gate on env config."""
    restricted = [
      (stem, rel) for stem, rel in _asset_index.ASSET_INDEX.items() if is_charmm_restricted(rel)
    ]
    assert restricted, "Expected at least one CHARMM-restricted entry (e.g. charmm36)"
    for _, rel in restricted:
      with pytest.raises(CharmmLicenseRequiredError):
        resolve_asset(rel)


class TestGaffXmlFiles:
  """Tests specifically for GAFF force field files (still git-vendored, unchanged)."""

  @pytest.fixture
  def gaff_dir(self) -> Path:
    """Get GAFF ffxml directory."""
    return ASSETS_DIR / "gaff" / "ffxml"

  def test_gaff_directory_exists(self, gaff_dir: Path) -> None:
    """Verify GAFF directory exists."""
    assert gaff_dir.exists(), f"GAFF directory not found: {gaff_dir}"

  @pytest.mark.parametrize(
    "version",
    ["gaff-1.4", "gaff-1.7", "gaff-1.8", "gaff-1.81", "gaff-2.1", "gaff-2.11", "gaff-2.2.20"],
  )
  def test_load_gaff_version(self, gaff_dir: Path, version: str) -> None:
    """Test loading each GAFF version."""
    xml_file = gaff_dir / f"{version}.xml"
    assert xml_file.exists(), f"GAFF file not found: {xml_file}"

    result = _proxider.load_forcefield(str(xml_file))

    assert len(result["atom_types"]) > 0, f"No atom types in {version}"
    assert len(result["harmonic_bonds"]) > 0, f"No bonds in {version}"
    assert len(result["harmonic_angles"]) > 0, f"No angles in {version}"
    assert len(result["proper_torsions"]) > 0, f"No proper torsions in {version}"
    assert len(result["nonbonded_params"]) > 0, f"No nonbonded params in {version}"

  def test_gaff_211_has_expected_types(self, gaff_dir: Path) -> None:
    """Test that GAFF 2.11 has expected atom types."""
    xml_file = gaff_dir / "gaff-2.11.xml"
    result = _proxider.load_forcefield(str(xml_file))

    atom_type_names = {at["name"] for at in result["atom_types"]}

    expected_types = {"c", "c1", "c2", "c3", "ca", "n", "n3", "o", "oh", "os", "h1", "hc", "ha", "hn", "ho"}
    assert expected_types.issubset(atom_type_names), f"Missing types: {expected_types - atom_type_names}"


class TestAmberXmlFiles:
  """Tests for Amber protein force field files (fetch-on-demand)."""

  def test_load_ff14sb(self) -> None:
    """Test loading ff14SB."""
    xml_file = _resolve_or_skip("amber/ff14SB.xml")
    result = _proxider.load_forcefield(str(xml_file))
    assert len(result["atom_types"]) > 0
    assert len(result["residue_templates"]) > 0

  def test_load_ff19sb(self) -> None:
    """Test loading ff19SB."""
    xml_file = _resolve_or_skip("amber/protein.ff19SB.xml")
    result = _proxider.load_forcefield(str(xml_file))
    assert len(result["atom_types"]) > 0
    assert len(result["residue_templates"]) > 0


class TestImplicitSolventFiles:
  """Tests for implicit solvent (GBSA-OBC) files (fetch-on-demand)."""

  @pytest.mark.parametrize(
    "version",
    ["amber96_obc", "amber99_obc", "amber03_obc", "amber10_obc"],
  )
  def test_load_obc_version(self, version: str) -> None:
    """Test loading each OBC version."""
    xml_file = _resolve_or_skip(f"implicit/{version}.xml")
    result = _proxider.load_forcefield(str(xml_file))

    assert "gbsa_obc_params" in result
    assert len(result["gbsa_obc_params"]) > 0, f"No GBSA params in {version}"


class TestWaterModels:
  """Tests for water model files (fetch-on-demand)."""

  @pytest.mark.parametrize(
    "model",
    ["tip3p_standard", "tip4pew_standard", "opc_standard"],
  )
  def test_load_water_model(self, model: str) -> None:
    """Test loading water models.

    Note: Some water models use constraint-only definitions that our parser
    may not fully support.
    """
    xml_file = _resolve_or_skip(f"water/{model}.xml")
    try:
      result = _proxider.load_forcefield(str(xml_file))
      assert isinstance(result, dict)
    except ValueError as e:
      pytest.skip(f"Water model format not fully supported: {e}")
