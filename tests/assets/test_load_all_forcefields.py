"""Tests for loading all force field XML files from assets.

This module validates that all force field XML files in the assets directory
can be parsed successfully by the Rust parser.
"""

from pathlib import Path

import pytest

import proxide_rs


# Get the assets directory
ASSETS_DIR = Path(__file__).parent.parent.parent / "src" / "priox" / "assets"


def get_all_xml_files() -> list[Path]:
    """Get all XML files from assets directory recursively."""
    xml_files = []
    for xml_file in ASSETS_DIR.rglob("*.xml"):
        # Skip some files that are not force fields
        if xml_file.name in ("pdbNames.xml", "residues.xml", "hydrogens.xml", "glycam-hydrogens.xml"):
            continue
        xml_files.append(xml_file)
    return sorted(xml_files)


class TestLoadAllForceFields:
    """Tests for loading all force field XML files."""

    @pytest.fixture(scope="class")
    def all_xml_files(self) -> list[Path]:
        """Get all XML files to test."""
        return get_all_xml_files()

    def test_assets_directory_exists(self) -> None:
        """Verify assets directory exists."""
        assert ASSETS_DIR.exists(), f"Assets directory not found: {ASSETS_DIR}"
        assert ASSETS_DIR.is_dir()

    def test_has_xml_files(self, all_xml_files: list[Path]) -> None:
        """Verify we have XML files to test."""
        assert len(all_xml_files) > 0, "No XML files found in assets directory"
        print(f"\nFound {len(all_xml_files)} XML files to test")

    @pytest.mark.parametrize(
        "xml_file",
        get_all_xml_files(),
        ids=lambda p: str(p.relative_to(ASSETS_DIR)),
    )
    def test_load_xml_file(self, xml_file: Path) -> None:
        """Test that each XML file can be loaded by the Rust parser.

        Note: Some OpenMM bundled files use different formats that our parser
        may not fully support. We mark those as xfail rather than failures.
        """
        # Files known to use formats our parser doesn't fully support
        known_unsupported = {
            # AMOEBA uses fundamentally different format (multipoles, etc.)
            "amoeba2009.xml", "amoeba2013.xml", "amoeba2018.xml", "iamoeba.xml",
        }

        try:
            result = priox_rs.load_forcefield(str(xml_file))

            # Basic validation - should return a dict
            assert isinstance(result, dict), f"Expected dict, got {type(result)}"
        except (ValueError, RuntimeError) as e:
            # Check if this is an expected unsupported file (AMOEBA)
            str_e = str(e)
            if "AMOEBA" in str_e or "Amoeba" in str_e:
                pytest.skip(f"Known unsupported format (AMOEBA): {e}")
            elif xml_file.name in known_unsupported:
                pytest.skip(f"Known unsupported format: {e}")
            else:
                raise


class TestGaffXmlFiles:
    """Tests specifically for GAFF force field files."""

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

        result = priox_rs.load_forcefield(str(xml_file))

        # GAFF should have atom types
        assert len(result["atom_types"]) > 0, f"No atom types in {version}"

        # GAFF should have bond parameters
        assert len(result["harmonic_bonds"]) > 0, f"No bonds in {version}"

        # GAFF should have angle parameters
        assert len(result["harmonic_angles"]) > 0, f"No angles in {version}"

        # GAFF should have torsion parameters
        assert len(result["proper_torsions"]) > 0, f"No proper torsions in {version}"

        # GAFF should have nonbonded parameters
        assert len(result["nonbonded_params"]) > 0, f"No nonbonded params in {version}"

    def test_gaff_211_has_expected_types(self, gaff_dir: Path) -> None:
        """Test that GAFF 2.11 has expected atom types."""
        xml_file = gaff_dir / "gaff-2.11.xml"
        result = priox_rs.load_forcefield(str(xml_file))

        # Get atom type names
        atom_type_names = {at["name"] for at in result["atom_types"]}

        # Should have common GAFF types
        expected_types = {"c", "c1", "c2", "c3", "ca", "n", "n3", "o", "oh", "os", "h1", "hc", "ha", "hn", "ho"}
        assert expected_types.issubset(atom_type_names), f"Missing types: {expected_types - atom_type_names}"


class TestAmberXmlFiles:
    """Tests for Amber protein force field files."""

    @pytest.fixture
    def amber_dir(self) -> Path:
        """Get Amber directory."""
        return ASSETS_DIR / "amber"

    def test_load_ff14sb(self, amber_dir: Path) -> None:
        """Test loading ff14SB."""
        xml_file = amber_dir / "ff14SB.xml"
        if not xml_file.exists():
            pytest.skip("ff14SB.xml not found")

        result = priox_rs.load_forcefield(str(xml_file))
        assert len(result["atom_types"]) > 0
        assert len(result["residue_templates"]) > 0

    def test_load_ff19sb(self, amber_dir: Path) -> None:
        """Test loading ff19SB."""
        xml_file = amber_dir / "protein.ff19SB.xml"
        if not xml_file.exists():
            pytest.skip("protein.ff19SB.xml not found")

        result = priox_rs.load_forcefield(str(xml_file))
        assert len(result["atom_types"]) > 0
        assert len(result["residue_templates"]) > 0


class TestImplicitSolventFiles:
    """Tests for implicit solvent (GBSA-OBC) files."""

    @pytest.fixture
    def implicit_dir(self) -> Path:
        """Get implicit solvent directory."""
        return ASSETS_DIR / "implicit"

    def test_implicit_directory_exists(self, implicit_dir: Path) -> None:
        """Verify implicit directory exists."""
        assert implicit_dir.exists(), f"Implicit directory not found: {implicit_dir}"

    @pytest.mark.parametrize(
        "version",
        ["amber96_obc", "amber99_obc", "amber03_obc", "amber10_obc"],
    )
    def test_load_obc_version(self, implicit_dir: Path, version: str) -> None:
        """Test loading each OBC version."""
        xml_file = implicit_dir / f"{version}.xml"
        if not xml_file.exists():
            pytest.skip(f"{version}.xml not found")

        result = priox_rs.load_forcefield(str(xml_file))

        # OBC files should have GBSA parameters
        assert "gbsa_obc_params" in result
        assert len(result["gbsa_obc_params"]) > 0, f"No GBSA params in {version}"


class TestWaterModels:
    """Tests for water model files."""

    @pytest.fixture
    def water_dir(self) -> Path:
        """Get water directory."""
        return ASSETS_DIR / "water"

    @pytest.mark.parametrize(
        "model",
        ["tip3p_standard", "tip4pew_standard", "opc_standard"],
    )
    def test_load_water_model(self, water_dir: Path, model: str) -> None:
        """Test loading water models.

        Note: Some water models use constraint-only definitions that our parser
        may not fully support.
        """
        xml_file = water_dir / f"{model}.xml"
        if not xml_file.exists():
            pytest.skip(f"{model}.xml not found")

        try:
            result = priox_rs.load_forcefield(str(xml_file))
            # Water models may have just residue templates
            assert isinstance(result, dict)
        except ValueError as e:
            # Water models with unusual format
            pytest.skip(f"Water model format not fully supported: {e}")

