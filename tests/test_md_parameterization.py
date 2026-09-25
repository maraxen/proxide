"""Test MD parameterization integration with force field."""

from pathlib import Path

import numpy as np
import pytest

# Skip if Rust extension not available
pytest.importorskip("proxide._proxider")

from proxide import MissingResidueMode
from proxide.io.parsing.backend import (
    OutputSpec,
    is_rust_parser_available,
    parse_structure,
)

# Path to test data
TEST_DATA_DIR = Path(__file__).parent.parent / "data"
FF_XML_PATH = Path(__file__).parent.parent / "src" / "proxide" / "physics" / "force_fields" / "xml" / "protein.ff19SB.xml"

# NOTE: FF_XML_PATH above does not exist (no such `physics/force_fields/xml/`
# directory in this repo), so every test that guards on it always skips.
# That's a pre-existing, separate bug -- left alone here to keep this diff
# focused on the solvent-parameterization fix. The path below is verified to
# exist and is used by the new solvent tests so they actually run.
FF14SB_XML_PATH = Path(__file__).parent.parent / "src" / "proxide" / "assets" / "protein.ff14SB.xml"


class TestMDParameterization:
    """Tests for MD parameterization from force field."""

    @pytest.fixture
    def simple_pdb(self, tmp_path):
        """Create a simple PDB file for testing."""
        pdb_content = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00 20.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00 20.00           C
ATOM      4  O   ALA A   1       1.251   2.390   0.000  1.00 20.00           O
ATOM      5  CB  ALA A   1       1.978  -0.760   1.230  1.00 20.00           C
ATOM      6  N   GLY A   2       3.320   1.520   0.000  1.00 20.00           N
ATOM      7  CA  GLY A   2       3.970   2.820   0.000  1.00 20.00           C
ATOM      8  C   GLY A   2       5.480   2.720   0.000  1.00 20.00           C
ATOM      9  O   GLY A   2       6.020   1.600   0.000  1.00 20.00           O
END
"""
        pdb_path = tmp_path / "test.pdb"
        pdb_path.write_text(pdb_content)
        return pdb_path

    def test_parameterization_basic(self, simple_pdb):
        """Test that parameterization produces charges and LJ params."""
        if not FF_XML_PATH.exists():
            pytest.skip(f"Force field file not found: {FF_XML_PATH}")
        
        # Create spec with parameterization enabled
        spec = OutputSpec(
            parameterize_md=True,
            force_field=str(FF_XML_PATH),
        )
        
        # Parse with parameterization
        from proxide import _proxider
        result = _proxider.parse_structure(str(simple_pdb), spec)
        
        # Check that charges were assigned
        assert "charges" in result, "Missing charges in result"
        charges = result["charges"]
        assert len(charges) == 9, f"Expected 9 atoms, got {len(charges)}"
        
        # Check that sigmas and epsilons were assigned
        assert "sigmas" in result, "Missing sigmas in result"
        assert "epsilons" in result, "Missing epsilons in result"
        
        # Check that atom_types was assigned
        assert "atom_types" in result, "Missing atom_types in result"
        
        # Check parameterization stats
        assert "num_parameterized" in result
        assert "num_skipped" in result
        print(f"Parameterized: {result['num_parameterized']}, Skipped: {result['num_skipped']}")

    def test_parameterization_charges_nonzero(self, simple_pdb):
        """Test that backbone atoms get non-zero charges."""
        if not FF_XML_PATH.exists():
            pytest.skip(f"Force field file not found: {FF_XML_PATH}")
        
        spec = OutputSpec(
            parameterize_md=True,
            force_field=str(FF_XML_PATH),
        )
        
        from proxide import _proxider
        result = _proxider.parse_structure(str(simple_pdb), spec)
        
        charges = result["charges"]
        
        # N should have negative charge, CA positive
        # Check that not all charges are zero
        assert not np.allclose(charges, 0.0), "All charges are zero - parameterization failed"
        
        # Check charge range is reasonable (-1 to +1 for amino acids)
        assert np.all(charges >= -2.0) and np.all(charges <= 2.0), \
            f"Charges out of expected range: min={charges.min()}, max={charges.max()}"

    def test_no_parameterization_by_default(self, simple_pdb):
        """Test that parameterization is disabled by default."""
        from proxide import _proxider
        result = _proxider.parse_structure(str(simple_pdb))
        
        # Should not have MD params when not requested
        assert "charges" not in result or result.get("charges") is None

    def test_missing_ff_warning(self, simple_pdb, caplog):
        """Test warning when parameterize_md=True but no force_field provided."""
        import logging
        caplog.set_level(logging.WARNING)
        
        spec = OutputSpec(
            parameterize_md=True,
            # force_field not set
        )
    
        from proxide import _proxider
        result = _proxider.parse_structure(str(simple_pdb), spec)
        
        # Should complete without error, but no charges assigned
        assert "charges" not in result or result.get("charges") is None


class TestSolventParameterization:
    """Regression coverage for the defect where solvent (HOH/WAT/TIP3/SOL/DOD)
    atoms were structurally excluded from `parameterize_structure`'s
    residue-template loop (they're recorded only in
    `ProcessedStructure.solvent_atoms`, never in `residue_info`) and came
    back with charge=sigma=epsilon=0.0 no matter what force field was
    supplied -- confirmed on a real 23,558-atom DHFR fixture (21,069 solvent
    atoms, all zero). See `crates/proxide-physics/src/physics/md_params.rs`
    for the fix (solvent parameterized from the `WaterModel` catalog) and
    `unparameterized_atoms`/`num_unparameterized` for the new safety net
    that surfaces this whole class of bug instead of returning zeros
    silently.
    """

    @pytest.fixture
    def solvated_pdb(self, tmp_path):
        """One ALA residue plus two explicit, TIP3P-geometry waters."""
        pdb_content = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00 20.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00 20.00           C
ATOM      4  O   ALA A   1       1.251   2.390   0.000  1.00 20.00           O
ATOM      5  CB  ALA A   1       1.978  -0.760   1.230  1.00 20.00           C
HETATM    6  O   HOH W   1      20.000   0.000   0.000  1.00 20.00           O
HETATM    7  H1  HOH W   1      20.957   0.000   0.000  1.00 20.00           H
HETATM    8  H2  HOH W   1      19.760   0.927   0.000  1.00 20.00           H
HETATM    9  O   HOH W   2      40.000   0.000   0.000  1.00 20.00           O
HETATM   10  H1  HOH W   2      40.957   0.000   0.000  1.00 20.00           H
HETATM   11  H2  HOH W   2      39.760   0.927   0.000  1.00 20.00           H
END
"""
        pdb_path = tmp_path / "solvated.pdb"
        pdb_path.write_text(pdb_content)
        return pdb_path

    def test_water_charges_and_epsilon_nonzero(self, solvated_pdb):
        """(1) Parameterizing a solvated structure yields nonzero water
        charges and a nonzero oxygen epsilon, matching TIP3P."""
        if not FF14SB_XML_PATH.exists():
            pytest.skip(f"Force field file not found: {FF14SB_XML_PATH}")

        spec = OutputSpec(
            parameterize_md=True,
            remove_solvent=False,
            force_field=str(FF14SB_XML_PATH),
        )

        from proxide import _proxider
        result = _proxider.parse_structure(str(solvated_pdb), spec)

        charges = np.asarray(result["charges"])
        epsilons = np.asarray(result["epsilons"])
        sigmas = np.asarray(result["sigmas"])

        # Atom order follows the PDB: 0-4 = ALA, 5-7 = water 1 (O,H1,H2),
        # 8-10 = water 2 (O,H1,H2).
        water_idx = [5, 6, 7, 8, 9, 10]
        assert not np.allclose(charges[water_idx], 0.0), (
            "solvent charges all zero -- solvent atoms were not parameterized"
        )
        assert not np.allclose(epsilons[[5, 8]], 0.0), (
            "solvent oxygen epsilon all zero -- solvent atoms were not parameterized"
        )

        # TIP3P reference values (default `unit_system=Amber`: Angstrom /
        # kcal-mol, matching the `WaterModel` catalog's native units).
        np.testing.assert_allclose(charges[5], -0.834, atol=1e-3)  # O
        np.testing.assert_allclose(charges[6], 0.417, atol=1e-3)  # H1
        np.testing.assert_allclose(charges[7], 0.417, atol=1e-3)  # H2
        np.testing.assert_allclose(sigmas[5], 3.15061, atol=1e-2)  # O
        np.testing.assert_allclose(epsilons[5], 0.1521, atol=1e-3)  # O
        assert epsilons[6] == 0.0  # H has zero LJ epsilon in TIP3P by design

    def test_unparameterized_report_nonempty_for_unknown_residue(self, tmp_path):
        """(2) The unparameterized-atoms report is non-empty -- and strict
        mode errors -- for a residue the force field cannot parameterize
        (here, a small-molecule ligand with no ff14SB template)."""
        if not FF14SB_XML_PATH.exists():
            pytest.skip(f"Force field file not found: {FF14SB_XML_PATH}")

        pdb_content = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00 20.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00 20.00           C
ATOM      4  O   ALA A   1       1.251   2.390   0.000  1.00 20.00           O
ATOM      5  CB  ALA A   1       1.978  -0.760   1.230  1.00 20.00           C
HETATM    6  C1  LIG L   1      50.000   0.000   0.000  1.00 20.00           C
HETATM    7  C2  LIG L   1      51.500   0.000   0.000  1.00 20.00           C
END
"""
        pdb_path = tmp_path / "ligand.pdb"
        pdb_path.write_text(pdb_content)

        spec = OutputSpec(
            parameterize_md=True,
            remove_solvent=False,
            force_field=str(FF14SB_XML_PATH),
        )

        from proxide import _proxider
        result = _proxider.parse_structure(str(pdb_path), spec)

        assert "unparameterized_atoms" in result
        unparameterized = list(result["unparameterized_atoms"])
        assert unparameterized, "expected the ligand atoms to be reported as unparameterized"
        assert set(unparameterized) == {5, 6}
        assert result["num_unparameterized"] == len(unparameterized)

        # Strict mode turns the same input into a hard error instead of
        # quietly returning zeros for the ligand atoms.
        strict_spec = OutputSpec(
            parameterize_md=True,
            remove_solvent=False,
            force_field=str(FF14SB_XML_PATH),
            strict_parameterization=True,
        )
        with pytest.raises(ValueError):
            _proxider.parse_structure(str(pdb_path), strict_spec)

    def test_water_oh_bonds_preserved(self, solvated_pdb):
        """(3) Water O-H bonds are still emitted -- exactly 2 bonds per water
        molecule, from the geometric bond-inference pass in
        `proxide-geometry`'s `topology::generate_topology`. This is
        pre-existing, correct behavior (bond inference works on distance +
        covalent radii and never consults `residue_info`/`molecule_type`);
        locked here so the solvent-parameterization fix can't regress it.
        """
        if not FF14SB_XML_PATH.exists():
            pytest.skip(f"Force field file not found: {FF14SB_XML_PATH}")

        spec = OutputSpec(
            parameterize_md=True,
            remove_solvent=False,
            force_field=str(FF14SB_XML_PATH),
        )

        from proxide import _proxider
        result = _proxider.parse_structure(str(solvated_pdb), spec)

        bonds = np.asarray(result["bonds"])
        water_atoms = {5, 6, 7, 8, 9, 10}
        water_bonds = [
            tuple(sorted((int(b[0]), int(b[1]))))
            for b in bonds
            if int(b[0]) in water_atoms and int(b[1]) in water_atoms
        ]
        assert len(water_bonds) == 4  # 2 waters x 2 O-H bonds each
        assert len(set(water_bonds)) == 4  # no duplicates


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
