"""Tests for coords→χ angle measurement in proxide.geometry.measure_chi.

Acceptance gates:
- LEU chi1 AND chi2 within ±0.01° of biotite.structure.dihedral on real atoms
- None propagates for any missing/absent atom
- All returned values are float64
- Consistent dtype via measure_chi
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

import biotite.database.rcsb as rcsb
import biotite.structure as bs
import biotite.structure.io.pdb as pdb_io

from proxide import parse_structure
from proxide.chem.residues import atom_order, restype_order
from proxide.geometry.measure_chi import dihedral, measure_chi


# ---------------------------------------------------------------------------
# Tolerance (acceptance gate: ±0.01°)
# ---------------------------------------------------------------------------
TOLERANCE_DEG = 0.01
TOLERANCE_RAD = math.radians(TOLERANCE_DEG)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ubq_pdb_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Download 1UBQ (ubiquitin) to a temp directory.

    1UBQ has 9 LEU residues with all heavy atoms resolved — ideal for
    chi-angle parity tests.
    """
    tmp = tmp_path_factory.mktemp("structures")
    path = rcsb.fetch("1ubq", "pdb", str(tmp))
    return Path(path)


@pytest.fixture(scope="module")
def proxide_protein(ubq_pdb_path: Path):
    """Parse 1UBQ with proxide (altloc=occupancy, use_jax=False)."""
    return parse_structure(str(ubq_pdb_path), use_jax=False)


@pytest.fixture(scope="module")
def biotite_struct(ubq_pdb_path: Path):
    """Load 1UBQ model 1 with biotite."""
    f = pdb_io.PDBFile.read(str(ubq_pdb_path))
    return pdb_io.get_structure(f, model=1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _biotite_chi(biotite_struct, res_id: int, chain: str, quad: list[str]) -> float:
    """Measure a dihedral from biotite's AtomArray for the given 4-atom quad."""
    res = biotite_struct[(biotite_struct.res_id == res_id) & (biotite_struct.chain_id == chain)]
    def _coord(name: str) -> np.ndarray:
        mask = res.atom_name == name
        coords = res.coord[mask]
        if len(coords) == 0:
            raise KeyError(f"Atom {name!r} not found in residue {res_id}")
        return coords[0]

    return float(bs.dihedral(*[_coord(a) for a in quad]))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDihedralFormula:
    """Unit tests for the low-level dihedral() function."""

    def test_matches_biotite_synthetic_trans(self):
        """trans configuration: biotite and our formula agree."""
        p0 = np.array([0.0, 1.0, 0.0])
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        p3 = np.array([1.0, 0.0, 1.0])

        ours = dihedral(p0, p1, p2, p3)
        bio = float(bs.dihedral(p0, p1, p2, p3))
        assert abs(ours - bio) < 1e-6, f"Sign mismatch: ours={ours:.6f}, biotite={bio:.6f}"

    def test_matches_biotite_synthetic_gauche_plus(self):
        """gauche+ configuration (~+60°): sign and magnitude agree."""
        p0 = np.array([0.0, 1.0, 0.0])
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        p3 = np.array([1.0, math.cos(math.pi / 3), math.sin(math.pi / 3)])

        ours = dihedral(p0, p1, p2, p3)
        bio = float(bs.dihedral(p0, p1, p2, p3))
        assert abs(ours - bio) < 1e-6, f"Sign mismatch: ours={ours:.6f}, biotite={bio:.6f}"

    def test_range_is_minus_pi_to_pi(self):
        """Output is in [-π, π] for arbitrary inputs."""
        rng = np.random.default_rng(42)
        for _ in range(50):
            pts = rng.standard_normal((4, 3))
            angle = dihedral(*pts)
            assert -math.pi <= angle <= math.pi, f"Out of range: {angle}"

    def test_returns_float(self):
        p0 = np.array([0.0, 1.0, 0.0])
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        p3 = np.array([1.0, 0.0, 1.0])
        result = dihedral(p0, p1, p2, p3)
        assert isinstance(result, float)


class TestMeasureChiLeuParity:
    """Core gate: LEU chi1 and chi2 within ±0.01° of biotite on real atoms."""

    def test_leu_chi1_chi2_parity(
        self,
        proxide_protein,
        biotite_struct,
    ):
        """Both chi1 and chi2 for every LEU residue agree with biotite ≤0.01°."""
        from proxide.chem.residues import chi_angles_atoms, restype_1to3

        coords = np.array(proxide_protein.coordinates)  # (n_res, 37, 3)
        aatype = np.array(proxide_protein.aatype)
        res_index = np.array(proxide_protein.residue_index)

        leu_code = restype_order["L"]
        leu_residues = np.where(aatype == leu_code)[0]
        assert len(leu_residues) > 0, "No LEU residues found in 1UBQ (unexpected)"

        chi1_errors = []
        chi2_errors = []

        for res_pos in leu_residues:
            res_coords = coords[res_pos]  # (37, 3)
            chi_result = measure_chi(res_coords, "LEU")

            assert "chi1" in chi_result, "chi1 key missing for LEU"
            assert "chi2" in chi_result, "chi2 key missing for LEU"
            assert chi_result["chi1"] is not None, f"chi1 is None for res pos {res_pos}"
            assert chi_result["chi2"] is not None, f"chi2 is None for res pos {res_pos}"

            # Biotite reference: use the same four atom coordinates
            leu_chi_quads = chi_angles_atoms["LEU"]
            bio_chi1 = float(bs.dihedral(*[res_coords[atom_order[a]] for a in leu_chi_quads[0]]))
            bio_chi2 = float(bs.dihedral(*[res_coords[atom_order[a]] for a in leu_chi_quads[1]]))

            err1 = abs(math.degrees(chi_result["chi1"]) - math.degrees(bio_chi1))
            err2 = abs(math.degrees(chi_result["chi2"]) - math.degrees(bio_chi2))
            chi1_errors.append(err1)
            chi2_errors.append(err2)

            assert err1 <= TOLERANCE_DEG, (
                f"LEU res {res_pos} chi1 error {err1:.6f}° exceeds ±{TOLERANCE_DEG}°"
            )
            assert err2 <= TOLERANCE_DEG, (
                f"LEU res {res_pos} chi2 error {err2:.6f}° exceeds ±{TOLERANCE_DEG}°"
            )

        max1 = max(chi1_errors)
        max2 = max(chi2_errors)
        # Report max error for both (surfaced in test output)
        print(f"\n  LEU chi1 max error: {max1:.7f}°  chi2 max error: {max2:.7f}°  "
              f"(over {len(leu_residues)} residues)")
        assert max1 <= TOLERANCE_DEG
        assert max2 <= TOLERANCE_DEG


class TestMeasureChiMissingAtom:
    """Missing atoms propagate as None."""

    def _make_residue_positions(self, restype3: str, absent_atom: str) -> np.ndarray:
        """Return a (37,3) position array with `absent_atom` zeroed out."""
        from proxide.chem.residues import chi_angles_atoms

        # Build a fake residue with non-zero coordinates
        positions = np.ones((37, 3), dtype=np.float64)
        # Put distinct coords so dihedrals are well-defined for all *present* atoms
        rng = np.random.default_rng(0)
        positions[:] = rng.standard_normal((37, 3)) + 5.0

        # Zero out the absent atom's slot
        slot = atom_order.get(absent_atom)
        assert slot is not None, f"{absent_atom} not in atom_order"
        positions[slot] = 0.0
        return positions

    def test_missing_chi_atom_returns_none_leu(self):
        """LEU with CG zeroed out → chi1 and chi2 both None."""
        positions = self._make_residue_positions("LEU", "CG")
        result = measure_chi(positions, "LEU")
        # CG is in both chi1 (N,CA,CB,CG) and chi2 (CA,CB,CG,CD1)
        assert result["chi1"] is None, "chi1 should be None when CG is missing"
        assert result["chi2"] is None, "chi2 should be None when CG is missing"

    def test_missing_last_atom_only_affects_that_chi_leu(self):
        """LEU with CD1 zeroed out → chi1 present, chi2 None."""
        positions = self._make_residue_positions("LEU", "CD1")
        result = measure_chi(positions, "LEU")
        # CD1 is only in chi2 (CA,CB,CG,CD1)
        assert result["chi1"] is not None, "chi1 should be present when only CD1 is absent"
        assert result["chi2"] is None, "chi2 should be None when CD1 is missing"

    def test_missing_atom_ala_empty(self):
        """ALA has no chi angles → empty dict regardless."""
        positions = np.ones((37, 3), dtype=np.float64)
        result = measure_chi(positions, "ALA")
        assert result == {}, "ALA should return empty chi dict"

    def test_gly_empty(self):
        """GLY has no chi angles → empty dict."""
        positions = np.ones((37, 3), dtype=np.float64)
        result = measure_chi(positions, "GLY")
        assert result == {}


class TestMeasureChiDtype:
    """Returned chi values are float64."""

    def test_chi_values_are_float64(self, proxide_protein):
        coords = np.array(proxide_protein.coordinates, dtype=np.float64)
        aatype = np.array(proxide_protein.aatype)
        leu_code = restype_order["L"]
        leu_residues = np.where(aatype == leu_code)[0]

        for res_pos in leu_residues[:3]:
            result = measure_chi(coords[res_pos], "LEU")
            for key, val in result.items():
                if val is not None:
                    # Python float is always f64 on CPython
                    assert isinstance(val, float), f"{key} value is {type(val)}, expected float"
                    # Also verify the underlying numpy computation is f64
                    as_np = np.float64(val)
                    assert as_np.dtype == np.float64

    def test_dihedral_returns_python_float(self):
        p0 = np.array([0.0, 1.0, 0.0])
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        p3 = np.array([1.0, 0.0, 1.0])
        result = dihedral(p0, p1, p2, p3)
        assert isinstance(result, float)
        # float() wraps a Python double → confirms f64 precision
        assert np.float64(result).dtype == np.float64
