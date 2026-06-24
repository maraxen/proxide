"""Tests for altloc (alternate-location) resolution in proxide parsers.

Covers:
- _resolve_altlocs deduplication logic (unit tests)
- parse_pdb_raw_rust: zero dup keys by default, max-occ selection, back-compat
- parse_structure: zero dup keys by default, max-occ selection, back-compat

Fixture: altloc_two_conf.pdb (committed under tests/io/parsing/)
  - ALA residue 1 chain A
  - CB has two altloc conformers: A occ=0.70 (coord [1,1,0]), B occ=0.30 (coord [1,2,0])
  - Backbone atoms N/CA/C/O have occupancy 1.00 (no altloc)
  - Expected winner: conformer A (max occupancy)

Network-free: all tests use the committed fixture.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

FIXTURE_DIR = Path(__file__).parent
ALTLOC_PDB = FIXTURE_DIR / "altloc_two_conf.pdb"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dup_key_count(raw) -> int:
    """Count distinct (chain_id, res_id, atom_name) keys that appear >1 time."""
    keys = list(zip(raw.chain_ids, raw.res_ids, raw.atom_names))
    return sum(1 for v in Counter(keys).values() if v > 1)


# ---------------------------------------------------------------------------
# Unit tests for _resolve_altlocs
# ---------------------------------------------------------------------------


class TestResolveAltlocs:
    """Direct tests of the _resolve_altlocs helper (no file I/O)."""

    def _make_raw(self, chain_ids, res_ids, atom_names, occupancies, b_factors, coords=None):
        """Build a minimal RawAtomData for testing."""
        from proxide.io.parsing.backend import RawAtomData

        n = len(atom_names)
        if coords is None:
            coords = np.zeros((n, 3), dtype=np.float32)
        return RawAtomData(
            num_atoms=n,
            atom_names=list(atom_names),
            res_names=["ALA"] * n,
            res_ids=list(res_ids),
            chain_ids=list(chain_ids),
            coords=np.asarray(coords, dtype=np.float32),
            elements=["C"] * n,
            occupancies=list(occupancies),
            b_factors=list(b_factors),
        )

    def test_no_duplicates_is_noop(self):
        """_resolve_altlocs returns the same object when there are no duplicates."""
        from proxide.io.parsing.backend import _resolve_altlocs

        raw = self._make_raw(
            chain_ids=["A", "A"],
            res_ids=[1, 1],
            atom_names=["N", "CA"],
            occupancies=[1.0, 1.0],
            b_factors=[10.0, 10.0],
        )
        result = _resolve_altlocs(raw)
        assert result is raw

    def test_mode_all_returns_same_object(self):
        """mode='all' returns the input unchanged, even with duplicates."""
        from proxide.io.parsing.backend import _resolve_altlocs

        raw = self._make_raw(
            chain_ids=["A", "A"],
            res_ids=[1, 1],
            atom_names=["CB", "CB"],
            occupancies=[0.70, 0.30],
            b_factors=[8.0, 12.0],
        )
        result = _resolve_altlocs(raw, mode="all")
        assert result is raw

    def test_max_occupancy_selected(self):
        """Atom with higher occupancy is kept."""
        from proxide.io.parsing.backend import _resolve_altlocs

        coords = np.array([[1.0, 1.0, 0.0], [1.0, 2.0, 0.0]], dtype=np.float32)
        raw = self._make_raw(
            chain_ids=["A", "A"],
            res_ids=[1, 1],
            atom_names=["CB", "CB"],
            occupancies=[0.70, 0.30],
            b_factors=[8.0, 12.0],
            coords=coords,
        )
        result = _resolve_altlocs(raw)
        assert result.num_atoms == 1
        assert _dup_key_count(result) == 0
        np.testing.assert_allclose(result.coords[0], [1.0, 1.0, 0.0])
        assert abs(result.occupancies[0] - 0.70) < 1e-6

    def test_tiebreak_lower_bfactor(self):
        """When occupancies are equal, lower B-factor wins."""
        from proxide.io.parsing.backend import _resolve_altlocs

        coords = np.array([[0.0, 0.0, 0.0], [9.9, 9.9, 9.9]], dtype=np.float32)
        raw = self._make_raw(
            chain_ids=["A", "A"],
            res_ids=[1, 1],
            atom_names=["CB", "CB"],
            occupancies=[0.50, 0.50],
            b_factors=[5.0, 20.0],
            coords=coords,
        )
        result = _resolve_altlocs(raw)
        assert result.num_atoms == 1
        np.testing.assert_allclose(result.coords[0], [0.0, 0.0, 0.0])

    def test_non_duplicate_atoms_preserved(self):
        """Atoms without altlocs (unique key) are always kept."""
        from proxide.io.parsing.backend import _resolve_altlocs

        raw = self._make_raw(
            chain_ids=["A", "A", "A"],
            res_ids=[1, 1, 1],
            atom_names=["N", "CB", "CB"],
            occupancies=[1.0, 0.70, 0.30],
            b_factors=[10.0, 8.0, 12.0],
        )
        result = _resolve_altlocs(raw)
        assert result.num_atoms == 2
        assert "N" in result.atom_names
        assert result.atom_names.count("CB") == 1


# ---------------------------------------------------------------------------
# parse_pdb_raw_rust tests
# ---------------------------------------------------------------------------


class TestParsePdbRawRust:
    """Tests for altloc resolution in parse_pdb_raw_rust."""

    def test_default_mode_zero_dup_keys(self):
        """Default altloc='occupancy' produces no duplicate (chain,res_id,atom_name) keys."""
        from proxide.io.parsing.backend import parse_pdb_raw_rust

        raw = parse_pdb_raw_rust(ALTLOC_PDB)
        assert _dup_key_count(raw) == 0, "Expected 0 duplicate keys after altloc resolution"

    def test_default_mode_max_occ_selected(self):
        """CB atom present is the one with higher occupancy (0.70, coord [1,1,0])."""
        from proxide.io.parsing.backend import parse_pdb_raw_rust

        raw = parse_pdb_raw_rust(ALTLOC_PDB)
        cb_indices = [i for i, a in enumerate(raw.atom_names) if a == "CB"]
        assert len(cb_indices) == 1, "Exactly one CB atom expected after dedup"
        i = cb_indices[0]
        assert abs(raw.occupancies[i] - 0.70) < 1e-5, (
            f"Expected occupancy 0.70, got {raw.occupancies[i]}"
        )
        np.testing.assert_allclose(
            np.asarray(raw.coords)[i],
            [1.0, 1.0, 0.0],
            atol=1e-3,
            err_msg="CB coord should match the max-occ conformer A",
        )

    def test_all_mode_preserves_duplicates(self):
        """altloc='all' preserves both CB altloc conformers (back-compat)."""
        from proxide.io.parsing.backend import parse_pdb_raw_rust

        raw_all = parse_pdb_raw_rust(ALTLOC_PDB, altloc="all")
        cb_count = raw_all.atom_names.count("CB")
        assert cb_count == 2, f"Expected 2 CB atoms with altloc='all', got {cb_count}"
        assert _dup_key_count(raw_all) == 1, "Expected 1 duplicate key with altloc='all'"

    def test_backbone_atoms_all_present(self):
        """Backbone atoms (N, CA, C, O) with occ=1.0 are never dropped."""
        from proxide.io.parsing.backend import parse_pdb_raw_rust

        raw = parse_pdb_raw_rust(ALTLOC_PDB)
        for backbone in ("N", "CA", "C", "O"):
            assert backbone in raw.atom_names, f"Backbone atom {backbone} missing after dedup"

    def test_total_atom_count_after_dedup(self):
        """After dedup: 4 backbone + 1 CB = 5 atoms total."""
        from proxide.io.parsing.backend import parse_pdb_raw_rust

        raw = parse_pdb_raw_rust(ALTLOC_PDB)
        assert raw.num_atoms == 5, f"Expected 5 atoms, got {raw.num_atoms}"


# ---------------------------------------------------------------------------
# parse_structure tests
# ---------------------------------------------------------------------------


class TestParseStructureAltloc:
    """Tests for altloc resolution in the high-level parse_structure API."""

    def test_default_mode_no_dup_keys_via_raw(self):
        """parse_structure default produces a Protein; underlying RawAtomData has 0 dup keys."""
        from proxide.io.parsing.backend import parse_pdb_raw_rust, parse_structure

        # Confirm the high-level parse doesn't regress raw dedup
        raw = parse_pdb_raw_rust(ALTLOC_PDB)
        assert _dup_key_count(raw) == 0

        # parse_structure should succeed and return 1 residue
        prot = parse_structure(ALTLOC_PDB)
        assert prot.coordinates.shape[0] == 1, "Expected 1 ALA residue"

    def test_altloc_all_returns_protein(self):
        """altloc='all' path still returns a valid Protein (back-compat)."""
        from proxide.io.parsing.backend import parse_structure

        prot = parse_structure(ALTLOC_PDB, altloc="all")
        assert prot.coordinates.shape[0] >= 1
