"""NeRF round-trip test: χ → coords → χ ≤ 0.1° RMS (B4, #2654).

Acceptance gates (all must pass):
  1. Round-trip RMS(Δχ) ≤ 0.1° for each covered residue type.
  2. Symmetric χ angles compared via fold_symmetric_chi (B3) representatives
     so forward/back landing on a symmetric partner is not counted as error.
  3. Circular angle differencing: Δ = atan2(sin(a−b), cos(a−b)).
  4. measure_chi matches biotite within ±0.01° on real LEU residues from 1UBQ
     (re-assertion of B2 gate, in-file so B4 is self-contained).

Residue types covered
---------------------
All 18 residue types that have at least one χ angle:
  ARG (4), ASN (2), ASP (2), CYS (1), GLN (3), GLU (3), HIS (2), ILE (2),
  LEU (2), LYS (4), MET (3), PHE (2), PRO (2), SER (1), THR (1), TRP (2),
  TYR (2), VAL (1).
ALA and GLY have no χ angles and are not included.

Forward NeRF used
-----------------
proxide.geometry.nerf_forward.chi_to_coords — pure-Python / NumPy implementation
built for this sprint.  PyO3/Rust deferred (see ADR 260623_coords_to_chi_path.md).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

import biotite.database.rcsb as rcsb
import biotite.structure as bs
import biotite.structure.io.pdb as pdb_io

from proxide import parse_structure
from proxide.chem.chi import CHI_ANGLES_ATOMS, CHI_SYMMETRY_CLASS
from proxide.chem.residues import atom_order, chi_angles_mask, restype_order, restype_1to3
from proxide.geometry.fold_chi import fold_symmetric_chi
from proxide.geometry.measure_chi import dihedral, measure_chi
from proxide.geometry.nerf_forward import chi_to_coords


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROUNDTRIP_RMS_GATE_DEG = 0.1   # B4 acceptance gate
BIOTITE_PARITY_DEG = 0.01      # B2 re-assertion gate

# All 18 residue types with at least one χ angle
CHI_RESIDUES = [
    "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _circular_diff(a: float, b: float) -> float:
    """Signed circular difference a − b in radians, result in (−π, π]."""
    return math.atan2(math.sin(a - b), math.cos(a - b))


def _is_symmetric(restype3: str, chi_idx: int) -> bool:
    """Return True if (restype3, chi_idx) has a symmetric folding."""
    sym = CHI_SYMMETRY_CLASS.get((restype3, chi_idx), "none")
    return sym in ("pi_periodic_180", "aromatic_180")


def _compare_chi(
    chi_input: float,
    chi_measured: float,
    restype3: str,
    chi_idx: int,
) -> float:
    """Return |circular difference| for one χ angle.

    For symmetric χ angles (Asp χ2, Glu χ3, Phe/Tyr χ2), compare via
    fold_symmetric_chi representatives so that landing on the symmetric
    partner is correctly counted as 0 error.
    """
    if _is_symmetric(restype3, chi_idx):
        a = fold_symmetric_chi(chi_input, restype3, chi_idx)
        b = fold_symmetric_chi(chi_measured, restype3, chi_idx)
        return abs(_circular_diff(a, b))
    return abs(_circular_diff(chi_input, chi_measured))


# ---------------------------------------------------------------------------
# Synthetic round-trip fixture
# ---------------------------------------------------------------------------

def _make_test_cases() -> list[dict]:
    """Build a list of test cases: one dict per (restype, chi_vector) pair.

    Each residue type gets multiple χ vectors to exercise different rotameric
    regions of the torus.
    """
    rng = np.random.default_rng(42)
    test_cases = []

    # Canonical backbone: standard extended-chain geometry, translated so that
    # CA is NOT at the origin (all-zero CA coords would be treated as absent
    # by measure_chi's all-zero heuristic).
    _offset = np.array([3.0, 7.0, -1.5])
    N_bb = np.array([-0.521, 1.363, 0.0]) + _offset
    CA_bb = np.array([0.0, 0.0, 0.0]) + _offset
    C_bb = np.array([1.525, 0.0, 0.0]) + _offset

    # A second backbone with an arbitrary global rotation+translation so that
    # the test is not degenerate at the standard orientation
    R = np.array([
        [0.0, -1.0,  0.0],
        [0.0,  0.0, -1.0],
        [1.0,  0.0,  0.0],
    ])
    t = np.array([5.0, -3.0, 2.5])
    N_g = R @ (N_bb - _offset) + t
    CA_g = R @ (CA_bb - _offset) + t
    C_g = R @ (C_bb - _offset) + t

    for restype3 in CHI_RESIDUES:
        rt1 = restype_1to3.get(None)  # Not used; determine n_chi from chi_mask
        from proxide.chem.residues import restype_3to1
        rt1_letter = restype_3to1[restype3]
        rt_idx = restype_order[rt1_letter]
        n_chi = int(sum(chi_angles_mask[rt_idx]))

        if n_chi == 0:
            continue

        # Several representative χ vectors per residue
        for _ in range(5):
            chi_vec = rng.uniform(-np.pi, np.pi, size=n_chi).tolist()

            # Standard backbone
            test_cases.append({
                "restype3": restype3,
                "N": N_bb,
                "CA": CA_bb,
                "C": C_bb,
                "chi_input": chi_vec,
            })
            # Rotated+translated backbone
            test_cases.append({
                "restype3": restype3,
                "N": N_g,
                "CA": CA_g,
                "C": C_g,
                "chi_input": chi_vec,
            })

    return test_cases


# ---------------------------------------------------------------------------
# Test 1: NeRF round-trip RMS ≤ 0.1° for all covered residue types
# ---------------------------------------------------------------------------

class TestNerfRoundtrip:
    """NeRF round-trip: χ → chi_to_coords → measure_chi → compare."""

    @pytest.fixture(scope="class")
    def all_cases(self):
        return _make_test_cases()

    def test_per_residue_rms(self, all_cases):
        """For each residue type, assert RMS(Δχ) ≤ 0.1° over all test cases."""
        from collections import defaultdict
        errors_by_restype: dict[str, list[float]] = defaultdict(list)

        for case in all_cases:
            restype3 = case["restype3"]
            N, CA, C = case["N"], case["CA"], case["C"]
            chi_input = case["chi_input"]

            # Forward: χ → coords
            atom37, present = chi_to_coords(restype3, N, CA, C, chi_input)

            # Backward: coords → χ
            measured = measure_chi(atom37, restype3)

            # Compare each χ
            for i, chi_val in enumerate(chi_input):
                chi_name = f"chi{i + 1}"
                measured_val = measured.get(chi_name)
                if measured_val is None:
                    pytest.fail(
                        f"{restype3} {chi_name}: measure_chi returned None — "
                        f"forward NeRF produced a missing atom?"
                    )

                err_rad = _compare_chi(chi_val, measured_val, restype3, i)
                err_deg = math.degrees(err_rad)
                errors_by_restype[restype3].append(err_deg)

        # Assert per-residue RMS ≤ 0.1°
        failures = []
        rms_results = {}
        for restype3, errs in sorted(errors_by_restype.items()):
            rms_deg = math.sqrt(sum(e ** 2 for e in errs) / len(errs))
            rms_results[restype3] = rms_deg
            if rms_deg > ROUNDTRIP_RMS_GATE_DEG:
                failures.append(f"{restype3}: RMS={rms_deg:.4f}° > {ROUNDTRIP_RMS_GATE_DEG}°")

        # Print summary for visibility even on pass
        print("\nNeRF round-trip RMS per residue type:")
        for restype3, rms_deg in sorted(rms_results.items()):
            tag = "PASS" if rms_deg <= ROUNDTRIP_RMS_GATE_DEG else "FAIL"
            print(f"  {restype3:4s}: RMS={rms_deg:.4f}°  [{tag}]")

        if failures:
            pytest.fail(
                f"Round-trip RMS gate ({ROUNDTRIP_RMS_GATE_DEG}°) failed for:\n"
                + "\n".join(f"  {f}" for f in failures)
            )

    def test_overall_rms(self, all_cases):
        """Global RMS across all residue types and χ angles ≤ 0.1°."""
        all_errors = []

        for case in all_cases:
            restype3 = case["restype3"]
            N, CA, C = case["N"], case["CA"], case["C"]
            chi_input = case["chi_input"]

            atom37, _ = chi_to_coords(restype3, N, CA, C, chi_input)
            measured = measure_chi(atom37, restype3)

            for i, chi_val in enumerate(chi_input):
                chi_name = f"chi{i + 1}"
                measured_val = measured.get(chi_name)
                if measured_val is not None:
                    err_rad = _compare_chi(chi_val, measured_val, restype3, i)
                    all_errors.append(math.degrees(err_rad))

        rms_deg = math.sqrt(sum(e ** 2 for e in all_errors) / len(all_errors))
        print(f"\nGlobal round-trip RMS: {rms_deg:.6f}° over {len(all_errors)} χ angles")
        assert rms_deg <= ROUNDTRIP_RMS_GATE_DEG, (
            f"Global round-trip RMS {rms_deg:.4f}° > {ROUNDTRIP_RMS_GATE_DEG}°"
        )

    def test_max_error(self, all_cases):
        """Maximum single-angle error ≤ 0.1° (no outliers)."""
        max_err_deg = 0.0
        worst = None

        for case in all_cases:
            restype3 = case["restype3"]
            N, CA, C = case["N"], case["CA"], case["C"]
            chi_input = case["chi_input"]

            atom37, _ = chi_to_coords(restype3, N, CA, C, chi_input)
            measured = measure_chi(atom37, restype3)

            for i, chi_val in enumerate(chi_input):
                chi_name = f"chi{i + 1}"
                measured_val = measured.get(chi_name)
                if measured_val is not None:
                    err_rad = _compare_chi(chi_val, measured_val, restype3, i)
                    err_deg = math.degrees(err_rad)
                    if err_deg > max_err_deg:
                        max_err_deg = err_deg
                        worst = (restype3, chi_name, math.degrees(chi_val),
                                 math.degrees(measured_val), err_deg)

        print(f"\nMax single-angle error: {max_err_deg:.6f}°")
        if worst:
            print(f"  Worst case: {worst[0]} {worst[1]} "
                  f"input={worst[2]:.2f}° measured={worst[3]:.2f}° err={worst[4]:.6f}°")

        assert max_err_deg <= ROUNDTRIP_RMS_GATE_DEG, (
            f"Max single-angle error {max_err_deg:.4f}° > {ROUNDTRIP_RMS_GATE_DEG}°"
        )

    def test_symmetric_chi_handling(self):
        """Symmetric χ angles round-trip correctly via fold_symmetric_chi comparison.

        Tests Asp χ2 (pi_periodic_180), Glu χ3 (pi_periodic_180),
        Phe χ2 (aromatic_180), Tyr χ2 (aromatic_180).
        """
        # Use non-origin CA to avoid all-zero heuristic in measure_chi
        _off = np.array([3.0, 7.0, -1.5])
        N_bb = np.array([-0.521, 1.363, 0.0]) + _off
        CA_bb = np.array([0.0, 0.0, 0.0]) + _off
        C_bb = np.array([1.525, 0.0, 0.0]) + _off

        symmetric_cases = [
            # (restype3, chi_idx_0based, chi_input_deg)
            ("ASP", 1, 100.0),   # Asp chi2: pi_periodic_180
            ("GLU", 2, 120.0),   # Glu chi3: pi_periodic_180
            ("PHE", 1, 80.0),    # Phe chi2: aromatic_180
            ("TYR", 1, -60.0),   # Tyr chi2: aromatic_180
        ]

        for restype3, chi_idx, chi_deg in symmetric_cases:
            from proxide.chem.residues import restype_3to1
            rt_idx = restype_order[restype_3to1[restype3]]
            n_chi = int(sum(chi_angles_mask[rt_idx]))

            chi_vec = [0.0] * n_chi
            chi_vec[chi_idx] = math.radians(chi_deg)

            atom37, _ = chi_to_coords(restype3, N_bb, CA_bb, C_bb, chi_vec)
            measured = measure_chi(atom37, restype3)

            chi_name = f"chi{chi_idx + 1}"
            measured_val = measured.get(chi_name)
            assert measured_val is not None, (
                f"{restype3} {chi_name}: measure_chi returned None"
            )

            err_rad = _compare_chi(chi_vec[chi_idx], measured_val, restype3, chi_idx)
            err_deg = math.degrees(err_rad)
            assert err_deg <= ROUNDTRIP_RMS_GATE_DEG, (
                f"{restype3} {chi_name} (symmetric): "
                f"input={chi_deg}° measured={math.degrees(measured_val):.3f}° "
                f"folded_err={err_deg:.4f}° > {ROUNDTRIP_RMS_GATE_DEG}°"
            )


# ---------------------------------------------------------------------------
# Test 2: measure_chi matches biotite within ±0.01° (B2 re-assertion)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ubq_pdb_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Download 1UBQ (ubiquitin) to a temp directory."""
    tmp = tmp_path_factory.mktemp("structures_nerf")
    path = rcsb.fetch("1ubq", "pdb", str(tmp))
    return Path(path)


@pytest.fixture(scope="module")
def proxide_protein(ubq_pdb_path: Path):
    return parse_structure(str(ubq_pdb_path), use_jax=False)


@pytest.fixture(scope="module")
def biotite_struct(ubq_pdb_path: Path):
    f = pdb_io.PDBFile.read(str(ubq_pdb_path))
    return pdb_io.get_structure(f, model=1)


def _biotite_chi(bstruct, res_id: int, chain: str, quad: list[str]) -> float:
    res = bstruct[(bstruct.res_id == res_id) & (bstruct.chain_id == chain)]
    def _coord(name: str) -> np.ndarray:
        mask = res.atom_name == name
        coords = res.coord[mask]
        if len(coords) == 0:
            raise KeyError(f"Atom {name!r} not found in residue {res_id}")
        return coords[0]
    return float(bs.dihedral(*[_coord(a) for a in quad]))


def test_measure_chi_parity_with_biotite(proxide_protein, biotite_struct):
    """B2 re-assertion: measure_chi matches biotite ±0.01° on 1UBQ LEU residues.

    Uses biotite's own dihedral function on the same coordinates as proxide
    (same atoms, so any difference comes purely from dihedral convention).
    """
    from proxide.chem.residues import restypes, restype_1to3

    coords = np.array(proxide_protein.coordinates)   # (n_res, 37, 3)
    aatype = np.array(proxide_protein.aatype)         # (n_res,) — indices into restypes

    leu_code = restype_order["L"]
    leu_residues = np.where(aatype == leu_code)[0]
    assert len(leu_residues) > 0, "No LEU residues found in 1UBQ — fixture problem?"

    leu_chi_quads = CHI_ANGLES_ATOMS["LEU"]
    n_checked = 0

    for res_pos in leu_residues:
        res_coords = coords[res_pos]   # (37, 3)
        result = measure_chi(res_coords, "LEU")

        for chi_i, quad in enumerate(leu_chi_quads):
            chi_name = f"chi{chi_i + 1}"
            proxide_val = result.get(chi_name)
            if proxide_val is None:
                continue

            # Biotite reference: call biotite's dihedral on the same coordinates
            atom_coords = [res_coords[atom_order[a]] for a in quad]
            bio_val = float(bs.dihedral(*atom_coords))

            diff_deg = abs(math.degrees(_circular_diff(proxide_val, bio_val)))
            assert diff_deg <= BIOTITE_PARITY_DEG, (
                f"LEU res_pos={res_pos} {chi_name}: "
                f"proxide={math.degrees(proxide_val):.5f}° "
                f"biotite={math.degrees(bio_val):.5f}° "
                f"diff={diff_deg:.6f}° > {BIOTITE_PARITY_DEG}°"
            )
            n_checked += 1

    assert n_checked > 0, "No LEU chi angles checked — fixture problem?"
    print(
        f"\nB2 re-assertion: {n_checked} (LEU, chi_i) pairs checked against biotite, "
        f"all within ±{BIOTITE_PARITY_DEG}°"
    )
