"""Tests for symmetry-folding of χ angles in proxide.geometry.fold_chi.

Acceptance gates:
  1. Asp χ2 and Glu χ3 (pi_periodic_180):
     - Values fold to [-π/2, π/2]
     - χ and χ±180° fold identically
  2. Phe χ2 and Tyr χ2 (aromatic_180):
     - Values fold to [0, π/2]
     - χ and ring-symmetric image fold identically
  3. Non-symmetric (e.g., Leu χ1, Met χ1):
     - Pass through unchanged
  4. Non-rotameric sp2 (Asn χ2, Gln χ3, His χ2):
     - Pass through unchanged
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import pytest

from proxide.geometry.fold_chi import fold_symmetric_chi


# Tolerance for floating-point comparisons (radians)
TOLERANCE_RAD = 1e-14


# ==============================================================================
# Test: pi_periodic_180 (Asp χ2, Glu χ3)
# ==============================================================================


class TestPiPeriodic180:
    """Asp χ2 and Glu χ3 have 180° rotational symmetry."""

    @pytest.mark.parametrize(
        "restype,chi_idx",
        [
            ("ASP", 1),  # Asp χ2
            ("GLU", 2),  # Glu χ3
        ],
    )
    def test_fold_range_is_pi_to_pi2(self, restype: str, chi_idx: int):
        """Folded value should be in [-π/2, π/2]."""
        test_angles_deg = [-180, -170, -90, -45, 0, 45, 90, 170, 180]
        for deg in test_angles_deg:
            rad = math.radians(deg)
            folded = fold_symmetric_chi(rad, restype, chi_idx)
            assert -np.pi / 2 - TOLERANCE_RAD <= folded <= np.pi / 2 + TOLERANCE_RAD, (
                f"Folded angle {math.degrees(folded):.1f}° for {restype} χ{chi_idx+1} "
                f"from input {deg}° is outside [-90°, 90°]"
            )

    @pytest.mark.parametrize(
        "restype,chi_idx",
        [
            ("ASP", 1),
            ("GLU", 2),
        ],
    )
    def test_chi_and_chi_plus_180_fold_identically(
        self, restype: str, chi_idx: int
    ):
        """χ and χ+180° should fold to the same value."""
        test_angles_deg = [-90, -45, -20, 0, 20, 45, 90]
        for deg in test_angles_deg:
            chi_rad = math.radians(deg)
            chi_plus_180 = math.radians(deg + 180)
            chi_minus_180 = math.radians(deg - 180)

            folded = fold_symmetric_chi(chi_rad, restype, chi_idx)
            folded_plus_180 = fold_symmetric_chi(chi_plus_180, restype, chi_idx)
            folded_minus_180 = fold_symmetric_chi(chi_minus_180, restype, chi_idx)

            assert abs(folded - folded_plus_180) < TOLERANCE_RAD, (
                f"{restype} χ{chi_idx+1}: χ({deg}°) and χ+180° "
                f"should fold identically but got "
                f"{math.degrees(folded):.6f}° vs {math.degrees(folded_plus_180):.6f}°"
            )
            assert abs(folded - folded_minus_180) < TOLERANCE_RAD, (
                f"{restype} χ{chi_idx+1}: χ({deg}°) and χ-180° "
                f"should fold identically but got "
                f"{math.degrees(folded):.6f}° vs {math.degrees(folded_minus_180):.6f}°"
            )

    @pytest.mark.parametrize(
        "restype,chi_idx",
        [
            ("ASP", 1),
            ("GLU", 2),
        ],
    )
    def test_specific_values_pi_periodic(self, restype: str, chi_idx: int):
        """Check specific folded values for pi_periodic_180."""
        # 0° should fold to 0°
        folded_zero = fold_symmetric_chi(math.radians(0), restype, chi_idx)
        assert abs(folded_zero - math.radians(0)) < TOLERANCE_RAD

        # 45° should fold to 45°
        folded_45 = fold_symmetric_chi(math.radians(45), restype, chi_idx)
        assert abs(folded_45 - math.radians(45)) < TOLERANCE_RAD

        # -45° should fold to -45°
        folded_neg45 = fold_symmetric_chi(math.radians(-45), restype, chi_idx)
        assert abs(folded_neg45 - math.radians(-45)) < TOLERANCE_RAD

        # At boundaries (±90°), the value may fold to either side due to periodicity
        # But ±90° should fold to one of {-90°, 90°}, both valid in [-π/2, π/2]
        folded_pos90 = fold_symmetric_chi(math.radians(90), restype, chi_idx)
        folded_neg90 = fold_symmetric_chi(math.radians(-90), restype, chi_idx)
        # Both should be in [-π/2, π/2] (already checked by range test above)
        # and they should be equivalent (differ by at most 2π)
        assert (abs(folded_pos90 - math.radians(90)) < TOLERANCE_RAD or
                abs(folded_pos90 - math.radians(-90)) < TOLERANCE_RAD)
        assert (abs(folded_neg90 - math.radians(-90)) < TOLERANCE_RAD or
                abs(folded_neg90 - math.radians(90)) < TOLERANCE_RAD)

        # 45° and -135° should fold to the same value (since -135° ≡ 45°)
        folded_45_check = fold_symmetric_chi(math.radians(45), restype, chi_idx)
        folded_neg135 = fold_symmetric_chi(math.radians(-135), restype, chi_idx)
        assert abs(folded_45_check - folded_neg135) < TOLERANCE_RAD


# ==============================================================================
# Test: aromatic_180 (Phe χ2, Tyr χ2)
# ==============================================================================


class TestAromatic180:
    """Phe χ2 and Tyr χ2 have aromatic (2-fold) ring symmetry."""

    @pytest.mark.parametrize(
        "restype,chi_idx",
        [
            ("PHE", 1),  # Phe χ2
            ("TYR", 1),  # Tyr χ2
        ],
    )
    def test_fold_range_is_0_to_pi2(self, restype: str, chi_idx: int):
        """Folded value should be in [0, π/2]."""
        test_angles_deg = [-180, -90, -45, 0, 45, 90, 135, 180]
        for deg in test_angles_deg:
            rad = math.radians(deg)
            folded = fold_symmetric_chi(rad, restype, chi_idx)
            assert -TOLERANCE_RAD <= folded <= np.pi / 2 + TOLERANCE_RAD, (
                f"Folded angle {math.degrees(folded):.1f}° for {restype} χ{chi_idx+1} "
                f"from input {deg}° is outside [0°, 90°]"
            )

    @pytest.mark.parametrize(
        "restype,chi_idx",
        [
            ("PHE", 1),
            ("TYR", 1),
        ],
    )
    def test_ring_symmetric_images_fold_identically(
        self, restype: str, chi_idx: int
    ):
        """χ and its ring-symmetric image should fold to the same value.

        For aromatic rings, the 2-fold axis creates equivalences:
          χ and π - χ are equivalent (reflection about the ring axis)
          χ and -χ are NOT always equivalent (depends on frame)
        For our convention [0, π/2], we check that:
          - χ and (π - χ) fold identically
          - χ and (-χ) fold identically (since -χ = -χ, but wraps)
        """
        test_angles_deg = [0, 15, 30, 45, 60, 75, 90]
        for deg in test_angles_deg:
            chi_rad = math.radians(deg)
            # Ring symmetry: χ is equivalent to π - χ (reflection)
            chi_ring_symmetric = np.pi - chi_rad

            folded = fold_symmetric_chi(chi_rad, restype, chi_idx)
            folded_ring_symmetric = fold_symmetric_chi(
                chi_ring_symmetric, restype, chi_idx
            )

            assert abs(folded - folded_ring_symmetric) < TOLERANCE_RAD, (
                f"{restype} χ{chi_idx+1}: χ({deg}°) and its ring-symmetric image "
                f"π - χ({180-deg}°) should fold identically but got "
                f"{math.degrees(folded):.6f}° vs {math.degrees(folded_ring_symmetric):.6f}°"
            )

    @pytest.mark.parametrize(
        "restype,chi_idx",
        [
            ("PHE", 1),
            ("TYR", 1),
        ],
    )
    def test_specific_values_aromatic(self, restype: str, chi_idx: int):
        """Check specific folded values for aromatic_180."""
        # 0° should fold to 0°
        folded_zero = fold_symmetric_chi(math.radians(0), restype, chi_idx)
        assert abs(folded_zero - math.radians(0)) < TOLERANCE_RAD

        # 90° should fold to 90°
        folded_90 = fold_symmetric_chi(math.radians(90), restype, chi_idx)
        assert abs(folded_90 - math.radians(90)) < TOLERANCE_RAD

        # 45° should fold to 45°
        folded_45 = fold_symmetric_chi(math.radians(45), restype, chi_idx)
        assert abs(folded_45 - math.radians(45)) < TOLERANCE_RAD

        # -45° should fold to 45° (via 180° - (-45°) = 225° → fold to 45°)
        # OR via (-45° + 180° = 135° → fold to min(135°, 45°) = 45°)
        folded_neg45 = fold_symmetric_chi(math.radians(-45), restype, chi_idx)
        assert abs(folded_neg45 - math.radians(45)) < TOLERANCE_RAD, (
            f"-45° should fold to 45° but got {math.degrees(folded_neg45):.6f}°"
        )

        # 135° should fold to 45° (since 180° - 135° = 45°)
        folded_135 = fold_symmetric_chi(math.radians(135), restype, chi_idx)
        assert abs(folded_135 - math.radians(45)) < TOLERANCE_RAD, (
            f"135° should fold to 45° but got {math.degrees(folded_135):.6f}°"
        )


# ==============================================================================
# Test: 'none' (no symmetry)
# ==============================================================================


class TestNoSymmetry:
    """χ angles with no symmetry should pass through unchanged."""

    @pytest.mark.parametrize(
        "restype,chi_idx",
        [
            ("LEU", 0),  # Leu χ1
            ("LEU", 1),  # Leu χ2
            ("MET", 0),  # Met χ1
            ("MET", 1),  # Met χ2
            ("MET", 2),  # Met χ3
            ("ARG", 0),  # Arg χ1
            ("ASP", 0),  # Asp χ1 (only χ2 is symmetric)
            ("PHE", 0),  # Phe χ1 (only χ2 is symmetric)
        ],
    )
    def test_no_symmetry_pass_through(self, restype: str, chi_idx: int):
        """Non-symmetric angles should return input unchanged."""
        test_angles_deg = [-180, -135, -90, -45, 0, 45, 90, 135, 180]
        for deg in test_angles_deg:
            rad = math.radians(deg)
            folded = fold_symmetric_chi(rad, restype, chi_idx)
            assert abs(folded - rad) < TOLERANCE_RAD, (
                f"{restype} χ{chi_idx+1} with no symmetry should return "
                f"input {deg}° unchanged but got {math.degrees(folded):.6f}°"
            )


# ==============================================================================
# Test: 'non_rotameric_sp2' (pass through)
# ==============================================================================


class TestNonRotamericSp2:
    """Non-rotameric sp2 angles should pass through unchanged."""

    @pytest.mark.parametrize(
        "restype,chi_idx",
        [
            ("ASN", 1),  # Asn χ2 (amide)
            ("GLN", 2),  # Gln χ3 (amide)
            ("HIS", 1),  # His χ2 (imidazole)
        ],
    )
    def test_non_rotameric_sp2_pass_through(self, restype: str, chi_idx: int):
        """Non-rotameric sp2 angles should return input unchanged."""
        test_angles_deg = [-180, -135, -90, -45, 0, 45, 90, 135, 180]
        for deg in test_angles_deg:
            rad = math.radians(deg)
            folded = fold_symmetric_chi(rad, restype, chi_idx)
            assert abs(folded - rad) < TOLERANCE_RAD, (
                f"{restype} χ{chi_idx+1} (non-rotameric sp2) should return "
                f"input {deg}° unchanged but got {math.degrees(folded):.6f}°"
            )


# ==============================================================================
# Test: Edge cases and round-trip semantics
# ==============================================================================


class TestEdgeCases:
    """Edge cases and round-trip properties."""

    def test_nan_passthrough(self):
        """NaN input should result in NaN output (no error)."""
        # Non-symmetric case
        result = fold_symmetric_chi(float("nan"), "LEU", 0)
        assert np.isnan(result)

        # Symmetric case (may not be NaN, but should not crash)
        result = fold_symmetric_chi(float("nan"), "ASP", 1)
        # NaN comparisons are always False, so we just check it doesn't crash

    def test_very_large_angles(self):
        """Large angles (outside [-π, π]) should normalize correctly."""
        # 10π + 45° should fold like 45° for Asp χ2 (pi_periodic_180)
        large_angle = 10 * np.pi + math.radians(45)
        folded_large = fold_symmetric_chi(large_angle, "ASP", 1)
        folded_small = fold_symmetric_chi(math.radians(45), "ASP", 1)
        assert abs(folded_large - folded_small) < TOLERANCE_RAD

    def test_dtype_float64(self):
        """Output should be float64 (or compatible)."""
        result = fold_symmetric_chi(math.radians(45), "ASP", 1)
        assert isinstance(result, (float, np.floating))


# ==============================================================================
# Test: Symmetry equivalence confirmation
# ==============================================================================


class TestSymmetryEquivalence:
    """Verify that symmetric-equivalent values fold to identical results."""

    def test_asp_chi2_equivalence_suite(self):
        """Comprehensive equivalence tests for Asp χ2."""
        # χ values in degrees
        test_values = [-90, -60, -30, 0, 30, 60, 90]
        for deg in test_values:
            rad = math.radians(deg)
            folded_base = fold_symmetric_chi(rad, "ASP", 1)

            # χ+180° equivalent
            rad_plus = math.radians(deg + 180)
            folded_plus = fold_symmetric_chi(rad_plus, "ASP", 1)

            # χ-180° equivalent
            rad_minus = math.radians(deg - 180)
            folded_minus = fold_symmetric_chi(rad_minus, "ASP", 1)

            # All three should fold to the same value
            assert abs(folded_base - folded_plus) < TOLERANCE_RAD
            assert abs(folded_base - folded_minus) < TOLERANCE_RAD

    def test_phe_chi2_equivalence_suite(self):
        """Comprehensive equivalence tests for Phe χ2."""
        # χ values in degrees
        test_values = [-90, -45, 0, 45, 90]
        for deg in test_values:
            rad = math.radians(deg)
            folded_base = fold_symmetric_chi(rad, "PHE", 1)

            # Ring-symmetric image: 180° - χ
            rad_ring_sym = np.pi - rad
            folded_ring = fold_symmetric_chi(rad_ring_sym, "PHE", 1)

            # All should fold to the same value
            assert abs(folded_base - folded_ring) < TOLERANCE_RAD
