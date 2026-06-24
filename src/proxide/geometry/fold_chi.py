"""Symmetry folding for symmetric χ angles.

This module provides utilities to fold symmetric dihedral angles onto their
canonical ranges, according to their point-group symmetry.

Symmetry classes (from proxide.chem.chi.CHI_SYMMETRY_CLASS):
  - 'none': No symmetry, pass through unchanged
  - 'pi_periodic_180': χ ≡ χ±180° (180° rotational symmetry)
    → fold onto [-π/2, π/2] (e.g., Asp χ2, Glu χ3)
  - 'aromatic_180': Ring 2-fold symmetry
    → fold onto [0, π/2] (e.g., Phe χ2, Tyr χ2)
  - 'non_rotameric_sp2': Continuous sp2 hybridization, not discretized
    → pass through unchanged (Asn χ2, Gln χ3, His χ2)

All arithmetic is float64; input angles are in radians.
"""

from __future__ import annotations

import numpy as np

from proxide.chem.chi import get_chi_symmetry_class


def fold_symmetric_chi(
    chi_value_rad: float,
    restype_3: str,
    chi_index: int,
) -> float:
    """Fold a χ angle onto its canonical range based on symmetry.

    This is a many-to-one map: multiple input angles fold to the same output.
    For example, under pi_periodic_180 symmetry, both χ and χ+180° fold to
    the same value in [-π/2, π/2].

    Parameters
    ----------
    chi_value_rad : float
        Dihedral angle in radians, typically in [-π, π].
    restype_3 : str
        Three-letter residue name, e.g. "ASP", "PHE".
    chi_index : int
        Zero-based χ index (0 = χ1, 1 = χ2, etc.).

    Returns
    -------
    float
        Folded angle in radians. For 'none' and 'non_rotameric_sp2',
        returns the input unchanged. For symmetric classes, returns a
        value in the canonical range:
          - pi_periodic_180: [-π/2, π/2]
          - aromatic_180: [0, π/2]

    Raises
    ------
    KeyError
        If (restype_3, chi_index) is not in CHI_SYMMETRY_CLASS.
    """
    chi_value_rad = float(chi_value_rad)
    symmetry_class = get_chi_symmetry_class(restype_3, chi_index)

    if symmetry_class == "none" or symmetry_class == "non_rotameric_sp2":
        # Pass through unchanged
        return chi_value_rad

    elif symmetry_class == "pi_periodic_180":
        # χ ≡ χ±180°; fold onto [-π/2, π/2]
        # Normalize χ to [-π, π] if needed
        chi_norm = _normalize_angle(chi_value_rad)
        # Shift to [0, π] and take mod π to get [0, π]
        chi_shifted = (chi_norm + np.pi / 2) % np.pi
        # Shift back to [-π/2, π/2]
        folded = chi_shifted - np.pi / 2
        return float(folded)

    elif symmetry_class == "aromatic_180":
        # Ring 2-fold symmetry; fold onto [0, π/2]
        # Normalize χ to [-π, π]
        chi_norm = _normalize_angle(chi_value_rad)
        # Fold negative values: if χ < 0, use χ + π (still in [-π/2, π/2] → [π/2, 3π/2])
        # Then clamp to [0, π/2] by taking min(χ, π - χ)
        if chi_norm < 0:
            chi_reflected = chi_norm + np.pi
        else:
            chi_reflected = chi_norm
        # Now chi_reflected is in [0, π]
        # Reflect to [0, π/2]: min(χ, π - χ)
        folded = min(chi_reflected, np.pi - chi_reflected)
        return float(folded)

    else:
        raise ValueError(
            f"Unknown symmetry class {symmetry_class!r} for "
            f"({restype_3}, {chi_index})"
        )


def _normalize_angle(angle_rad: float) -> float:
    """Normalize angle to [-π, π]."""
    # Reduce to [-2π, 2π]
    angle_rad = float(angle_rad) % (2 * np.pi)
    # Shift to [-π, π]
    if angle_rad > np.pi:
        angle_rad -= 2 * np.pi
    elif angle_rad < -np.pi:
        angle_rad += 2 * np.pi
    return angle_rad
