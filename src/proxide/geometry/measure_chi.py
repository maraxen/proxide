"""coords→χ angle measurement for protein residues.

All arithmetic is float64. Missing atoms propagate as None.

Sign convention
---------------
Matches IUPAC / biotite.structure.dihedral:
    v1 = p1 - p0  (normalised)
    v2 = p2 - p1  (normalised)
    v3 = p3 - p2  (normalised)
    n1 = cross(v1, v2)
    n2 = cross(v2, v3)
    angle = atan2(dot(cross(n1, n2), v2), dot(n1, n2))

This is the standard Prelog/IUPAC torsion convention; biotite uses the same
formula (see biotite.structure.geometry.dihedral source).  A common bug is
using atan2(dot(m, n2), dot(n1, n2)) with m = cross(n1, b2/||b2||) which
gives the opposite sign — confirmed fixed here.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from proxide.chem.chi import CHI_ANGLES_ATOMS
from proxide.chem.residues import atom_order


def dihedral(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
) -> float:
    """Compute the torsion angle defined by four 3D points.

    Uses the IUPAC / biotite sign convention.  All inputs are cast to f64.

    Parameters
    ----------
    p0, p1, p2, p3:
        Cartesian coordinates of the four atoms, shape (3,).

    Returns
    -------
    float
        Dihedral angle in radians, in [-π, π].
    """
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    p3 = np.asarray(p3, dtype=np.float64)

    v1 = p1 - p0
    v2 = p2 - p1
    v3 = p3 - p2

    n1v2 = np.linalg.norm(v1)
    n2v2 = np.linalg.norm(v2)
    n3v2 = np.linalg.norm(v3)

    if n1v2 == 0.0 or n2v2 == 0.0 or n3v2 == 0.0:
        return float("nan")

    v1 = v1 / n1v2
    v2 = v2 / n2v2
    v3 = v3 / n3v2

    n1 = np.cross(v1, v2)
    n2 = np.cross(v2, v3)

    x = np.dot(n1, n2)
    y = np.dot(np.cross(n1, n2), v2)
    return float(np.arctan2(y, x))


# ---------------------------------------------------------------------------
# Per-residue measurement
# ---------------------------------------------------------------------------

def measure_chi(
    residue_positions: np.ndarray,
    restype3: str,
    atom_name_to_index: dict[str, int] | None = None,
) -> dict[str, Optional[float]]:
    """Measure all χ angles for a single residue.

    Parameters
    ----------
    residue_positions:
        Atom coordinates for this residue, shape (37, 3), in **atom37** order
        (the slot ordering defined by ``proxide.chem.residues.atom_order``).
        Atoms absent from the structure should have coordinates set to zeros
        *and* be absent from the atom mask — but this function checks for NaN
        or all-zero coordinates as a secondary guard.  Pass ``atom_mask`` via
        ``missing_atom_slots`` if you have it (preferred).
    restype3:
        Three-letter residue name, e.g. ``"LEU"``.
    atom_name_to_index:
        Mapping from atom name to slot index.  Defaults to the global
        ``proxide.chem.residues.atom_order`` (atom37 ordering).  Override
        for atom14 or custom layouts.

    Returns
    -------
    dict
        Keys ``"chi1"`` … ``"chiN"`` (only χ angles defined for this residue
        type).  Value is ``float`` (radians, f64) or ``None`` if any of the
        four defining atoms is missing (zero coordinates or not in the
        mapping).
    """
    if atom_name_to_index is None:
        atom_name_to_index = atom_order

    chi_quads = CHI_ANGLES_ATOMS.get(restype3, [])
    result: dict[str, Optional[float]] = {}
    positions = np.asarray(residue_positions, dtype=np.float64)

    for chi_idx, quad in enumerate(chi_quads):
        chi_name = f"chi{chi_idx + 1}"

        # Resolve each atom name → coordinate
        pts: list[np.ndarray] = []
        missing = False
        for atom_name in quad:
            slot = atom_name_to_index.get(atom_name)
            if slot is None:
                missing = True
                break
            coord = positions[slot]
            # Treat all-zero coordinate as absent (atom37 fill value)
            if np.all(coord == 0.0) or np.any(np.isnan(coord)):
                missing = True
                break
            pts.append(coord)

        if missing:
            result[chi_name] = None
        else:
            result[chi_name] = dihedral(*pts)

    return result


# ---------------------------------------------------------------------------
# Vectorised variant (optional, nice-to-have)
# ---------------------------------------------------------------------------

def measure_chi_vectorized(
    all_positions: np.ndarray,
    restypes: np.ndarray | list[str],
    atom_mask: np.ndarray | None = None,
    atom_name_to_index: dict[str, int] | None = None,
) -> list[dict[str, Optional[float]]]:
    """Measure χ angles for every residue in a protein.

    Parameters
    ----------
    all_positions:
        Shape ``(n_res, 37, 3)``, atom37 coordinates.
    restypes:
        Per-residue 3-letter codes, length ``n_res``.
    atom_mask:
        Optional boolean/float mask, shape ``(n_res, 37)``.  Slot ``1`` =
        present.  When provided, overrides the all-zero heuristic.
    atom_name_to_index:
        Atom name → slot index mapping (defaults to ``atom_order``).

    Returns
    -------
    list of dict
        One dict per residue (same format as ``measure_chi``).
    """
    if atom_name_to_index is None:
        atom_name_to_index = atom_order

    all_positions = np.asarray(all_positions, dtype=np.float64)
    n_res = all_positions.shape[0]

    results = []
    for i in range(n_res):
        res_positions = all_positions[i]

        # Build per-residue mask-aware coordinate accessor
        if atom_mask is not None:
            res_mask = np.asarray(atom_mask[i], dtype=bool)
            # Zero out absent atoms so measure_chi all-zero check fires
            masked_positions = res_positions.copy()
            masked_positions[~res_mask] = 0.0
            results.append(
                measure_chi(masked_positions, restypes[i], atom_name_to_index)
            )
        else:
            results.append(
                measure_chi(res_positions, restypes[i], atom_name_to_index)
            )

    return results
