"""Forward NeRF: χ angles + backbone (N, CA, C) → side-chain Cartesian coordinates.

This module places side-chain atoms in global Cartesian space from:
  - The three backbone atoms (N, CA, C) that define the residue's backbone frame.
  - A vector of χ angles (radians) for the side chain.

Algorithm
---------
The implementation uses the **rigid-group frame** decomposition from AlphaFold2
(Jumper et al. 2021, Supp. §1.8) as encoded in
``proxide.chem.residues.restype_rigid_group_default_frame``:

  1. Build a 4×4 backbone frame T_bb from N, CA, C (maps local backbone coords
     to global Cartesian space; CA is the origin, x-axis toward C).
  2. For each chi group *i* (1-indexed, stored as group 4+i-1):
       T_chi_i_global = T_bb × F_1 × … × F_i × Rot_x(χ_i)
     where F_j = restype_rigid_group_default_frame[restype, 4+j-1] is the
     *relative* default frame of chi_j with respect to its parent frame, and
     Rot_x(χ_i) is a rotation by χ_i around the x-axis of the chi_i frame.
  3. For each atom belonging to group i, its global position is:
       T_chi_i_global[:3,:3] @ atom_local_pos + T_chi_i_global[:3,3]

Atom positions and group assignments are stored in the pre-computed arrays
``restype_atom37_rigid_group_positions`` and ``restype_atom37_to_rigid_group``.

Design decisions
----------------
- Pure Python / NumPy: coords→χ (measure_chi) runs in data-prep pipelines, not
  the hot inference loop; NumPy precision suffices and avoids JAX compilation.
  PyO3/Rust binding deferred (see ADR 260623_coords_to_chi_path.md).
- float64 throughout to match measure_chi precision.
- Returns a dense atom37 array (shape (37, 3)) with zeros for absent atoms;
  an atom37 mask indicates which atoms were placed.
- Backbone atoms (N, CA, C, O, CB) are placed from the backbone group (group 0)
  and are always returned regardless of χ angles.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from proxide.chem.residues import (
    chi_angles_mask,
    restype_atom37_rigid_group_positions,
    restype_atom37_to_rigid_group,
    restype_order,
    restype_rigid_group_default_frame,
)
from proxide.chem.residues import (
    restype_atom37_mask as _restype_atom37_mask,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rot_x(angle: float) -> np.ndarray:
    """4×4 rotation matrix about the x-axis by *angle* radians."""
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array(
        [[1.0, 0.0, 0.0, 0.0],
         [0.0,  c,  -s,  0.0],
         [0.0,  s,   c,  0.0],
         [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _backbone_frame(
    N: np.ndarray,
    CA: np.ndarray,
    C: np.ndarray,
) -> np.ndarray:
    """Build a 4×4 transform from the *local backbone frame* to global space.

    The local backbone frame is:
      - Origin = CA
      - x-axis = (C − CA) / ‖C − CA‖
      - y-axis = component of (N − CA) orthogonal to x
      - z-axis = x × y

    Parameters
    ----------
    N, CA, C:
        Global Cartesian coordinates (shape (3,)) of the three backbone atoms.

    Returns
    -------
    np.ndarray
        4×4 rigid transformation matrix, float64.
    """
    N = np.asarray(N, dtype=np.float64)
    CA = np.asarray(CA, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)

    ex = C - CA
    ex_norm = np.linalg.norm(ex)
    if ex_norm < 1e-12:
        raise ValueError("CA and C are coincident — cannot build backbone frame.")
    ex = ex / ex_norm

    ey = N - CA
    ey = ey - np.dot(ey, ex) * ex
    ey_norm = np.linalg.norm(ey)
    if ey_norm < 1e-12:
        raise ValueError("N, CA, C are collinear — cannot build backbone frame.")
    ey = ey / ey_norm

    ez = np.cross(ex, ey)

    T = np.eye(4, dtype=np.float64)
    T[:3, 0] = ex
    T[:3, 1] = ey
    T[:3, 2] = ez
    T[:3, 3] = CA
    return T


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chi_to_coords(
    restype3: str,
    N: np.ndarray,
    CA: np.ndarray,
    C: np.ndarray,
    chi_angles_rad: Sequence[float | None],
) -> tuple[np.ndarray, np.ndarray]:
    """Place side-chain atoms given backbone atoms and χ angles.

    Parameters
    ----------
    restype3:
        Three-letter residue name, e.g. ``"LEU"``.
    N, CA, C:
        Global Cartesian coordinates of the three backbone atoms, shape (3,).
    chi_angles_rad:
        χ angles in radians, length ≤ 4.  None entries (or a shorter list) are
        treated as χ = 0.  Angles for chi groups that don't exist for this
        residue type are ignored.

    Returns
    -------
    atom37_positions : np.ndarray
        Shape (37, 3), float64.  Global coordinates for every atom present in
        the residue; absent atoms are filled with zeros.
    atom37_present : np.ndarray
        Shape (37,), bool.  True for atoms that were placed (have non-zero
        entries in the source residue-geometry tables).
    """
    if restype3 not in restype_order:
        # Try via restype_order which uses 1-letter; we accept 3-letter directly
        pass
    restype3 = restype3.upper()

    # Map restype3 → integer index (same ordering as restypes)
    from proxide.chem.residues import restype_3to1
    try:
        restype1 = restype_3to1[restype3]
    except KeyError as exc:
        raise ValueError(f"Unknown residue type: {restype3!r}") from exc
    rt = restype_order[restype1]

    # Build backbone frame
    T_bb = _backbone_frame(N, CA, C)

    # Accumulate per-group frames: T_group[i] = global frame for rigid group i
    # Groups: 0=backbone, 1=pre-omega(unused), 2=phi, 3=psi, 4=chi1, 5=chi2, ...
    default_frames = restype_rigid_group_default_frame[rt].astype(np.float64)  # (8,4,4)
    chi_mask = chi_angles_mask[rt]  # list of 4 floats: 1=exists, 0=absent

    # Pad chi_angles_rad to length 4, filling with 0.0
    chi_vals = [0.0] * 4
    for i, v in enumerate(chi_angles_rad):
        if i >= 4:
            break
        if v is not None:
            chi_vals[i] = float(v)

    # Compute global frame for each of the 8 rigid groups
    T_global = [np.eye(4, dtype=np.float64)] * 8

    # Group 0: backbone → global
    T_global[0] = T_bb @ default_frames[0]  # default_frames[0] = identity

    # Groups 1–3: phi/psi frames (we don't rotate these; chi angles only)
    for grp in (1, 2, 3):
        T_global[grp] = T_bb @ default_frames[grp]

    # Groups 4–7: chi1–chi4 frames
    # T_chi1_global = T_bb × F_chi1 × Rot_x(chi1)
    # T_chi2_global = T_chi1_global × F_chi2_rel_chi1 × Rot_x(chi2)
    # where F_chi2_rel_chi1 = restype_rigid_group_default_frame[rt, 5]
    # Note: chi2 default frame is expressed relative to chi1 (parent) frame, etc.
    # So T_chi_{i+1} = T_chi_i × F_chi_{i+1} × Rot_x(chi_{i+1})
    parent_frame = T_bb  # chi1 parent is backbone
    for chi_idx in range(4):
        grp = 4 + chi_idx
        if chi_mask[chi_idx] == 0.0:
            # No chi angle for this group — set to parent (won't be used)
            T_global[grp] = parent_frame @ default_frames[grp]
        else:
            chi_rad = chi_vals[chi_idx]
            T_global[grp] = parent_frame @ default_frames[grp] @ _rot_x(chi_rad)
        parent_frame = T_global[grp]

    # Place atoms
    local_positions = restype_atom37_rigid_group_positions[rt].astype(np.float64)  # (37,3)
    group_assignments = restype_atom37_to_rigid_group[rt]  # (37,)
    present_mask = _restype_atom37_mask[rt].astype(bool)  # (37,)

    atom37_positions = np.zeros((37, 3), dtype=np.float64)
    for slot in range(37):
        if not present_mask[slot]:
            continue
        grp = group_assignments[slot]
        local_pos = local_positions[slot]  # (3,)
        T = T_global[grp]
        atom37_positions[slot] = T[:3, :3] @ local_pos + T[:3, 3]

    return atom37_positions, present_mask
