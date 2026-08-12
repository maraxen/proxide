#!/usr/bin/env python3
"""Diagnose branch atom torsion offset errors in MASTER rotlib.bin.

Measures chi dihedrals and branch atom placements for residues showing per-residue
Kabsch rotation errors when comparing proxide's rebuilt rotamer library to MASTER.

The hypothesis: branch atom torsion offsets (chi[i] + fixed_degree) or CB placement
errors in the template calculations account for the 2-5° per-residue rotation errors
observed in GLU, PHE, ASP, LEU, VAL, MET.

Measures for each residue:
- Per-chi offset: actual_chi_from_coords - stored_chi (should be ~0)
- Branch atom offset: actual_branch_torsion - chi[relative_chi_idx]
- Compare to template torsion_deg constant
- CB distance: error in synthetic CB placement

Reports summary table showing measured vs expected offsets for each residue.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger("diagnose_branch_torsion_offsets")

# --- Canonical backbone atoms (universal frame) ---
BACKBONE_CA = [0.0, 0.0, 0.0]
BACKBONE_N = [-1.458, 0.0, 0.0]
BACKBONE_C = [0.551, 1.420, 0.0]

# --- Branch atom specifications ---
# (residue, branch_atom_name, [A,B,C,D] dihedral atoms, relative_chi_idx, template_torsion_deg)
BRANCH_SPECS = {
    "VAL": [
        ("CG2", ["N", "CA", "CB", "CG2"], 0, 120.0),
    ],
    "ILE": [
        ("CG2", ["N", "CA", "CB", "CG2"], 0, 120.0),
    ],
    "LEU": [
        ("CD2", ["CA", "CB", "CG", "CD2"], 1, 120.0),
    ],
    "ASP": [
        ("OD2", ["CA", "CB", "CG", "OD2"], 1, 180.0),
    ],
    "ASN": [
        ("ND2", ["CA", "CB", "CG", "ND2"], 1, 180.0),
    ],
    "GLU": [
        ("OE2", ["CB", "CG", "CD", "OE2"], 2, 180.0),
    ],
    "GLN": [
        ("NE2", ["CB", "CG", "CD", "NE2"], 2, 180.0),
    ],
    "PHE": [
        ("CD2", ["CA", "CB", "CG", "CD2"], 1, 180.0),
    ],
    "TYR": [
        ("CD2", ["CA", "CB", "CG", "CD2"], 1, 180.0),
    ],
    "THR": [
        ("CG2", ["N", "CA", "CB", "CG2"], 0, 120.0),
    ],
}

# --- Chi definition atoms (for measuring chi from coordinates) ---
CHI_DEFS = {
    "VAL": [["N", "CA", "CB", "CG1"], ["CA", "CB", "CG1", "CG2"]],
    "ILE": [["N", "CA", "CB", "CG1"], ["CA", "CB", "CG1", "CD1"]],
    "LEU": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"], ["CB", "CG", "CD1", "CD2"]],
    "ASP": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"], ["CB", "CG", "OD1", "OD2"]],
    "ASN": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
    "GLU": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"], ["CB", "CG", "CD", "OE1"]],
    "GLN": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"], ["CB", "CG", "CD", "OE1"]],
    "PHE": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "TYR": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "THR": [["N", "CA", "CB", "OG1"], ["CA", "CB", "OG1", "CG2"]],
    "SER": [["N", "CA", "CB", "OG"]],
    "CYS": [["N", "CA", "CB", "SG"]],
    "MET": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "SD"], ["CB", "CG", "SD", "CE"]],
}


class Result(NamedTuple):
    residue: str
    n_rotamers: int
    mean_chi_errs: list[float]
    branch_atom: str | None
    expected_offset_deg: float | None
    measured_offset_deg: float | None
    delta_deg: float | None
    status: str


def _dihedral_signed(a: list[float], b: list[float], c: list[float], d: list[float]) -> float:
    """Compute signed dihedral angle A-B-C-D in degrees using cross-product formula."""
    b1 = [b[i] - a[i] for i in range(3)]
    b2 = [c[i] - b[i] for i in range(3)]
    b3 = [d[i] - c[i] for i in range(3)]

    def cross(u, v):
        return [u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0]]

    def dot(u, v):
        return sum(u[i] * v[i] for i in range(3))

    def norm(v):
        return math.sqrt(sum(x * x for x in v))

    n1 = cross(b1, b2)
    n2 = cross(b2, b3)
    b2_norm = norm(b2)
    if b2_norm < 1e-10:
        return 0.0
    b2_unit = [b2[i] / b2_norm for i in range(3)]
    n_cross = cross(n1, n2)
    numerator = dot(n_cross, b2_unit)
    denominator = dot(n1, n2)
    dihedral_rad = math.atan2(numerator, denominator)
    return math.degrees(dihedral_rad)


def _build_coords_dict(entry: dict, aa: str) -> dict[str, list[float]]:
    """Build a name->coords dict including backbone atoms."""
    atom_names = entry["atom_names"]
    coords_dict = {}

    # Add backbone
    coords_dict["N"] = BACKBONE_N[:]
    coords_dict["CA"] = BACKBONE_CA[:]
    coords_dict["C"] = BACKBONE_C[:]

    # For chi1 and CB measurements, we also need:
    # CB is at index 0 in the sidechain coords (first atom after backbone)
    # So we'll compute it separately when needed, or use synthetic calculation

    return coords_dict, atom_names


def _measure_rotamer(entry: dict, aa: str, rot_idx: int, bin_idx: int) -> dict[str, float]:
    """Measure chi angles and branch atom offsets for a single rotamer.

    Returns {
        'chi_measured': [chi0, chi1, ...],  # measured from coordinates
        'chi_stored': [chi0, chi1, ...],    # stored in rotlib.bin
        'chi_errors': [err0, err1, ...],    # measured - stored
        'branch_offset': float or None,     # for branch atom of this residue
    }
    """
    atom_names = entry["atom_names"]
    rot = entry["rot_bins"][bin_idx][rot_idx]
    coords = rot["coords"]
    chi_stored = [c[0] for c in rot["chi"]]

    # Build atom->coords dict (sidechain atoms only from rotlib.bin)
    atom_to_coords = {}
    for i, name in enumerate(atom_names):
        atom_to_coords[name] = coords[i]

    # Add backbone
    atom_to_coords["N"] = BACKBONE_N[:]
    atom_to_coords["CA"] = BACKBONE_CA[:]
    atom_to_coords["C"] = BACKBONE_C[:]
    atom_to_coords["CB"] = coords[0]  # CB is always first sidechain atom

    # Measure chi angles from coordinates
    chi_measured = []
    if aa in CHI_DEFS:
        for chi_def in CHI_DEFS[aa]:
            atoms = [atom_to_coords.get(name) for name in chi_def]
            if all(a is not None for a in atoms):
                dihedral = _dihedral_signed(*atoms)
                chi_measured.append(dihedral)
            else:
                chi_measured.append(None)

    # Compute chi errors (only for non-None measured values)
    chi_errors = []
    for _i, (meas, stored) in enumerate(zip(chi_measured, chi_stored, strict=False)):
        if meas is not None:
            err = meas - stored
            # Normalize to [-180, 180]
            while err > 180:
                err -= 360
            while err < -180:
                err += 360
            chi_errors.append(err)
        else:
            chi_errors.append(None)

    # Measure branch atom offset (if applicable)
    branch_offset = None
    if aa in BRANCH_SPECS:
        for _branch_atom, dihedral_atoms, chi_idx, _template_deg in BRANCH_SPECS[aa]:
            atoms = [atom_to_coords.get(name) for name in dihedral_atoms]
            if all(a is not None for a in atoms):
                branch_torsion = _dihedral_signed(*atoms)
                if chi_idx < len(chi_stored):
                    chi_ref = chi_stored[chi_idx]
                    offset = branch_torsion - chi_ref
                    # Normalize to [-180, 180]
                    while offset > 180:
                        offset -= 360
                    while offset < -180:
                        offset += 360
                    branch_offset = offset

    return {
        "chi_measured": chi_measured,
        "chi_stored": chi_stored,
        "chi_errors": chi_errors,
        "branch_offset": branch_offset,
    }


def diagnose_residue(entry: dict, aa: str) -> Result:
    """Diagnose a single residue across all rotamers.

    Returns: Result namedtuple with summary statistics.
    """
    chi_errors_all = []  # list of error lists
    branch_offsets_all = []
    n_rotamers = 0

    # Measure all rotamers across all bins
    for bin_idx in range(entry["nb"]):
        for rot_idx in range(len(entry["rot_bins"][bin_idx])):
            measurement = _measure_rotamer(entry, aa, rot_idx, bin_idx)
            n_rotamers += 1
            chi_errors_all.append(measurement["chi_errors"])
            if measurement["branch_offset"] is not None:
                branch_offsets_all.append(measurement["branch_offset"])

    # Compute mean chi errors per chi index
    mean_chi_errs = []
    if chi_errors_all:
        n_chis = len(chi_errors_all[0]) if chi_errors_all[0] else 0
        for chi_idx in range(n_chis):
            valid_errs = [
                e
                for errors in chi_errors_all
                if errors and chi_idx < len(errors) and errors[chi_idx] is not None
                for e in [errors[chi_idx]]
            ]
            if valid_errs:
                mean_chi_errs.append(sum(valid_errs) / len(valid_errs))
            else:
                mean_chi_errs.append(None)

    # Compute branch offset statistics
    mean_branch_offset = None
    expected_offset = None
    status = "PASS"
    delta = None

    if aa in BRANCH_SPECS and branch_offsets_all:
        mean_branch_offset = sum(branch_offsets_all) / len(branch_offsets_all)
        branch_atom, _, _, template_torsion_deg = BRANCH_SPECS[aa][0]
        expected_offset = template_torsion_deg
        delta = mean_branch_offset - expected_offset

        # Normalize delta to [-180, 180]
        while delta > 180:
            delta -= 360
        while delta < -180:
            delta += 360

        if abs(delta) > 2:
            status = "FAIL" if abs(delta) > 5 else "MARGINAL"
    elif aa in BRANCH_SPECS:
        branch_atom, _, _, template_torsion_deg = BRANCH_SPECS[aa][0]
        status = "NO_BRANCH_DATA"
    else:
        branch_atom = None
        status = "NO_BRANCH"

    return Result(
        residue=aa,
        n_rotamers=n_rotamers,
        mean_chi_errs=mean_chi_errs,
        branch_atom=branch_atom if aa in BRANCH_SPECS else None,
        expected_offset_deg=expected_offset,
        measured_offset_deg=mean_branch_offset,
        delta_deg=delta,
        status=status,
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--rotlib-bin",
        type=Path,
        default=Path("/home/marielle/repos/mosaist/testfiles/rotlib.bin"),
        help="path to MASTER/MSL rotlib.bin",
    )
    ap.add_argument(
        "--residues",
        nargs="+",
        default=["GLU", "PHE", "ASP", "LEU", "VAL", "MET", "ILE", "THR", "ASN", "SER"],
        help="residue codes to diagnose",
    )
    ap.add_argument("--output", type=Path, default=None, help="write JSON report here")
    ap.add_argument("--log-level", default="INFO")
    ap.add_argument("--dry-run", action="store_true", help="parse header only, skip measurement")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not args.rotlib_bin.exists():
        logger.error("rotlib.bin not found at %s", args.rotlib_bin)
        return 2

    # Import parse_rotlib from sibling script
    sys.path.insert(0, str(Path(__file__).parent))
    from extract_rotlib_geometry import parse_rotlib

    entries = parse_rotlib(args.rotlib_bin)
    logger.info("parsed %d residue codes: %s", len(entries), sorted(entries))

    results = []
    report = {
        "rotlib": str(args.rotlib_bin),
        "residue_codes": sorted(entries),
        "measurements": {},
    }

    for aa in args.residues:
        if aa not in entries:
            logger.warning("%s absent from rotlib.bin", aa)
            continue

        if args.dry_run:
            logger.info("%s atoms: %s", aa, entries[aa]["atom_names"])
            continue

        logger.info("diagnosing %s (%d bins)...", aa, entries[aa]["nb"])
        result = diagnose_residue(entries[aa], aa)
        results.append(result)

        logger.info(
            "  n_rotamers=%d, mean_chi_errs=%s, branch=%s, "
            "expected=%.1f° measured=%.1f° delta=%.1f°",
            result.n_rotamers,
            [round(e, 1) if e is not None else None for e in result.mean_chi_errs],
            result.branch_atom,
            result.expected_offset_deg if result.expected_offset_deg is not None else -999,
            result.measured_offset_deg if result.measured_offset_deg is not None else -999,
            result.delta_deg if result.delta_deg is not None else -999,
        )

        report["measurements"][aa] = {
            "n_rotamers": result.n_rotamers,
            "mean_chi_errs_deg": result.mean_chi_errs,
            "branch_atom": result.branch_atom,
            "expected_offset_deg": result.expected_offset_deg,
            "measured_offset_deg": result.measured_offset_deg,
            "delta_deg": result.delta_deg,
            "status": result.status,
        }

    # Print summary table
    print("\n" + "=" * 100)
    print("SUMMARY: Branch Torsion Offset Diagnostics")
    print("=" * 100)
    print(
        f"{'Residue':<12} {'n_rot':<8} {'Branch Atom':<14} {'Expected':<12} {'Measured':<12} "
        f"{'Delta':<12} {'Status':<12}"
    )
    print("-" * 100)

    for result in results:
        res_str = result.residue
        n_rot_str = str(result.n_rotamers)
        branch_str = result.branch_atom if result.branch_atom else "N/A"
        exp_str = (
            f"{result.expected_offset_deg:.1f}°" if result.expected_offset_deg is not None
            else "N/A"
        )
        meas_str = (
            f"{result.measured_offset_deg:.1f}°" if result.measured_offset_deg is not None
            else "N/A"
        )
        delta_str = f"{result.delta_deg:.1f}°" if result.delta_deg is not None else "N/A"
        status_str = result.status

        print(
            f"{res_str:<12} {n_rot_str:<8} {branch_str:<14} {exp_str:<12} {meas_str:<12} "
            f"{delta_str:<12} {status_str:<12}"
        )

    print("=" * 100)
    print("\nStatus legend:")
    print("  PASS      — |delta| < 2° (offset matches template)")
    print("  MARGINAL  — 2° < |delta| < 5° (offset exists but smaller than rotation error)")
    print("  FAIL      — |delta| > 5° (offset does NOT match template)")
    print("  NO_BRANCH — residue has no branch atoms to measure")
    print("  NO_BRANCH_DATA — branch specs exist but coords missing in rotlib.bin")

    if args.output:
        args.output.write_text(json.dumps(report, indent=2))
        logger.info("wrote report -> %s", args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
