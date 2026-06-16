#!/usr/bin/env python3
"""Measure MASTER rotlib.bin's baked C-N-CA-CB improper dihedral.

Backlog #884: Determine whether MASTER's canonical CB orientation (the C-N-CA-CB
improper dihedral) differs from proxide's fixed placeholder of -119.7 deg.

This is MEASUREMENT ONLY — do NOT edit template.rs or charmm_ic.rs.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

# Add scripts/analysis to path to import parse_rotlib
sys.path.insert(0, str(Path(__file__).parent))

from extract_rotlib_geometry import parse_rotlib

logger = logging.getLogger("measure_master_cb_improper")

# Canonical backbone frame atoms in proxide's test convention
IDENTITY_N = [-1.0, 0.0, 0.0]
IDENTITY_CA = [0.0, 0.0, 0.0]
IDENTITY_C = [0.0, 1.0, 0.0]

def dihedral_iupac(a, b, c, d):
    """Compute dihedral angle A-B-C-D using IUPAC standard convention.
    
    Uses the atan2-based formula from proxide-geometry/src/geometry/angles.rs.
    Returns angle in radians in range [-π, π].
    """
    b1 = [b[i] - a[i] for i in range(3)]
    b2 = [c[i] - b[i] for i in range(3)]
    b3 = [d[i] - c[i] for i in range(3)]

    def cross(u, v):
        return [
            u[1]*v[2] - u[2]*v[1],
            u[2]*v[0] - u[0]*v[2],
            u[0]*v[1] - u[1]*v[0],
        ]

    def dot(u, v):
        return sum(u[i]*v[i] for i in range(3))

    def magnitude(v):
        return math.sqrt(sum(x*x for x in v))

    def normalize(v):
        mag = magnitude(v)
        if mag < 1e-6:
            return [0.0, 0.0, 0.0]
        return [x / mag for x in v]

    n1 = cross(b1, b2)
    n2 = cross(b2, b3)
    b2_norm = normalize(b2)
    m1 = cross(n1, b2_norm)

    x = dot(n1, n2)
    y = dot(m1, n2)

    return math.atan2(y, x)

def dihedral_iupac_deg(a, b, c, d):
    """Compute dihedral angle A-B-C-D in degrees (IUPAC)."""
    rad = dihedral_iupac(a, b, c, d)
    return math.degrees(rad)

def dihedral_nerf_deg(a, b, c, d):
    """Compute dihedral angle A-B-C-D in degrees (NeRF/torsion convention).
    
    NeRF uses phi directly in cos(phi) and sin(phi), so the sign is opposite
    to the IUPAC standard formula.
    """
    return -dihedral_iupac_deg(a, b, c, d)

def _self_test():
    """BATHOS measurement-verification: verify dihedral function."""
    logger.info("Running --self-test...")

    a = [0.0, 1.0, 0.0]
    b = [0.0, 0.0, 0.0]
    c = [1.0, 0.0, 0.0]
    d = [1.0, 0.0, 1.0]
    angle_deg = dihedral_iupac_deg(a, b, c, d)
    is_90 = abs(abs(angle_deg) - 90.0) < 0.01
    logger.info("Test 1 (perpendicular): %.2f deg (expect ±90) %s",
                angle_deg, "PASS" if is_90 else "FAIL")

    if not is_90:
        logger.error("SELF-TEST FAILED")
        return False

    a = [0.0, 1.0, 0.0]
    b = [0.0, 0.0, 0.0]
    c = [1.0, 0.0, 0.0]
    d = [1.0, math.cos(math.radians(60)), math.sin(math.radians(60))]
    angle_deg = dihedral_iupac_deg(a, b, c, d)
    is_60 = abs(abs(angle_deg) - 60.0) < 0.01
    logger.info("Test 2 (60 deg): %.2f deg (expect ±60) %s",
                angle_deg, "PASS" if is_60 else "FAIL")

    if not is_60:
        logger.error("SELF-TEST FAILED")
        return False

    logger.info("SELF-TEST PASSED")
    return True

def guard_cb_constancy(entry: dict, aa: str) -> bool:
    """GUARD: assert CB is constant across all rotamers (pstdev < 1e-3 A)."""
    names = entry["atom_names"]
    if "CB" not in names:
        logger.warning("%s has no CB atom", aa)
        return False

    cb_idx = names.index("CB")
    cb_coords = []

    for bin_rotamers in entry["rot_bins"]:
        for rot in bin_rotamers:
            cb_coords.append(rot["coords"][cb_idx])

    if not cb_coords:
        logger.error("%s: no CB coordinates", aa)
        return False

    cb_x = [c[0] for c in cb_coords]
    cb_y = [c[1] for c in cb_coords]
    cb_z = [c[2] for c in cb_coords]

    def pstdev(vals):
        if len(vals) <= 1:
            return 0.0
        mean = sum(vals) / len(vals)
        var = sum((x - mean) ** 2 for x in vals) / len(vals)
        return math.sqrt(var)

    std_x = pstdev(cb_x)
    std_y = pstdev(cb_y)
    std_z = pstdev(cb_z)

    threshold = 1e-3
    passed = std_x < threshold and std_y < threshold and std_z < threshold

    if passed:
        logger.info("%s CB constancy PASS: std=[%.6f, %.6f, %.6f] A",
                   aa, std_x, std_y, std_z)
    else:
        logger.error("%s CB constancy FAIL: std=[%.6f, %.6f, %.6f] (threshold=%.6f)",
                    aa, std_x, std_y, std_z, threshold)

    return passed

def measure_cb_improper(entry: dict, aa: str) -> dict | None:
    """Measure the C-N-CA-CB improper dihedral."""
    if not guard_cb_constancy(entry, aa):
        return None

    names = entry["atom_names"]
    if "CB" not in names:
        return None

    cb_idx = names.index("CB")
    rot = entry["rot_bins"][0][0]
    cb_coord = rot["coords"][cb_idx]

    cb_coords = []
    for bin_rotamers in entry["rot_bins"]:
        for r in bin_rotamers:
            cb_coords.append(r["coords"][cb_idx])

    def pstdev(vals):
        if len(vals) <= 1:
            return 0.0
        mean = sum(vals) / len(vals)
        var = sum((x - mean) ** 2 for x in vals) / len(vals)
        return math.sqrt(var)

    cb_x = [c[0] for c in cb_coords]
    cb_y = [c[1] for c in cb_coords]
    cb_z = [c[2] for c in cb_coords]

    std_x = pstdev(cb_x)
    std_y = pstdev(cb_y)
    std_z = pstdev(cb_z)

    # Compute in both conventions
    iupac_improper_deg = dihedral_iupac_deg(IDENTITY_C, IDENTITY_N, IDENTITY_CA, cb_coord)
    nerf_improper_deg = dihedral_nerf_deg(IDENTITY_C, IDENTITY_N, IDENTITY_CA, cb_coord)

    placeholder = -119.7
    iupac_delta = iupac_improper_deg - placeholder
    nerf_delta = nerf_improper_deg - placeholder

    logger.info("%s: CB improper = %.2f deg (IUPAC) / %.2f deg (NeRF), delta = %.2f / %.2f deg",
               aa, iupac_improper_deg, nerf_improper_deg, iupac_delta, nerf_delta)

    return {
        "cb_coord": list(cb_coord),
        "cb_std": [std_x, std_y, std_z],
        "cb_improper_iupac_deg": round(iupac_improper_deg, 2),
        "cb_improper_nerf_deg": round(nerf_improper_deg, 2),
        "delta_vs_placeholder_iupac": round(iupac_delta, 2),
        "delta_vs_placeholder_nerf": round(nerf_delta, 2),
    }

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rotlib", type=Path,
                    default=Path("/home/marielle/repos/mosaist/testfiles/rotlib.bin"),
                    help="path to MASTER/MSL rotlib.bin")
    ap.add_argument("--residues", nargs="+",
                    default=["ARG", "MET", "LYS", "GLN", "LEU", "GLU", "ASP"],
                    help="residue codes to measure")
    ap.add_argument("--out", type=Path, default=None, help="write JSON report")
    ap.add_argument("--log-level", default="INFO")
    ap.add_argument("--dry-run", action="store_true", help="parse only; skip measurement")
    ap.add_argument("--self-test", action="store_true", help="run synthetic verification")
    args = ap.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format="%(levelname)s %(name)s: %(message)s")

    if args.self_test:
        return 0 if _self_test() else 1

    if not args.rotlib.exists():
        logger.error("rotlib.bin not found: %s", args.rotlib)
        return 2

    logger.info("Parsing rotlib.bin: %s", args.rotlib)
    entries = parse_rotlib(args.rotlib)
    logger.info("Parsed %d residues: %s", len(entries), sorted(entries))

    import hashlib
    with args.rotlib.open("rb") as f:
        rotlib_sha256 = hashlib.sha256(f.read()).hexdigest()
    logger.info("rotlib.bin SHA256: %s", rotlib_sha256)

    report = {
        "rotlib": str(args.rotlib),
        "rotlib_sha256": rotlib_sha256,
        "residue_codes": sorted(entries),
        "measurements": {},
        "metadata": {
            "canonical_backbone": {
                "N": IDENTITY_N, "CA": IDENTITY_CA, "C": IDENTITY_C,
                "description": "canonical identity-frame (proxide test_ac_r_routing.rs:185)"
            },
            "placeholder_improper_deg": -119.7,
            "conventions": {
                "iupac": "Standard atan2-based formula (proxide angles.rs:dihedral_angle)",
                "nerf": "NeRF torsion convention (opposite sign to IUPAC; used for atom placement)"
            }
        }
    }

    for aa in args.residues:
        if aa not in entries:
            logger.warning("%s absent", aa)
            continue

        if args.dry_run:
            logger.info("%s atoms: %s", aa, entries[aa]["atom_names"])
            continue

        m = measure_cb_improper(entries[aa], aa)
        if m is not None:
            report["measurements"][aa] = m

    if args.out:
        args.out.write_text(json.dumps(report, indent=2))
        logger.info("wrote -> %s", args.out)
    else:
        print("\n=== CB Improper Measurement ===")
        print(f"{'Residue':<10} {'IUPAC (deg)':<20} {'NeRF (deg)':<20} {'IUPAC Δ':<15} {'NeRF Δ':<15}")
        print("-" * 80)
        for aa in args.residues:
            if aa in report["measurements"]:
                m = report["measurements"][aa]
                print(f"{aa:<10} {m['cb_improper_iupac_deg']:<20.2f} {m['cb_improper_nerf_deg']:<20.2f} {m['delta_vs_placeholder_iupac']:<15.2f} {m['delta_vs_placeholder_nerf']:<15.2f}")
        print()

    return 0

if __name__ == "__main__":
    sys.exit(main())
