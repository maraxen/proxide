#!/usr/bin/env python3
"""Audit LEU sidechain internal coordinates against MASTER rotlib.bin.

MEASUREMENT ONLY — do NOT edit template.rs / charmm_ic.rs / any production code.

For LEU, measure all-atom internal coordinates (bonds, angles, torsions) from
MASTER's stored Cartesian coordinates and compare against the template.rs fixed
geometry. The audit computes:
  - CB bond to CA, CB angle N-CA-CB, CB improper C-N-CA-CB
  - CG bond to CB, CG angle CA-CB-CG (should equal chi1 rotamer value)
  - CD1 bond to CG, CD1 angle CB-CG-CD1 (should equal chi2)
  - CD2 bond to CG, CD2 angle CB-CG-CD2 (should equal chi2 + 120)

Reports per-atom, per-rotamer table and summary max|delta| for geometry validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from extract_rotlib_geometry import parse_rotlib

logger = logging.getLogger("audit_leu_ic_vs_master")

# Canonical backbone frame atoms (proxide test convention)
IDENTITY_N = [-1.0, 0.0, 0.0]
IDENTITY_CA = [0.0, 0.0, 0.0]
IDENTITY_C = [0.0, 1.0, 0.0]

# LEU template fixed geometry (from template.rs)
TEMPLATE_GEOMETRY = {
    "CB": {"parent": "CA", "bond_len_A": 1.540, "angle_deg": 110.5, "improper_nerf_deg": -119.7},
    "CG": {"parent": "CB", "bond_len_A": 1.530, "angle_deg": 116.3},
    "CD1": {"parent": "CG", "bond_len_A": 1.524, "angle_deg": 111.1},
    "CD2": {"parent": "CG", "bond_len_A": 1.524, "angle_deg": 111.1},
}


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
    """Compute dihedral angle A-B-C-D in degrees (NeRF convention).

    NeRF uses sign opposite to IUPAC.
    """
    return -dihedral_iupac_deg(a, b, c, d)


def bond_length(a, b):
    """Compute bond length in Angstroms."""
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def bond_angle(a, b, c):
    """Compute bond angle A-B-C in degrees."""
    v1 = [a[i] - b[i] for i in range(3)]
    v2 = [c[i] - b[i] for i in range(3)]
    dot = sum(v1[i] * v2[i] for i in range(3))
    n1 = math.sqrt(sum(x * x for x in v1))
    n2 = math.sqrt(sum(x * x for x in v2))
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cos_angle = max(-1.0, min(1.0, dot / (n1 * n2)))
    return math.degrees(math.acos(cos_angle))


def self_test():
    """Verify dihedral and angle functions on synthetic atoms."""
    logger.info("Running --self-test...")
    all_pass = True

    # Test 1: perpendicular dihedral (expect ±90 deg)
    a = [0.0, 1.0, 0.0]
    b = [0.0, 0.0, 0.0]
    c = [1.0, 0.0, 0.0]
    d = [1.0, 0.0, 1.0]
    angle_deg = dihedral_iupac_deg(a, b, c, d)
    is_pass = abs(abs(angle_deg) - 90.0) < 0.01
    logger.info("Test dihedral_perpendicular: %.2f deg (expect ±90) %s",
                angle_deg, "PASS" if is_pass else "FAIL")
    all_pass = all_pass and is_pass

    # Test 2: 60 degree dihedral
    a = [0.0, 1.0, 0.0]
    b = [0.0, 0.0, 0.0]
    c = [1.0, 0.0, 0.0]
    d = [1.0, math.cos(math.radians(60)), math.sin(math.radians(60))]
    angle_deg = dihedral_iupac_deg(a, b, c, d)
    is_pass = abs(abs(angle_deg) - 60.0) < 0.01
    logger.info("Test dihedral_60deg: %.2f deg (expect ±60) %s",
                angle_deg, "PASS" if is_pass else "FAIL")
    all_pass = all_pass and is_pass

    # Test 3: bond angle 90 degrees
    a = [1.0, 0.0, 0.0]
    b = [0.0, 0.0, 0.0]
    c = [0.0, 1.0, 0.0]
    angle_deg = bond_angle(a, b, c)
    is_pass = abs(angle_deg - 90.0) < 0.01
    logger.info("Test bond_angle_90deg: %.2f deg (expect 90) %s",
                angle_deg, "PASS" if is_pass else "FAIL")
    all_pass = all_pass and is_pass

    # Test 4: bond length 2.0 Å
    a = [0.0, 0.0, 0.0]
    b = [2.0, 0.0, 0.0]
    dist = bond_length(a, b)
    is_pass = abs(dist - 2.0) < 0.01
    logger.info("Test bond_length_2A: %.3f A (expect 2.0) %s",
                dist, "PASS" if is_pass else "FAIL")
    all_pass = all_pass and is_pass

    # Test 5: NeRF sign negation
    a = [0.0, 1.0, 0.0]
    b = [0.0, 0.0, 0.0]
    c = [1.0, 0.0, 0.0]
    d = [1.0, 0.0, 1.0]
    iupac = dihedral_iupac_deg(a, b, c, d)
    nerf = dihedral_nerf_deg(a, b, c, d)
    is_pass = abs(iupac + nerf) < 0.01
    logger.info("Test nerf_sign_negation: IUPAC=%.2f NeRF=%.2f sum=%.2f %s",
                iupac, nerf, iupac + nerf, "PASS" if is_pass else "FAIL")
    all_pass = all_pass and is_pass

    if all_pass:
        logger.info("SELF-TEST PASSED")
    else:
        logger.error("SELF-TEST FAILED")

    return all_pass


def audit_leu_rotamers(entry: dict, rotlib_path: Path, n_rotamers: int = 5) -> dict:
    """Audit LEU sidechain ICs for the top rotamers.

    For each rotamer in the top-frequency bin, compute from MASTER's coordinates
    and the canonical backbone all internal coordinates (bonds, angles, torsions)
    and compare to template.rs expectations.

    Returns dict with per-atom, per-rotamer measurements and summary statistics.
    """
    names = entry["atom_names"]
    name_to_idx = {n: i for i, n in enumerate(names)}

    if "CB" not in names or "CG" not in names or "CD1" not in names or "CD2" not in names:
        logger.error("LEU missing required sidechain atoms; have %s", names)
        return {}

    # Get the highest-frequency phi/psi bin
    best_bin_idx = max(range(entry["nb"]), key=lambda i: entry["bins"][i][2])
    rotamers = entry["rot_bins"][best_bin_idx]
    phi, psi, freq = entry["bins"][best_bin_idx]

    logger.info("LEU audit: best_bin_idx=%d, phi/psi=(%.1f/%.1f), freq=%.4f, n_rotamers=%d",
                best_bin_idx, phi, psi, freq, len(rotamers))

    # Select top N rotamers by probability
    top_rotamers = sorted(rotamers, key=lambda r: r["prob"], reverse=True)[:n_rotamers]
    logger.info("Selected top %d rotamers for audit", len(top_rotamers))

    measurements = []

    for rot_idx, rotamer in enumerate(top_rotamers):
        chi_vals = [c[0] for c in rotamer["chi"]]  # chi1, chi2
        coords = rotamer["coords"]

        cb_coord = coords[name_to_idx["CB"]]
        cg_coord = coords[name_to_idx["CG"]]
        cd1_coord = coords[name_to_idx["CD1"]]
        cd2_coord = coords[name_to_idx["CD2"]]

        # Compute internal coordinates
        cb_bond = bond_length(IDENTITY_CA, cb_coord)
        # Template angle for CB is N-CA-CB (NeRF places CB via [C,N,CA] -> angle b-c-d = N-CA-CB).
        # Using IDENTITY_C here was a bug: it measured C-CA-CB (118.9deg) and compared it to the
        # N-CA-CB template value (110.5), producing a spurious ~8.4deg "mismatch".
        cb_angle = bond_angle(IDENTITY_N, IDENTITY_CA, cb_coord)
        cb_improper_nerf = dihedral_nerf_deg(IDENTITY_C, IDENTITY_N, IDENTITY_CA, cb_coord)

        cg_bond = bond_length(cb_coord, cg_coord)
        cg_angle = bond_angle(IDENTITY_CA, cb_coord, cg_coord)
        cg_torsion_nerf = dihedral_nerf_deg(IDENTITY_N, IDENTITY_CA, cb_coord, cg_coord)

        cd1_bond = bond_length(cg_coord, cd1_coord)
        cd1_angle = bond_angle(cb_coord, cg_coord, cd1_coord)
        cd1_torsion_nerf = dihedral_nerf_deg(IDENTITY_CA, cb_coord, cg_coord, cd1_coord)

        cd2_bond = bond_length(cg_coord, cd2_coord)
        cd2_angle = bond_angle(cb_coord, cg_coord, cd2_coord)
        cd2_torsion_nerf = dihedral_nerf_deg(IDENTITY_CA, cb_coord, cg_coord, cd2_coord)

        # Compute deltas from template / chi expectations
        cb_bond_delta = cb_bond - TEMPLATE_GEOMETRY["CB"]["bond_len_A"]
        cb_angle_delta = cb_angle - TEMPLATE_GEOMETRY["CB"]["angle_deg"]
        cb_improper_delta = cb_improper_nerf - TEMPLATE_GEOMETRY["CB"]["improper_nerf_deg"]

        cg_bond_delta = cg_bond - TEMPLATE_GEOMETRY["CG"]["bond_len_A"]
        cg_angle_delta = cg_angle - TEMPLATE_GEOMETRY["CG"]["angle_deg"]
        cg_torsion_delta = cg_torsion_nerf - chi_vals[0]  # chi1

        cd1_bond_delta = cd1_bond - TEMPLATE_GEOMETRY["CD1"]["bond_len_A"]
        cd1_angle_delta = cd1_angle - TEMPLATE_GEOMETRY["CD1"]["angle_deg"]
        cd1_torsion_delta = cd1_torsion_nerf - chi_vals[1]  # chi2

        cd2_bond_delta = cd2_bond - TEMPLATE_GEOMETRY["CD2"]["bond_len_A"]
        cd2_angle_delta = cd2_angle - TEMPLATE_GEOMETRY["CD2"]["angle_deg"]
        # CD2 torsion is chi2 + 120 offset
        cd2_torsion_expected = chi_vals[1] + 120.0
        cd2_torsion_delta = cd2_torsion_nerf - cd2_torsion_expected

        measurement = {
            "rotamer_idx": rot_idx,
            "prob": rotamer["prob"],
            "chi_deg": chi_vals,
            "CB": {
                "bond_len_A": round(cb_bond, 4),
                "bond_delta_A": round(cb_bond_delta, 4),
                "angle_deg": round(cb_angle, 2),
                "angle_delta_deg": round(cb_angle_delta, 2),
                "improper_nerf_deg": round(cb_improper_nerf, 2),
                "improper_delta_deg": round(cb_improper_delta, 2),
            },
            "CG": {
                "bond_len_A": round(cg_bond, 4),
                "bond_delta_A": round(cg_bond_delta, 4),
                "angle_deg": round(cg_angle, 2),
                "angle_delta_deg": round(cg_angle_delta, 2),
                "torsion_nerf_deg": round(cg_torsion_nerf, 2),
                "torsion_vs_chi1_deg": round(cg_torsion_delta, 2),
            },
            "CD1": {
                "bond_len_A": round(cd1_bond, 4),
                "bond_delta_A": round(cd1_bond_delta, 4),
                "angle_deg": round(cd1_angle, 2),
                "angle_delta_deg": round(cd1_angle_delta, 2),
                "torsion_nerf_deg": round(cd1_torsion_nerf, 2),
                "torsion_vs_chi2_deg": round(cd1_torsion_delta, 2),
            },
            "CD2": {
                "bond_len_A": round(cd2_bond, 4),
                "bond_delta_A": round(cd2_bond_delta, 4),
                "angle_deg": round(cd2_angle, 2),
                "angle_delta_deg": round(cd2_angle_delta, 2),
                "torsion_nerf_deg": round(cd2_torsion_nerf, 2),
                "torsion_vs_chi2_plus_120_deg": round(cd2_torsion_delta, 2),
            },
        }
        measurements.append(measurement)

    # Compute summary statistics
    def max_abs_delta(key_path):
        """Extract max absolute delta across all measurements."""
        vals = []
        for m in measurements:
            d = m
            for k in key_path:
                d = d[k]
            vals.append(abs(d))
        return max(vals) if vals else 0.0

    summary = {
        "n_rotamers_audited": len(measurements),
        "max_abs_delta": {
            "bonds_A": {
                "CB": round(max_abs_delta(["CB", "bond_delta_A"]), 4),
                "CG": round(max_abs_delta(["CG", "bond_delta_A"]), 4),
                "CD1": round(max_abs_delta(["CD1", "bond_delta_A"]), 4),
                "CD2": round(max_abs_delta(["CD2", "bond_delta_A"]), 4),
            },
            "angles_deg": {
                "CB": round(max_abs_delta(["CB", "angle_delta_deg"]), 2),
                "CG": round(max_abs_delta(["CG", "angle_delta_deg"]), 2),
                "CD1": round(max_abs_delta(["CD1", "angle_delta_deg"]), 2),
                "CD2": round(max_abs_delta(["CD2", "angle_delta_deg"]), 2),
            },
            "torsions_deg": {
                "CB_improper": round(max_abs_delta(["CB", "improper_delta_deg"]), 2),
                "CG_vs_chi1": round(max_abs_delta(["CG", "torsion_vs_chi1_deg"]), 2),
                "CD1_vs_chi2": round(max_abs_delta(["CD1", "torsion_vs_chi2_deg"]), 2),
                "CD2_vs_chi2_plus_120": round(max_abs_delta(["CD2", "torsion_vs_chi2_plus_120_deg"]), 2),
            },
        },
    }

    return {
        "rotlib": str(rotlib_path),
        "residue": "LEU",
        "phi_deg": phi,
        "psi_deg": psi,
        "bin_frequency": freq,
        "measurements": measurements,
        "summary": summary,
        "canonical_backbone": {
            "N": IDENTITY_N,
            "CA": IDENTITY_CA,
            "C": IDENTITY_C,
        },
        "template_geometry": TEMPLATE_GEOMETRY,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rotlib", type=Path,
                    default=Path("/home/marielle/repos/mosaist/testfiles/rotlib.bin"),
                    help="path to MASTER/MSL rotlib.bin")
    ap.add_argument("--out", type=Path, default=None, help="write JSON audit report")
    ap.add_argument("--n-rotamers", type=int, default=5,
                    help="number of top rotamers to audit (default 5)")
    ap.add_argument("--log-level", default="INFO")
    ap.add_argument("--dry-run", action="store_true", help="parse header only")
    ap.add_argument("--self-test", action="store_true", help="run synthetic IC verification")
    args = ap.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format="%(levelname)s %(name)s: %(message)s")

    if args.self_test:
        return 0 if self_test() else 1

    if not args.rotlib.exists():
        logger.error("rotlib.bin not found: %s", args.rotlib)
        return 2

    logger.info("Parsing rotlib.bin: %s", args.rotlib)
    entries = parse_rotlib(args.rotlib)
    logger.info("Parsed %d residues: %s", len(entries), sorted(entries))

    if "LEU" not in entries:
        logger.error("LEU not found in rotlib.bin")
        return 2

    if args.dry_run:
        logger.info("LEU atoms: %s", entries["LEU"]["atom_names"])
        return 0

    # Compute rotlib SHA256 for provenance
    with args.rotlib.open("rb") as f:
        rotlib_sha256 = hashlib.sha256(f.read()).hexdigest()
    logger.info("rotlib.bin SHA256: %s", rotlib_sha256)

    report = audit_leu_rotamers(entries["LEU"], args.rotlib, n_rotamers=args.n_rotamers)
    report["rotlib_sha256"] = rotlib_sha256

    # Pretty-print summary to log
    logger.info("=== AUDIT SUMMARY ===")
    logger.info("Audited %d LEU rotamers (phi=%.1f, psi=%.1f)",
                report["summary"]["n_rotamers_audited"], report["phi_deg"], report["psi_deg"])
    logger.info("Max |delta| (bonds):")
    for atom in ("CB", "CG", "CD1", "CD2"):
        delta = report["summary"]["max_abs_delta"]["bonds_A"][atom]
        template_len = TEMPLATE_GEOMETRY[atom]["bond_len_A"]
        logger.info("  %s: %.4f A (template=%.4f)", atom, delta, template_len)
    logger.info("Max |delta| (angles):")
    for atom in ("CB", "CG", "CD1", "CD2"):
        delta = report["summary"]["max_abs_delta"]["angles_deg"][atom]
        template_angle = TEMPLATE_GEOMETRY[atom]["angle_deg"]
        logger.info("  %s: %.2f deg (template=%.2f)", atom, delta, template_angle)
    logger.info("Max |delta| (torsions):")
    for key, val in report["summary"]["max_abs_delta"]["torsions_deg"].items():
        logger.info("  %s: %.2f deg", key, val)

    # Per-rotamer table
    logger.info("=== PER-ROTAMER MEASUREMENTS ===")
    for m in report["measurements"]:
        logger.info("Rotamer %d (prob=%.4f, chi=[%.1f, %.1f]):",
                    m["rotamer_idx"], m["prob"], m["chi_deg"][0], m["chi_deg"][1])
        for atom in ("CB", "CG", "CD1", "CD2"):
            atom_data = m[atom]
            logger.info("  %s: bond=%.4f(Δ%.4f) A, angle=%.2f(Δ%.2f) deg",
                        atom, atom_data["bond_len_A"], atom_data["bond_delta_A"],
                        atom_data["angle_deg"], atom_data["angle_delta_deg"])
            if atom == "CB":
                logger.info("       improper(NeRF)=%.2f(Δ%.2f) deg",
                            atom_data["improper_nerf_deg"], atom_data["improper_delta_deg"])
            elif atom == "CG":
                logger.info("       torsion(NeRF)=%.2f vs chi1=%.2f (Δ%.2f) deg",
                            atom_data["torsion_nerf_deg"], m["chi_deg"][0],
                            atom_data["torsion_vs_chi1_deg"])
            elif atom == "CD1":
                logger.info("       torsion(NeRF)=%.2f vs chi2=%.2f (Δ%.2f) deg",
                            atom_data["torsion_nerf_deg"], m["chi_deg"][1],
                            atom_data["torsion_vs_chi2_deg"])
            elif atom == "CD2":
                expected = m["chi_deg"][1] + 120.0
                logger.info("       torsion(NeRF)=%.2f vs chi2+120=%.2f (Δ%.2f) deg",
                            atom_data["torsion_nerf_deg"], expected,
                            atom_data["torsion_vs_chi2_plus_120_deg"])

    if args.out:
        args.out.write_text(json.dumps(report, indent=2))
        logger.info("wrote audit report -> %s", args.out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
