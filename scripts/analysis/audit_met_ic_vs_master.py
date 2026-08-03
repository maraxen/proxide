#!/usr/bin/env python3
"""Audit MET sidechain internal coordinates against MASTER rotlib.bin.

MEASUREMENT ONLY — do NOT edit template.rs / charmm_ic.rs / any production code.

For MET (CB-CG-SD-CE sidechain), measure all-atom internal coordinates
(bonds, angles, torsions) from MASTER's stored Cartesian coordinates and compare
against the template.rs fixed geometry:

  CB: bond CA-CB, angle N-CA-CB, improper C-N-CA-CB
  CG: bond CB-CG, angle CA-CB-CG
  SD: bond CG-SD, angle CB-CG-SD
  CE: bond SD-CE, angle CG-SD-CE

Reports per-atom, per-rotamer table and summary max|delta|.
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

logger = logging.getLogger("audit_met_ic_vs_master")

# Canonical backbone frame atoms (proxide CA-origin convention)
IDENTITY_N = [-1.0, 0.0, 0.0]
IDENTITY_CA = [0.0, 0.0, 0.0]
IDENTITY_C = [0.0, 1.0, 0.0]

# MET template fixed geometry (from template.rs methionine_template())
TEMPLATE_GEOMETRY = {
    "CB": {"bond_len_A": 1.540, "angle_deg": 110.5, "improper_nerf_deg": -119.7},
    "CG": {"bond_len_A": 1.530, "angle_deg": 114.1},
    "SD": {"bond_len_A": 1.807, "angle_deg": 112.7},
    "CE": {"bond_len_A": 1.791, "angle_deg": 100.9},
}


def bond_length(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def bond_angle(a, b, c):
    v1 = [a[i] - b[i] for i in range(3)]
    v2 = [c[i] - b[i] for i in range(3)]
    dot = sum(v1[i] * v2[i] for i in range(3))
    n1 = math.sqrt(sum(x * x for x in v1))
    n2 = math.sqrt(sum(x * x for x in v2))
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / (n1 * n2)))))


def dihedral_iupac(a, b, c, d):
    b1 = [b[i] - a[i] for i in range(3)]
    b2 = [c[i] - b[i] for i in range(3)]
    b3 = [d[i] - c[i] for i in range(3)]

    def cross(u, v):
        return [u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]]

    def dot(u, v):
        return sum(u[i]*v[i] for i in range(3))

    def normalize(v):
        m = math.sqrt(sum(x*x for x in v))
        return [x/m for x in v] if m > 1e-6 else [0.0]*3

    n1 = cross(b1, b2)
    n2 = cross(b2, b3)
    m1 = cross(n1, normalize(b2))
    return math.degrees(math.atan2(dot(m1, n2), dot(n1, n2)))


def dihedral_nerf_deg(a, b, c, d):
    return -dihedral_iupac(a, b, c, d)


def audit_met_rotamers(entry: dict, rotlib_path: Path, n_rotamers: int = 5) -> dict:
    names = entry["atom_names"]
    name_to_idx = {n: i for i, n in enumerate(names)}

    for req in ("CB", "CG", "SD", "CE"):
        if req not in names:
            logger.error("MET missing required atom %s; have %s", req, names)
            return {}

    best_bin_idx = max(range(entry["nb"]), key=lambda i: entry["bins"][i][2])
    rotamers = entry["rot_bins"][best_bin_idx]
    phi, psi, freq = entry["bins"][best_bin_idx]

    logger.info("MET audit: best_bin_idx=%d, phi/psi=(%.1f/%.1f), freq=%.4f, n_rotamers=%d",
                best_bin_idx, phi, psi, freq, len(rotamers))

    top_rotamers = sorted(rotamers, key=lambda r: r["prob"], reverse=True)[:n_rotamers]
    measurements = []

    for rot_idx, rotamer in enumerate(top_rotamers):
        chi_vals = [c[0] for c in rotamer["chi"]]
        coords = rotamer["coords"]

        cb = coords[name_to_idx["CB"]]
        cg = coords[name_to_idx["CG"]]
        sd = coords[name_to_idx["SD"]]
        ce = coords[name_to_idx["CE"]]

        cb_bond  = bond_length(IDENTITY_CA, cb)
        cb_angle = bond_angle(IDENTITY_N, IDENTITY_CA, cb)
        cb_imp   = dihedral_nerf_deg(IDENTITY_C, IDENTITY_N, IDENTITY_CA, cb)

        cg_bond  = bond_length(cb, cg)
        cg_angle = bond_angle(IDENTITY_CA, cb, cg)
        cg_tor   = dihedral_nerf_deg(IDENTITY_N, IDENTITY_CA, cb, cg)

        sd_bond  = bond_length(cg, sd)
        sd_angle = bond_angle(cb, cg, sd)
        sd_tor   = dihedral_nerf_deg(IDENTITY_CA, cb, cg, sd)

        ce_bond  = bond_length(sd, ce)
        ce_angle = bond_angle(cg, sd, ce)
        ce_tor   = dihedral_nerf_deg(cb, cg, sd, ce)

        measurements.append({
            "rotamer_idx": rot_idx,
            "prob": rotamer["prob"],
            "chi_deg": chi_vals,
            "CB": {
                "bond_len_A":       round(cb_bond, 4),
                "bond_delta_A":     round(cb_bond - TEMPLATE_GEOMETRY["CB"]["bond_len_A"], 4),
                "angle_deg":        round(cb_angle, 2),
                "angle_delta_deg":  round(cb_angle - TEMPLATE_GEOMETRY["CB"]["angle_deg"], 2),
                "improper_nerf_deg":   round(cb_imp, 2),
                "improper_delta_deg":  round(
                    cb_imp - TEMPLATE_GEOMETRY["CB"]["improper_nerf_deg"], 2
                ),
            },
            "CG": {
                "bond_len_A":       round(cg_bond, 4),
                "bond_delta_A":     round(cg_bond - TEMPLATE_GEOMETRY["CG"]["bond_len_A"], 4),
                "angle_deg":        round(cg_angle, 2),
                "angle_delta_deg":  round(cg_angle - TEMPLATE_GEOMETRY["CG"]["angle_deg"], 2),
                "torsion_nerf_deg": round(cg_tor, 2),
                "torsion_vs_chi1_deg": round(cg_tor - chi_vals[0], 2) if chi_vals else None,
            },
            "SD": {
                "bond_len_A":       round(sd_bond, 4),
                "bond_delta_A":     round(sd_bond - TEMPLATE_GEOMETRY["SD"]["bond_len_A"], 4),
                "angle_deg":        round(sd_angle, 2),
                "angle_delta_deg":  round(sd_angle - TEMPLATE_GEOMETRY["SD"]["angle_deg"], 2),
                "torsion_nerf_deg": round(sd_tor, 2),
                "torsion_vs_chi2_deg": (
                    round(sd_tor - chi_vals[1], 2) if len(chi_vals) > 1 else None
                ),
            },
            "CE": {
                "bond_len_A":       round(ce_bond, 4),
                "bond_delta_A":     round(ce_bond - TEMPLATE_GEOMETRY["CE"]["bond_len_A"], 4),
                "angle_deg":        round(ce_angle, 2),
                "angle_delta_deg":  round(ce_angle - TEMPLATE_GEOMETRY["CE"]["angle_deg"], 2),
                "torsion_nerf_deg": round(ce_tor, 2),
                "torsion_vs_chi3_deg": (
                    round(ce_tor - chi_vals[2], 2) if len(chi_vals) > 2 else None
                ),
            },
        })

    def max_abs_delta(key_path):
        vals = []
        for m in measurements:
            d = m
            for k in key_path:
                d = d.get(k) if isinstance(d, dict) else None
                if d is None:
                    break
            if d is not None:
                vals.append(abs(d))
        return max(vals) if vals else 0.0

    summary = {
        "n_rotamers_audited": len(measurements),
        "max_abs_delta": {
            "bonds_A": {a: round(max_abs_delta([a, "bond_delta_A"]), 4)
                        for a in ("CB", "CG", "SD", "CE")},
            "angles_deg": {a: round(max_abs_delta([a, "angle_delta_deg"]), 2)
                           for a in ("CB", "CG", "SD", "CE")},
        },
    }

    return {
        "rotlib": str(rotlib_path),
        "residue": "MET",
        "phi_deg": phi,
        "psi_deg": psi,
        "bin_frequency": freq,
        "measurements": measurements,
        "summary": summary,
        "template_geometry": TEMPLATE_GEOMETRY,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rotlib", type=Path,
                    default=Path("/home/marielle/repos/mosaist/testfiles/rotlib.bin"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--n-rotamers", type=int, default=5)
    ap.add_argument("--log-level", default="INFO")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format="%(levelname)s %(name)s: %(message)s")

    if not args.rotlib.exists():
        logger.error("rotlib.bin not found: %s", args.rotlib)
        return 2

    logger.info("Parsing rotlib.bin: %s", args.rotlib)
    entries = parse_rotlib(args.rotlib)

    if "MET" not in entries:
        logger.error("MET not found in rotlib.bin")
        return 2

    if args.dry_run:
        logger.info("MET atoms: %s", entries["MET"]["atom_names"])
        return 0

    sha = hashlib.sha256(args.rotlib.open("rb").read()).hexdigest()
    logger.info("rotlib.bin SHA256: %s", sha)

    report = audit_met_rotamers(entries["MET"], args.rotlib, n_rotamers=args.n_rotamers)
    report["rotlib_sha256"] = sha

    logger.info("=== AUDIT SUMMARY ===")
    logger.info("Audited %d MET rotamers (phi=%.1f, psi=%.1f)",
                report["summary"]["n_rotamers_audited"], report["phi_deg"], report["psi_deg"])
    logger.info("Max |delta| bonds (A):   CB=%.4f  CG=%.4f  SD=%.4f  CE=%.4f",
                *[report["summary"]["max_abs_delta"]["bonds_A"][a] for a in ("CB","CG","SD","CE")])
    logger.info(
        "Max |delta| angles (deg): CB=%.2f  CG=%.2f  SD=%.2f  CE=%.2f",
        *[report["summary"]["max_abs_delta"]["angles_deg"][a] for a in ("CB", "CG", "SD", "CE")],
    )

    logger.info("=== PER-ROTAMER MEASUREMENTS ===")
    for m in report["measurements"]:
        logger.info("Rotamer %d (prob=%.4f, chi=%s):",
                    m["rotamer_idx"], m["prob"],
                    [round(c, 1) for c in m["chi_deg"]])
        for atom in ("CB", "CG", "SD", "CE"):
            d = m[atom]
            logger.info("  %s: bond=%.4f(Δ%.4f)A  angle=%.2f(Δ%.2f)deg",
                        atom, d["bond_len_A"], d["bond_delta_A"],
                        d["angle_deg"], d["angle_delta_deg"])

    if args.out:
        args.out.write_text(json.dumps(report, indent=2))
        logger.info("wrote -> %s", args.out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
