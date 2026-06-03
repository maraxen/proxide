#!/usr/bin/env python3
"""Extract and measure baked sidechain geometry from MASTER/MSL ``rotlib.bin``.

Backlog #820: root-cause the contact-degree drift between proxide's rebuilt
``*.rotlib.pb.zst`` (Engh-Huber / CCD ``PRO.cif`` placeholder geometry) and the
MASTER golden ``rotlib.bin``. The hypothesis (from code comment in
``crates/proxide-rotlib/src/geometry/template.rs`` and a NotebookLM query against
the Dunbrack/CHARMM proline parameterization literature) is that MASTER built its
Cartesian coordinates from **CHARMM ideal internal coordinates**, not Engh-Huber.

This script parses the MSL binary rotamer-library format (decoded from
``mosaist/src/mstrotlib.cpp::readRotamerLibrary``) and measures the actual
bond lengths and bond angles baked into ``rotlib.bin`` for a chosen residue,
then compares them against the CHARMM and Engh-Huber/CCD reference tables so the
geometry component of the drift can be quantified directly from ground truth.

The MSL binary stores only **sidechain** atoms (backbone N/CA/C/O are fetched
separately by ``getBackbone`` at placement time), so the frame-independent
measurements available here are intra-sidechain bonds/angles (e.g. for PRO:
CB-CG, CG-CD, CB-CG-CD). Those alone discriminate CHARMM (~1.537 Å, 108.5°) from
the CCD/Engh-Huber placeholder (~1.543 Å, ~105°).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import struct
import sys
from pathlib import Path

logger = logging.getLogger("extract_rotlib_geometry")

# --- Reference geometry tables (Å, degrees) --------------------------------
# CHARMM22 all-hydrogen proline ideals (atom types CP1=CA, CP2=CB/CG, CP3=CD).
# Source: Dunbrack-lab / CHARMM proline parameterization appendix (via NotebookLM
# notebook 171c5c8b "proxide: Rotamer Library Theory", source 4dc6a5ab).
CHARMM_PRO = {
    "bonds": {"N-CA": 1.434, "CA-CB": 1.527, "CB-CG": 1.537, "CG-CD": 1.537, "CD-N": 1.455},
    "angles": {"N-CA-CB": 110.80, "CA-CB-CG": 108.50, "CB-CG-CD": 108.50,
               "CG-CD-N": 110.50, "CD-N-CA": 114.20},
}
# Current proxide placeholder: CCD PRO.cif ideals (public-domain), used in
# crates/proxide-rotlib/src/geometry/template.rs.
CCD_PRO = {
    "bonds": {"N-CA": 1.486, "CA-CB": 1.543, "CB-CG": 1.543, "CG-CD": 1.544, "CD-N": 1.487},
    "angles": {"N-CA-CB": 104.7, "CA-CB-CG": 105.1, "CB-CG-CD": 105.1,
               "CG-CD-N": 104.7, "CD-N-CA": 104.1},
}


def _read_cstr(fp) -> str:
    out = bytearray()
    while True:
        b = fp.read(1)
        if not b or b == b"\x00":
            break
        out += b
    return out.decode("ascii", "replace")


def _read_i32(fp) -> int:
    return struct.unpack("<i", fp.read(4))[0]


def _read_f32(fp) -> float:
    return struct.unpack("<f", fp.read(4))[0]


def parse_rotlib(path: Path):
    """Parse the MSL binary rotamer library into a dict keyed by residue code.

    Mirrors ``RotamerLibrary::readRotamerLibrary`` (mosaist/src/mstrotlib.cpp).
    """
    entries: dict[str, dict] = {}
    with path.open("rb") as fp:
        while True:
            aa = _read_cstr(fp)
            # EOF: readNullTerminatedString returns empty at end of file.
            peek = fp.read(1)
            if not peek:
                break
            fp.seek(-1, 1)
            if aa == "":
                break
            nc = _read_i32(fp)  # num chi
            na = _read_i32(fp)  # num sidechain atoms
            nb = _read_i32(fp)  # num phi/psi bins
            chidef = [[_read_cstr(fp) for _ in range(4)] for _ in range(nc)]
            atom_names = [_read_cstr(fp) for _ in range(na)]
            bins = []
            for _ in range(nb):
                phi = _read_f32(fp)
                psi = _read_f32(fp)
                freq = _read_f32(fp)
                bins.append((phi, psi, freq))
            rot_bins = []
            for _ in range(nb):
                nr = _read_i32(fp)
                rotamers = []
                for _ in range(nr):
                    prob = _read_f32(fp)
                    chi = [(_read_f32(fp), _read_f32(fp)) for _ in range(nc)]
                    coords = [[_read_f32(fp), _read_f32(fp), _read_f32(fp)] for _ in range(na)]
                    rotamers.append({"prob": prob, "chi": chi, "coords": coords})
                rot_bins.append(rotamers)
            entries[aa] = {
                "nc": nc, "na": na, "nb": nb,
                "chidef": chidef, "atom_names": atom_names,
                "bins": bins, "rot_bins": rot_bins,
            }
            logger.debug("parsed %s: nc=%d na=%d nb=%d atoms=%s", aa, nc, na, nb, atom_names)
    return entries


def charmm_pro_params(xml_path: Path) -> dict:
    """Parse proline atom types and the equilibrium bond lengths/angles that
    bracket them from an OpenMM CHARMM36 FFXML, to verify the bundled
    forcefield reproduces the geometry baked into rotlib.bin.

    Returns {'atom_types': {atomname: type}, 'bonds_A': {...}, 'angles_deg': {...}}
    with lengths converted nm->Å and angles rad->deg.
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(xml_path)
    root = tree.getroot()
    # Residue PRO atom name -> class/type
    a2type: dict[str, str] = {}
    for res in root.iter("Residue"):
        if res.get("name") == "PRO":
            for atom in res.findall("Atom"):
                a2type[atom.get("name")] = atom.get("type")
            break
    # OpenMM CHARMM FFXML AtomTypes map type-id -> class (bonds use class*)
    type2class: dict[str, str] = {}
    for at in root.iter("Type"):
        type2class[at.get("name")] = at.get("class")

    def cls(name):
        t = a2type.get(name)
        return type2class.get(t, t)

    # index bond/angle params by frozenset/tuple of classes
    bonds = {}
    for b in root.iter("Bond"):
        c1, c2, length = b.get("class1") or b.get("type1"), b.get("class2") or b.get("type2"), b.get("length")
        if c1 and c2 and length is not None:
            bonds[frozenset((c1, c2))] = float(length) * 10.0  # nm -> Å
    angles = {}
    for a in root.iter("Angle"):
        c1 = a.get("class1") or a.get("type1")
        c2 = a.get("class2") or a.get("type2")
        c3 = a.get("class3") or a.get("type3")
        ang = a.get("angle")
        if c1 and c2 and c3 and ang is not None:
            angles[(c1, c2, c3)] = math.degrees(float(ang))

    def bond(x, y):
        return bonds.get(frozenset((cls(x), cls(y))))

    def angle(x, y, z):
        key, rev = (cls(x), cls(y), cls(z)), (cls(z), cls(y), cls(x))
        return angles.get(key) or angles.get(rev)

    out_b, out_a = {}, {}
    for x, y in (("N", "CA"), ("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "N")):
        v = bond(x, y)
        if v:
            out_b[f"{x}-{y}"] = round(v, 3)
    for x, y, z in (("N", "CA", "CB"), ("CA", "CB", "CG"), ("CB", "CG", "CD"),
                    ("CG", "CD", "N"), ("CD", "N", "CA")):
        v = angle(x, y, z)
        if v:
            out_a[f"{x}-{y}-{z}"] = round(v, 2)
    return {"atom_types": a2type, "bonds_A": out_b, "angles_deg": out_a}


def _dist(a, b):
    return math.dist(a, b)


def _angle(a, b, c):
    """Bond angle a-b-c in degrees."""
    v1 = [a[i] - b[i] for i in range(3)]
    v2 = [c[i] - b[i] for i in range(3)]
    dot = sum(v1[i] * v2[i] for i in range(3))
    n1 = math.sqrt(sum(x * x for x in v1))
    n2 = math.sqrt(sum(x * x for x in v2))
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / (n1 * n2)))))


def measure_residue(entry: dict, aa: str) -> dict:
    """Measure intra-sidechain bonds/angles for the highest-frequency bin,
    rotamer 0. Returns a dict of named measurements over stored atoms."""
    names = entry["atom_names"]
    idx = {n: i for i, n in enumerate(names)}
    # pick the most frequent bin
    best_bin = max(range(entry["nb"]), key=lambda i: entry["bins"][i][2])
    rot = entry["rot_bins"][best_bin][0]
    coords = rot["coords"]

    def C(name):
        return coords[idx[name]] if name in idx else None

    bonds, angles = {}, {}
    # Generic: consecutive sidechain bonds we can resolve for common residues.
    chain_defs = {
        "PRO": [("CB", "CG"), ("CG", "CD")],
        "LEU": [("CB", "CG"), ("CG", "CD1"), ("CG", "CD2")],
        "ARG": [("CB", "CG"), ("CG", "CD"), ("CD", "NE"), ("NE", "CZ")],
        "LYS": [("CB", "CG"), ("CG", "CD"), ("CD", "CE"), ("CE", "NZ")],
    }
    angle_defs = {
        "PRO": [("CB", "CG", "CD")],
        "LEU": [("CB", "CG", "CD1"), ("CB", "CG", "CD2"), ("CD1", "CG", "CD2")],
        "ARG": [("CB", "CG", "CD"), ("CG", "CD", "NE"), ("CD", "NE", "CZ")],
    }
    for x, y in chain_defs.get(aa, []):
        if C(x) and C(y):
            bonds[f"{x}-{y}"] = _dist(C(x), C(y))
    for x, y, z in angle_defs.get(aa, []):
        if C(x) and C(y) and C(z):
            angles[f"{x}-{y}-{z}"] = _angle(C(x), C(y), C(z))

    return {
        "aa": aa,
        "atom_names": names,
        "best_bin": best_bin,
        "best_bin_phipsi": entry["bins"][best_bin][:2],
        "rot0_prob": rot["prob"],
        "rot0_chi_deg": [c[0] for c in rot["chi"]],
        "bonds_A": bonds,
        "angles_deg": angles,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rotlib", type=Path,
                    default=Path("/home/marielle/repos/mosaist/testfiles/rotlib.bin"),
                    help="path to MASTER/MSL rotlib.bin")
    ap.add_argument("--residues", nargs="+", default=["PRO", "LEU", "ARG"],
                    help="residue codes to measure")
    ap.add_argument("--charmm-xml", type=Path,
                    default=Path("src/proxide/assets/charmm/charmm36_protein.xml"),
                    help="OpenMM CHARMM36 FFXML to cross-check PRO ideals against (proxide-bundled)")
    ap.add_argument("--out", type=Path, default=None, help="write JSON report here")
    ap.add_argument("--log-level", default="INFO")
    ap.add_argument("--dry-run", action="store_true",
                    help="parse header/atom names only; skip measurement")
    args = ap.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format="%(levelname)s %(name)s: %(message)s")

    if not args.rotlib.exists():
        logger.error("rotlib.bin not found at %s", args.rotlib)
        return 2

    entries = parse_rotlib(args.rotlib)
    logger.info("parsed %d residue codes: %s", len(entries), sorted(entries))

    report = {"rotlib": str(args.rotlib), "residue_codes": sorted(entries), "measurements": {}}
    for aa in args.residues:
        if aa not in entries:
            logger.warning("%s absent from rotlib.bin", aa)
            continue
        if args.dry_run:
            logger.info("%s atoms: %s", aa, entries[aa]["atom_names"])
            continue
        m = measure_residue(entries[aa], aa)
        report["measurements"][aa] = m
        logger.info("=== %s (bin phi/psi=%s, rot0 prob=%.3f) ===",
                    aa, m["best_bin_phipsi"], m["rot0_prob"])
        logger.info("  atoms: %s", m["atom_names"])
        logger.info("  chi(deg): %s", [round(c, 1) for c in m["rot0_chi_deg"]])
        for k, v in m["bonds_A"].items():
            logger.info("  bond  %-10s = %.3f A", k, v)
        for k, v in m["angles_deg"].items():
            logger.info("  angle %-12s = %.2f deg", k, v)

    # PRO comparison table vs references
    if "PRO" in report["measurements"]:
        pro = report["measurements"]["PRO"]
        cmp = {"bonds": {}, "angles": {}}
        for k, v in pro["bonds_A"].items():
            cmp["bonds"][k] = {"measured": round(v, 3),
                               "charmm": CHARMM_PRO["bonds"].get(k),
                               "ccd_placeholder": CCD_PRO["bonds"].get(k)}
        for k, v in pro["angles_deg"].items():
            cmp["angles"][k] = {"measured": round(v, 2),
                                "charmm": CHARMM_PRO["angles"].get(k),
                                "ccd_placeholder": CCD_PRO["angles"].get(k)}
        report["pro_comparison"] = cmp
        logger.info("--- PRO measured vs CHARMM vs CCD-placeholder ---")
        for grp in ("bonds", "angles"):
            for k, d in cmp[grp].items():
                logger.info("  %-12s measured=%-8s charmm=%-8s ccd=%-8s",
                            k, d["measured"], d["charmm"], d["ccd_placeholder"])

    # Cross-check against proxide's bundled CHARMM36 FFXML (the parser the fix will use).
    if args.charmm_xml and args.charmm_xml.exists():
        try:
            cp = charmm_pro_params(args.charmm_xml)
            report["charmm36_ffxml"] = {"path": str(args.charmm_xml), **cp}
            logger.info("--- CHARMM36 FFXML (%s) PRO ideals ---", args.charmm_xml.name)
            logger.info("  PRO atom types: %s", cp["atom_types"])
            for k, v in cp["bonds_A"].items():
                logger.info("  bond  %-10s = %.3f A", k, v)
            for k, v in cp["angles_deg"].items():
                logger.info("  angle %-12s = %.2f deg", k, v)
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not parse CHARMM FFXML %s: %s", args.charmm_xml, exc)
    else:
        logger.info("CHARMM FFXML not found at %s (skipping cross-check)", args.charmm_xml)

    if args.out:
        args.out.write_text(json.dumps(report, indent=2))
        logger.info("wrote report -> %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
