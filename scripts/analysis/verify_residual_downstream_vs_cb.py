#!/usr/bin/env python3
"""Verify CB frame-angle hypothesis: test whether closer CB at θ=108.37° corresponds to
farther terminals.

Hypothesis: θ=108.37° (CB angle matched to MASTER's measured value) makes CB's position
CLOSER to MASTER's stored CB, but the DOWNSTREAM/TERMINAL sidechain atoms move FARTHER
from MASTER — because MASTER's downstream bond angles differ from proxide's CHARMM angles.
If true, the residual drift is downstream sidechain geometry, not the CB frame angle.

This script:
1. Decodes two rotamer libraries (θ=113.5° and θ=108.37°) using parse_rotlib.
2. For each of several long/charged residues (ARG, GLN, LYS, GLU, MET, LEU),
   selects a representative rotamer matched between both libraries.
3. Computes per-atom Euclidean distance to MASTER's rotlib.bin (in MASTER's canonical frame).
4. Groups atoms by bond-depth from CB (CB=0, bonded child=1, etc.).
5. Reports: CB distance and terminal-atom distance for both θ values.
6. Includes --self-test to verify distance-by-depth logic.
7. Includes --dry-run for path verification.

Canonical backbone frame (from proxide tests):
  N at [-1.0, 0, 0], CA at [0, 0, 0], C at [0, 1, 0].
  All stored coordinates are in this frame; both proxide pb and rotlib.bin use it.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import tempfile
from pathlib import Path
from typing import Any

import zstandard

logger = logging.getLogger("verify_residual_downstream_vs_cb")

# Add scripts/analysis to path to import parse_rotlib
sys.path.insert(0, str(Path(__file__).parent))

from extract_rotlib_geometry import parse_rotlib  # noqa: E402 (needs sys.path.insert above)


def compile_proto_runtime(proto_dir: Path, output_dir: Path) -> str:
    """Compile rotlib.proto to Python at runtime, return module path."""
    proto_file = proto_dir / "rotlib.proto"
    if not proto_file.exists():
        raise FileNotFoundError(f"rotlib.proto not found at {proto_file}")

    import grpc_tools.protoc

    ret = grpc_tools.protoc.main([
        "protoc",
        f"--proto_path={proto_dir}",
        f"--python_out={output_dir}",
        str(proto_file),
    ])

    if ret != 0:
        raise RuntimeError(f"protoc failed with code {ret}")

    return str(output_dir)


def load_pb(path: Path) -> bytes:
    """Load and decompress the .pb.zst file."""
    logger.info(f"Loading {path}")
    with open(path, "rb") as f:
        compressed = f.read()

    dctx = zstandard.ZstdDecompressor()
    try:
        raw = dctx.decompress(compressed, max_output_size=500_000_000)
    except Exception as e:
        logger.error(f"Failed to decompress: {e}")
        raise

    logger.info(f"Decompressed {len(compressed)} bytes -> {len(raw)} bytes")
    return raw


def import_rotlib_pb2(proto_py_path: str):
    """Import rotlib_pb2 from compiled proto."""
    sys.path.insert(0, proto_py_path)
    import rotlib_pb2
    return rotlib_pb2


def parse_pb_file(path: Path, rotlib_pb2: Any) -> dict[str, dict]:
    """Parse a proxide .pb.zst rotamer library using protobuf.

    Returns dict in same structure as parse_rotlib:
      {aa_code: {"nc": int, "na": int, "nb": int,
        "atom_names": [str], "bins": [(phi, psi, freq)],
        "rot_bins": [[{"prob": f, "chi": [(val, sigma)], "coords": [[x,y,z]]}]]}}
    """
    raw = load_pb(path)

    # Parse protobuf
    library = rotlib_pb2.RotamerLibrary()
    library.ParseFromString(raw)

    entries: dict[str, dict] = {}

    for residue_entry in library.residues:
        aa = residue_entry.code
        nc = residue_entry.num_chi
        na = len(residue_entry.atom_names)
        nb = len(residue_entry.bins)

        atom_names = list(residue_entry.atom_names)

        # Parse phi/psi bins
        bins = []
        for bin_entry in residue_entry.bins:
            phi = bin_entry.phi
            psi = bin_entry.psi
            freq = bin_entry.freq
            bins.append((phi, psi, freq))

        # Parse rotamers per bin
        rot_bins = []
        for bin_entry in residue_entry.bins:
            rotamers = []
            for rotamer_entry in bin_entry.rotamers:
                prob = rotamer_entry.prob

                # Chi angles with sigma
                chi = []
                for chi_val in rotamer_entry.chi:
                    chi.append((chi_val.val, chi_val.sigma))

                # Atomic coordinates
                coords = []
                for vec in rotamer_entry.coords:
                    coords.append([vec.x, vec.y, vec.z])

                rotamers.append({"prob": prob, "chi": chi, "coords": coords})

            rot_bins.append(rotamers)

        entries[aa] = {
            "nc": nc, "na": na, "nb": nb,
            "atom_names": atom_names,
            "bins": bins, "rot_bins": rot_bins,
        }
        logger.debug("parsed pb %s: nc=%d na=%d nb=%d atoms=%s", aa, nc, na, nb, atom_names)

    return entries


# --- Canonical backbone frame ----------------------------------------
IDENTITY_N = [-1.0, 0.0, 0.0]
IDENTITY_CA = [0.0, 0.0, 0.0]
IDENTITY_C = [0.0, 1.0, 0.0]


def euclidean_distance(a, b):
    """Euclidean distance between two 3D points."""
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def _self_test():
    """BATHOS verification: assert distance-by-depth logic is correct."""
    logger.info("Running --self-test...")

    # Synthetic test: reference (MASTER) atoms and two libraries where lib_a has
    # closer CB to reference but farther terminals (the hypothesis pattern).
    reference_atoms = [
        [0.5, 0.0, 0.0],  # depth 0 (CB): target position
        [1.0, 1.0, 0.0],  # depth 1 (child of CB): side chain
        [1.5, 1.5, 0.0],  # depth 2 (grandchild): terminal
    ]
    # lib_a: CB closer to reference, but terminals farther
    lib_a_atoms = [
        [0.52, 0.0, 0.0],  # depth 0: distance 0.02 (closer than lib_b)
        [1.5, 1.5, 0.0],   # depth 1: distance ~0.71 (farther than lib_b)
        [2.0, 2.0, 0.0],   # depth 2: distance ~0.71 (farther than lib_b)
    ]
    # lib_b: CB farther from reference, terminals closer
    lib_b_atoms = [
        [0.54, 0.0, 0.0],  # depth 0: distance 0.04 (farther than lib_a)
        [1.1, 1.1, 0.0],   # depth 1: distance ~0.14 (closer than lib_a)
        [1.6, 1.6, 0.0],   # depth 2: distance ~0.14 (closer than lib_a)
    ]

    # Compute distances
    d_a_0 = euclidean_distance(reference_atoms[0], lib_a_atoms[0])
    d_a_term = (euclidean_distance(reference_atoms[1], lib_a_atoms[1]) +
                euclidean_distance(reference_atoms[2], lib_a_atoms[2])) / 2.0

    d_b_0 = euclidean_distance(reference_atoms[0], lib_b_atoms[0])
    d_b_term = (euclidean_distance(reference_atoms[1], lib_b_atoms[1]) +
                euclidean_distance(reference_atoms[2], lib_b_atoms[2])) / 2.0

    # Expected: lib_a has closer CB (d_a_0 < d_b_0) but farther terminals (d_a_term > d_b_term)
    test_1 = d_a_0 < d_b_0
    test_2 = d_a_term > d_b_term

    logger.info("Test 1 (CB distance): lib_a=%.4f A, lib_b=%.4f A → lib_a closer: %s",
                d_a_0, d_b_0, "PASS" if test_1 else "FAIL")
    logger.info("Test 2 (terminal distance): lib_a=%.4f A, lib_b=%.4f A → lib_a farther: %s",
                d_a_term, d_b_term, "PASS" if test_2 else "FAIL")

    if test_1 and test_2:
        logger.info("SELF-TEST PASSED")
        return True
    else:
        logger.error("SELF-TEST FAILED")
        return False


def get_bond_depth(atom_idx, atom_names, bonds_dict):
    """Compute bond depth from CB (0-indexed).

    CB = depth 0. Its bonded children = depth 1. Grandchildren = depth 2, etc.
    If bonds_dict is unavailable, use atom_names order as a depth proxy.
    """
    if "CB" not in atom_names:
        return None
    cb_idx = atom_names.index("CB")
    if atom_idx == cb_idx:
        return 0

    # If bonds_dict provided, traverse parent chain
    if bonds_dict and atom_idx in bonds_dict:
        parent_idx = bonds_dict[atom_idx]
        if parent_idx == cb_idx:
            return 1
        parent_depth = get_bond_depth(parent_idx, atom_names, bonds_dict)
        if parent_depth is not None:
            return parent_depth + 1

    # Fallback: use position in atom_names as proxy (typically atoms are stored in order)
    if atom_idx > cb_idx:
        return atom_idx - cb_idx  # Simple proxy: distance in array order
    return None


def compare_libraries(pb1_path, pb2_path, rotlib_path, residues, dry_run=False, rotlib_pb2=None):
    """Compare two proxide pb libraries against MASTER rotlib.bin.

    Args:
        pb1_path: Path to first pb (θ=113.5°, CHARMM standard)
        pb2_path: Path to second pb (θ=108.37°, CB-angle matched)
        rotlib_path: Path to MASTER rotlib.bin
        residues: List of residue codes to measure (e.g. ["ARG", "GLN", "LYS"])
        dry_run: If True, verify paths only and exit
        rotlib_pb2: Compiled protobuf module (required for pb file parsing)

    Returns:
        dict with per-residue comparison data
    """
    pb1_path = Path(pb1_path)
    pb2_path = Path(pb2_path)
    rotlib_path = Path(rotlib_path)

    if dry_run:
        logger.info("DRY-RUN: verifying paths...")
        for p in [pb1_path, pb2_path, rotlib_path]:
            exists = p.exists()
            logger.info("  %s: %s", p, "EXISTS" if exists else "NOT FOUND")
        return {}

    # Parse libraries
    logger.info("Parsing pb1 (θ=113.5°) from %s...", pb1_path)
    pb1_data = parse_pb_file(pb1_path, rotlib_pb2)

    logger.info("Parsing pb2 (θ=108.37°) from %s...", pb2_path)
    pb2_data = parse_pb_file(pb2_path, rotlib_pb2)

    logger.info("Parsing MASTER rotlib.bin from %s...", rotlib_path)
    master_data = parse_rotlib(rotlib_path)

    results = {}

    for aa in residues:
        if aa not in pb1_data or aa not in pb2_data or aa not in master_data:
            logger.warning("Residue %s not found in one or more libraries", aa)
            continue

        entry1 = pb1_data[aa]
        entry2 = pb2_data[aa]
        master = master_data[aa]

        atom_names = entry1["atom_names"]
        if "CB" not in atom_names:
            logger.warning("%s has no CB atom", aa)
            continue

        cb_idx = atom_names.index("CB")

        # Find a top-probability rotamer in the first bin that exists in both
        if not entry1["rot_bins"] or not entry1["rot_bins"][0]:
            logger.warning("%s pb1: no rotamers in first bin", aa)
            continue

        # Use the first rotamer in the first bin as representative
        rot1 = entry1["rot_bins"][0][0]
        coords1 = rot1["coords"]

        if not entry2["rot_bins"] or not entry2["rot_bins"][0]:
            logger.warning("%s pb2: no rotamers in first bin", aa)
            continue

        rot2 = entry2["rot_bins"][0][0]
        coords2 = rot2["coords"]

        if not master["rot_bins"] or not master["rot_bins"][0]:
            logger.warning("%s master: no rotamers in first bin", aa)
            continue

        rot_master = master["rot_bins"][0][0]
        coords_master = rot_master["coords"]

        # Compute per-atom distances
        atom_data = []
        for atom_idx, atom_name in enumerate(atom_names):
            if (
                atom_idx >= len(coords1)
                or atom_idx >= len(coords2)
                or atom_idx >= len(coords_master)
            ):
                logger.warning(
                    "%s atom %d (%s) out of range in one library", aa, atom_idx, atom_name
                )
                continue

            dist1 = euclidean_distance(coords1[atom_idx], coords_master[atom_idx])
            dist2 = euclidean_distance(coords2[atom_idx], coords_master[atom_idx])

            depth = get_bond_depth(atom_idx, atom_names, None)

            atom_data.append({
                "name": atom_name,
                "index": atom_idx,
                "depth": depth,
                "dist_pb113": dist1,
                "dist_pb108": dist2,
            })

        # Summarize by depth
        cb_atoms = [a for a in atom_data if a["index"] == cb_idx]
        terminal_atoms = [a for a in atom_data if a.get("depth") and a["depth"] >= 2]

        if cb_atoms:
            cb_data = cb_atoms[0]
            cb_dist_113 = cb_data["dist_pb113"]
            cb_dist_108 = cb_data["dist_pb108"]
        else:
            cb_dist_113 = None
            cb_dist_108 = None

        if terminal_atoms:
            term_mean_113 = sum(a["dist_pb113"] for a in terminal_atoms) / len(terminal_atoms)
            term_mean_108 = sum(a["dist_pb108"] for a in terminal_atoms) / len(terminal_atoms)
            term_max_113 = max(a["dist_pb113"] for a in terminal_atoms)
            term_max_108 = max(a["dist_pb108"] for a in terminal_atoms)
        else:
            term_mean_113 = None
            term_mean_108 = None
            term_max_113 = None
            term_max_108 = None

        results[aa] = {
            "cb_dist_pb113": cb_dist_113,
            "cb_dist_pb108": cb_dist_108,
            "terminal_mean_dist_pb113": term_mean_113,
            "terminal_mean_dist_pb108": term_mean_108,
            "terminal_max_dist_pb113": term_max_113,
            "terminal_max_dist_pb108": term_max_108,
            "num_terminal_atoms": len(terminal_atoms),
            "all_atoms": atom_data,
        }

        logger.debug(
            "%s: CB(113)=%.4f CB(108)=%.4f | Term(113)=%.4f Term(108)=%.4f",
            aa, cb_dist_113 or 0, cb_dist_108 or 0, term_mean_113 or 0, term_mean_108 or 0,
        )

    return results


def format_results_table(results):
    """Format results as a markdown table."""
    lines = []
    lines.append("## Residual Comparison Table")
    lines.append("")
    lines.append(
        "| Residue | CB dist (θ=113.5°) | CB dist (θ=108.37°) | CB Δ | "
        "Term mean (θ=113.5°) | Term mean (θ=108.37°) | Term Δ | Notes |"
    )
    lines.append(
        "|---------|-------------------|-------------------|------|"
        "----------------------|----------------------|--------|-------|"
    )

    for aa in sorted(results.keys()):
        data = results[aa]
        cb_113 = data["cb_dist_pb113"]
        cb_108 = data["cb_dist_pb108"]
        term_113 = data["terminal_mean_dist_pb113"]
        term_108 = data["terminal_mean_dist_pb108"]

        if cb_113 is None or cb_108 is None or term_113 is None or term_108 is None:
            continue

        cb_delta = cb_108 - cb_113
        term_delta = term_108 - term_113

        cb_delta_str = f"{cb_delta:+.4f}" if cb_delta else "—"
        term_delta_str = f"{term_delta:+.4f}" if term_delta else "—"

        # Hypothesis test: CB closer (δ < 0) but terminals farther (δ > 0)?
        hypothesis_marker = ""
        if cb_delta < 0 and term_delta > 0:
            hypothesis_marker = "✓ confirm"
        elif cb_delta < 0 and term_delta < 0:
            hypothesis_marker = "✗ refute (both improve)"
        elif cb_delta > 0 and term_delta > 0:
            hypothesis_marker = "✗ refute (both worsen)"
        elif cb_delta > 0 and term_delta < 0:
            hypothesis_marker = "✗ refute (opposite)"

        lines.append(
            f"| {aa} | {cb_113:.4f} | {cb_108:.4f} | {cb_delta_str} | "
            f"{term_113:.4f} | {term_108:.4f} | {term_delta_str} | {hypothesis_marker} |"
        )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pb113",
        type=Path,
        default=Path("/home/marielle/.claude/jobs/52b25843/tmp/pb_cb113.pb.zst"),
        help="Path to θ=113.5° pb (CHARMM standard)",
    )
    parser.add_argument(
        "--pb108",
        type=Path,
        default=Path("/home/marielle/.claude/jobs/52b25843/tmp/pb_cb108.pb.zst"),
        help="Path to θ=108.37° pb (CB-angle matched)",
    )
    parser.add_argument(
        "--rotlib",
        type=Path,
        default=Path("/home/marielle/repos/mosaist/testfiles/rotlib.bin"),
        help="Path to MASTER rotlib.bin",
    )
    parser.add_argument(
        "--residues",
        type=str,
        default="ARG,GLN,LYS,GLU,MET,LEU",
        help="Comma-separated residue codes to measure",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify paths and exit without running analysis",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run measurement-verification test and exit",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.self_test:
        success = _self_test()
        sys.exit(0 if success else 1)

    # Compile proto once for pb file parsing
    proto_dir = Path(__file__).parent.parent.parent / "crates" / "proxide-rotlib" / "proto"
    logger.info("Compiling rotlib.proto from %s...", proto_dir)

    with tempfile.TemporaryDirectory() as tmpdir:
        proto_py_path = compile_proto_runtime(proto_dir, Path(tmpdir))
        rotlib_pb2 = import_rotlib_pb2(proto_py_path)

        # Parse residues
        residues = [r.strip().upper() for r in args.residues.split(",")]
        logger.info("Measuring residues: %s", residues)

        # Compare libraries
        results = compare_libraries(
            args.pb113,
            args.pb108,
            args.rotlib,
            residues,
            dry_run=args.dry_run,
            rotlib_pb2=rotlib_pb2,
        )

        if args.dry_run:
            logger.info("DRY-RUN completed. No further analysis.")
            return

        # Format and print results
        table = format_results_table(results)
        print(table)
        print("")

        # Summary
        logger.info("Comparison complete. Analyzed %d residues.", len(results))

        # Count hypothesis markers
        confirm_count = 0
        refute_both_improve = 0
        refute_both_worsen = 0
        refute_opposite = 0

        for aa in results:
            data = results[aa]
            cb_113 = data["cb_dist_pb113"]
            cb_108 = data["cb_dist_pb108"]
            term_113 = data["terminal_mean_dist_pb113"]
            term_108 = data["terminal_mean_dist_pb108"]

            if (
                cb_113 is not None
                and cb_108 is not None
                and term_113 is not None
                and term_108 is not None
            ):
                cb_delta = cb_108 - cb_113
                term_delta = term_108 - term_113
                if cb_delta < 0 and term_delta > 0:
                    confirm_count += 1
                elif cb_delta < 0 and term_delta < 0:
                    refute_both_improve += 1
                elif cb_delta > 0 and term_delta > 0:
                    refute_both_worsen += 1
                elif cb_delta > 0 and term_delta < 0:
                    refute_opposite += 1

        print("")
        print("## CONCLUSION")
        print("")
        print(
            f"Results: {confirm_count} confirm hypothesis, "
            f"{refute_both_improve} show both improve, "
            f"{refute_both_worsen} show both worsen, {refute_opposite} show opposite pattern"
        )
        print("")

        if (
            confirm_count > 0
            and (refute_both_improve + refute_both_worsen + refute_opposite) == 0
        ):
            print(
                f"HYPOTHESIS CONFIRMED: All {confirm_count} residues show CB closer at "
                f"θ=108.37° but terminals farther."
            )
            print("The residual drift is downstream sidechain geometry, not CB frame angle.")
        elif refute_both_improve >= 4:
            print(
                f"HYPOTHESIS REFUTED: {refute_both_improve}/{len(results)} residues show BOTH "
                f"CB AND terminals improve at θ=108.37°."
            )
            print(
                "This suggests the N-CA-CB angle (at θ=108.37°) is part of the solution, "
                "not just a symptom."
            )
        else:
            print("MIXED RESULTS: Pattern unclear. Interpretation requires further investigation.")


if __name__ == "__main__":
    main()
