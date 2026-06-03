#!/usr/bin/env python3
"""Measure the residual between CA-origin pb and MASTER rotlib.bin after primary fix.

HYPOTHESIS: The residual contact-degree drift is caused by one of:
  (a) uniform N-CA-CB angle offset (template angle is consistently wrong by ~2°)
  (b) single global backbone-axis rotation (frame is rotated uniformly)
  (c) per-residue scatter (not a simple fix)
  (d) other

MEASUREMENT PLAN:
1. Atom-order sanity check: confirm pb and rotlib.bin have same atom order per residue
2. N-CA-CB angle: compute from both pb and rotlib.bin coords (using N at [-1.458,0,0], CA at [0,0,0])
3. Per-residue Kabsch: top rotamer of most-populated shared (phi,psi) bin
4. Global Kabsch: pool all matched atoms across residues/bins
5. Lever-arm analysis: RMSD growth with distance from CA

DEPENDENCIES: uv run --with grpcio-tools --with zstandard --with protobuf --with scipy --with numpy
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import tempfile
from pathlib import Path
from subprocess import run as subprocess_run
from typing import Any

import numpy as np
import zstandard
from scipy.spatial.transform import Rotation

logger = logging.getLogger("measure_caorigin_residual_vs_master")


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


def parse_rotlib(path: Path) -> dict[str, dict]:
    """Parse the MSL binary rotamer library (from extract_rotlib_geometry.py)."""
    def _read_cstr(fp) -> str:
        import struct
        out = bytearray()
        while True:
            b = fp.read(1)
            if not b or b == b"\x00":
                break
            out += b
        return out.decode("ascii", "replace")

    def _read_i32(fp) -> int:
        import struct
        return struct.unpack("<i", fp.read(4))[0]

    def _read_f32(fp) -> float:
        import struct
        return struct.unpack("<f", fp.read(4))[0]

    entries: dict[str, dict] = {}
    with path.open("rb") as fp:
        while True:
            aa = _read_cstr(fp)
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


def compute_ncacb_angle(coords: list[list[float]], atom_names: list[str]) -> float | None:
    """Compute N-CA-CB angle (degrees) from sidechain coords in CA-origin frame.

    In the canonical backbone_frame: CA=[0,0,0], N at [-1.458,0,0], C in +xy half-plane.
    CB is the first sidechain atom (atom_names[0]) for non-glycine residues.

    Returns: angle at CA between N-CA direction [-1,0,0] and CB vector, or None if no CB.
    """
    if not atom_names or len(atom_names) == 0:
        return None  # Glycine has no sidechain atoms

    # CB is always the first atom in the protobuf
    cb_idx = 0
    if cb_idx >= len(coords):
        return None

    cb = np.array(coords[cb_idx], dtype=float)
    ca = np.array([0.0, 0.0, 0.0])  # CA is at origin in the frame
    n = np.array([-1.458, 0.0, 0.0])  # N is at [-1.458, 0, 0]

    # Vectors from CA
    v_n_ca = n - ca  # [-1.458, 0, 0]
    v_cb_ca = cb - ca  # CB vector

    # Angle between them
    dot = np.dot(v_n_ca, v_cb_ca)
    norm_n = np.linalg.norm(v_n_ca)
    norm_cb = np.linalg.norm(v_cb_ca)

    if norm_cb < 1e-6:
        return None

    cos_angle = dot / (norm_n * norm_cb)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)

    return float(angle_deg)


def kabsch(src_coords: np.ndarray, tgt_coords: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Kabsch algorithm: fit src onto tgt, return (rotation_matrix, translation, rmsd)."""
    # Center
    src_center = src_coords.mean(axis=0)
    tgt_center = tgt_coords.mean(axis=0)
    src_centered = src_coords - src_center
    tgt_centered = tgt_coords - tgt_center

    # SVD for rotation
    h = src_centered.T @ tgt_centered
    u, s, vt = np.linalg.svd(h)
    r = vt.T @ u.T

    # Ensure proper rotation (det(R) = 1, not -1)
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T

    # Translation
    t = tgt_center - r @ src_center

    # RMSD
    src_transformed = src_coords @ r.T + t
    rmsd = float(np.sqrt(np.mean(np.sum((src_transformed - tgt_coords) ** 2, axis=1))))

    return r, t, rmsd


def rotation_to_angle_axis(r: np.ndarray) -> tuple[float, np.ndarray]:
    """Convert rotation matrix to angle (degrees) and axis (unit vector)."""
    rotation = Rotation.from_matrix(r)
    angle_rad = rotation.magnitude()

    if angle_rad < 1e-6:
        return 0.0, np.array([0.0, 0.0, 1.0])

    axis = rotation.as_rotvec() / angle_rad
    angle_deg = np.degrees(angle_rad)

    return angle_deg, axis


def self_test_kabsch() -> bool:
    """Verify Kabsch fit on synthetic rotated point set."""
    logger.info("Running Kabsch self-test...")

    # Create source points
    src = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)

    # Rotate by exactly 30 degrees about Z-axis
    theta_deg = 30.0
    theta_rad = np.radians(theta_deg)
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)

    r_expected = np.array([
        [cos_t, -sin_t, 0.0],
        [sin_t, cos_t, 0.0],
        [0.0, 0.0, 1.0],
    ])

    tgt = src @ r_expected.T

    # Fit
    r_recovered, t, rmsd = kabsch(src, tgt)

    # Verify recovered rotation
    angle_deg_recovered, axis_recovered = rotation_to_angle_axis(r_recovered)

    logger.info(f"  Expected angle: {theta_deg:.2f}°, recovered: {angle_deg_recovered:.2f}°")
    logger.info(f"  Expected axis: Z, recovered: {axis_recovered}")
    logger.info(f"  RMSD: {rmsd:.2e}")

    # Tolerance: recovered angle should match within 0.01 degrees
    if abs(angle_deg_recovered - theta_deg) > 0.01:
        logger.error(f"Kabsch test FAILED: angle mismatch {angle_deg_recovered:.2f}° vs {theta_deg:.2f}°")
        return False

    if rmsd > 1e-6:
        logger.error(f"Kabsch test FAILED: RMSD {rmsd:.2e} too large")
        return False

    logger.info("Kabsch self-test PASSED")
    return True


def measure_ncacb_per_residue(
    pb_lib: Any,
    rotlib_bin_path: Path,
    residues: list[str] | None = None,
) -> dict[str, Any]:
    """Measure N-CA-CB angles per residue from both pb and rotlib.bin."""
    master_lib = parse_rotlib(rotlib_bin_path)

    if residues is None:
        residues = sorted(set(r.code for r in pb_lib.residues if r.code in master_lib))

    results = {}

    for res_code in residues:
        pb_res = None
        for r in pb_lib.residues:
            if r.code == res_code:
                pb_res = r
                break

        if pb_res is None:
            logger.warning(f"{res_code} not in pb")
            continue

        if res_code not in master_lib:
            logger.warning(f"{res_code} not in rotlib.bin")
            continue

        pb_names = list(pb_res.atom_names)
        master_names = master_lib[res_code]["atom_names"]

        # Sanity check: atom order
        if pb_names != master_names:
            logger.error(f"{res_code}: atom order mismatch. pb={pb_names}, master={master_names}")
            results[res_code] = {
                "error": "atom_order_mismatch",
                "pb_atoms": pb_names,
                "master_atoms": master_names,
            }
            continue

        # Find shared bins (by phi, psi center match)
        pb_bins_dict = {(float(b.phi), float(b.psi)): b for b in pb_res.bins}

        master_entry = master_lib[res_code]
        master_bins_list = master_entry["bins"]
        master_rot_bins = master_entry["rot_bins"]

        # Measure N-CA-CB angle for the most-populated shared bin
        best_pb_angle = None
        best_master_angle = None
        best_phi = None
        best_psi = None
        best_freq = -1

        for master_idx, (phi, psi, freq) in enumerate(master_bins_list):
            key = (phi, psi)
            if key not in pb_bins_dict:
                continue

            pb_bin = pb_bins_dict[key]

            # Use top rotamer
            if len(pb_bin.rotamers) == 0 or len(master_rot_bins[master_idx]) == 0:
                continue

            pb_rot = pb_bin.rotamers[0]
            master_rot = master_rot_bins[master_idx][0]

            pb_coords = [[c.x, c.y, c.z] for c in pb_rot.coords]
            master_coords = master_rot["coords"]

            pb_angle = compute_ncacb_angle(pb_coords, pb_names)
            master_angle = compute_ncacb_angle(master_coords, master_names)

            # Track most-populated bin
            if freq > best_freq:
                best_freq = freq
                best_pb_angle = pb_angle
                best_master_angle = master_angle
                best_phi = phi
                best_psi = psi

        if best_pb_angle is None or best_master_angle is None:
            logger.warning(f"{res_code}: no shared bins with valid coords")
            results[res_code] = {"error": "no_shared_bins"}
            continue

        delta_angle = best_pb_angle - best_master_angle
        results[res_code] = {
            "pb_ncacb_deg": round(float(best_pb_angle), 3),
            "master_ncacb_deg": round(float(best_master_angle), 3),
            "delta_deg": round(float(delta_angle), 3),
            "best_bin_phi": float(best_phi),
            "best_bin_psi": float(best_psi),
            "best_bin_freq": float(best_freq),
        }

    return results


def measure_per_residue_kabsch(
    pb_lib: Any,
    rotlib_bin_path: Path,
    residues: list[str] | None = None,
) -> dict[str, Any]:
    """Per-residue Kabsch fit of top rotamer in most-populated shared bin."""
    master_lib = parse_rotlib(rotlib_bin_path)

    if residues is None:
        residues = sorted(set(r.code for r in pb_lib.residues if r.code in master_lib))

    results = {}

    for res_code in residues:
        pb_res = None
        for r in pb_lib.residues:
            if r.code == res_code:
                pb_res = r
                break

        if pb_res is None or res_code not in master_lib:
            continue

        pb_names = list(pb_res.atom_names)
        master_names = master_lib[res_code]["atom_names"]

        if pb_names != master_names:
            continue

        pb_bins_dict = {(float(b.phi), float(b.psi)): b for b in pb_res.bins}
        master_entry = master_lib[res_code]
        master_bins_list = master_entry["bins"]
        master_rot_bins = master_entry["rot_bins"]

        # Most-populated shared bin
        best_pb_coords = None
        best_master_coords = None
        best_phi = None
        best_psi = None
        best_freq = -1

        for master_idx, (phi, psi, freq) in enumerate(master_bins_list):
            key = (phi, psi)
            if key not in pb_bins_dict:
                continue

            pb_bin = pb_bins_dict[key]
            if len(pb_bin.rotamers) == 0 or len(master_rot_bins[master_idx]) == 0:
                continue

            pb_rot = pb_bin.rotamers[0]
            master_rot = master_rot_bins[master_idx][0]

            if freq > best_freq:
                best_freq = freq
                best_pb_coords = np.array([[c.x, c.y, c.z] for c in pb_rot.coords], dtype=float)
                best_master_coords = np.array(master_rot["coords"], dtype=float)
                best_phi = phi
                best_psi = psi

        if best_pb_coords is None or best_master_coords is None:
            continue

        # Kabsch
        r, t, rmsd = kabsch(best_pb_coords, best_master_coords)
        angle_deg, axis = rotation_to_angle_axis(r)

        results[res_code] = {
            "best_bin_phi": float(best_phi),
            "best_bin_psi": float(best_psi),
            "rmsd_A": round(float(rmsd), 4),
            "rotation_angle_deg": round(float(angle_deg), 3),
            "rotation_axis": [round(float(x), 4) for x in axis],
            "translation_A": [round(float(x), 4) for x in t],
        }

    return results


def measure_global_kabsch(
    pb_lib: Any,
    rotlib_bin_path: Path,
    residues: list[str] | None = None,
) -> dict[str, Any]:
    """Global Kabsch: pool all matched atoms across residues and bins."""
    master_lib = parse_rotlib(rotlib_bin_path)

    if residues is None:
        residues = sorted(set(r.code for r in pb_lib.residues if r.code in master_lib))

    all_pb_coords = []
    all_master_coords = []

    for res_code in residues:
        pb_res = None
        for r in pb_lib.residues:
            if r.code == res_code:
                pb_res = r
                break

        if pb_res is None or res_code not in master_lib:
            continue

        pb_names = list(pb_res.atom_names)
        master_names = master_lib[res_code]["atom_names"]

        if pb_names != master_names:
            continue

        pb_bins_dict = {(float(b.phi), float(b.psi)): b for b in pb_res.bins}
        master_entry = master_lib[res_code]
        master_bins_list = master_entry["bins"]
        master_rot_bins = master_entry["rot_bins"]

        # All shared bins, all rotamers
        for master_idx, (phi, psi, freq) in enumerate(master_bins_list):
            key = (phi, psi)
            if key not in pb_bins_dict:
                continue

            pb_bin = pb_bins_dict[key]
            if len(pb_bin.rotamers) == 0 or len(master_rot_bins[master_idx]) == 0:
                continue

            pb_rot = pb_bin.rotamers[0]
            master_rot = master_rot_bins[master_idx][0]

            pb_coords = [[c.x, c.y, c.z] for c in pb_rot.coords]
            master_coords = master_rot["coords"]

            all_pb_coords.extend(pb_coords)
            all_master_coords.extend(master_coords)

    if len(all_pb_coords) == 0:
        return {"error": "no_matched_atoms"}

    pb_array = np.array(all_pb_coords, dtype=float)
    master_array = np.array(all_master_coords, dtype=float)

    r, t, rmsd = kabsch(pb_array, master_array)
    angle_deg, axis = rotation_to_angle_axis(r)

    return {
        "n_matched_atoms": len(all_pb_coords),
        "rmsd_A": round(float(rmsd), 4),
        "rotation_angle_deg": round(float(angle_deg), 3),
        "rotation_axis": [round(float(x), 4) for x in axis],
        "translation_A": [round(float(x), 4) for x in t],
    }


def measure_lever_arm(
    pb_lib: Any,
    rotlib_bin_path: Path,
    residues: list[str] | None = None,
) -> dict[str, Any]:
    """Lever-arm check: RMSD growth with distance from CA."""
    master_lib = parse_rotlib(rotlib_bin_path)

    if residues is None:
        residues = sorted(set(r.code for r in pb_lib.residues if r.code in master_lib))

    # Per-atom distance from CB and associated RMSD contribution
    atom_distances = []  # (distance_from_ca, rmsd_contrib)

    for res_code in residues:
        pb_res = None
        for r in pb_lib.residues:
            if r.code == res_code:
                pb_res = r
                break

        if pb_res is None or res_code not in master_lib:
            continue

        pb_names = list(pb_res.atom_names)
        master_names = master_lib[res_code]["atom_names"]

        if pb_names != master_names:
            continue

        pb_bins_dict = {(float(b.phi), float(b.psi)): b for b in pb_res.bins}
        master_entry = master_lib[res_code]
        master_bins_list = master_entry["bins"]
        master_rot_bins = master_entry["rot_bins"]

        for master_idx, (phi, psi, freq) in enumerate(master_bins_list):
            key = (phi, psi)
            if key not in pb_bins_dict:
                continue

            pb_bin = pb_bins_dict[key]
            if len(pb_bin.rotamers) == 0 or len(master_rot_bins[master_idx]) == 0:
                continue

            pb_rot = pb_bin.rotamers[0]
            master_rot = master_rot_bins[master_idx][0]

            pb_coords = [[c.x, c.y, c.z] for c in pb_rot.coords]
            master_coords = master_rot["coords"]

            # CB is first atom, at index 0
            cb_pb = np.array(pb_coords[0], dtype=float)
            cb_master = np.array(master_coords[0], dtype=float)

            for i, (pb_c, master_c) in enumerate(zip(pb_coords, master_coords)):
                pb_a = np.array(pb_c, dtype=float)
                master_a = np.array(master_c, dtype=float)

                # Distance from CB
                dist_from_cb = float(np.linalg.norm(pb_a - cb_pb))

                # Per-atom difference
                delta = pb_a - master_a
                rmsd_contrib = float(np.sqrt(np.sum(delta ** 2)))

                atom_distances.append((dist_from_cb, rmsd_contrib))

    if not atom_distances:
        return {"error": "no_atoms_measured"}

    # Sort by distance and report statistics
    atom_distances.sort(key=lambda x: x[0])

    # Bin by distance ranges
    bins = {
        "0_1A": [],
        "1_2A": [],
        "2_3A": [],
        "3_4A": [],
        "4plus_A": [],
    }

    for dist, rmsd_c in atom_distances:
        if dist < 1.0:
            bins["0_1A"].append(rmsd_c)
        elif dist < 2.0:
            bins["1_2A"].append(rmsd_c)
        elif dist < 3.0:
            bins["2_3A"].append(rmsd_c)
        elif dist < 4.0:
            bins["3_4A"].append(rmsd_c)
        else:
            bins["4plus_A"].append(rmsd_c)

    result = {}
    for bin_name, values in bins.items():
        if values:
            result[bin_name] = {
                "n_atoms": len(values),
                "mean_rmsd_A": round(float(np.mean(values)), 4),
                "median_rmsd_A": round(float(np.median(values)), 4),
                "max_rmsd_A": round(float(np.max(values)), 4),
            }

    # Linear regression: distance vs rmsd to check for lever-arm signature
    distances_arr = np.array([d for d, _ in atom_distances])
    rmsd_arr = np.array([r for _, r in atom_distances])

    if len(distances_arr) > 1:
        coeffs = np.polyfit(distances_arr, rmsd_arr, 1)
        slope = float(coeffs[0])
        result["linear_fit"] = {
            "slope_rmsd_per_A": round(slope, 4),
            "interpretation": "positive slope indicates lever-arm signature (rotation)",
        }

    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pb", type=Path,
                    default=Path("/home/marielle/projects/proxide/.claude/worktrees/loadpb-frame-fix/data/rotlibs/proxide-rotlib-bbdep2010.pb.zst"),
                    help="input proxide rotlib .pb.zst (CA-origin)")
    ap.add_argument("--rotlib-bin", type=Path,
                    default=Path("/home/marielle/repos/mosaist/testfiles/rotlib.bin"),
                    help="MASTER rotlib.bin oracle")
    ap.add_argument("--residues", nargs="+", default=None,
                    help="residue codes to measure (default: all in both libraries)")
    ap.add_argument("--output", type=Path, default=None,
                    help="write JSON report here")
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    ap.add_argument("--dry-run", action="store_true",
                    help="skip actual measurements")
    ap.add_argument("--self-test", action="store_true",
                    help="run self-tests and exit")

    args = ap.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # Self-tests
    if args.self_test:
        if not self_test_kabsch():
            return 1
        logger.info("All self-tests passed")
        return 0

    # Compile proto
    logger.info("Compiling rotlib.proto...")
    proto_dir = Path(__file__).parent.parent.parent / "crates" / "proxide-rotlib" / "proto"
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            proto_py_path = compile_proto_runtime(proto_dir, Path(tmpdir))
            rotlib_pb2 = import_rotlib_pb2(proto_py_path)
        except Exception as e:
            logger.error(f"Proto compilation failed: {e}", exc_info=True)
            return 1

        try:
            # Load pb
            logger.info(f"Loading pb from {args.pb}")
            raw_pb = load_pb(args.pb)
            pb_lib = rotlib_pb2.RotamerLibrary()
            pb_lib.ParseFromString(raw_pb)
            logger.info(f"Loaded pb with {len(pb_lib.residues)} residues")

            # Determine residues to measure
            if args.residues is None:
                pb_codes = set(r.code for r in pb_lib.residues)
                master_lib = parse_rotlib(args.rotlib_bin)
                master_codes = set(master_lib.keys())
                measure_codes = sorted(pb_codes & master_codes)
                logger.info(f"Measuring {len(measure_codes)} residues in both libraries: {measure_codes}")
            else:
                measure_codes = args.residues

            if args.dry_run:
                logger.info("Dry-run: skipping measurements")
                return 0

            # Measurements
            logger.info("Measuring N-CA-CB angles...")
            ncacb_results = measure_ncacb_per_residue(pb_lib, args.rotlib_bin, measure_codes)

            logger.info("Measuring per-residue Kabsch fits...")
            per_res_kabsch = measure_per_residue_kabsch(pb_lib, args.rotlib_bin, measure_codes)

            logger.info("Measuring global Kabsch fit...")
            global_kabsch = measure_global_kabsch(pb_lib, args.rotlib_bin, measure_codes)

            logger.info("Measuring lever-arm signature...")
            lever_arm = measure_lever_arm(pb_lib, args.rotlib_bin, measure_codes)

            # Assemble report
            report = {
                "pb_path": str(args.pb),
                "rotlib_bin_path": str(args.rotlib_bin),
                "residues_measured": measure_codes,
                "ncacb_angles": ncacb_results,
                "per_residue_kabsch": per_res_kabsch,
                "global_kabsch": global_kabsch,
                "lever_arm_analysis": lever_arm,
            }

            # Determine conclusion
            # Check if N-CA-CB angles are uniformly offset
            ncacb_valid = [r["delta_deg"] for r in ncacb_results.values() if "delta_deg" in r]
            if ncacb_valid:
                mean_delta = float(np.mean(ncacb_valid))
                std_delta = float(np.std(ncacb_valid))
                report["ncacb_analysis"] = {
                    "mean_delta_deg": round(mean_delta, 3),
                    "std_delta_deg": round(std_delta, 3),
                    "interpretation": "uniform offset" if std_delta < 0.5 else "per-residue scatter",
                }

            # Check global rotation
            if "rotation_angle_deg" in global_kabsch:
                rot_angle = global_kabsch["rotation_angle_deg"]
                report["rotation_analysis"] = {
                    "global_rotation_deg": rot_angle,
                    "is_significant": rot_angle > 0.5,
                }

            # Final conclusion
            conclusions = []

            if "ncacb_analysis" in report and report["ncacb_analysis"]["std_delta_deg"] < 0.5:
                mean_d = report["ncacb_analysis"]["mean_delta_deg"]
                conclusions.append(f"Uniform N-CA-CB offset of {mean_d:+.2f}° detected")

            if "rotation_analysis" in report and report["rotation_analysis"]["is_significant"]:
                rot_angle = report["rotation_analysis"]["global_rotation_deg"]
                rot_axis = global_kabsch.get("rotation_axis", [0, 0, 0])
                conclusions.append(f"Global rotation of {rot_angle:.2f}° about axis {rot_axis} detected")

            if not conclusions:
                conclusions.append("Residual is per-residue scatter or other (not simple uniform offset/rotation)")

            report["conclusion"] = " | ".join(conclusions)

            # Output
            report_json = json.dumps(report, indent=2)
            print(report_json)

            if args.output:
                logger.info(f"Writing report to {args.output}")
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(report_json)

            return 0

        except Exception as e:
            logger.error(f"Measurement failed: {e}", exc_info=True)
            return 1


if __name__ == "__main__":
    sys.exit(main())
