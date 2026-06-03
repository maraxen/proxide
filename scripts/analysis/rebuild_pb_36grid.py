#!/usr/bin/env python3
"""Rebuild the proxide rotamer library protobuf with 36x36 phi/psi grid.

HYPOTHESIS: The ConFind load_pb contact-degree drift vs MASTER rotlib.bin
is caused by a redundant +180° wrap column in proxide's protobuf grid
(37×37 instead of 36×36). The +180 column is a byte-duplicate of -180.
Removing it should collapse the drift to ~0, proving grid-packaging is
the sole root cause.

DEPENDENCIES: Must be run as:
  uv run --with grpcio-tools --with zstandard --with protobuf python \
    scripts/analysis/rebuild_pb_36grid.py

Compiles proto at runtime via grpc_tools.protoc.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from subprocess import run as subprocess_run

# grpcio-tools, zstandard, protobuf available via uv --with
import zstandard

logger = logging.getLogger("rebuild_pb_36grid")


def compile_proto_runtime(proto_dir: Path, output_dir: Path) -> str:
    """Compile rotlib.proto to Python at runtime, return module path."""
    proto_file = proto_dir / "rotlib.proto"
    if not proto_file.exists():
        raise FileNotFoundError(f"rotlib.proto not found at {proto_file}")

    # grpc_tools.protoc.main expects sys.argv[0] = program name, [1:] = args
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


def filter_grid_36(lib):
    """Remove all bins where phi==180 or psi==180 (within 1e-6).

    Returns a new RotamerLibrary with 36x36 grids per residue.
    """
    import rotlib_pb2

    new_lib = rotlib_pb2.RotamerLibrary()
    new_lib.version = lib.version
    new_lib.provenance = lib.provenance
    new_lib.attribution = lib.attribution
    new_lib.data_license = lib.data_license
    new_lib.geometry_mode = lib.geometry_mode

    for res in lib.residues:
        new_res = rotlib_pb2.ResidueEntry()
        new_res.code = res.code
        new_res.atom_names.extend(res.atom_names)
        new_res.num_chi = res.num_chi
        new_res.default_bin = res.default_bin

        # Filter phi and psi centers: remove 180.0
        eps = 1e-6
        new_res.phi_centers.extend([p for p in res.phi_centers if abs(p - 180.0) > eps])
        new_res.psi_centers.extend([p for p in res.psi_centers if abs(p - 180.0) > eps])

        # Filter bins: drop any bin with phi==180 or psi==180
        for bin_entry in res.bins:
            phi, psi = bin_entry.phi, bin_entry.psi
            if abs(phi - 180.0) > eps and abs(psi - 180.0) > eps:
                new_bin = rotlib_pb2.Bin()
                new_bin.phi = phi
                new_bin.psi = psi
                new_bin.freq = bin_entry.freq
                for rot in bin_entry.rotamers:
                    new_rot = rotlib_pb2.Rotamer()
                    new_rot.prob = rot.prob
                    new_rot.chi.extend(rot.chi)
                    new_rot.coords.extend(rot.coords)
                    new_bin.rotamers.append(new_rot)
                new_res.bins.append(new_bin)

        new_lib.residues.append(new_res)

    return new_lib


def verify_identity_with_master(pb_lib, rotlib_bin_path: Path) -> dict:
    """Compare rotamer sets between filtered pb and rotlib.bin for all shared bins.

    Returns dict: {
        total_shared_bins: int,
        n_bins_identical: int,
        n_bins_mismatch: int,
        mismatches: list of (code, phi, psi, first_diff_field) tuples (up to 10),
    }
    """
    # Import parse_rotlib from the sibling script
    sys.path.insert(0, str(Path(__file__).parent))
    from extract_rotlib_geometry import parse_rotlib

    logger.info(f"Parsing rotlib.bin from {rotlib_bin_path}")
    master_lib = parse_rotlib(rotlib_bin_path)

    total_shared = 0
    identical = 0
    mismatches_list = []

    for res in pb_lib.residues:
        code = res.code

        if code not in master_lib:
            logger.debug(f"Residue {code} not in rotlib.bin, skipping")
            continue

        master_entry = master_lib[code]
        master_bins_dict = {(b[0], b[1]): (b[2], rotamers)
                            for b, rotamers in zip(master_entry["bins"], master_entry["rot_bins"])}

        for bin_entry in res.bins:
            phi, psi = bin_entry.phi, bin_entry.psi
            key = (phi, psi)

            if key not in master_bins_dict:
                logger.debug(f"{code} ({phi}, {psi}) not in master")
                continue

            total_shared += 1
            master_freq, master_rotamers = master_bins_dict[key]

            # Compare rotamer counts
            if len(bin_entry.rotamers) != len(master_rotamers):
                logger.warning(f"{code} ({phi}, {psi}): rotamer count mismatch "
                               f"{len(bin_entry.rotamers)} vs {len(master_rotamers)}")
                if len(mismatches_list) < 10:
                    mismatches_list.append((code, phi, psi, "rotamer_count"))
                continue

            # Compare each rotamer
            match = True
            for i, (pb_rot, master_rot) in enumerate(zip(bin_entry.rotamers, master_rotamers)):
                pb_prob = pb_rot.prob
                master_prob = master_rot["prob"]
                if abs(pb_prob - master_prob) > 1e-5:
                    logger.warning(f"{code} ({phi}, {psi}) rot{i}: prob mismatch "
                                   f"{pb_prob} vs {master_prob}")
                    if len(mismatches_list) < 10:
                        mismatches_list.append((code, phi, psi, f"prob_rot{i}"))
                    match = False
                    break

                # Compare chi values
                pb_chi = [c.val for c in pb_rot.chi]
                master_chi = [c[0] for c in master_rot["chi"]]
                if len(pb_chi) != len(master_chi):
                    logger.warning(f"{code} ({phi}, {psi}) rot{i}: chi count mismatch")
                    if len(mismatches_list) < 10:
                        mismatches_list.append((code, phi, psi, f"chi_count_rot{i}"))
                    match = False
                    break

                for j, (pb_c, master_c) in enumerate(zip(pb_chi, master_chi)):
                    if abs(pb_c - master_c) > 1e-2:
                        logger.warning(f"{code} ({phi}, {psi}) rot{i} chi{j}: "
                                       f"{pb_c} vs {master_c}")
                        if len(mismatches_list) < 10:
                            mismatches_list.append((code, phi, psi, f"chi{j}_rot{i}"))
                        match = False
                        break

                if not match:
                    break

            if match:
                identical += 1

    return {
        "total_shared_bins": total_shared,
        "n_bins_identical": identical,
        "n_bins_mismatch": total_shared - identical,
        "mismatches": [(code, float(phi), float(psi), field)
                       for code, phi, psi, field in mismatches_list],
    }


def self_test():
    """Verify the filter logic with synthetic data."""
    import rotlib_pb2

    logger.info("Running self-test...")

    # Build a tiny library
    lib = rotlib_pb2.RotamerLibrary()
    lib.version = 1
    lib.provenance = "test"
    lib.attribution = "test"
    lib.data_license = "test"
    lib.geometry_mode = rotlib_pb2.PRECOMPUTED

    res = rotlib_pb2.ResidueEntry()
    res.code = "ALA"
    res.atom_names.extend(["CB"])
    res.num_chi = 1
    res.default_bin = 0

    # Centers: -180, 170, 180
    res.phi_centers.extend([-180.0, 170.0, 180.0])
    # Centers: -180, 180
    res.psi_centers.extend([-180.0, 180.0])

    # Create 6 bins covering phi×psi:
    # (-180, -180), (-180, 180), (170, -180), (170, 180), (180, -180), (180, 180)
    for phi in [-180.0, 170.0, 180.0]:
        for psi in [-180.0, 180.0]:
            bin_entry = rotlib_pb2.Bin()
            bin_entry.phi = phi
            bin_entry.psi = psi
            bin_entry.freq = 0.1

            rot = rotlib_pb2.Rotamer()
            rot.prob = 0.5
            chi_val = rotlib_pb2.ChiValue()
            chi_val.val = 120.0
            chi_val.sigma = 1.0
            rot.chi.append(chi_val)

            bin_entry.rotamers.append(rot)
            res.bins.append(bin_entry)

    lib.residues.append(res)

    # Filter
    filtered = filter_grid_36(lib)
    filtered_res = filtered.residues[0]

    # Checks
    expected_phi = [-180.0, 170.0]
    expected_psi = [-180.0]
    expected_bins = [
        (-180.0, -180.0),
        (170.0, -180.0),
    ]

    assert list(filtered_res.phi_centers) == expected_phi, \
        f"phi_centers: expected {expected_phi}, got {list(filtered_res.phi_centers)}"
    assert list(filtered_res.psi_centers) == expected_psi, \
        f"psi_centers: expected {expected_psi}, got {list(filtered_res.psi_centers)}"
    assert len(filtered_res.bins) == 2, \
        f"bin count: expected 2, got {len(filtered_res.bins)}"

    for i, (phi, psi) in enumerate(expected_bins):
        assert filtered_res.bins[i].phi == phi and filtered_res.bins[i].psi == psi, \
            f"bin {i}: expected ({phi}, {psi}), got ({filtered_res.bins[i].phi}, {filtered_res.bins[i].psi})"

    logger.info("Self-test passed")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-pb", type=Path,
                    default=Path("/home/marielle/projects/proxide/data/rotlibs/proxide-rotlib-bbdep2010.pb.zst"),
                    help="input proxide rotlib .pb.zst")
    ap.add_argument("--rotlib-bin", type=Path,
                    default=Path("/home/marielle/repos/mosaist/testfiles/rotlib.bin"),
                    help="MASTER rotlib.bin for identity verification")
    ap.add_argument("--out-pb", type=Path,
                    help="output path for filtered .pb.zst (default: $CLAUDE_JOB_DIR/tmp/pb36.pb.zst)")
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                    help="logging level")
    ap.add_argument("--dry-run", action="store_true",
                    help="do not write output file")
    ap.add_argument("--self-test", action="store_true",
                    help="run self-test and exit")

    args = ap.parse_args(argv)

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # Self-test
    if args.self_test:
        try:
            # Compile proto first
            proto_dir = Path(__file__).parent.parent.parent / "crates" / "proxide-rotlib" / "proto"
            with tempfile.TemporaryDirectory() as tmpdir:
                proto_py_path = compile_proto_runtime(proto_dir, Path(tmpdir))
                rotlib_pb2 = import_rotlib_pb2(proto_py_path)
                return self_test()
        except Exception as e:
            logger.error(f"Self-test failed: {e}", exc_info=True)
            return 1

    # Default output path
    if args.out_pb is None:
        job_dir = os.environ.get("CLAUDE_JOB_DIR", "/tmp")
        out_dir = Path(job_dir) / "tmp"
        out_dir.mkdir(parents=True, exist_ok=True)
        args.out_pb = out_dir / "pb36.pb.zst"

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
            # Load input
            raw_pb = load_pb(args.in_pb)
            lib = rotlib_pb2.RotamerLibrary()
            lib.ParseFromString(raw_pb)

            # Count before
            in_residues = len(lib.residues)
            in_bins_total = sum(len(r.bins) for r in lib.residues)
            in_phi_centers = {r.code: len(r.phi_centers) for r in lib.residues}
            in_psi_centers = {r.code: len(r.psi_centers) for r in lib.residues}
            logger.info(f"Loaded {in_residues} residues, {in_bins_total} total bins")

            # Filter
            logger.info("Filtering to 36x36 grid...")
            filtered_lib = filter_grid_36(lib)

            # Count after
            out_residues = len(filtered_lib.residues)
            out_bins_total = sum(len(r.bins) for r in filtered_lib.residues)
            out_phi_centers = {r.code: len(r.phi_centers) for r in filtered_lib.residues}
            out_psi_centers = {r.code: len(r.psi_centers) for r in filtered_lib.residues}
            logger.info(f"Filtered to {out_residues} residues, {out_bins_total} total bins")

            # Verify identity
            logger.info("Verifying identity with rotlib.bin...")
            verify_result = verify_identity_with_master(filtered_lib, args.rotlib_bin)
            logger.info(f"Shared bins: {verify_result['total_shared_bins']}, "
                        f"identical: {verify_result['n_bins_identical']}, "
                        f"mismatches: {verify_result['n_bins_mismatch']}")

            # Serialize and compress
            logger.info("Encoding filtered library...")
            serialized = filtered_lib.SerializeToString()
            cctx = zstandard.ZstdCompressor()
            compressed = cctx.compress(serialized)
            logger.info(f"Serialized {len(serialized)} bytes, compressed to {len(compressed)} bytes")

            # Compute SHA256
            import hashlib
            out_sha256 = hashlib.sha256(compressed).hexdigest()

            # Write (unless dry-run)
            if not args.dry_run:
                logger.info(f"Writing to {args.out_pb}")
                args.out_pb.parent.mkdir(parents=True, exist_ok=True)
                with open(args.out_pb, "wb") as f:
                    f.write(compressed)
            else:
                logger.info("Dry-run: skipping write")

            # Report
            report = {
                "in_grid": f"37x37 (before filtering)",
                "out_grid": f"36x36 (after filtering)",
                "residues": out_residues,
                "total_shared_bins": verify_result["total_shared_bins"],
                "n_bins_identical": verify_result["n_bins_identical"],
                "n_bins_mismatch": verify_result["n_bins_mismatch"],
                "mismatches": verify_result["mismatches"],
                "out_path": str(args.out_pb),
                "out_sha256": out_sha256,
            }

            # JSON to stdout and log
            import sys
            report_json = json.dumps(report, indent=2)
            print(report_json)
            logger.info(f"Report: {report_json}")

            return 0

        except Exception as e:
            logger.error(f"Processing failed: {e}", exc_info=True)
            return 1


if __name__ == "__main__":
    sys.exit(main())
