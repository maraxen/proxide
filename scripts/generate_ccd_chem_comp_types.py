#!/usr/bin/env python3
"""Generate the bundled CCD id->chem_comp.type asset used by
crates/proxide-core/src/chem/ccd.rs.

Downloads the official PDB Chemical Component Dictionary
(https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz, ~116MB
gzipped) and extracts ONLY the two scalar tags needed for residue
classification -- _chem_comp.id and _chem_comp.type -- for every component
block, discarding the (much larger) per-component atom/bond geometry that
proxide-core does not need here (it already has its own heavy-atom templates
for the residues it builds atom37 representations for).

Output is a zstd-compressed protobuf message (proto/ccd_chem_comp_types.proto,
ChemCompTypeTable), matching the existing proxide-rotlib .pb.zst convention
(crates/proxide-rotlib, e.g. data/rotlibs/proxide-rotlib-dunbrack2010-ccd.pb.zst).
This is a one-time/occasional data-refresh script, not run at build time --
the output is checked into the repo like the rotlib asset.

Usage:
    uv run python3 scripts/generate_ccd_chem_comp_types.py \
        --out crates/proxide-core/data/ccd_chem_comp_types.pb.zst
"""

from __future__ import annotations

import argparse
import gzip
import logging
import subprocess
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("generate_ccd_chem_comp_types")

CCD_URL = "https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz"
REPO_ROOT = Path(__file__).resolve().parent.parent
PROTO_PATH = REPO_ROOT / "crates" / "proxide-core" / "proto" / "ccd_chem_comp_types.proto"


def _compile_proto(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "protoc",
            f"--python_out={out_dir}",
            f"--proto_path={PROTO_PATH.parent}",
            str(PROTO_PATH),
        ],
        check=True,
    )


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        return value[1:-1]
    if len(value) >= 2 and value[0] == "'" and value[-1] == "'":
        return value[1:-1]
    return value


def parse_chem_comp_id_and_type(cif_lines) -> dict[str, str]:
    """Stream (id -> chem_comp_type) pairs out of a components.cif iterator.

    Each component is a `data_XXX` block. Within it, `_chem_comp.id` and
    `_chem_comp.type` appear as simple `tag value` lines before the first
    `loop_` (which starts the much larger per-atom/per-bond records we don't
    need). A block missing either tag, or with an unresolved (`?`/`.`) type,
    is skipped and counted -- logged loudly rather than silently dropped,
    since a parsing gap here would silently corrupt downstream classification.
    """
    entries: dict[str, str] = {}
    current_id: str | None = None
    current_type: str | None = None
    in_scalar_section = False
    n_blocks = 0
    n_skipped_incomplete = 0

    def _flush():
        nonlocal current_id, current_type
        if current_id is not None and current_type is not None and current_type not in ("?", "."):
            if current_id in entries and entries[current_id] != current_type:
                log.warning(
                    f"Duplicate CCD id {current_id!r} with conflicting types "
                    f"({entries[current_id]!r} vs {current_type!r}) -- keeping first seen.",
                )
            else:
                entries.setdefault(current_id, current_type)
        elif current_id is not None:
            nonlocal n_skipped_incomplete
            n_skipped_incomplete += 1
        current_id = None
        current_type = None

    for raw_line in cif_lines:
        line = raw_line.rstrip("\n")
        if line.startswith("data_"):
            _flush()
            n_blocks += 1
            in_scalar_section = True
            continue
        if line.startswith("loop_"):
            in_scalar_section = False
            continue
        if not in_scalar_section:
            continue
        if line.startswith("_chem_comp.id"):
            current_id = _strip_quotes(line[len("_chem_comp.id") :])
        elif line.startswith("_chem_comp.type"):
            current_type = _strip_quotes(line[len("_chem_comp.type") :])
    _flush()

    log.info(f"Parsed {n_blocks} CCD component blocks -> {len(entries)} usable (id, type) pairs")
    if n_skipped_incomplete:
        log.warning(f"Skipped {n_skipped_incomplete} blocks with a missing/unresolved id or type")
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "crates" / "proxide-core" / "data" / "ccd_chem_comp_types.pb.zst",
    )
    parser.add_argument(
        "--source-cif-gz",
        type=Path,
        default=None,
        help="Use an already-downloaded components.cif.gz instead of fetching from wwPDB.",
    )
    args = parser.parse_args()

    import tempfile

    if args.source_cif_gz and args.source_cif_gz.exists():
        log.info(f"Using local {args.source_cif_gz}")
        gz_path = args.source_cif_gz
    else:
        gz_path = Path(tempfile.mkstemp(suffix=".cif.gz")[1])
        log.info(f"Downloading {CCD_URL} -> {gz_path}")
        urllib.request.urlretrieve(CCD_URL, gz_path)  # noqa: S310

    with gzip.open(gz_path, "rt", encoding="utf-8", errors="replace") as f:
        entries = parse_chem_comp_id_and_type(f)

    if len(entries) < 40_000:
        log.error(
            f"Only extracted {len(entries)} entries -- expected ~45k+. "
            "Treating this as a parsing bug, not accepting a truncated dataset.",
        )
        sys.exit(1)

    pyproto_dir = Path(tempfile.mkdtemp())
    _compile_proto(pyproto_dir)
    sys.path.insert(0, str(pyproto_dir))
    import ccd_chem_comp_types_pb2 as pb  # noqa: E402

    table = pb.ChemCompTypeTable()
    table.ccd_version = f"wwpdb-components.cif.gz fetched {datetime.now(UTC).date().isoformat()}"
    for comp_id, comp_type in sorted(entries.items()):
        entry = table.entries.add()
        entry.id = comp_id
        entry.chem_comp_type = comp_type

    serialized = table.SerializeToString()
    log.info(f"Serialized protobuf: {len(serialized) / 1024:.1f} KiB uncompressed")

    import zstandard

    compressed = zstandard.ZstdCompressor(level=19).compress(serialized)
    log.info(f"Compressed (zstd level 19): {len(compressed) / 1024:.1f} KiB")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(compressed)
    log.info(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
