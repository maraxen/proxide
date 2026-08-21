"""Fetch AmberTools/antechamber-derived assets that must NOT be vendored in git.

`src/proxide/assets/gaff/dat/ATOMTYPE_GFF2.DEF` is the GAFF2 atom-type rule
grammar antechamber ships. Every other file in that directory comes from
`openmmforcefields` (see `scripts/sync_forcefields.py`'s `FILE_MAPPINGS`,
which only globs `*.dat`/`*.xml`) -- ATOMTYPE_GFF2.DEF does not, and never
did; it is antechamber's own file, sourced directly from
`Amber-MD/AmberClassic` (`dat/antechamber/ATOMTYPE_GFF2.DEF`), which is
GPL-licensed. Unlike the openmmforcefields-sourced assets, this file is
deliberately NOT committed to this (MIT-licensed) repository's git history
-- fetch it here instead, at dev-setup/CI time, verified against a pinned
content digest so a silent upstream change (or a corrupted fetch) is
caught immediately rather than silently shipping a different rule set.

See `.praxia/docs/reference/260821_gaff2-rust-port-lessons.md` (Open Item
#3) for the full rationale and the licensing research this pin is based on.

This mirrors `scripts/validation/gaff2_geostd_sample.py`'s fetch
conventions (pinned ref, User-Agent header, optional GITHUB_TOKEN/GH_TOKEN
auth to avoid the unauthenticated 60 req/hr GitHub rate limit).

Usage:
    uv run python scripts/fetch_amber_assets.py
    uv run python scripts/fetch_amber_assets.py --check   # verify only, no fetch/write
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("fetch_amber_assets")

_REPO = "Amber-MD/AmberClassic"
_DEFAULT_REF = "20e92d4b44e84cc0ca84bdf7f640eba0c1d1f2ed"  # main, captured 2026-08-21

# Each entry: where the file lives upstream, where it lands locally, and the
# sha256 it must match -- verified 2026-08-21 by fetching the real upstream
# file and diffing it byte-for-byte against this repo's then-vendored copy
# (they were identical), so this digest is a real confirmed pin, not a
# guess.
ASSETS = [
    {
        "upstream_path": "dat/antechamber/ATOMTYPE_GFF2.DEF",
        "dest": Path("src/proxide/assets/gaff/dat/ATOMTYPE_GFF2.DEF"),
        "sha256": "7a076ac2e667ab87057befc7a5985be4cead83e01ff5d2d3dab9f1d65bff637e",
    },
]


def _fetch(url: str) -> bytes:
    headers = {"User-Agent": "Mozilla/5.0"}
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response:
        return response.read()


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ref", default=_DEFAULT_REF, help="Amber-MD/AmberClassic commit SHA to fetch from"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify local files' digests only -- don't fetch or write anything "
        "(for a fast CI gate that confirms the fetch step actually ran)",
    )
    args = parser.parse_args()

    ok = True
    for asset in ASSETS:
        dest: Path = asset["dest"]

        if args.check:
            if not dest.exists():
                logger.error("MISSING: %s -- run without --check to fetch it", dest)
                ok = False
                continue
            actual = _digest(dest.read_bytes())
            if actual != asset["sha256"]:
                logger.error(
                    "DIGEST MISMATCH: %s -- expected %s, got %s",
                    dest, asset["sha256"], actual,
                )
                ok = False
            else:
                logger.info("OK: %s", dest)
            continue

        if dest.exists() and _digest(dest.read_bytes()) == asset["sha256"]:
            logger.info("already present and verified: %s", dest)
            continue

        url = f"https://raw.githubusercontent.com/{_REPO}/{args.ref}/{asset['upstream_path']}"
        logger.info("fetching %s -> %s", url, dest)
        data = _fetch(url)
        actual = _digest(data)
        if actual != asset["sha256"]:
            logger.error(
                "DIGEST MISMATCH after fetch: %s -- expected %s, got %s. This means "
                "AmberTools/antechamber's upstream file changed, or the fetch was "
                "corrupted. If this is a deliberate upstream update: update the pinned "
                "digest here AND in crates/proxide-gaff2/src/rules_loader.rs's "
                "embedded_default_def_content_digest_is_pinned test, then re-run the "
                "full geostd parity campaign (scripts/validation/gaff2_rust_parity.py "
                "--full) before merging.",
                dest, asset["sha256"], actual,
            )
            ok = False
            continue

        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        logger.info("wrote %s (%d bytes, digest verified)", dest, len(data))

    if not ok:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
