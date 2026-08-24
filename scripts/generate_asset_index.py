#!/usr/bin/env python3
"""Regenerate src/proxide/assets/_asset_index.py from a local asset tree.

Only needed if the upstream pinned commits in `proxide.assets._fetch` are
bumped and the file catalog changes (new/renamed/removed files). Requires
the assets subdirectories to exist locally -- run scripts/sync_forcefields.py
first (or point --assets-dir at an old checkout / a manual download of the
pinned commits) to populate them before regenerating.

Usage:
    uv run scripts/generate_asset_index.py [--assets-dir PATH]
"""

import argparse
from pathlib import Path

SUBDIRS = ("amber", "water", "implicit", "openmm_bundled")


def build_index(assets_dir: Path) -> dict[str, str]:
  index: dict[str, str] = {}
  for sub in SUBDIRS:
    d = assets_dir / sub
    if not d.exists():
      continue
    for xml in sorted(d.rglob("*.xml")):
      rel = xml.relative_to(assets_dir).as_posix()
      index.setdefault(xml.stem, rel)
  return index


def render(index: dict[str, str]) -> str:
  lines = [
    '"""Static name -> relative-path index for fetchable/restricted force field assets.',
    "",
    "Generated from the asset tree as it existed before amber/water/implicit/openmm_bundled",
    "were converted from git-vendored to fetch-on-demand (see",
    ".praxia/docs/specs/260824_license-provenance-verification.md). This lets",
    'load_force_field("protein.ff14SB")-style bare-name lookups resolve to the right',
    "relative path without needing the files present on disk (rglob can't find what isn't",
    "vendored anymore) -- regenerate with scripts/generate_asset_index.py if the upstream",
    "pinned commits in _fetch.py are ever bumped and the file catalog changes.",
    '"""',
    "",
    "ASSET_INDEX: dict[str, str] = {",
  ]
  for k in sorted(index):
    lines.append(f"  {k!r}: {index[k]!r},")
  lines.append("}")
  lines.append("")
  return "\n".join(lines)


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--assets-dir",
    type=Path,
    default=Path(__file__).resolve().parent.parent / "src" / "proxide" / "assets",
  )
  args = parser.parse_args()

  index = build_index(args.assets_dir)
  if not index:
    subdirs = ",".join(SUBDIRS)
    print(f"No XML files found under {args.assets_dir}/{{{subdirs}}} -- nothing to index.")
    return 1

  out_path = args.assets_dir / "_asset_index.py"
  out_path.write_text(render(index))
  print(f"Wrote {len(index)} entries to {out_path}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
