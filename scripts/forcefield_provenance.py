#!/usr/bin/env python3
"""Match every vendored force-field XML to a byte-identical file upstream.

For each ``*.xml`` under ``src/proxide/assets`` this computes the git blob id
(``sha1(b"blob <len>\\0" + data)``) and looks it up in the recursive tree of
each upstream repository at a pinned commit, fetched through ``gh api``. A
blob-id match means the vendored file is byte-identical to that upstream path
at that commit. With ``--history``, an asset that matches nothing at the pin
is compared against past versions of same-named upstream files. Anything still
unmatched is reported as ``UNMATCHED`` and never guessed at.

With ``--emit-tsv`` it writes ``src/proxide/assets/FORCEFIELD-LICENCES.tsv``
(spec 261102, track T2): one ``# source:`` comment and one
``path<TAB>kind<TAB>target`` row per XML. The licence target is the pinned
upstream licence text (``sha256``), except for XMLs converted from
MacKerell-lab toppar files, which point (``file``) at the repo-authored
CHARMM notice. Emitting refuses while any asset is unmatched.

Re-run after every asset sync, and review the diff of the TSV.

Usage:
    uv run python scripts/forcefield_provenance.py --history [--json OUT] [--emit-tsv]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import pathlib
import re
import subprocess
import sys

log = logging.getLogger("forcefield_provenance")

ASSETS = pathlib.Path("src/proxide/assets")
TSV = ASSETS / "FORCEFIELD-LICENCES.tsv"

# Pinned upstream commits (2026-11-24). Change these deliberately, never to a
# branch name: an unpinned ref makes the evidence drift silently.
UPSTREAMS = {
  "openmm/openmmforcefields": "0edd3bb8111b400db01f0a9666091bf8a9247baf",
  "openmm/openmm": "b5282630278e6c78e92691ed411543f73aabba75",
}

# Upstream files that no longer exist at the pinned commit, so a same-name
# search of the pinned tree cannot find them. Asset basename -> [(repo, path)].
DELETED_UPSTREAM = {
  # Removed by openmm/openmm#5143 (2025-11-18) as an incomplete, abandoned force field.
  "protein.ff19ipq.xml": [
    ("openmm/openmm", "wrappers/python/openmm/app/data/amber19/protein.ff19ipq.xml"),
  ],
}

# Verbatim licence text of each upstream at its pinned commit, checked in.
LICENCE_TEXTS = {
  "openmm/openmmforcefields": "src/proxide/assets/licences/openmmforcefields-LICENSE",
  "openmm/openmm": "src/proxide/assets/licences/openmm-Licenses.txt",
}
# Repo-authored notice for XMLs converted from MacKerell-lab toppar files,
# which carry no stated licence upstream (spec 261102, decision D5a).
CHARMM_NOTICE = "src/proxide/assets/licences/CHARMM-TOPPAR-NOTICE.md"
CHARMM_SOURCE = re.compile(r'sourcePackage="(charmm|drude)|<Source[^>]*>[^<]*toppar')
CHARMM_INCLUDE = re.compile(r'<Include\s+file="([^"]+)"')
# Directories vendored from OpenMM's own app/data; everything else is synced
# from openmmforcefields (scripts/sync_forcefields.py).
OPENMM_DIRS = ("openmm_bundled/", "implicit/")


def git_blob_id(data: bytes) -> str:
  """Return the git blob id of ``data``, as ``git hash-object`` would."""
  return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def gh_json(endpoint: str) -> object:
  out = subprocess.run(["gh", "api", endpoint], check=True, capture_output=True, text=True).stdout
  return json.loads(out)


def upstream_blobs(repo: str, commit: str) -> dict[str, list[str]]:
  """Map blob id -> upstream paths for every ``*.xml`` in ``repo`` at ``commit``."""
  tree = gh_json(f"repos/{repo}/git/trees/{commit}?recursive=1")
  if tree.get("truncated"):
    # A truncated listing would turn real matches into UNMATCHED silently.
    raise SystemExit(f"tree listing for {repo}@{commit} is truncated")
  blobs: dict[str, list[str]] = {}
  for entry in tree["tree"]:
    if entry["type"] == "blob" and entry["path"].endswith(".xml"):
      blobs.setdefault(entry["sha"], []).append(entry["path"])
  log.info("%s@%s: %d xml blobs", repo, commit[:7], len(blobs))
  return blobs


def history_matches(
  blob: str, basename: str, indexes: dict[str, dict[str, list[str]]]
) -> list[dict]:
  """Search past versions of same-named upstream files for ``blob``.

  Candidates are the pinned-tree paths whose basename equals the asset's, plus
  any ``DELETED_UPSTREAM`` entries. For each, walk the commits that touched it
  (up to 100, newest first, ending at the pin) and compare the blob id at each.
  """
  candidates = list(DELETED_UPSTREAM.get(basename, []))
  for repo, index in indexes.items():
    paths = sorted({p for ps in index.values() for p in ps if p.rsplit("/", 1)[-1] == basename})
    candidates += [(repo, p) for p in paths]
  found = []
  for repo, path in candidates:
    commits = gh_json(f"repos/{repo}/commits?path={path}&sha={UPSTREAMS[repo]}&per_page=100")
    for c in commits:
      try:
        meta = gh_json(f"repos/{repo}/contents/{path}?ref={c['sha']}")
      except subprocess.CalledProcessError:
        # The commit that deleted the file has no contents at that path.
        log.info("  %s:%s absent at %s", repo, path, c["sha"][:7])
        continue
      if meta.get("sha") == blob:
        found.append(
          {"repo": repo, "commit": c["sha"], "path": path, "date": c["commit"]["committer"]["date"]}
        )
        break
  return found


def pick_source(rel: str, matches: list[dict]) -> dict:
  """Choose the upstream the asset was vendored from, among byte-identical matches.

  Prefer the expected repo for the asset's directory, then an upstream path with
  the asset's own filename: identical siblings (amber96_obc.xml and
  amber99_obc.xml share one blob) must not be recorded under each other's name.
  """
  want = "openmm/openmm" if rel.startswith(OPENMM_DIRS) else "openmm/openmmforcefields"
  ours = rel.split("/")[::-1]

  def shared_tail(path: str) -> int:
    """Number of trailing path components shared with the asset's path."""
    n = 0
    for a, b in zip(ours, path.split("/")[::-1], strict=False):
      if a != b:
        break
      n += 1
    return n

  return min(matches, key=lambda m: (m["repo"] != want, -shared_tail(m["path"])))


def is_charmm_derived(rel: str, path: pathlib.Path) -> bool:
  """True for XMLs built from MacKerell-lab toppar data.

  Signals, any of which suffices: a <Source> naming a CHARMM or Drude toppar
  file; a DrudeForce (the CHARMM Drude polarizable model, including the
  SWM4-NDP water, whose OpenMM XML has no <Source>); or a wrapper whose
  <Include>s all resolve to CHARMM-derived files.
  """
  text = path.read_text(encoding="utf-8", errors="replace")
  if CHARMM_SOURCE.search(text[:20000]) or "<DrudeForce" in text:
    return True
  includes = CHARMM_INCLUDE.findall(text)
  if includes and not re.search(r"<(AtomTypes|Residues|\w+Force)\b", text):
    targets = [path.parent / inc for inc in includes]
    return all(t.is_file() and is_charmm_derived(rel, t) for t in targets)
  return False


def sha256_of(path: str) -> str:
  return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def emit_tsv(rows: list[dict]) -> str:
  lines = [
    "# Source and licence record for every vendored force-field XML under src/proxide/assets.",
    "# Spec 261102, track T2. Generated by scripts/forcefield_provenance.py --history --emit-tsv;",
    "# do not edit rows by hand. Re-run after every asset sync and review the diff.",
    "#",
    "# Columns (tab-separated): path relative to src/proxide/assets; kind; target.",
    "#   sha256 DIGEST PATH  upstream licence text, checked in verbatim at the pinned commit",
    "#   file PATH           repo-authored notice (CHARMM toppar-derived XMLs)",
    "# Each '# source:' line names the upstream file the next row's XML is byte-identical",
    "# to (git blob id match).",
    "#",
    "# Pinned upstreams:",
    *(f"#   {repo}@{sha}" for repo, sha in UPSTREAMS.items()),
    "# Licence texts are each upstream's at its pinned commit. A '# source:' line with an",
    "# older commit means the XML matches an earlier version of that upstream file; for",
    "# OpenMM, Licenses.txt section 1 (MIT) is unchanged between 654c6c9 (2021) and the",
    "# pin: only the preamble, copyright years and an added VkFFT section differ.",
    "#",
    "# Original-source caveats not resolved here; each row records the licence of the",
    "# upstream that redistributes the file:",
    "#   - CHARMM36, CGenFF, Drude and CHARMM water/ion XMLs: no licence is stated for the",
    "#     MacKerell toppar files; see licences/CHARMM-TOPPAR-NOTICE.md and debt #3258.",
    "#   - amoeba*.xml and iamoeba.xml: AMOEBA parameters from Tinker; terms not verified.",
    "#   - GLYCAM_06j-1.xml: Woods lab (glycam.org) parameters; terms not verified.",
    "#   - openmm_bundled/amber19/protein.ff19ipq.xml: removed upstream as abandoned",
    "#     (openmm/openmm#5143).",
    "",
  ]
  for r in rows:
    src = r["source"]
    when = f"@{src['commit'][:12]}"
    lines.append(f"# source: {src['repo']}:{src['path']}{when}")
    if r["charmm"]:
      lines.append(f"{r['asset']}\tfile\t{CHARMM_NOTICE}")
    else:
      text = LICENCE_TEXTS[src["repo"]]
      lines.append(f"{r['asset']}\tsha256\t{sha256_of(text)} {text}")
  return "\n".join(lines) + "\n"


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument("--json", type=pathlib.Path, help="write the full result here")
  ap.add_argument(
    "--history",
    action="store_true",
    help="for unmatched assets, search past versions of same-named upstream files",
  )
  ap.add_argument("--emit-tsv", action="store_true", help=f"write {TSV}")
  args = ap.parse_args()
  logging.basicConfig(level=logging.INFO, format="%(message)s")

  indexes = {repo: upstream_blobs(repo, sha) for repo, sha in UPSTREAMS.items()}
  rows = []
  for path in sorted(ASSETS.rglob("*.xml")):
    rel = path.relative_to(ASSETS).as_posix()
    blob = git_blob_id(path.read_bytes())
    matches = [
      {"repo": repo, "commit": UPSTREAMS[repo], "path": p}
      for repo, index in indexes.items()
      for p in index.get(blob, [])
    ]
    if not matches and args.history:
      matches = history_matches(blob, path.name, indexes)
    rows.append(
      {
        "asset": rel,
        "blob": blob,
        "matches": matches,
        "source": pick_source(rel, matches) if matches else None,
        "charmm": is_charmm_derived(rel, path),
      }
    )

  unmatched = [r["asset"] for r in rows if not r["matches"]]
  for r in rows:
    where = (
      "; ".join(
        f"{m['repo']}:{m['path']}"
        + (f"@{m['commit'][:7]} ({m['date'][:10]})" if "date" in m else "")
        for m in r["matches"]
      )
      or "UNMATCHED"
    )
    print(f"{r['asset']}\t{'CHARMM' if r['charmm'] else '-'}\t{where}")
  log.info(
    "%d assets, %d matched, %d unmatched", len(rows), len(rows) - len(unmatched), len(unmatched)
  )
  if args.json:
    args.json.write_text(json.dumps({"upstreams": UPSTREAMS, "rows": rows}, indent=2) + "\n")
  if args.emit_tsv:
    if unmatched:
      log.error("refusing to emit %s with unmatched assets: %s", TSV, ", ".join(unmatched))
      return 1
    TSV.write_text(emit_tsv(rows))
    log.info("wrote %s", TSV)
  return 1 if unmatched else 0


if __name__ == "__main__":
  sys.exit(main())
