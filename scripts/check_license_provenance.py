#!/usr/bin/env python3
"""Tier-1 license-provenance lint (see .praxia/docs/specs/260824_license-provenance-verification.md).

Deterministic, recall-oriented tripwire: scans crate manifests, crate doc
comments, and internal spec/decision docs for signals that a crate reimplements
or ports external code, then checks whether that signal has a corresponding
resolution (a crate-local NOTICE/LICENSE file, or an `append_audit` record in
`.praxia/audits.jsonl`). It does not judge whether a flagged crate is actually
a problem -- that's the tier-2 agentic pass (see the
`license-provenance-check` skill), invoked manually by a developer.

Usage:
    uv run scripts/check_license_provenance.py
    uv run scripts/check_license_provenance.py --crate proxide-confind
    uv run scripts/check_license_provenance.py --json

Exit status: 0 if every signal found has a NOTICE/LICENSE or a matching audit
record; 1 if any flagged crate is unresolved (for use as an optional local
pre-push check -- this is not wired into CI, see the spec's "Open decisions").
"""

import argparse
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
CRATES_DIR = REPO_ROOT / "crates"
SPECS_DIRS = [
  REPO_ROOT / ".praxia" / "docs" / "specs",
  REPO_ROOT / ".praxia" / "docs" / "decisions",
]
AUDITS_JSONL = REPO_ROOT / ".praxia" / "audits.jsonl"

# Deliberately broad -- false positives are cheap (one tier-2 dispatch),
# false negatives are the actual risk (see spec's "why a lint alone doesn't
# fix this").
PHRASE_SIGNALS = re.compile(
  r"\b(port(?:ed)?\s+of|based\s+on|derived\s+from|reimplement(?:s|ation\s+of)?"
  r"|ported\s+from|matches\s+\S+\s+exactly|source\s+reference)\b",
  re.IGNORECASE,
)
REPO_URL_SIGNAL = re.compile(r"https?://(?:github|gitlab)\.com/[\w.\-]+/[\w.\-]+")
PAPER_SIGNAL = re.compile(r"\b10\.\d{4,9}/\S+|arXiv:\d{4}\.\d{4,5}\b")

# A tier-2 verdict record must carry this to be recognized -- see the
# `license-provenance-check` skill. Without a marker, matching audits.jsonl
# by substring-contains-crate-name is a real false-positive source: this repo's
# audit log is a general project log, and plenty of unrelated entries (a code
# review, a bugfix audit) mention a crate name in passing with no license
# verdict attached. Caught by manual testing of this script against the live
# repo -- exactly the flakiness this two-tier design exists to avoid.
AUDIT_MARKER = "license_provenance_crate"


@dataclass
class CrateFinding:
  crate: str
  signals: list[str] = field(default_factory=list)
  has_notice: bool = False
  license_override: str | None = None
  audited: bool = False

  @property
  def flagged(self) -> bool:
    return bool(self.signals)

  @property
  def resolved(self) -> bool:
    return not self.flagged or self.has_notice or self.audited


def _find_signals(text: str) -> list[str]:
  hits = []
  if m := PHRASE_SIGNALS.search(text):
    hits.append(f'phrase:"{m.group(0)}"')
  if m := REPO_URL_SIGNAL.search(text):
    hits.append(f"repo-url:{m.group(0)}")
  if m := PAPER_SIGNAL.search(text):
    hits.append(f"citation:{m.group(0)}")
  return hits


def _audited_crates() -> set[str]:
  """Crates with a tier-2 verdict already recorded.

  Only counts records carrying the explicit `license_provenance_crate`
  marker (written by the `license-provenance-check` skill) -- matching by
  "does the crate name appear anywhere in this audit record" is exactly the
  kind of loose substring match that produces false "resolved" readings in a
  project-wide audit log full of unrelated entries that happen to mention a
  crate name in passing.
  """
  if not AUDITS_JSONL.exists():
    return set()
  audited = set()
  for line in AUDITS_JSONL.read_text().splitlines():
    line = line.strip()
    if not line:
      continue
    try:
      record = json.loads(line)
    except json.JSONDecodeError:
      continue
    payload = record.get("payload", record)
    crate = payload.get(AUDIT_MARKER)
    if crate:
      audited.add(crate)
  return audited


def scan_crate(crate_dir: Path, audited: set[str]) -> CrateFinding:
  finding = CrateFinding(crate=crate_dir.name)

  manifest = crate_dir / "Cargo.toml"
  if manifest.exists():
    manifest_text = manifest.read_text()
    finding.signals += _find_signals(manifest_text)
    if m := re.search(r'^license\s*=\s*"([^"]+)"', manifest_text, re.MULTILINE):
      finding.license_override = m.group(1)

  for lib_path in (crate_dir / "src" / "lib.rs", crate_dir / "src" / "main.rs"):
    if lib_path.exists():
      doc_lines = [
        line
        for line in lib_path.read_text().splitlines()[:80]
        if line.lstrip().startswith("//!")
      ]
      finding.signals += _find_signals("\n".join(doc_lines))

  # (?![\w-]) so "proxide-gaff" doesn't match inside a "proxide-gaff2"
  # mention -- a real false positive hit while testing this against the live
  # repo (this doc's own appendix discusses both crates in adjacent text).
  crate_token = re.compile(re.escape(crate_dir.name) + r"(?![\w-])", re.IGNORECASE)
  for specs_dir in SPECS_DIRS:
    if not specs_dir.exists():
      continue
    for doc in specs_dir.glob("*.md"):
      # Paragraph-scoped, not whole-document: a large cross-cutting doc
      # (e.g. this very spec's own appendix, which names every crate in a
      # results table) mentions most crate names somewhere, and whole-doc
      # matching would then attribute that doc's own vocabulary ("port of",
      # "reimplementation", ...) to every crate it lists, not just the ones
      # it's actually making a provenance claim about.
      for paragraph in doc.read_text().split("\n\n"):
        if crate_token.search(paragraph):
          finding.signals += _find_signals(paragraph)

  finding.signals = sorted(set(finding.signals))
  finding.has_notice = (crate_dir / "NOTICE").exists() or (crate_dir / "LICENSE").exists()
  finding.audited = crate_dir.name in audited
  return finding


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--crate", help="only check this crate (directory name under crates/)")
  parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
  args = parser.parse_args()

  if not CRATES_DIR.exists():
    logger.error("No crates/ directory found at %s", CRATES_DIR)
    return 1

  audited = _audited_crates()
  crate_dirs = sorted(d for d in CRATES_DIR.iterdir() if d.is_dir())
  if args.crate:
    crate_dirs = [d for d in crate_dirs if d.name == args.crate]
    if not crate_dirs:
      logger.error("No crate named %s under %s", args.crate, CRATES_DIR)
      return 1

  findings = [scan_crate(d, audited) for d in crate_dirs]

  if args.json:
    print(json.dumps([f.__dict__ for f in findings], indent=2))
  else:
    flagged = [f for f in findings if f.flagged]
    if not flagged:
      logger.info("No provenance signals found across %d crates.", len(findings))
    for f in flagged:
      status = "OK (resolved)" if f.resolved else "UNRESOLVED"
      logger.info("%-24s %-12s signals=%s", f.crate, status, f.signals)

  unresolved = [f for f in findings if f.flagged and not f.resolved]
  if unresolved:
    logger.warning(
      "\n%d crate(s) flagged with no NOTICE/LICENSE or audit record: %s\n"
      "Run the license-provenance-check skill (tier 2) on each before merging.",
      len(unresolved),
      ", ".join(f.crate for f in unresolved),
    )
    return 1
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
