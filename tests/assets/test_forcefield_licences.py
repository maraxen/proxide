"""Acceptance for spec 261102 track T2: every vendored force-field XML has a licence record.

``ACCEPT`` is the program from the spec, verbatim. It exits 0 only when every
``*.xml`` under ``src/proxide/assets`` has a row in ``FORCEFIELD-LICENCES.tsv``
and every row's licence target exists (``file``) or matches its recorded
sha256 (``sha256``). The red controls below break one thing each in a mirror
of the tree and require a non-zero exit, so a check that always passes would
fail here.
"""

from __future__ import annotations

import pathlib
import re
import shutil
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
ASSETS_REL = pathlib.Path("src/proxide/assets")
TSV_NAME = "FORCEFIELD-LICENCES.tsv"

# Verbatim from .praxia/docs/specs/261102_proxide-stable-hardening-and-cisternal.md, T2.
ACCEPT = r'''
import hashlib, pathlib, sys
root = pathlib.Path("src/proxide/assets")
xmls = {p.relative_to(root).as_posix() for p in root.rglob("*.xml")}
rec = root / "FORCEFIELD-LICENCES.tsv"
rows = {}
if rec.is_file():
    for line in rec.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        path, kind, target = line.split("\t")
        rows[path] = (kind, target)
missing = sorted(xmls - rows.keys())
bad = []
for path, (kind, target) in rows.items():
    if path not in xmls:
        bad.append(path)
    elif kind == "file":
        if not pathlib.Path(target).is_file():
            bad.append(path)
    elif kind == "sha256":
        digest, _, text = target.partition(" ")
        blob = pathlib.Path(text)
        if not blob.is_file() or hashlib.sha256(blob.read_bytes()).hexdigest() != digest:
            bad.append(path)
    else:
        bad.append(path)
sys.exit(1 if missing or bad else 0)
'''

# Mirrors scripts/forcefield_provenance.py: an XML converted from MacKerell
# toppar files must point at the CHARMM notice, never at an MIT text.
CHARMM_SOURCE = re.compile(r'sourcePackage="(charmm|drude)|<Source[^>]*>[^<]*toppar')
CHARMM_NOTICE = "src/proxide/assets/licences/CHARMM-TOPPAR-NOTICE.md"


def run_accept(cwd: pathlib.Path) -> int:
  return subprocess.run([sys.executable, "-c", ACCEPT], cwd=cwd, check=False).returncode


def data_rows(tsv: pathlib.Path) -> list[list[str]]:
  return [
    line.split("\t")
    for line in tsv.read_text().splitlines()
    if line.strip() and not line.startswith("#")
  ]


@pytest.fixture
def mirror(tmp_path: pathlib.Path) -> pathlib.Path:
  """A copy of the asset tree: XMLs symlinked, TSV and licence texts copied."""
  src = REPO / ASSETS_REL
  dst = tmp_path / ASSETS_REL
  for xml in src.rglob("*.xml"):
    link = dst / xml.relative_to(src)
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(xml)
  shutil.copy(src / TSV_NAME, dst / TSV_NAME)
  shutil.copytree(src / "licences", dst / "licences")
  return tmp_path


def test_accept_program_passes_on_repo() -> None:
  assert run_accept(REPO) == 0


def test_accept_program_passes_on_unmodified_mirror(mirror: pathlib.Path) -> None:
  assert run_accept(mirror) == 0


def test_red_missing_row(mirror: pathlib.Path) -> None:
  tsv = mirror / ASSETS_REL / TSV_NAME
  lines = tsv.read_text().splitlines()
  first_row = next(i for i, line in enumerate(lines) if line.strip() and not line.startswith("#"))
  del lines[first_row]
  tsv.write_text("\n".join(lines) + "\n")
  assert run_accept(mirror) == 1


def test_red_stray_xml(mirror: pathlib.Path) -> None:
  (mirror / ASSETS_REL / "amber" / "unrecorded.xml").write_text("<ForceField/>\n")
  assert run_accept(mirror) == 1


def test_red_wrong_digest(mirror: pathlib.Path) -> None:
  tsv = mirror / ASSETS_REL / TSV_NAME
  text = tsv.read_text()
  digest = re.search(r"\tsha256\t([0-9a-f]{64}) ", text).group(1)
  tsv.write_text(text.replace(digest, "0" * 64, 1))
  assert run_accept(mirror) == 1


def test_red_edited_licence_text(mirror: pathlib.Path) -> None:
  lic = mirror / ASSETS_REL / "licences" / "openmmforcefields-LICENSE"
  lic.write_bytes(lic.read_bytes() + b"\n")
  assert run_accept(mirror) == 1


def test_red_missing_charmm_notice(mirror: pathlib.Path) -> None:
  (mirror / CHARMM_NOTICE).unlink()
  assert run_accept(mirror) == 1


def test_charmm_derived_xmls_point_at_notice() -> None:
  """No CHARMM-toppar-derived XML may be recorded under an MIT licence text."""
  assets = REPO / ASSETS_REL
  rows = {path: (kind, target) for path, kind, target in data_rows(assets / TSV_NAME)}
  wrong = []
  for path, (kind, target) in rows.items():
    head = (assets / path).read_bytes()[:20000].decode("utf-8", errors="replace")
    if CHARMM_SOURCE.search(head) and (kind, target) != ("file", CHARMM_NOTICE):
      wrong.append(path)
  assert not wrong, wrong
  # Positive control: the check above must have something to find.
  assert sum(1 for kind, target in rows.values() if target == CHARMM_NOTICE) >= 20
