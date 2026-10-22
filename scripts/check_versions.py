"""Check that proxide's version strings agree with the workspace version.

Reads the PEP 440 version from pyproject.toml ``[project].version`` and the
Cargo version from the root Cargo.toml ``[workspace.package].version``, converts
the PEP 440 form to the Cargo form, and checks that:

* every ``version =`` on a path dependency to a workspace crate (in the root
  Cargo.toml and in ``crates/*/Cargo.toml``) equals the workspace version;
* the Python version agrees with the workspace version;
* ``src/proxide/__init__.py`` and ``docs/source/conf.py`` contain no quoted
  literal version string (they must derive the version from package metadata).

Exit status: 0 when everything agrees, 1 when any mismatch is found.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import tomllib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

log = logging.getLogger("check_versions")

PEP440_RE = re.compile(r"^(\d+\.\d+\.\d+)(?:(a|b|rc)(\d+))?$")
PRE_RELEASE_NAMES = {"a": "alpha", "b": "beta", "rc": "rc"}
LITERAL_VERSION_RE = re.compile(r'"\d+\.\d+\.\d+')
LITERAL_FILES = ("src/proxide/__init__.py", "docs/source/conf.py")


def pep440_to_cargo(version: str) -> str:
  """Convert a PEP 440 version (e.g. ``0.1.0a17``) to the Cargo form (``0.1.0-alpha.17``)."""
  match = PEP440_RE.match(version)
  if match is None:
    raise ValueError(f"unsupported PEP 440 version: {version!r}")
  base, pre, num = match.groups()
  if pre is None:
    return base
  return f"{base}-{PRE_RELEASE_NAMES[pre]}.{num}"


def _iter_path_deps(node: Any, name: str | None = None) -> Iterator[tuple[str, dict[str, Any]]]:
  """Yield (dependency name, table) for every table carrying both ``path`` and ``version``."""
  if not isinstance(node, dict):
    return
  if "path" in node and "version" in node and name is not None:
    yield name, node
  for key, value in node.items():
    if isinstance(value, dict):
      yield from _iter_path_deps(value, key)


def _workspace_members(root_manifest: dict[str, Any]) -> set[str]:
  members = root_manifest.get("workspace", {}).get("members", [])
  return {os.path.normpath(member) for member in members}


def check_versions(root: Path) -> list[str]:
  """Return a list of human-readable mismatch descriptions (empty when all agree)."""
  mismatches: list[str] = []

  pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
  root_manifest = tomllib.loads((root / "Cargo.toml").read_text(encoding="utf-8"))

  py_version = pyproject["project"]["version"]
  cargo_version = root_manifest["workspace"]["package"]["version"]
  py_as_cargo = pep440_to_cargo(py_version)
  log.info("pyproject version %s -> cargo form %s", py_version, py_as_cargo)
  log.info("workspace version %s", cargo_version)
  if py_as_cargo != cargo_version:
    mismatches.append(
      f"pyproject.toml: [project].version {py_version!r} is {py_as_cargo!r} "
      f"in cargo form; Cargo.toml [workspace.package].version is {cargo_version!r}"
    )

  members = _workspace_members(root_manifest)
  manifests = [("Cargo.toml", root_manifest)]
  for crate_manifest in sorted((root / "crates").glob("*/Cargo.toml")):
    rel = crate_manifest.relative_to(root).as_posix()
    manifests.append((rel, tomllib.loads(crate_manifest.read_text(encoding="utf-8"))))

  for rel, manifest in manifests:
    manifest_dir = os.path.dirname(rel)
    for dep_name, table in _iter_path_deps(manifest):
      target = os.path.normpath(os.path.join(manifest_dir, table["path"]))
      if target not in members:
        continue
      pinned = table["version"]
      if pinned != cargo_version:
        mismatches.append(
          f"{rel}: dependency {dep_name!r} pins version {pinned!r}, "
          f"expected workspace version {cargo_version!r}"
        )

  for rel in LITERAL_FILES:
    path = root / rel
    if not path.is_file():
      log.debug("skipping missing file %s", rel)
      continue
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
      if LITERAL_VERSION_RE.search(line):
        mismatches.append(
          f"{rel}:{lineno}: hard-coded version literal {line.strip()!r}; "
          "derive it from importlib.metadata instead"
        )

  return mismatches


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument(
    "--root",
    type=Path,
    default=Path(__file__).resolve().parents[1],
    help="repository root to check (default: the repository containing this script)",
  )
  parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")
  args = parser.parse_args(argv)

  logging.basicConfig(
    level=logging.DEBUG if args.verbose else logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
  )

  mismatches = check_versions(args.root.resolve())
  if not mismatches:
    log.info("all version strings agree")
    return 0

  log.error("%d version mismatch(es):", len(mismatches))
  for item in mismatches:
    log.error("  %s", item)
  return 1


if __name__ == "__main__":
  sys.exit(main())
