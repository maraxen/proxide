"""Check that proxide's version strings agree with the workspace version.

Reads the PEP 440 version from pyproject.toml ``[project].version`` and the
Cargo version from the root Cargo.toml ``[workspace.package].version``, converts
the PEP 440 form to the Cargo form, and checks that:

* the Python version (converted) equals the workspace version, or pyproject is a
  final ``X.Y.Z`` and the workspace is a pre-release ``X.Y.Z-<alpha|beta|rc>.N``
  (PyPI stable, crates.io gated);
* every ``[package]`` manifest (root and ``crates/*/Cargo.toml``) either
  inherits ``version.workspace = true`` or pins the workspace version;
* every path dependency on a workspace member carries a ``version`` equal to
  the workspace version (``cargo publish`` rejects a path dependency without
  one). ``workspace = true`` entries resolve through the root pin, which is
  itself checked;
* ``src/proxide/__init__.py`` and ``docs/source/conf.py`` do not assign a
  hard-coded version string (the fallback ``"0+unknown"`` is allowed);
* ``uv.lock``'s ``proxide`` entry matches the Python version (reported as a
  warning only; see the docstring of ``check_uv_lock``).

Exit status: 0 when everything agrees, 1 when any mismatch is found.
"""

from __future__ import annotations

import argparse
import ast
import logging
import os
import re
import sys
import tomllib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

log = logging.getLogger("check_versions")

PEP440_RE = re.compile(r"(\d+)\.(\d+)\.(\d+)(?:(a|b|rc)(\d+))?")
PRE_RELEASE_NAMES = {"a": "alpha", "b": "beta", "rc": "rc"}
FALLBACK_VERSION = "0+unknown"
SOURCE_LITERAL_FILES = {
  "src/proxide/__init__.py": ("__version__",),
  "docs/source/conf.py": ("release", "version"),
}
DEPENDENCY_SECTIONS = ("dependencies", "dev-dependencies", "build-dependencies")


FINAL_RE = re.compile(r"\d+\.\d+\.\d+")
CARGO_PRERELEASE_RE = re.compile(r"(\d+\.\d+\.\d+)-(alpha|beta|rc)\.\d+")


def _final_pypi_prerelease_cargo(py_version: str, cargo_version: str) -> bool:
  """True when pyproject is final X.Y.Z and Cargo is a pre-release of the same X.Y.Z."""
  cargo = CARGO_PRERELEASE_RE.fullmatch(cargo_version)
  return (
    FINAL_RE.fullmatch(py_version) is not None
    and cargo is not None
    and cargo.group(1) == py_version
  )


def pep440_to_cargo(version: str) -> str:
  """Convert a PEP 440 release or pre-release (e.g. ``0.1.0a17``) to Cargo form.

  Raises ValueError for anything else (dev, post, epoch, local segments).
  """
  match = PEP440_RE.fullmatch(version)
  if match is None:
    raise ValueError(f"unsupported PEP 440 version {version!r} (only X.Y.Z[a|b|rc]N is handled)")
  major, minor, patch, pre, num = match.groups()
  base = f"{int(major)}.{int(minor)}.{int(patch)}"
  if pre is None:
    return base
  return f"{base}-{PRE_RELEASE_NAMES[pre]}.{int(num)}"


def _dependency_tables(manifest: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
  """Yield (name, table) for every entry under the dependency sections.

  Covers top-level sections, ``[workspace.dependencies]`` and
  ``[target.<cfg>.<section>]``. ``[lib]``/``[[bin]]`` ``path`` keys are not
  dependencies and are not visited.
  """
  sections: list[dict[str, Any]] = []
  for section in DEPENDENCY_SECTIONS:
    if isinstance(manifest.get(section), dict):
      sections.append(manifest[section])
  workspace = manifest.get("workspace")
  if isinstance(workspace, dict) and isinstance(workspace.get("dependencies"), dict):
    sections.append(workspace["dependencies"])
  targets = manifest.get("target")
  if isinstance(targets, dict):
    for cfg in targets.values():
      if isinstance(cfg, dict):
        for section in DEPENDENCY_SECTIONS:
          if isinstance(cfg.get(section), dict):
            sections.append(cfg[section])
  for section in sections:
    for name, table in section.items():
      if isinstance(table, dict):
        yield name, table


def _workspace_members(root_manifest: dict[str, Any]) -> set[str]:
  members = root_manifest.get("workspace", {}).get("members", [])
  return {os.path.normpath(member) for member in members}


def check_source_literals(root: Path) -> list[str]:
  """Flag str constants assigned to version names in the package and docs config."""
  problems: list[str] = []
  for rel, names in SOURCE_LITERAL_FILES.items():
    path = root / rel
    if not path.is_file():
      problems.append(f"{rel}: file is missing (cannot verify its version handling)")
      continue
    try:
      tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
    except SyntaxError as exc:
      problems.append(f"{rel}:{exc.lineno}: cannot parse ({exc.msg})")
      continue
    for node in ast.walk(tree):
      if isinstance(node, ast.Assign):
        targets = node.targets
        value = node.value
      elif isinstance(node, ast.AnnAssign) and node.value is not None:
        targets = [node.target]
        value = node.value
      else:
        continue
      assigned = [t.id for t in targets if isinstance(t, ast.Name) and t.id in names]
      if not assigned or not isinstance(value, ast.Constant) or not isinstance(value.value, str):
        continue
      if value.value == FALLBACK_VERSION:
        continue
      problems.append(
        f"{rel}:{node.lineno}: {assigned[0]} is hard-coded to {value.value!r}; "
        "derive it from importlib.metadata instead"
      )
  return problems


def check_uv_lock(root: Path, py_version: str) -> list[str]:
  """Warn when uv.lock's ``proxide`` entry disagrees with pyproject.toml.

  Reported as a warning, not a mismatch: the lock could only be refreshed
  with a uv build that also rewrites unrelated marker lines, so it is left to
  the maintainer to run ``uv lock`` on a matching uv.
  """
  lock_path = root / "uv.lock"
  if not lock_path.is_file():
    log.warning("uv.lock not found; skipping lock check")
    return []
  lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
  entries = [p for p in lock.get("package", []) if p.get("name") == "proxide"]
  if not entries:
    log.warning("uv.lock has no [[package]] entry for proxide")
    return []
  for entry in entries:
    if entry.get("version") != py_version:
      log.warning(
        "uv.lock proxide entry is %r, pyproject.toml is %r; run `uv lock` to refresh",
        entry.get("version"),
        py_version,
      )
  return []


def check_versions(root: Path) -> list[str]:
  """Return a list of human-readable mismatch descriptions (empty when all agree)."""
  mismatches: list[str] = []

  pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
  root_manifest = tomllib.loads((root / "Cargo.toml").read_text(encoding="utf-8"))

  py_version = pyproject["project"]["version"]
  try:
    py_as_cargo = pep440_to_cargo(py_version)
  except ValueError as exc:
    py_as_cargo = None
    mismatches.append(f"pyproject.toml: [project].version: {exc}")

  cargo_version = root_manifest.get("workspace", {}).get("package", {}).get("version")
  if cargo_version is None:
    mismatches.append("Cargo.toml: [workspace.package].version is missing")
  elif py_as_cargo is not None and py_as_cargo != cargo_version:
    if _final_pypi_prerelease_cargo(py_version, cargo_version):
      log.info(
        "pyproject is final %s; cargo stays pre-release %s (crates.io gated)",
        py_version,
        cargo_version,
      )
    else:
      mismatches.append(
        f"pyproject.toml: [project].version {py_version!r} is {py_as_cargo!r} in cargo form; "
        f"Cargo.toml [workspace.package].version is {cargo_version!r}"
      )
  log.info("pyproject version %s -> cargo form %s", py_version, py_as_cargo)
  log.info("workspace version %s", cargo_version)

  members = _workspace_members(root_manifest)
  manifests = [("Cargo.toml", root_manifest)]
  for crate_manifest in sorted((root / "crates").glob("*/Cargo.toml")):
    rel = crate_manifest.relative_to(root).as_posix()
    manifests.append((rel, tomllib.loads(crate_manifest.read_text(encoding="utf-8"))))

  for rel, manifest in manifests:
    package = manifest.get("package")
    if isinstance(package, dict):
      pkg_version = package.get("version")
      inherits = isinstance(pkg_version, dict) and pkg_version.get("workspace") is True
      if not inherits and pkg_version != cargo_version:
        mismatches.append(
          f"{rel}: [package].version is {pkg_version!r}; expected "
          f"`version.workspace = true` or {cargo_version!r}"
        )

    manifest_dir = os.path.dirname(rel)
    for dep_name, table in _dependency_tables(manifest):
      if "path" not in table:
        continue
      target = os.path.normpath(os.path.join(manifest_dir, table["path"]))
      if target not in members:
        log.warning(
          "%s: path dependency %r points at %r, which is not a workspace member; not checked",
          rel,
          dep_name,
          target,
        )
        continue
      if "version" not in table:
        mismatches.append(
          f"{rel}: path dependency {dep_name!r} on workspace member has no `version` "
          "(cargo publish rejects it)"
        )
        continue
      if table["version"] != cargo_version:
        mismatches.append(
          f"{rel}: dependency {dep_name!r} pins version {table['version']!r}, "
          f"expected workspace version {cargo_version!r}"
        )

  mismatches.extend(check_source_literals(root))
  check_uv_lock(root, py_version)
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
