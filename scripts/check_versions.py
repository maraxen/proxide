"""Check that proxide's version strings agree with the workspace version.

Reads the PEP 440 version from pyproject.toml ``[project].version`` and the
Cargo version from the root Cargo.toml ``[workspace.package].version``, converts
the PEP 440 form to the Cargo form (``X.Y.Z``, ``X.Y.Z-alpha.N``,
``X.Y.Z-beta.N``, ``X.Y.Z-rc.N``, ``X.Y.Z-dev.N``), and checks that:

* the Python version (converted) equals the workspace version, or pyproject is a
  final ``X.Y.Z`` and the workspace is a pre-release (alpha, beta, rc or dev)
  of the same ``X.Y.Z`` (PyPI stable, crates.io gated);
* the workspace members in ``[workspace].members`` (globs expanded) all exist
  and have a Cargo.toml; a ``crates/*/Cargo.toml`` that is not a member is only
  a warning;
* every ``[package]`` manifest either inherits ``version.workspace = true`` or
  pins the workspace version;
* every path dependency points at a workspace member and carries a ``version``
  equal to the workspace version (``cargo publish`` rejects a path dependency
  without one). ``workspace = true`` entries resolve through the root pin,
  which is itself checked;
* ``src/proxide/__init__.py`` and ``docs/source/conf.py`` do not assign a
  hard-coded version string (the fallback ``"0+unknown"`` is allowed);
* ``uv.lock``'s ``proxide`` entry matches the Python version (warning only; see
  ``check_uv_lock``).

Missing files, unparsable TOML, dynamic or missing versions and unsupported
PEP 440 forms (epoch, post, local) are reported as mismatches, never as
tracebacks.

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

PEP440_RE = re.compile(r"(\d+)\.(\d+)\.(\d+)(?:(?:(a|b|rc)(\d+))|(?:\.dev(\d+)))?")
PRE_RELEASE_NAMES = {"a": "alpha", "b": "beta", "rc": "rc"}
FINAL_RE = re.compile(r"\d+\.\d+\.\d+")
CARGO_PRERELEASE_RE = re.compile(r"(\d+\.\d+\.\d+)-(alpha|beta|rc|dev)\.\d+")
FALLBACK_VERSION = "0+unknown"
SOURCE_LITERAL_FILES = {
  "src/proxide/__init__.py": ("__version__",),
  "docs/source/conf.py": ("release", "version"),
}
DEPENDENCY_SECTIONS = ("dependencies", "dev-dependencies", "build-dependencies")


def pep440_to_cargo(version: str) -> str:
  """Convert a PEP 440 version to the Cargo form.

  Supports ``X.Y.Z``, ``X.Y.Za|b|rcN`` and ``X.Y.Z.devN``; anything else
  (epoch, post, local segments) raises ValueError.
  """
  match = PEP440_RE.fullmatch(version)
  if match is None:
    raise ValueError(
      f"unsupported PEP 440 version {version!r} "
      "(only X.Y.Z, X.Y.Za|b|rcN and X.Y.Z.devN are handled)"
    )
  major, minor, patch, pre, pre_num, dev_num = match.groups()
  base = f"{int(major)}.{int(minor)}.{int(patch)}"
  if dev_num is not None:
    return f"{base}-dev.{int(dev_num)}"
  if pre is None:
    return base
  return f"{base}-{PRE_RELEASE_NAMES[pre]}.{int(pre_num)}"


def _final_pypi_prerelease_cargo(py_version: str, cargo_version: str) -> bool:
  """True when pyproject is final X.Y.Z and Cargo is a pre-release of the same X.Y.Z."""
  cargo = CARGO_PRERELEASE_RE.fullmatch(cargo_version)
  return (
    FINAL_RE.fullmatch(py_version) is not None
    and cargo is not None
    and cargo.group(1) == py_version
  )


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


def _read_toml(path: Path, rel: str, mismatches: list[str]) -> dict[str, Any] | None:
  if not path.is_file():
    mismatches.append(f"{rel}: file is missing")
    return None
  try:
    return tomllib.loads(path.read_text(encoding="utf-8"))
  except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
    mismatches.append(f"{rel}: cannot parse ({exc})")
    return None


def _static_version(
  doc: dict[str, Any] | None, rel: str, table: str, path: tuple[str, ...], mismatches: list[str]
) -> str | None:
  """Return a static version string from ``doc[path]``, or record why it is missing."""
  if doc is None:
    return None
  node: Any = doc
  for key in path:
    node = node.get(key) if isinstance(node, dict) else None
  if isinstance(node, str):
    return node
  mismatches.append(
    f"{rel}: {table} is missing or not a static string "
    "(dynamic versions are not supported; set it literally)"
  )
  return None


def check_uv_lock(root: Path, py_version: str | None) -> None:
  """Warn when uv.lock's ``proxide`` entry disagrees with pyproject.toml.

  Reported as a warning, not a mismatch: the lock could only be refreshed
  with a uv build that also rewrites unrelated marker lines, so it is left to
  the maintainer to run ``uv lock`` on a matching uv.
  """
  lock_path = root / "uv.lock"
  if not lock_path.is_file():
    log.warning("uv.lock not found; skipping lock check")
    return
  try:
    lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
  except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
    log.warning("uv.lock cannot be parsed (%s); skipping lock check", exc)
    return
  entries = [p for p in lock.get("package", []) if p.get("name") == "proxide"]
  if not entries:
    log.warning("uv.lock has no [[package]] entry for proxide")
    return
  for entry in entries:
    if py_version is not None and entry.get("version") != py_version:
      log.warning(
        "uv.lock proxide entry is %r, pyproject.toml is %r; run `uv lock` to refresh",
        entry.get("version"),
        py_version,
      )


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
    except (SyntaxError, UnicodeDecodeError) as exc:
      problems.append(f"{rel}: cannot parse ({exc})")
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


def _expand_members(root: Path, patterns: list[str], mismatches: list[str]) -> dict[str, Path]:
  """Map each workspace member directory (relative, normalised) to its manifest path."""
  members: dict[str, Path] = {}
  for pattern in patterns:
    dirs = sorted(p for p in root.glob(pattern) if p.is_dir())
    if not dirs:
      mismatches.append(f"Cargo.toml: workspace member {pattern!r} matches no directory")
      continue
    for directory in dirs:
      rel_dir = os.path.normpath(directory.relative_to(root).as_posix())
      manifest = directory / "Cargo.toml"
      if not manifest.is_file():
        mismatches.append(f"{rel_dir}/Cargo.toml: workspace member has no manifest")
        continue
      members[rel_dir] = manifest
  return members


def check_versions(root: Path) -> list[str]:
  """Return a list of human-readable mismatch descriptions (empty when all agree)."""
  mismatches: list[str] = []

  pyproject = _read_toml(root / "pyproject.toml", "pyproject.toml", mismatches)
  root_manifest = _read_toml(root / "Cargo.toml", "Cargo.toml", mismatches)

  py_version = _static_version(
    pyproject, "pyproject.toml", "[project].version", ("project", "version"), mismatches
  )
  cargo_version = _static_version(
    root_manifest,
    "Cargo.toml",
    "[workspace.package].version",
    ("workspace", "package", "version"),
    mismatches,
  )

  py_as_cargo: str | None = None
  if py_version is not None:
    try:
      py_as_cargo = pep440_to_cargo(py_version)
    except ValueError as exc:
      mismatches.append(f"pyproject.toml: [project].version: {exc}")
  log.info("pyproject version %s -> cargo form %s", py_version, py_as_cargo)
  log.info("workspace version %s", cargo_version)

  if py_as_cargo is not None and cargo_version is not None and py_as_cargo != cargo_version:
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

  patterns = []
  if root_manifest is not None:
    patterns = root_manifest.get("workspace", {}).get("members", [])
  members = _expand_members(root, patterns, mismatches)

  for crate_manifest in sorted((root / "crates").glob("*/Cargo.toml")):
    rel_dir = os.path.normpath(crate_manifest.parent.relative_to(root).as_posix())
    if rel_dir not in members:
      log.warning("%s/Cargo.toml is not a workspace member; not checked", rel_dir)

  manifests: list[tuple[str, str, dict[str, Any]]] = []
  if root_manifest is not None:
    manifests.append(("Cargo.toml", "", root_manifest))
  for rel_dir, manifest_path in sorted(members.items()):
    rel_file = f"{rel_dir}/Cargo.toml"
    manifest = _read_toml(manifest_path, rel_file, mismatches)
    if manifest is not None:
      manifests.append((rel_file, rel_dir, manifest))

  for rel_file, rel_dir, manifest in manifests:
    package = manifest.get("package")
    if isinstance(package, dict) and cargo_version is not None:
      pkg_version = package.get("version")
      inherits = isinstance(pkg_version, dict) and pkg_version.get("workspace") is True
      if not inherits and pkg_version != cargo_version:
        mismatches.append(
          f"{rel_file}: [package].version is {pkg_version!r}; expected "
          f"`version.workspace = true` or {cargo_version!r}"
        )

    for dep_name, table in _dependency_tables(manifest):
      if "path" not in table:
        continue
      target = os.path.normpath(os.path.join(rel_dir, table["path"]))
      if target not in members:
        mismatches.append(
          f"{rel_file}: path dependency {dep_name!r} points at {target!r}, "
          "which is not a workspace member"
        )
        continue
      if "version" not in table:
        mismatches.append(
          f"{rel_file}: path dependency {dep_name!r} on workspace member has no `version` "
          "(cargo publish rejects it)"
        )
        continue
      if cargo_version is not None and table["version"] != cargo_version:
        mismatches.append(
          f"{rel_file}: dependency {dep_name!r} pins version {table['version']!r}, "
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
