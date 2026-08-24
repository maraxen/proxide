#!/usr/bin/env python3
"""Script to sync force field XML files from openmmforcefields and OpenMM.

This script clones/updates the openmmforcefields repository and copies
the relevant XML files into a local scratch copy of the proxide assets
directory structure. It also copies bundled force field files from OpenMM
itself.

These directories are gitignored and no longer vendored/bundled in the
package (see src/proxide/assets/README.md) -- this script exists to
populate a local scratch copy for scripts/generate_asset_index.py to walk,
not to prepare files for committing.

Usage:
    uv run python scripts/sync_forcefields.py [--clone-path /path/to/clone] [--force]

Options:
    --clone-path    Path where openmmforcefields will be cloned (default: /tmp/openmmforcefields)
    --force         Force re-clone even if directory exists
    --dry-run       Show what would be copied without actually copying
"""

import argparse
import shutil
import subprocess
from pathlib import Path

# Repository URL
REPO_URL = "https://github.com/openmm/openmmforcefields.git"

# Asset directory relative to project root
ASSETS_DIR = Path("src/proxide/assets")

# Source paths relative to cloned repo, with organization
FILE_MAPPINGS = {
  # GAFF force field files
  "gaff/ffxml": [
    "openmmforcefields/ffxml/amber/gaff/ffxml/*.xml",
  ],
  "gaff/dat": [
    "openmmforcefields/ffxml/amber/gaff/dat/*.dat",
  ],
  # Amber protein force fields
  "amber": [
    "openmmforcefields/ffxml/amber/ff14SB.xml",
    "openmmforcefields/ffxml/amber/protein.ff19SB.xml",
    "openmmforcefields/ffxml/amber/ff99SBildn.xml",
    "openmmforcefields/ffxml/amber/ff99SB.xml",
    "openmmforcefields/ffxml/amber/ff99.xml",
    "openmmforcefields/ffxml/amber/lipid17.xml",
    "openmmforcefields/ffxml/amber/lipid17_merged.xml",
    "openmmforcefields/ffxml/amber/lipid21_merged.xml",
    "openmmforcefields/ffxml/amber/GLYCAM_06j-1.xml",
    "openmmforcefields/ffxml/amber/phosaa10.xml",
    "openmmforcefields/ffxml/amber/protein.ff03ua.xml",
    "openmmforcefields/ffxml/amber/protein.fb15.xml",
    "openmmforcefields/ffxml/amber/protein.ff15ipq.xml",
    # DNA/RNA
    "openmmforcefields/ffxml/amber/DNA.OL15.xml",
    "openmmforcefields/ffxml/amber/DNA.OL21.xml",
    "openmmforcefields/ffxml/amber/DNA.bsc1.xml",
    "openmmforcefields/ffxml/amber/RNA.OL3.xml",
    "openmmforcefields/ffxml/amber/RNA.ROC.xml",
  ],
  # Water models
  "water": [
    "openmmforcefields/ffxml/amber/tip3p_standard.xml",
    "openmmforcefields/ffxml/amber/tip4pew_standard.xml",
    "openmmforcefields/ffxml/amber/tip4pfb_standard.xml",
    "openmmforcefields/ffxml/amber/opc_standard.xml",
    # Ion parameters for different water models
    "openmmforcefields/ffxml/amber/tip3p_HFE_multivalent.xml",
    "openmmforcefields/ffxml/amber/tip3p_IOD_multivalent.xml",
  ],
  # CHARMM force fields
  # NOTE: charmm36_cgenff.xml deliberately excluded -- CGenFF requires a
  # signed not-for-profit license agreement from the MacKerell lab
  # (mackerell.umaryland.edu/charmm_ff.shtml), not redistributable by
  # default. See .praxia/docs/specs/260824_license-provenance-verification.md.
  # Do not re-add without that agreement in place.
  "charmm": [
    "openmmforcefields/ffxml/charmm/charmm36.xml",
    "openmmforcefields/ffxml/charmm/charmm36_nowaters.xml",
    "openmmforcefields/ffxml/charmm/charmm36_protein.xml",
    "openmmforcefields/ffxml/charmm/waters_ions_default.xml",
    "openmmforcefields/ffxml/charmm/waters_ions_tip3p_pme_b.xml",
  ],
}


def clone_or_update_repo(clone_path: Path, *, force: bool = False) -> bool:
  """Clone or update the openmmforcefields repository."""
  if clone_path.exists():
    if force:
      print(f"Removing existing clone at {clone_path}...")
      shutil.rmtree(clone_path)
    else:
      print(f"Repository already exists at {clone_path}")
      # Try to pull latest
      try:
        subprocess.run(
          ["git", "pull"],
          cwd=clone_path,
          check=True,
          capture_output=True,
        )
        print("Updated to latest version")
      except subprocess.CalledProcessError:
        print("Warning: Could not update repository")
      return True

  print(f"Cloning {REPO_URL} to {clone_path}...")
  result = subprocess.run(
    ["git", "clone", "--depth", "1", REPO_URL, str(clone_path)],
    capture_output=True,
    text=True,
    check=False,
  )

  if result.returncode != 0:
    print(f"Error cloning repository: {result.stderr}")
    return False

  print("Clone successful!")
  return True


def resolve_glob_pattern(repo_path: Path, pattern: str) -> list[Path]:
  """Resolve a glob pattern relative to repo path."""
  full_pattern = repo_path / pattern
  # Use glob on the parent directory with the pattern
  parent = full_pattern.parent
  glob_part = full_pattern.name

  if "*" in str(parent):
    # Pattern in parent path too, use full rglob
    return list(repo_path.glob(pattern))
  return list(parent.glob(glob_part))


def copy_files(
  repo_path: Path,
  assets_path: Path,
  *,
  dry_run: bool = False,
) -> dict[str, list[str]]:
  """Copy files from repo to assets directory following the mapping."""
  results = {"copied": [], "skipped": [], "errors": []}

  for dest_subdir, patterns in FILE_MAPPINGS.items():
    dest_dir = assets_path / dest_subdir

    if not dry_run:
      dest_dir.mkdir(parents=True, exist_ok=True)

    for pattern in patterns:
      source_files = resolve_glob_pattern(repo_path, pattern)

      if not source_files:
        print(f"  Warning: No files match pattern '{pattern}'")
        results["skipped"].append(pattern)
        continue

      for src_file in source_files:
        dest_file = dest_dir / src_file.name

        if dry_run:
          print(f"  Would copy: {src_file.name} -> {dest_subdir}/")
          results["copied"].append(str(dest_file))
        else:
          try:
            shutil.copy2(src_file, dest_file)
            print(f"  Copied: {src_file.name} -> {dest_subdir}/")
            results["copied"].append(str(dest_file))
          except Exception as e:
            print(f"  Error copying {src_file.name}: {e}")
            results["errors"].append(str(src_file))

  return results


def copy_openmm_bundled_files(
  assets_path: Path,
  *,
  dry_run: bool = False,
) -> dict[str, list[str]]:
  """Copy force field files bundled with OpenMM itself."""
  results = {"copied": [], "skipped": [], "errors": []}

  try:
    import openmm.app

    openmm_data = Path(openmm.app.__file__).parent / "data"
  except ImportError:
    print("  Warning: OpenMM not installed, skipping bundled files")
    return results

  print("\nCopying OpenMM bundled files...")
  print("-" * 40)

  # Implicit solvent (GBSA-OBC) files
  implicit_dir = assets_path / "implicit"
  if not dry_run:
    implicit_dir.mkdir(parents=True, exist_ok=True)

  obc_files = list(openmm_data.glob("*_obc.xml"))
  for src_file in obc_files:
    dest_file = implicit_dir / src_file.name
    if dry_run:
      print(f"  Would copy: {src_file.name} -> implicit/")
      results["copied"].append(str(dest_file))
    else:
      shutil.copy2(src_file, dest_file)
      print(f"  Copied: {src_file.name} -> implicit/")
      results["copied"].append(str(dest_file))

  # OpenMM bundled amber files (that may not be in openmmforcefields)
  openmm_amber_dir = assets_path / "openmm_bundled"
  if not dry_run:
    openmm_amber_dir.mkdir(parents=True, exist_ok=True)

  # Copy all top-level XML files from OpenMM data dir
  for src_file in openmm_data.glob("*.xml"):
    # Skip if it's an OBC file (already copied)
    if "_obc" in src_file.name:
      continue
    dest_file = openmm_amber_dir / src_file.name
    if dry_run:
      print(f"  Would copy: {src_file.name} -> openmm_bundled/")
      results["copied"].append(str(dest_file))
    else:
      shutil.copy2(src_file, dest_file)
      print(f"  Copied: {src_file.name} -> openmm_bundled/")
      results["copied"].append(str(dest_file))

  # Copy subdirectory force fields (amber14, amber19, charmm36, etc.)
  for subdir in openmm_data.iterdir():
    if subdir.is_dir() and not subdir.name.startswith("."):
      dest_subdir = openmm_amber_dir / subdir.name
      if not dry_run:
        dest_subdir.mkdir(parents=True, exist_ok=True)
      for src_file in subdir.glob("*.xml"):
        dest_file = dest_subdir / src_file.name
        if dry_run:
          print(f"  Would copy: {src_file.name} -> openmm_bundled/{subdir.name}/")
          results["copied"].append(str(dest_file))
        else:
          shutil.copy2(src_file, dest_file)
          print(f"  Copied: {src_file.name} -> openmm_bundled/{subdir.name}/")
          results["copied"].append(str(dest_file))

  return results


def main() -> int:
  """Main entry point."""
  parser = argparse.ArgumentParser(
    description="Sync force field files from openmmforcefields and OpenMM",
  )
  parser.add_argument(
    "--clone-path",
    type=Path,
    default=Path("/tmp/openmmforcefields"),
    help="Path where openmmforcefields will be cloned",
  )
  parser.add_argument(
    "--force",
    action="store_true",
    help="Force re-clone even if directory exists",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Show what would be copied without copying",
  )
  parser.add_argument(
    "--assets-dir",
    type=Path,
    default=None,
    help="Override assets directory path",
  )

  args = parser.parse_args()

  # Determine project root (script is in scripts/)
  script_dir = Path(__file__).parent
  project_root = script_dir.parent

  assets_path = args.assets_dir or (project_root / ASSETS_DIR)

  print(f"Project root: {project_root}")
  print(f"Assets directory: {assets_path}")
  print()

  # Clone or update repo
  if not clone_or_update_repo(args.clone_path, force=args.force):
    return 1

  print()
  print("Copying openmmforcefields files...")
  print("-" * 40)

  results = copy_files(args.clone_path, assets_path, dry_run=args.dry_run)

  print("-" * 40)
  print(f"Copied from openmmforcefields: {len(results['copied'])} files")

  # Copy OpenMM bundled files
  openmm_results = copy_openmm_bundled_files(assets_path, dry_run=args.dry_run)
  print("-" * 40)
  print(f"Copied from OpenMM: {len(openmm_results['copied'])} files")

  if results["skipped"]:
    print(f"Skipped patterns: {len(results['skipped'])}")
  if results["errors"]:
    print(f"Errors: {len(results['errors'])}")

  print()
  print("Done! This populates a local scratch copy only -- these directories are")
  print("gitignored and not vendored/bundled anymore (fetch-on-demand, see")
  print("src/proxide/assets/README.md). Typically you'd run this only to")
  print("regenerate _asset_index.py: uv run scripts/generate_asset_index.py")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
