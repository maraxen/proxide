"""Runtime fetch-and-cache for bundled force field assets.

See .praxia/docs/specs/260824_license-provenance-verification.md for the
full rationale. Two categories of content:

- **Fetchable** (amber/, water/, implicit/, most of openmm_bundled/): no
  restrictive license found on audit, but the data is large and was
  previously git-vendored from an unpinned branch clone (drift risk). Now
  fetched on first use from a pinned commit and cached locally -- not
  committed to git, not bundled in the wheel.
- **CHARMM-restricted** (charmm/, and the CGenFF/CHARMM36-derived files
  bundled directly by OpenMM itself under openmm_bundled/): CGenFF requires
  a signed not-for-profit license agreement from the MacKerell lab; the base
  CHARMM36 toppar files have no positive redistribution grant we could find
  either. proxide never fetches these on a user's behalf -- that would just
  move the same unresolved redistribution question from "bundled in git" to
  "silently fetched at runtime," not resolve it (a script can't complete a
  signature requirement). Users who have their own rights to this content
  point proxide at it directly via `PROXIDE_CHARMM_TOPPAR_DIR`.
"""

from __future__ import annotations

import hashlib
import logging
import os
import urllib.error
import urllib.request
from pathlib import Path

import platformdirs

logger = logging.getLogger(__name__)

# Pinned commits (not branches) -- content-addressed, so a raw fetch against
# one of these SHAs can't silently drift the way the old unpinned
# `git clone` in scripts/sync_forcefields.py could. Re-pin deliberately via
# `scripts/pin_asset_sources.py` when a newer upstream release is wanted.
OMF_REPO = "openmm/openmmforcefields"
OMF_REF = "fb86b916b76393392c39b8b73e2eb5c770908942"  # 2026-08-24
OPENMM_REPO = "openmm/openmm"
OPENMM_REF = "8eb6462c41c5c8c6ff456108f594bc5a9ed9429a"  # 2026-08-24

# subdir prefix (as used under src/proxide/assets/) -> (repo, ref, path prefix in that repo)
_SOURCE_MAP: dict[str, tuple[str, str, str]] = {
  "amber": (OMF_REPO, OMF_REF, "openmmforcefields/ffxml/amber"),
  "water": (OMF_REPO, OMF_REF, "openmmforcefields/ffxml/amber"),
  "implicit": (OPENMM_REPO, OPENMM_REF, "wrappers/python/openmm/app/data"),
  "openmm_bundled": (OPENMM_REPO, OPENMM_REF, "wrappers/python/openmm/app/data"),
}

# Explicit clean subdirectories inside openmm_bundled/ -- confirmed by audit
# to be water models only, not CHARMM protein/CGenFF content (see spec
# Appendix B). Everything else with "charmm" in the path is restricted.
_OPENMM_BUNDLED_CHARMM_ALLOWLIST = ("openmm_bundled/charmm36/", "openmm_bundled/charmm36_2024/")


class CharmmLicenseRequiredError(RuntimeError):
  """Raised when CGenFF/CHARMM36 content is requested without a user-supplied path.

  proxide does not fetch or bundle this content -- see module docstring.
  """


def is_charmm_restricted(relative_path: str) -> bool:
  """True if `relative_path` (posix-style, relative to assets/) is CGenFF/CHARMM-derived."""
  normalized = relative_path.replace("\\", "/").lstrip("/")
  if normalized.startswith("charmm/"):
    return True
  if "charmm" in normalized.lower():
    return not any(normalized.startswith(prefix) for prefix in _OPENMM_BUNDLED_CHARMM_ALLOWLIST)
  return False


def _cache_dir() -> Path:
  override = os.environ.get("PROXIDE_ASSET_CACHE_DIR")
  base = Path(override) if override else Path(platformdirs.user_cache_dir("proxide"))
  return base / "assets"


def _source_for(relative_path: str) -> tuple[str, str, str] | None:
  top = relative_path.split("/", 1)[0]
  return _SOURCE_MAP.get(top)


def resolve_asset(relative_path: str) -> Path:
  """Return a local path to `relative_path` (e.g. "amber/ff14SB.xml"), fetching if needed.

  Raises CharmmLicenseRequiredError for CGenFF/CHARMM-derived paths -- use
  resolve_charmm_toppar for those instead.
  """
  normalized = relative_path.replace("\\", "/").lstrip("/")
  if is_charmm_restricted(normalized):
    raise CharmmLicenseRequiredError(
      f"'{normalized}' is CGenFF/CHARMM36-derived content. CGenFF requires a signed "
      "not-for-profit license agreement from the MacKerell lab "
      "(mackerell.umaryland.edu/charmm_ff.shtml); proxide does not fetch or bundle it. "
      "If you have your own rights to these files, set PROXIDE_CHARMM_TOPPAR_DIR to the "
      "directory containing them. See "
      ".praxia/docs/specs/260824_license-provenance-verification.md for the full rationale."
    )

  cache_path = _cache_dir() / normalized
  if cache_path.exists():
    return cache_path

  source = _source_for(normalized)
  if source is None:
    raise ValueError(f"No known fetch source for asset '{normalized}'")
  repo, ref, prefix = source
  # relative_path's first component is the assets/ subdir name (amber, water,
  # implicit, openmm_bundled); the upstream layout doesn't have that
  # component, so strip it before joining to the source prefix.
  tail = normalized.split("/", 1)[1] if "/" in normalized else normalized
  url = f"https://raw.githubusercontent.com/{repo}/{ref}/{prefix}/{tail}"

  logger.info("Fetching %s from %s (pinned commit %s)", normalized, repo, ref[:12])
  try:
    with urllib.request.urlopen(  # noqa: S310 -- fixed https raw.githubusercontent.com host
      urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}),
      timeout=30,
    ) as response:
      data = response.read()
  except urllib.error.HTTPError as e:
    raise FileNotFoundError(
      f"Could not fetch '{normalized}' from {url} (pinned commit {ref}): {e}"
    ) from e

  cache_path.parent.mkdir(parents=True, exist_ok=True)
  cache_path.write_bytes(data)
  return cache_path


def resolve_charmm_toppar(name: str) -> Path:
  """Resolve a CGenFF/CHARMM toppar file from the user-supplied directory.

  Set PROXIDE_CHARMM_TOPPAR_DIR to a directory containing your own,
  properly-licensed copy of the CHARMM additive toppar files.
  """
  toppar_dir = os.environ.get("PROXIDE_CHARMM_TOPPAR_DIR")
  if not toppar_dir:
    raise CharmmLicenseRequiredError(
      f"'{name}' requires CGenFF/CHARMM36 toppar files, which proxide does not bundle or "
      "fetch (see .praxia/docs/specs/260824_license-provenance-verification.md). Set "
      "PROXIDE_CHARMM_TOPPAR_DIR to a directory containing your own copy, obtained per "
      "the terms at mackerell.umaryland.edu/charmm_ff.shtml -- CGenFF specifically requires "
      "a signed not-for-profit license agreement (or a SilcsBio license for commercial use)."
    )
  candidate = Path(toppar_dir) / name
  if not candidate.exists() and not candidate.with_suffix(".xml").exists():
    raise FileNotFoundError(
      f"'{name}' not found under PROXIDE_CHARMM_TOPPAR_DIR ({toppar_dir})"
    )
  return candidate if candidate.exists() else candidate.with_suffix(".xml")


def sha256_of(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()
