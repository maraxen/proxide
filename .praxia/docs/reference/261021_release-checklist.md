---
title: proxide Release Checklist
description: Numbered steps for cutting a proxide release to PyPI, and the open blockers that gate the first stable release (0.1.0)
status: final
task_id: 261021_proxide-stable-release-scope
date: '261021'
---

# proxide Release Checklist

Cutting a release publishes wheels to PyPI through `publish.yml`. Work through the steps in order.

## Cutting a release

1. Confirm the version agrees everywhere: `python scripts/check_versions.py` (lands with debt #2909). Until it exists, compare `version` in `pyproject.toml` with the Cargo workspace version by hand.
2. In `CHANGELOG.md`, rename `## [Unreleased]` to `## [<version>] - <YYYY-MM-DD>` and open a fresh empty `## [Unreleased]` above it. Check that every **Changed (behaviour)** entry is intended for this release.
3. Confirm CI is green on `main` for both `ci.yml` and `trajectory_parity.yml`. Do not tag a commit whose CI is still running or red.
4. Run `publish.yml` as a dry run against TestPyPI. Confirm every platform wheel and the sdist build, and that the abi3 wheels are present.
5. Merge the release PR to `main`. Note the merge commit SHA.
6. Tag the merge commit: `git tag v<version> <merge-sha>` and `git push origin v<version>`.
7. Publish a GitHub Release for the tag. Publishing the release triggers `publish.yml` for the real PyPI upload.
8. Verify the upload on PyPI: the wheel set covers the expected platforms, the wheels are abi3 where declared, and the sdist is present.
9. Smoke test from a clean environment: `pip install proxide==<version>` then `python -c "import proxide"`, and exercise one call from the stable surface (parse a PDB, write it back, read an XTC frame).
10. Bump `version` on `main` to the next development version (for example `0.1.0a18` after `0.1.0a17`), and open a fresh `## [Unreleased]` entry if step 2 did not already.

## Before 0.1.0 (stable)

The stable contract is core IO: structure parsing, formatting (Atom37, Atom14, Full), writing (PDB, mmCIF) and XTC trajectory reading. GAFF2, confind, rotlib and CCD/fixer ship as experimental. crates.io stays pre-release.

Open blockers, by debt id:

- #2909: version agreement check (`scripts/check_versions.py`)
- #2910: release scaffolding (this checklist, CHANGELOG, Beta classifier)
- #2911, #2912, #2913, #2914
- #1930, #2372, #2355, #2362, #2353, #2366, #2367, #2233, #2252, #909, #1891, #1331

Clear the blockers, then cut 0.1.0 by the steps above. Update the stable and experimental split in the README if it changes.
