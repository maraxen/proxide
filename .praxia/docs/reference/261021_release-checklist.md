---
title: proxide Release Checklist
description: Numbered steps for cutting a proxide release to PyPI, and the open blockers that gate the first stable release (0.1.0)
status: final
task_id: 261021_proxide-stable-release-scope
date: '261021'
---

# proxide Release Checklist

Pushing a `v*` tag publishes to PyPI through `publish.yml`. Work through the steps in order.

Versioning rule: `crates.io` stays pre-release. For a stable PyPI tag, `pyproject.toml` is the final `X.Y.Z` while the Cargo workspace stays `X.Y.Z-beta.N`. `scripts/check_versions.py` (debt #2909) accepts exactly that relation. For a pre-release PyPI tag, the two versions agree.

## Cutting a release

1. Confirm the versions match the rule above: `python scripts/check_versions.py` (lands with debt #2909). Until it exists, compare `version` in `pyproject.toml` with the Cargo workspace version by hand.
2. In `CHANGELOG.md`, rename `## [Unreleased]` to `## [<version>] - <YYYY-MM-DD>` and open a fresh empty `## [Unreleased]` above it. Check that every **Changed (behaviour)** entry is intended for this release.
3. Confirm CI is green on `main` for both `ci.yml` and `trajectory_parity.yml`. Do not tag a commit whose CI is still running or red.
4. Run `publish.yml` with `workflow_dispatch` on the release branch. It builds all platform wheels and the sdist and publishes nothing; no TestPyPI path exists. Confirm the wheel set and the abi3 wheels.
5. Merge the release PR to `main`. Note the merge commit SHA.
6. Tag the merge commit and push the tag: `git tag v<version> <merge-sha>` then `git push origin v<version>`.
   **WARNING: this push is the irreversible publish.** `publish.yml` triggers on `push: tags: v*`, and its `release` job (environment `pypi`) uploads to PyPI. PyPI does not accept a re-upload of the same version, so push the tag only when the release is final. The `publish-crates` job also runs on any tag until the gate in debt #2914 lands.
7. Optionally publish GitHub Release notes for the tag. This is informational and does not trigger `publish.yml`.
8. Verify the upload on PyPI: the wheel set covers the expected platforms, the wheels are abi3 where declared, and the sdist is present.
9. Smoke test from a clean environment: `pip install proxide==<version>` then `python -c "import proxide"`, and exercise one call from the stable surface (parse a PDB, write it back, read an XTC frame).
10. Bump the version on `main` to the next development version. After a pre-release tag, bump to the next pre-release (for example `0.1.0a18`). After stable `0.1.0`, bump to `0.1.1.dev0` (Cargo `0.1.1-dev.0`, or the next beta). Open a fresh `## [Unreleased]` entry if step 2 did not already.

## Before 0.1.0 (stable)

The stable contract is core IO: structure parsing, formatting (Atom37, Atom14, Full), writing (PDB, mmCIF) and XTC trajectory reading. GAFF2, confind, rotlib and CCD/fixer ship as experimental. `crates.io` stays pre-release. Anything not listed as stable is provisional.

Open blockers, by debt id:

- #2909: version agreement check (`scripts/check_versions.py`)
- #2911, #2912, #2913, #2914 (crates publish gating)
- #1930, #2372, #2362, #2366, #2367, #2233, #2252, #909, #1891, #1331

Resolved 2026-10-21 and removed from this list: #2353 and #2355 (already fixed on `main`). This release-scaffolding change (#2910) is also complete.

Clear the blockers, then cut 0.1.0 by the steps above. Update the stable and experimental split in the README if it changes.
