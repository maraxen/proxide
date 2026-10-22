---
title: proxide Release Checklist
description: Numbered steps for cutting a proxide release to PyPI, and the open blockers that gate the first stable release (0.1.0)
status: final
task_id: 261021_proxide-stable-release-scope
date: '261021'
---

# proxide Release Checklist

Pushing a `v*` tag publishes to PyPI through `publish.yml`. Work through the steps in order.

Versioning rule: `crates.io` stays pre-release. A stable PyPI tag has a final `X.Y.Z` in `pyproject.toml`, while the Cargo workspace carries a pre-release `X.Y.Z-(alpha|beta|rc|dev).N`. `scripts/check_versions.py` (debt #2909, PR #58) accepts exactly that relation, and will support `.devN` in pyproject against `-dev.N` in Cargo.

## Cutting a release

1. Check the version relation: `python scripts/check_versions.py`. It must accept a final pyproject `X.Y.Z` with a Cargo `X.Y.Z-(alpha|beta|rc|dev).N` workspace version.
2. In `CHANGELOG.md`, rename `## [Unreleased]` to `## [<version>] - <YYYY-MM-DD>` and open a fresh empty `## [Unreleased]` above it. Check that every **Changed (behaviour)** entry is intended for this release.
3. Confirm CI is green on `main` for both `ci.yml` and `trajectory_parity.yml`. Do not tag a commit whose CI is still running or red.
4. Dry run: dispatch `publish.yml` with `workflow_dispatch` on a branch ref, never a tag ref. A tag ref runs the `release` job and publishes. The dry run builds all platform wheels and the sdist and publishes nothing, and it skips `verify-crates`. Confirm the wheel set and the abi3 wheels.
5. Version PR, before tagging. Set `[project].version` in `pyproject.toml` to the release version (for example `0.1.0`). Set the Cargo workspace version to the chosen pre-release (for example `0.1.0-beta.1`). Bump every workspace path-dependency pin to match. Run `uv lock`, then `python scripts/check_versions.py`. Commit through a PR and merge it.
6. Note the merge commit SHA of the version PR on `main`.
7. Tag the merge commit and push the tag: `git tag v<version> <merge-sha>` then `git push origin v<version>`.
   **WARNING: this push is the irreversible publish.** `publish.yml` triggers on `push: tags: v*`, and its `release` job (environment `pypi`) uploads to PyPI. PyPI does not accept a re-upload of the same version, so push the tag only when the release is final. `publish-crates` runs only when the Cargo workspace version contains a `-` (pre-release). A stable PyPI tag with Cargo `X.Y.Z-beta.N` still publishes those beta crates; the PyPI release job does not depend on it. The push is irreversible unless the `pypi` environment requires a reviewer approval (check repo Settings, then Environments).
8. Optionally publish GitHub Release notes for the tag. This is informational and does not trigger `publish.yml`.
9. Verify the upload on PyPI: the wheel set covers the expected platforms, the wheels are abi3 where declared, and the sdist is present.
10. Smoke test from a clean environment: `pip install proxide==<version>` then `python -c "import proxide"`, and exercise one call from the stable surface (parse a PDB, write it back, read an XTC frame).
11. Bump the version on `main` to the next development version, in a PR. After stable `0.1.0`, set `pyproject.toml` to `0.1.1.dev0` and the Cargo workspace to `0.1.1-dev.0`, bump the path pins to match, run `uv lock`, and run `python scripts/check_versions.py`. After a pre-release tag, bump both to the next pre-release pair and run the same checks. Open a fresh `## [Unreleased]` entry if step 2 did not already.

## Before 0.1.0 (stable)

The stable contract is core IO: structure parsing, formatting (Atom37, Atom14, Full), writing (PDB, mmCIF) and XTC trajectory reading. GAFF2, confind, rotlib and CCD/fixer ship as experimental. `crates.io` stays pre-release. Anything not listed as stable is provisional.

Open blockers, by debt id:

- #2909: version agreement check (`scripts/check_versions.py`, PR #58 not yet merged)
- #2911, #2912, #2913
- #1930, #2372, #2362, #2366, #2367, #2233, #2252, #909, #1891, #1331

Resolved and removed from this list: #2353 and #2355 (already fixed on `main`, 2026-10-21); #2910 (this change); #2914 (publish-crates gating, PR #60 merged); #2915 (PR #57 merged).

Note, not a blocker: the residual TestPyPI and trusted-publisher check stays tracked under #2914.

Clear the blockers, then cut 0.1.0 by the steps above. Update the stable and experimental split in the README if it changes.
