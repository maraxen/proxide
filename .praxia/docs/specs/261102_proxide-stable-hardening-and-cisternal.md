---
title: proxide stable hardening and cisternal integration
description: 'Deeper audit and remaining work after 0.1.0: licensing, sdist, stable-surface gaps, cisternal skill and plugin, provenance and telemetry, unified CLI and MCP'
status: draft
task_id: 261102_proxide-stable-hardening
date: '261102'
backlog_ids: ''
adversarial_review: ''
---
# proxide stable hardening and cisternal integration

## 1. Where things stand

proxide 0.1.0 is on PyPI as `Production/Stable`, with stable scope limited to core IO (README, Stability). This spec makes that label true and adds the agent surface. It is not a plan to reach a first stable release.

Owner decisions, 2026-11-02:

- D1. Keep the Stable classifier and fix forward.
- D2. The first cisternal integration is the skill and plugin. Stable MCP tools and a unified CLI and MCP interface are debt #3177, specified here.
- D3. Provenance and telemetry are stored with cisternal's tooling.
- D4. `get_mass`, CI lock use and other deferred items are filed as debt, not fixed silently.
- D5. CGenFF (#3182): remove the file, ship the replacement release, then yank 0.1.0. Yanking stays a manual owner step.
- D6. sdist (#3181): restructure the optional GPL dependency so an MIT-only sdist builds.
- D7. Intel macOS (#3183): drop the target. Done on `ci/smoke-built-wheels`, PR #80.

Already in draft PRs, which this spec does not redo:

| PR | Branch | Content |
|---|---|---|
| #79 | `fix/unknown-element-raises` | `infer_element` returns `Option`; parsers raise on unknown names |
| #80 | `ci/smoke-built-wheels` | wheel and sdist smoke jobs gate `release` |
| #81 | `fix/stop-publishing-mosaist-ports` | held crates off crates.io; `NOTICE` files |

Base all new work on `origin/main`. Local `fix/parameterize-solvent-atoms` is 146 commits behind it.

## 2. Owner-only decisions

No agent makes these. A track that reaches one stops, parks the item as `awaiting_feedback`, and continues elsewhere.

| ID | Decision | Blocks |
|---|---|---|
| O1 | Mosaist code (#1420): seek permission from the Grigoryan lab, relicense the affected crates CC BY-NC-SA 4.0, or rewrite clean-room. The owner asked for a module-level map of what actually derives before choosing | T1 end state |
| O2 | Yank the published crates (alpha.9 to alpha.18). A counsel question; yanked versions stay downloadable | nothing in code |
| O3 | The git history rewrite for CGenFF that the unmerged branch deferred, and carrying out the 0.1.0 yank decided in D5 | T2 follow-up |
| O5 | Repo settings: required status checks on the `main` ruleset, a reviewer on the `pypi` environment, and whether the signed-commit bypass stays | T8 |
| O6 | Whether `Dunbrack` attribution needs correcting: the licence URL cited as ODC-BY now serves CC BY 4.0 text (#1917) | nothing in code |

Agents never merge a PR, push a tag, publish, or yank.

## 3. Assumptions

Verified means probed on 2026-11-02 or read at the cited location. These probes are exploratory; re-run under a sidecar anything a deliverable cites.

| Assumption | State | Evidence |
|---|---|---|
| 0.1.0 wheel parses a PDB on 3.11 and 3.13 | verified | clean venv install |
| proxide and cisternal 0.1.5 co-install on 3.13 with no prerelease flag | verified | resolved fastmcp 4.1.5 |
| `@tool`, `wire()` and an in-memory `Client` call work with proxide | verified | throwaway probe |
| `import proxide` does not load cisternal | verified | `sys.modules` check |
| stdout stays clean during a parse, including a missing-atom message | verified, one case | stdout and stderr captured separately |
| jaxlint's typer plugin command works unchanged in proxide | verified by reading only | `jaxlint/cli/plugin_command.py` |
| Stable features are linked and smoke-tested on Linux x86_64 and aarch64, Apple Silicon and Windows wheels, on 3.11 and 3.13 | verified | dry run 41491125466 |
| The Intel macOS wheel installs | verified false, target dropped (D7) | same run: no `grain` distribution for that platform |
| maturin includes `src/proxide/agent_plugin.json` in the wheel | unverified | `include` globs suggest yes; T5 must list a built wheel |
| Real stdio MCP transport works at import and parse cold start | unverified | T7 |
| `wire()` can drive a typer app | unverified, likely no | cisternal documents cyclopts; T7 must decide |
| CGenFF is in every 0.1.0 wheel | verified on x86_64 Linux only | subagent inspection; tracked on `main` |

## 4. Tracks

Each track is one branch and one draft PR off `origin/main`, safe to run in parallel unless marked. Acceptance is a command that fails on the defect, per ledger B1, B3 and B4. A pass with no positive control does not count.

### T1. Licensing records (no code decision)

- Port `260824_license-provenance-verification.md` from `chore/license-provenance-verification` to `main`, adding what the audit corrected: `proxide_fixer` was missed, and `proxide-frag` does not depend on confind.
- Update debts #1420 and #1421 to match.
- Line-compare `proxide-frag` with `mstfasst.cpp` and report evidence only.
- Assess `~/projects/plegadx/vendor/proxide`, which vendors confind and rotlib. Report; do not change that repo.
- Consider a runtime notice in the `confind` and `parse_master` binaries, emitted through `tracing`.

Accept: the spec is on `main`, and `rg -l Mosaist crates/*/NOTICE` lists confind, rotlib and fixer. Depends on #81.

### T2. Remove CGenFF

- Port `14e623d` and `bb4d018`. Take from `55bdfef` only what removal needs, and report the rest.
- Find every code path that loads the file, and make it raise a named error.
- Add a `smoke_wheel.py` check that no `*cgenff*` file is present. Depends on #80.
- Audit the other bundled force fields and record a source and licence for each. `openmm_bundled` and `charmm36*.xml` need checking, and the branch's asset audit (`cdb48c1`) is the starting point.

Accept: `git ls-files | rg -i cgenff` is empty, and the smoke check fails when the file is put back.

### T3. Make the sdist installable

Per D6, restructure so the sdist builds without the GPL crates, for example by removing the workspace path dependencies on `proxide-gaff2` and `proxide-ligand-frame` from what the sdist must resolve, while source builds can still opt in. This approach is untested: spike it first and report if Cargo or maturin rules it out, and do not fall back to shipping GPL sources or dropping the sdist without asking. `smoke-sdist` in #80 is the acceptance test: it is red today and stays a release gate. Do not weaken or remove it to get a green run.

### T4. Deeper stable-surface audit

Audit only what README Stability promises. Findings go to the debt store with a repro; fixes go in separate PRs.

- Readers: `parse_pdb`, `parse_mmcif`, `parse_pqr`, `parse_parm7`, `parse_amber_trajectory`, XTC, DCD, TRR, MDTraj and mdCATH HDF5, foldcomp.
- Known gaps: foldcomp tests never run (#2923), HDF5 parity skips (#2925), Atom37 `UNK` silence (#2920), numbering lost in `project_to_mpnn_batch` (#2928), concatenate crash (#2921), deuterium and ions (#2918).
- New: a file with only a `FE` HETATM returned zero residues and no warning in the default format, while an empty file raises. Decide and test it.
- Sweep each stable path for ledger A1 to A6: `unwrap_or`, `unwrap_or_default`, `.ok()`, `nan_to_num`, `except Exception`, and `documented-default` comments.
- Add a property or fuzz harness for the parsers. 13 hand-written inputs caused no panics, which is weak evidence. A Rust panic surfaces as `PanicException`, which is a `BaseException`.
- Triage stale debts: #2368, #1931, #1932 and #1905 look resolved by the 0.1.0 changelog. Verify each, then resolve it or say why not.
- A count, breakdown or pass rate from this audit is a finding: pre-register it with a committed `.bth.toml` first, and use `bth`, not `uv run bth`.

Accept: every stable name has a test that fails when the behaviour breaks, and calibration uses real historical defects from `calibration/element-inference-*` (ledger B2), not planted ones.

### T5. Cisternal skill and plugin

Copy jaxlint's pattern. cisternal needs Python >=3.13 and proxide supports >=3.11, so it stays optional.

1. Extra: `plugin = ["cisternal>=0.1.5; python_version >= '3.13'"]`.
2. `src/proxide/cli/plugin_command.py`: forward raw argv to `plugin_app(PluginSpec(name="proxide", package="proxide", cli="proxide"))`, import cisternal lazily, and print a clear error naming Python 3.13 when it is absent. Register it with `allow_extra_args` and `ignore_unknown_options`.
3. `.praxia/manifest.toml` with `skills_delivered_by_plugin = true`. Extend `scripts/check_versions.py` to cover its version.
4. `agent_assets/skills/using-proxide/SKILL.md`, triggered on the wording of the mistake: hand-parsing a PDB, `atom37`, mmCIF numbering, NaN for missing atoms. Content: stable and experimental split, auth vs label numbering, `atom_mask` not NaN, unknown elements raise, the GAFF2 DEF is not bundled, licensing of optional crates. Every example must run.
5. Snapshot at `src/proxide/agent_plugin.json`, with `cisternal assets snapshot --check` in CI.
6. Add Python 3.13 to the CI test matrix. Run `proxide plugin info` from a non-editable install and assert it resolves the packaged snapshot.
7. Treat any stderr from `cisternal assets export` as failure, because it exits 0 when it drops assets.

Accept: `unzip -l` on a built wheel lists the snapshot; on 3.11 `proxide plugin` fails with the named message and every other command is unaffected.

### T6. Provenance and telemetry

Goal: any result from the CLI or MCP surface can be traced to the code and input bytes that produced it, stored with cisternal.

Design constraints:

- **Library calls stay silent.** `import proxide` and `parse_structure` never call `cisternal.init` and never write. Only CLI and MCP entry points record.
- **Storage is configuration.** `cisternal.init` defaults to `~/.cisternal/logs`; proxide always passes `log_dir`. Resolve it with the order in `261001_proxide-config-file.md`: explicit argument > `PROXIDE_TELEMETRY_LOG_DIR` (`none` or empty disables) > `[tool.proxide.telemetry]` > `~/.config/proxide/config.toml` > off. Expose which layer decided. A malformed file fails loudly. That spec is still a draft, so land its resolver for this one key rather than adding a second resolver.
- **Absent is its own state.** Without cisternal, or below 3.13, or disabled, the result reports `provenance: {"source": "unavailable", "reason": "..."}`, following `xtrax/telemetry/record.py`. Never a blank or a default that looks like a real record.
- **Git state** comes from `cisternal.provenance.capture_git_state`, as in `alynxr/provenance.py`. Installed-wheel runs have no repo, so also record the distribution version and wheel build, and treat `nogit` as an expected value.
- **Per-operation record:** operation, proxide version, input path and sha256, output path and sha256, features linked, error code if any, and duration. Write it with `cisternal.emit_event`. Include counts from provenance-bearing outputs (`ParameterSource`, `unparameterized_terms`) when an experimental command produces them.
- **Telemetry failure never changes a result,** and is itself a recorded warning (ledger B9).
- **No personal data or file contents** in events; paths only. Note this in the skill.

Link to the observability spec: this is the transport for its Tier 1 `Report`, not a replacement. Do not implement Tier 1 here.

Accept: a test runs `proxide validate` with a temp log dir and checks the record's sha256 against the file; one with telemetry off writes nothing; one with a malformed config exits non-zero; one with cisternal absent reports `unavailable`.

### T7. Unified CLI and MCP (debt #3177, after T5 and T6)

Recommendation, to be challenged in review: plain tool functions in `src/proxide/tools.py` return JSON-serialisable results and raise coded errors. typer stays the CLI on all Python versions as a thin formatter with `--json`. `src/proxide/mcp.py` registers the same bodies with `@cisternal.tool` and `wire()` on 3.13+. Moving the CLI to cyclopts through `wire()` waits until cisternal supports 3.11, because it would otherwise break the CLI for those users.

Scope: stable functions only (`inspect`, `validate`, `fetch`, `convert`). Summaries and paths, never arrays. Long calls need a timeout. `fetch` writes to a caller-supplied directory. Register `PROX-IO-*` codes (#1921) so an agent can branch on them.

Accept: one parametrized test calls each tool through the function, the CLI and an MCP client over real stdio, and asserts identical JSON. A guard fails when a CLI command has no tool, or a tool is not on the stable list.

### T8. Packaging and supply chain

- Add `py.typed` and check a downstream type check resolves the stable API.
- Pin `test_public_api.py` to signatures, not just names.
- CI: run tests from `uv.lock`, plus one named unlocked job (#3180).
- Add `cargo-deny` and `pip-audit`, and Dependabot for actions and dependencies. Pin actions by SHA. Replace `curl | sh` for uv with `astral-sh/setup-uv`.
- The macOS and Windows jobs pass `--features xtc` and are labelled "Slim", yet the smoke check finds all stable features linked, because default features stay on. Fix the labels and flags, and correct the CHANGELOG statement about wheel feature sets.
- Fix the README: the "Pre-release toward 0.1.0" banner, JAX MD mention, and "requires a Rust toolchain". Create the missing GitHub release for `v0.1.0`, which needs the owner's approval because it is outward-facing.
- Add `SECURITY.md`, `CONTRIBUTING.md` and `CITATION.cff`.

Accept: each item is a CI check or a file that exists; none is a statement that it was done.

## 5. Ordering

1. Owner: merge #81, then #79 and #80. Each carries the pytest fix where needed. Decide O1 to O6.
2. In parallel: T1, T2, T3, T4, T8.
3. T5, then T6, then T7.
4. Release as 0.2.0, because #79 changes accepted input on the stable surface. Follow `261021_release-checklist.md`. The owner then yanks 0.1.0 (D5).

## 6. Verification rules

- Read the code and the record, not a summary. Subagent and self-reports do not count as evidence (ledger B1).
- Never run a whole JAX or pytest suite locally. Use narrow selections, or the remote.
- Physics constants change only with a citation and in their own commit.
- Any new default, lookup or fallback must satisfy the ledger entry it resembles. Add an attributed ledger entry for each new instance.

## 7. Risks

| Risk | Mitigation |
|---|---|
| #79 breaks real blank-element files | Stable-surface behaviour change, recorded, and released as 0.2.0; T4 tests downstream fixtures from the proxide-dependent repos |
| Release blocked by red `smoke-sdist` | Intended; T3 resolves it |
| MCP cold start too slow because of JAX | Tools import lazily; T7 measures it at production shapes and records it in the sidecar |
| Telemetry adds a hard dependency | Extra stays optional; T6 tests the absent state |
| Skill drifts from the API | Examples run in CI |
