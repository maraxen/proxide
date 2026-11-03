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
| O1 | Mosaist code (#1420): seek permission from the Grigoryan lab, relicense the affected crates CC BY-NC-SA 4.0, or rewrite clean-room. Evidence: `audits/261102_mosaist-derivation-map.md` | T1 end state |
| O2 | Yank the published crates (alpha.9 to alpha.18). A counsel question; yanked versions stay downloadable | nothing in code |
| O3 | The git history rewrite for CGenFF that the unmerged branch deferred, and carrying out the 0.1.0 yank decided in D5 | T2 follow-up |
| O5 | Repo settings: required status checks on the `main` ruleset, a reviewer on the `pypi` environment, and whether the signed-commit bypass stays | T8 |
| O6 | Whether `Dunbrack` attribution needs correcting: the licence URL cited as ODC-BY now serves CC BY 4.0 text (#1917) | nothing in code |

Agents never merge a PR, push a tag, publish, or yank.

## 3. Assumptions

Verified means probed on 2026-11-02 or read at the cited location, and the probe or the file is in this repo. A session probe with no checked-in artifact is unverified. A workflow-file fact is recorded separately from a dry-run result. These probes are exploratory; re-run under a sidecar anything a deliverable cites.

| Assumption | State | Evidence |
|---|---|---|
| 0.1.0 wheel parses a PDB on 3.11 and 3.13 | unverified | observed in session, no checked-in artifact (clean venv install) |
| proxide and cisternal 0.1.5 co-install on 3.13 with no prerelease flag | unverified | observed in session, no checked-in artifact (fastmcp 4.1.5 resolution) |
| `@tool`, `wire()` and an in-memory `Client` call work with proxide | unverified | observed in session, no checked-in artifact (`@tool` / `wire()` / `Client` probe) |
| `import proxide` does not load cisternal | unverified | observed in session, no checked-in artifact (`sys.modules` check) |
| stdout stays clean during a parse, including a missing-atom message | unverified | observed in session, no checked-in artifact (stdout and stderr captured separately) |
| jaxlint's typer plugin command works unchanged in proxide | verified by reading only | `jaxlint/cli/plugin_command.py` |
| Stable features are linked and smoke-tested on Linux x86_64 and aarch64, Apple Silicon and Windows wheels, on 3.11 and 3.13 | unverified | dry run 41491125466 is not recorded in the repo. Workflow-file fact, separate from that run: on this tree `.github/workflows/publish.yml` the linux job passes `--features full` and the macOS and Windows jobs pass `--features xtc` |
| The Intel macOS wheel installs | unverified | dry-run claim (no `grain` distribution on run 41491125466) is not recorded in the repo. Workflow-file fact, separate: on `ci/smoke-built-wheels`, `.github/workflows/publish.yml` drops `macos-15-intel` and comments that grain has no build for that platform (debt #3183, D7). This tree's `publish.yml` still lists `macos-15-intel` |
| maturin includes `src/proxide/agent_plugin.json` in the wheel | unverified | `include` globs suggest yes; T5 must list a built wheel |
| Real stdio MCP transport works at import and parse cold start | unverified | T7 |
| `wire()` can drive a typer app | unverified, likely no | cisternal documents cyclopts; T7 must decide |
| CGenFF is in every 0.1.0 wheel | verified on x86_64 Linux only | subagent inspection; tracked on `main` |

## 4. Tracks

Each track is one branch and one draft PR off `origin/main`, safe to run in parallel unless marked. Acceptance is a command that fails on the defect, per ledger B1, B3 and B4. A pass with no positive control does not count.

### T1. Licensing records (no code decision)

- Port `260824_license-provenance-verification.md` from `chore/license-provenance-verification` to `main`, correcting rows that are wrong, not missing. On that branch the `proxide_fixer` row (line 259) already exists and says OK, subprocesses only (`invoked as external subprocesses`), and it omits the `proxide-rotlib` dependency and the confind `CANONICAL_AA_NAMES` constant. The `proxide-frag` row (line 270) says the crate depends on `proxide-confind`; `crates/proxide-frag/Cargo.toml` has no such dependency.
- Update debts #1420 and #1421 to match those corrected rows. The debt store is not a file in this repo, so the accept is the ported spec text, not a debt-id edit (ledger B6).
- Carry the corrections from `audits/261102_mosaist-derivation-map.md` into those records. Its `proxide-frag` comparison replaces the line-compare this track originally asked for.
- Assess `~/projects/plegadx/vendor/proxide`, which vendors confind and rotlib. Report; do not change that repo.

Accept: with `spec=.praxia/docs/specs/260824_license-provenance-verification.md` on `main`, `rg -n 'proxide-confind dependency|invoked as external subprocesses' "$spec"` exits 1, and both `rg -n CANONICAL_AA_NAMES "$spec"` and `rg -n proxide-rotlib "$spec"` exit 0. `rg -n proxide-confind crates/proxide-frag/Cargo.toml` exits 1. Depends on #81 for the NOTICE files. `rg -l Mosaist crates/*/NOTICE` already lists confind, rotlib and fixer on `fix/stop-publishing-mosaist-ports`, so that command does not decide these wording corrections.

### T2. Remove CGenFF

- Port `14e623d` and `bb4d018`. Take from `55bdfef` only what removal needs, and report the rest.
- Find every code path that loads the file, and make it raise a named error.
- Add a `smoke_wheel.py` check that no `*cgenff*` file is present. Depends on #80.
- Audit the other bundled force fields and record a source and licence for each vendored XML under `src/proxide/assets`. `openmm_bundled` and `charmm36*.xml` need checking, and the branch's asset audit (`cdb48c1`) is the starting point. Write `src/proxide/assets/FORCEFIELD-LICENCES.tsv`, one data row per XML. Columns, tab-separated: path relative to `src/proxide/assets`; `file` or `sha256`; then a repo-relative licence file, or `DIGEST PATH` where `DIGEST` is the sha256 of the checked-in upstream licence text at `PATH`.

Accept: `git ls-files | rg -i cgenff` exits 1 (no output), and the smoke check fails when the file is put back. Force-field records: the program below exits 0. It exits 1 when a vendored `*.xml` under `src/proxide/assets` has no matching row, when a `file` target is missing, or when a `sha256` target's digest does not equal `hashlib.sha256` of the checked-in upstream text.

```python
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
```

### T3. Make the sdist installable

Per D6, restructure so the sdist builds without the GPL crates, for example by removing the workspace path dependencies on `proxide-gaff2` and `proxide-ligand-frame` from what the sdist must resolve, while source builds can still opt in. This approach is untested: spike it first and report if Cargo or maturin rules it out, and do not fall back to shipping GPL sources or dropping the sdist without asking. `smoke-sdist` in #80 is the acceptance test. The job exists on `ci/smoke-built-wheels` (`.github/workflows/publish.yml`, job `smoke-sdist`, `needs: [build-sdist]`). Its current result was not recorded in the repo. It stays a release gate. Do not weaken or remove it to get a green run.

### T4. Deeper stable-surface audit

Audit only what README Stability promises. Findings go to the debt store with a repro; fixes go in separate PRs.

- Readers: `parse_pdb`, `parse_mmcif`, `parse_pqr`, `parse_parm7`, `parse_amber_trajectory`, XTC, DCD, TRR, MDTraj and mdCATH HDF5, foldcomp.
- Known gaps: foldcomp tests never run (#2923), HDF5 parity skips (#2925), Atom37 `UNK` silence (#2920), numbering lost in `project_to_mpnn_batch` (#2928), concatenate crash (#2921), deuterium and ions (#2918).
- New: a file with only a `FE` HETATM returned zero residues and no warning in the default format, while an empty file raises. Decide and test it.
- Sweep each stable path for ledger A1 to A6: `unwrap_or`, `unwrap_or_default`, `.ok()`, `nan_to_num`, `except Exception`, and `documented-default` comments.
- Add a property or fuzz harness for the parsers. 13 hand-written inputs caused no panics, which is weak evidence. A Rust panic surfaces as `PanicException`, which is a `BaseException`.
- Triage stale debts: #2368, #1931, #1932 and #1905 look resolved by the 0.1.0 changelog. Verify each, then resolve it or say why not.
- A count, breakdown or pass rate from this audit is a finding: pre-register it with a committed `.bth.toml` first, and use `bth`, not `uv run bth`.

Accept: the command below exits 0. Each named test is the one that fails when that README Stability function's checked behaviour breaks. `write_dcd` and `TrajectoryStream` have no such test in the tree today (`rg write_dcd tests` matches only `tests/test_public_api.py`; `rg TrajectoryStream tests` matches only that file), so their nodes are part of the command and the command fails until those tests exist. Calibration is the `git rev-parse` of the three tags plus the three ground-truth files, not a planted regex.

| README Stability function | Test that fails when the behaviour breaks | Fixture |
|---|---|---|
| `parse_pdb` | `tests/io/parsing/test_real_fixtures.py::test_1crn_pdb_num_atoms_matches_independent_count` (`parse_pdb_raw_rust` calls `_proxider.parse_pdb`) | `crates/proxide-io/tests/data/1CRN.pdb`, `num_atoms == 327` |
| `parse_mmcif` | `tests/io/test_mmcif_numbering.py::test_parse_mmcif_defaults_to_auth_numbering` | inline CIF in that file; auth chain `A`, res ids `7,7,7,7,8,8,8,8` |
| `parse_pqr` | `tests/io/parsing/test_pqr.py::test_parse_pqr_basic` (`parse_pqr_rust` calls `_proxider.parse_pqr`) | `tests/data/1a00.pqr` |
| `parse_structure` | `tests/io/parsing/test_altloc_resolution.py::TestParseStructureMmcif::test_parse_structure_pdb_still_works` | `ALTLOC_PDB` in that file; `coordinates.shape[0] == 1` |
| `parse_parm7` | `tests/io/test_parm7_trajectory.py::test_parse_parm7_low_level` | parm7 written by `_write_parm7`; `n_atoms == 22` |
| `parse_amber_trajectory` | `tests/io/test_parm7_trajectory.py::test_frame0_matches_xtc_reader` | same parm7 plus `tests/data/trajectories` XTC |
| `write_pdb` | `tests/io/test_writing.py::test_pdb_element_column_is_upper_case_and_right_justified` | in-test protein; cols 77-78 are ` C` and `CL` |
| `write_mmcif` | `tests/io/test_writing.py::test_cif_type_symbol_is_upper_case` | in-test protein; output contains `ATOM 1 CL ` |
| `write_npz` | `tests/io/test_atom14_variants.py::test_npz_round_trip_keeps_mse_residue_name` | in-test MSE protein |
| `write_dcd` | `tests/io/test_write_dcd.py::test_write_dcd_round_trip_parse_dcd` (absent; must write with `write_dcd` and assert `parse_dcd` returns those coordinates) | temp file created by the test |
| `parse_xtc` | `tests/validation/test_trajectory_parity.py::test_xtc_frame_coordinates_vs_mdtraj` | `tests/data/trajectories/frame0.xtc` |
| `parse_dcd` | `tests/validation/test_trajectory_parity.py::test_dcd_frame_coordinates_vs_mdtraj` | `tests/data/trajectories/frame0.dcd` |
| `parse_trr` | `tests/validation/test_trajectory_parity.py::test_trr_frame_coordinates_vs_mdtraj` | `tests/data/trajectories/frame0.trr` |
| `read_xtc_lazy` | `tests/validation/test_xtc_reader_parity.py::test_read_xtc_lazy_matches_mdtraj` | `synthetic_xtc` fixture in that file |
| `read_xtc_parallel` | `tests/validation/test_xtc_reader_parity.py::test_read_xtc_parallel_matches_mdtraj` | `synthetic_xtc` |
| `read_xtc_ca_distogram` | `tests/validation/test_xtc_distogram_parity.py::test_read_xtc_ca_distogram_synthetic_ground_truth_and_pair_order` | `tmp_path` synthetic XTC |
| `frame_count` | `tests/validation/test_xtc_reader_parity.py::test_frame_count_matches_mdtraj` | `synthetic_xtc` |
| `n_atoms` | `tests/validation/test_xtc_reader_parity.py::test_n_atoms_matches_mdtraj` | `synthetic_xtc` |
| `TrajectoryStream` | `tests/io/streaming/test_trajectory_stream.py::test_trajectory_stream_yields_a_chunk` (absent; must iterate `TrajectoryStream` and assert one chunk with coordinates) | `tests/data/trajectories/frame0.xtc` |
| `parse_mdtraj_h5_metadata` | `tests/io/parsing/test_hdf5_integration.py::TestMdtrajH5Parser::test_parse_metadata` | `mdtraj_h5_file`; `num_frames == 10`, `num_atoms == 50` |
| `parse_mdtraj_h5_frame` | `tests/io/parsing/test_hdf5_integration.py::TestMdtrajH5Parser::test_parse_frame` | `mdtraj_h5_file`; frame 0 shape `(50, 3)` |
| `parse_mdcath_metadata` | `tests/io/parsing/test_hdf5_integration.py::TestMdcathH5Parser::test_parse_domain_metadata` | `mdcath_h5_file`; domain `1abc00` |
| `parse_mdcath_frame` | `tests/io/parsing/test_hdf5_integration.py::TestMdcathH5Parser::test_parse_frame` | `mdcath_h5_file` |
| `get_mdcath_replicas` | `tests/io/parsing/test_hdf5_integration.py::TestMdcathH5Parser::test_get_replicas` | `mdcath_h5_file`; replicas include `0` and `1` |
| `fetch_rcsb` | `tests/io/test_fetch_security.py::test_fetch_rcsb_invalid_id` | id `../../../etc/passwd`; raises with `Invalid ID` |
| `fetch_afdb` | `tests/io/test_fetch_security.py::test_fetch_afdb_invalid_id` | id `P12345/../../bad`; raises with `Invalid ID` |
| `fetch_foldcomp_database` | `tests/io/test_fetch_security.py::test_fetch_foldcomp_invalid_id` | name `bad;rm -rf /`; raises with `Invalid ID` |

```bash
set -e
git rev-parse --verify 'calibration/element-inference-defective^{commit}'
git rev-parse --verify 'calibration/element-inference-fixed-initial^{commit}'
git rev-parse --verify 'calibration/element-inference-fixed^{commit}'
test -s tests/data/element_inference_calibration/writing_d4ac812.py.txt
test -s tests/data/element_inference_calibration/projector_495ef10.py.txt
test -s tests/data/element_inference_calibration/containers_12c68e6.py.txt
uv run pytest --collect-only -q \
  tests/io/parsing/test_real_fixtures.py::test_1crn_pdb_num_atoms_matches_independent_count \
  tests/io/test_mmcif_numbering.py::test_parse_mmcif_defaults_to_auth_numbering \
  tests/io/parsing/test_pqr.py::test_parse_pqr_basic \
  tests/io/parsing/test_altloc_resolution.py::TestParseStructureMmcif::test_parse_structure_pdb_still_works \
  tests/io/test_parm7_trajectory.py::test_parse_parm7_low_level \
  tests/io/test_parm7_trajectory.py::test_frame0_matches_xtc_reader \
  tests/io/test_writing.py::test_pdb_element_column_is_upper_case_and_right_justified \
  tests/io/test_writing.py::test_cif_type_symbol_is_upper_case \
  tests/io/test_atom14_variants.py::test_npz_round_trip_keeps_mse_residue_name \
  tests/io/test_write_dcd.py::test_write_dcd_round_trip_parse_dcd \
  tests/validation/test_trajectory_parity.py::test_xtc_frame_coordinates_vs_mdtraj \
  tests/validation/test_trajectory_parity.py::test_dcd_frame_coordinates_vs_mdtraj \
  tests/validation/test_trajectory_parity.py::test_trr_frame_coordinates_vs_mdtraj \
  tests/validation/test_xtc_reader_parity.py::test_read_xtc_lazy_matches_mdtraj \
  tests/validation/test_xtc_reader_parity.py::test_read_xtc_parallel_matches_mdtraj \
  tests/validation/test_xtc_distogram_parity.py::test_read_xtc_ca_distogram_synthetic_ground_truth_and_pair_order \
  tests/validation/test_xtc_reader_parity.py::test_frame_count_matches_mdtraj \
  tests/validation/test_xtc_reader_parity.py::test_n_atoms_matches_mdtraj \
  tests/io/streaming/test_trajectory_stream.py::test_trajectory_stream_yields_a_chunk \
  tests/io/parsing/test_hdf5_integration.py::TestMdtrajH5Parser::test_parse_metadata \
  tests/io/parsing/test_hdf5_integration.py::TestMdtrajH5Parser::test_parse_frame \
  tests/io/parsing/test_hdf5_integration.py::TestMdcathH5Parser::test_parse_domain_metadata \
  tests/io/parsing/test_hdf5_integration.py::TestMdcathH5Parser::test_parse_frame \
  tests/io/parsing/test_hdf5_integration.py::TestMdcathH5Parser::test_get_replicas \
  tests/io/test_fetch_security.py::test_fetch_rcsb_invalid_id \
  tests/io/test_fetch_security.py::test_fetch_afdb_invalid_id \
  tests/io/test_fetch_security.py::test_fetch_foldcomp_invalid_id
```

### T5. Cisternal skill and plugin

Copy jaxlint's pattern. cisternal needs Python >=3.13 and proxide supports >=3.11, so it stays optional.

1. Extra: `plugin = ["cisternal>=0.1.5; python_version >= '3.13'"]`.
2. `src/proxide/cli/plugin_command.py`: forward raw argv to `plugin_app(PluginSpec(name="proxide", package="proxide", cli="proxide"))`, import cisternal lazily, and when cisternal is absent or Python is below 3.13 write this exact substring to stderr: `proxide plugin requires Python 3.13`. Register it with `allow_extra_args` and `ignore_unknown_options`.
3. `.praxia/manifest.toml` with `skills_delivered_by_plugin = true`. Extend `scripts/check_versions.py` to cover its version.
4. `agent_assets/skills/using-proxide/SKILL.md`, triggered on the wording of the mistake: hand-parsing a PDB, `atom37`, mmCIF numbering, NaN for missing atoms. Content: stable and experimental split, auth vs label numbering, `atom_mask` not NaN, unknown elements raise, the GAFF2 DEF is not bundled, licensing of optional crates. Every example must run.
5. Snapshot at `src/proxide/agent_plugin.json`, with `cisternal assets snapshot --check` in CI.
6. Add Python 3.13 to the CI test matrix. Run `proxide plugin info` from a non-editable install and assert it resolves the packaged snapshot.
7. Treat any stderr from `cisternal assets export` as failure, because it exits 0 when it drops assets.

Accept: `unzip -l` on a built wheel lists the snapshot. On Python 3.11, `proxide plugin` exits non-zero and its stderr contains `proxide plugin requires Python 3.13`. These commands still exit 0 on Python 3.11: `proxide version`, `proxide --help`, `proxide fetch --help`, `proxide info --help`, `proxide convert --help`, `proxide validate --help`, `proxide bench --help`, `proxide parameterize --help`, `proxide charges --help`.

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

Scope: the stable surface is the README Stability set and `_STABLE` in `src/proxide/__init__.py` (parsers, writers, trajectory readers, and `fetch_rcsb`, `fetch_afdb`, `fetch_foldcomp_database`). There is no `inspect` command. The summary command is `info` (`src/proxide/cli/main.py`). Tools for that surface are `info`, `validate`, `fetch`, and `convert`. Summaries and paths, never arrays. Long calls need a timeout. `fetch` writes to a caller-supplied directory. Register `PROX-IO-*` codes (#1921) so an agent can branch on them.

Accept: one parametrized test calls each stable tool (`info`, `validate`, `fetch`, `convert`) through the function, the CLI and an MCP client over real stdio, and asserts identical JSON. A guard fails when any of `fetch`, `info`, `convert`, `validate`, or `bench` has no tool (`bench` is in that guard; `src/proxide/cli/main.py` defines it beside `validate`). A guard fails when a tool other than the `bench` tool is outside `_STABLE`. `parameterize`, `charges`, and `version` are CLI commands and are outside `_STABLE`.

### T8. Packaging and supply chain

- Add `py.typed` and check a downstream type check resolves the stable API.
- Pin `test_public_api.py` to signatures, not just names.
- CI: run tests from `uv.lock`, plus one named unlocked job (#3180).
- Add `cargo-deny` and `pip-audit`, and Dependabot for actions and dependencies. Pin actions by SHA. Replace `curl | sh` for uv with `astral-sh/setup-uv`.
- The macOS and Windows jobs are labelled "Slim" and pass `--features xtc` without `--no-default-features`, so default features stay on (`.github/workflows/publish.yml`). Fix the labels and flags, and correct the CHANGELOG statement that PyPI wheels are built with the `full` and `xtc` feature sets. The claim that a smoke run found every stable feature linked is the unrecorded dry run 41491125466, not a workflow-file fact.
- Fix the README: the "Pre-release toward 0.1.0" banner, the JAX MD mentions, and "requires a Rust toolchain".
- Add `SECURITY.md`, `CONTRIBUTING.md` and `CITATION.cff`.

Owner step, outside the accept: create the GitHub release for `v0.1.0`. It is outward-facing and needs the owner's approval.

Accept: the shell below exits 0. It exits non-zero while the stale README or CHANGELOG strings remain, or while a macOS or Windows maturin `args:` line in `.github/workflows/publish.yml` contains `--features xtc` and does not contain `--no-default-features`.

```bash
set -e
if rg -q 'Pre-release toward 0\.1\.0' README.md; then exit 1; fi
if rg -q 'JAX MD' README.md; then exit 1; fi
if rg -q 'requires a Rust toolchain' README.md; then exit 1; fi
if rg -q 'built with the `full` and `xtc` feature sets' CHANGELOG.md; then exit 1; fi
python3 - <<'PY'
import pathlib, re, sys
text = pathlib.Path(".github/workflows/publish.yml").read_text()
for job in ("build-wheels-macos", "build-wheels-windows"):
    m = re.search(rf"^  {job}:\n(.*?)(?=^  \S|\Z)", text, re.M | re.S)
    if not m:
        sys.exit(1)
    args = re.findall(r"^ +args:.*$", m.group(1), re.M)
    if not args or any("--features xtc" in a and "--no-default-features" not in a for a in args):
        sys.exit(1)
PY
```

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
| Release blocked while `smoke-sdist` is red | Intended; the job exists on #80 and its current result was not recorded. T3 resolves a red run |
| MCP cold start too slow because of JAX | Tools import lazily; T7 measures it at production shapes and records it in the sidecar |
| Telemetry adds a hard dependency | Extra stays optional; T6 tests the absent state |
| Skill drifts from the API | Examples run in CI |
