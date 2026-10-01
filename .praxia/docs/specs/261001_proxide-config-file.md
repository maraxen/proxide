---
title: 'proxide configuration: layered config file for machine-specific knobs'
description: One resolver (explicit > PROXIDE_* env > [tool.proxide] > ~/.config/proxide/config.toml > default) for threads, XTC offsets dir, fetch cache/mirrors, tool paths and dev fixtures; science parameters and secrets deliberately excluded
status: draft
task_id: 261001_proxide-config
date: '261001'
backlog_ids: ''
adversarial_review: ''
---
# proxide configuration: layered config file for machine-specific knobs

## 1. Problem

proxide's machine- and site-specific settings are spread across ad-hoc environment variables,
hard-coded constants and per-call defaults, with no single place to set them and no way to see what
is in effect. Inventory on `main` (`d4ac812`, 2026-10-01):

**Environment variables (all in Rust; the Python package reads none):**

| variable | kind | site |
|---|---|---|
| `PDBFIXER_EXEC` | runtime: external tool path, else `PATH` search | `crates/proxide_fixer/src/solvate.rs:466` |
| `MODELLER_KEY` | runtime: license **secret** | `crates/proxide_fixer/src/loop_model.rs:171` |
| `ROTLIB_PATH` | test-only (9 sites after `#[cfg(test)]` at `mutate.rs:263`), fallback hard-coded to `/home/marielle/repos/mosaist/testfiles/rotlib.bin` | `crates/proxide_fixer/src/mutate.rs:401-1003`, `proxide-rotlib/tests`, `proxide-confind/tests` |
| `PDB_PATH`, `DC7_PDB_PATH`, `PDB_2ZTA_PATH`, `RLIB`, `PROXIDE_ROTLIB_PB` | test-only fixture paths | `proxide-rotlib/tests/helpers.rs`, `proxide-confind/tests/common/mod.rs`, `test_drift_loadpb_small_pdb.rs` |
| `USALIGN_REPO` | test-only, fallback `~/repos/USalign` | `crates/proxide-tmalign/src/structure.rs:110` (test module from :101) |
| `PROTOC` | build time | `crates/proxide-core/build.rs` |
| `RAYON_NUM_THREADS` | implicit: read by rayon itself | wherever rayon's global pool is used |

**Hard-coded knobs:**

| knob | today | site |
|---|---|---|
| thread count | **two inconsistent defaults**: `proxide-parallel-rt` = 1 (`static NUM_THREADS = 1`, used by confind, fasta, wasm), while `read_frames_parallel` and other `into_par` paths use rayon's global pool = all node cores, ignoring the SLURM allocation; Python cannot set either | `proxide-parallel-rt/src/lib.rs:3`, `proxide-io/src/formats/xtc.rs:634` |
| download location | `output_dir="."` default; repeats re-download | `src/proxide/io/fetching.py:6-47` |
| endpoints | RCSB / AFDB / mdCATH / foldcomp base URLs are constants; AFDB `version=4` default | `proxide-io/src/io/fetching.rs:13-16, 227` |
| retry policy | `max_retries = 3`, 1 s initial backoff | `fetching.rs:178-179` |
| XTC offsets sidecar | always `<xtc>.offsets` next to the trajectory | `proxide-io/src/formats/xtc.rs:115-290` |

## 2. Principles

1. **Config is for where and how, never for what result.** Locations, resources, endpoints and tool paths
   vary by machine. Scientific parameters (force fields, GB parameter sets, protonation, strides, cutoffs)
   must stay explicit arguments or pinned run configs, so a result never depends on which machine's
   config file was read.
2. **No secrets in config files.** `MODELLER_KEY` stays an env var or keyring; config may only name which
   env var holds it.
3. **Same layering as `isochore.traj_cache`** (the reference implementation, `cache_root_source`), so one
   mental model covers the ecosystem. First match wins:
   1. explicit argument (Rust param / Python kwarg / CLI flag);
   2. `PROXIDE_<SECTION>_<KEY>` env var (e.g. `PROXIDE_PARALLEL_NUM_THREADS`), if set at all;
   3. `[tool.proxide.<section>]` in the nearest `pyproject.toml` at or above the cwd;
   4. `${XDG_CONFIG_HOME:-~/.config}/proxide/config.toml` (per machine);
   5. the built-in default (today's behaviour unless noted).
   Values expand `~` and `$VARS`; relative paths are relative to the file that set them; a malformed file
   is an error naming the file and key; every resolved value can be reported with its source layer.
4. **Derived data never lands in someone else's data directories** (global CLAUDE.md rule): offsets
   sidecars and downloads get configurable homes.

## 3. Knobs in scope

| section.key | type | default | replaces | priority |
|---|---|---|---|---|
| `parallel.num_threads` | int ≥ 1 | `$SLURM_CPUS_PER_TASK` if set, else `available_parallelism()` | `parallel-rt` default 1 **and** rayon's all-cores pool; `RAYON_NUM_THREADS` honoured as the env layer's alias | **P1** |
| `xtc.offsets_dir` | path or unset | unset = next to the XTC (today) | always-adjacent sidecar | **P1** (blocks isochore spec `260930_parallel-xtc-read-via-proxide` T1/O1) |
| `xtc.import_mdanalysis_offsets` | bool | `true` (today) | — | P1 |
| `fetch.cache_dir` | path or unset | unset = today's `output_dir="."` | cwd downloads | P2 |
| `fetch.skip_existing` | bool | `true` when `cache_dir` is set | re-downloads | P2 |
| `fetch.rcsb_url`, `fetch.afdb_url`, `fetch.mdcath_url`, `fetch.foldcomp_url` | URL | today's constants | constants | P3 |
| `fetch.retries`, `fetch.backoff_s`, `fetch.timeout_s` | int / float | 3, 1.0, (none today → 60) | constants | P3 |
| `fetch.afdb_version` | int | 4 | kwarg default | P3 |
| `tools.pdbfixer` | path | unset → `PDBFIXER_EXEC` → `PATH` | `PDBFIXER_EXEC` (kept as env layer) | P3 |
| `tools.modeller_key_env` | env var name | `"MODELLER_KEY"` | — (value never in config) | P3 |
| `dev.fixtures.<name>` | path | unset → test **skips** | test env vars + `/home/marielle/...` fallbacks | P4 |

**Explicitly out of scope:** `PROTOC` (build time), the formatter LRU capacity (a per-call argument),
temp files (`TMPDIR` already governs them), and every scientific parameter (Principle 1).

## 4. Design

- **New crate `proxide-config`** (deps: `toml`, `serde`, `dirs`; no I/O beyond reading config files):
  `Config::load(cwd) -> Result<Config, ConfigError>` builds the layered view once; typed getters
  (`num_threads()`, `offsets_dir()`, `fetch()`, ...) each return `(value, Source)`.
  `Source = Explicit | Env(name) | Project(path) | User(path) | Default`. Process-global, lazily
  initialised (`OnceLock`), with `Config::reload()` for tests.
- **wasm:** `cfg(target_arch = "wasm32")` builds use defaults plus explicit arguments only (no
  filesystem, no env).
- **Threads:** at first use, initialise rayon's global pool and `parallel-rt` from `parallel.num_threads`,
  so every parallel path agrees. Python: `proxide.set_num_threads(n)` / `proxide.get_num_threads()`.
- **Python:** `proxide.config.show()` returns `{key: (value, source)}`; the `proxide config show` CLI
  prints it. Python reads the Rust-resolved view (one implementation, not two).
- **Unknown keys:** warn (`tracing::warn!`) naming the file, so typos are visible. **Wrong types:** error.

## 5. Tests (red before green)

1. Layer precedence for each layer, including "env set to empty or `none`" where a key accepts unset.
2. `~` / `$VAR` expansion; relative paths resolve against the defining file.
3. Malformed TOML → error naming the file; wrong type → error naming the key; unknown key → warning.
4. Threads: with `SLURM_CPUS_PER_TASK=2` and no config, both rayon and `parallel-rt` report 2; an explicit
   `set_num_threads(3)` wins. Negative control: without the fix, rayon reports all cores.
5. `offsets_dir`: the sidecar lands there and is found on reopen; nothing is written next to the XTC;
   unset keeps today's location (existing XTC tests pass unchanged).
6. `fetch.cache_dir`: a second fetch of the same ID makes no network call (mock transport).
7. Dev fixtures: a test whose fixture is unset **skips** with a message; no `/home/marielle` path remains
   in the tree (`rg '/home/marielle' crates src` returns nothing).
8. `config show` lists every key with its source.

## 6. Rollout

1. **T1** `proxide-config` crate + precedence/expansion/error tests (§5.1–5.3).
2. **T2** threads (§5.4) — fixes the live oversubscription on cluster jobs.
3. **T3** `xtc.offsets_dir` (+ import flag) (§5.5) — unblocks isochore's parallel-reader spec.
4. **T4** `fetch.*` (§5.6).
5. **T5** `tools.*`.
6. **T6** `dev.fixtures.*` and removal of hard-coded user paths (§5.7).
7. **T7** `proxide config show` + Python `config.show()`, README section with an example `config.toml`.

Each task lands with today's behaviour as the default, so nothing changes until a config value is set,
except T2, which deliberately changes the thread default (documented in the changelog).

## 7. Open questions

- **Q1** T2 changes default behaviour (rayon no longer uses every node core under SLURM). Acceptable as a
  fix, or keep the old default behind a flag for one release?
- **Q2** Should `xtc.offsets_dir` default to a subdirectory of the `traj_cache` root when that is
  configured, or stay independent? (Recommendation: independent; set it explicitly to point inside the
  traj_cache root. No cross-project coupling.)
- **Q3** Env var naming: `PROXIDE_<SECTION>_<KEY>` for everything, keeping `PDBFIXER_EXEC` and
  `RAYON_NUM_THREADS` as aliases?
