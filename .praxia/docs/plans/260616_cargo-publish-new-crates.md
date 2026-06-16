---
task_id: 260616_cargo-publish-new-crates
created: 260616
status: ready
sprint_id: 260616_cargo-publish-new-crates
---

# Sprint: cargo-publish-new-crates

**Goal:** Make the 4 new Rust crates (`proxide-confind`, `proxide-rotlib`, `proxide-frag`, `proxide-parallel-rt`) publishable to crates.io, exclude `proxide-wasm`, and update the CI publish pipeline to cover the complete workspace.

**Context:** The original 7 crates have full publish metadata (added in the sprightly-kindling-pinwheel worktree, now on `main`). The 5 new crates from the `cheerful-rolling-beaver` merge are not yet publish-ready. This sprint closes that gap so a single `v*` tag push publishes the complete workspace.

---

## Backlog Items

| ID | Title | Priority | Difficulty | Depends On |
|----|-------|----------|------------|------------|
| #1402 | Add missing license to proxide-parallel-rt | P1 | quick | — |
| #1403 | Mark proxide-wasm publish = false | P1 | quick | — |
| #1405 | Fix CI README cp step (no-clobber) | P1 | quick | — |
| #1406 | Migrate new crate path deps to workspace = true | P1 | quick | — |
| #1401 | Add publish metadata to 4 new crates | P1 | standard | #1402 |
| #1404 | Update verify-crates + publish-crates CI jobs | P1 | standard | #1401 #1402 #1403 |
| #1407 | Triage 8 stale backlog branches | P2 | quick | — |
| #1408 | Write per-crate README.md files | P3 | standard | #1401–#1406 |

**Sprint scope:** #1402, #1403, #1405, #1406, #1401, #1404 (all P1). Items #1407 and #1408 deferred to follow-up.

---

## Dependency DAG

```
#1402 ──┐
#1403 ──┼──► #1401 ──► #1404
#1405   │
#1406 ──┘
```

Wave 1 (parallel, no deps): **#1402, #1403, #1405, #1406**
Wave 2 (after Wave 1): **#1401**
Wave 3 (after Wave 2): **#1404**

---

## Wave 1 — Parallel Quick Fixes

### #1402 — License on proxide-parallel-rt

**File:** `crates/proxide-parallel-rt/Cargo.toml`

Add to `[package]`:
```toml
license.workspace = true
authors.workspace = true
```

**Verification:** `cargo package -p proxide-parallel-rt --no-verify` exits 0; `cargo publish --dry-run -p proxide-parallel-rt` exits 0.

---

### #1403 — proxide-wasm publish = false

**File:** `crates/proxide-wasm/Cargo.toml`

Add to `[package]`:
```toml
publish = false
```

Confirm `proxide-wasm` does not appear in any `cargo publish -p ...` line in `.github/workflows/publish.yml`.

**Verification:** `cargo package -p proxide-wasm` exits with "cannot publish" error (confirming the flag works).

---

### #1405 — Fix README copy step (no-clobber)

**File:** `.github/workflows/publish.yml`

In both `verify-crates` and `publish-crates` jobs, change the copy step from:
```bash
cp README.md crates/$crate/README.md
```
to:
```bash
cp -n README.md crates/$crate/README.md
```
(`-n` = no-clobber: skips if destination already exists, preserving proxide-rotlib and proxide-wasm READMEs).

**Verification:** Confirm `crates/proxide-rotlib/README.md` and `crates/proxide-wasm/README.md` contents are unchanged after a local dry-run.

---

### #1406 — Migrate relative path deps to workspace = true

**Files:** `crates/proxide-confind/Cargo.toml`, `crates/proxide-rotlib/Cargo.toml`, `crates/proxide-frag/Cargo.toml`, `crates/proxide-wasm/Cargo.toml`

For each internal dep using `{ path = "../proxide-X" }`, change to `{ workspace = true }`. The workspace dep entry already carries `version = "0.1.0-alpha.7"` so cargo publish will resolve correctly.

**Examples:**
```toml
# Before
proxide-core = { path = "../proxide-core" }
proxide-geometry = { path = "../proxide-geometry" }

# After
proxide-core.workspace = true
proxide-geometry.workspace = true
```

For deps that use `default-features = false` or specific features (e.g. proxide-wasm), keep the path temporarily or add a version field: `proxide-core = { path = "../proxide-core", version = "0.1.0-alpha.7", default-features = false }`.

Note: `proxide-rotlib` uses `prost`, `zstd`, `clap`, `tracing`, `tracing-subscriber` which are NOT in workspace.dependencies — keep those as direct version deps.

**Verification:** `cargo check --workspace` exits 0. `cargo publish --dry-run -p proxide-rotlib` exits 0.

---

## Wave 2 — Publish Metadata (#1401)

**Files:** `crates/proxide-confind/Cargo.toml`, `crates/proxide-rotlib/Cargo.toml`, `crates/proxide-frag/Cargo.toml`, `crates/proxide-parallel-rt/Cargo.toml`

Add to `[package]` in each:
```toml
repository.workspace = true
homepage.workspace = true
readme.workspace = true   # or omit if per-crate README exists (rotlib has one)
```

Add `categories` (must be crates.io whitelist slugs):
- `proxide-confind`: `categories = ["science", "algorithms"]`
- `proxide-rotlib`: `categories = ["science", "algorithms"]`
- `proxide-frag`: `categories = ["science", "algorithms"]`
- `proxide-parallel-rt`: `categories = ["concurrency"]`

Add `[package.metadata.docs.rs]` per crate:
- `proxide-confind`: `features = ["serde"]` (avoids wasm32 target-dep complications on docs.rs)
- `proxide-rotlib`: `features = ["serde"]`
- `proxide-frag`: `all-features = true` (no native/net deps)
- `proxide-parallel-rt`: `all-features = true`

Note: `proxide-rotlib` already has a per-crate `README.md` — set `readme = "README.md"` directly (not workspace) so the CI no-clobber fix doesn't interfere.

**Verification:** `cargo publish --dry-run -p <crate>` exits 0 for all 4 crates (run after Wave 1 is complete).

---

## Wave 3 — CI Job Update (#1404)

**File:** `.github/workflows/publish.yml`

Update `verify-crates` (dry-run) and `publish-crates` jobs to include new crates.

**New publish order (complete workspace):**

```
Tier 1 (no internal deps):
  cargo publish -p proxide-units
  sleep 30
  cargo publish -p proxide-core
  cargo publish -p proxide-parallel-rt   # ← NEW (no internal deps)
  sleep 30

Tier 2:
  cargo publish -p proxide-geometry
  sleep 30

Tier 3 (deps: core + geometry):
  cargo publish -p proxide-gaff
  sleep 30
  cargo publish -p proxide-io
  cargo publish -p proxide-rotlib        # ← NEW (deps: core + geometry)
  cargo publish -p proxide-frag          # ← NEW (no internal deps on native; parallel-rt via wasm32 target only)
  sleep 30

Tier 4:
  cargo publish -p proxide-physics
  cargo publish -p proxide-confind       # ← NEW (deps: core + geometry + io + rotlib)
  sleep 30

Tier 5:
  cargo publish -p proxide_rs
  sleep 30

Tier 6:
  cargo publish -p proxide_fixer
```

Update the `verify-crates` README copy loop to include all new crates:
```bash
for crate in proxide-units proxide-core proxide-parallel-rt proxide-geometry \
             proxide-gaff proxide-io proxide-rotlib proxide-frag \
             proxide-physics proxide-confind proxide_rs proxide_fixer; do
  cp -n README.md crates/$crate/README.md
done
```

**Verification:** `cargo publish --dry-run` passes for all 11 crates in order (gated by `verify-crates` job running clean on CI).

---

## Success Criteria

- [ ] `cargo check --workspace` exits 0
- [ ] `cargo publish --dry-run -p <crate>` exits 0 for: `proxide-parallel-rt`, `proxide-rotlib`, `proxide-frag`, `proxide-confind`
- [ ] `cargo package -p proxide-wasm` exits with publish-blocked error
- [ ] `crates/proxide-rotlib/README.md` is unchanged from its committed content after the CI copy step runs
- [ ] `.github/workflows/publish.yml` verify-crates dry-run covers all 11 publishable crates
- [ ] `git push && git tag v0.1.0-alpha.9 && git push --tags` triggers a full clean CI run

---

## Deferred (Not in This Sprint)

- **#1407** Triage 8 stale backlog branches — P2, standalone audit, not blocking publish
- **#1408** Per-crate README.md files — P3, nice-to-have; CI copy workaround is sufficient for initial publish
