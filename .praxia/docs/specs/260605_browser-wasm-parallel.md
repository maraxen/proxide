---
title: Browser WASM Parallelism via orx-parallel + build-std atomics
slug: 260605_browser-wasm-parallel
status: revised — post oracle critique; ready for sprint composition
created: 260605
revised: 260605
---

# Browser WASM Parallelism for proxide

## Summary

Enable real multi-threaded parallel execution in browser WASM (wasm32-unknown-unknown) for
the orx-parallel call sites in proxide-confind, proxide-io, and proxide-master. The mechanism
is: nightly Rust + `build-std` + `-C target-feature=+atomics,+bulk-memory`, which rebuilds
`std` so that `std::thread::scope` + `s.spawn` use SharedArrayBuffer-backed Web Workers.
No rayon, no wasm-bindgen-rayon — orx-parallel's existing spawn path works natively once
std is rebuilt with atomic thread support.

A new `proxide-parallel-rt` crate stores the JS-injected thread count in an `AtomicUsize`
so all parallel crates can call `.num_threads(proxide_parallel_rt::num_threads())` without
depending on `proxide-wasm`.

**Revision note (post oracle critique):** The prior draft proposed `proxide-parallel-rt` as
an `OnceLock<rayon::ThreadPool>` bridge via wasm-bindgen-rayon. Oracle and librarian research
confirmed orx-parallel 2.4.0 uses `std::thread::scope` directly (no rayon at runtime), so
the rayon bridge is unnecessary and the architecture is simpler.

## Research findings (confidence: high)

| Question | Answer | Source |
|----------|--------|--------|
| Does orx-parallel use rayon internally for spawning? | No — uses `std::thread::scope` + `s.spawn` exclusively | orx-parallel 2.4.0 `runner/parallel_runner_compute/reduce.rs:22` |
| Does `with_runner<Q>()` override the spawn mechanism? | No — controls scheduling policy only; spawn site is not in the trait | orx-parallel 2.4.0 `par_iter.rs:259`, `runner/parallel_runner.rs` |
| Can `std::thread::scope` use Web Workers on wasm32? | Yes, when std is rebuilt with `+atomics,+bulk-memory` via nightly + build-std | wasm-bindgen-rayon README, web.dev/webassembly-threads |
| Does build-std + atomics enable real threads without a bridge? | Yes — std::thread::scope becomes SharedArrayBuffer-backed | Proven path via rayon/wasm-bindgen-rayon mechanism; applies to any std thread use |
| Is wasm-bindgen-rayon needed? | No — that's the rayon-specific glue; orx-parallel bypasses rayon entirely | orx-parallel source; no rayon dep in runner paths |
| Does `num_threads(1)` give sequential execution on stock wasm32? | No — spawns exactly 1 thread, which panics on stock wasm32 | `fixed_chunk_runner/num_threads.rs`; `do_spawn_new` fires once at num_spawned=0 |
| Requires nightly? | Yes — `build-std` is unstable-only | rust docs |

## Architecture

### New crate: `proxide-parallel-rt`

A tiny runtime crate that stores the JS-injected thread count. All parallel crates depend
on it (wasm-gated) to read the configured count without depending on `proxide-wasm`.

```
proxide-confind  ──┐
proxide-io       ──┤──► proxide-parallel-rt  (AtomicUsize NUM_THREADS)
proxide-master   ──┘

proxide-wasm ──► proxide-parallel-rt  (calls set_num_threads from JS)
```

### Dependency additions

**`proxide-parallel-rt/Cargo.toml`** (new crate, no external deps):
```toml
[package]
name = "proxide-parallel-rt"
edition.workspace = true
version.workspace = true
```

**`Cargo.toml`** (workspace) — add member:
```toml
members = [
    # ... existing ...
    "crates/proxide-parallel-rt",
]
```

**Each parallel crate** (`proxide-confind`, `proxide-io`, `proxide-master`) — add wasm-gated dep:
```toml
[target.'cfg(target_arch = "wasm32")'.dependencies]
proxide-parallel-rt = { path = "../proxide-parallel-rt" }
```

**`proxide-wasm/Cargo.toml`** — add dep (not target-gated, it's already wasm-only):
```toml
[dependencies]
proxide-parallel-rt = { path = "../proxide-parallel-rt" }
```

### proxide-parallel-rt/src/lib.rs

```rust
use std::sync::atomic::{AtomicUsize, Ordering};

static NUM_THREADS: AtomicUsize = AtomicUsize::new(1);

pub fn set_num_threads(n: usize) {
    NUM_THREADS.store(n.max(1), Ordering::Relaxed);
}

pub fn num_threads() -> usize {
    NUM_THREADS.load(Ordering::Relaxed)
}
```

No `cfg` guards needed — this compiles and is correct on all targets. Native builds just
never call `set_num_threads`, and the AtomicUsize is zero-cost when not read.

### proxide-wasm/src/lib.rs additions

```rust
use wasm_bindgen::prelude::*;

/// Set the thread count used by all parallel iterators.
/// Call this before any parallel operations with `navigator.hardwareConcurrency`.
#[wasm_bindgen]
pub fn init_parallel(num_threads: usize) {
    proxide_parallel_rt::set_num_threads(num_threads);
}
```

Note: `init_parallel` is now synchronous (no `async`, no `.await`) — we are not initializing
a thread pool, just storing a count. Web Worker lifecycle is managed by the browser runtime
via the rebuilt std.

### Call-site transformation (all parallel crates)

Every `#[cfg(target_arch = "wasm32")] let par = par.num_threads(1);` becomes:

```rust
// Before (current — buggy: num_threads(1) still spawns 1 thread and panics on stock wasm32):
#[cfg(target_arch = "wasm32")]
let par = par.num_threads(1);

// After (correct — injects JS-provided thread count for the atomics wasm build):
#[cfg(target_arch = "wasm32")]
let par = par.num_threads(proxide_parallel_rt::num_threads());
```

The default `NUM_THREADS` value is 1. On non-atomics wasm builds (stock wasm32 without
build-std), even 1 thread will panic — but that is an unsupported build configuration;
the `.cargo/config.toml` enforces `+atomics,+bulk-memory` for all wasm32 targets.

**Call-site inventory (10 sites across 5 files):**

| File | Lines | Description |
|------|-------|-------------|
| `crates/proxide-confind/src/confind.rs` | ~104 | `cache_all()` parallel residue caching |
| `crates/proxide-confind/src/confind.rs` | ~151 | `contacts()` Phase A residue caching |
| `crates/proxide-confind/src/parallel.rs` | ~147 | Phase B1 pair enumeration |
| `crates/proxide-confind/src/parallel.rs` | ~160 | Phase B1 contact-degree computation |
| `crates/proxide-confind/src/parallel.rs` | ~211 | Phase C freedom computation |
| `crates/proxide-io/src/formatters/full.rs` | ~81 | Full formatter par iter #1 |
| `crates/proxide-io/src/formatters/full.rs` | ~92 | Full formatter par iter #2 |
| `crates/proxide-io/src/formats/fasta.rs` | ~68 | FASTA parallel records |
| `crates/proxide-master/src/search.rs` | ~46 | Search flat_map |
| `crates/proxide-master/src/search.rs` | ~119 | Search prefiltered flat_map |

Note: `proxide-io` parallelism is feature-gated (`parallel = ["dep:orx-parallel"]`). The
`proxide-parallel-rt` dep must be added under the same feature gate:
```toml
[target.'cfg(target_arch = "wasm32")'.dependencies]
proxide-parallel-rt = { path = "../proxide-parallel-rt", optional = true }

[features]
parallel = ["dep:orx-parallel", "dep:proxide-parallel-rt"]
```

### Build configuration

**`.cargo/config.toml`** (project root, new file):
```toml
[target.wasm32-unknown-unknown]
rustflags = [
    "-C", "target-feature=+atomics,+bulk-memory",
    "-C", "link-arg=--shared-memory",
    "-C", "link-arg=--max-memory=1073741824",
    "-C", "link-arg=--import-memory",
    "-C", "link-arg=--export=__wasm_init_tls",
    "-C", "link-arg=--export=__tls_size",
    "-C", "link-arg=--export=__tls_align",
    "-C", "link-arg=--export=__tls_base",
]

[unstable]
build-std = ["panic_abort", "std"]
```

**Toolchain pinning** — do NOT add `rust-toolchain.toml` at the workspace root; this would
force nightly on all native crate builds. Instead, scope nightly to the wasm build command:
```bash
# CI / local wasm build:
rustup run nightly-2025-11-15 wasm-pack build --target web --release -p proxide-wasm

# Or set toolchain in the wasm-pack script / Makefile target only
```
If a lockfile is needed for reproducibility, add `rust-toolchain.toml` inside
`crates/proxide-wasm/` and always build that crate from that subdirectory.

**Build command:**
```bash
cd crates/proxide-wasm
wasm-pack build --target web --release
```

### JS integration

```js
import init, { init_parallel } from './pkg/proxide_wasm.js';

await init();
init_parallel(navigator.hardwareConcurrency);
// all par() calls now run on Web Workers via std::thread::scope + SharedArrayBuffer
```

Note: `init_parallel` is now a synchronous function (no `await` needed).

**Required HTTP headers (SharedArrayBuffer requires cross-origin isolation):**
```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

## Open questions / risks

1. **`std::thread::scope` on wasm32 with atomics — empirical validation needed.**
   The mechanism is well-documented for rayon (via wasm-bindgen-rayon) but no published
   example exists for orx-parallel specifically. A proof-of-concept spike (any par() call
   running on ≥2 Web Workers in a browser) is a required precondition before the full
   implementation sprint. If the spike fails, the fallback is keeping `num_threads(1)` as
   the wasm path and abandoning browser parallelism.

2. **WASI gate narrowing.** The current gate `#[cfg(target_arch = "wasm32")]` also covers
   WASI targets. After this change, WASI builds will use `proxide_parallel_rt::num_threads()`
   (defaulting to 1 unless set). For `wasm32-wasip1-threads`, `std::thread::scope` works
   natively, so this is correct behavior — but should be verified with a WASI compile check.

3. **nightly requirement for wasm build only.** Per the corrected toolchain guidance above:
   no root `rust-toolchain.toml`; nightly scoped to the wasm build command. This is
   straightforward in CI (separate job) but needs explicit documentation.

4. **Send + Sync audit on parallel closures.** Removing the `num_threads(1)` gates and
   replacing with real thread counts exposes the closures to multi-threading enforcement.
   Captured types (`Arc<RotamerLibrary>`, `DashMap<...>`, `&HashSet<...>`) are expected to
   be `Send + Sync` — needs a compile test targeting wasm32 with atomics to confirm.

5. **COOP/COEP cross-origin isolation.** COOP/COEP headers required for SharedArrayBuffer
   may break OAuth flows or cross-origin iframes on the same page. Deployment concern only.

6. **`available_parallelism()` returns `Err` on wasm32.** orx-parallel calls this at runner
   construction; on wasm32 it hits the `debug_assert!(false, ...)` path and falls back to
   an 8-thread cap. With `proxide_parallel_rt::num_threads()` injected via `num_threads()`,
   this fallback is bypassed — the explicit count wins. But debug builds will emit the
   assert noise in the console; this is cosmetic.

## Acceptance criteria

- [ ] A proof-of-concept spike: `cache_all` on a 100-residue structure runs on ≥2 Web Workers
      in a browser (confirm via browser DevTools timeline or Worker thread count)
- [ ] `wasm-pack build --target web -p proxide-wasm` (from `crates/proxide-wasm/`) succeeds
      on nightly-2025-11-15 + build-std
- [ ] `init_parallel(4)` stores 4 in NUM_THREADS and subsequent `par()` calls observe it
- [ ] `cargo test -p proxide-confind` (native) still passes — no regression from call-site change
- [ ] `cargo check --target wasm32-unknown-unknown -p proxide-confind` passes Send/Sync check

## Backlog items

- [ ] Empirical spike: prove `std::thread::scope` routes to Web Workers on wasm32+atomics
- [ ] Create `proxide-parallel-rt` crate (AtomicUsize NUM_THREADS; no external deps)
- [ ] Add `rust-toolchain.toml` and `.cargo/config.toml` for wasm build (no root toolchain)
- [ ] Update 10 parallel call sites: `num_threads(1)` → `num_threads(proxide_parallel_rt::num_threads())`
- [ ] Wire `proxide-parallel-rt` dep into confind/io/master (wasm-gated; io under feature flag)
- [ ] Expose `init_parallel(n)` (sync) in proxide-wasm
- [ ] WASI gate narrowing: `target_arch="wasm32"` → `all(wasm32, target_os="unknown")` (quick win, independent)
