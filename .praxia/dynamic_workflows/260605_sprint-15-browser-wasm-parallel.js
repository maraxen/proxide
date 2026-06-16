// Sprint 15 runner — emitted by `praxia dw emit-sprint`
// Source: .praxia/sprint_plans/sprint_plan.toml
// Regenerate: praxia dw emit-sprint sprint_plan.toml
// task_id: 260605_sprint-15-browser-wasm-parallel   sprint_id: 15
//
// RACE SAFETY (memory: parallel fixers race on git-status scope checks in praxia):
//   the writing chain (E,F,G) runs STRICTLY SEQUENTIAL —
//   exactly one fixer touches the working tree at a time. Only the read-only
//   research/concurrent tracks (A,B,C,D) run concurrently.

export const meta = {
  name: "260605_sprint-15-browser-wasm-parallel",
  description: "Browser WASM parallelism for proxide: WASI gate narrowing (quick win), spike to validate std::thread::scope on wasm32+atomics, then proxide-parallel-rt crate + build config + dep wiring + call-site updates + init_parallel JS exposure.",
  phases: [
    { title: "Track E — Wire proxide-parallel-rt wasm-gated dep into confind, io, master (#1174)" },
    { title: "Track F — Update 10 parallel gates: num_threads(1) -> num_threads(proxide_parallel_rt::num_threads()) (#1173)" },
    { title: "Track G — Expose init_parallel(n) synchronous wasm_bindgen fn in proxide-wasm (#1175)" },
    { title: "Track A — WASI gate narrowing: #[cfg(wasm32)] -> #[cfg(all(wasm32, target_os=\"unknown\"))] (#1176)" },
    { title: "Track B — Spike: verify std::thread::scope runs on Web Workers with wasm32+atomics (#1170)" },
    { title: "Track C — Create proxide-parallel-rt crate (AtomicUsize thread-count registry) (#1171)" },
    { title: "Track D — Add .cargo/config.toml wasm build flags; no root rust-toolchain.toml (#1172)" },
  ],
};

const TASK_ID = "260605_sprint-15-browser-wasm-parallel";
const MAX_FIX_RETRIES = 2;

function extractVerdict(text) {
  const m = String(text ?? "").match(/verdict:\s*([a-z_]+)/i);
  return m ? m[1].toLowerCase() : "advance";
}

const VERDICT_SCHEMA = {
  type: "object",
  additionalProperties: false,
  required: ["item_id", "verdict", "summary"],
  properties: {
    item_id: { type: "string" },
    verdict: { type: "string", enum: ["PASS", "NEEDS_WORK", "FAIL"] },
    summary: { type: "string" },
    issues: {
      type: "array",
      items: {
        type: "object",
        additionalProperties: false,
        required: ["where", "problem", "fix"],
        properties: {
          where: { type: "string" },
          problem: { type: "string" },
          fix: { type: "string" },
        },
      },
    },
  },
};

const RESEARCH_SCHEMA = {
  type: "object",
  additionalProperties: false,
  required: ["surfaces", "recommendation"],
  properties: {
    surfaces: {
      type: "array",
      items: {
        type: "object",
        additionalProperties: false,
        required: ["surface", "rules_location", "frontmatter", "settings_triggers", "distribution"],
        properties: {
          surface: { type: "string" },
          rules_location: { type: "string" },
          frontmatter: { type: "string" },
          settings_triggers: { type: "string" },
          distribution: { type: "string" },
          confidence: { type: "string" },
        },
      },
    },
    cross_surface_differences: { type: "array", items: { type: "string" } },
    recommendation: { type: "string" },
    open_questions: { type: "array", items: { type: "string" } },
  },
};

// Shared context for the writing tracks (from recon, task 260605_sprint-15-browser-wasm-parallel).
const EMITTER_CTX = `Spec: .praxia/docs/specs/260605_browser-wasm-parallel.md\ntask_id: 260605_browser-wasm-parallel\nBranch: worktree-orx-parallel-wasm (clean; 288 tests green)\n\nCWD: Your working directory is /home/marielle/projects/proxide/.claude/worktrees/orx-parallel-wasm\nThis IS the project root. Do NOT cd to /home/marielle/projects/proxide. Run all commands (cargo, git, grep) from cwd directly — no cd prefix needed.\n\nMECHANISM SUMMARY:\norx-parallel 2.4.0 spawns via std::thread::scope internally. With nightly + build-std +\n-C target-feature=+atomics,+bulk-memory, std is rebuilt so std::thread::scope uses\nSharedArrayBuffer-backed Web Workers on wasm32-unknown-unknown. No rayon, no wasm-bindgen-rayon.\n\nJS calls init_parallel(navigator.hardwareConcurrency) (synchronous) -> stores n in\nproxide-parallel-rt AtomicUsize -> all par() sites call .num_threads(proxide_parallel_rt::num_threads()).\n\nTrack A (WASI narrowing) and Track B (spike) are independent and run concurrently.\nTracks C and D depend on B passing; they run concurrently after B.\nTrack E depends on C+D. Track F depends on A+E. Track G depends on C+F.\n`;

// ---- per-track stage helpers ---------------------------------------------
const fixer = (prompt, label, phaseName) =>
  agent(`${prompt}\n\nWhen done, end your message with 'verdict: done' on its own line.`, {
    agentType: "fixer",
    label,
    phase: phaseName,
  });

const reviewer = (itemId, prompt, label, phaseName) =>
  agent(prompt, { agentType: "reviewer", label, phase: phaseName, schema: VERDICT_SCHEMA });

// Sequential implement->review with bounded NEEDS_WORK repair cycles.
async function track(itemId, phaseName, fixerPrompt, reviewerPrompt) {
  log(`[${itemId}] implement`);
  await fixer(fixerPrompt, `fix:${itemId}`, phaseName);
  let verdict = await reviewer(itemId, reviewerPrompt, `review:${itemId}`, phaseName);
  for (let retry = 0; retry < MAX_FIX_RETRIES && verdict && verdict.verdict === "NEEDS_WORK"; retry++) {
    log(`[${itemId}] NEEDS_WORK — repair cycle ${retry + 1}/${MAX_FIX_RETRIES}`);
    const issues = (verdict.issues || [])
      .map((i) => `- ${i.where}: ${i.problem} -> ${i.fix}`)
      .join("\n");
    await fixer(
      `${fixerPrompt}\n\nA reviewer found issues — fix exactly these, nothing else:\n${issues}`,
      `fix:${itemId}:repair:${retry}`,
      phaseName
    );
    verdict = await reviewer(itemId, reviewerPrompt, `review:${itemId}:re:${retry}`, phaseName);
  }
  return verdict;
}

// ===== TRACK E — Track E — Wire proxide-parallel-rt wasm-gated dep into confind, io, master (#1174) =========================
const trackE = () =>
  track(
    "1174",
    "Track E — Wire proxide-parallel-rt wasm-gated dep into confind, io, master (#1174)",
    `task_id: ${TASK_ID}. task_id: 260605_browser-wasm-parallel | track: E | backlog: #1174\n\nPREREQUISITES: Track C (#1171, crate created) and Track D (#1172, config.toml added) both done.\n\n## Goal\nAdd proxide-parallel-rt as a wasm-gated dependency to the three parallel crates.\n\n### crates/proxide-confind/Cargo.toml — add section:\n[target.'cfg(target_arch = "wasm32")'.dependencies]\nproxide-parallel-rt = { path = "../proxide-parallel-rt" }\n\n### crates/proxide-master/Cargo.toml — add same section:\n[target.'cfg(target_arch = "wasm32")'.dependencies]\nproxide-parallel-rt = { path = "../proxide-parallel-rt" }\n\n### crates/proxide-io/Cargo.toml — add as optional dep tied to the parallel feature:\nIn [dependencies], add:\n  proxide-parallel-rt = { path = "../proxide-parallel-rt", optional = true }\nIn [features], find the existing \`parallel = [...]\` line and append "dep:proxide-parallel-rt":\n  parallel = ["dep:orx-parallel", "dep:proxide-parallel-rt"]\n\n## Acceptance criteria\n1. cargo build -p proxide-confind -p proxide-master exits 0\n2. cargo build -p proxide-io exits 0\n3. cargo build -p proxide-io --features parallel exits 0\n4. cargo check --target wasm32-unknown-unknown -p proxide-confind exits 0\n5. cargo test --workspace exits 0\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. task_id: 260605_browser-wasm-parallel | track: E | backlog: #1174\n\n## Verification steps\n\ngrep -A3 "cfg(target_arch" crates/proxide-confind/Cargo.toml  ->  shows proxide-parallel-rt\ngrep -A3 "cfg(target_arch" crates/proxide-master/Cargo.toml  ->  shows proxide-parallel-rt\ngrep "proxide-parallel-rt" crates/proxide-io/Cargo.toml  ->  appears as optional AND in parallel feature\ngrep "parallel" crates/proxide-io/Cargo.toml  ->  includes "dep:proxide-parallel-rt"\ncargo build -p proxide-confind -p proxide-master  ->  exits 0\ncargo build -p proxide-io --features parallel  ->  exits 0\ncargo build -p proxide-io --no-default-features  ->  exits 0 (proxide-parallel-rt absent)\ncargo test --workspace  ->  exits 0\n`,
  );

// ===== TRACK F — Track F — Update 10 parallel gates: num_threads(1) -> num_threads(proxide_parallel_rt::num_threads()) (#1173) =========================
const trackF = () =>
  track(
    "1173",
    "Track F — Update 10 parallel gates: num_threads(1) -> num_threads(proxide_parallel_rt::num_threads()) (#1173)",
    `task_id: ${TASK_ID}. task_id: 260605_browser-wasm-parallel | track: F | backlog: #1173\n\nPREREQUISITES: Track A (#1176, gates narrowed) and Track E (#1174, dep wired) both done.\n\n## Goal\nReplace all 10 \`num_threads(1)\` calls (now inside \`all(wasm32, target_os="unknown")\` gates)\nwith \`num_threads(proxide_parallel_rt::num_threads())\`.\n\n## Procedure\n1. grep -rn 'num_threads(1)' crates/proxide-confind/src crates/proxide-io/src crates/proxide-master/src\n   — enumerate all 10 sites\n2. For each, change:\n     let par = par.num_threads(1);\n   to:\n     let par = par.num_threads(proxide_parallel_rt::num_threads());\n   Use full path; no \`use\` import needed.\n\nFiles: confind.rs (~2), parallel.rs (~3), formatters/full.rs (~2), fasta.rs (~1), search.rs (~2)\n\n## Acceptance criteria\n1. grep -rn 'num_threads(1)' crates/  ->  ZERO lines\n2. grep -rn 'num_threads(proxide_parallel_rt' crates/  ->  exactly 10 lines\n3. cargo test -p proxide-confind -p proxide-io -p proxide-master exits 0\n4. cargo check --target wasm32-unknown-unknown -p proxide-confind exits 0\n5. cargo check --target wasm32-unknown-unknown -p proxide-master exits 0\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. task_id: 260605_browser-wasm-parallel | track: F | backlog: #1173\n\n## Verification steps\n\ngrep -rn 'num_threads(1)' crates/  ->  ZERO lines\ngrep -rn 'num_threads(proxide_parallel_rt' crates/  ->  exactly 10 lines\ncargo test -p proxide-confind -p proxide-io -p proxide-master  ->  exits 0\ncargo check --target wasm32-unknown-unknown -p proxide-confind  ->  exits 0\ncargo check --target wasm32-unknown-unknown -p proxide-master  ->  exits 0\ngit diff HEAD  ->  touches only the 10 num_threads(1) lines, nothing else\n`,
  );

// ===== TRACK G — Track G — Expose init_parallel(n) synchronous wasm_bindgen fn in proxide-wasm (#1175) =========================
const trackG = () =>
  track(
    "1175",
    "Track G — Expose init_parallel(n) synchronous wasm_bindgen fn in proxide-wasm (#1175)",
    `task_id: ${TASK_ID}. task_id: 260605_browser-wasm-parallel | track: G | backlog: #1175\n\nPREREQUISITES: Track C (#1171, crate created) and Track F (#1173, call sites updated) both done.\n\n## Goal\nExpose a synchronous \`init_parallel(n: usize)\` JS-callable function from proxide-wasm.\n\n### crates/proxide-wasm/Cargo.toml — add to [dependencies]:\nproxide-parallel-rt = { path = "../proxide-parallel-rt" }\n\n### crates/proxide-wasm/src/lib.rs — add near other #[wasm_bindgen] exports:\n\`\`\`rust\n/// Set the number of threads for parallel iterators.\n/// Call before any parallel work: \`init_parallel(navigator.hardwareConcurrency)\`.\n#[wasm_bindgen]\npub fn init_parallel(num_threads: usize) {\n    proxide_parallel_rt::set_num_threads(num_threads);\n}\n\`\`\`\n\nSynchronous — no async, no await, no Promise. Intentional.\n\n## Acceptance criteria\n1. grep "init_parallel" crates/proxide-wasm/src/lib.rs  ->  finds the function\n2. grep "async fn init_parallel" crates/proxide-wasm/src/lib.rs  ->  ZERO lines\n3. grep "proxide-parallel-rt" crates/proxide-wasm/Cargo.toml  ->  finds the dep\n4. cargo build -p proxide-wasm exits 0\n5. cargo test --workspace exits 0\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. task_id: 260605_browser-wasm-parallel | track: G | backlog: #1175\n\n## Verification steps\n\ngrep -A5 "fn init_parallel" crates/proxide-wasm/src/lib.rs  ->  shows #[wasm_bindgen] + set_num_threads\ngrep "async fn init_parallel" crates/proxide-wasm/src/lib.rs  ->  ZERO lines\ngrep "proxide-parallel-rt" crates/proxide-wasm/Cargo.toml  ->  in [dependencies], not optional\ncargo build -p proxide-wasm  ->  exits 0\ncargo test --workspace  ->  exits 0\ngit diff HEAD  ->  touches only proxide-wasm/src/lib.rs and proxide-wasm/Cargo.toml\n`,
  );

// ===== TRACK A — Track A — WASI gate narrowing: #[cfg(wasm32)] -> #[cfg(all(wasm32, target_os="unknown"))] (#1176) =========================
const trackA = () =>
  track(
    "1176",
    "Track A — WASI gate narrowing: #[cfg(wasm32)] -> #[cfg(all(wasm32, target_os=\"unknown\"))] (#1176)",
    `task_id: ${TASK_ID}. task_id: 260605_browser-wasm-parallel | track: A | backlog: #1176\n\n## Goal\nNarrow all 10 wasm32 parallel gates from \`target_arch = "wasm32"\` to\n\`all(target_arch = "wasm32", target_os = "unknown")\`. This ensures wasm32-wasip1-threads\nbuilds use full std::thread parallelism rather than the browser-specific num_threads injection.\nIndependent quick win — does not require the spike or the new crate.\n\n## Why\n\`target_os = "unknown"\` is the browser target. WASI targets set target_os to "wasi" —\nthe narrower gate correctly excludes them.\n\n## All 10 sites (grep first, then change)\nRun: grep -rn 'num_threads(1)' crates/ to find all sites. The files are:\n  crates/proxide-confind/src/confind.rs\n  crates/proxide-confind/src/parallel.rs\n  crates/proxide-io/src/formatters/full.rs\n  crates/proxide-io/src/formats/fasta.rs\n  crates/proxide-master/src/search.rs\n\nFor each match inside a \`let par = par.num_threads(1)\` block, change the cfg attribute above it:\n  #[cfg(target_arch = "wasm32")]\nto:\n  #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]\n\nDo NOT change any other #[cfg(target_arch = "wasm32")] guards unrelated to the num_threads gate.\n\n## Acceptance criteria\n1. grep -rn 'cfg(target_arch = "wasm32")' crates/ | grep num_threads  ->  ZERO lines\n2. grep -rn 'all(target_arch = "wasm32", target_os = "unknown")' crates/ | grep num_threads  ->  10 lines\n3. cargo test -p proxide-confind -p proxide-io -p proxide-master exits 0\n4. cargo check --target wasm32-unknown-unknown -p proxide-confind exits 0\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. task_id: 260605_browser-wasm-parallel | track: A | backlog: #1176\n\n## Verification steps\n\n### Grep checks\ngrep -rn 'cfg(target_arch = "wasm32")' crates/ | grep num_threads\n— must return ZERO lines\n\ngrep -rn 'all(target_arch = "wasm32", target_os = "unknown")' crates/ | grep num_threads\n— must return exactly 10 lines\n\n### Test suite\ncargo test -p proxide-confind -p proxide-io -p proxide-master\n— exits 0\n\n### wasm32 compile check\ncargo check --target wasm32-unknown-unknown -p proxide-confind\n— exits 0 (target_os="unknown" on this target, so the cfg fires correctly)\n\n### Scope of change\ngit diff HEAD — must touch ONLY the cfg attributes; no logic changes, no new deps\n`,
  );

// ===== TRACK B — Track B — Spike: verify std::thread::scope runs on Web Workers with wasm32+atomics (#1170) =========================
const trackB = () =>
  agent(
    `task_id: ${TASK_ID}. task_id: 260605_browser-wasm-parallel | track: B | backlog: #1170\n\n## Goal\nProve that nightly + build-std + +atomics,+bulk-memory enables std::thread::scope + s.spawn\non wasm32-unknown-unknown and produces a binary with SharedArrayBuffer TLS exports.\nRequired empirical precondition before Tracks C-G.\n\n## Validation approach (compilation + binary inspection)\nA browser run is not required. Validate at the compilation and binary level.\n\n### Step 1 — Ensure .cargo/config.toml has wasm build flags\ncat .cargo/config.toml must show +atomics,+bulk-memory under [target.wasm32-unknown-unknown].\nIf Track D is not yet done, create it now as described in Track D before proceeding.\n\n### Step 2 — Attempt wasm32 compilation\nrustup run nightly-2025-11-15 cargo check --target wasm32-unknown-unknown -p proxide-wasm\nRecord whether this succeeds or produces an error.\n\n### Step 3 — Check for TLS exports in binary (if step 2 succeeded)\nrustup run nightly-2025-11-15 cargo build --target wasm32-unknown-unknown -p proxide-wasm --release\nwasm-objdump -x target/wasm32-unknown-unknown/release/proxide_wasm.wasm | grep -E "(tls|__wasm_init)"\n(If wasm-objdump is unavailable, use: wasm-nm target/wasm32-unknown-unknown/release/proxide_wasm.wasm)\n\n### Step 4 — Document findings\nWrite .praxia/docs/research/260605_wasm-thread-scope-spike.md with:\n- Commands run and their exit codes\n- Whether __wasm_init_tls or equivalent TLS symbols are present\n- PASS or FAIL verdict with evidence\n- If FAIL: exact error, recommended next step\n\n## Acceptance criteria\n1. Compilation succeeds on nightly-2025-11-15 with build-std + atomics\n2. __wasm_init_tls present in .wasm binary (confirms SharedArrayBuffer TLS support)\n3. .praxia/docs/research/260605_wasm-thread-scope-spike.md exists with PASS verdict\n4. No temporary test code left in any source file\n`,
    { agentType: "librarian", label: "research:1170", phase: "Track B — Spike: verify std::thread::scope runs on Web Workers with wasm32+atomics (#1170)", schema: RESEARCH_SCHEMA }
  );

// ===== TRACK C — Track C — Create proxide-parallel-rt crate (AtomicUsize thread-count registry) (#1171) =========================
const trackC = () =>
  track(
    "1171",
    "Track C — Create proxide-parallel-rt crate (AtomicUsize thread-count registry) (#1171)",
    `task_id: ${TASK_ID}. task_id: 260605_browser-wasm-parallel | track: C | backlog: #1171\n\nPREREQUISITE: Track B must have PASS verdict before starting.\n\n## Goal\nCreate the proxide-parallel-rt workspace crate: a minimal AtomicUsize thread-count registry.\nNo external dependencies.\n\n## Files to create\n\n### crates/proxide-parallel-rt/Cargo.toml\n\`\`\`toml\n[package]\nname = "proxide-parallel-rt"\nedition.workspace = true\nversion.workspace = true\ndescription = "Minimal parallel runtime: thread-count registry for wasm32 builds"\n\`\`\`\n\n### crates/proxide-parallel-rt/src/lib.rs\n\`\`\`rust\nuse std::sync::atomic::{AtomicUsize, Ordering};\n\nstatic NUM_THREADS: AtomicUsize = AtomicUsize::new(1);\n\npub fn set_num_threads(n: usize) {\n    NUM_THREADS.store(n.max(1), Ordering::Relaxed);\n}\n\npub fn num_threads() -> usize {\n    NUM_THREADS.load(Ordering::Relaxed)\n}\n\n#[cfg(test)]\nmod tests {\n    use super::*;\n\n    #[test]\n    fn default_is_one() {\n        assert_eq!(num_threads(), 1);\n    }\n\n    #[test]\n    fn set_and_get() {\n        set_num_threads(4);\n        assert_eq!(num_threads(), 4);\n        set_num_threads(1);\n    }\n\n    #[test]\n    fn zero_clamps_to_one() {\n        set_num_threads(0);\n        assert_eq!(num_threads(), 1);\n        set_num_threads(1);\n    }\n}\n\`\`\`\n\n## Workspace registration\nIn root Cargo.toml, add "crates/proxide-parallel-rt" to the members array.\n\n## Acceptance criteria\n1. cargo build -p proxide-parallel-rt exits 0\n2. cargo test -p proxide-parallel-rt -- --test-threads=1 exits 0 (3 tests pass)\n3. cargo check --target wasm32-unknown-unknown -p proxide-parallel-rt exits 0\n4. cargo test --workspace exits 0 (no regressions)\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. task_id: 260605_browser-wasm-parallel | track: C | backlog: #1171\n\n## Verification steps\n\nls crates/proxide-parallel-rt/src/lib.rs crates/proxide-parallel-rt/Cargo.toml  ->  both exist\ngrep "proxide-parallel-rt" Cargo.toml  ->  in members array\ncargo test -p proxide-parallel-rt -- --test-threads=1  ->  3 tests pass\ncargo check --target wasm32-unknown-unknown -p proxide-parallel-rt  ->  exits 0\ngrep "dependencies" crates/proxide-parallel-rt/Cargo.toml  ->  no [dependencies] section (no external deps)\ncargo test --workspace  ->  exits 0\n`,
  );

// ===== TRACK D — Track D — Add .cargo/config.toml wasm build flags; no root rust-toolchain.toml (#1172) =========================
const trackD = () =>
  track(
    "1172",
    "Track D — Add .cargo/config.toml wasm build flags; no root rust-toolchain.toml (#1172)",
    `task_id: ${TASK_ID}. task_id: 260605_browser-wasm-parallel | track: D | backlog: #1172\n\nPREREQUISITE: Track B must have PASS verdict before starting.\n\n## Goal\nAdd .cargo/config.toml at the workspace root with wasm32 build flags.\nDo NOT add rust-toolchain.toml at workspace root.\n\n## Check existing state\nls .cargo/config.toml\nIf it exists, ADD these sections without removing existing content.\nIf it does not exist, create .cargo/ directory and the file.\n\n## Content to add\n\`\`\`toml\n# Requires nightly for build-std:\n#   rustup run nightly-2025-11-15 wasm-pack build --target web -p proxide-wasm\n[target.wasm32-unknown-unknown]\nrustflags = [\n    "-C", "target-feature=+atomics,+bulk-memory",\n    "-C", "link-arg=--shared-memory",\n    "-C", "link-arg=--max-memory=1073741824",\n    "-C", "link-arg=--import-memory",\n    "-C", "link-arg=--export=__wasm_init_tls",\n    "-C", "link-arg=--export=__tls_size",\n    "-C", "link-arg=--export=__tls_align",\n    "-C", "link-arg=--export=__tls_base",\n]\n\n[unstable]\nbuild-std = ["panic_abort", "std"]\n\`\`\`\n\n## Acceptance criteria\n1. cat .cargo/config.toml shows +atomics,+bulk-memory and build-std entries\n2. ls rust-toolchain.toml  ->  "No such file" (no root toolchain file)\n3. cargo build --workspace (stable, native) exits 0\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. task_id: 260605_browser-wasm-parallel | track: D | backlog: #1172\n\n## Verification steps\n\ncat .cargo/config.toml  ->  contains [target.wasm32-unknown-unknown] and [unstable] build-std\ngrep "atomics" .cargo/config.toml  ->  returns the +atomics line\nls rust-toolchain.toml 2>&1  ->  "No such file or directory"\ncargo build --workspace  ->  exits 0, no new errors\n`,
  );

// ---- orchestrate: sequential writing chain || read-only research ----------
log("sprint-15-browser-wasm-parallel: writing chain (E -> F -> G, sequential) || research (A, B, C, D, read-only)");
const [writing, resA, resB, resC, resD] = await Promise.all([
  (async () => {
    const e = await trackE();
    const f = await trackF();
    const g = await trackG();
    return { e, f, g };
  })(),
  trackA(),
  trackB(),
  trackC(),
  trackD(),
]);

return {
  task_id: TASK_ID,
  sprint_id: 15,
  verdicts: {
    "1174": writing.e,
    "1173": writing.f,
    "1175": writing.g
  },
  research_1176: resA,
  research_1170: resB,
  research_1171: resC,
  research_1172: resD
};
