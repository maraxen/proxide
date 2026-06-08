// Sprint 16 runner — emitted by `praxia dw emit-sprint`, revised after oracle critique
// Source: .praxia/sprint_plans/260608_sprint-16-rotlib-source-pyobindings.toml
// Regenerate: praxia dw emit-sprint .praxia/sprint_plans/260608_sprint-16-rotlib-source-pyobindings.toml
// task_id: 260608_sprint-16-rotlib-source-pyobindings   sprint_id: 16
//
// ORCHESTRATION (oracle critique 260608):
//   Tracks A/B/C: SEQUENTIAL — all touch crates/proxide-rotlib/ (shared target/ lock;
//     A: convert_rotlib.rs + new rotlib_source/, B: parse_master.rs, C: tests/ + README)
//   Track D: CONCURRENT with A/B/C chain, isolation:"worktree" — separate proxide-py crate
//
// REVISION NOTES (oracle critique 260608):
//   - Fixed topology: emitter mis-routed all writing tracks into empty sequential chain
//     + read-only concurrent lane; tracks A/B/C are now the sequential writing chain
//   - Removed extractVerdict() dead code (was never called)
//   - Removed verdict:done fixer sentinel (reviewer VERDICT_SCHEMA is the source of truth)
//   - Added isolation:"worktree" on Track D fixer (runs concurrent with A/B/C chain)
//   - Fixed return shape: verdicts map populated by return_key, research_* keys removed
//   - Added per-track FAIL guard in sequential chain

export const meta = {
  name: "260608_sprint-16-rotlib-source-pyobindings",
  description: "Sprint 16 — RotlibSource trait (A), parse_master --validate (B), PRO P6 verify+docs (C) sequential; Python bindings skeleton (D) concurrent",
  phases: [
    { title: "Track A: RotlibSource trait", detail: "New rotlib_source/ module + DunbrackSource + --rotlib-source CLI (#1114)" },
    { title: "Track B: parse_master --validate", detail: "Add --validate flag + when-to-use docs to parse_master (#1115)" },
    { title: "Track C: P6 verify+docs", detail: "cis-PRO fallback test + ODC-BY README + no-CC-artifact check (#819)" },
    { title: "Track D: Python bindings", detail: "py_confind/py_rotlib/py_frag stubs wired into _proxider (#1306)" },
  ],
};

const TASK_ID = "260608_sprint-16-rotlib-source-pyobindings";
const MAX_FIX_RETRIES = 2;

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

const EMITTER_CTX = `Prior sprint context (Sprint 14 + 15 outputs):
- Sprint 14: RTF/CCD IC importers + residue_geometry.proto merged; convert_rotlib --ic-source URI wired
  (ccd: and rtf: both implemented; see convert_rotlib.rs:52-78)
- Sprint 15: orx-parallel migration + proxide-parallel-rt + init_parallel WASM export merged
- Dunbrack+CCD confind migration (#869) complete: load_pb() is the production path in confind.rs

Active spec: .praxia/docs/specs/260604_rotlib-multi-source-arch.md
  - §3 defines RotlibSource trait (RotamerEntry, BinData, all 6 methods)
  - §6 defines new CLI shape: --rotlib-source dunbrack:<path> + existing --ic-source flags
  - Phase 1 work (#869, #1112, #1113) is complete; this sprint is Phases 2+3

Key invariants from spec §10:
  - attribution field must be non-empty in every pb.zst proto
  - *.master.pb.zst files must never be committed
  - apply_ic_table() skips proline (ProlineBuilder manages ring geometry)
  - PRECOMPUTED mode is the shipped default`;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

const fixer = (prompt, label, phaseName, extraOpts = {}) =>
  agent(prompt, { agentType: "fixer", label, phase: phaseName, ...extraOpts });

// itemId is NOT a param — identity is embedded in the label and already in the prompt text
const reviewer = (prompt, label, phaseName) =>
  agent(prompt, { agentType: "reviewer", label, phase: phaseName, schema: VERDICT_SCHEMA });

async function track(itemId, phaseName, fixerPrompt, reviewerPrompt, fixerOpts = {}) {
  log(`[${itemId}] implement`);
  await fixer(fixerPrompt, `fix:${itemId}`, phaseName, fixerOpts);
  let verdict = await reviewer(reviewerPrompt, `review:${itemId}`, phaseName);
  for (let retry = 0; retry < MAX_FIX_RETRIES && verdict && verdict.verdict === "NEEDS_WORK"; retry++) {
    log(`[${itemId}] NEEDS_WORK — repair cycle ${retry + 1}/${MAX_FIX_RETRIES}`);
    const issues = (verdict.issues || [])
      .map((i) => `- ${i.where}: ${i.problem} -> ${i.fix}`)
      .join("\n");
    await fixer(
      `${fixerPrompt}\n\nA reviewer found issues — fix exactly these, nothing else:\n${issues}`,
      `fix:${itemId}:repair:${retry}`,
      phaseName,
      fixerOpts
    );
    verdict = await reviewer(reviewerPrompt, `review:${itemId}:re:${retry}`, phaseName);
  }
  return verdict;
}

// ─────────────────────────────────────────────────────────────────────────────
// Track A — RotlibSource trait + DunbrackSource refactor (#1114)
// ─────────────────────────────────────────────────────────────────────────────
const trackA = () =>
  track(
    "1114",
    "Track A: RotlibSource trait",
    `task_id: ${TASK_ID} | track: A | backlog: #1114

## Goal
Implement the RotlibSource abstraction layer in proxide-rotlib as specified in
.praxia/docs/specs/260604_rotlib-multi-source-arch.md §3. Extract the existing
Dunbrack text parser from convert_rotlib.rs into a DunbrackSource implementing
the trait. Wire build_library() to accept Box<dyn RotlibSource>. Add --rotlib-source CLI flag.

## Key files
- crates/proxide-rotlib/src/bin/convert_rotlib.rs:30-78
  Current state: DunbrackRotamer struct (line 38-46) and ic_source string matching
  (lines 52-78) live inline. DunbrackRotamer holds res_code, phi, psi, count,
  probability, chi_values, chi_sigmas — these map directly to BinData + RotamerEntry.
- crates/proxide-rotlib/src/lib.rs — add \`pub mod rotlib_source;\` here
- NEW: crates/proxide-rotlib/src/rotlib_source/mod.rs — trait + types
- NEW: crates/proxide-rotlib/src/rotlib_source/dunbrack.rs — DunbrackSource impl

## Trait definition (from spec §3 — copy exactly)
\`\`\`rust
pub struct RotamerEntry {
    pub chi_values: Vec<f32>,
    pub chi_sigmas: Vec<f32>,
    pub probability: f64,
    pub count: u32,
}
pub struct BinData {
    pub phi: f64,
    pub psi: f64,
    pub freq: f64,
    pub rotamers: Vec<RotamerEntry>,
}
pub trait RotlibSource: Send + Sync {
    fn residue_codes(&self) -> Vec<String>;
    fn bins(&self, code: &str) -> Vec<BinData>;
    fn default_bin_index(&self, code: &str) -> usize;
    fn data_license(&self) -> &str;
    fn attribution(&self) -> &str;
    fn source_tag(&self) -> &str;
}
\`\`\`

## DunbrackSource implementation
- Constructor: \`DunbrackSource::from_file(path: &Path) -> Result<Self, ...>\`
  Reads the Dunbrack *.bbdep.rotamers.lib text file; populates an internal
  HashMap<String, Vec<BinData>> keyed by residue code.
- \`residue_codes()\` returns the HashMap keys sorted.
- \`bins()\` returns the Vec<BinData> for the given code (empty Vec if unknown).
- \`default_bin_index()\` returns the index of the BinData with maximum freq.
- \`data_license()\` returns \`"ODC-BY-1.0"\`.
- \`attribution()\` returns \`"Dunbrack BBDEP2010 SimpleOpt1-5; Shapovalov & Dunbrack 2011"\`.
- \`source_tag()\` returns \`"dunbrack2010_simpleopt1"\`.

The existing Dunbrack parsing logic in convert_rotlib.rs should be MOVED
(not duplicated) into DunbrackSource::from_file. Remove DunbrackRotamer from
convert_rotlib.rs once the move is complete.

## build_library() change
Find the existing build_library() function in convert_rotlib.rs.
Change its signature to accept \`Box<dyn RotlibSource>\` instead of directly
reading the grouped rotamers. Thread RotlibSource methods through to produce
the same RotamerLibrary proto output.

## CLI change — convert_rotlib.rs Args struct
Add a new optional argument:
\`\`\`rust
/// Rotamer library source: "dunbrack:<path>" for Dunbrack BBDEP text file.
/// Default: uses --input path if present, for backwards compatibility.
#[arg(long)]
rotlib_source: Option<String>,
\`\`\`
In main(), parse "dunbrack:<path>" to construct DunbrackSource.
If --rotlib-source is absent and --input is present, default to DunbrackSource
from --input path (backwards compat; log a deprecation hint to stderr).
If both absent, return a clear error: "Provide --rotlib-source dunbrack:<path>".

## Edit-only — no Write on existing files
## Do not touch: ic_source parsing, apply_ic_table(), load_pb(), confind.rs

## Acceptance criteria
1. cargo build --workspace exits 0
2. cargo test -p proxide-rotlib exits 0 (no test regressions)
3. grep -n "DunbrackRotamer" crates/proxide-rotlib/src/bin/convert_rotlib.rs → empty
4. cargo run -p proxide-rotlib --bin convert_rotlib -- --help | grep "rotlib-source" → shows the flag
5. The new rotlib_source module is exported from lib.rs (pub mod rotlib_source)

${EMITTER_CTX}`,
    `task_id: ${TASK_ID} | track: A | backlog: #1114

## What correct looks like

### Module structure
- ls crates/proxide-rotlib/src/rotlib_source/ must show at least mod.rs and dunbrack.rs
- grep "pub mod rotlib_source" crates/proxide-rotlib/src/lib.rs → present
- grep "RotlibSource" crates/proxide-rotlib/src/rotlib_source/mod.rs → trait defined

### DunbrackRotamer gone from convert_rotlib
- grep -n "DunbrackRotamer" crates/proxide-rotlib/src/bin/convert_rotlib.rs → empty
  (it should now live only in dunbrack.rs or be replaced entirely by BinData/RotamerEntry)

### Trait completeness
- All 6 required methods present: residue_codes, bins, default_bin_index,
  data_license, attribution, source_tag
- RotamerEntry and BinData structs have all required fields (no missing pub fields)

### Build and tests
- cargo build --workspace exits 0, no new compile errors
- cargo test -p proxide-rotlib exits 0; no test regressions
  (pay attention to test_converter_ac3.rs and test_schema_roundtrip.rs — these
   exercise convert_rotlib logic most directly)

### CLI smoke test
- cargo run -p proxide-rotlib --bin convert_rotlib -- --help
  Must show --rotlib-source in the output
  Must not have regressed any other flags (--input, --output, --ic-source)

### License fields
- DunbrackSource::data_license() must return exactly "ODC-BY-1.0"
- DunbrackSource::attribution() must be non-empty and mention "Shapovalov" or "Dunbrack 2011"`
  );

// ─────────────────────────────────────────────────────────────────────────────
// Track B — parse_master --validate + docs (#1115)
// ─────────────────────────────────────────────────────────────────────────────
const trackB = () =>
  track(
    "1115",
    "Track B: parse_master --validate",
    `task_id: ${TASK_ID} | track: B | backlog: #1115

## Goal
Expose parse_master as a first-class user-facing tool with a --validate flag
and clear license documentation. See spec §8 Phase 3.

## Key file
- crates/proxide-rotlib/src/bin/parse_master.rs:23-31
  Current Args struct has only --input (PathBuf) and --output (PathBuf).
  No --validate flag exists yet.

## Code change — Args struct (parse_master.rs:23-31)
Add:
\`\`\`rust
/// Validate output: run drift test against small.pdb and report max|delta|.
/// Requires ROTLIB_PATH env var pointing to Mosaist rotlib.bin.
#[arg(long, default_value_t = false)]
validate: bool,
\`\`\`

## Validation logic (in main(), after writing output)
When --validate is set:
1. Check ROTLIB_PATH env var is set; if not, print to stderr and skip gracefully.
2. Load the freshly written output .pb.zst with load_pb().
3. Find the small.pdb test fixture — check these paths in order:
   a. crates/proxide-rotlib/tests/data/small.pdb
   b. tests/data/small.pdb
   If not found, skip with a message.
4. Run confind on small.pdb using the loaded library and compute max|delta| vs
   the MASTER reference contact list (use the existing drift test logic from
   tests/test_parse_master.rs as reference — do NOT import test helpers; inline
   the relevant measurement).
5. Print to stdout: "VALIDATE max|delta|: {:.6} (threshold 5e-4: {})"
   where {} is PASS or FAIL based on < 5e-4.

## Documentation change
Extend the existing doc comment at the top of parse_master.rs (lines 1-9)
with a new section:

\`\`\`
## When to use parse_master vs dunbrack+ccd
- parse_master: exact MASTER parity, max|delta|=0. Use when you need contact degrees
  numerically identical to MASTER. Requires a Mosaist install (ROTLIB_PATH set).
  License: CC-BY-NC-SA 4.0 — personal use OK; output must not be committed or distributed.
- convert_rotlib (Dunbrack+CCD): redistributable, max|delta|<0.05, 0 contact flips at
  CD > 0.001. Use for any output you share or bundle in a repo.

## License note
Output is CC-BY-NC-SA 4.0 (Grigoryan lab / Mosaist). You may use the output
for personal research. Do NOT commit the .pb.zst output to version control or
distribute it to others.
\`\`\`

## Edit-only — no Write on existing files

## Acceptance criteria
1. cargo build -p proxide-rotlib exits 0
2. cargo run -p proxide-rotlib --bin parse_master -- --help | grep validate → shows the flag
3. The doc comment section "When to use parse_master vs dunbrack+ccd" is present in the file
4. cargo test -p proxide-rotlib exits 0 (no regressions in test_parse_master.rs)

${EMITTER_CTX}`,
    `task_id: ${TASK_ID} | track: B | backlog: #1115

## What correct looks like

### Flag presence
- cargo run -p proxide-rotlib --bin parse_master -- --help
  Must show --validate in the output with a helpful description
  Must still show --input and --output unchanged

### Documentation
- grep -A5 "When to use parse_master" crates/proxide-rotlib/src/bin/parse_master.rs
  Must show the section heading and at least one bullet point
- grep "CC-BY-NC-SA" crates/proxide-rotlib/src/bin/parse_master.rs
  Must appear in the doc comment (license note section)
- The doc comment must NOT say "distribute" is OK — it should say it is NOT

### Validation logic
- Read the --validate branch in main(): it must check ROTLIB_PATH before attempting anything
- If ROTLIB_PATH is unset, it must not panic — it must print a clear skip message
- The validation output format must include "max|delta|" and either "PASS" or "FAIL"

### No test regressions
- cargo test -p proxide-rotlib exits 0
- test_parse_master.rs tests must all pass`
  );

// ─────────────────────────────────────────────────────────────────────────────
// Track C — P6 Verify+docs: cis-PRO fallback + ODC-BY attribution (#819)
// ─────────────────────────────────────────────────────────────────────────────
const trackC = () =>
  track(
    "819",
    "Track C: P6 verify+docs",
    `task_id: ${TASK_ID} | track: C | backlog: #819

## Goal
Close the P6 verification gap from Sprint 12: add a real-library cis-PRO fallback
integration test and add ODC-BY attribution + Shapovalov-Dunbrack citation to the
rotlib README. Verify test suite is clean and no CC-BY-NC-SA artifact exists.

## Background
The AC-R routing tests in tests/test_ac_r_routing.rs already verify that
PRO + is_cis_peptide=true routes to CPR when a CPR entry is present in a
SYNTHETIC library (build_pro_cpr_library(), line ~44). The gap is:
- The shipped Dunbrack+CCD pb.zst has no CPRO entry
- When CPRO is absent from the loaded library, the router must fall back to PRO
  coords and NOT panic or return an error
- This fallback path is not exercised by an integration test

## Key files
- crates/proxide-confind/src/coords.rs:40-42
  is_cis_peptide: bool field — set to true when |omega| < 30.0 degrees
- crates/proxide-rotlib/tests/test_ac_r_routing.rs — existing synthetic tests
- crates/proxide-rotlib/tests/helpers.rs — shared test helpers

## Change 1 — New integration test
Add a new test to tests/test_ac_r_routing.rs (or a new file test_cpr_fallback.rs):

\`\`\`rust
/// When the loaded library has no CPRO entry, place_rotamer("PRO", ..., is_cis=true)
/// must succeed and return the same result as place_rotamer("PRO", ..., is_cis=false).
#[test]
fn test_cpr_fallback_when_cpro_absent() {
    // Build a library with PRO but NO CPR entry (adapt build_pro_cpr_library()
    // from test_ac_r_routing.rs line ~44 but omit the CPR ResidueEntry)
    let lib = /* PRO-only library */;
    let n = [0.0f32; 3]; let ca = [1.458, 0.0, 0.0]; let c = [2.1, 1.2, 0.0];
    let p_normal = lib.place_rotamer("PRO", -60.0, -40.0, 0, false, n, ca, c).unwrap();
    let p_cis    = lib.place_rotamer("PRO", -60.0, -40.0, 0, true,  n, ca, c).unwrap();
    assert_eq!(p_normal.atoms.len(), p_cis.atoms.len(), "atom count must match in fallback");
    for (a, b) in p_normal.atoms.iter().zip(p_cis.atoms.iter()) {
        for i in 0..3 {
            assert!((a.xyz[i] - b.xyz[i]).abs() < 1e-6,
                "fallback PRO(cis) must return same coords as PRO(trans) when CPRO absent");
        }
    }
}
\`\`\`

## Change 2 — README attribution
Add an ODC-BY notice to crates/proxide-rotlib/README.md (create if absent):
\`\`\`
## Data attribution

This crate ships pre-built rotamer libraries derived from:
- **Dunbrack BBDEP 2010 (SimpleOpt1-5)** — Shapovalov & Dunbrack (2011),
  Structure 19(6):844-858. License: ODC-BY-1.0 (Open Data Commons Attribution).
  Attribution: "Contains data from the Dunbrack Backbone-Dependent Rotamer Library."
- **PDB Chemical Component Dictionary (CCD)** — RCSB PDB. License: CC0-1.0.
\`\`\`

## Change 3 — Confirm no CC-BY-NC-SA artifact
Run: git ls-files | grep -E "(rotlib\\.bin|\\.master\\.pb\\.zst)"
Must return empty. If any appear, add them to .gitignore only — do not remove source files.

## Edit-only — no Write on existing files except README.md (create if absent) and .gitignore

## Acceptance criteria
1. cargo test -p proxide-rotlib exits 0, including the new cpr_fallback test
2. crates/proxide-rotlib/README.md contains "ODC-BY" and "Shapovalov"
3. git ls-files | grep -E "(rotlib\\.bin|\\.master\\.pb\\.zst)" → empty
4. cargo test --workspace --all-targets exits 0

${EMITTER_CTX}`,
    `task_id: ${TASK_ID} | track: C | backlog: #819

## What correct looks like

### New cis-PRO fallback test
- grep -rn "test_cpr_fallback" crates/proxide-rotlib/tests/ → found
- cargo test -p proxide-rotlib test_cpr_fallback -- --nocapture → prints "ok"
- The test setup must NOT call build_pro_cpr_library() (that helper has both PRO and CPR)

### README attribution
- cat crates/proxide-rotlib/README.md | grep "ODC-BY" → present
- cat crates/proxide-rotlib/README.md | grep "Shapovalov" → present
- cat crates/proxide-rotlib/README.md | grep "CC0" → present (CCD citation)

### No CC artifacts
- git ls-files | grep -E "(rotlib\\.bin|\\.master\\.pb\\.zst)" → empty
- git ls-files | grep "\\.master\\." → empty

### Clean test suite
- cargo test -p proxide-rotlib --all-targets exits 0
- test_ac_r_routing.rs existing tests must not regress`
  );

// ─────────────────────────────────────────────────────────────────────────────
// Track D — Python bindings skeleton (#1306)  [isolation: worktree]
// ─────────────────────────────────────────────────────────────────────────────
const trackD = () =>
  track(
    "1306",
    "Track D: Python bindings",
    `task_id: ${TASK_ID} | track: D | backlog: #1306

## Goal
Add ConFind, RotLib, and fragment-search stubs to the existing proxide-py crate.
The crate already exposes IO/chemistry surfaces (proxide_py/src/lib.rs:72-146).
This track adds three stub modules and wires them into the _proxider PyModule.

## Key file
- crates/proxide_py/src/lib.rs:72-146 — _proxider PyModule registration function.
  New registrations go at the end of this function, before Ok(()).

## New modules to create

### crates/proxide_py/src/py_confind.rs
\`\`\`rust
use pyo3::prelude::*;
/// Run ConFind on a structure and return contact pairs as a list of dicts.
/// Each dict: {"res_i": str, "res_j": str, "contact_degree": f64}
#[pyfunction]
#[pyo3(signature = (structure, cd_threshold = 0.0))]
pub fn run_confind(
    structure: &crate::py_parsers::PyAtomicSystem,
    cd_threshold: f64,
) -> PyResult<Vec<std::collections::HashMap<String, pyo3::PyObject>>> {
    // TODO: wire to proxide_confind::run_confind()
    let _ = (structure, cd_threshold);
    Ok(vec![])
}
\`\`\`

### crates/proxide_py/src/py_rotlib.rs
\`\`\`rust
use pyo3::prelude::*;
#[pyclass]
pub struct PyRotamerLibrary {}
#[pymethods]
impl PyRotamerLibrary {
    #[staticmethod]
    pub fn load_pb(path: &str) -> PyResult<Self> {
        // TODO: wire to proxide_rotlib::RotamerLibrary::load_pb()
        let _ = path;
        Ok(Self {})
    }
    pub fn residue_codes(&self) -> PyResult<Vec<String>> {
        // TODO: return real codes from inner library
        Ok(vec![])
    }
}
\`\`\`

### crates/proxide_py/src/py_frag.rs
\`\`\`rust
use pyo3::prelude::*;
/// Search a fragment database (proxide-frag). Stub — returns empty results.
#[pyfunction]
#[pyo3(signature = (query, db_path, n_results = 100, n_threads = None))]
pub fn search_fragments(
    query: &crate::py_parsers::PyAtomicSystem,
    db_path: &str,
    n_results: usize,
    n_threads: Option<usize>,
) -> PyResult<Vec<std::collections::HashMap<String, pyo3::PyObject>>> {
    let _ = (query, db_path, n_results, n_threads);
    Ok(vec![])
}
\`\`\`

## Wire into _proxider module (lib.rs)
Add at the top of lib.rs:
\`\`\`rust
mod py_confind;
mod py_frag;
mod py_rotlib;
\`\`\`
After existing registrations (line ~143, before Ok(())):
\`\`\`rust
m.add_function(wrap_pyfunction!(py_confind::run_confind, m)?)?;
m.add_class::<py_rotlib::PyRotamerLibrary>()?;
m.add_function(wrap_pyfunction!(py_frag::search_fragments, m)?)?;
\`\`\`

## Dependencies (crates/proxide_py/Cargo.toml)
If proxide-confind is not already a dep: add \`proxide-confind = { path = "../proxide-confind" }\`
If proxide-rotlib is not already a dep: add \`proxide-rotlib = { path = "../proxide-rotlib" }\`

## Edit-only — no Write on existing files except the 3 new .rs files
## Stubs only — do not attempt full wiring (full wiring is Track D+ in a future sprint)

## Acceptance criteria
1. cargo build -p proxide-py exits 0
2. grep "run_confind\\|PyRotamerLibrary\\|search_fragments" crates/proxide_py/src/lib.rs → all three present
3. cargo test -p proxide-py exits 0 (no regressions)

${EMITTER_CTX}`,
    `task_id: ${TASK_ID} | track: D | backlog: #1306

## What correct looks like

### New source files
- ls crates/proxide_py/src/ | grep -E "py_confind|py_rotlib|py_frag" → all three present

### Module declarations in lib.rs
- grep "mod py_confind\\|mod py_rotlib\\|mod py_frag" crates/proxide_py/src/lib.rs → all three present

### PyModule registrations in lib.rs
- grep "run_confind\\|PyRotamerLibrary\\|search_fragments" crates/proxide_py/src/lib.rs → all three present

### Build
- cargo build -p proxide-py exits 0; no new compile errors

### Stubs are stubs
- grep "TODO" crates/proxide_py/src/py_confind.rs → present
- The stubs return empty results, not panic or unimplemented!()
- PyRotamerLibrary::load_pb returns Ok(Self {}) not Err()

### Dependencies declared
- grep "proxide-confind\\|proxide-rotlib" crates/proxide_py/Cargo.toml → both present

### No regressions
- cargo test -p proxide-py exits 0`,
    { isolation: "worktree" }
  );

// ─────────────────────────────────────────────────────────────────────────────
// Orchestration: A/B/C sequential (proxide-rotlib, shared target/ lock)
//               D concurrent in own worktree (proxide-py, isolated)
//
// Both arms are throwing IIFEs so Promise.all fails fast symmetrically:
//   - A/B/C arm: throws on any FAIL verdict → short-circuits the sequential chain
//   - D arm: throws on FAIL verdict → does not wait for A/B/C to finish
// Note: Promise.all does not cancel the surviving arm when one rejects — the
// harness is responsible for worktree cleanup of an orphaned D agent.
// ─────────────────────────────────────────────────────────────────────────────
log(`${TASK_ID}: A/B/C sequential (proxide-rotlib crate) || D isolated worktree (proxide-py)`);

const [abcChain, resD] = await Promise.all([
  (async () => {
    const resA = await trackA();
    if (resA?.verdict !== "PASS") throw new Error(`Track A (#1114) ${resA?.verdict ?? "NULL"}: ${resA?.summary ?? "agent returned null"}`);
    const resB = await trackB();
    if (resB?.verdict !== "PASS") throw new Error(`Track B (#1115) ${resB?.verdict ?? "NULL"}: ${resB?.summary ?? "agent returned null"}`);
    const resC = await trackC();
    if (resC?.verdict !== "PASS") throw new Error(`Track C (#819) ${resC?.verdict ?? "NULL"}: ${resC?.summary ?? "agent returned null"}`);
    return { resA, resB, resC };
  })(),
  (async () => {
    const d = await trackD();
    if (d?.verdict !== "PASS") throw new Error(`Track D (#1306) ${d?.verdict ?? "NULL"}: ${d?.summary ?? "agent returned null"}`);
    return d;
  })(),
]);

const { resA, resB, resC } = abcChain;

return {
  task_id: TASK_ID,
  sprint_id: 16,
  verdicts: {
    "1114": resA,
    "1115": resB,
    "819":  resC,
    "1306": resD,
  },
};
