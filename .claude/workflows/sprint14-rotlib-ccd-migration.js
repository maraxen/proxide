// Sprint 14: Track A (--ic-source flag) → Track B (vendor CCD) → pb.zst regen
//          → Track C (migrate confind) → 1DC7 calibration → final tests
// task_id: 260604_rotlib_multi_source
// working_directory: /home/marielle/projects/proxide
//
// REVISION NOTES (oracle critique 260604):
//   - Added Phase 0 for Track A (#1112) — convert_rotlib --ic-source was missing entirely
//   - Fixed CCD path: tests/data/ccd/ (not data/ccd/)
//   - Decoupled Phase 1 git commits: download-only agents, single commit agent
//   - Moved confind migration (Track C) before calibration — binary must use load_pb()
//     before it can run against the new CCD pb.zst
//   - Removed MASTER reference gate; calibration uses numeric sanity + threshold monotonicity
//   - Fixed confind CLI flags: --p, --rLib, --o, --cdcut (not --input/--output/--cd-threshold)
//   - Binary uses --rLib, not PROXIDE_ROTLIB_PB env var
//   - Fixed drift test: measure_loadpb_drift_vs_master with --ignored (was test_drift_loadpb_small_pdb)
//   - Fixed load_pb file reference: bin/confind.rs:22, not src/confind.rs

export const meta = {
  name: 'sprint14-rotlib-ccd-migration',
  description: 'Track A: --ic-source flag; Track B: vendor CCD .cif; regen pb.zst; Track C: migrate confind to load_pb(); calibrate on 1DC7; final tests.',
  phases: [
    { title: 'Track A: --ic-source', detail: 'Add --ic-source ccd default to convert_rotlib.rs (#1112)' },
    { title: 'Track B: Vendor CCD', detail: 'Download 19 .cif files to tests/data/ccd/ (download-only agents, single commit)' },
    { title: 'Regenerate pb.zst', detail: 'cargo run convert_rotlib with --ic-source ccd → data/rotlibs/proxide-rotlib-dunbrack2010-ccd.pb.zst' },
    { title: 'Track C: Migrate confind', detail: 'bin/confind.rs load() → load_pb() (#869)' },
    { title: 'Calibrate 1DC7', detail: 'Run migrated confind on 1DC7 at 4 --cdcut thresholds' },
    { title: 'Verify Calibration', detail: 'Two adversarial verifiers: methodology + numeric sanity (no MASTER required)' },
    { title: 'Final Test', detail: 'Drift gate (measure_loadpb_drift_vs_master --ignored), cargo test --workspace, CC-NC scan' },
  ],
}

// ─────────────────────────────────────────────────────────────────────────────
// Schemas
// ─────────────────────────────────────────────────────────────────────────────

const TRACK_A_SCHEMA = {
  type: "object",
  properties: {
    success: { type: "boolean" },
    ic_source_arg_verified: { type: "boolean" },
    commit_sha: { type: "string" },
    error: { type: "string" },
  },
  required: ["success"],
}

const CCD_DOWNLOAD_SCHEMA = {
  type: "object",
  properties: {
    aa_code: { type: "string" },
    local_path: { type: "string" },
    success: { type: "boolean" },
    error: { type: "string" },
  },
  required: ["aa_code", "success"],
}

const REGEN_SCHEMA = {
  type: "object",
  properties: {
    success: { type: "boolean" },
    output_path: { type: "string" },
    file_size_bytes: { type: "number" },
    smoke_test_passed: { type: "boolean" },
    commit_sha: { type: "string" },
    error: { type: "string" },
  },
  required: ["success"],
}

const MIGRATE_SCHEMA = {
  type: "object",
  properties: {
    success: { type: "boolean" },
    files_changed: { type: "array", items: { type: "string" } },
    drift_test_passed: { type: "boolean" },
    drift_max_delta: { type: "number" },
    drift_ccd_max_delta: { type: "number" },
    license_gate_clean: { type: "boolean" },
    commit_sha: { type: "string" },
    error: { type: "string" },
  },
  required: ["success"],
}

const CALIBRATION_SCHEMA = {
  type: "object",
  properties: {
    thresholds: {
      type: "array",
      items: {
        type: "object",
        properties: {
          cdcut: { type: "number" },
          contact_count: { type: "number" },
          run_succeeded: { type: "boolean" },
          output_file: { type: "string" },
        },
        required: ["cdcut", "run_succeeded"],
      },
    },
    residue_count: { type: "number" },
    notes: { type: "string" },
  },
  required: ["thresholds"],
}

const VERIFY_SCHEMA = {
  type: "object",
  properties: {
    verdict: { type: "string", enum: ["CONFIRMED", "REFUTED", "UNCERTAIN"] },
    reasoning: { type: "string" },
    gate_passes: { type: "boolean" },
    concerns: { type: "array", items: { type: "string" } },
  },
  required: ["verdict", "gate_passes"],
}

// 19 amino acids to vendor (PRO already present at tests/data/ccd/PRO.cif)
const AA_CODES = [
  'ALA', 'ARG', 'ASN', 'ASP', 'CYS',
  'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
  'LEU', 'LYS', 'MET', 'PHE', 'SER',
  'THR', 'TRP', 'TYR', 'VAL',
]

const CCD_PB_PATH = '/tmp/proxide-rotlib-dunbrack2010-ccd.pb.zst'
const CCD_PB_DEST = 'data/rotlibs/proxide-rotlib-dunbrack2010-ccd.pb.zst'
const DC7_PDB = '/home/marielle/repos/mosaist/testfiles/1DC7.pdb'

// ─────────────────────────────────────────────────────────────────────────────
// Phase 0: Track A — Implement --ic-source flag in convert_rotlib.rs (#1112)
// ─────────────────────────────────────────────────────────────────────────────

phase('Track A: --ic-source')
log('Implementing --ic-source ccd default in convert_rotlib.rs (Track A, #1112)')

const trackAResult = await agent(
  `You are implementing Track A (#1112) for the proxide Sprint 14.

task_id: 260604_rotlib_multi_source
working_directory: /home/marielle/projects/proxide

## Context
convert_rotlib.rs currently has no --ic-source argument — it unconditionally applies CHARMM IC
overrides to all non-proline residues via apply_charmm_ideals(). Track A adds an --ic-source flag
with default "ccd" that, when set to "ccd", skips CHARMM overrides and keeps the CCD-derived
template geometry (which is already embedded in the templates in template.rs).

When --ic-source is "charmm", behavior is identical to the current code.

## Exact file to modify (ONLY this file)
crates/proxide-rotlib/src/bin/convert_rotlib.rs

Read this file in full before making any changes.

## Changes to make (edit only, no Write on existing files)

### 1. Add --ic-source to the Args struct (after the charmm_xml field)

  /// IC source for sidechain geometry: "ccd" keeps CCD template values;
  /// "charmm" applies CHARMM36 force field overrides.
  #[arg(long, default_value = "ccd")]
  ic_source: String,

### 2. Make CHARMM loading conditional in main()

Before the change, main() unconditionally calls:
  let charmm_ideals = load_charmm_ideals(args.charmm_xml.to_str().unwrap())...;

After the change:
  let charmm_ideals = if args.ic_source == "charmm" {
      Some(
          load_charmm_ideals(args.charmm_xml.to_str().unwrap())
              .map_err(|e| format!("Failed to load CHARMM ideals: {}", e))?,
      )
  } else {
      eprintln!("IC source: ccd (using CCD template geometry; CHARMM overrides disabled)");
      None
  };

### 3. Update the build_library call signature

build_library now receives an Option<&CharmmIdeals> instead of &CharmmIdeals.
Change the call in main():
  let lib = build_library(&grouped, &charmm_ideals)?;
  // becomes:
  let lib = build_library(&grouped, charmm_ideals.as_deref())?;

### 4. Update build_library signature and body

Change the function signature:
  fn build_library(
      grouped: &GroupedRotamers,
      charmm_ideals: Option<&proxide_rotlib::geometry::charmm_ic::CharmmIdeals>,
  ) -> Result<rotlib_v1::RotamerLibrary, Box<dyn std::error::Error>> {

Inside the loop, the CHARMM application block currently reads:
  let is_proline = matches!(res_code.as_str(), "PRO" | "CPR" | "TPR");
  if is_proline {
      ic_proline_skipped += 1;
  } else {
      let charmm_resname = map_template_to_charmm_name(res_code);
      match apply_charmm_ideals(&mut template, charmm_ideals, charmm_resname) {
          Ok(applied) => ic_overrides_count += applied,
          Err(e) => { ... }
      }
  }

Change to:
  let is_proline = matches!(res_code.as_str(), "PRO" | "CPR" | "TPR");
  if is_proline {
      ic_proline_skipped += 1;
  } else if let Some(ideals) = charmm_ideals {
      let charmm_resname = map_template_to_charmm_name(res_code);
      match apply_charmm_ideals(&mut template, ideals, charmm_resname) {
          Ok(applied) => ic_overrides_count += applied,
          Err(e) => {
              eprintln!("Warning: could not apply CHARMM ICs to {}: {}", res_code, e);
              ic_misses_count += 1;
          }
      }
  }

### 5. Verify the change compiles
  cargo check -p proxide-rotlib 2>&1 | grep -E "^error|^warning.*unused" | head -20

  If it compiles clean (no errors), stage and commit:
  git add crates/proxide-rotlib/src/bin/convert_rotlib.rs
  git commit -m "feat(#1112): add --ic-source ccd default to convert_rotlib [260604_rotlib_multi_source]"

## Constraints
- Edit only — do NOT use Write on convert_rotlib.rs
- Commit ONLY convert_rotlib.rs — nothing else
- Do NOT run cargo build or cargo run
- If cargo check fails, describe the errors and stop; do not guess at fixes

Return: success (bool), ic_source_arg_verified (bool — did --help show --ic-source after change?),
commit_sha (from git log --oneline -1), error (if failed).`,
  { label: 'track-a-ic-source', phase: 'Track A: --ic-source', schema: TRACK_A_SCHEMA }
)

if (!trackAResult?.success) {
  throw new Error(
    `Track A (#1112) failed: ${trackAResult?.error ?? 'unknown error'}. ` +
    `Cannot proceed — convert_rotlib must support --ic-source before regenerating pb.zst.`
  )
}
log(`Track A done. ic_source_arg_verified=${trackAResult?.ic_source_arg_verified}, sha=${trackAResult?.commit_sha ?? 'unknown'}`)

// ─────────────────────────────────────────────────────────────────────────────
// Phase 1: Track B — Vendor CCD .cif files (#1113)
// Two-step: parallel downloads (no git), then single commit agent
// ─────────────────────────────────────────────────────────────────────────────

phase('Track B: Vendor CCD')
log('Downloading 19 CCD .cif files (download-only, no git commits yet)')

// Step 1: 19 agents download only — no git operations
const downloadResults = await parallel(
  AA_CODES.map(aa => () => agent(
    `You are downloading a CCD .cif file for the proxide Sprint 14 Track B.

task_id: 260604_rotlib_multi_source
working_directory: /home/marielle/projects/proxide

## Task: download-only for amino acid ${aa}

1. Download the CCD .cif file:
   curl -fL "https://files.rcsb.org/ligands/download/${aa}.cif" \\
     -o crates/proxide-rotlib/tests/data/ccd/${aa}.cif

   The target directory crates/proxide-rotlib/tests/data/ccd/ ALREADY EXISTS (PRO.cif is there).
   Do NOT mkdir — the directory exists.

2. Verify the download:
   ls -lh crates/proxide-rotlib/tests/data/ccd/${aa}.cif
   - Must be > 500 bytes
   - Must contain _chem_comp.id field:
     grep "_chem_comp.id" crates/proxide-rotlib/tests/data/ccd/${aa}.cif | head -1

3. Return: aa_code="${aa}", local_path (absolute), success (bool), error (if failed).

## CRITICAL CONSTRAINTS
- Do NOT run git add, git commit, or any git command
- Do NOT run cargo build, cargo check, or any compilation
- Download ONLY the .cif file for ${aa} — nothing else`,
    { label: `download-${aa}`, phase: 'Track B: Vendor CCD', schema: CCD_DOWNLOAD_SCHEMA }
  ))
)

const failedDownloads = downloadResults.filter(Boolean).filter(r => !r.success)
const succeededDownloads = downloadResults.filter(Boolean).filter(r => r.success)

log(`CCD downloads: ${succeededDownloads.length}/${AA_CODES.length} succeeded`)
if (failedDownloads.length > 0) {
  log(`Failed: ${failedDownloads.map(r => `${r.aa_code}(${r.error ?? 'unknown'})`).join(', ')}`)
}

if (succeededDownloads.length < 17) {
  throw new Error(
    `Too many CCD download failures (${failedDownloads.length}/${AA_CODES.length}). ` +
    `Cannot proceed. Failed: ${failedDownloads.map(r => r.aa_code).join(', ')}`
  )
}

// Step 2: Single agent commits ALL downloaded .cif files atomically
log('Committing all downloaded .cif files in a single atomic commit')

const commitCcdResult = await agent(
  `You are making a single atomic git commit of all downloaded CCD .cif files for proxide Sprint 14.

task_id: 260604_rotlib_multi_source
working_directory: /home/marielle/projects/proxide

## Context
${succeededDownloads.length} CCD .cif files were just downloaded to:
  crates/proxide-rotlib/tests/data/ccd/

## Task
1. Verify the files are present and untracked:
   git status --short crates/proxide-rotlib/tests/data/ccd/

2. Stage all new .cif files (ONLY .cif files in that directory):
   git add crates/proxide-rotlib/tests/data/ccd/*.cif

3. Verify staging (should show 19 new files, all .cif):
   git status --short crates/proxide-rotlib/tests/data/ccd/

4. Commit:
   git commit -m "chore(#1113): vendor 19 CCD .cif files from RCSB CCD (CC0) [260604_rotlib_multi_source]"

5. Verify commit:
   git log --oneline -1

Return the commit SHA.

## Constraints
- Stage ONLY *.cif files in crates/proxide-rotlib/tests/data/ccd/
- Do NOT stage any other files
- Do NOT run cargo build or any compilation`,
  { label: 'commit-ccd-files', phase: 'Track B: Vendor CCD' }
)

log(`CCD commit: ${typeof commitCcdResult === 'string' ? commitCcdResult.slice(0, 150) : 'done'}`)

// ─────────────────────────────────────────────────────────────────────────────
// Phase 2: Regenerate pb.zst with CCD geometry
// ─────────────────────────────────────────────────────────────────────────────

phase('Regenerate pb.zst')
log(`Regenerating pb.zst with --ic-source ccd → ${CCD_PB_DEST}`)

const regenResult = await agent(
  `You are regenerating the proxide rotamer library with CCD geometry (Track A complete, #1112).

task_id: 260604_rotlib_multi_source
working_directory: /home/marielle/projects/proxide

## Context
- convert_rotlib now accepts --ic-source (default: "ccd") — Track A is done
- 19 CCD .cif files are in crates/proxide-rotlib/tests/data/ccd/ — Track B is done
- CCD mode: CHARMM overrides disabled; CCD template geometry retained

## Steps

1. Verify --ic-source exists:
   cargo run -p proxide-rotlib --bin convert_rotlib -- --help 2>&1 | grep -A3 "ic.source"
   STOP and return success=false if --ic-source is not shown — Track A must be verified first.

2. Verify the Dunbrack input file exists (convert_rotlib needs it; default path):
   ls -lh data/rotlibs/SimpleOpt1-5/ALL.bbdep.rotamers.lib
   STOP and return success=false with error="Dunbrack source library not found" if this fails.

3. Run convert_rotlib with CCD default (omit --ic-source; it defaults to "ccd"):
   cargo run --release -p proxide-rotlib --bin convert_rotlib -- \\
     --output ${CCD_PB_PATH} \\
     2>&1 | tee /home/marielle/.claude/jobs/bd147e9b/tmp/convert_rotlib.log
   grep -E "^error\\[|^error|Finished|IC source|CHARMM IC" \\
     /home/marielle/.claude/jobs/bd147e9b/tmp/convert_rotlib.log | head -30

4. Verify output:
   ls -lh ${CCD_PB_PATH}
   (must be > 1MB; existing CHARMM pb.zst at data/rotlibs/proxide-rotlib-bbdep2010.pb.zst is ~20MB for reference)

5. Validate the pb.zst is a well-formed zstd-compressed file (proxide-rotlib tests use a
   different env var and don't exercise this file directly — verify at the file level):
   zstd -d ${CCD_PB_PATH} -o /dev/null && echo "zstd_valid=true" || echo "zstd_valid=false"
   (If zstd command is unavailable: python3 -c "import zstandard,open; zstandard.ZstdDecompressor().read_to_iter(open('${CCD_PB_PATH}','rb')).__next__()")
   STOP and return success=false if decompression fails.
   Note: full load_pb() validation occurs in Phase 3 drift test using the CHARMM pb.zst;
   CCD pb.zst is validated end-to-end in Phase 4 calibration.

6. Copy to project location:
   cp ${CCD_PB_PATH} ${CCD_PB_DEST}
   ls -lh ${CCD_PB_DEST}

7. Stage and commit the new pb.zst:
   git add ${CCD_PB_DEST}
   git commit -m "chore(#1113): add CCD-geometry pb.zst (Dunbrack+CCD, no CHARMM) [260604_rotlib_multi_source]"
   git log --oneline -1

Return: success (bool), output_path (=${CCD_PB_DEST}), file_size_bytes, smoke_test_passed (bool), commit_sha, error if failed.`,
  { label: 'regen-pbzst', phase: 'Regenerate pb.zst', schema: REGEN_SCHEMA }
)

if (!regenResult?.success) {
  throw new Error(
    `pb.zst regeneration failed: ${regenResult?.error ?? 'unknown error'}. ` +
    `Check /home/marielle/.claude/jobs/bd147e9b/tmp/convert_rotlib.log`
  )
}
log(`pb.zst regenerated: ${regenResult?.output_path}, size=${regenResult?.file_size_bytes ?? '?'} bytes, smoke=${regenResult?.smoke_test_passed}`)

// ─────────────────────────────────────────────────────────────────────────────
// Phase 3: Track C — Migrate confind binary to load_pb() (#869)
// load() is in crates/proxide-confind/src/bin/confind.rs:22
// load_pb() is defined in crates/proxide-rotlib/src/rotlib.rs:191
// ─────────────────────────────────────────────────────────────────────────────

phase('Track C: Migrate confind')
log('Migrating confind binary: load() → load_pb() in crates/proxide-confind/src/bin/confind.rs')

const migrateResult = await agent(
  `You are migrating the proxide confind binary to use the CCD-sourced rotamer library (Track C, #869).

task_id: 260604_rotlib_multi_source
working_directory: /home/marielle/projects/proxide

## Context
- CCD pb.zst is at: ${CCD_PB_DEST}
- The drift test (measure_loadpb_drift_vs_master) uses PROXIDE_ROTLIB_PB env var + load_pb()
- The confind BINARY (not the library) must be migrated; the library struct (confind.rs) is unchanged

## Exact locations
- Binary entry point to change: crates/proxide-confind/src/bin/confind.rs (the file with fn main())
- load_pb() definition: crates/proxide-rotlib/src/rotlib.rs:191
- Drift test: crates/proxide-confind/tests/test_drift_loadpb_small_pdb.rs

## Steps

1. Read crates/proxide-confind/src/bin/confind.rs in full.
   Find: let rotlib = Arc::new(RotamerLibrary::load(&opts.rlib)?);  ← line ~22
   This is the ONLY change needed in this file.

2. Read crates/proxide-rotlib/src/rotlib.rs lines 185-210 to confirm load_pb() signature.
   Verify it takes &Path and returns Result<Self, RotlibError> (same as load()).

3. Edit crates/proxide-confind/src/bin/confind.rs (edit only, not Write):
   Change ONLY line ~22:
     BEFORE: let rotlib = Arc::new(RotamerLibrary::load(&opts.rlib)?);
     AFTER:  let rotlib = Arc::new(RotamerLibrary::load_pb(&opts.rlib)?);
   No other changes to this file.

4. Compile check (debug + release):
   cargo check -p proxide-confind 2>&1 | grep -E "^error" | head -20
   STOP if errors — do not continue past compile failures.
   cargo build --release -p proxide-confind 2>&1 | grep -E "^error" | head -20
   STOP if release build errors.

5. Drift plumbing gate (uses CHARMM pb.zst — same geometry as the Mosaist rotlib.bin reference):
   The drift test compares load_pb() output against Mosaist-derived reference values that were
   generated from the CHARMM/Mosaist rotlib.bin. To verify load_pb() plumbing is correct, run
   the test with the EXISTING CHARMM pb.zst (NOT the new CCD pb.zst) — they share geometry.

   PROXIDE_ROTLIB_PB=data/rotlibs/proxide-rotlib-bbdep2010.pb.zst \\
   PDB_PATH=/home/marielle/repos/mosaist/testfiles/small.pdb \\
   cargo test -p proxide-confind measure_loadpb_drift_vs_master -- --ignored --nocapture \\
   2>&1 | tee /home/marielle/.claude/jobs/bd147e9b/tmp/drift_charmm.log

   Parse: grep -E "PASSED|FAILED|max.delta|test result" /home/marielle/.claude/jobs/bd147e9b/tmp/drift_charmm.log | head -10
   The test MUST show "test result: ok. 1 passed" — STOP if it shows FAILED.
   Record max|Δ| (should be < 5e-4 since geometries match).

6. Drift informational run (CCD pb.zst — expected to drift from CHARMM reference; NOT a gate):
   PROXIDE_ROTLIB_PB=${CCD_PB_PATH} \\
   PDB_PATH=/home/marielle/repos/mosaist/testfiles/small.pdb \\
   cargo test -p proxide-confind measure_loadpb_drift_vs_master -- --ignored --nocapture \\
   2>&1 | tee /home/marielle/.claude/jobs/bd147e9b/tmp/drift_ccd.log

   grep -E "max.delta|exceeding|test result" /home/marielle/.claude/jobs/bd147e9b/tmp/drift_ccd.log | head -10
   Record max|Δ| from CCD run — informational only, do NOT fail on this value.
   (Some drift is expected because CCD geometry differs from CHARMM by design.)

7. License gate: verify no CC-BY-NC-SA artifacts are tracked:
   git ls-files | grep -E "(rotlib\\.bin|\\.master\\.pb\\.zst)"
   (must be empty)

8. Stage and commit ONLY bin/confind.rs:
   git add crates/proxide-confind/src/bin/confind.rs
   git commit -m "feat(#869): migrate confind binary to load_pb() — retire CC-BY-NC-SA rotlib.bin [260604_rotlib_multi_source]"
   git log --oneline -1

Return: success (bool), files_changed (["crates/proxide-confind/src/bin/confind.rs"]),
drift_test_passed (bool — the CHARMM plumbing gate from step 5),
drift_max_delta (f64 — max|Δ| from the CHARMM plumbing run),
drift_ccd_max_delta (f64 — informational only, from the CCD run),
license_gate_clean (bool),
commit_sha, error if failed.

## Constraints
- Edit only — do NOT use Write on confind.rs
- Only one line changes in bin/confind.rs — nothing else
- STOP if the CHARMM plumbing drift test (step 5) fails
- Do NOT stop on CCD drift (step 6) regardless of max|Δ| — it is informational only`,
  { label: 'migrate-confind', phase: 'Track C: Migrate confind', schema: MIGRATE_SCHEMA }
)

if (!migrateResult?.success) {
  throw new Error(
    `Track C migration failed: ${migrateResult?.error ?? 'unknown'}. ` +
    `Drift test passed=${migrateResult?.drift_test_passed}, max|Δ|=${migrateResult?.drift_max_delta ?? '?'}. ` +
    `See /home/marielle/.claude/jobs/bd147e9b/tmp/drift_test.log`
  )
}
if (!migrateResult?.drift_test_passed) {
  throw new Error(
    `Drift gate FAILED: measure_loadpb_drift_vs_master did not pass. ` +
    `max|Δ|=${migrateResult?.drift_max_delta}. Do NOT proceed to calibration.`
  )
}
log(`Track C done. Drift gate passed, max|Δ|=${migrateResult?.drift_max_delta ?? '<reported in log>'}, sha=${migrateResult?.commit_sha ?? 'unknown'}`)

// ─────────────────────────────────────────────────────────────────────────────
// Phase 4: Calibrate on 1DC7
// Uses the migrated binary with --rLib pointing to CCD pb.zst
// Correct flags: --p (pdb), --rLib (lib), --o (output), --cdcut (threshold)
// ─────────────────────────────────────────────────────────────────────────────

phase('Calibrate 1DC7')
log('Running migrated confind on 1DC7 at thresholds: 0.001, 0.005, 0.01, 0.05')

const THRESHOLDS = [0.001, 0.005, 0.01, 0.05]

const calibResult = await agent(
  `You are calibrating the proxide confind contact detector on 1DC7.

task_id: 260604_rotlib_multi_source
working_directory: /home/marielle/projects/proxide

## Context
- confind binary now uses load_pb() (migration complete)
- CCD pb.zst is at: ${CCD_PB_DEST}
- 1DC7.pdb is at: ${DC7_PDB} (use this directly; do NOT download from RCSB)

## Calibration: 4 thresholds
For each threshold in [0.001, 0.005, 0.01, 0.05], run confind and count "contact" lines.

## Steps

1. Verify 1DC7.pdb exists:
   ls -lh ${DC7_PDB}
   STOP and report if not found.

2. Run confind four times, once per threshold, with EXACT filenames as shown:

   Threshold 0.001:
   cargo run --release -p proxide-confind -- \\
     --p ${DC7_PDB} --rLib ${CCD_PB_DEST} \\
     --o /home/marielle/.claude/jobs/bd147e9b/tmp/1dc7_contacts_0001.tsv --cdcut 0.001 \\
     2>/home/marielle/.claude/jobs/bd147e9b/tmp/confind_1dc7_0001.log
   grep -c "^contact" /home/marielle/.claude/jobs/bd147e9b/tmp/1dc7_contacts_0001.tsv

   Threshold 0.005:
   cargo run --release -p proxide-confind -- \\
     --p ${DC7_PDB} --rLib ${CCD_PB_DEST} \\
     --o /home/marielle/.claude/jobs/bd147e9b/tmp/1dc7_contacts_0005.tsv --cdcut 0.005 \\
     2>/home/marielle/.claude/jobs/bd147e9b/tmp/confind_1dc7_0005.log
   grep -c "^contact" /home/marielle/.claude/jobs/bd147e9b/tmp/1dc7_contacts_0005.tsv

   Threshold 0.01:
   cargo run --release -p proxide-confind -- \\
     --p ${DC7_PDB} --rLib ${CCD_PB_DEST} \\
     --o /home/marielle/.claude/jobs/bd147e9b/tmp/1dc7_contacts_001.tsv --cdcut 0.01 \\
     2>/home/marielle/.claude/jobs/bd147e9b/tmp/confind_1dc7_001.log
   grep -c "^contact" /home/marielle/.claude/jobs/bd147e9b/tmp/1dc7_contacts_001.tsv

   Threshold 0.05:
   cargo run --release -p proxide-confind -- \\
     --p ${DC7_PDB} --rLib ${CCD_PB_DEST} \\
     --o /home/marielle/.claude/jobs/bd147e9b/tmp/1dc7_contacts_005.tsv --cdcut 0.05 \\
     2>/home/marielle/.claude/jobs/bd147e9b/tmp/confind_1dc7_005.log
   grep -c "^contact" /home/marielle/.claude/jobs/bd147e9b/tmp/1dc7_contacts_005.tsv

   After each run, check exit code (must be 0) and record the contact count.

3. Count residues from the 0.05 run's SEQUENCE line:
   grep "^SEQUENCE:" /home/marielle/.claude/jobs/bd147e9b/tmp/1dc7_contacts_005.tsv | tr ' ' '\\n' | wc -l

4. Return CALIBRATION_SCHEMA with all 4 thresholds, each entry containing:
   - cdcut (the threshold value)
   - contact_count (number of "contact" lines)
   - run_succeeded (bool — exit code 0)
   - output_file (path to the .tsv)
   Also: residue_count (from SEQUENCE line), notes (any warnings or issues).`,
  { label: 'calibrate-1dc7', phase: 'Calibrate 1DC7', schema: CALIBRATION_SCHEMA }
)

const allRansOk = calibResult?.thresholds?.every(t => t.run_succeeded) ?? false
log(`Calibration: ${calibResult?.thresholds?.length ?? 0} thresholds run, all_ok=${allRansOk}`)
log(`Contact counts: ${calibResult?.thresholds?.map(t => `@${t.cdcut}:${t.contact_count ?? '?'}`).join(', ') ?? 'none'}`)
if (calibResult?.notes) log(`Calibration notes: ${calibResult.notes}`)

if (!allRansOk) {
  throw new Error(
    `Calibration runs failed for some thresholds. ` +
    `Failed: ${calibResult?.thresholds?.filter(t => !t.run_succeeded).map(t => t.cdcut).join(', ')}`
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 5: Adversarial verification (parallel — two independent skeptics)
// Gate: methodology soundness + numeric sanity (no MASTER reference required)
// ─────────────────────────────────────────────────────────────────────────────

phase('Verify Calibration')
log('Spawning 2 adversarial verifiers to audit calibration methodology and numeric sanity')

const calibJSON = JSON.stringify(calibResult, null, 2)

const verifierPrompt = (idx) => `You are adversarial verifier #${idx} for proxide Sprint 14 calibration.

task_id: 260604_rotlib_multi_source
working_directory: /home/marielle/projects/proxide

## Your role
You are a SKEPTIC. Audit the METHODOLOGY of the calibration results and determine whether
the contact degree numbers should be trusted. There is no MASTER reference file — evaluate
methodology and numeric sanity instead.

## Calibration results under review
${calibJSON}

## What to check

### 1. Threshold application correctness (check source code)
Read crates/proxide-confind/src/confind.rs (the contacts() method) to understand how
cd_cut is applied to the contact list. Is it > cd_cut or >= cd_cut?
Verify the calibration used --cdcut which maps to cd_cut in the binary's Opts struct.
Report whether contacts are correctly filtered by the threshold.

### 2. Threshold monotonicity
Contact counts MUST decrease (or stay equal) as threshold increases.
With thresholds [0.001, 0.005, 0.01, 0.05]:
  count(0.001) >= count(0.005) >= count(0.01) >= count(0.05)
If this is violated, the threshold application is wrong. Report specific violations.

### 3. Plausible contact counts for 1DC7
1DC7 has ${calibResult?.residue_count ?? '~100'} residues. Unique residue pairs: ~N*(N-1)/2.
At CD > 0.01, expect 50–500 contacts (biological range for a protein this size).
At CD > 0.05, expect fewer than at CD > 0.01.
Suspicious: zero contacts at any threshold, or > 5000 contacts at CD > 0.01.
Report whether counts are in plausible biological range.

### 4. Binary format check
The binary now uses load_pb() (migration complete). The pb.zst was generated with CCD
geometry (--ic-source ccd, CHARMM overrides disabled).
Read crates/proxide-rotlib/src/rotlib.rs lines 191-220 to verify load_pb() reads protobuf format.
If the binary called load() with a .pb.zst file, it would silently produce garbage.
Verify the binary does call load_pb() (check crates/proxide-confind/src/bin/confind.rs:22).

### 5. Gate assessment
gate_passes = true ONLY if ALL of the following:
  - All 4 threshold runs succeeded (exit 0, non-empty output)
  - Threshold monotonicity holds
  - Contact counts are in plausible biological range (not zero, not > 10000)
  - Binary verified to use load_pb() at line ~22 of bin/confind.rs
  - No evidence of threshold application inversion

gate_passes = false if ANY of the above fail.

Return verdict (CONFIRMED / REFUTED / UNCERTAIN), reasoning (2-4 sentences per concern),
gate_passes (bool), concerns (specific issues found).`

const [verifier1, verifier2] = await parallel([
  () => agent(verifierPrompt(1), { label: 'verify-calib-1', phase: 'Verify Calibration', schema: VERIFY_SCHEMA }),
  () => agent(verifierPrompt(2), { label: 'verify-calib-2', phase: 'Verify Calibration', schema: VERIFY_SCHEMA }),
])

log(`Verifier 1: ${verifier1?.verdict ?? 'null'}, gate_passes=${verifier1?.gate_passes}`)
log(`Verifier 2: ${verifier2?.verdict ?? 'null'}, gate_passes=${verifier2?.gate_passes}`)
if (verifier1?.concerns?.length) log(`V1 concerns: ${verifier1.concerns.join('; ')}`)
if (verifier2?.concerns?.length) log(`V2 concerns: ${verifier2.concerns.join('; ')}`)

const gateOk = verifier1?.gate_passes && verifier2?.gate_passes

if (!gateOk) {
  const reasons = []
  if (!verifier1?.gate_passes) reasons.push(`V1 (${verifier1?.verdict}): ${verifier1?.reasoning ?? 'no reasoning'}`)
  if (!verifier2?.gate_passes) reasons.push(`V2 (${verifier2?.verdict}): ${verifier2?.reasoning ?? 'no reasoning'}`)
  throw new Error(
    `CALIBRATION GATE FAILED.\n\n${reasons.join('\n\n')}\n\nCalibration data:\n${calibJSON}`
  )
}

log('Calibration gate PASSED. Proceeding to final tests.')

// ─────────────────────────────────────────────────────────────────────────────
// Phase 6: Final Test
// Drift gate: measure_loadpb_drift_vs_master (--ignored, correct test name)
// Workspace tests, CC-NC scan
// ─────────────────────────────────────────────────────────────────────────────

phase('Final Test')
log('Running drift gate, workspace tests, CC-NC scan')

const testResult = await agent(
  `You are running the final verification for proxide Sprint 14.

task_id: 260604_rotlib_multi_source
working_directory: /home/marielle/projects/proxide

## Task: final integration verification

### Step 1: Drift gate (re-verify load_pb() plumbing after all changes)
The drift test compares load_pb() output against Mosaist-derived reference (CHARMM geometry).
Run with the CHARMM pb.zst (not the CCD one) — both should match to < 5e-4.
Test function: measure_loadpb_drift_vs_master (#[ignore] — must use --ignored)

   PROXIDE_ROTLIB_PB=data/rotlibs/proxide-rotlib-bbdep2010.pb.zst \\
   PDB_PATH=/home/marielle/repos/mosaist/testfiles/small.pdb \\
   cargo test -p proxide-confind measure_loadpb_drift_vs_master -- --ignored --nocapture \\
   2>&1 | tee /home/marielle/.claude/jobs/bd147e9b/tmp/final_drift.log

   grep -E "test result|PASSED|FAILED|max.delta|exceeded" /home/marielle/.claude/jobs/bd147e9b/tmp/final_drift.log | head -10

   MUST show "test result: ok. 1 passed". STOP and report if FAILED.
   Note: some drift between CCD and CHARMM pb.zst is expected (Track A changed geometry).
         The gate uses CHARMM pb.zst because the reference values in the test are CHARMM-derived.

### Step 2: Full workspace test suite
   cargo test --workspace 2>&1 | tee /home/marielle/.claude/jobs/bd147e9b/tmp/workspace_test.log
   grep -E "^test result|FAILED|^error\\[" /home/marielle/.claude/jobs/bd147e9b/tmp/workspace_test.log

### Step 3: Git state
   git status --short   # must be clean
   git log --oneline -8  # show recent commits from this sprint

### Step 4: CC-NC artifact scan
   git ls-files | grep -E "(rotlib\\.bin|\\.master\\.pb\\.zst)"  # must be empty
   grep -rl "BY-NC\\|non-commercial\\|NonCommercial" crates/ data/ 2>/dev/null | grep -v target | head
   grep -l "BY-NC" crates/proxide-rotlib/tests/data/ccd/*.cif 2>/dev/null  # must be empty

### Step 5: CCD .cif license spot-check
   grep -h "^_chem_comp.pdbx_synonyms\\|^_pdbx_chem_comp_descriptor.descriptor" \\
     crates/proxide-rotlib/tests/data/ccd/ALA.cif 2>/dev/null | head -3
   # Just verify CCD files are present and readable

Report: drift_gate (passed/failed), workspace_tests (passed/N_failed/errors),
cc_nc_scan (clean or list offending files), git_status (clean or list uncommitted),
recent_commits (last 8 from git log).`,
  { label: 'final-test', phase: 'Final Test' }
)

log(`Final test: ${typeof testResult === 'string' ? testResult.slice(0, 400) : JSON.stringify(testResult).slice(0, 400)}`)

// ─────────────────────────────────────────────────────────────────────────────
// Summary
// ─────────────────────────────────────────────────────────────────────────────

log('Sprint 14 complete.')
log(`  Track A (#1112): ic_source_verified=${trackAResult?.ic_source_arg_verified}, sha=${trackAResult?.commit_sha ?? '?'}`)
log(`  Track B (#1113): ${succeededDownloads.length}/${AA_CODES.length} CCD files vendored`)
log(`  pb.zst: ${regenResult?.output_path ?? CCD_PB_DEST}, size=${regenResult?.file_size_bytes ?? '?'} bytes`)
log(`  Track C (#869): charmm_plumbing_delta=${migrateResult?.drift_max_delta ?? '<see log>'}, ccd_drift_delta=${migrateResult?.drift_ccd_max_delta ?? '<see log>'}, sha=${migrateResult?.commit_sha ?? '?'}`)
log(`  Calibration: thresholds=${calibResult?.thresholds?.map(t => t.cdcut).join(',')}, verifiers=(${verifier1?.verdict},${verifier2?.verdict})`)

return {
  track_a_sha: trackAResult?.commit_sha,
  ccd_vendored: succeededDownloads.length,
  ccd_failed: failedDownloads.map(r => r.aa_code),
  pbzst_path: regenResult?.output_path,
  pbzst_size_bytes: regenResult?.file_size_bytes,
  drift_max_delta: migrateResult?.drift_max_delta,
  track_c_sha: migrateResult?.commit_sha,
  calibration_thresholds: calibResult?.thresholds?.map(t => ({ cdcut: t.cdcut, contacts: t.contact_count })),
  verifier1_verdict: verifier1?.verdict,
  verifier2_verdict: verifier2?.verdict,
  test_summary: typeof testResult === 'string' ? testResult.slice(0, 600) : testResult,
}
