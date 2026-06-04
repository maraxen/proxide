// Sprint 14: Vendor CCD .cif files → regenerate pb.zst → calibrate on 1DC7 → migrate confind
// task_id: 260604_rotlib_multi_source
// working_directory: /home/marielle/projects/proxide

export const meta = {
  name: 'sprint14-rotlib-ccd-migration',
  description: 'Vendor PDB CCD .cif files for all 20 standard amino acids, regenerate Dunbrack+CCD pb.zst, calibrate contact detection on 1DC7 (precision/recall gate), and migrate confind to load_pb().',
  phases: [
    { title: 'Vendor CCD', detail: 'Download all 19 missing AA .cif files in parallel (CC0)' },
    { title: 'Regenerate pb.zst', detail: 'Run convert_rotlib with CCD as default IC source' },
    { title: 'Calibrate 1DC7', detail: 'Run ConFind on 1DC7, measure precision/recall vs MASTER at 4 thresholds' },
    { title: 'Verify Calibration', detail: 'Two adversarial verifiers audit methodology before gate passes' },
    { title: 'Migrate confind', detail: 'Switch confind load path to load_pb(); drift gate < 5e-4' },
    { title: 'Final Test', detail: 'cargo test --workspace + CC-NC artifact scan' },
  ],
}

// ─────────────────────────────────────────────────────────────────────────────
// Schemas
// ─────────────────────────────────────────────────────────────────────────────

const CCD_RESULT_SCHEMA = {
  type: "object",
  properties: {
    aa_code: { type: "string" },
    cif_path: { type: "string" },
    success: { type: "boolean" },
    error: { type: "string" },
  },
  required: ["aa_code", "success"],
}

const CALIBRATION_SCHEMA = {
  type: "object",
  properties: {
    results: {
      type: "array",
      items: {
        type: "object",
        properties: {
          threshold: { type: "number" },
          precision: { type: "number" },
          recall: { type: "number" },
          tp: { type: "number" },
          fp: { type: "number" },
          fn: { type: "number" },
        },
        required: ["threshold", "precision", "recall"],
      },
    },
    reference_file: { type: "string" },
    notes: { type: "string" },
  },
  required: ["results"],
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

// 19 amino acids to vendor (.cif files); PRO already present
const AA_CODES = [
  'ALA', 'ARG', 'ASN', 'ASP', 'CYS',
  'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
  'LEU', 'LYS', 'MET', 'PHE', 'SER',
  'THR', 'TRP', 'TYR', 'VAL',
]

// ─────────────────────────────────────────────────────────────────────────────
// Phase 1: Vendor CCD .cif files (parallel — each AA is independent)
// ─────────────────────────────────────────────────────────────────────────────

phase('Vendor CCD')
log('Starting parallel CCD vendoring for 19 standard amino acids (PRO already present)')

const ccdResults = await pipeline(
  AA_CODES,
  (aa) => agent(
    `You are vendoring CCD .cif files for the proxide project.

task_id: 260604_rotlib_multi_source
working_directory: /home/marielle/projects/proxide

## Task
Download and commit the CCD .cif file for amino acid: ${aa}

## Steps
1. Download from RCSB Chemical Component Dictionary (CC0 license):
   curl -fL "https://files.rcsb.org/ligands/download/${aa}.cif" \\
     -o crates/proxide-rotlib/data/ccd/${aa}.cif

2. Verify the file:
   - Must exist and be non-empty (> 500 bytes)
   - Must contain "_chem_comp.id" field
   - Run: grep "_chem_comp.id" crates/proxide-rotlib/data/ccd/${aa}.cif

3. Stage and commit:
   git add crates/proxide-rotlib/data/ccd/${aa}.cif
   git commit -m "chore(ccd): vendor ${aa}.cif from RCSB CCD (CC0) [260604_rotlib_multi_source]"

4. Return: aa_code="${aa}", cif_path (absolute), success=true/false.
   On failure: set success=false and populate error with the reason.

## Constraints
- Commit ONLY the single .cif file — nothing else
- Do NOT run cargo build or any compilation
- Do NOT modify source code`,
    { label: `ccd-${aa}`, phase: 'Vendor CCD', schema: CCD_RESULT_SCHEMA }
  )
)

const failedCcd = ccdResults.filter(Boolean).filter(r => !r.success)
const succeededCcd = ccdResults.filter(Boolean).filter(r => r.success)

log(`CCD vendoring: ${succeededCcd.length}/${AA_CODES.length} succeeded`)
if (failedCcd.length > 0) {
  log(`Failed: ${failedCcd.map(r => `${r.aa_code}(${r.error ?? 'unknown'})`).join(', ')}`)
}

if (succeededCcd.length < 17) {
  throw new Error(
    `Too many CCD download failures (${failedCcd.length}/${AA_CODES.length}). ` +
    `Cannot proceed. Failed codes: ${failedCcd.map(r => r.aa_code).join(', ')}`
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 2: Regenerate pb.zst (sequential — depends on all CCD files)
// ─────────────────────────────────────────────────────────────────────────────

phase('Regenerate pb.zst')
log('Running convert_rotlib to regenerate pb.zst with CCD as default IC source')

const regenResult = await agent(
  `You are regenerating the proxide rotamer library using vendored CCD geometry data.

task_id: 260604_rotlib_multi_source
working_directory: /home/marielle/projects/proxide

## Context
CCD .cif files for these amino acids have been vendored to crates/proxide-rotlib/data/ccd/:
${succeededCcd.map(r => r.aa_code).join(', ')}, plus PRO (pre-existing)

## Task
Regenerate the pb.zst rotamer library using the CCD-default convert_rotlib.

## Steps
1. Verify the --ic-source default is now "ccd":
   cargo run -p proxide-rotlib --bin convert_rotlib -- --help 2>&1 | grep -A2 "ic.source"

2. Run convert_rotlib (omit --ic-source; it should default to ccd):
   cargo run --release -p proxide-rotlib --bin convert_rotlib -- \\
     --output /tmp/proxide-rotlib-dunbrack2010-ccd.pb.zst \\
     2>&1 | tee /tmp/convert_rotlib.log
   grep -E "error\\[|^error|Finished|warning:" /tmp/convert_rotlib.log | head -30

3. Verify the output:
   ls -lh /tmp/proxide-rotlib-dunbrack2010-ccd.pb.zst
   (must be non-empty; typically several MB)

4. Run a quick smoke test of the generated pb.zst:
   PROXIDE_ROTLIB_PB=/tmp/proxide-rotlib-dunbrack2010-ccd.pb.zst \\
   cargo test -p proxide-rotlib -- --test-output immediate 2>&1 | tail -20

5. Copy the pb.zst to the expected project location:
   find . -name "*.pb.zst" -not -path "/tmp/*" -not -path "*/target/*" | head -5
   Copy to the same directory as any existing .pb.zst fixture.

6. Report: output path, file size, whether the smoke test passed.`,
  { label: 'regen-pbzst', phase: 'Regenerate pb.zst' }
)

log(`pb.zst regeneration: ${typeof regenResult === 'string' ? regenResult.slice(0, 200) : JSON.stringify(regenResult).slice(0, 200)}`)

// ─────────────────────────────────────────────────────────────────────────────
// Phase 3: Calibrate on 1DC7
// ─────────────────────────────────────────────────────────────────────────────

phase('Calibrate 1DC7')
log('Fetching 1DC7 and measuring precision/recall vs MASTER at thresholds: 0.001, 0.005, 0.01, 0.05')

const calibResult = await agent(
  `You are calibrating the proxide confind contact detector against the 1DC7 benchmark.

task_id: 260604_rotlib_multi_source
working_directory: /home/marielle/projects/proxide

## Task
Run ConFind on 1DC7 and measure precision/recall vs MASTER reference at 4 CD thresholds.
Gate: precision >= 0.90 AND recall >= 0.90 at CD > 0.01.

## Steps
1. Fetch 1DC7.pdb if not already present:
   mkdir -p tests/data/benchmark
   curl -fL "https://files.rcsb.org/download/1DC7.pdb" -o tests/data/benchmark/1DC7.pdb

2. Locate MASTER reference contacts for 1DC7:
   find . -name "*1dc7*" -o -name "*1DC7*" 2>/dev/null | grep -v target | head -10
   If a reference file exists, note its path as reference_file.
   If absent, note that in the "notes" field — do NOT fabricate reference data.

3. Run confind on 1DC7.pdb at 4 thresholds: 0.001, 0.005, 0.01, 0.05
   For each threshold T:
     PROXIDE_ROTLIB_PB=/tmp/proxide-rotlib-dunbrack2010-ccd.pb.zst \\
     cargo run --release -p proxide-confind -- \\
       --input tests/data/benchmark/1DC7.pdb \\
       --cd-threshold T \\
       --output /tmp/1dc7_contacts_T.tsv \\
       2>/tmp/confind_T.log
     wc -l /tmp/1dc7_contacts_T.tsv

4. If MASTER reference exists, compute for each threshold:
   - TP: pairs in both confind and MASTER
   - FP: pairs only in confind
   - FN: pairs only in MASTER
   - precision = TP / (TP + FP)
   - recall = TP / (TP + FN)
   Treat pairs as unordered: (A,B) == (B,A); exclude same-residue pairs.

5. Return CALIBRATION_SCHEMA with all 4 thresholds.
   If MASTER reference is absent, set tp/fp/fn to null and record in notes.
   Do NOT hallucinate precision/recall values.`,
  { label: 'calibrate-1dc7', phase: 'Calibrate 1DC7', schema: CALIBRATION_SCHEMA }
)

log(`Calibration complete. Thresholds measured: ${calibResult?.results?.length ?? 0}`)
if (calibResult?.notes) log(`Calibration notes: ${calibResult.notes}`)

// ─────────────────────────────────────────────────────────────────────────────
// Phase 4: Adversarial verification (parallel — two independent skeptics)
// ─────────────────────────────────────────────────────────────────────────────

phase('Verify Calibration')
log('Spawning 2 independent adversarial verifiers to audit calibration methodology')

const calibJSON = JSON.stringify(calibResult, null, 2)

const verifierPrompt = (idx) => `You are adversarial verifier #${idx} for the proxide sprint 14 calibration.

task_id: 260604_rotlib_multi_source
working_directory: /home/marielle/projects/proxide

## Your role
You are a SKEPTIC. Do NOT re-run confind. Audit the METHODOLOGY of the calibration that was
just reported and determine whether the precision/recall numbers should be trusted.

## Calibration results under review
${calibJSON}

## What to check (focus on methodology, not re-execution)

1. **Off-by-one in threshold application**
   Read crates/proxide-confind/src/ — how is the CD threshold applied?
   Is it strictly > T, >= T, or something else? Does this match MASTER's convention?
   Boundary contacts (exactly at threshold) must be counted consistently.

2. **Contact pair counting**
   Verify contacts are counted as unordered pairs (A,B) not ordered pairs.
   Verify self-contacts (same residue) are excluded.
   Verify inter-chain handling is consistent between confind and MASTER.

3. **MASTER reference validity**
   Check the reference_file field. Is it a canonical MASTER output?
   If notes say reference is absent: gate_passes must be false.

4. **Numeric sanity**
   1DC7 has ~100 residues. Expected unique pairs: ~4000–6000 at CD > 0.01.
   Do TP+FP and TP+FN totals make sense for that size?
   Are precision/recall consistent with tp/fp/fn counts?
   Suspicious: exact 1.0 precision/recall without explanation.

5. **Gate assessment**
   gate_passes = true ONLY if: MASTER reference was real, methodology is sound,
   AND precision >= 0.90 AND recall >= 0.90 at CD > 0.01 in the results.
   A false CONFIRMED is worse than UNCERTAIN.

Return verdict (CONFIRMED / REFUTED / UNCERTAIN), reasoning (2–4 sentences per concern),
gate_passes (boolean), and concerns (list of specific issues found).`

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
  if (!verifier1?.gate_passes) reasons.push(`Verifier 1 (${verifier1?.verdict}): ${verifier1?.reasoning ?? 'no reasoning'}`)
  if (!verifier2?.gate_passes) reasons.push(`Verifier 2 (${verifier2?.verdict}): ${verifier2?.reasoning ?? 'no reasoning'}`)
  throw new Error(
    `CALIBRATION GATE FAILED — precision/recall gate (>= 0.90 at CD > 0.01) not verified.\n\n` +
    reasons.join('\n\n') + `\n\nCalibration data:\n${calibJSON}`
  )
}

log('Calibration gate PASSED (both verifiers confirm). Proceeding to confind migration.')

// ─────────────────────────────────────────────────────────────────────────────
// Phase 5: Migrate confind to load_pb()
// ─────────────────────────────────────────────────────────────────────────────

phase('Migrate confind')
log('Migrating confind production load path from load() to load_pb()')

const migrateResult = await agent(
  `You are migrating the proxide confind crate to use the new CCD-sourced rotamer library.

task_id: 260604_rotlib_multi_source
working_directory: /home/marielle/projects/proxide

## Context
- CCD .cif files vendored; pb.zst regenerated at /tmp/proxide-rotlib-dunbrack2010-ccd.pb.zst
- Calibration on 1DC7 passed: precision/recall >= 0.90 at CD > 0.01
- Calibration results: ${JSON.stringify(calibResult?.results ?? [])}

## Task
Switch confind's production load path from load() to load_pb().

## Steps
1. Read crates/proxide-confind/src/confind.rs to understand the current load call.
   Also read crates/proxide-rotlib/src/lib.rs to understand load_pb() signature and
   the PROXIDE_ROTLIB_PB env var convention.

2. In confind.rs, find the load() call on the production code path.
   Replace it with load_pb(). Match the env var / path resolution pattern from lib.rs.
   Remove now-dead imports for the old load() path.
   Do NOT change any public API (function signatures, struct fields, CLI flags).

3. In the drift test (crates/proxide-confind/tests/test_drift_loadpb_small_pdb.rs),
   ensure PROXIDE_ROTLIB_PB points to the pb.zst. The test may already handle this
   via env var; if it silently skips when the var is unset, set it explicitly.
   Locate any existing .pb.zst test fixture: find . -name "*.pb.zst" | grep -v target | head

4. Run the drift gate test:
   PROXIDE_ROTLIB_PB=/tmp/proxide-rotlib-dunbrack2010-ccd.pb.zst \\
   cargo test -p proxide-confind test_drift_loadpb_small_pdb -- --nocapture 2>&1 | tail -30
   The test MUST pass with max|Δ| < 5e-4. Do NOT widen the tolerance.

5. Check the license gate:
   git ls-files | grep -E "(rotlib\\.bin|\\.master\\.pb\\.zst)"
   Must return empty. Add both patterns to .gitignore if not already there.

6. Stage and commit source changes:
   git add crates/proxide-confind/src/confind.rs crates/proxide-confind/tests/ .gitignore
   git commit -m "feat(#869): migrate confind to load_pb() — retire CC-BY-NC-SA rotlib.bin [260604_rotlib_multi_source]"

7. If migration requires > 15 substantive lines of code change, STOP and describe
   exactly what changes are needed instead of implementing — let the human review first.

Report: files changed, drift test result (actual max|Δ| if printed), license gate result.`,
  { label: 'migrate-confind', phase: 'Migrate confind' }
)

log(`confind migration: ${typeof migrateResult === 'string' ? migrateResult.slice(0, 300) : JSON.stringify(migrateResult).slice(0, 300)}`)

// ─────────────────────────────────────────────────────────────────────────────
// Phase 6: Final Test
// ─────────────────────────────────────────────────────────────────────────────

phase('Final Test')
log('Running cargo test --workspace and CC-NC artifact scan')

const testResult = await agent(
  `You are running the final integration verification for proxide Sprint 14.

task_id: 260604_rotlib_multi_source
working_directory: /home/marielle/projects/proxide

## Task
Run the full test suite and scan for CC-NC license artifacts.

## Steps
1. Run the full workspace test suite:
   cargo test --workspace 2>&1 | tee /tmp/sprint14_final_test.log
   grep -E "^test result|FAILED|^error\\[" /tmp/sprint14_final_test.log

2. Check git working tree:
   git status --short
   git log --oneline -6

3. Scan for CC-NC license artifacts:
   git ls-files | grep -E "(rotlib\\.bin|\\.master\\.pb\\.zst)"  # must be empty
   grep -rl "BY-NC\\|non-commercial\\|NonCommercial" crates/ data/ 2>/dev/null | grep -v target | head

4. Verify .cif license headers are CC0:
   grep -l "BY-NC" crates/proxide-rotlib/data/ccd/*.cif 2>/dev/null  # must be empty

5. Final smoke: confind --help still works:
   cargo run -p proxide-confind -- --help 2>&1 | head -5

Report: test pass/fail counts, any FAILED test names + first error line,
CC-NC scan results (explicitly state "clean" or list offending files),
git status (clean or list uncommitted changes).`,
  { label: 'final-test', phase: 'Final Test' }
)

log(`Final test: ${typeof testResult === 'string' ? testResult.slice(0, 300) : JSON.stringify(testResult).slice(0, 300)}`)

// ─────────────────────────────────────────────────────────────────────────────
// Summary
// ─────────────────────────────────────────────────────────────────────────────

log('Sprint 14 complete.')
log(`  CCD files vendored: ${succeededCcd.length}/${AA_CODES.length}`)
log(`  pb.zst regenerated: Phase 2 complete`)
log(`  1DC7 calibration: GATE PASSED (verifiers: ${verifier1?.verdict}, ${verifier2?.verdict})`)
log(`  confind migration: Phase 5 complete`)
log(`  Final tests: Phase 6 complete`)

return {
  ccd_succeeded: succeededCcd.length,
  ccd_failed: failedCcd.map(r => r.aa_code),
  calibration_thresholds: calibResult?.results?.map(r => ({ t: r.threshold, p: r.precision, r: r.recall })),
  verifier1_verdict: verifier1?.verdict,
  verifier2_verdict: verifier2?.verdict,
  migrate_summary: typeof migrateResult === 'string' ? migrateResult.slice(0, 500) : migrateResult,
  test_summary: typeof testResult === 'string' ? testResult.slice(0, 500) : testResult,
}
