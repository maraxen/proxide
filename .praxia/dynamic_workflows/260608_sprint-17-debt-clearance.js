// Sprint 17 runner — emitted by `praxia dw emit-sprint`
// Source: .praxia/sprint_plans/sprint_plan.toml
// Regenerate: praxia dw emit-sprint sprint_plan.toml
// task_id: 260608_sprint-17-debt-clearance   sprint_id: 17
//
// RACE SAFETY (memory: parallel fixers race on git-status scope checks in praxia):
//   the writing chain (A,B,C,D,E) runs STRICTLY SEQUENTIAL —
//   exactly one fixer touches the working tree at a time.

export const meta = {
  name: "260608_sprint-17-debt-clearance",
  description: "Clear workspace clippy failures, ccd_parser lint, parse_master drift test stub, confind rustdoc, and confind algorithm citations.",
  phases: [
    { title: "Track A — workspace clippy: proxide-frag needless_range_loop (#1370)" },
    { title: "Track B — ccd_parser.rs clippy: while_let_on_iterator + type_complexity (#1368)" },
    { title: "Track C — parse_master --validate: wire real inline drift test (#1369)" },
    { title: "Track D — proxide-confind: rustdoc for all public API items (#1371)" },
    { title: "Track E — proxide-confind: fix algorithm citation (Grigoryan/DeGrado 2011) (#1372)" },
  ],
};

const TASK_ID = "260608_sprint-17-debt-clearance";
const MAX_FIX_RETRIES = 1;

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

// Shared context for the writing tracks (from recon, task 260608_sprint-17-debt-clearance).
const EMITTER_CTX = `Codebase: proxide (Rust workspace). Sprint 16 closed clean at 13ec61a.\nActive failures: \`cargo clippy --workspace -- -D warnings\` exits 101 due to two lint groups:\n  1. crates/proxide-rotlib/src/geometry/ccd_parser.rs — while_let_on_iterator (line 99) + type_complexity (line 125)\n  2. crates/proxide-frag/src/fragment.rs — needless_range_loop (lines 90–91)\nPreviously documented proxide-io/proxide-physics violations are already suppressed with #[allow] attributes.\nStack: cargo / tracing / rustfmt. Diagnostics via tracing::warn!/info!, never eprintln!.\n`;

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

// ===== TRACK A — Track A — workspace clippy: proxide-frag needless_range_loop (#1370) =========================
const trackA = () =>
  track(
    "1370",
    "Track A — workspace clippy: proxide-frag needless_range_loop (#1370)",
    `task_id: ${TASK_ID}. task_id: 260608_sprint-17-debt-clearance\n\n## Objective\nFix the two remaining \`cargo clippy --workspace -- -D warnings\` failures in proxide-frag.\nEdit only — no Write on existing files.\n\n## Confirmed violations\n\nFile: crates/proxide-frag/src/fragment.rs, lines 90–91\n  lint: clippy::needless_range_loop\n  90: for r in 0..N {\n  91:     for at in 0..4 {\n  92:         centered_coords[r][at][0] -= centroid[0];\n  ...\n  }\nBoth loop variables are used only as index expressions into \`centered_coords\`.\n\n## Fix strategy\nThese loops operate on a generic const-parameter array Fragment<N> using [N][[4][3]] coords.\nBecause N is a const generic, iter_mut() over a const-generic array is not ergonomic.\nThe idiomatic fix is a targeted allow at the loop site:\n  #[allow(clippy::needless_range_loop)]\n  for r in 0..N { ... }\nAdd a brief inline comment explaining the const-generic constraint so future reviewers understand why this is deliberate.\n\n## Success criteria\n\`cargo clippy -p proxide-frag -- -D warnings\` exits 0.\ncrates/proxide-rotlib and crates/proxide-io/proxide-physics are unaffected (already clean).\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. task_id: 260608_sprint-17-debt-clearance\n\n## Objective\nVerify that proxide-frag clippy violations are resolved without breaking anything else.\n\n## Checks\n1. Run: \`cargo clippy -p proxide-frag -- -D warnings\` — must exit 0.\n2. Run: \`cargo clippy --workspace -- -D warnings 2>&1 | grep "proxide-frag"\` — no proxide-frag errors.\n3. Run: \`cargo test -p proxide-frag\` — all tests pass.\n4. Verify no #[allow] was placed higher than the loop site (not on fn or struct).\n\n## Pass criterion\nproxide-frag is clippy-clean; crates outside proxide-frag are unmodified.\n`,
  );

// ===== TRACK B — Track B — ccd_parser.rs clippy: while_let_on_iterator + type_complexity (#1368) =========================
const trackB = () =>
  track(
    "1368",
    "Track B — ccd_parser.rs clippy: while_let_on_iterator + type_complexity (#1368)",
    `task_id: ${TASK_ID}. task_id: 260608_sprint-17-debt-clearance\n\n## Objective\nFix two clippy violations in ccd_parser.rs. Edit only — no Write on existing files.\n\n## Confirmed violations\n\n### Violation 1 — while_let_on_iterator\nFile: crates/proxide-rotlib/src/geometry/ccd_parser.rs, line 99\nThe tokenize() method has:\n  while let Some(c) = chars.next() { match c { ... } }\nwhere \`chars\` is a Peekable<Chars> from \`line.chars().peekable()\`.\nFix: convert to \`for c in chars.by_ref() { match c { ... } }\`.\nThe loop body (match arms for quote toggling, whitespace splitting, and char accumulation) is unchanged.\n\n### Violation 2 — type_complexity\nFile: crates/proxide-rotlib/src/geometry/ccd_parser.rs, line 125\nThe parse_cif_file() function return type is:\n  Result<(HashMap<(String, String), f32>, HashMap<(String, String, String), f32>), RotlibError>\nFix: introduce two file-private type aliases immediately above parse_cif_file:\n  type BondMap = HashMap<(String, String), f32>;\n  type AngleMap = HashMap<(String, String, String), f32>;\nThen rewrite the signature as:\n  fn parse_cif_file(path: &Path) -> Result<(BondMap, AngleMap), RotlibError>\nThe aliases must NOT be pub.\n\n## Success criteria\n\`cargo clippy -p proxide-rotlib -- -D warnings\` exits 0.\nThe tokenize() and parse_cif_file() functions behave identically (no logic change).\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. task_id: 260608_sprint-17-debt-clearance\n\n## Objective\nVerify ccd_parser.rs lint violations are resolved and logic is preserved.\n\n## Checks\n1. Run: \`cargo clippy -p proxide-rotlib -- -D warnings\` — must exit 0.\n2. Run: \`cargo test -p proxide-rotlib\` — all tests pass.\n3. Inspect the for-loop fix: confirm loop body is identical to the original (no logic change).\n4. Inspect type aliases: confirm BondMap/AngleMap are file-private (no pub), function compiles.\n\n## Pass criterion\nproxide-rotlib is clippy-clean; tokenize() and parse_cif_file() behaviour unchanged.\n`,
  );

// ===== TRACK C — Track C — parse_master --validate: wire real inline drift test (#1369) =========================
const trackC = () =>
  track(
    "1369",
    "Track C — parse_master --validate: wire real inline drift test (#1369)",
    `task_id: ${TASK_ID}. task_id: 260608_sprint-17-debt-clearance\n\n## Objective\nReplace the load-sanity stub in run_validation() with a real inline drift measurement.\nEdit only — no Write on existing files.\n\n## Current stub location\nFile: crates/proxide-rotlib/src/bin/parse_master.rs, lines 99–137\n  fn run_validation(pb_path: &PathBuf) -> Result<(), Box<dyn std::error::Error>>\nCurrent behaviour: loads .pb.zst, checks for ALA, prints "VALIDATE max|delta|: N/A PASS (stub)".\n\n## Required behaviour (spec §8 Phase 3 step 1)\n\n1. Read ROTLIB_PATH env var. If unset, warn and return Ok(()) — same as current.\n\n2. Locate small.pdb. It lives at:\n     crates/proxide-confind/tests/common/small.pdb\n   Resolve relative to CARGO_MANIFEST_DIR at compile time using:\n     const SMALL_PDB: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../proxide-confind/tests/common/small.pdb");\n   If the file does not exist at runtime, warn and return Ok(()) — do not hard-error.\n\n3. Construct a ConFind instance with the freshly written .pb.zst:\n     use proxide_confind::{ConFind, load_pdb_f64};\n     let rotlib = Arc::new(RotamerLibrary::load_pb(pb_path)?);\n     let backbone = Arc::new(load_pdb_f64(Path::new(SMALL_PDB))?);\n     let cf = ConFind::new(Arc::clone(&rotlib), Arc::clone(&backbone), false);\n   Check proxide-rotlib/Cargo.toml for proxide-confind dependency before adding it.\n\n4. Run cf.cache_all()? then call cf.contacts() for all residue pairs.\n\n5. Inline a reference subset (first 8–10 pairs) as a const array in run_validation, taken directly\n   from crates/proxide-confind/tests/test_drift_loadpb_small_pdb.rs (REF_CONTACTS).\n   Do NOT import the test module — copy the values.\n\n6. Compute max|delta| over matched pairs of |computed - reference|.\n\n7. Print:\n     VALIDATE max|delta|: {:.6}  PASS     (if max|delta| <= 1e-4)\n     VALIDATE max|delta|: {:.6}  FAIL     (if max|delta| > 1e-4, also print the worst pair)\n\n## Success criteria\nBinary compiles: \`cargo build --bin parse_master\` exits 0.\nWith ROTLIB_PATH set and small.pdb reachable, prints real numeric max|delta| and PASS/FAIL.\nWithout ROTLIB_PATH or with small.pdb absent, warns and exits 0.\nNo "stub" or "N/A" in the output path.\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. task_id: 260608_sprint-17-debt-clearance\n\n## Objective\nVerify the drift test is real (not a stub) and produces meaningful output.\n\n## Checks\n1. Read run_validation() and confirm:\n   a. The string "N/A" and "stub" do NOT appear in the printed output.\n   b. max|delta| is computed from actual ConFind outputs vs. inlined reference values.\n   c. PASS threshold is <= 1e-4.\n2. Run: \`cargo build --bin parse_master\` — must exit 0.\n3. Run: \`cargo clippy -p proxide-rotlib -- -D warnings\` — still exits 0.\n4. Verify the inlined reference const has at least 5 pairs matching REF_CONTACTS in test_drift_loadpb_small_pdb.rs.\n5. Verify graceful skip (warn + return Ok(())) when ROTLIB_PATH is unset or small.pdb is absent.\n\n## Pass criterion\nrun_validation() performs a real drift measurement; binary compiles; no new clippy errors.\n`,
  );

// ===== TRACK D — Track D — proxide-confind: rustdoc for all public API items (#1371) =========================
const trackD = () =>
  track(
    "1371",
    "Track D — proxide-confind: rustdoc for all public API items (#1371)",
    `task_id: ${TASK_ID}. task_id: 260608_sprint-17-debt-clearance\n\n## Objective\nAdd /// rustdoc to every pub fn, pub struct, and pub const in proxide-confind.\nEdit only — no Write on existing files. One file per Edit call.\n\n## Context\ncrates/proxide-confind/src/lib.rs already has a crate-level //! doc block (lines 1–37) — leave it unchanged.\n\n## Files and targets\n\n### crates/proxide-confind/src/confind.rs\n- ConFind struct (line 19): one-line summary\n- ConFind::new (line 39): describe rotlib, backbone, and \`parallel\` flag\n- ConFind::n_residues (line 78)\n- ConFind::cache_all (line 99)\n- ConFind::contact_degree_with_clashes (line 125)\n- ConFind::contacts (line 141)\n- ConFind::interference (line 172)\n- ConFind::bb_interaction (line 205)\n\n### crates/proxide-confind/src/coords.rs\n- ResidueIndex: newtype index into ProteinBackbone residue list\n- ProteinBackbone: per-residue 3D backbone coordinate store\n- ResidueBackbone: backbone atom positions for one residue\n- extract_f64_backbone: describe input and return\n- load_pdb_f64: describe path parameter and return type\n\n### crates/proxide-confind/src/cache.rs\n- ResidueCache (line 10)\n- weight_of_available_rotamers (line 25)\n- cache_residue_impl (line 95)\n\n### crates/proxide-confind/src/freedom.rs\n- compute_freedom: describe what "freedom" means in the contact-degree context\n\n### crates/proxide-confind/src/params.rs\n- aa_propensity\n- AA_NAMES, CLASH_DIST, CONT_DIST, DCUT, HI_COLL_PROB_CUT, LO_COLL_PROB_CUT: one line each, including units\n\n## Standard\nOne-line /// summary per item. For non-obvious parameters add a brief description.\nDo NOT add comments explaining what trivial code does.\nDo NOT modify any logic, #[allow], #[derive], or other attributes.\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. task_id: 260608_sprint-17-debt-clearance\n\n## Objective\nVerify all pub items have rustdoc and docs are accurate.\n\n## Checks\n1. Run: \`cargo doc -p proxide-confind --no-deps 2>&1 | grep "missing documentation"\` — zero hits.\n2. Spot-check: ConFind::contacts, compute_freedom, aa_propensity each have at least a one-line /// doc.\n3. Verify lib.rs crate-level //! block is unchanged.\n4. Run: \`cargo test -p proxide-confind\` — all tests pass.\n\n## Pass criterion\ncargo doc produces no missing-doc warnings; all tests pass.\n`,
  );

// ===== TRACK E — Track E — proxide-confind: fix algorithm citation (Grigoryan/DeGrado 2011) (#1372) =========================
const trackE = () =>
  track(
    "1372",
    "Track E — proxide-confind: fix algorithm citation (Grigoryan/DeGrado 2011) (#1372)",
    `task_id: ${TASK_ID}. task_id: 260608_sprint-17-debt-clearance\n\n## Objective\nFix the incorrect algorithm citation in proxide-confind/src/lib.rs.\nEdit only — no Write on existing files.\n\n## Current state (lib.rs lines 33–37)\n  //! # References\n  //!\n  //! - Mosaist protein design suite: <https://grigoryanlab.org/mosaist/>\n  //! - Grigoryan G, Keating AE. "Structural specificity in coiled-coil\n  //!   interactions." Curr Opin Struct Biol. 2008;18(4):477-483.\n\n## Required change\nThe Grigoryan/Keating 2008 paper describes coiled-coil specificity — not the ConFind algorithm.\nReplace with the correct citation for the contact-degree/designability metric:\n\n  //! # References\n  //!\n  //! - Mosaist protein design suite: <https://grigoryanlab.org/mosaist/>\n  //! - Grigoryan G, DeGrado WF. "Probing designability via a generalized model of helical bundle\n  //!   geometry." J Mol Biol. 2011;405(4):1079-1100.\n\nOptionally: if Track D (rustdoc) has already added a /// doc to ConFind struct in confind.rs,\nadd a brief citation reference there too:\n  /// Implements the contact-degree algorithm from Grigoryan & DeGrado (2011).\nCheck confind.rs before editing — do not duplicate if Track D already added it.\n\n## Scope\nPrimary edit: crates/proxide-confind/src/lib.rs (References block only).\nOptional edit: crates/proxide-confind/src/confind.rs (ConFind struct doc) if not already done.\nNo other files.\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. task_id: 260608_sprint-17-debt-clearance\n\n## Objective\nVerify the citation fix is accurate and complete.\n\n## Checks\n1. Read crates/proxide-confind/src/lib.rs lines 33–40: must reference "Grigoryan G, DeGrado WF" and "J Mol Biol. 2011" — NOT Keating 2008.\n2. Run: \`cargo doc -p proxide-confind --no-deps\` — zero errors.\n3. Confirm Mosaist URL is still present in the References block.\n4. Run: \`git diff --name-only HEAD\` — should show only lib.rs and optionally confind.rs.\n\n## Pass criterion\nCorrect citation in place; doc build passes; no unintended edits.\n`,
  );

// ---- orchestrate: writing chain (A -> B -> C -> D -> E, sequential) ----
log("Sprint 17 — Debt Clearance: writing chain (A -> B -> C -> D -> E, sequential)");
const a = await trackA();
const b = await trackB();
const c = await trackC();
const d = await trackD();
const e = await trackE();

return {
  task_id: TASK_ID,
  sprint_id: 17,
  verdicts: {
    "1370": a,
    "1368": b,
    "1369": c,
    "1371": d,
    "1372": e
  },
};
