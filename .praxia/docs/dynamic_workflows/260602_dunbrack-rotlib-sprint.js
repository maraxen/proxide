export const meta = {
  name: 'dunbrack-rotlib-protobuf-cis-pro',
  description: 'Sprint #12: Dunbrack 2010 (ODC-BY) -> proxide protobuf+zstd rotamer library with cis-PRO (CPR). Risk-first ordering with independent reviewer audit gates. Decisions A/5%/Engh-Huber locked. Rev 2 (post oracle round 1).',
  phases: [
    { title: 'Research', detail: 'query existing NotebookLM rotamer notebooks + web/code; write synthesis.jsonl (non-blocking; feeds follow-up #820)' },
    { title: 'P0 Preflight', detail: 'assert Dunbrack input present + pin SHA256; vendor CCD PRO.cif (offline-safe)' },
    { title: 'P1 Extract', detail: 'parse ALL.bbdep.rotamers.lib (T in CPR/PRO/TPR) -> JSON' },
    { title: 'P3 Geometry', detail: 'proline chi->Cartesian via NeRF; endo/exo via r1; ring closure; Engh-Huber placeholder' },
    { title: 'A3 Geometry Audit', detail: 'independent reviewer re-runs tests + external geometry validation; GATES P4' },
    { title: 'P2 Schema', detail: 'rotlib.proto + prost(build.rs)/zstd + round-trip test' },
    { title: 'P4 Converter', detail: 'Dunbrack text -> *.rotlib.pb.zst (PRECOMPUTED coords); embeds ODC-BY attribution' },
    { title: 'P5 Loader', detail: 'load_pb + route num_rotamers/place_rotamer to CPR via one helper' },
    { title: 'A5 Routing Audit', detail: 'independent reviewer verifies AC-R by identity; re-runs full suite' },
    { title: 'P6 Verify', detail: 'cross-check vs PDB CCD PRO.cif (NOT MASTER); attribution; --all-targets clean' },
  ],
}

// ---------------------------------------------------------------------------
// Dunbrack 2010 -> protobuf rotamer library (cis-PRO) — dynamic workflow, rev 2.
// Run from repo root:  Workflow({ scriptPath: ".praxia/docs/dynamic_workflows/260602_dunbrack-rotlib-sprint.js" })
//
// Locked decisions (do NOT relitigate in agents): A=precompute coords; 5% stepdown
// (Opt1-5); cis-PRO key "CPR"; reuse proxide-geometry Nerf::place_atom (it is f32 —
// build in f32, store f64); build ONLY from Dunbrack ODC-BY text + standard geometry,
// NEVER from MASTER's CC BY-NC-SA rotlib.bin; MIT code + ODC-BY data kept separate.
//
// Ordering is RISK-FIRST: the hardest phase (P3 proline geometry) runs early, and an
// independent `reviewer` audit (A3) GATES the converter (P4). A second reviewer (A5)
// verifies cis-PRO routing by identity. Reviewer gates exist because a fixer
// self-attesting "tests pass" is NOT trusted (one false-greened earlier this sprint);
// reviewers have Bash and report their OWN measured numbers. Phases are sequential
// (real deps + avoids shared-worktree concurrent-fixer git clobber). Research is
// read-only and runs in parallel. All paths repo-relative.
// ---------------------------------------------------------------------------

const SPEC = '.praxia/docs/specs/260602_dunbrack-rotlib-protobuf-cis-pro.md'
const DATA = 'data/rotlibs/SimpleOpt1-5/ALL.bbdep.rotamers.lib'
const SYNTH = '.praxia/docs/research/260602_dunbrack-geometry-synthesis.jsonl'
// Known rotamer/geometry-relevant NotebookLM notebooks (verified to exist this session):
const NLM_NOTEBOOKS = 'PRIMARY "proxide: Rotamer Library Theory" id=171c5c8b-8bae-48c1-9e6b-6cb3a45b7a8a; SECONDARY "proxide: ConFind, Contact Degree & Protein Design" id=a2302b01-05b9-44b3-af62-58ea2e892298; "Protein Conformational State Metrics: Density-Based Evaluation" id=004a4e38-61ba-4b35-9350-45df33d33cef'

const COMMON = `
Read the spec first: ${SPEC} (authoritative; do not contradict it).
Locked: A=precompute coords; 5% stepdown; cis-PRO key "CPR"; reuse proxide-geometry
Nerf::place_atom; build ONLY from Dunbrack ODC-BY text + standard geometry, never from
MASTER's CC BY-NC-SA rotlib.bin. Edit existing files (no Write over them); Write only
new files. Keep proxide-rotlib/proxide-master warning-free (#![deny(warnings)]).
NOTE: your work will be re-verified by an INDEPENDENT reviewer that re-runs cargo
itself — do not paste fabricated test output; run the real command and report truthfully.`

const EXTRACT_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    output_path: { type: 'string' },
    residues_parsed: { type: 'array', items: { type: 'string' } },
    cpr_rotamers_per_bin: { type: 'integer' },
    n_bins_pro: { type: 'integer' },
    n_bins_cpr: { type: 'integer' },
    prob_sum_ok: { type: 'boolean' },
    grid_rectangular: { type: 'boolean' },
  },
  required: ['output_path', 'residues_parsed', 'cpr_rotamers_per_bin', 'prob_sum_ok', 'grid_rectangular'],
}

const GEOM_AUDIT_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    ac_g_pass: { type: 'boolean', description: 'auditor-measured overall AC-G verdict' },
    chi_recovery_max_err_deg: { type: 'number' },
    both_puckers_distinct: { type: 'boolean' },
    endocyclic_angles_ok: { type: 'boolean' },
    cd_n_distance_ang: { type: 'number' },
    ccd_ref_rmsd_ang: { type: 'number', description: 'RMSD vs CCD PRO.cif' },
    roundtrip_identity_ok: { type: 'boolean' },
    cargo_test_summary: { type: 'string', description: 'reviewer re-ran it' },
    notes: { type: 'string' },
  },
  required: ['ac_g_pass', 'chi_recovery_max_err_deg', 'both_puckers_distinct', 'cd_n_distance_ang', 'roundtrip_identity_ok'],
}

const ROUTING_AUDIT_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    ac_r_pass: { type: 'boolean' },
    cis_coords_identity: { type: 'boolean', description: 'cis coords == CPR entry coords' },
    num_rotamers_matches_cpr: { type: 'boolean' },
    chi1_closer_to_cis: { type: 'boolean', description: 'placed cis chi1 closer to 32.5 than 27.3 at (-180,-180)' },
    cargo_test_summary: { type: 'string' },
    notes: { type: 'string' },
  },
  required: ['ac_r_pass', 'cis_coords_identity', 'num_rotamers_matches_cpr', 'chi1_closer_to_cis'],
}

const VERIFY_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    all_tests_pass: { type: 'boolean' },
    geometry_reference_rmsd_ang: { type: 'number', description: 'rebuilt PRO vs CCD PRO.cif' },
    attribution_present: { type: 'boolean' },
    no_nc_sa_committed: { type: 'boolean' },
    cargo_test_summary: { type: 'string' },
    notes: { type: 'string' },
  },
  required: ['all_tests_pass', 'attribution_present', 'no_nc_sa_committed'],
}

// Non-blocking, read-only research — feeds a FOLLOW-UP refinement of the Engh-Huber
// placeholder (backlog #820); does NOT gate the build.
async function research() {
  return agent(
    `Research (spec ${SPEC} §7 RESEARCH FLAG, backlog #820). Do NOT edit source code; you WILL write ONE artifact: ${SYNTH}.

STEP 1 — NotebookLM (we already have rotamer notebooks). The mcp__notebooklm__* tools are deferred: load them via ToolSearch("select:mcp__notebooklm__refresh_auth,mcp__notebooklm__notebook_list,mcp__notebooklm__notebook_query"). Call refresh_auth first (tokens were refreshed on disk this session). Call notebook_list to confirm, then QUERY the known-relevant notebooks: ${NLM_NOTEBOOKS}. Use notebook_query (asks existing sources; NOT research_start) on each, asking: (a) how MASTER/SCWRL/Dunbrack derive baked Cartesian sidechain coords from chi; (b) which idealized bond-length/angle parameter set is used (Engh-Huber / CHARMM / AMBER / SCWRL); (c) how proline ring closure / Cgamma-endo-exo pucker is handled; (d) what backbone-relative frame convention places the rotamer. If any notebook_query errors (auth/headless), record the failure and continue.

STEP 2 — Triangulate: web lit-review + mosaist code (/home/marielle/repos/mosaist/src/mstrotlib.cpp) + Shapovalov-Dunbrack 2011 + SCWRL4/MASTER literature.

STEP 3 — WRITE ${SYNTH} (mkdir -p its dir): JSONL, ONE compact JSON object per line, schema:
{"source": "...", "origin": "notebooklm"|"web"|"code", "notebook_id": "...optional...", "claim": "...", "evidence": "...", "relevance_to_P3_geometry": "...", "confidence": "high"|"medium"|"low"}

RETURN a decision-record summary: which ideal-geometry params proxide should adopt to match MASTER's convention, and whether the Engh-Huber placeholder is adequate or needs replacement. This feeds follow-up #820 ONLY — it CANNOT override the locked Engh-Huber decision for THIS sprint and does NOT block cis-PRO landing.`,
    { label: 'research-master-geometry', phase: 'Research', agentType: 'librarian' }
  )
}

async function build() {
  // --- P0 Preflight (vendor reference + verify input; blocks all) -----------
  await agent(
    `P0 Preflight (spec §10 P0). Make the sprint's data prerequisites reproducible and offline-safe.
1. Assert the Dunbrack input exists at ${DATA}. It is GITIGNORED — if absent, extract it from data/rotlibs/dunbrack2010-everything.tar.zst (zstd --long=31 -dc | tar -x the SimpleOpt1-5/ALL.bbdep.rotamers.lib member). Compute and record its SHA256.
2. Vendor the validation reference: download RCSB CCD proline to crates/proxide-rotlib/tests/data/ccd/PRO.cif from https://files.rcsb.org/ligands/view/PRO.cif (public-domain CCD; license-clean) so A3/P6 read it from disk (CI/cluster is offline, reviewers have no network). Record its SHA256.
Report both paths + SHA256s. If the download is unreachable, STOP and report — do not fabricate a reference.`,
    { label: 'p0-preflight', phase: 'P0 Preflight', agentType: 'fixer' }
  )

  // --- P1 Extract -----------------------------------------------------------
  const p1 = await agent(
    `P1 Extract (spec §5/§10, backlog #814, AC-1).${COMMON}
Write TRACKED scripts/data/extract_dunbrack_rotamers.py (+ bathos .bth.toml sidecar: argparse, logging, --dry-run, input SHA256 provenance). Parse ${DATA} FILTERED to T in {CPR, PRO, TPR}; emit JSON per residue: sorted phi/psi centers, default_bin, per-bin [{prob, chi[1..n] val/sigma, r1..r4}]. Columns: T Phi Psi Count r1..r4 Probability chi1..4Val chi1..4Sig. Use uv run python.
AC-1: all three of CPR/PRO/TPR present; CPR 2 rotamers/bin; each bin Sum(prob)=1.0+/-1e-3; rectangular grid; record PRO & CPR bin counts. Report output path + measured checks.`,
    { label: 'p1-extract', phase: 'P1 Extract', agentType: 'fixer', schema: EXTRACT_SCHEMA }
  )

  // --- P3 Geometry (RISK-FIRST) --------------------------------------------
  await agent(
    `P3 Geometry — proline chi->Cartesian (spec §7/§10, backlog #816, AC-G). HIGHEST-RISK phase.${COMMON}
REUSE proxide_geometry::geometry::nerf::Nerf::place_atom (it is f32 -> build in f32, convert to f64 for storage; do NOT add a parallel f64 NeRF). Add a proline residue template (atoms N,CA,C,O,CB,CG,CD; bonds; the 4-atom def per chi) with Engh-Huber ideal bond lengths/angles as a clearly-commented PLACEHOLDER (// PLACEHOLDER: Engh-Huber; backlog #820).
RING CLOSURE — named algorithm (NOT generic): the two Dunbrack rotamers per bin ARE the Cgamma-endo/exo puckers (opposite chi signs). Build CB,CG,CD by NeRF using the rotamer's ENDOCYCLIC torsions chi1=N-CA-CB-CG, chi2=CA-CB-CG-CD, chi3=CB-CG-CD-N; select endo/exo by r1. If |CD-N - 1.47| > 0.02A after the NeRF build, run CCD: iteratively rotate chi2 (moves CG,CD) then chi3 (moves CD) in small steps toward CD-N=1.47A; MAX 100 iterations; CONVERGE when |CD-N-1.47| <= 0.02A AND each chi stays within +/-5deg of its Dunbrack value. A rotamer that cannot converge within those bounds FAILS AC-G and is NOT shipped — there is no "document the residual" fallback. Refs: Ho et al. proline pucker; Cremer-Pople.
FRAME (convention trap): store coords in frame::backbone_frame(N,CA,C) (x=CA-N, z=x x (C-CA), y=z x x, origin=CA) so place_rotamer's rigid transform is correct. Add a round-trip identity unit test: place_rotamer onto a backbone equal to the build-frame backbone returns the stored coords within 1e-2 A.
Add tests asserting AC-G (§11) with PINNED tolerances: chi recovered within +/-2 deg; BOTH puckers (r1=1,2) build and give distinct CG (>=0.5A apart); endocyclic angles +/-3 deg; CD-N within +/-0.03A of 1.47A; rebuilt PRO ring heavy-atom RMSD vs the vendored crates/proxide-rotlib/tests/data/ccd/PRO.cif (from P0) <= 0.05A; round-trip identity <= 1e-2 A. cargo test -p proxide-rotlib.`,
    { label: 'p3-geometry', phase: 'P3 Geometry', agentType: 'fixer' }
  )

  // --- A3 Geometry audit (GATES P4) ----------------------------------------
  const a3 = await agent(
    `A3 INDEPENDENT geometry audit of P3 (spec §10 A3 + §11 AC-G). You are a reviewer with Bash — do NOT trust the implementer; MEASURE yourself.
1. Re-run \`cargo test -p proxide-rotlib\` and \`cargo check -p proxide-rotlib --all-targets\`; capture the real summary.
2. Independently validate the rebuilt proline geometry against the vendored crates/proxide-rotlib/tests/data/ccd/PRO.cif (public-domain CCD, NOT MASTER) — dump/recompute coords with your OWN script if needed (uv run python). PINNED thresholds: chi recovery <=2 deg; BOTH puckers distinct (CG >=0.5A apart); endocyclic angles within 3 deg; CD-N within 0.03A of 1.47A; heavy-atom RMSD vs PRO.cif <= 0.05A; round-trip identity <= 1e-2A.
Report each AC-G sub-criterion with your MEASURED number (chi_recovery_max_err_deg, cd_n_distance_ang, ccd_ref_rmsd_ang, etc.). Set ac_g_pass=true ONLY if every pinned threshold holds. This GATES the converter (a false pass ships wrong geometry).`,
    { label: 'a3-geometry-audit', phase: 'A3 Geometry Audit', agentType: 'reviewer', schema: GEOM_AUDIT_SCHEMA }
  )
  if (!a3 || !a3.ac_g_pass) {
    log(`A3 GATE FAILED — geometry not validated (cd_n=${a3 && a3.cd_n_distance_ang}A, chi_err=${a3 && a3.chi_recovery_max_err_deg}deg, ccd_rmsd=${a3 && a3.ccd_ref_rmsd_ang}A). Blocking P4.`)
    throw new Error(`A3 geometry gate failed (AC-G not met): ${a3 ? JSON.stringify(a3) : 'no audit returned'}. Fix P3 and re-run; converter (P4) must not bake unvalidated coords.`)
  }

  // --- P2 Schema ------------------------------------------------------------
  await agent(
    `P2 Schema (spec §6/§10, backlog #815, AC-2).${COMMON}
Add crates/proxide-rotlib/proto/rotlib.proto (proto3) per spec §6 (RotamerLibrary -> ResidueEntry -> Bin -> Rotamer; chi always; Vec3 coords for GeometryMode=PRECOMPUTED; REQUIRED non-empty attribution + data_license). Wire prost via build.rs (prost-build) unless the repo already uses a committed-generated-module convention (check first, match it). Add prost (Apache-2.0) + zstd (MIT). #[cfg(test)] round-trip: build a small RotamerLibrary, prost-encode, zstd (--long-capable) compress, decompress, decode, assert equality (AC-2). cargo test -p proxide-rotlib && cargo build -p proxide-master (same deny-warnings).`,
    { label: 'p2-schema', phase: 'P2 Schema', agentType: 'fixer' }
  )

  // --- P4 Converter (only after A3 passed) ---------------------------------
  await agent(
    `P4 Converter (spec §10, backlog #817, AC-3). Runs only because A3 validated geometry.${COMMON}
Offline converter (Rust bin in proxide-rotlib, or tracked script): parse Dunbrack text (reuse P1), build PRECOMPUTED coords (P3), assemble proto RotamerLibrary, prost-encode, zstd --long -> data/rotlibs/dunbrack2010-cpr-pro.rotlib.pb.zst. MUST populate attribution (ODC-BY notice, spec §4 L3) + data_license="ODC-BY-1.0" and emit an ODC-BY license sidecar. Convert CPR + PRO (5% stepdown). AC-3: output produced, attribution non-empty.`,
    { label: 'p4-converter', phase: 'P4 Converter', agentType: 'fixer' }
  )

  // --- P5 Loader + routing --------------------------------------------------
  await agent(
    `P5 Loader + routing (spec §8/§10, backlog #818, AC-R).${COMMON}
Add RotamerLibrary::load_pb(path): zstd --long decompress -> prost decode -> existing AaEntry/BinData map; reject empty attribution (RotlibError::MissingAttribution); add RotlibError::Protobuf. PRECOMPUTED -> read coords directly.
FIX ROUTING GAP: today only backbone_bin is CPR-aware. Factor effective-key resolution into ONE helper used by backbone_bin, num_rotamers, AND place_rotamer (CPR when cis_proline && aa=="PRO" && CPR present).
load_pb MUST read PRECOMPUTED coords DIRECTLY (no NeRF rebuild on the read path), so the cis path returns the stored CPR coords. Tests for AC-R (IDENTITY not inequality): with synthetic IDENTICAL PRO & CPR grids, place_rotamer(cis=true) coords EQUAL the CPR entry's stored coords within 1e-6 A (bit-identity modulo the rigid transform); num_rotamers(cis=true) EQUALS the CPR bin rotamer count; on real data at (-180,-180) the placed cis chi1 is closer to 32.5 than 27.3. cargo test -p proxide-rotlib.`,
    { label: 'p5-loader', phase: 'P5 Loader', agentType: 'fixer' }
  )

  // --- A5 Routing audit -----------------------------------------------------
  const a5 = await agent(
    `A5 INDEPENDENT routing audit of P5 (spec §10 A5 + §11 AC-R). Reviewer with Bash — MEASURE yourself, do not trust the implementer.
Re-run full \`cargo test -p proxide-rotlib\`. Verify AC-R by IDENTITY (not inequality): (1) cis coords == CPR entry stored coords within 1e-6 A (confirm load_pb does NOT rebuild coords on the read path); (2) num_rotamers(cis=true) == CPR bin count; (3) placed cis chi1 closer to 32.5 than 27.3 at (-180,-180). Confirm a SINGLE shared effective-key helper is used by backbone_bin, num_rotamers, AND place_rotamer (grep). Report measured numbers; ac_r_pass only if all hold.`,
    { label: 'a5-routing-audit', phase: 'A5 Routing Audit', agentType: 'reviewer', schema: ROUTING_AUDIT_SCHEMA }
  )

  // --- P6 Verify + docs -----------------------------------------------------
  const p6 = await agent(
    `P6 Verify + docs (spec §10/§11, backlog #819, AC-4/AC-5). Reviewer with Bash.${COMMON}
1. Cross-check rebuilt PRO coords vs PDB CCD PRO.cif (public-domain) — report geometry_reference_rmsd_ang. Do NOT use MASTER rotlib.bin as a correctness reference (license + convention-conformance risk); at most an informational dev-only note.
2. Add ODC-BY attribution notice + Shapovalov-Dunbrack 2011 citation to proxide-rotlib README/docs where data ships.
3. AC-4: cargo test -p proxide-rotlib AND cargo check -p proxide-rotlib --all-targets pass warning-free.
4. AC-5: git ls-files shows NO rotlib.bin / CC BY-NC-SA artifact committed.
Report verdict fields + test summary.`,
    { label: 'p6-verify', phase: 'P6 Verify', agentType: 'reviewer', schema: VERIFY_SCHEMA }
  )

  return { extract: p1, geometry_audit: a3, routing_audit: a5, verify: p6 }
}

// build() throws at the A3 gate on geometry failure; parallel() maps a throwing
// thunk to null, so buildOut === null signals a hard failure (loud, not silent).
const [researchOut, buildOut] = await parallel([research, build])

log(buildOut
  ? `build completed: tests_pass=${buildOut.verify && buildOut.verify.all_tests_pass}, ac_r=${buildOut.routing_audit && buildOut.routing_audit.ac_r_pass}`
  : `build FAILED (gate threw — likely A3 geometry). See run log; fix and re-run.`)

return { sprint: 'Sprint #12 (260602-rotlib-protobuf)', spec: SPEC, research: researchOut, build: buildOut, build_failed: buildOut === null }
