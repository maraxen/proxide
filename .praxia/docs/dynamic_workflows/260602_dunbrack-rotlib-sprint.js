export const meta = {
  name: 'dunbrack-rotlib-protobuf-cis-pro',
  description: 'Sprint #12: convert Dunbrack 2010 (ODC-BY) -> proxide protobuf+zstd rotamer library with cis-PRO (CPR); precompute coords (decision A) reusing proxide-geometry NeRF; Engh-Huber placeholder geometry flagged for research.',
  phases: [
    { title: 'Research', detail: 'lit-review how MASTER derived rotlib.bin Cartesians (non-blocking, read-only)' },
    { title: 'P1 Extract', detail: 'parse cpr.bbdep.rotamers.lib -> JSON (bins/probs/chi)' },
    { title: 'P2 Schema', detail: 'rotlib.proto + prost/zstd deps + round-trip test' },
    { title: 'P3 Geometry', detail: 'proline chi->Cartesian via NeRF + Engh-Huber placeholder' },
    { title: 'P4 Converter', detail: 'Dunbrack text -> *.rotlib.pb.zst CLI (embeds ODC-BY attribution)' },
    { title: 'P5 Loader', detail: 'load_pb + route num_rotamers/place_rotamer to CPR' },
    { title: 'P6 Verify', detail: 'cross-check vs MASTER PRO (dev-only), attribution, tests green' },
  ],
}

// ---------------------------------------------------------------------------
// Dunbrack 2010 -> protobuf rotamer library (cis-PRO) — dynamic workflow.
//
// Run from the proxide repo root:  Workflow({ scriptPath: ".praxia/docs/dynamic_workflows/260602_dunbrack-rotlib-sprint.js" })
//
// Authoritative detail lives in the spec; each agent reads it. Locked decisions
// (do NOT re-litigate in agents): A=precompute coords; 5% stepdown (Opt1-5);
// cis-PRO key = "CPR"; reuse proxide-geometry Nerf::place_atom; data built ONLY
// from Dunbrack ODC-BY text + standard geometry, NEVER from MASTER's CC BY-NC-SA
// rotlib.bin; MIT code + ODC-BY data file kept separate.
//
// Phases P1..P6 are SEQUENTIAL (real dependencies; also avoids the shared-worktree
// concurrent-fixer git-clobber hazard seen earlier this sprint). The Research agent
// is read-only and runs in parallel. All paths are repo-relative.
// ---------------------------------------------------------------------------

const SPEC = '.praxia/docs/specs/260602_dunbrack-rotlib-protobuf-cis-pro.md'
const DATA = 'data/rotlibs/SimpleOpt1-5/ALL.bbdep.rotamers.lib'

const COMMON = `
Read the spec first: ${SPEC} (authoritative; do not contradict it).
Locked decisions: A=precompute coords; 5% stepdown; cis-PRO key "CPR"; reuse
proxide-geometry Nerf::place_atom; build ONLY from Dunbrack ODC-BY text + standard
geometry, never from MASTER's CC BY-NC-SA rotlib.bin. Edit existing files (no Write
over them); Write only genuinely-new files. After your change run the relevant
cargo/uv command and paste the pass/fail summary line. Keep crates warning-free
(proxide-rotlib / proxide-master have #![deny(warnings)]).`

const EXTRACT_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    output_path: { type: 'string' },
    residues_parsed: { type: 'array', items: { type: 'string' } },
    cpr_rotamers_per_bin: { type: 'integer' },
    n_bins: { type: 'integer' },
    prob_sum_ok: { type: 'boolean', description: 'per-bin probabilities sum to 1.0 +/- 1e-3' },
    grid_rectangular: { type: 'boolean' },
  },
  required: ['output_path', 'cpr_rotamers_per_bin', 'prob_sum_ok', 'grid_rectangular'],
}

const VERIFY_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    all_tests_pass: { type: 'boolean' },
    cis_vs_trans_differ: { type: 'boolean', description: 'place_rotamer PRO cis vs trans give different prob/coords' },
    geometry_valid: { type: 'boolean', description: 'AC-G: proline bonds/angles in tol, ring closes' },
    attribution_present: { type: 'boolean' },
    no_nc_sa_committed: { type: 'boolean', description: 'AC-5: no CC BY-NC-SA artifact committed' },
    test_summary: { type: 'string' },
    notes: { type: 'string' },
  },
  required: ['all_tests_pass', 'cis_vs_trans_differ', 'geometry_valid', 'attribution_present', 'no_nc_sa_committed'],
}

// Non-blocking, read-only research (runs alongside the build).
async function research() {
  return agent(
    `Research task (spec ${SPEC} §7 RESEARCH FLAG, backlog #820). Read-only — produce a findings report only, do not edit code.
Question: how did MASTER (Grigoryan lab) / SCWRL / Dunbrack derive the baked Cartesian sidechain coordinates in rotlib.bin from the Dunbrack chi angles? Which idealized bond-length/bond-angle parameter set (Engh-Huber? CHARMM? AMBER? SCWRL internal?), and how is proline ring closure handled? Triangulate code (mosaist /home/marielle/repos/mosaist src/mstrotlib.cpp), the Shapovalov-Dunbrack 2011 paper, and SCWRL4 / MASTER literature.
Output: a concise decision-record-style report recommending which ideal-geometry params proxide should adopt to match conventions, and whether the Engh-Huber placeholder (P3) is adequate or needs replacement. This refines P3 later; it does NOT block cis-PRO landing.`,
    { label: 'research-master-geometry', phase: 'Research', agentType: 'librarian' }
  )
}

// Sequential build pipeline (P1 -> P6).
async function build() {
  const p1 = await agent(
    `P1 Extract (spec §5/§10 P1, backlog #814, AC-1).${COMMON}
Write a TRACKED Python script scripts/data/extract_dunbrack_rotamers.py (+ bathos sidecar .bth.toml per the ephemeral-scripts rule: argparse, logging, --dry-run, input SHA256 provenance) that parses ${DATA} for residue codes CPR (and PRO, TPR for comparison) and emits JSON: per residue -> sorted phi/psi centers, default_bin, and per-bin list of rotamers {prob, chi[1..n] val/sigma}. Columns: T Phi Psi Count r1..r4 Probability chi1..4Val chi1..4Sig. Use uv run python.
Verify AC-1: CPR has 2 rotamers/bin; each (phi,psi) bin's probabilities sum to 1.0 +/- 1e-3; bin grid is rectangular. Report the output JSON path and the checks.`,
    { label: 'p1-extract', phase: 'P1 Extract', agentType: 'fixer', schema: EXTRACT_SCHEMA }
  )

  await agent(
    `P2 Schema (spec §6/§10 P2, backlog #815, AC-2).${COMMON}
Add crates/proxide-rotlib/proto/rotlib.proto (proto3) exactly per spec §6 (RotamerLibrary -> ResidueEntry -> Bin -> Rotamer; chi always present; Vec3 coords present for GeometryMode=PRECOMPUTED; required non-empty attribution + data_license fields). Add deps: prost (Apache-2.0) and zstd (MIT) to the workspace + proxide-rotlib (use prost-build in build.rs or a committed generated module — match repo conventions). Add a #[cfg(test)] round-trip test: build a small RotamerLibrary message, prost-encode, zstd compress (--long-capable), decompress, decode, assert equality. AC-2: lossless round-trip. cargo test -p proxide-rotlib.`,
    { label: 'p2-schema', phase: 'P2 Schema', agentType: 'fixer' }
  )

  await agent(
    `P3 Geometry (spec §7/§10 P3, backlog #816, AC-G).${COMMON}
REUSE proxide-geometry Nerf::place_atom (crates/proxide-geometry/src/geometry/nerf.rs) — do not reimplement NeRF. Add a proline residue template (atom names N,CA,C,O,CB,CG,CD; bonds; the 4-atom defs for each chi) with Engh-Huber ideal bond lengths/angles as a clearly-commented PLACEHOLDER (// PLACEHOLDER: Engh-Huber; see backlog #820 research). Implement a chi->Cartesian builder for proline that places CB,CG,CD from the Dunbrack chi values in the canonical backbone frame.
CRITICAL: proline's pyrrolidine ring (N-CA-CB-CG-CD-N) is CLOSED — chi are not independent; naive open-chain rotation will NOT close the ring. Use a proline-specific pucker/ring-closure approach parameterized by the Dunbrack chi.
AC-G: rebuilt proline sidechain bond lengths/angles within tolerance of the template; ring closure satisfied (CD-N distance in range). Add a unit test asserting AC-G. cargo test.`,
    { label: 'p3-geometry', phase: 'P3 Geometry', agentType: 'fixer' }
  )

  await agent(
    `P4 Converter (spec §10 P4, backlog #817, AC-3).${COMMON}
Add an offline converter (Rust bin target in proxide-rotlib, or a tracked script) that: parses the Dunbrack text (reuse P1 logic), builds coords via P3 (PRECOMPUTED mode), assembles the proto RotamerLibrary, prost-encodes, and zstd-compresses (--long) to data/rotlibs/dunbrack2010-cpr-pro.rotlib.pb.zst. It MUST populate attribution (the ODC-BY notice from spec §4 L3) and data_license="ODC-BY-1.0", and emit an ODC-BY license sidecar next to the output. Convert CPR + PRO for the 5% stepdown. AC-3: output is produced and its attribution field is non-empty (the P5 loader will reject empty attribution).`,
    { label: 'p4-converter', phase: 'P4 Converter', agentType: 'fixer' }
  )

  await agent(
    `P5 Loader + routing (spec §8/§10 P5, backlog #818, AC-R).${COMMON}
Add RotamerLibrary::load_pb(path) that reads the *.rotlib.pb.zst (zstd --long decompress -> prost decode -> existing AaEntry/BinData map); reject empty attribution (RotlibError::MissingAttribution); add RotlibError::Protobuf. For PRECOMPUTED, read coords directly.
FIX THE ROUTING GAP: today only backbone_bin is CPR-aware (const CIS_PRO_KEY="CPR"). Factor effective-key resolution into ONE helper and use it in backbone_bin, num_rotamers, AND place_rotamer so cis-PRO probabilities/coords are actually used when cis_proline && aa=="PRO" && CPR present.
AC-R: load the P4 output; assert place_rotamer("PRO",phi,psi,ri,cis=true,..) yields different prob/coords than cis=false; num_rotamers likewise. All proxide-rotlib tests green. cargo test -p proxide-rotlib.`,
    { label: 'p5-loader', phase: 'P5 Loader', agentType: 'fixer' }
  )

  return agent(
    `P6 Verify + docs (spec §10 P6, backlog #819, AC-4/AC-5).${COMMON}
1) Dev-only cross-check: compare rebuilt PRO coords vs the MASTER rotlib.bin PRO entry (load via ROTLIB_PATH; this is local verification only — do NOT commit or redistribute rotlib.bin) and report agreement within tolerance.
2) Add the ODC-BY attribution notice + Shapovalov-Dunbrack 2011 citation to the proxide-rotlib README / docs where the data ships.
3) AC-4: cargo test -p proxide-rotlib AND cargo check -p proxide-rotlib --all-targets pass warning-free.
4) AC-5: confirm no CC BY-NC-SA artifact (rotlib.bin) is committed (git ls-files check).
Report the verdict fields and the test summary line.`,
    { label: 'p6-verify', phase: 'P6 Verify', agentType: 'reviewer', schema: VERIFY_SCHEMA }
  )
}

const [researchOut, verify] = await parallel([research, build])

log(`build verdict: tests_pass=${verify && verify.all_tests_pass} cis!=trans=${verify && verify.cis_vs_trans_differ} geom_valid=${verify && verify.geometry_valid}`)

return {
  sprint: 'Sprint #12 (260602-rotlib-protobuf)',
  spec: SPEC,
  research: researchOut,
  verify,
}
