export const meta = {
  name: 'dunbrack-rotlib-protobuf-cis-pro',
  description: 'Sprint #12: Dunbrack 2010 (ODC-BY) -> proxide protobuf+zstd rotamer library with cis-PRO (CPR). Risk-first ordering with independent reviewer audit gates. Decisions A/5%/Engh-Huber locked. Rev 2 (post oracle round 1).',
  phases: [
    { title: 'Research', detail: 'lit-review how MASTER derived rotlib.bin Cartesians (non-blocking, read-only; feeds follow-up #820)' },
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
    `Research (spec ${SPEC} §7 RESEARCH FLAG, backlog #820). READ-ONLY — report only, no edits.
How did MASTER/SCWRL/Dunbrack derive the baked Cartesian sidechain coords in rotlib.bin from Dunbrack chi? Which ideal bond-length/angle param set (Engh-Huber? CHARMM? AMBER? SCWRL?), and how is proline ring closure treated? Triangulate mosaist (/home/marielle/repos/mosaist/src/mstrotlib.cpp), Shapovalov-Dunbrack 2011, SCWRL4/MASTER literature.
Output: a decision-record recommending which ideal-geometry params proxide should adopt to match conventions, and whether the Engh-Huber placeholder is adequate or needs replacement. Feeds a follow-up sprint; does NOT block cis-PRO landing.`,
    { label: 'research-master-geometry', phase: 'Research', agentType: 'librarian' }
  )
}

async function build() {
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
RING CLOSURE — named algorithm (NOT generic): the two Dunbrack rotamers per bin ARE the Cgamma-endo/exo puckers (opposite chi signs). Build CB,CG,CD by NeRF using the rotamer's ENDOCYCLIC torsions chi1=N-CA-CB-CG, chi2=CA-CB-CG-CD, chi3=CB-CG-CD-N; select endo/exo by r1. Verify CD-N ~= 1.47A; if idealized geometry leaves residual closure error beyond tol, apply a minimal cyclic-coordinate-descent (CCD) ring closure or document residual. Refs: Ho et al. proline pucker; Cremer-Pople.
FRAME (convention trap): store coords in frame::backbone_frame(N,CA,C) (x=CA-N, z=x x (C-CA), y=z x x, origin=CA) so place_rotamer's rigid transform is correct. Add a round-trip identity unit test: place_rotamer onto a backbone equal to the build-frame backbone returns the stored coords within tol.
Add tests asserting AC-G (§11): chi recovered within +/-2 deg; BOTH puckers (r1=1,2) build and give distinct CG (>0.3A apart); endocyclic angles +/-3 deg; CD-N within +/-0.03A of 1.47A; RMSD vs idealized CCD PRO.cif <= tol; round-trip identity. cargo test -p proxide-rotlib.`,
    { label: 'p3-geometry', phase: 'P3 Geometry', agentType: 'fixer' }
  )

  // --- A3 Geometry audit (GATES P4) ----------------------------------------
  const a3 = await agent(
    `A3 INDEPENDENT geometry audit of P3 (spec §10 A3 + §11 AC-G). You are a reviewer with Bash — do NOT trust the implementer; MEASURE yourself.
1. Re-run \`cargo test -p proxide-rotlib\` and \`cargo check -p proxide-rotlib --all-targets\`; capture the real summary.
2. Independently validate the rebuilt proline geometry: dump/recompute the proline rotamer coords and check, with your OWN script if needed (uv run python), against CCD PRO.cif (public-domain, NOT MASTER): chi recovery <=2 deg, BOTH puckers distinct (CG >0.3A apart), endocyclic angles within 3 deg, CD-N within 0.03A of 1.47A, RMSD vs PRO.cif within tol, and the round-trip identity test exists and passes.
Report each AC-G sub-criterion with your measured number. Set ac_g_pass=true ONLY if all hold. This GATES the converter.`,
    { label: 'a3-geometry-audit', phase: 'A3 Geometry Audit', agentType: 'reviewer', schema: GEOM_AUDIT_SCHEMA }
  )
  if (!a3 || !a3.ac_g_pass) {
    log(`A3 GATE FAILED — geometry not validated (cd_n=${a3 && a3.cd_n_distance_ang}, chi_err=${a3 && a3.chi_recovery_max_err_deg}). Halting before P4; fix P3 and re-run.`)
    return { halted_at: 'A3', geometry_audit: a3, extract: p1 }
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
Tests for AC-R (strengthened, IDENTITY not inequality): with synthetic IDENTICAL PRO & CPR grids, place_rotamer(cis=true) coords EQUAL the CPR entry's stored coords; num_rotamers(cis=true) EQUALS the CPR bin rotamer count; on real data at (-180,-180) the placed cis chi1 is closer to 32.5 than 27.3. cargo test -p proxide-rotlib.`,
    { label: 'p5-loader', phase: 'P5 Loader', agentType: 'fixer' }
  )

  // --- A5 Routing audit -----------------------------------------------------
  const a5 = await agent(
    `A5 INDEPENDENT routing audit of P5 (spec §10 A5 + §11 AC-R). Reviewer with Bash — MEASURE yourself, do not trust the implementer.
Re-run full \`cargo test -p proxide-rotlib\`. Verify AC-R by IDENTITY (not inequality): (1) cis coords == CPR entry stored coords; (2) num_rotamers(cis=true) == CPR bin count; (3) placed cis chi1 closer to 32.5 than 27.3 at (-180,-180). Confirm a SINGLE shared effective-key helper is used by backbone_bin, num_rotamers, AND place_rotamer (grep). Report measured numbers; ac_r_pass only if all hold.`,
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

const [researchOut, buildOut] = await parallel([research, build])

log(`research done; build ${buildOut && buildOut.halted_at ? 'HALTED at ' + buildOut.halted_at : 'completed: tests_pass=' + (buildOut && buildOut.verify && buildOut.verify.all_tests_pass)}`)

return { sprint: 'Sprint #12 (260602-rotlib-protobuf)', spec: SPEC, research: researchOut, build: buildOut }
