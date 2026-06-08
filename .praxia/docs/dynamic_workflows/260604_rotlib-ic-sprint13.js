export const meta = {
  name: 'rotlib-ic-geometry-sprint13',
  description: 'Sprint 13: residue_geometry.proto + RTF IC parser + CCD importer + convert_rotlib wiring + parse_master binary + confind load_pb() migration. Closes #987/#988, gates #869.',
  phases: [
    { title: 'P0 Preflight', detail: 'Verify RTF at conda path; confirm CCD PRO.cif vendor; measure MET IC from RTF vs Phase D' },
    { title: 'P1 Schema', detail: 'residue_geometry.proto (IcRecord build-tree) + geometry_source/license in rotlib.proto + prost wiring + round-trip test' },
    { title: 'P2 RTF Importer', detail: 'rtf_parser.rs: parse CHARMM36 IC section -> ResidueGeometryTable; all 20 AA' },
    { title: 'A1 IC Audit', detail: 'Independent: verify RTF-extracted MET/LEU/ARG values vs Phase D measurements; GATES P3' },
    { title: 'P2B CCD Importer', detail: 'ccd_parser.rs: pairwise bonds/angles from .cif -> IC build tree (CC0 fallback path)' },
    { title: 'P3 Wire + Regenerate', detail: 'convert_rotlib --ic-source flag; replace template.rs IC values; regenerate pb.zst with RTF source' },
    { title: 'A2 Drift Gate', detail: 'CRITICAL: run drift test; max|Δ| < 5e-4 on small.pdb GATES confind migration (#869)' },
    { title: 'P4 parse_master', detail: '#988: MASTER rotlib.bin -> RotamerLibrary.pb.zst (dev-only; CC-BY-NC-SA; gitignored output)' },
    { title: 'P5 Migrate confind', detail: '#869: switch confind load() -> load_pb(); run full confind test suite' },
    { title: 'A3 Final', detail: 'cargo test --workspace; no CC-BY-NC-SA artifacts committed; attribution present' },
  ],
}

// ---------------------------------------------------------------------------
// Sprint 13: IC geometry schema + RTF/CCD importers + confind migration
// Run via: Workflow({ scriptPath: ".praxia/docs/dynamic_workflows/260604_rotlib-ic-sprint13.js" })
//
// Context (do NOT relitigate these in agents):
//  - Root cause of 0.043 max|Δ| drift is confirmed: cumulative IC error; MET CE
//    displaced 0.235 Å due to template.rs Engh-Huber placeholders diverging from
//    CHARMM IC table values (research 260603_master-rotlib-cartesian-derivation.md).
//  - charmm_ic.rs uses HARMONIC PARAMETERS from the XML (theta0=113.5° for NH1-CT1-CT2A).
//    This is the FALSIFIED Phase C approach — DO NOT EXTEND IT.
//  - The RTF IC section (IC record rows) is the correct source: these are equilibrium
//    geometry values (not force constants) used to BUILD coordinates.
//  - RTF is available locally but NOT bundled in repo (MacKerell license).
//    Point to: /home/marielle/.local/share/mamba/envs/espaloma-bench/dat/chamber/top_all36_prot.rtf
//  - MASTER rotlib.bin: /home/marielle/repos/mosaist/testfiles/rotlib.bin (CC-BY-NC-SA; dev-only).
//  - Existing pb.zst: data/rotlibs/proxide-rotlib-bbdep2010.pb.zst (Sprint 12 output).
//  - Drift test: PROXIDE_ROTLIB_PB must be absolute path; test is #[ignore]; run with --ignored.
//  - All paths repo-relative unless stated. Use uv run python for Python. cargo for Rust.
//  - proxide-rotlib and proxide-master: #![deny(warnings)].
// ---------------------------------------------------------------------------

const SPEC = '.praxia/docs/specs/260604_rotlib-ic-geometry-schema.md'
const RTF_PATH = '/home/marielle/.local/share/mamba/envs/espaloma-bench/dat/chamber/top_all36_prot.rtf'
const PB_OLD = 'data/rotlibs/proxide-rotlib-bbdep2010.pb.zst'
const PB_NEW = 'data/rotlibs/proxide-rotlib-bbdep2010-charmm36ic.pb.zst'
const CCD_DIR = 'crates/proxide-rotlib/tests/data/ccd'
const ROTLIB_BIN = '/home/marielle/repos/mosaist/testfiles/rotlib.bin'

const COMMON = `
Read the spec first: ${SPEC}. Locked context (do not relitigate):
- charmm_ic.rs uses HARMONIC params from XML (113.5°) — FALSIFIED Phase C. Do NOT extend it.
- RTF IC section is a separate format: one record per atom, gives b_ij/theta_ijk/phi/theta_jkl/b_kl.
  RTF IC records in the RESI block look like:
    IC  CA   CB   CG   SD    1.5546 115.9200  180.0000 110.2800  1.8219
    IC  CB   CG   SD   CE    1.5460 110.2800  180.0000  98.9400  1.8206
  The * prefix on atom_k marks a branch (e.g. IC CA  CB  *CG  CD2 for branched residues).
- template.rs BondDef: bond_length (Å) + bond_angle_deg + torsion_deg + relative_chi.
  We replace ONLY bond_length and bond_angle_deg from the IC table.
  torsion_deg and relative_chi encode chi rotamer placement — leave them unchanged.
- Edit existing files; Write only new files. Keep #![deny(warnings)] clean.
- Reviewer agents independently re-run cargo; do NOT trust implementer self-attestation.`

// ── Schemas ────────────────────────────────────────────────────────────────

const IC_AUDIT_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    met_sd_ce_ang:     { type: 'number', description: 'b(SD-CE) from RTF, Å' },
    met_cg_sd_ce_deg:  { type: 'number', description: 'θ(CG-SD-CE) from RTF, deg' },
    met_cb_cg_sd_deg:  { type: 'number', description: 'θ(CB-CG-SD) from RTF, deg' },
    all_20_aa_present: { type: 'boolean' },
    match_phase_d:     { type: 'boolean', description: 'RTF values match Phase D measurements within 0.01Å/0.5°' },
    residues_with_ic:  { type: 'array', items: { type: 'string' } },
    notes:             { type: 'string' },
  },
  required: ['met_sd_ce_ang', 'met_cg_sd_ce_deg', 'all_20_aa_present', 'match_phase_d'],
}

const DRIFT_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    drift_max_delta:      { type: 'number', description: 'max|Δ| over all tested pairs' },
    pairs_over_threshold: { type: 'integer', description: 'pairs with |Δ| > 5e-4 out of total' },
    pairs_total:          { type: 'integer' },
    drift_gate_pass:      { type: 'boolean', description: 'max|Δ| < 5e-4' },
    baseline_max_delta:   { type: 'number', description: '0.043 from Phase D for reference' },
    improvement_factor:   { type: 'number', description: 'baseline / new max|Δ|' },
    cargo_test_summary:   { type: 'string' },
    notes:                { type: 'string' },
  },
  required: ['drift_max_delta', 'pairs_over_threshold', 'drift_gate_pass'],
}

const FINAL_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    all_tests_pass:         { type: 'boolean' },
    no_nc_sa_committed:     { type: 'boolean' },
    attribution_present:    { type: 'boolean' },
    pb_new_geometry_source: { type: 'string', description: 'value of geometry_source field in new pb.zst' },
    cargo_test_summary:     { type: 'string' },
    notes:                  { type: 'string' },
  },
  required: ['all_tests_pass', 'no_nc_sa_committed', 'attribution_present'],
}

// ── Research (non-blocking, parallel with build) ──────────────────────────

async function research() {
  return agent(
    `Research companion for Sprint 13 (#820 partial). Read spec ${SPEC}.
Do NOT edit any source code.

TASK: Verify and document the IC table values MASTER used for the most drift-critical residues.

1. Read scripts/analysis/audit_met_ic_vs_master.py and scripts/analysis/audit_leu_ic_vs_master.py
   to understand the Phase D methodology for measuring IC values from MASTER's rotlib.bin.

2. Run an extended IC audit for the 17 interior-residue pairs that were drifting.
   Using ROTLIB_BIN=${ROTLIB_BIN}, write a NEW tracked script:
   scripts/analysis/audit_all_ic_vs_master.py
   that measures b(atom-parent) and theta(atom-parent-grandparent) for ALL 20 standard AA
   from MASTER's pre-baked Cartesian coordinates.
   Output: scripts/analysis/audit_all_ic_vs_master_output.jsonl (one line per atom, per residue)

3. Cross-check the measured values against the RTF at ${RTF_PATH}:
   For each residue, compare RTF IC record values with MASTER-measured values.
   Document the delta and flag any residue where RTF differs from MASTER by >0.01Å or >0.5°.

Write findings to .praxia/docs/research/260604_all-ic-vs-master-audit.md.
Return a summary: which residues have meaningful CHARMM-IC vs MASTER discrepancy, and
what confidence level we have that applying RTF values will fully close the drift.`,
    { label: 'research-all-ic-audit', phase: 'P0 Preflight', agentType: 'librarian' }
  )
}

// ── Main build pipeline (sequential) ─────────────────────────────────────

async function build() {
  // P0 — Preflight ──────────────────────────────────────────────────────────
  await agent(
    `P0 Preflight. Read spec ${SPEC}. Verify all sprint prerequisites before any code changes.

1. Confirm RTF is readable and contains IC records:
     head -200 ${RTF_PATH} | grep -A5 "RESI MET"
     grep "^IC" ${RTF_PATH} | grep -E "CB|CG|SD|CE" | head -10
   Record the exact IC values for MET's sidechain chain (CA→CB→CG→SD→CE).
   Phase D reference: b(SD-CE)=1.8206Å, θ(CG-SD-CE)=98.94°. Verify they match.

2. Vendor CCD .cif files for all 20 standard AAs (CC0 fallback path).
   They go in ${CCD_DIR}/. PRO.cif already exists. For the remaining 19, fetch from
   https://files.rcsb.org/ligands/view/<CODE>.cif for each of:
   ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE SER THR TRP TYR VAL
   If a fetch fails, record it and continue (network may be restricted; fallback is OK).
   Record SHA256 for each successfully vendored file.

3. Confirm existing pb.zst and drift test infrastructure:
     ls -lh ${PB_OLD}
     grep -c "PROXIDE_ROTLIB_PB" crates/proxide-confind/tests/test_drift_loadpb_small_pdb.rs

Report: RTF MET IC values vs Phase D reference; list of vendored CCD files; any blockers.`,
    { label: 'p0-preflight', phase: 'P0 Preflight', agentType: 'fixer' }
  )

  // P1 — Schema ─────────────────────────────────────────────────────────────
  await agent(
    `P1 Schema (spec §4, backlog #987). ${COMMON}

Two schema changes:

A. NEW FILE: crates/proxide-rotlib/proto/residue_geometry.proto
   Content per spec §4 verbatim — the ResidueGeometryTable / ResidueGeometry / IcRecord messages.
   Make sure build.rs compiles it (check if prost-build already compiles all protos in the dir
   or if you need to add it explicitly).

B. EXTEND: crates/proxide-rotlib/proto/rotlib.proto
   Add to RotamerLibrary: geometry_source (field 7, string) + geometry_license (field 8, string).
   These carry the IC source provenance in every generated pb.zst.

C. ROUND-TRIP TEST:
   In crates/proxide-rotlib/src/ (or tests/), add a #[cfg(test)] round-trip:
   build a minimal ResidueGeometryTable (one residue, one IcRecord), encode via prost,
   decode, assert equality. Same for the RotamerLibrary with the new fields populated.

cargo test -p proxide-rotlib && cargo check -p proxide-rotlib --all-targets`,
    { label: 'p1-schema', phase: 'P1 Schema', agentType: 'fixer' }
  )

  // P2 — RTF IC Parser ──────────────────────────────────────────────────────
  await agent(
    `P2 RTF IC Parser (spec §5, backlog #987). ${COMMON}

NEW MODULE: crates/proxide-rotlib/src/geometry/rtf_parser.rs

Parse the CHARMM RTF IC section into a ResidueGeometryTable protobuf.

RTF IC record format (one record per line within a RESI block):
  IC  atom_i  atom_j  [*]atom_k  atom_l   b_ij  theta_ijk  phi_ijkl  theta_jkl  b_kl
  (whitespace-delimited; the * prefix on atom_k means branch=true)

Algorithm:
1. Read the RTF line by line. Track current RESI name (RESI <name> <charge> lines).
2. On "IC " lines within a RESI block, parse the 5 atoms + 5 floats.
   atom_k: strip leading "*" and set branch=true if present.
3. For each RESI, collect all IC records into a ResidueGeometry message.
4. Wrap everything in ResidueGeometryTable with:
     source = "charmm36"
     version = "charmm36_rtf"  (or read the RTF version comment if present)
     license = "NOT-OSI: MacKerell lab academic use only; see https://mackerell.umaryland.edu/charmm_ff.shtml"
     citation = "doi:10.1021/jp973084f"

PUBLIC API:
  pub fn parse_rtf_ic_table(rtf_path: &str) -> Result<ResidueGeometryTable, RotlibError>

Handle: skip PRES (patch residue) blocks; skip IC records with dummy coords (0.0000 0.00).
Map CHARMM residue names to proxide codes using map_template_to_charmm_name from charmm_ic.rs
(or a similar mapping: HSD/HSE/HSP -> HIS, etc.).

TESTS:
- parse_rtf_ic_table("${RTF_PATH}") succeeds and returns ≥20 residues
- MET entry contains IC record for (CB,CG,SD,CE): b_kl ≈ 1.8206 Å ± 0.001, theta_jkl ≈ 98.94° ± 0.1°
- All 20 standard AA names present in result

cargo test -p proxide-rotlib`,
    { label: 'p2-rtf-importer', phase: 'P2 RTF Importer', agentType: 'fixer' }
  )

  // A1 — IC Audit (GATES P3) ────────────────────────────────────────────────
  const a1 = await agent(
    `A1 INDEPENDENT IC audit (spec §9 AC-2, #987). You are a reviewer — MEASURE yourself.
Do NOT trust P2 implementer's self-attestation; run the code and inspect the output.

1. cargo test -p proxide-rotlib -- rtf_parser (or the relevant test module). Capture output.

2. Write a small Python script (uv run python3 inline or tracked) to call the parser
   via the CLI or to directly cat/grep the RTF and verify IC values manually:
     grep "^IC" ${RTF_PATH} | awk '/^IC/ && /SD/ && /CE/ {print}'
   Phase D reference values:
     MET: b(SD-CE) = 1.8206 Å, θ(CG-SD-CE) = 98.94°, θ(CB-CG-SD) = 110.28°
   Report your measured values from the parser output AND the raw RTF grep.

3. Confirm all 20 standard AA residue names are present in the parsed table.
   List any missing residues.

4. Set match_phase_d=true ONLY if RTF values agree with Phase D measurements
   within 0.01 Å / 0.5°.

This GATES P3. A false pass means wrong geometry baked into the pb.zst.`,
    { label: 'a1-ic-audit', phase: 'A1 IC Audit', agentType: 'reviewer', schema: IC_AUDIT_SCHEMA }
  )

  if (!a1 || !a1.match_phase_d || !a1.all_20_aa_present) {
    log(`A1 GATE FAILED: match_phase_d=${a1?.match_phase_d}, all_20_aa=${a1?.all_20_aa_present}, met_sd_ce=${a1?.met_sd_ce_ang}Å. Blocking P3.`)
    throw new Error(`A1 IC audit gate failed: ${JSON.stringify(a1)}. Fix RTF parser (P2) before wiring into convert_rotlib.`)
  }
  log(`A1 PASSED: MET SD-CE=${a1.met_sd_ce_ang}Å, θ(CG-SD-CE)=${a1.met_cg_sd_ce_deg}°, all_20=${a1.all_20_aa_present}`)

  // P2B — CCD Importer (parallel opportunity — runs after A1 since same phase) ──
  await agent(
    `P2B CCD Importer (spec §6, #987). ${COMMON}

This is the CC0 / license-clean alternative IC source. Runs after A1.

NEW MODULE: crates/proxide-rotlib/src/geometry/ccd_parser.rs

Parse _chem_comp_bond (value_dist_ideal) and _chem_comp_angle (value_angle_ideal) from
.cif files in ${CCD_DIR}/ into a ResidueGeometryTable.

CIF format (relevant lines):
  _chem_comp_bond.atom_id_1
  _chem_comp_bond.atom_id_2
  _chem_comp_bond.value_dist_ideal
  (loop_ block, space-delimited)
  Similar for _chem_comp_angle with .atom_id_1 .atom_id_2 .atom_id_3 .value_angle_ideal

Algorithm:
1. For each .cif file in ${CCD_DIR}/:
   a. Read all _chem_comp_bond records → HashMap<(atom1, atom2), f32> (unordered key)
   b. Read all _chem_comp_angle records → HashMap<(atom1, center, atom3), f32>
   c. Build IC records matching the atom order in the existing ResidueTemplate
      (use standard_residue_template() from template.rs to get atom ordering + connectivity)
      For each atom in template order (idx >=3), build IcRecord with:
        atom_i = grandparent name
        atom_j = parent name
        atom_k = atom name
        atom_l = first child of atom_k (or a sibling for branch atoms)
        b_ij = bond length (i-j) from CCD
        theta_ijk = angle (i-j-k) from CCD
        phi_ijkl = 0.0 (CCD does not provide dihedral; chi placement comes from Dunbrack)
        theta_jkl = angle (j-k-l) from CCD (if l exists)
        b_kl = bond length (k-l) from CCD (if l exists)

2. Output: ResidueGeometryTable with source="pdb_ccd", license="CC0",
   citation="doi:10.1093/nar/gku1178"

Only process residues where the .cif file exists; log missing ones via tracing::warn!.

TESTS:
- parse_ccd_ic_table("${CCD_DIR}") returns entries for at least PRO and MET (if vendored)
- MET SD-CE bond from CCD ≈ expected value (1.82 Å ± 0.02)

cargo test -p proxide-rotlib`,
    { label: 'p2b-ccd-importer', phase: 'P2B CCD Importer', agentType: 'fixer' }
  )

  // P3 — Wire + Regenerate ──────────────────────────────────────────────────
  await agent(
    `P3 Wire ResidueGeometryTable into convert_rotlib + regenerate pb.zst (spec §8, #987). ${COMMON}

TWO PARTS:

PART A: New apply function in geometry/mod.rs (or a new geometry/ic_apply.rs):
  pub fn apply_ic_table(template: &mut ResidueTemplate, table: &ResidueGeometryTable)
  For each IcRecord in the ResidueGeometry matching template.code:
    Find the atom_k in template.atom_names (by name).
    If found at atom_idx, update template.bonds[atom_idx]:
      bond_length = ic_record.b_kl  (the bond being placed is k-l, so b_kl is the atom_k->parent length)
      Actually: the RTF IC record places atom_k; the relevant bond is atom_j -> atom_k (b_ij is j-k length).
      Use ic_record.b_ij for the j-k bond length and ic_record.theta_ijk for the j-k-i angle.
      Map RTF conventions carefully:
        IC i j *k l: places k; b_ij = bond(i-j); theta_ijk = angle(i-j-k); b_kl = bond(k-l)
        But in BondDef: bond_length is bond(parent->atom), bond_angle_deg is angle(grandparent-parent-atom)
        So: bond_length for atom_k = find the IC record where atom_k appears as atom_k, use b_kl
            bond_angle_deg for atom_k = theta_jkl (angle j-k-l where j=parent of k)
        Wait — re-read the RTF IC record meaning: b_kl is the bond k->l (the next bond IN the chain),
        NOT the bond being placed. The bond being placed is j->k, which is b_ij in the NEXT record's i-j.
        Check the actual MET records and work out which field maps to BondDef.bond_length for atom k.
        The spec says the IC record PLACES atom_k; b_kl is the length from k to the next atom l.
        So for atom SD in MET: IC CB CG SD CE -> b_kl = bond(SD-CE).
        For the BondDef of SD: bond_length = bond(CG-SD) = b_ij of the IC record "IC CA CB *CG SD".

        IMPLEMENTATION APPROACH: build a lookup {atom_k_name -> IcRecord} from the table.
        For each atom at atom_idx in the template (idx>=3):
          Look up ic_record where ic_record.atom_k == atom_names[atom_idx]
          bond_length = <the b_ij of the record where atom_j == parent of this atom>
                     OR b_kl of the PREVIOUS atom's record where atom_l == this atom.
          bond_angle_deg = theta_jkl (angle from the record: j=parent, k=this atom, l=child)

        Simplest: build two lookups:
          bond_lookup: (atom_j, atom_k) -> b_ij (length from j to k)
          angle_lookup: (atom_j, atom_k, atom_l) -> theta_jkl (angle j-k-l where k is the center)
        Then for each template atom_idx:
          parent = atom_names[bonds[atom_idx].parent_idx]
          bond_length from bond_lookup[(parent, this_atom)]
          grandparent = atom_names[grandparent_idx]
          child = first child atom name (atom whose parent = atom_idx)
          bond_angle_deg from angle_lookup[(parent, this_atom, child)] or theta_ijk where k=this

PART B: Extend convert_rotlib CLI with --ic-source flag:
  --ic-source rtf:<path>   -> parse RTF IC table
  --ic-source ccd:<dir>    -> parse CCD directory
  (default: no --ic-source -> keep existing Engh-Huber placeholders)

Apply the IC table to each residue template before building coordinates.
Set geometry_source and geometry_license in the output RotamerLibrary proto.

PART C: Regenerate pb.zst using RTF source:
  cargo run -p proxide-rotlib --bin convert_rotlib -- \\
    --dunbrack data/rotlibs/SimpleOpt1-5/ALL.bbdep.rotamers.lib \\
    --ic-source rtf:${RTF_PATH} \\
    --output ${PB_NEW}
The output must have geometry_source="charmm36" and geometry_license non-empty.

cargo test -p proxide-rotlib && cargo build -p proxide-rotlib`,
    { label: 'p3-wire-regenerate', phase: 'P3 Wire + Regenerate', agentType: 'fixer' }
  )

  // A2 — Drift Gate (CRITICAL, GATES P5) ────────────────────────────────────
  const a2 = await agent(
    `A2 CRITICAL DRIFT GATE (spec §9 AC-3, #869/#987). You are an INDEPENDENT reviewer.
MEASURE EVERYTHING YOURSELF — do not trust the fixer's numbers.

The drift gate is the sprint's critical acceptance criterion.
Baseline: max|Δ| = 0.043 (Phase D, Engh-Huber geometry).
Target: max|Δ| < 5e-4 (the confind parity gate).

1. Confirm ${PB_NEW} exists and was built with the CHARMM36 RTF IC source:
   Use uv run python3 to decode the pb.zst header and read geometry_source + geometry_license fields.
   (prost-encoded proto — decode with: python3 -c "import zstd,sys; data=zstd.decompress(open('${PB_NEW}','rb').read()); print(len(data),'bytes')")
   Or: cargo run -p proxide-rotlib --bin convert_rotlib -- --info ${PB_NEW}  (if such a flag exists)
   At minimum verify the file exists and is non-empty.

2. Run the drift test with an absolute path:
   PROXIDE_ROTLIB_PB=$(realpath ${PB_NEW}) \\
   cargo test -p proxide-confind -- test_drift_loadpb_small_pdb --ignored 2>&1 | tail -50

   Parse the output for: max|Δ|, number of pairs over 5e-4 threshold, total pairs.
   The test prints these values. Record them precisely.

3. Compute improvement_factor = 0.043 / new_max_delta.

4. Set drift_gate_pass = (drift_max_delta < 5e-4).

5. If the drift is reduced but NOT below 5e-4, report the new value and note which
   pairs are still drifting — this informs whether residual drift is terminal-residue
   default_bin issue (separate from IC geometry fix).

This GATES P5 (confind migration). A drift_gate_pass=false means the IC geometry fix
alone is not sufficient; do not migrate confind until this passes.`,
    { label: 'a2-drift-gate', phase: 'A2 Drift Gate', agentType: 'reviewer', schema: DRIFT_SCHEMA }
  )

  if (!a2) {
    log('A2: no result returned (reviewer error). Blocking P5.')
    throw new Error('A2 drift gate: reviewer returned null. Cannot proceed to confind migration.')
  }
  log(`A2 drift gate: max|Δ|=${a2.drift_max_delta}, pairs_over=${a2.pairs_over_threshold}/${a2.pairs_total}, pass=${a2.drift_gate_pass}, improvement=${a2.improvement_factor?.toFixed(1)}×`)

  // P4 — parse_master binary (non-blocking, no gate dependency on A2) ────────
  await agent(
    `P4 parse_master dev binary (spec §7, backlog #988). ${COMMON}

NEW FILE: crates/proxide-rotlib/src/bin/parse_master.rs

This is a DEVELOPER TOOL ONLY. Its output MUST NOT be committed (CC-BY-NC-SA data).

CLI:
  parse_master --input <rotlib.bin path> --output <out.pb.zst>

Behavior:
1. Read the input via RotamerLibrary::load(path) (the existing MASTER binary reader).
2. Convert all AaEntry/BinData entries to RotamerLibrary proto:
   - geometry_mode = PRECOMPUTED (coords already baked)
   - For each residue: populate ResidueEntry with atom_names, phi/psi centers, default_bin, bins.
   - For each bin: populate Bin with phi, psi, freq, rotamers.
   - For each rotamer: populate Rotamer with prob + coords (from BinData.coords).
   - chi values: BinData from the MASTER binary may not carry chi explicitly; leave chi empty
     if not available (PRECOMPUTED mode; coords are what matters).
3. Set on RotamerLibrary:
   - data_license = "CC-BY-NC-SA-4.0"
   - provenance = "Mosaist Grigoryan lab; testfiles/rotlib.bin; NON-COMMERCIAL ONLY; NOT FOR REDISTRIBUTION"
   - attribution = "Mosaist (https://github.com/Grigoryanlab/Mosaist), CC-BY-NC-SA 4.0, Grigoryan lab, Dartmouth"
   - geometry_source = "master_precomputed"
   - geometry_license = "CC-BY-NC-SA-4.0"
4. prost-encode + zstd-compress to output path.
5. Print to STDERR before writing: "WARNING: output is CC-BY-NC-SA 4.0 (Mosaist/Grigoryan lab). Do NOT commit or redistribute."

GITIGNORE: Add to .gitignore (repo root):
  # MASTER-derived data — CC-BY-NC-SA 4.0; NOT for redistribution
  *.master.pb.zst

TESTS: A test that runs parse_master on ${ROTLIB_BIN} (if accessible under ROTLIB_PATH env var;
skip if absent) and verifies the output:
  - geometry_mode == PRECOMPUTED
  - data_license starts with "CC-BY-NC-SA"
  - at least 18 residue entries present

cargo test -p proxide-rotlib && cargo build -p proxide-master`,
    { label: 'p4-parse-master', phase: 'P4 parse_master', agentType: 'fixer' }
  )

  // P5 — Migrate confind (only if A2 passes) ────────────────────────────────
  if (a2.drift_gate_pass) {
    await agent(
      `P5 Migrate confind to load_pb() (backlog #869). ${COMMON}
A2 drift gate PASSED (max|Δ|=${a2.drift_max_delta} < 5e-4). Proceeding with migration.

The generated pb.zst is: ${PB_NEW} (CHARMM36 IC geometry, geometry_source="charmm36").

CHANGES:
1. In crates/proxide-confind/src/confind.rs (or wherever RotamerLibrary::load() is called for
   production use — check with grep -r "RotamerLibrary::load" crates/proxide-confind/src/):
   Switch from load() to load_pb(), reading from PROXIDE_ROTLIB_PB env var.
   Keep the load() path behind a #[cfg(feature = "master-compat")] or similar optional feature
   for dev cross-checking (do NOT delete the MASTER binary reader entirely).

2. Update any test infrastructure that sets up the RotamerLibrary to use load_pb() +
   PROXIDE_ROTLIB_PB env var pointing to ${PB_NEW}.

3. The drift test (test_drift_loadpb_small_pdb.rs) already uses PROXIDE_ROTLIB_PB —
   verify it still passes with the new pb.zst.

4. Run: cargo test -p proxide-confind (both ignored and non-ignored suites).
   Report: which tests pass/fail; any new failures introduced by the migration.

DO NOT change the contact-degree algorithm logic — only the library loading path.`,
      { label: 'p5-migrate-confind', phase: 'P5 Migrate confind', agentType: 'fixer' }
    )
  } else {
    log(`P5 SKIPPED: A2 drift gate did not pass (max|Δ|=${a2.drift_max_delta}). ` +
        `Confind stays on load() until drift is resolved. ` +
        `Residual drift pairs: ${a2.pairs_over_threshold}/${a2.pairs_total} over threshold. ` +
        `Investigate: are remaining drifting pairs terminal-residue default_bin mismatches (deferred) ` +
        `or a new IC geometry gap?`)
  }

  // A3 — Final verification ─────────────────────────────────────────────────
  const a3 = await agent(
    `A3 Final verification. Independent reviewer. ${COMMON}

1. cargo test --workspace 2>&1 | tail -30   (run the full suite; report summary)
2. cargo check --workspace --all-targets    (check for warnings — deny(warnings) is active)
3. Verify no CC-BY-NC-SA artifacts committed:
     git ls-files | grep -E "rotlib\\.bin|\\.master\\.pb\\.zst|top_all36.*\\.rtf"
   Must return empty.
4. Verify attribution is present somewhere in proxide-rotlib docs/README:
     grep -r "ODC-BY\|Dunbrack\|Richardson" crates/proxide-rotlib/README.md 2>/dev/null || echo "missing"
5. Decode the new pb.zst header and confirm geometry_source field is populated:
     PROXIDE_ROTLIB_PB=$(realpath ${PB_NEW}) cargo run -p proxide-rotlib --bin convert_rotlib -- --info 2>/dev/null || true
   Or read the geometry_source from the proto with a small Python snippet.

Report all_tests_pass, no_nc_sa_committed, attribution_present, pb_new_geometry_source.`,
    { label: 'a3-final', phase: 'A3 Final', agentType: 'reviewer', schema: FINAL_SCHEMA }
  )

  return {
    ic_audit: a1,
    drift: a2,
    final: a3,
    confind_migrated: a2.drift_gate_pass,
  }
}

// ── Fan-out: research + build in parallel ─────────────────────────────────
const [researchOut, buildOut] = await parallel([research, build])

log(buildOut
  ? `Sprint 13 complete. drift_gate=${buildOut.drift?.drift_gate_pass}, max|Δ|=${buildOut.drift?.drift_max_delta}, confind_migrated=${buildOut.confind_migrated}, tests=${buildOut.final?.all_tests_pass}`
  : `Sprint 13 BUILD FAILED (gate threw). Check A1 (IC audit) or A2 (drift gate) log.`)

return {
  sprint: 'Sprint 13 (#27)',
  spec: SPEC,
  research: researchOut,
  build: buildOut,
  build_failed: buildOut === null,
}
