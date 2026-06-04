# Spec: Unified residue IC geometry schema + parse_master dev tool

- **task_id:** `260604_rotlib_ic_schema`
- **Status:** DRAFT
- **Author:** orchestrator (2026-06-04)
- **Backlog:** #987 (residue_geometry.proto + importers), #988 (parse_master)
- **Related:** #869, #987, #988; supersedes #976; see also `260602_dunbrack-rotlib-protobuf-cis-pro.md`

---

## 1. Summary

Two complementary additions to proxide-rotlib's rotamer library architecture:

1. **`residue_geometry.proto`** — a unified IC geometry schema covering bond lengths,
   valence angles, and dihedral reference frames for all 20 standard amino acid
   sidechains. Multi-source: CHARMM36 RTF, PDB CCD, and future sources. Every table
   carries explicit `source`, `version`, `license`, and `citation` fields.

2. **`parse_master` binary** — a dev-only tool that converts an existing Mosaist
   `rotlib.bin` into a `RotamerLibrary.pb.zst`, enabling MASTER-exact cross-checking
   for developers who have Mosaist installed, without proxide redistributing
   CC-BY-NC-SA data.

These complement (not replace) the Dunbrack → protobuf spec
(`260602_dunbrack-rotlib-protobuf-cis-pro.md`). That spec's Engh-Huber geometry
placeholder (§7 / backlog #820) is replaced by a `ResidueGeometryTable` loaded from
a user-supplied IC source.

---

## 2. Motivation

### 2.1 The IC geometry gap

`convert_rotlib.rs` builds sidechain Cartesian coordinates from Dunbrack χ angles +
`geometry/template.rs`. That template uses Engh-Huber placeholder values, which
diverge from the CHARMM IC table values MASTER used when building its `rotlib.bin`.
Measured consequence: 0.043 max|Δ| contact-degree drift (17/30 interior-residue
pairs; root cause: cumulative IC error reaching 0.235 Å at MET CE — see research
doc `260603_master-rotlib-cartesian-derivation.md`).

Hardcoding any single IC source creates a single point of failure and makes the
provenance/license implicit. The fix is to make IC geometry a first-class, typed,
attributed input.

### 2.2 License landscape

| IC source | License | Coverage | Notes |
|-----------|---------|----------|-------|
| CHARMM36 RTF (`top_all36_prot.rtf`) | No OSI license; "free for academic use" — MacKerell lab | All 20 AA; complete IC build tree; explicit per-atom records | Cannot be bundled in MIT project without ambiguity; user-supplied |
| PDB CCD (`.cif` per residue) | Public domain / CC0 (wwPDB) | All 20 AA; pairwise bonds + angles | Bundleable; needs build-tree conversion |
| AMBER ff14SB (AmberTools) | Apache-2.0 | All 20 AA; atom-type keyed, not per-residue build tree | Deferred v1 |
| OpenFF ff14sb port | MIT | All 20 AA; SMIRKS-keyed | Deferred v1 |
| Mosaist `rotlib.bin` | **CC-BY-NC-SA 4.0** (Grigoryanlab/Mosaist README; file in testfiles/) | 20 AA; pre-baked Cartesian coords | Non-commercial + ShareAlike; **must not** appear in proxide repo or distributed artifacts |

The `parse_master` tool enables MASTER-exact parity for developers without bundling
CC-BY-NC-SA data.

---

## 3. Goals / Non-goals

**Goals**
- G1. `residue_geometry.proto` with source/version/license/citation metadata per table.
- G2. CHARMM36 RTF IC importer (user-supplied file; not bundled in repo).
- G3. PDB CCD importer (pairwise bonds/angles → IC build tree; 20 standard AA `.cif` files can be vendored as public-domain data).
- G4. Wire `ResidueGeometryTable` into `convert_rotlib.rs` as the IC geometry input.
- G5. `geometry_source` + `geometry_license` fields added to `rotlib.proto`.
- G6. `parse_master` binary: Mosaist `rotlib.bin` → `RotamerLibrary.pb.zst` (dev-only).
- G7. Docs: per-source license table, NOTICE-style attribution text, CI check that no CC-BY-NC-SA artifact is committed.

**Non-goals**
- N1. AMBER ff14SB or OpenFF importers in v1.
- N2. Bundling any CHARMM36 RTF file in the repo.
- N3. Committing or distributing any `parse_master` output (CC-BY-NC-SA by inheritance).
- N4. Runtime IC geometry lookup (`PRECOMPUTED` mode only; geometry builds offline).

---

## 4. `residue_geometry.proto`

New file: `crates/proxide-rotlib/proto/residue_geometry.proto`

```proto
syntax = "proto3";
package proxide.rotlib.v1;

// Table of sidechain IC geometry for one force-field / geometry source.
// Every table must carry non-empty source, license, and citation fields.
message ResidueGeometryTable {
  string source   = 1;  // "charmm36" | "pdb_ccd" | "amber_ff14sb" | "openff_ff14sb"
  string version  = 2;  // e.g. "charmm36_jul2024", "pdb_ccd_2024"
  string license  = 3;  // SPDX id or prose: "NOT-OSI: MacKerell lab academic only" | "CC0"
  string citation = 4;  // DOI or BibTeX key for the source publication
  repeated ResidueGeometry residues = 5;
}

// IC build tree for one amino acid residue.
message ResidueGeometry {
  string name = 1;          // "MET", "ARG", "PRO", etc.
  repeated IcRecord ic = 2; // ordered: each record places atom_k given i, j, l placed
}

// One NERF placement record — mirrors CHARMM RTF IC format:
//   i j *k l | b(i-j)  θ(i-j-k)  φ(i-j-k-l)  θ(j-k-l)  b(k-l)
message IcRecord {
  string atom_i    = 1;
  string atom_j    = 2;
  string atom_k    = 3;  // atom being placed
  string atom_l    = 4;
  bool   branch    = 5;  // true = asterisk in RTF (atom_k is a branch point)
  float  b_ij      = 6;  // bond length i–j, Å
  float  theta_ijk = 7;  // valence angle i–j–k, degrees
  float  phi_ijkl  = 8;  // dihedral i–j–k–l, degrees
  float  theta_jkl = 9;  // valence angle j–k–l, degrees
  float  b_kl      = 10; // bond length k–l, Å
}
```

### 4.1 Additions to `rotlib.proto`

```proto
message RotamerLibrary {
  // existing fields 1–6 unchanged
  string geometry_source  = 7; // ResidueGeometryTable.source used to build coords
  string geometry_license = 8; // ResidueGeometryTable.license of that source
}
```

---

## 5. CHARMM36 RTF importer (#987)

**Module:** `crates/proxide-rotlib/src/geometry/rtf_parser.rs`

**Input:** user-supplied `top_all36_prot.rtf` (path via CLI flag; not committed to repo).

**IC record parsing:** each line matching `IC <i> <j> [*]<k> <l>  <b_ij> <theta_ijk> <phi> <theta_jkl> <b_kl>` within a `RESI` block is parsed. The `*` prefix on `atom_k` sets `branch=true`.

**Output:** `ResidueGeometryTable` with:
- `source = "charmm36"`
- `license = "NOT-OSI: MacKerell lab academic use only; see https://mackerell.umaryland.edu/charmm_ff.shtml"`
- `citation = "doi:10.1021/jp973084f"` (MacKerell 1998) + `"doi:10.1021/jp0621210"` (Best 2012)

**Coverage confirmed (Phase D audit):** all 20 standard AA have explicit IC sections.
Reference values for MET (critical path): `b(SD-CE) = 1.8206 Å`, `θ(CG-SD-CE) = 98.94°`.

---

## 6. PDB CCD importer (#987)

**Module:** `crates/proxide-rotlib/src/geometry/ccd_parser.rs`

**Input:** `<AA>.cif` files from RCSB (public domain; 20 standard AA `.cif` files vendored under `crates/proxide-rotlib/data/ccd/`; fetched once at dev time, committed as CC0 data).

**Conversion:** `_chem_comp_bond.value_dist_ideal` + `_chem_comp_angle.value_angle_ideal` → IC build tree via DFS from N/CA backbone anchors. Dihedral references (`phi_ijkl`) taken from standard χ definitions, not from CCD (CCD does not contain dihedral IC records).

**Output:** `ResidueGeometryTable` with `source = "pdb_ccd"`, `license = "CC0"`, `citation = "doi:10.1093/nar/gku1178"`.

**Limitation:** CCD ideals are symmetric averages (e.g. PRO χ2≈0°); use for bond lengths and valence angles only. Dihedral starters come from rotamer definitions.

---

## 7. `parse_master` binary (#988)

**File:** `crates/proxide-rotlib/src/bin/parse_master.rs`

```
parse_master --input <path/to/rotlib.bin> --output <path/to/out.pb.zst>
```

- Reads `rotlib.bin` via `RotamerLibrary::load()` (existing reader).
- Serializes all entries to `RotamerLibrary` proto + zstd with `geometry_mode = PRECOMPUTED`.
- Embeds `data_license = "CC-BY-NC-SA-4.0"`, `provenance = "Mosaist testfiles/rotlib.bin — Grigoryan lab, Dartmouth; non-commercial only; not for redistribution"`.
- Prints a license reminder to stderr on every run.

**gitignore:** output `.pb.zst` files must never be committed. Add pattern to `.gitignore`:
```
# MASTER-derived data — CC-BY-NC-SA; not for redistribution
*.master.pb.zst
```

**Use case:** developers with a local Mosaist install generate a local MASTER-exact library once. The resulting file can be passed to the existing `confind` integration for cross-checking. It must not be shared or published.

---

## 8. `convert_rotlib.rs` integration (#987)

CLI extension:
```
convert_rotlib --dunbrack ALL.bbdep.rotamers.lib \
               --ic-geometry charmm36.residue_geometry.pb.zst \
               --output proxide-rotlib-bbdep2010.pb.zst
```

The `--ic-geometry` argument replaces `geometry/template.rs` as the IC source.
`template.rs` is kept as a fallback for residues not present in the supplied table
(with a `tracing::warn!`), and for the Engh-Huber baseline for regression comparison.

---

## 9. Acceptance criteria

- **AC-1.** `ResidueGeometryTable` round-trips through prost + zstd losslessly.
- **AC-2.** CHARMM36 RTF importer: all 20 standard AA present; MET verified: `b(SD-CE) = 1.8206 ± 0.001 Å`, `θ(CG-SD-CE) = 98.94 ± 0.1°`.
- **AC-3.** `convert_rotlib` with CHARMM36 IC source: contact-degree drift vs MASTER closes to `<5e-4` max|Δ| on `small.pdb` (closes #869).
- **AC-4.** `parse_master` output: `RotamerLibrary::load()` in-memory struct matches `load_pb()` in-memory struct for all 20 AA entries (validated by `test_parity_small_pdb` suite passing with the generated `.pb.zst`).
- **AC-5.** CI check: `git ls-files` contains no `top_all36_prot.rtf`, `rotlib.bin`, or `*.master.pb.zst`.
- **AC-6.** Docs: `docs/data-sources.md` (or equivalent) carries the per-source license table from §2.2 and NOTICE-style attribution for CHARMM36 and Dunbrack.

---

## 10. Risks & mitigations

| Risk | Sev | Mitigation |
|------|-----|-----------|
| CHARMM36 RTF not available in CI (user-supplied) | Med | CCD importer is bundleable and provides a CI-safe alternative for AC-3 regression |
| CCD build-tree conversion produces wrong dihedral order for branched residues | Med | Validate with round-trip χ-recovery test per residue; fail loudly on mismatch |
| parse_master output accidentally committed | High | `.gitignore` pattern + CI `git ls-files` check |
| AMBER/OpenFF deferred — leaves gap if CHARMM36 RTF unavailable | Low | CCD covers bond lengths/angles; acceptable for v1; deferred importers tracked |
