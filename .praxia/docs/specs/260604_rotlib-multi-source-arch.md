# Spec: Multi-source rotamer library architecture

- **task_id:** `260604_rotlib_multi_source`
- **Status:** DRAFT
- **Author:** orchestrator (2026-06-04)
- **Related:** #869, #987, #988; `260602_dunbrack-rotlib-protobuf-cis-pro.md`,
  `260604_rotlib-ic-geometry-schema.md`

---

## 1. Problem statement

proxide-rotlib currently has one production path: load the Mosaist `rotlib.bin`
(CC-BY-NC-SA 4.0, non-commercial). Sprint 12 built a Dunbrack→protobuf converter
but the geometry drift vs MASTER remained open. Sprint 13 resolved the geometry
question and found:

- **parse_master**: 0 drift, perfect MASTER parity — but CC-BY-NC-SA data, no
  redistribution. Suitable for users who have a Mosaist install.
- **Dunbrack + any IC source**: 0 contact flips at CD > 0.001 (measured on small.pdb).
  Magnitude drift is ~0.04 max|Δ| but does not change which pairs are classified as
  contacts at any threshold a user would actually set.

The remaining work is to make the Dunbrack path first-class and to build the
abstraction layer that lets us plug in other rotlib sources and IC parameterizations
without structural changes to confind or the hot path.

---

## 2. Architecture overview

Three independent axes of variation, each with a clear abstraction boundary:

```
┌───────────────────────┐    ┌───────────────────────────┐
│  Rotlib source        │    │  IC geometry source        │
│  (chi + probs)        │    │  (bond lengths + angles)   │
│                       │    │                            │
│  DunbrackSource  ──── │    │  CcdSource (CC0, bundled)  │
│  MasterSource (CC-NC) │ ×  │  RtfSource (user-supplied) │
│  [future: Richardson] │    │  EnghHuberSource (builtin) │
│                       │    │  [future: AmberSource]     │
└──────────┬────────────┘    └──────────────┬─────────────┘
           │                                │
           └──────────────┬─────────────────┘
                          ▼
           ┌─────────────────────────────────┐
           │       convert_rotlib            │
           │  --rotlib-source <spec>         │
           │  --ic-source <spec>             │
           │  --output <named.pb.zst>        │
           └──────────────┬──────────────────┘
                          ▼
           ┌─────────────────────────────────┐
           │   RotamerLibrary.pb.zst         │
           │   (PRECOMPUTED mode; Cartesian   │
           │    coords baked in)             │
           │   provenance / attribution /    │
           │   data_license / geometry_*     │
           └──────────────┬──────────────────┘
                          ▼
           ┌─────────────────────────────────┐
           │   proxide-confind (unchanged)   │
           │   load_pb() → ConFind           │
           └─────────────────────────────────┘
```

### 2.1 What does NOT change

- `RotamerLibrary::load_pb()` — hot path is untouched
- `place_rotamer()` / `backbone_bin()` / `num_rotamers()` — all unchanged
- `confind.rs` — switches to `load_pb()` once a validated pb.zst is available
- The proto schema (`rotlib.proto`) — already has `geometry_source`/`geometry_license`

### 2.2 What changes

- `convert_rotlib.rs` gains a `RotlibSource` trait and a `--rotlib-source` flag
- IC geometry application (`apply_ic_table`) is already wired; it just needs the CCD
  source to be the default when no `--ic-source` is specified
- Naming convention for output files (§3)
- A small calibration test suite (§5)

---

## 3. Rotlib source trait

Lives in a new `crates/proxide-rotlib/src/rotlib_source/` module. Keeps
`convert_rotlib.rs` decoupled from any specific text format.

```rust
/// One rotamer entry: chi angles, per-rotamer probability, Dunbrack observation count.
pub struct RotamerEntry {
    pub chi_values: Vec<f32>,  // degrees, chi1..n
    pub chi_sigmas: Vec<f32>,
    pub probability: f64,
    pub count: u32,
}

/// One (phi, psi) bin's contents for a single residue.
pub struct BinData {
    pub phi: f64,
    pub psi: f64,
    pub freq: f64,       // sum-of-probs or observation fraction
    pub rotamers: Vec<RotamerEntry>,
}

/// Source of chi-angle rotamer data. Implemented by DunbrackSource; extendable.
pub trait RotlibSource: Send + Sync {
    fn residue_codes(&self) -> Vec<String>;
    fn bins(&self, code: &str) -> Vec<BinData>;
    fn default_bin_index(&self, code: &str) -> usize;  // argmax freq bin
    fn data_license(&self) -> &str;
    fn attribution(&self) -> &str;
    fn source_tag(&self) -> &str;  // e.g. "dunbrack2010_simpleopt1"
}
```

### 3.1 DunbrackSource

Wraps the existing Dunbrack text parser from `convert_rotlib.rs`. Reads
`*.bbdep.rotamers.lib` and implements `RotlibSource`. Path passed at construction.

**License**: ODC-BY-1.0 (Open Data Commons Attribution). Attribution notice is
embedded as a required non-empty field in every output proto.

### 3.2 MasterSource (via parse_master)

`parse_master` already converts MASTER binary → proto directly (PRECOMPUTED mode).
It does NOT go through the `RotlibSource` trait because the coords are pre-baked —
there is no chi+IC→Cartesian step. It is a separate binary, not a `convert_rotlib`
source type.

**License**: CC-BY-NC-SA 4.0. Output must not be committed or redistributed.
Suitable for user personal use (user already has a Mosaist license that covers this).

---

## 4. IC geometry sources

Already built in Sprint 13. Summary of status and recommendation:

| Source | License | Bundleable | Quality | Recommendation |
|--------|---------|-----------|---------|---------------|
| PDB CCD (ccd_parser.rs) | CC0 | ✅ Yes | Good (crystal structures) | **Default for Dunbrack** |
| CHARMM36 RTF (rtf_parser.rs) | Not OSI | ❌ User-supplied | Best for MASTER proximity | `--ic-source rtf:<path>` |
| Engh-Huber (template.rs) | Published | ✅ Hardcoded | Acceptable | Fallback / baseline |
| AMBER ff14SB | Apache-2.0 | ✅ Yes | Comparable to CCD | Deferred |
| OpenFF ff14sb | MIT | ✅ Yes | Comparable to CCD | Deferred |

**Default for the shipped Dunbrack pb.zst**: PDB CCD.

- CC0 — no license complications in any distribution model
- 20 standard AA `.cif` files already vendored (or can be committed)
- Contact classification quality: 0 flips vs MASTER at CD > 0.001 (measured, §5.1)
- Bond lengths/angles derived from high-resolution PDB structures; adequate for
  all realistic contact-degree use cases

**When to use CHARMM36 RTF**: users who run workflows that compare contact degrees
numerically against MASTER output and need the magnitudes to be closer (reduces
max|Δ| from 0.043 to 0.038). Does not change contact classifications. User-supplied.

### 4.1 IC source interaction with proline

Proline is skipped by `apply_ic_table()` in all cases. The `ProlineBuilder`
manages the pyrrolidine ring geometry via CCD ring closure, keeping it consistent
regardless of IC source. Proline coords remain calibrated to PDB-observed ring
geometry.

---

## 5. Output naming and provenance

Files are named `proxide-rotlib-<rotlib>-<icsource>.pb.zst`.

| File | Rotlib | IC source | License | Redistributable |
|------|--------|-----------|---------|-----------------|
| `proxide-rotlib-dunbrack2010-ccd.pb.zst` | Dunbrack 2010 (ODC-BY) | PDB CCD (CC0) | ODC-BY + CC0 | ✅ Yes |
| `proxide-rotlib-dunbrack2010-rtf.pb.zst` | Dunbrack 2010 (ODC-BY) | CHARMM36 RTF | ODC-BY + MacKerell | ❌ User-built |
| `*.master.pb.zst` | MASTER binary | N/A (PRECOMPUTED) | CC-BY-NC-SA | ❌ Dev-only |

The default shipped artifact is `proxide-rotlib-dunbrack2010-ccd.pb.zst`.

Every pb.zst proto encodes:
```
RotamerLibrary {
  provenance:       "Dunbrack BBDEP2010 SimpleOpt1-5; convert_rotlib <version>"
  attribution:      "Contains information from Dunbrack 2010 BBDep ... ODC-BY"
  data_license:     "ODC-BY-1.0"
  geometry_source:  "pdb_ccd"
  geometry_license: "CC0"
  ...
}
```

---

## 6. convert_rotlib CLI after this work

```
convert_rotlib \
  --rotlib-source dunbrack:<path/to/ALL.bbdep.rotamers.lib> \
  --ic-source ccd:<path/to/ccd/dir>          # default: bundled CCD
  --ic-source rtf:<path/to/top_all36_prot.rtf>  # override: CHARMM36 RTF
  --ic-source engh-huber                     # override: hardcoded template.rs
  --output data/rotlibs/proxide-rotlib-dunbrack2010-ccd.pb.zst
```

The `--rotlib-source` flag currently defaults to Dunbrack. Later, adding
`--rotlib-source richardson:<path>` or `--rotlib-source bbdep2023:<path>` is a
matter of implementing `RotlibSource` for that format and wiring the CLI.

`parse_master` remains a separate binary (no trait needed; it copies pre-baked coords).

---

## 7. Calibration

### 7.1 What we know (measured Sprint 13)

On `small.pdb` (7-residue test case, 56 MASTER reference pairs):

| pb.zst | max\|Δ\| vs MASTER | Flips at CD > 0.001 | Flips at CD > 0.01 |
|--------|--------------------|---------------------|--------------------|
| parse_master (MASTER exact) | 0.000 | 0 | 0 |
| Dunbrack + Engh-Huber | 0.043 | 0 | 1 |
| Dunbrack + CHARMM36 RTF | 0.038 | 0 | 1 |

Direction: 24/30 drifting pairs over-report vs MASTER (proxide's Dunbrack-sourced
contacts tend to be slightly higher than MASTER's, not lower).

### 7.2 What we still need

- **Full-protein calibration**: run on 1DC7 (124 residues, all 20 AAs) and compare
  contact lists. small.pdb has 7 residues; 1DC7 gives a realistic distribution.
- **Precision/recall framing**: at CD > threshold, what fraction of MASTER's contacts
  does Dunbrack+CCD recover? (precision = TP / (TP+FP); recall = TP / (TP+FN))
- **Threshold recommendation**: at what CD threshold do Dunbrack+CCD and MASTER agree
  most closely on contact classification? This becomes the "recommended threshold"
  in user docs.

### 7.3 Calibration test (to add)

A new `#[ignore]` test in `test_drift_loadpb_small_pdb.rs` or a companion
`test_contact_precision_1dc7.rs`:
- Load the shipped `dunbrack2010-ccd.pb.zst`
- Run ConFind on 1DC7
- Compare contact lists against MASTER output at CD > 0.001, 0.005, 0.01, 0.05
- Assert precision ≥ 0.90 and recall ≥ 0.90 at CD > 0.01 (threshold to be refined)

---

## 8. Phased implementation plan

### Phase 1 — Default Dunbrack+CCD pb.zst and confind migration (#869)

**Goal**: ship a working, license-clean `proxide-rotlib-dunbrack2010-ccd.pb.zst`
and migrate confind off the CC-BY-NC-SA `rotlib.bin` for production use.

Steps:
1. Vendor all 20 standard AA CCD `.cif` files (they go in
   `crates/proxide-rotlib/data/ccd/`; currently only PRO.cif exists)
2. Set PDB CCD as the default IC source when `--ic-source` is omitted from
   `convert_rotlib`
3. Regenerate `proxide-rotlib-dunbrack2010-ccd.pb.zst` and commit it
4. Run full-protein calibration on 1DC7; document precision/recall
5. Migrate `confind.rs` load site from `load()` → `load_pb()` once calibration passes
6. Update user docs with threshold guidance

**Acceptance**: `cargo test --workspace` passes; confind 1DC7 parity tests pass;
no CC-BY-NC-SA artifact in `git ls-files`.

### Phase 2 — RotlibSource trait + DunbrackSource refactor

**Goal**: decouple the Dunbrack text parser from `convert_rotlib.rs` into a proper
trait, making it straightforward to add future sources.

Steps:
1. Define `RotlibSource` trait and `RotamerEntry` / `BinData` types in new module
2. Extract existing Dunbrack parser into `DunbrackSource` implementing the trait
3. `build_library()` takes `Box<dyn RotlibSource>` instead of `GroupedRotamers`
4. Add `--rotlib-source dunbrack:<path>` flag (current behavior becomes explicit default)
5. `parse_master` is documented as the MasterSource equivalent but stays a separate binary

**Acceptance**: all existing convert_rotlib tests pass; trait is documented; adding a
new RotlibSource requires only implementing the trait.

### Phase 3 — parse_master as user tool (not just dev)

**Goal**: expose `parse_master` as a first-class user-facing tool with clear license
guidance, since it enables MASTER-exact contact degrees for users with Mosaist.

Steps:
1. Add `--validate` flag to `parse_master`: runs the drift test inline and reports
   max|Δ|, confirming the output is MASTER-equivalent
2. Add man-page style documentation: "When to use parse_master vs dunbrack+ccd"
3. Add to proxide CLI top-level help
4. Note in docs: CC-BY-NC-SA means use is fine; distribution of the output is not

### Phase 4 — Additional rotlib sources (future, not scoped)

Richardson 2010, Dunbrack 2023, SCWRL4 (if licensable), others.
Each is a new `impl RotlibSource` and new IC compatibility tests.
No structural changes to convert_rotlib or confind required.

---

## 9. Backlog items

| # | Title | Phase | Depends on |
|---|-------|-------|-----------|
| tbd | Vendor 20 AA CCD .cif files; set CCD as default IC source | 1 | #987 |
| tbd | Full-protein 1DC7 calibration: precision/recall Dunbrack+CCD vs MASTER | 1 | above |
| #869 | Migrate confind to load_pb() | 1 | calibration pass |
| tbd | RotlibSource trait + DunbrackSource refactor | 2 | #869 |
| tbd | parse_master --validate + user-facing docs | 3 | #988 |

---

## 10. Key invariants (do not violate)

- Every `RotamerLibrary.pb.zst` must have non-empty `attribution` (enforced by loader)
- `*.master.pb.zst` files must never be committed (`git ls-files` CI check)
- CHARMM36 RTF must never be committed
- `apply_ic_table()` skips proline (ring geometry managed by ProlineBuilder)
- PRECOMPUTED mode is the shipped default; CHI_ONLY mode exists in the proto for
  future runtime parameterization but is not used in production paths
