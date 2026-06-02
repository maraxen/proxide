# Spec: Dunbrack 2010 → protobuf rotamer library, with cis-PRO support

- **task_id:** `260602_proxcon_deferred4` (item 3) + follow-on
- **Status:** DRAFT — one open decision (geometry storage A vs B, §9)
- **Author:** orchestrator (proxcon session, 2026-06-02)
- **Backlog:** #799
- **Related:** `crates/proxide-rotlib`, memory `project-rotlib-path`

---

## 1. Summary

Replace proxide-rotlib's dependency on the MASTER/MSL binary `rotlib.bin` with a
**proxide-owned protobuf + zstd** rotamer-library format, generated directly from
the **Dunbrack 2010 backbone-dependent rotamer library** (ODC-BY licensed text).
This (a) adds the missing **cis-proline (`CPR`)** entry — closing item 3 — and
(b) removes proxide's runtime dependency on a CC BY-NC-SA file, giving proxide a
clean, **MIT-code + ODC-BY-data** library it can actually ship.

---

## 2. Motivation & background

### 2.1 The format blocker
`RotamerLibrary::load` (rotlib.rs) reads a MASTER binary that stores, per rotamer:
probability, χ values+sigmas, **and pre-baked Cartesian sidechain coordinates +
atom names** in a canonical backbone frame. `place_rotamer` simply rigid-transforms
those frozen coords onto a target N–CA–C frame; it has **no χ→Cartesian builder**.

The Dunbrack text (`*.bbdep.rotamers.lib`) contains only **χ angles, sigmas,
probabilities, and (φ,ψ) bins** — no coordinates, no atom names. mosaist's
`mstrotlib` has only a reader (no writer); the original `rotlib.bin` shipped
pre-built with MASTER. So text→binary requires regenerating geometry, which no
code in the proxide/mosaist tree currently does.

### 2.2 Why not just reuse MASTER's coords
The shipped `rotlib.bin` is **CC BY-NC-SA 4.0** (NonCommercial + ShareAlike +
non-sublicensable). Copying its baked coords into proxide's distributed data would
force proxide's data to be NC-SA — incompatible with proxide's **MIT** license.
Therefore all coordinates must be derived from the **Dunbrack ODC-BY text** plus
**standard, non-copyrightable idealized geometry** (bond lengths/angles), never
from MASTER. (Also: cis-PRO χ means differ from trans-PRO — e.g. χ1 32.5° vs
27.3° at (−180,−180) — so coords must be built from CPR's own χ regardless.)

---

## 3. Goals / Non-goals

**Goals**
- G1. A `.proto` schema for backbone-dependent rotamer libraries.
- G2. An offline converter: Dunbrack text → `*.rotlib.pb.zst`.
- G3. A χ→Cartesian geometry engine sufficient for **proline first**, designed to
  generalize to all 20 residues.
- G4. A protobuf+zstd loader in proxide-rotlib producing today's `AaEntry` map.
- G5. cis-PRO (`CPR`) fully wired: `backbone_bin`, `num_rotamers`, `place_rotamer`.
- G6. ODC-BY attribution shipped with any generated data artifact.

**Non-goals**
- N1. Regenerating all 20 residues in v1 (proline-first; others tracked separately).
- N2. Reading the MASTER binary for production (kept only for dev/migration/cross-check).
- N3. χ(n) density (Extended-mode) modeling — rotamers-only, as today.
- N4. Redistributing MASTER's `rotlib.bin` or any CC BY-NC-SA artifact.

---

## 4. Licensing & compliance (normative)

- **Source data:** Dunbrack BBDEP2010, **ODC-BY** (Open Data Commons Attribution).
  §3.1 explicitly permits Derivative Databases, reproduction "in any form,"
  distribution, and commercial use. A protobuf re-encoding is a Derivative Database.
- **Obligations on every generated data artifact (`*.rotlib.pb.zst`):**
  - L1. The artifact (data) remains under **ODC-BY** (distinct from proxide's MIT code).
  - L2. Ship the ODC-BY license text + URI alongside the data and in docs.
  - L3. Embed the attribution notice (also as a protobuf field, §6):
    *"Contains information from the 2010 Backbone-Dependent Rotamer Library
    (http://dunbrack.fccc.edu/bbdep2010), made available under the ODC Attribution
    License (http://dunbrack.fccc.edu/bbdep2010/license/bbdep2010_license.txt)."*
  - L4. Cite Shapovalov & Dunbrack, *Structure* 19:844–858 (2011) in papers/README.
- **Tooling licenses:** protobuf (`prost`, Apache-2.0), `zstd` crate (MIT) — both
  MIT-compatible. ✅
- **MASTER/mosaist (`rotlib.bin`):** CC BY-NC-SA — **must not** be copied into any
  distributed proxide artifact. Dev-only use behind `ROTLIB_PATH` for cross-checks
  is acceptable (no redistribution).

---

## 5. Source data format (Dunbrack `*.bbdep.rotamers.lib`)

Whitespace columns (per the `#`-comment header), one row per (residue, φ, ψ, rotamer):

```
T  Phi  Psi  Count  r1 r2 r3 r4  Probability  chi1Val chi2Val chi3Val chi4Val  chi1Sig chi2Sig chi3Sig chi4Sig
```

- `T` — 3-letter residue code. Relevant codes: standard 18 + `PRO` (trans),
  `TPR` (trans), `CPR` (**cis-proline**), `CYS`/`CYD` (disulfide)/`CYH` (reduced).
- `Phi`,`Psi` — bin centers (deg); sentinel handling matches existing loader
  (`9999` → default bin). Grid is rectangular (validated as today).
- `r1..r4` — rotamer-well indices per χ (unused χ → 0).
- `chiNVal/chiNSig` — mean and sigma (deg); unused χ → 0.

Concrete shapes (verified against `SimpleOpt1-5/ALL.bbdep.rotamers.lib`):
- **CPR / PRO:** 3 active χ (χ4=0), **2 rotamers/bin** (ring puckers; r1∈{1,2}).
- **ARG:** 4 active χ, multiple rotamers/bin.

Atom names + connectivity are **not** in this file — they come from §7 templates.

---

## 6. Target format — protobuf schema

`crates/proxide-rotlib/proto/rotlib.proto` (proto3). Serialized message, then
`zstd` compressed (`--long` window; document the decompression flag). Suggested:

```proto
syntax = "proto3";
package proxide.rotlib.v1;

message RotamerLibrary {
  uint32 version = 1;                     // schema version (start at 1)
  string provenance = 2;                  // free text: source, stepdown, git SHA
  string attribution = 3;                 // ODC-BY notice (L3) — REQUIRED non-empty
  string data_license = 4;                // "ODC-BY-1.0"
  GeometryMode geometry_mode = 5;         // PRECOMPUTED or CHI_ONLY (see §9)
  repeated ResidueEntry residues = 6;
}

enum GeometryMode { GEOMETRY_MODE_UNSPECIFIED = 0; PRECOMPUTED = 1; CHI_ONLY = 2; }

message ResidueEntry {
  string code = 1;                        // "ARG", "CPR", ...
  repeated string atom_names = 2;         // sidechain heavy atoms, library order
  uint32 num_chi = 3;
  repeated double phi_centers = 4;        // sorted ascending
  repeated double psi_centers = 5;        // sorted ascending
  uint32 default_bin = 6;                 // argmax freq (first-max wins)
  repeated Bin bins = 7;                  // length = |phi|*|psi|, phi-major
}

message Bin {
  double phi = 1;
  double psi = 2;
  double freq = 3;
  repeated Rotamer rotamers = 4;
}

message Rotamer {
  float prob = 1;
  repeated ChiValue chi = 2;              // length = num_chi
  // Present iff geometry_mode == PRECOMPUTED:
  repeated Vec3 coords = 3;               // length = |atom_names|
}

message ChiValue { float val = 1; float sigma = 2; }
message Vec3 { float x = 1; float y = 2; float z = 3; }
```

Notes: `attribution` is a required, validated field (loader errors if empty) so the
ODC-BY notice travels with the data. Floats match the source precision (f32).

---

## 7. Geometry engine (χ → Cartesian)

Required for `GEOMETRY_MODE=PRECOMPUTED` (at convert time) and/or `CHI_ONLY` (at
load/place time). Lives in `crates/proxide-rotlib/src/geometry/`.

- **Residue templates:** per-residue ideal internal coordinates — atom names,
  connectivity, bond lengths, bond angles, and which 4 atoms define each χ. Sourced
  from **standard published values** (e.g. Engh–Huber / CCD ideal geometry) — facts,
  not MASTER-derived. v1 ships **PRO only**.
- **Builder:** place backbone N,CA,C,O in the canonical frame, then build sidechain
  atoms by NeRF/IC placement walking the χ tree.
- **Proline ring closure (RISK):** proline's pyrrolidine ring (N–CA–CB–CG–CD–N) is
  **closed**, so χ are not independent — naive open-chain χ rotation will not close
  the ring. The builder must enforce ring closure (or use a small proline-specific
  pucker model parameterized by the Dunbrack χ). This is the single hardest part of
  v1 and must be validated geometrically (§11 AC-G).

---

## 8. proxide-rotlib integration

- **Loader:** `RotamerLibrary::load_pb(path)` → existing `AaEntry`/`BinData` map.
  For `CHI_ONLY`, build coords on read via §7; for `PRECOMPUTED`, read coords
  directly. Keep `load` (MASTER reader) under a `dev`/`legacy` path for cross-checks.
- **cis-PRO routing (fix existing gap):** today only `backbone_bin` is `CPR`-aware
  (`const CIS_PRO_KEY = "CPR"`). `num_rotamers` and `place_rotamer` still do
  `entries.get(aa)` (= "PRO"). They MUST resolve the effective key (CPR when
  `cis_proline && aa=="PRO"` and a CPR entry exists) so cis-PRO probabilities/coords
  are actually used. Factor the key resolution into one helper used by all three.
- **Error type:** extend `RotlibError` with `Protobuf(...)`, `MissingAttribution`,
  `UnsupportedGeometryMode`.

---

## 9. OPEN DECISION — geometry storage (A vs B)

| | **A. PRECOMPUTED (recommended)** | **B. CHI_ONLY** |
|---|---|---|
| Protobuf stores | coords (+χ) | χ only |
| Geometry runs | offline, in converter | at load/place time |
| `place_rotamer` change | none (coords ready) | needs runtime builder |
| File size | larger | smallest, purest data |
| Hot-path code | unchanged | more |
| License cleanliness | clean (std geometry) | cleanest (only Dunbrack #s) |

**Recommendation: A** — least invasive to the existing hot path, geometry runs once
offline, fastest route to working cis-PRO. The geometry engine (§7) is needed either
way; A keeps it out of the shipped library. B remains a future option (enables runtime
χ perturbation) and is cheap to add later since the schema carries χ in both modes.

---

## 10. Phased implementation plan

| Phase | Tasks | Gate / verification |
|------|-------|---------------------|
| **P1. Extract** | Tracked Python script (+bathos sidecar) parsing `cpr.bbdep.rotamers.lib` (and PRO/TPR for compare) → JSON (bins, probs, χ). License-clean, no geometry. | golden counts: 2 rotamers/bin for CPR; probabilities sum≈1 per bin; bin grid rectangular. |
| **P2. Schema** | Add `prost`/`zstd` deps; `rotlib.proto`; generated types; round-trip unit test. | `cargo test` round-trips a synthetic `RotamerLibrary`. |
| **P3. Geometry (PRO)** | Proline residue template + ring-closing builder. | **AC-G**: rebuilt PRO coords match a reference (idealized PDB proline) within tol; ring closes (CD–N bond length in range). |
| **P4. Converter** | CLI: Dunbrack text → `*.rotlib.pb.zst`; embeds attribution; emits ODC-BY sidecar. Convert CPR (+PRO). | output loads via P5; attribution field non-empty. |
| **P5. Loader + routing** | `load_pb`; route `num_rotamers`/`place_rotamer` to CPR; `RotlibError` variants. | **AC-R**: cis vs trans PRO give different rotamer probs/coords; all rotlib tests green. |
| **P6. Verify + docs** | Cross-check vs MASTER `rotlib.bin` PRO entry (dev-only); README attribution; INDEX/memory. | parity report; `cargo test -p proxide-rotlib` + `--all-targets` clean. |

P1 is safe to start immediately under either §9 option.

---

## 11. Acceptance criteria

- **AC-1.** `cpr.bbdep.rotamers.lib` parsed; per-bin probabilities sum to 1.0±1e-3; CPR has 2 rotamers/bin.
- **AC-2.** `rotlib.proto` round-trips losslessly through prost + zstd (`--long`).
- **AC-G.** Geometry: rebuilt proline sidechain is chemically valid — all bond lengths/angles within ±tol of template; ring closure CD–N satisfied.
- **AC-3.** Converter emits `*.rotlib.pb.zst` whose `attribution`/`data_license` fields are populated (loader rejects empty `attribution`).
- **AC-R.** `place_rotamer("PRO", φ, ψ, ri, cis=true, …)` uses the CPR entry (different prob/coords from `cis=false`); `num_rotamers` likewise.
- **AC-4.** `cargo test -p proxide-rotlib` and `cargo check --all-targets` pass, warning-free.
- **AC-5.** Repo carries ODC-BY notice + citation where the data ships; no CC BY-NC-SA artifact is committed/redistributed.

---

## 12. Risks & mitigations

| Risk | Sev | Mitigation |
|------|-----|-----------|
| Proline ring closure geometry (χ not independent) | High | Proline-specific pucker builder; validate AC-G against reference before trusting. |
| Rebuilt coords drift vs MASTER conventions | Med | Cross-check PRO vs `rotlib.bin` (dev-only) within tolerance; document tol. |
| Atom-name/order mismatch breaks `place_rotamer` consumers | Med | Pin atom order in template; assert against existing PRO atom_names. |
| License contamination from MASTER coords | High | Hard rule §4/N4: build only from Dunbrack + std geometry; CI check no `rotlib.bin` committed. |
| zstd `--long` decompression footgun | Low | Document `--long=31`; store window in provenance; loader sets window. |

---

## 13. Open questions

1. **§9 A vs B** — confirm PRECOMPUTED (A) for v1? (recommended)
2. Which stepdown is canonical for shipping? (default **5%** = Opt1-5, per Dunbrack benchmark guidance.)
3. Ship per-residue `.pb.zst` or one library file? (lean: one file, all residues, once the geometry engine covers them; CPR+PRO-only interim file to unblock now.)
4. Source of ideal residue geometry templates — adopt Engh–Huber values, or reuse any existing proxide-geometry/proxide-gaff params?
