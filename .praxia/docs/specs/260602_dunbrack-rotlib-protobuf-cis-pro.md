# Spec: Dunbrack 2010 → protobuf rotamer library, with cis-PRO support

- **task_id:** `260602_proxcon_deferred4` (item 3) + follow-on
- **Status:** APPROVED (oracle, round 3 of critique→revise) — decisions A/5%/Engh-Huber locked (§13); ring-closure algorithm + CCD bounds, frame convention, f32/f64, risk-first ordering, independent reviewer audit gates, P0 preflight (vendored CCD PRO.cif), numeric AC-G/AC-R tolerances
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

- **Reuse, don't reimplement:** the IC→Cartesian primitive is
  `proxide_geometry::geometry::nerf::Nerf::place_atom` (verified **f32** signature,
  returns `[f32;3]`). Build the sidechain in f32 via NeRF, convert to the `f64` coords
  `BinData` uses; AC-G tolerances (≥1e-3 Å) are set so f32 rounding is immaterial. Do
  **not** add a parallel f64 NeRF.
- **Residue templates:** per-residue ideal internal coords (atoms, bonds, bond
  lengths/angles, the 4 atoms defining each χ) from **standard published values** —
  **Engh–Huber as a v1 PLACEHOLDER**, flagged for the research item (backlog #820,
  §7 RESEARCH FLAG) which decides whether MASTER's convention needs a different param
  set. v1 ships **PRO + CPR only**.
- **Canonical frame (convention trap):** rebuilt coords MUST live in the same
  backbone-relative frame `place_rotamer` assumes — `frame::backbone_frame(N,CA,C)`
  (x=CA−N, z=x×(C−CA), y=z×x, origin=CA; matches MSL/MASTER, frame.rs). Place
  idealized N,CA,C at canonical positions, build there, store. **Required test:**
  `place_rotamer` onto a backbone equal to the build-frame backbone returns the stored
  coords within tol (round-trip identity) — otherwise the rigid transform is wrong.
- **Proline ring closure — NAMED algorithm (was the hand-waved risk):** the
  pyrrolidine ring N–CA–CB–CG–CD–N is closed; the **two Dunbrack rotamers per (φ,ψ)
  bin ARE the Cγ-endo / Cγ-exo puckers** — their χ sets have opposite signs
  (CPR r1 χ=(32.5,−36.0,25.1) vs r2 (−20.3,34.0,−33.8)). Build CB,CG,CD by NeRF using
  the rotamer's **endocyclic torsions** (χ1=N-CA-CB-CG, χ2=CA-CB-CG-CD, χ3=CB-CG-CD-N)
  + ideal bond lengths/angles; select endo/exo by rotamer index `r1`. Because the χ
  come from real closed rings, NeRF lands CD near N. **Ring closure (specified — no
  escape hatch):** if `|CD–N − 1.47| > 0.02 Å` after the NeRF build, run
  **cyclic-coordinate-descent (CCD)**: iteratively rotate χ2 (moves CG,CD) then χ3
  (moves CD) in small steps to drive CD–N toward 1.47 Å; **max 100 iterations**;
  **converge** when `|CD–N − 1.47| ≤ 0.02 Å` **AND** each χ stays within **±5°** of its
  Dunbrack value (closure must NOT be absorbed into bond angles or large χ drift). A
  rotamer that cannot converge within those bounds **fails AC-G and is not shipped** —
  there is no "document the residual" fallback. Refs: Ho et al. (proline pucker),
  Cremer–Pople puckering. Validated by §11 AC-G.

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

Ordering is **risk-first**: the hardest phase (P3 geometry) runs early so failures
surface before later work is built on it. Independent **audit gates** (a separate
`reviewer` agent that re-runs tests + an external validation and reports its OWN
measured numbers — never the implementer's pasted claim) follow the two
correctness-critical phases, because an implementer self-attesting "tests pass" is
not trusted (a fixer false-greened earlier this sprint).

| # | Phase | Tasks | Gate |
|---|------|-------|------|
| 0 | **P0 Preflight** | Assert the Dunbrack input exists at `data/rotlibs/SimpleOpt1-5/ALL.bbdep.rotamers.lib` (it is **gitignored** — if absent, extract from `data/rotlibs/dunbrack2010-everything.tar.zst` with `zstd --long=31`); pin its SHA256. Vendor the validation reference `crates/proxide-rotlib/tests/data/ccd/PRO.cif` from RCSB CCD (`https://files.rcsb.org/ligands/view/PRO.cif`, public-domain) so A3/P6 read it from disk (CI/cluster is offline). | input present + SHA pinned; `PRO.cif` vendored. **Blocks all.** |
| 1 | **P1 Extract** | Tracked Python script (+bathos) parsing `ALL.bbdep.rotamers.lib` **filtered to T∈{CPR,PRO,TPR}** (5% stepdown) → JSON (bins/probs/χ). | AC-1. |
| 2 | **P3 Geometry** | Reuse `Nerf::place_atom` (§7); proline template (Engh–Huber placeholder); endocyclic-χ ring build + CCD closure; endo/exo via `r1`; canonical-frame identity test. | **AC-G**. |
| 3 | **A3 Geometry audit** | Independent `reviewer`: re-runs `cargo test -p proxide-rotlib`, runs external geometry validation (debug-dump coords; assert bonds/angles/closure/χ-recovery vs CCD `PRO.cif`), reports measured numbers. | AC-G confirmed by auditor — **blocks P4**. |
| 4 | **P2 Schema** | `rotlib.proto`; `prost` (build.rs) + `zstd` deps; round-trip test. | AC-2. |
| 5 | **P4 Converter** | Dunbrack text → `*.rotlib.pb.zst` (PRECOMPUTED coords from P3); embed attribution + sidecar. **Only after A3 passes.** | AC-3. |
| 6 | **P5 Loader+routing** | `load_pb`; route `num_rotamers`/`place_rotamer` to CPR via one shared helper; `RotlibError` variants. | AC-R. |
| 7 | **A5 Routing audit** | Independent `reviewer`: full `cargo test -p proxide-rotlib`, verifies AC-R **by identity** (not inequality), reports numbers. | AC-R confirmed by auditor. |
| 8 | **P6 Verify+docs** | Cross-check rebuilt PRO vs **PDB CCD `PRO.cif`** (public-domain) — **NOT** MASTER; README ODC-BY notice + citation; `--all-targets` clean. | AC-4, AC-5. |

P1 and P3 are safe to start immediately. The Research item (#820) runs in **parallel,
read-only**, and feeds a FOLLOW-UP refinement of the Engh–Huber placeholder — it does
not block this sprint.

---

## 11. Acceptance criteria

- **AC-1.** `ALL.bbdep.rotamers.lib` parsed; T∈{CPR,PRO,TPR} all present; CPR has 2 rotamers/bin; per-bin Σprob = 1.0±1e-3; grid rectangular; PRO/CPR (φ,ψ) bin count recorded and matches the file.
- **AC-2.** `rotlib.proto` round-trips losslessly through prost + zstd (`--long`).
- **AC-G (strengthened).** For proline, with the Engh–Huber template:
  - (a) all 3 Dunbrack χ are **recovered** from the rebuilt coords within ±2°;
  - (b) **both** puckers build — `r1=1` (endo) and `r1=2` (exo) produce geometrically **distinct** CG positions (**≥0.5 Å apart**; published endo↔exo CG displacement ≈0.5–0.7 Å);
  - (c) ring bond angles within ±3° of CCD `PRO.cif` ideals (**~104–105°**, measured: N-CA-CB 104.7, CA-CB-CG 105.1, CB-CG-CD 105.1, CG-CD-N 104.7, CD-N-CA 104.1 — NOT 110°); CD–N within ±0.03 Å of **1.487 Å** (CCD);
  - (d) ring **bond lengths** within ±0.02 Å of CCD ideals (N-CA 1.486, CA-CB 1.543, CB-CG 1.543, CG-CD 1.544, CD-N 1.487). NOTE: CCD `PRO.cif` is a *symmetric* idealization (χ2≈0°), so it is a bond/angle reference, **not** a whole-ring RMSD target for the χ2≈±35° Dunbrack puckers — ring fidelity is covered by (a) χ-recovery + (c) closure;
  - (e) **round-trip identity:** `place_rotamer` onto a backbone equal to the build-frame backbone returns the stored coords within **≤1e-2 Å**.
  - All numeric thresholds above are the AC-G pass conditions; A3 must report measured values, not a bare boolean.
- **AC-3.** Converter emits `*.rotlib.pb.zst` whose `attribution`/`data_license` are populated (loader rejects empty `attribution`).
- **AC-R (strengthened).** With synthetic **identical** PRO & CPR grids:
  - (1) `place_rotamer("PRO",φ,ψ,ri,cis=true)` coords **equal the CPR entry's stored coords within ≤1e-6 Å** (PRECOMPUTED coords are read directly by `load_pb` with NO rebuild on the read path, so this is bit-identity modulo the rigid transform — not a fresh NeRF build);
  - (2) `num_rotamers("PRO",φ,ψ,cis=true)` equals the CPR bin's rotamer count;
  - (3) on real data at (−180,−180), the placed cis rotamer χ1 is **closer to 32.5° than 27.3°**.
- **AC-4.** `cargo test -p proxide-rotlib` and `cargo check -p proxide-rotlib --all-targets` pass, warning-free (also `cargo build -p proxide-frag` — same `deny(warnings)`).
- **AC-5.** Repo carries ODC-BY notice + citation where the data ships; **no** CC BY-NC-SA artifact (`rotlib.bin`) is committed/redistributed (verified via `git ls-files`).

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

## 13. Decisions (closed) & open questions

**Closed (locked by user, 2026-06-02):**
1. **§9 A vs B → A (PRECOMPUTED).** Coords baked at convert time; `place_rotamer` unchanged.
2. **Canonical stepdown → 5% (Opt1-5).**
3. **Geometry templates → Engh–Huber placeholder**, flagged for research (backlog #820) to match MASTER's convention; reuse `proxide-geometry` NeRF for the IC build.

**Resolved in deferred sprint (2026-06-02, task 260602_rotlib_deferred_sprint):**
4. **Q4 → one combined `.pb.zst`.** The geometry engine now covers all 22 rotameric residue codes (`standard_residue_template` for the 19 non-proline + `proline_template` for PRO/CPR/TPR), so the interim CPR+PRO-only constraint is retired. The converter emits a single `proxide-rotlib-bbdep2010.pb.zst` (all residues) — verified by the AC-3 full-library test (740,629 entries → 22 residue types, 11.9 MB compressed; input SHA256 `aade9d4f…34bdf`).
5. **Full-library regeneration → DONE (generation validated); confind runtime migration → GATED on backlog #820.** The full 22-residue library generates and passes AC-3. However, swapping confind's production call-site (`confind.rs:22`) from `load()` (MASTER `rotlib.bin`) to `load_pb()` is **not drop-in safe**: measured contact-degree drift on `small.pdb` is large (max |Δ| 0.223, mean 0.021, 69% of 43 matched contacts exceed the 5e-4 parity tolerance, plus 13 MASTER pairs structurally absent). Reproduced by the `#[ignore]` harness `crates/proxide-confind/tests/test_drift_loadpb_small_pdb.rs` (anchored: the shared ConFind harness reproduces MASTER `REF_CONTACTS` to <5e-4 via `load()`). The migration therefore waits on #820 (Engh–Huber geometry tuned to MASTER's convention); confind stays on `load()` until then.

**Still open:**
6. Root-cause the drift: how much is pure geometry (Engh–Huber vs MASTER ideals) vs. rotamer-set / atom-order / residue-variant semantics? Needed to scope #820's acceptance criterion (target: confind drift within 5e-4 of MASTER).
