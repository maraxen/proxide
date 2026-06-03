---
name: 260603_charmm-ic-sourcing
description: Implementation spec — source rotamer template internal coordinates from bundled CHARMM36 FFXML (all residues) to match MASTER rotlib.bin geometry and reduce confind load_pb drift
metadata:
  type: spec
  task_id: 260602_rotlib-confind-actions
  backlog: "#820 -> #869"
  status: ready-for-impl
---

# Spec: CHARMM-sourced internal coordinates for proxide-rotlib geometry

**Goal.** Replace the Engh-Huber / CCD placeholder bond lengths and bond angles in
`proxide-rotlib`'s residue templates with **CHARMM ideal internal coordinates**,
sourced at convert time from proxide's already-bundled
`src/proxide/assets/charmm/charmm36_protein.xml` via the existing
`proxide_core::forcefield` parser. This matches MASTER/MSL's coordinate-derivation
method (established in research `260603_master-rotlib-cartesian-derivation.md`,
backlog #820) and is the path to clearing the #869 confind `load_pb()` drift gate.

**Why this approach (user decision 2026-06-03):** parser-sourced (not hand-edited
per-residue) fixes all 20 residues at once with no transcription risk; the FFXML +
parser already ship in proxide. Drift gate stays **5e-4**; we **measure and report**
the achieved residual honestly (FFXML harmonic equilibria may leave ~1.5° vs MASTER's
IC-table coords — if so, surface it; do NOT loosen the gate unilaterally).

---

## Ground-truth CHARMM proline ideals (verified — measured from FFXML + rotlib.bin)
N-CA 1.434, CA-CB 1.527, CB-CG 1.537, CG-CD 1.537, CD-N 1.455 Å;
N-CA-CB 110.80, CA-CB-CG 108.50, CB-CG-CD 108.50, CG-CD-N 110.50, CD-N-CA 114.20°.
(MASTER baked PRO measures CB-CG=CG-CD≈1.521 Å, CB-CG-CD≈109.96° — CHARMM-family,
NOT the CCD ~105° currently in the template.)

---

## Tasks

### T1 — Dependency
`crates/proxide-rotlib/Cargo.toml`: add `proxide-core = { path = "../proxide-core" }`
to `[dependencies]`. Verified no cycle (proxide-core does not depend on proxide-rotlib).

### T2 — New module `crates/proxide-rotlib/src/geometry/charmm_ic.rs`
Public API:
```rust
pub struct CharmmIdeals { /* class maps */ }
pub fn load_charmm_ideals(ffxml_path: &str) -> Result<CharmmIdeals, RotlibError>;
pub fn apply_charmm_ideals(t: &mut ResidueTemplate, ideals: &CharmmIdeals, charmm_resname: &str);
```
- `load_charmm_ideals`: call `proxide_core::forcefield::parse_forcefield_xml(path)`.
  Build three maps from the returned `ForceField`:
  1. `(resname, atomname) -> class`  — from the FFXML `<Residue><Atom name=.. type=..>`
     plus the `<Type name=.. class=..>` table (use `proxide_core::forcefield` structs;
     read `xml_parser`/`types` to find where Residue atom lists + Type classes live —
     if the parser does not currently expose Residue atom→type, extend it minimally).
  2. `frozenset{class,class} -> length_A`  — from `harmonic_bonds` (`length` nm × 10).
  3. `(class,class,class) -> angle_deg` (+ reversed key) — from `harmonic_angles`
     (`angle` rad → deg).
- Residue-name mapping (template code → CHARMM36 FFXML residue name):
  `CYS|CYH|CYD -> "CYS"`, `HIS -> "HSD"`, all others identity. (GLY has no sidechain
  template; skip. Variants share heavy-atom ring geometry, so HSD is fine for ICs.)
- `apply_charmm_ideals`: for each atom index `i` with `Some(BondDef)`:
  - `parent = bond.parent_idx`; `grandparent = bonds[parent].map(|b| b.parent_idx)`
    — **but match the SAME third-atom convention the NeRF builder uses for the
    bond angle** (see `geometry/mod.rs build_standard_sidechain` — read it and
    replicate exactly which atom is the angle's grandparent, especially for atoms
    whose parent is a backbone atom N/CA/C). Get atom names for `(grandparent,
    parent, i)` and `(parent, i)`.
  - Look up CHARMM `bond_length` and `bond_angle_deg`; **override** the BondDef fields
    if found. **Leave `torsion_deg` and `relative_chi` unchanged** (improper/branch
    torsions are not sourced from HarmonicBond/Angle; that residual is acceptable v1).
  - `tracing::warn!` on any miss (atom/class/bond/angle not found) so coverage is auditable.

### T3 — Proline (`geometry/proline.rs` + `proline_template`)
- The proline ring is built by `ProlineBuilder` (angle-relaxation), not the generic
  NeRF. Feed it CHARMM ring ideals:
  - `apply_charmm_ideals` to the proline template bonds (CB-CG, CG-CD, CA-CB).
  - Change ring-closure target `CD_N_IDEAL` from `1.487` (CCD) to the CHARMM CD-N
    bond `1.455` (source it from `CharmmIdeals` for "PRO" rather than hardcoding if
    practical; otherwise a named const with a `// CHARMM N-CP3` comment).
  - Ensure the canonical build-frame backbone uses CHARMM proline N-CA = **1.434**
    (proline-specific; not the generic 1.458/1.486) wherever the proline builder or
    AC-G test constructs the N/CA/C frame.
- Update the `proline_template()` doc comment: remove "Engh-Huber placeholder";
  state ICs are CHARMM-sourced at convert time (cite #820 research doc).

### T4 — Converter (`src/bin/convert_rotlib.rs`)
- Add CLI flag `--charmm-xml` (default `src/proxide/assets/charmm/charmm36_protein.xml`).
- In `main`/`build_library`: call `load_charmm_ideals(&args.charmm_xml)` once; for each
  residue, fetch its template, `apply_charmm_ideals(&mut t, &ideals, charmm_name)`,
  then build coords. Proline path uses the CHARMM-adjusted proline template + builder.
- `eprintln!` a one-line coverage summary (residues processed, IC overrides applied,
  misses) so regeneration is auditable.

### T5 — Tests
- `tests/test_geometry_ac_g.rs`: change the expected proline ideals from CCD to CHARMM
  (bonds 1.434/1.527/1.537/1.537/1.455; angles 110.8/108.5/108.5(solved)/110.5/114.2),
  keeping the existing ±3°/±6°/±0.02-0.03 tolerance structure. The endo/exo distinctness
  (≥0.5 Å CG split) and χ-recovery (±2°) assertions stay. Update the test's canonical
  backbone N-CA to CHARMM proline 1.434.
- `cargo test -p proxide-rotlib` (AC-G, AC-R, converter AC-3) must pass.
- `cargo check -p proxide-rotlib --all-targets` must be warning-free
  (`#![deny(warnings)]` is in effect).

### T6 — Regenerate + measure drift (REPORT NUMBERS, do not claim)
- Build: `cargo build -p proxide-rotlib --bin convert_rotlib`
- Generate (input lib lives in the PARENT working dir; use the absolute path):
  ```
  ./target/debug/convert_rotlib \
    --input /home/marielle/projects/proxide/data/rotlibs/SimpleOpt1-5/ALL.bbdep.rotamers.lib \
    --charmm-xml src/proxide/assets/charmm/charmm36_protein.xml \
    --output /home/marielle/.claude/jobs/704bac3c/tmp/charmm-lib.pb.zst
  ```
- Run the drift harness (the `#[ignore]` measurement test) pointing `load_pb` at the
  new lib: `crates/proxide-confind/tests/test_drift_loadpb_small_pdb.rs`
  (`fn measure_loadpb_drift_vs_master`). Read the test to find how it locates the
  library + `small.pdb` (small.pdb is at `/home/marielle/repos/mosaist/testfiles/small.pdb`);
  adjust the path to the regenerated lib if needed (temporarily, do not commit a hack).
  Run: `cargo test -p proxide-confind --test test_drift_loadpb_small_pdb -- --ignored --nocapture`
- **Report** the new drift vs the previous baseline (max |Δ| 0.223, mean 0.021,
  median 0.003, 69% of 43 contacts over 5e-4, 13 pairs absent). State whether 5e-4
  is met; if not, by how much, and which residues dominate the residual.

---

## Acceptance
1. `cargo test -p proxide-rotlib` + `cargo check -p proxide-rotlib --all-targets` green.
2. AC-G validates against CHARMM ideals (not CCD) and passes for both PRO puckers.
3. convert_rotlib regenerates the full library using CHARMM ICs with a coverage summary.
4. Drift harness re-run; **measured** numbers reported vs the 0.223 baseline.
5. No `rotlib.bin` (CC BY-NC-SA) committed or redistributed; CHARMM36 FFXML is already
   bundled (license OK).

## Out of scope (residual / follow-up)
- CHARMM improper/IC-table torsions (CB improper, etc.) — not sourced from HarmonicBond/
  Angle. If the FFXML-harmonic rebuild does not reach 5e-4, the next lever is CHARMM IC
  tables; record as follow-on, do not attempt here.
- Forcefield XML → protobuf (praxia idea idea-001).
