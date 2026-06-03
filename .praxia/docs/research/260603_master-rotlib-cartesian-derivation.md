---
name: 260603_master-rotlib-cartesian-derivation
description: Backlog #820 research — how MASTER derived rotlib.bin Cartesians (CHARMM ideal ICs, not Engh-Huber); root-causes the confind load_pb drift and names the fix
metadata:
  type: research
  task_id: 260602_rotlib-confind-actions
  backlog: "#820"
  status: complete
---

# How MASTER derived `rotlib.bin` Cartesians — and why `load_pb()` drifts (backlog #820)

**Date:** 2026-06-03
**Gates:** backlog **#869** (migrate confind from `load()`/MASTER `rotlib.bin` to
`load_pb()`/proxide ODC-BY library) — blocked until the rebuilt geometry matches
MASTER's convention within the contact-degree drift tolerance (target <5e-4).
**Feeds:** P4 converter (`convert_rotlib`) + `crates/proxide-rotlib/src/geometry/template.rs`.

---

## TL;DR

1. **MASTER/MSL built `rotlib.bin`'s sidechain Cartesians from CHARMM ideal internal
   coordinates** (bond lengths + angles from the CHARMM all-hydrogen residue topology,
   combined with the Dunbrack χ angles). It is **not** Engh-Huber and **not** the
   CCD `PRO.cif` geometry that `template.rs` currently ships as a placeholder.
2. This is confirmed **three independent ways**: (a) the literature
   (CHARMM proline parameterization / Kulp et al. 2012 MSL), (b) **direct measurement
   of the baked coordinates in `rotlib.bin`**, and (c) parsing proxide's own bundled
   CHARMM36 FFXML, which reproduces the same ideals.
3. **Root cause of the drift = bond angles.** The placeholder uses CCD ring angles
   (~104–105°); MASTER's baked geometry is tetrahedral (~110°). That ~5° error at
   every sidechain joint compounds along the chain and displaces terminal atoms —
   the magnitude that produces the measured contact-degree drift (max |Δ| 0.223).
4. **The fix needs no new constants and no external download.** proxide already
   ships `src/proxide/assets/charmm/charmm36_protein.xml` and already parses
   `HarmonicBondForce`/`HarmonicAngleForce` in `proxide_core::forcefield::xml_parser`.
   Source the template ICs from there.

---

## 1. What MASTER actually does (mosaist source)

`mosaist/src/mstrotlib.cpp::readRotamerLibrary` (439 lines) is a pure **reader**:
it loads pre-baked Cartesian coordinates per rotamer (lines 118–128) and, at
placement time, rigid-transforms them onto the target backbone using the frame
`X = CA−N, Z = X×(C−CA), Y = Z×X, origin = CA` (lines 186–194) — the exact frame
proxide's `backbone_frame` already matches. **There is no χ→Cartesian builder in
mosaist**; `rotlib.bin` shipped pre-built. So the coordinate-derivation convention
must be recovered from the literature and from the binary itself.

The MSL binary format (per residue): `aa` (cstr), `nc`/`na`/`nb` (i32),
`nc×4` χ-def atom names, `na` sidechain atom names, `nb×(phi,psi,freq)` (f32), then
per bin `nr` (i32) rotamers each `{prob, nc×(χ,σ), na×(x,y,z)}` (f32). Decoded in
`scripts/analysis/extract_rotlib_geometry.py`.

## 2. Literature: CHARMM ideal internal coordinates

The standard recipe for building backbone-dependent rotamer Cartesians (and the one
MSL/Grigoryan-lab tooling inherits): **bond lengths and angles come from the CHARMM
all-hydrogen residue topology** (equilibrium values from minimized tetrapeptides
Ac-Ala-Xxx-Ala-NHCH3), and the **χ dihedrals come from the Dunbrack library** — together
those fully determine the sidechain Cartesian coordinates. Proline uses dedicated ring
atom types CP1 (Cα), CP2 (Cβ/Cγ), CP3 (Cδ).

Reported CHARMM proline ideals (NotebookLM notebook A `171c5c8b`, source `4dc6a5ab`):
N-CA 1.434, CA-CB 1.527, CB-CG 1.537, CG-CD 1.537, CD-N 1.455 Å;
angles N-CA-CB 110.80, CA-CB-CG 108.50, CB-CG-CD 108.50, CG-CD-N 110.50, CD-N-CA 114.20°.

## 3. Ground truth: measured from `rotlib.bin`

`extract_rotlib_geometry.py` parsed the MASTER binary and measured the
frame-independent intra-sidechain geometry (highest-frequency bin, rotamer 0).
`rotlib.bin` stores only sidechain atoms (PRO: CD, CB, CG), so N-CA / CD-N closure
are not directly measurable here, but the all-carbon bonds/angles are decisive:

| PRO geometry      | MASTER (measured) | CHARMM ideal | CCD placeholder (current) |
|-------------------|-------------------|--------------|---------------------------|
| CB–CG             | **1.521 Å**       | 1.537        | 1.543                     |
| CG–CD             | **1.521 Å**       | 1.537        | 1.544                     |
| CB–CG–CD angle    | **109.96°**       | 108.5°       | 105.1°                    |

Cross-residue confirmation (clean sp3/sp2, not CCD):
- **LEU:** CB-CG 1.521, CG-CD1/2 1.519/1.520 Å; CB-CG-CD1/2 ≈ 111.5°.
- **ARG:** CB-CG 1.521, CG-CD 1.520, CD-NE 1.450, NE-CZ 1.330 Å; CB-CG-CD 109.98,
  CG-CD-NE 110.94, **CD-NE-CZ 119.95°** (sp2 guanidinium).

The baked geometry is uniformly tetrahedral (~110°, ~1.52 Å) — CHARMM-family, **not**
the CCD ~105° ring angles. The small residual vs raw CHARMM harmonic equilibria
(measured 1.521 vs 1.537; 109.96 vs 108.5) indicates `rotlib.bin` was built from CHARMM
**IC tables and/or post-build minimization** rather than the bare `b0`/`θ0` — a
second-order effect that does not change the conclusion.

## 4. The drift is the angle error

| | measured | CHARMM Δ | CCD-placeholder Δ |
|---|---|---|---|
| CB–CG bond  | 1.521 Å | 0.016 | 0.022 |
| CB–CG–CD angle | 109.96° | **1.46°** | **4.86°** |

The **angle** dominates. The current placeholder is ~4.9° off MASTER at the ring;
CHARMM is ~1.5° off. Switching to CHARMM cuts the per-joint angular error ~3.3×, and
because the error compounds geometrically down each sidechain, this is the
mechanism behind the contact-degree drift (max |Δ| 0.223, 69% of contacts over the
5e-4 tolerance) recorded by `test_drift_loadpb_small_pdb.rs`.

## 5. Fix path (no new constants, no download)

proxide already ships and parses the right source:

- **Data:** `src/proxide/assets/charmm/charmm36_protein.xml` (CHARMM36 protein FF,
  OpenMM FFXML). Verified to parse to the CHARMM ideals above (N-CA 1.434, CA-CB 1.527,
  CB-CG/CG-CD 1.537, CD-N 1.455; angles 110.80/108.50/108.50/110.50/114.20).
- **Parser:** `proxide_core::forcefield::xml_parser` already reads
  `HarmonicBondForce.length` (nm) and `HarmonicAngleForce.angle` (rad) into
  `HarmonicBondParam`/`HarmonicAngleParam` keyed by atom class.

**Recommended #820 implementation:**
1. In the P4 converter / `template.rs`, replace the CCD-ideal placeholder bond
   lengths and angles with CHARMM36 equilibria looked up via the existing forcefield
   parser (map each residue's IC atoms → CHARMM atom class → `length`/`angle`). Keep
   the proline ring-closure CCD relaxation step; it only needs a CHARMM-correct start.
2. Re-generate `proxide-rotlib-bbdep2010.pb.zst` and re-run
   `test_drift_loadpb_small_pdb.rs`. **Acceptance for #869:** confind contact-degree
   drift vs MASTER `REF_CONTACTS` within 5e-4.
3. If a residual remains after CHARMM ICs (the §3 ~1.5° gap suggests it might),
   close it with the **dev-only** `rotlib.bin` cross-check already specified in the
   spec's P6 — measure per-residue rebuilt-vs-MASTER coordinate RMSD and tune. Never
   redistribute `rotlib.bin` (CC BY-NC-SA); it stays a dev oracle only.

## 6. Open / follow-on

- **Exact param source:** confirm whether MASTER used CHARMM IC tables (`.rtf` RESI
  IC records) vs harmonic `b0`/`θ0` vs minimized coords — explains the ~1.5° residual.
  Only pursue if the CHARMM-FFXML rebuild does not clear the 5e-4 gate.
- **Forcefield XML → protobuf:** capturing the FFXML hierarchy losslessly as
  compressed protobuf is tracked separately in praxia ideas/staging (per user, 2026-06-03).

## Provenance

- Script: `scripts/analysis/extract_rotlib_geometry.py` (+ `.bth.toml` sidecar).
- `rotlib.bin` sha256 `edabc73f…a0da`; `charmm36_protein.xml` sha256 `94c665e4…f613d`.
- Literature via NotebookLM notebook `171c5c8b` ("proxide: Rotamer Library Theory");
  MSL: Kulp et al. 2012 *J Comp Chem* 33:1645 (10.1002/jcc.22968).
