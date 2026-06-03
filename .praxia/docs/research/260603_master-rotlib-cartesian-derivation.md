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

## 5a. IMPLEMENTED + MEASURED (2026-06-03) — geometry-equilibria hypothesis FALSIFIED

The fix was implemented (CHARMM ICs sourced from the bundled `charmm36_protein.xml` via
`proxide_core::forcefield`, applied to the 19 non-proline residue templates at convert
time; see `geometry/charmm_ic.rs`) and the drift harness re-run. **Result: the swap did
not reduce the drift.**

| Metric (small.pdb)      | Baseline (Engh-Huber/CCD) | After CHARMM (non-proline) |
|-------------------------|---------------------------|----------------------------|
| Max \|Δ\|               | 0.223                     | **0.212**                  |
| Mean \|Δ\|              | 0.021                     | 0.020                      |
| Median \|Δ\|            | 0.003                     | 0.003                      |
| Count over 5e-4         | 69% (43)                  | **71% (42)**               |

**Why ~no change — and why it's a real (negative) result:**
- `small.pdb` contains **no proline** (residues: ARG MET LYS GLN LEU GLU ASP). The top
  drift contacts are LEU↔LEU (Δ0.212), GLN↔ASP (Δ0.168), LYS/MET — **all residues that
  WERE converted to exact CHARMM bonds+angles.** The drift barely moved.
- The non-proline residues were *already* near-tetrahedral in Engh-Huber (~110° ≈ CHARMM),
  so sourcing CHARMM bond/angle equilibria was nearly a no-op for them.
- **Proline** — the one residue with a large (~5°) CCD-vs-MASTER angle gap — was kept on
  CCD because CHARMM's unstrained equilibrium ring angles collapse the single-DOF ring
  closure (solved CB-CG-CD → 85.5°, unphysical). And proline isn't even in this fixture.

**Conclusion:** ideal bond-length/angle geometry is **not** the dominant driver of the
confind `load_pb` contact-degree drift. The residual (~0.21 max, 71% over 5e-4) is
dominated by something else — most plausibly:
1. **CB improper / chi-independent torsions** (left unchanged at Engh-Huber/CCD): these set
   sidechain *orientation*; a wrong improper rotates the whole sidechain and changes contact
   degree even with perfect bonds/angles. Strongest suspect for the LEU↔LEU max drift.
2. **Rotamer-set / atom-order / chi-convention / variant semantics** (spec open question #6).
3. NeRF branch-torsion application and the shared backbone build frame.

`#869` (confind→`load_pb` migration) therefore **cannot be unblocked by IC geometry alone**;
the next lever is the improper/orientation torsions and a rotamer-set/atom-order audit, not
bond/angle equilibria.

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

## 7. #884 resolution — CB improper FALSIFIED; LEU geometry + chi convention are faithful

Follow-on investigation (task `260603_cb_improper_drift`) tested §5 suspects #1 and #2
directly against MASTER's stored Cartesians. Both are **measurement-verified** (synthetic
self-tests + CB-constancy invariant guard; convention disambiguated as NeRF = −IUPAC).

**Frame:** MASTER/MSL and proxide use the *identical* backbone frame (origin=CA, x=CA−N,
z=x×(C−CA); mosaist `mstrotlib.cpp:186`, proxide `frame.rs:115`), so MASTER's stored
sidechains compare directly in the canonical identity frame N=[-1,0,0] CA=[0,0,0] C=[0,1,0].
MASTER stores CB at a *single fixed* coord (0.479,−0.735,−1.241), std (0,0,0) across all
rotamers and identical across residues — there is no per-residue CB orientation.

**Suspect #1 — CB improper (C–N–CA–CB):** MASTER's value is **−120.64° (NeRF convention)**
vs proxide's −119.7° placeholder → **Δ −0.94°. FALSIFIED** as the drift driver. The
placeholder was already correct to ~1°. (`scripts/analysis/measure_master_cb_improper.py`.)

**Suspect #2 — LEU all-atom IC + chi/atom-order audit** (LEU = max-drift pair, 0.212;
`scripts/analysis/audit_leu_ic_vs_master.py`): bonds match within 0.02 Å; χ1, χ2, and the
CD2 χ2+120 branch all match MASTER's stored chi within ~0.02° → **chi convention and atom
order are faithful**. Largest residual is N–CA–CB = 108.37° vs template 110.5° (Δ −2.13°, a
benign Engh-Huber-vs-CCD difference the §5 CHARMM swap already showed does not move drift).
(An earlier audit pass reported a spurious 8.4° CB-angle mismatch — an artifact of measuring
C–CA–CB instead of N–CA–CB; corrected in commit `ecdb5f7`.)

**Conclusion:** sidechain *geometry* (bonds, angles, CB orientation) and the *chi/atom-order
convention* are **not** the source of the confind `load_pb` drift. With §5 (bonds/angles) and
both §7 suspects eliminated, the residual must lie in **rotamer-set composition / ordering /
per-bin probabilities** or in the **`load_pb` contact-degree algorithm itself** — a new
investigation (tracked as a fresh backlog item), not further IC geometry work.

## 8. #925 resolution — load_pb drift root cause is a backbone-FRAME bug in the converter

Follow-on investigation (task `260603_loadpb_rotamerset_audit`) localized the residual
ConFind `load_pb` contact-degree drift (max 0.223, 30/43 pairs > 5e-4 gate, 13 missing
pairs on small.pdb). All findings are measurement-verified (synthetic self-tests + the
parity test as an independent oracle).

**Isolation (algorithm & data exonerated).** The ConFind contact-degree algorithm and
rotamer enumeration are line-for-line identical to MASTER (`parallel.rs:94/107` ≡
`mstcondeg.cpp:265/272`; `cache.rs:103-115` ≡ `mstcondeg.cpp:122-124`). The parity test
`test_parity_small_pdb` — proxide ConFind fed MASTER's *own* `rotlib.bin` via `load()` —
passes 5/5 within 5e-4, proving the algorithm + MSL loader faithfully reproduce MASTER on
identical input. So the entire `load_pb` drift is in the `.pb.zst` path. Per-bin rotamer
data is otherwise identical: probs + chi are bit-identical for all shared interior/edge
bins; both have 9 rotamers/bin for LEU with per-bin prob-sum 1.0.

**Falsified suspect — grid.** The `.pb.zst` carries a redundant +180° φ/ψ wrap column
(37×37 vs MASTER's 36×36; the +180 bin is a byte-duplicate of −180). Rebuilding the pb as
36×36 left the drift **byte-identical** — `find_closest_angle` resolves ±180 ties to −180
(idx 0), so the duplicate bin is never uniquely selected. Grid duplication is cosmetically
wrong but operationally inert.

**ROOT CAUSE — N-origin backbone frame (PRIMARY).** `convert_rotlib.rs:171-174` builds
sidechain Cartesians with **N at the origin** (`backbone_n=[0,0,0]`, `backbone_ca=[1.458,0,0]`),
i.e. an N-origin frame, but `place_rotamer` (`rotlib.rs:412`) applies a **CA-origin**
`backbone_frame` to the stored coords. A Kabsch fit of pb-coords vs `rotlib.bin` coords gives
**RMSD 0.018 Å** over 1800 atoms with translation **t = (−1.458, 0.03, 0.05)** (−1.458 Å =
the N–CA bond) and a ~2° rotation. So every stored sidechain is mis-placed by ~1.458 Å in x
after placement → the dominant drift. **Confirmation experiment** (`rebuild_pb_36grid.py
--reframe-ca-origin`, subtract (1.458,0,0)): repairs contact topology (missing 13→2, matched
43→54) and cuts median drift 3.3× (0.00297→0.00090).

**RESIDUAL — Engh-Huber backbone idealization (SECONDARY, above gate).** After the
translation fix the per-atom residual vs MASTER is mean 0.10 Å / max 0.18 Å, concentrated in
long/charged sidechains (worst pairs A1↔A4 ARG-GLN 0.436, B4↔B7, A4↔A7) — the signature of a
~2° rotation about CA. This is the converter's idealized backbone axes (the `backbone_c`
direction / CHARMM-IC idealization) differing from MASTER's actual stored Cartesian frame —
the still-open #820 "how MASTER derived rotlib.bin Cartesians" question.

**Fix path (for #869).** (1) PRIMARY: change `convert_rotlib.rs` to build with **CA at the
origin** (translation; high-confidence, dominant). (2) RESIDUAL: match MASTER's exact
backbone frame axes to clear the 5e-4 gate (gated by #820). Regeneration is feasible — the
Dunbrack source `data/rotlibs/SimpleOpt1-5/ALL.bbdep.rotamers.lib` is present.

Artifacts: `scripts/analysis/rebuild_pb_36grid.py` (+ `.bth.toml`) — grid-rebuild and
CA-origin reframe experiments with self-tests and per-bin identity verification.
