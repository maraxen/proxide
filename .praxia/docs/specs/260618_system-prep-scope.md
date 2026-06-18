---
title: "System-Preparation Layer — Comprehensive Scope Audit"
epic: 977
task_id: 260618_sysprep-audit
date: 2026-06-18
status: draft (awaiting human review before child-item registration)
category: specs
sources:
  - recon (codebase surface inventory, task_id 260618_sysprep-audit)
  - librarian (external tool comparison + parameter-source licensing, task_id 260618_sysprep-audit)
  - "#869 load_pb drift investigation (CHARMM36 RTF IC gap — the triggering symptom)"
---

# System-Preparation Layer — Comprehensive Scope Audit (Epic #977)

## 1. Why this audit exists

The CHARMM36 RTF internal-coordinate gap found during the #869 `load_pb` drift
investigation (missing cis-PRO entry; frame drift 0.457 Å → 0.043 Å once patched)
was not an isolated bug. It is a **symptom of a layer built incrementally without a
map**: proxide's "system preparation" pipeline — everything that turns a raw
PDB/mmCIF structure into a scoring-/simulation-ready model — has accreted feature by
feature, with whole stages still stubbed or absent.

This document is the deliverable for the #977 RESEARCH stage. It (a) defines the full
component set of protein system prep, (b) inventories proxide's current coverage with
file anchors, (c) benchmarks the field's reference tools and the licensing of every
candidate data source, and (d) proposes a prioritized child-item DAG. **No code is
written and no child items are registered until this scope is reviewed.**

## 2. The 12-component system-prep model

System prep is a pipeline of ~12 sequentially-dependent operations. No single
open-source tool covers all of them (the closest all-in-one is the proprietary
Schrödinger PrepWizard). The components, in rough dependency order:

| # | Component | "Done right" means |
|---|-----------|--------------------|
| 1 | Structure I/O & parsing | Robust PDB/mmCIF/PQR read; altloc selection; insertion codes; SEQRES vs ATOM reconciliation |
| 2 | Geometry / IC placement | Heavy-atom + H placement matching Engh-Huber ideal bond/angle/dihedral; chirality & planarity enforced |
| 3 | Protonation (pH-dependent) | pKa-aware state assignment for ASP/GLU/HIS/LYS/ARG/CYS/TYR; HIS tautomer (HID/HIE/HIP) resolution |
| 4 | Missing heavy-atom completion | Place absent backbone/sidechain atoms from residue template geometry |
| 5 | Missing sidechain / rotamer placement | Rebuild truncated sidechains via backbone-dependent rotamer library + clash avoidance |
| 6 | Disulfide detection | Sγ–Sγ scan (~2.05 Å, threshold 2.5 Å) → CYS→CYX rename / DISU patch; drop SH H |
| 7 | Chain termini patching | Charged NTER/CTER for true ends; distinguish from internal chain breaks |
| 8 | Capping groups (ACE/NME) | Neutral caps on internal breaks where a loop can't be built |
| 9 | Point mutations | Residue swap + sidechain rebuild + local repack |
| 10 | Insertions/deletions/loop modeling | Build missing loops (fragment/KIC/NGK ≤10 res; AF2 for long disordered) |
| 11 | Waters & ions | Crystal-water triage; solvation box; counterion neutralization + ionic strength |
| 12 | Forcefield parameter assignment | atom types + LJ + partial charges for every atom; custom params for non-standard residues/ligands |

## 3. proxide current coverage (from recon)

Status legend: **present** (works) · **partial** (stub / untested / classification-only) · **missing**.

| # | Component | Status | Anchor (file:line) | Note |
|---|-----------|--------|--------------------|------|
| 1 | Structure I/O | **present** | `crates/proxide-io/src/formats/{pdb.rs:13,mmcif.rs}`, `src/proxide/io/parsing/{pqr.py:130,foldcomp.py,mdtraj.py,dispatch.py:36}` | PDB/mmCIF/PQR/FoldComp/trajectory; unified `load_structure` |
| 2 | Geometry / IC | **partial** | `crates/proxide-rotlib/src/geometry/{charmm_ic.rs:72,proline.rs}`; `crates/proxide-core/src/processing/residues.rs:16` | CHARMM36 ideals load; proline closure works; **non-proline IC untested; no Engh-Huber fallback** |
| 3 | Protonation | **partial** | `crates/proxide_fixer/src/protonate.rs:6` | **ALA-only stub; dummy HA; no pH / pKa / HIS tautomers** |
| 4 | Missing heavy atoms | **partial→missing** | `crates/proxide_fixer/src/builder.rs` | Builder module exists, scope unknown; no backbone-gap completion |
| 5 | Sidechain / rotamer | **partial** | `crates/proxide-rotlib/src/rotlib.rs` | Dunbrack 2010 lookup present; **not wired into a completion/repack path** |
| 6 | Disulfide detection | **missing** | — | No Sγ–Sγ scan, no CYX rename |
| 7 | Termini patching | **partial** | `crates/proxide_fixer/src/sanitizers/capping.rs:23,36` | H1/OXT placeholders; **no formal NTER/CTER; no break-vs-end logic** |
| 8 | Capping (ACE/NME) | **missing** | — | Only H1/OXT placeholders, no ACE/NME |
| 9 | Point mutations | **missing** | — | No API |
| 10 | Indels / loops | **missing** | `pdb.rs:40` (insertion codes parsed only) | No loop closure / coordinate inference |
| 11 | Waters & ions | **partial** | `crates/proxide-core/src/processing/residues.rs:101,105` | Classification only (solvent/ion); **no placement / neutralization** |
| 12 | FF params | **present (CHARMM/GAFF) / partial (gaps)** | `src/proxide/assets/charmm/charmm36_protein.xml`, `src/proxide/chem/gaff2.py`, `src/proxide/physics/force_fields/loader.py:126` | CHARMM36 + GAFF2 present; **no Amber ff14SB/ff19SB; CHARMM cis-PRO gap (#869)** |

**Crate ownership:** `proxide-io` (parse) → `proxide-core` (topology/classification) →
`proxide_fixer` (the system-prep home — *mostly stubbed*) → `proxide-rotlib`
(IC + rotamers) → `proxide-physics` (FF params) → `proxide-confind` (scoring readiness).

**Scoring-readiness risk:** ConFind (`crates/proxide-confind`) *assumes* every residue
has N/CA/C and valid φ/ψ and a consistent topology — preconditions nothing currently
enforces. Missing atoms / unhandled breaks fail silently downstream. Test coverage is
strong for ConFind itself (17 test files) but there are **no end-to-end system-prep
integration tests**.

## 4. Reference-tool coverage matrix (from librarian)

Condensed; full citations in the librarian report (28 sources). Y=covered,
P=partial/fixed-rules, —=absent.

| Component | Modeller | tLeap | OpenMM/PDBFixer | Rosetta | PDBFixer-only |
|-----------|:--------:|:-----:|:---------------:|:-------:|:-------------:|
| IC / H placement | — | Y | Y | P | Y |
| pH protonation | — | P | P | — | P |
| **pKa prediction** | — | — | — | — | — (all delegate to PROPKA/PDB2PQR) |
| Missing heavy atoms | Y | — | Y | Y | Y |
| Sidechain rotamer | Y | — | P (no rotlib) | **Y (Packer+BBDEP)** | P |
| Disulfide detect | Y | manual | Y | Y | Y |
| Termini patch | P | Y | Y | Y | Y |
| ACE/NME cap | — | Y | P | Y | — |
| Point mutation | Y | — | — | Y | — |
| Loop modeling | Y | — | — | **Y (KIC/NGK)** | P (short only) |
| Water/ion | — | Y | Y | — | Y |
| FF assignment | — | Y | Y | Y | — |

**Takeaways:** (1) *No* OSS tool computes pKa itself — the field standard is to chain
**PROPKA3/PDB2PQR** before tLeap/OpenMM. (2) Rosetta dominates sidechain packing and
loop modeling but is non-commercial-licensed. (3) **PDBFixer (MIT)** is the single most
license-clean repair tool and covers most of components 4,6,7,11. (4) Classical loop
modeling degrades sharply >10 residues — honest behavior is to delegate long loops.

## 5. Parameter-source licensing (MIT-compatibility)

Decision-critical, since proxide is MIT-code / open-data.

**Green — use freely:**
- **CHARMM36 parameters** — effectively **public domain** (CHARMM forum: "the CHARMM
  force fields are public domain, so no license is required"; MacKerell will issue a
  permissive CC license on request). Caveat is courtesy, not legal: don't ship *stale*
  copies. NB the CHARMM **program** is commercial — a separate thing from the params.
- **PDBFixer** (MIT), **OpenMM** (MIT/LGPL) — repair + solvation reference impls
- **PROPKA3** (LGPL-2.1 — linkable from MIT) — the only embeddable pKa engine
- **PDB2PQR** (BSD) — charge/radius assignment, wraps PROPKA
- **wwPDB CCD** (public domain) — ideal geometry/SMILES for all heterogens
- **OpenFF Sage** (MIT) — small-molecule params (SMIRNOFF)
- **Dunbrack BBDEP2010** (free, attribution) — already vendored in `proxide-rotlib`
- **Engh-Huber** ideal geometry — published tables, replicable

**Yellow — usable under own license, bundle per-asset:**
- **Amber ff14SB / ff19SB / GAFF2** — **GPL** parameter files (via AmberTools). Usable;
  vendoring puts the GPL on *that asset directory* (data, not code) — proxide's MIT code
  is unaffected. Note: `openmmforcefields` excludes **ff19SB** (OPC-water coupling, not
  licensing).

**Red — cannot redistribute in MIT project:** Modeller (proprietary academic),
Rosetta/PyRosetta (non-commercial), H++ (server-only), CSD/GRADE.

**Correction vs first draft:** CHARMM36 params are *more* permissive (public domain)
than Amber params (GPL), not less — the earlier "ambiguous" framing conflated the
commercial CHARMM *program* with its public-domain *parameters*.

## 6. Gap prioritization — ranked by ConFind/scoring/design impact

| Rank | Gap | Impact rationale | Complexity |
|------|-----|------------------|------------|
| 1 | **Input validation / precondition gate** for ConFind | Silent failures today; cheap; unblocks honest error reporting for every other gap | low |
| 2 | **Disulfide detection** (#6) | Wrong topology → wrong scoring; low complexity (Sγ–Sγ scan); coords already in `ProcessedStructure` | low |
| 3 | **Non-proline IC validation + Engh-Huber fallback** (#2) | Directly extends the #869 fix; geometry correctness underpins all scoring | medium |
| 4 | **Termini patching + ACE/NME** (#7,#8) | Required for any valid FF assignment & charge-neutral interior | medium |
| 5 | **IC-based H placement** (#3 geometry half) | Blocks FF param assignment; templates already parsed in `xml_parser` | medium |
| 6 | **pH protonation via PROPKA3** (#3 chemistry half) | Highest correctness impact; LGPL-clean; default to pH-7.4 fixed rules without it | medium-high |
| 7 | **Sidechain completion + repack** (#4,#5) | Natural home in `proxide-rotlib`; SCWRL-style greedy over BBDEP2010 | high |
| 8 | **Point mutations** (#9) | Builds on #7 | medium (after #7) |
| 9 | **Loop modeling** (#10) | High value, high complexity; recommend delegate long loops, short via fragments | high |
| 10 | **Waters/ions placement** (#11) | MD-oriented; lower priority for scoring/design use cases; delegate to PDBFixer | medium |
| 11 | **Amber ff14SB/ff19SB support** (#12) | Breadth; gated on GPL-bundling decision | medium |
| 12 | **End-to-end system-prep integration tests** | Cross-cutting; should land alongside each gap, not as one item | — |

## 7. Recommended sequencing

Three waves, dependency-ordered:

- **Wave A (foundation, parallelizable):** ConFind precondition gate · disulfide
  detection · non-proline IC validation + Engh-Huber fallback. All low/medium, mostly
  independent, immediately improve correctness.
- **Wave B (topology completeness):** termini patching + ACE/NME → IC-based H placement
  → PROPKA3 protonation. Sequential (H placement depends on patched topology;
  protonation tunes which H exist).
- **Wave C (rebuild capabilities):** sidechain completion/repack → point mutations →
  loop modeling → waters/ions → Amber FF. Higher complexity; #9/#10 build on #7.

Each wave should ship with integration-test fixtures (the cross-cutting test gap).

## 8. Proposed child-item DAG — FOR REVIEW (not yet registered)

Parent: #977. Suggested fields per item: `category`, `difficulty`, `priority`,
`depends_on`, `workflow_hint`.

| Tag | Title | cat | diff | prio | depends_on |
|-----|-------|-----|------|------|------------|
| C1 | ConFind input-precondition gate (N/CA/C + φ/ψ + topology) with diagnostics | feature | standard | P1 | — |
| C2 | Disulfide detection (Sγ–Sγ scan → CYX rename / DISU) | feature | quick | P1 | — |
| C3 | Non-proline CHARMM IC validation + Engh-Huber fallback | feature | standard | P1 | — |
| C4 | Formal N/C termini patching (NTER/CTER; break-vs-end detection) | feature | standard | P2 | C1 |
| C5 | ACE/NME capping for internal chain breaks | feature | standard | P2 | C4 |
| C6 | IC-based hydrogen placement from FF residue templates | feature | standard | P2 | C4 |
| C7 | pH-dependent protonation via PROPKA3 (LGPL); pH-7.4 fallback | feature | hard | P2 | C6 |
| C8 | Sidechain completion + greedy repack over BBDEP2010 | feature | hard | P2 | C3 |
| C9 | Point mutation API (swap + rebuild + local repack) | feature | standard | P3 | C8 |
| C10 | Loop modeling (fragment/KIC short; delegate long) | feature | hard | P3 | C8 |
| C11 | Water/ion placement + neutralization (or delegate to PDBFixer) | feature | standard | P3 | — |
| C12 | Amber ff14SB/ff19SB parameter support | feature | standard | P3 | — |
| C13 | End-to-end system-prep integration test harness + fixtures | test | standard | P1 | — |

## 9. Open decisions needed from the PI

1. **FFI vs reimplementation for PROPKA3 / PDBFixer** — wrap (subprocess/FFI) the MIT/LGPL
   reference impls, or reimplement natively in Rust? (Affects C7, C11 scope a lot.)
2. **GPL bundling** — is bundling Amber GPL parameter assets acceptable for an MIT project
   (C12), or keep Amber out and stay CHARMM/OpenFF only?
3. **Loop modeling ambition** (C10) — implement short-loop building, or just detect +
   delegate/flag for now?
4. **Waters/ions** (C11) — in scope for proxide (design/scoring focus), or defer as MD-only?
5. **Scope of #977 closure** — does #977 close when this scope doc + registered children
   exist, or should it also include Wave A implementation?

## 10. PI review decisions (2026-06-18)

1. **Implementation strategy — wrap → native, staged, CI-parity-gated.** Start each
   science-heavy stage by *wrapping* the license-clean reference impl (PDBFixer for
   repair/solvation; PROPKA3 for pKa), then migrate to native Rust **only behind a CI
   parity test** that asserts the native output matches the wrapped reference within
   tolerance. This is a cross-cutting principle for C7, C10, C11 (and any stage where a
   reference impl exists). Each native migration is its own follow-up item gated on the
   parity harness. Cheap geometry stages (C2/C3/C4/C5/C6) go native directly.
2. **Forcefield licensing — bundle per-asset under each FF's own license.** proxide code
   stays MIT; each vendored FF lives in its own dir with its upstream LICENSE/NOTICE.
   CHARMM36 = public-domain + attribution (already present); Amber = GPL-in-subdir;
   OpenFF Sage = MIT. C12 retained as a per-asset-licensed bundle (the corrected
   licensing picture in §5 supports this — CHARMM params are public domain, not
   commercial; the CHARMM *program* is the commercial part).
3. **Next step — register C1–C13 under #977, then DESIGN + implement Wave A** (C1
   precondition gate, C2 disulfide detection, C3 non-proline IC validation).

## 11. Sources (licensing verification)

- CHARMM params public domain: [CHARMM forums — parameter file distribution](https://forums-academiccharmm.org/viewtopic.php?t=8358); [MacKerell CHARMM FF](https://mackerell.umaryland.edu/charmm_ff.shtml)
- Amber params GPL: [AmberTools (ambermd.org)](https://ambermd.org/AmberTools.php); [openmm/openmmforcefields](https://github.com/openmm/openmmforcefields)
- (Full external-research citation list: librarian report, task_id 260618_sysprep-audit, 28 sources.)

---

*RESEARCH stage complete; PI review folded in (2026-06-18). Proceeding to REGISTER + DESIGN Wave A.*
