# GAFF2 Parity Benchmark Sources

## AmberMD Geostd Tarball

**Resource:** https://ambermd.org/downloads/amber_geostd.tar.bz2 (~29,000 pre-parameterized PDB-ligand files)

**Description:** Official AmberMD geometric standard library containing pre-parameterized ligand structures keyed by PDB 3-letter residue codes. Provides direct ground-truth comparisons for GAFF2 parameterization validation.

**Suggested use:** Supplement-tier benchmark — extract a representative subset (e.g., top 100-200 by PDB frequency) for parity validation against the GAFF2 paper's reference parameters.

**Access:** Direct download from AmberMD mirrors; no credentials required.

---

## Biosensors Integration Recommendation

**Target file:** `/home/marielle/projects/biosensors/scripts/md/validate_gaff2_vs_smirnoff.py` (out of scope for this task, but noted as a dependency)

**Integration strategy:** NOT A FINAL DECISION — this recommendation is for future evaluation by the biosensors team.

- **Option A (preferred if possible):** Import biosensors as a pip-installable dependency and reuse validation functions directly
- **Option B (fallback):** Parse biosensors validation via subprocess + stdout capture, if biosensors is not installable in the proxide environment

**Rationale:** Avoid reimplementing biosensors' GAFF2-vs-SMIRNOFF parity checks; consolidate validation logic in biosensors, then wire it as an external oracle for proxide's GAFF2 validation pipeline.

**Status:** Requires discussion with biosensors team (out of current scope). This file serves as a note for that future conversation.
