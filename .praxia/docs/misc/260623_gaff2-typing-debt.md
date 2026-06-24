---
name: gaff2-typing-debt
description: Tech debt — deeper audit of GAFF2 atom typing + parity CI tests against AmberTools/OpenFF reference
metadata:
  type: project
---

Three bugs in `proxide/chem/gaff2.py` were found while integrating GAFF2 into the biosensors naurmalade pipeline (validated on 17-OHP / cpOHPFA1952-20-BLA-253). Fixed in commit `{see git log}`, but the fixes are targeted rather than comprehensive.

**Why:** The implementation has no automated parity tests against a reference (AmberTools `antechamber` or the OpenFF GAFF2 plugin). The bug-finding process required manual tracing through DEF rule parsing and molecule-specific debugging. A systematic audit + CI suite would make future regressions obvious.

**How to apply:** Before wiring GAFF2 into any production pipeline, run the parity suite (once it exists). Until then, use SMIRNOFF as the production FF and treat proxide GAFF2 as experimental.

---

## Debt items

### 1. Audit h_ew checking (not implemented in `matches()`)

`h_ew` stores bond-type patterns like `2DL`, `1DB,0DL`, `3sb` from the ATD rules (f7 field). The `matches()` method never evaluates these — meaning the three `cs` rules and three `c` rules that differ only in h_ew are all equivalent from the matcher's perspective.

**Impact:** For any sp2 carbon, the first matching `cs` or `c` rule fires regardless of whether the actual bond pattern matches `[2DL]`, `[1DB,0DL]`, or `[3sb]`. This may produce incorrect types for delocalized systems (amides, enolates) vs. pure double-bond systems.

**Fix needed:** Implement h_ew matching in `matches()`:
- Parse `[2DL]`, `[1DB]`, `[3sb]`, `[AR1]`, etc. from h_ew
- DL = delocalized (check bond order ≈ 1.5 via rdkit AROMATIC bond type or resonance)
- DB = double bond (bond type DOUBLE)
- sb = single bond (bond type SINGLE)
- Match against the atom's actual bonds

### 2. Parser audit for multi-wildcard ATD lines

The parser consumes a leading `*` before h_ew as a prefix marker (Bug C origin). Lines with multiple wildcards like `ATD op * 8 2 * * [RG3] &` place the ring constraint in h_ew instead of chem_env. The workaround in `matches()` catches `RGn` in h_ew, but a deeper audit should verify no other rules have mis-placed fields.

**Action:** Grep all parsed rules for cases where h_ew contains non-bond-type tokens (anything besides `sb`, `db`, `AR`, `DL`, `tb` patterns).

### 3. Parity test suite against OpenFF GAFF2 reference

Add a CI test that:
1. Takes a set of test molecules (17-OHP, a simple amino acid, an amide, a thioester)
2. Runs `assign_gaff2_atom_types()` to get proxide GAFF2 types
3. Compares against the OpenFF GAFF2 plugin (`openmmforcefields.generators.GAFFTemplateGenerator` using antechamber output) or a stored reference fixture
4. Asserts exact match on all heavy atoms

This should be a `pytest` fixture with stored ground-truth JSON — not a live antechamber call (antechamber not available in CI).

**Reference molecule set:**
- 17-OHP (steroid, enone + ketone + hydroxyl): `C[C@]12CC[C@H]3...C(=O)CO`
- Acetamide (amide carbon): `CC(=O)N`
- Dimethyl sulfide (thioester alpha C): `CSC`
- Furan (aromatic O): `c1ccoc1`
- 3-methylenecyclopropane (exocyclic double bond in 3-ring): check op vs other

### 4. Integration: wire proxide GAFF2 into naurmalade holo pipeline

See biosensors project. Currently naurmalade uses SMIRNOFFTemplateGenerator (Bug 13 fix). Proxide GAFF2 integration requires:
- Building an OpenMM FFXML from proxide output (bonds, angles, torsions, charges)
- Or using GAFFTemplateGenerator with proxide-typed SYBYL mol2 output
- Then validating against SMIRNOFF smoke runs for energy/force parity

Status: deferred pending parity CI (item 3 above).
