---
title: proxide-confind — full-protein CB distogram parity test
task_id: 260601_confind-fullprot-distogram
session_id: 521fb08b-a39b-47e1-9069-a4796924fbc6
status: in_progress
date: 260601
branch: main
head: 689e98c
suite_confind: "52 passed, 36 ignored (cargo test -p proxide-confind)"
suite_rotlib: "80 passed, 1 ignored (cargo test -p proxide-rotlib)"
---

## Session summary

Started from the proxconfind3 close handoff (status: complete) and the rotlib-fixture-expansion
handoff. Commit `4c9fa11` had already closed chain B + GLY/PRO distogram items from the rotlib
handoff, leaving only the "diverse PDB" item open.

Added `689e98c` this session:
- `test_distogram_2zta_chain_a` to `proxide-rotlib` — 30×30 CB-CB matrix over GCN4 leucine
  zipper chain A (14 unique AAs, real helical backbone, sentinel phi/psi)
- `real_pdb_path_1dc7()` and `real_pdb_path_2zta()` helpers to rotlib `helpers.rs`
- Fixed hardcoded 1DC7 absolute path in `test_distogram_with_gly_and_pro`

**User correction:** The distogram concept belongs in `proxide-confind`, not `proxide-rotlib`.
The next session should add a full-protein distogram test (all 20 AAs) to confind, and decide
whether to revert/remove the 2ZTA distogram from rotlib.

---

## Fixture confirmed

`1DC7.pdb` (124 residues, chain A) contains all 20 standard amino acids:
ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL

This is already used in `test_parity_1dc7.rs` and has helpers in `common/mod.rs`.

---

## Next steps

1. **Decide on 689e98c**: Revert the 2ZTA distogram from rotlib, OR keep it as a lower-level
   rotlib placement unit test and also add the full-protein distogram to confind. User said
   "distogram should not be in rotlib" — interpret this decision at the start of the session.

2. **Resolve GLY and PRO convention** (see open questions below) before computing the reference.

3. **Run the Python reference computation** from `.praxia/docs/handoffs/260601_rotlib-fixture-expansion.md`
   (lines 68–156 — the full `load_rotlib` + frame math + `place_first_atom` + `parse_pdb_backbone`
   script) against 1DC7 chain A. Adjust for GLY/PRO convention chosen in step 2.

4. **Add `test_distogram_1dc7_chain_a`** to `crates/proxide-confind/tests/` (likely a new
   `test_distogram.rs` file, modeled on `test_parity_1dc7.rs`). Use `real_pdb_path_1dc7()` from
   `common/mod.rs`. Reference matrix at 1e-6 Å tolerance.

5. **Run `cargo test -p proxide-confind`** — confirm all 52 existing parity tests still pass.

---

## Open questions

1. **GLY in distogram**: rotlib gives `na=0` for GLY so there is no sidechain atom. Options:
   - Skip GLY rows/cols entirely → ~117×117 matrix (7 GLY in 1DC7 chain A)
   - Use CA as CB surrogate for GLY (Mosaist convention for some operations)
   - Include GLY with a sentinel (NaN or 0.0) and document it
   Check what Mosaist's contact-finding code does for GLY before deciding.

2. **PRO in distogram**: rotlib places CD (not CB) as PRO's first sidechain atom. Options:
   - Use CD in the distance matrix (consistent with what rotlib places)
   - Use CA as surrogate
   - Skip PRO (6 PRO in 1DC7 chain A)
   Check Mosaist's contact/distogram convention for PRO.

3. **Should 689e98c be reverted?** User direction was "distogram should not be in rotlib."
   If interpreted strictly, revert the 2ZTA test. If lower-level placement tests are acceptable
   in rotlib, keep it and add the protein-level distogram to confind.

---

## Key files

| Path | Role |
|------|------|
| `crates/proxide-confind/tests/common/mod.rs` (lines 231–248) | `real_pdb_path_1dc7()` + `load_1dc7_backbone()` helpers — reuse |
| `crates/proxide-confind/tests/test_parity_1dc7.rs` | Existing 1DC7 test; model distogram test structure here |
| `crates/proxide-rotlib/tests/test_distogram.rs` | `689e98c` changes — `dist()` helper and const pattern to replicate; possibly revert |
| `.praxia/docs/handoffs/260601_rotlib-fixture-expansion.md` (lines 68–156) | Python reference-computation script (frame math) — run against 1DC7 |
| `/home/marielle/repos/mosaist/testfiles/1DC7.pdb` | Fixture: all 20 AAs, 124 residues chain A |

---

## Context refs

- Prior: `.praxia/docs/handoffs/260601_proxconfind3-close.md` — confind complete (52/36)
- Prior: `.praxia/docs/handoffs/260601_rotlib-fixture-expansion.md` — rotlib fixture gaps + Python reference script
- This session commit: `689e98c` — 2ZTA distogram in rotlib (may revert)

---

## Deferred

- **Real phi/psi backbone_bin exercise**: compute actual φ/ψ from adjacent backbone atoms and
  pass to `place_rotamer`, exercising the backbone_bin grid lookup path. Requires replicating
  Mosaist bin selection in Python. Non-trivial; not on critical path.
- **Debt #67**: Dunbrack / multi-rotlib / PRO cis-trans support. Architectural; separate item.
