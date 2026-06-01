---
title: proxide-confind — proxconfind3 session close
task_id: 260601_rotlib-contact-tests
session_id: proxconfind3-260601
status: complete
date: 260601
branch: main
head: ac8776e
suite: "52 passed, 36 ignored (cargo test -p proxide-confind)"
---

## Session summary

Closed both deferred parity items from the proxconfind2 handoff:

1. **`test_seq_const_contacts_small_pdb`** — added to `test_parity_small_pdb.rs`
   - 1330 reference entries from Mosaist `--seq_const` on `small.pdb`
   - Validates `contact_degree_with_clashes(ri_a, ri_b, Some(&[aa]), None)` for every directed pair × constrained AA that Mosaist outputs with CD ≥ 0
   - Corresponds to Mosaist `getConstrainedContacts()` / the `--seq_const` flag

2. **`test_parity_1dc7.rs`** — new test file
   - Loads 1DC7.pdb chain A (124 residues, 7 GLY + 6 PRO positions)
   - `test_crowdedness_parity_1dc7_gly_pro`: 13 reference values at GLY/PRO positions
   - `test_freedom_parity_1dc7_gly_pro`: 13 reference values at GLY/PRO positions
   - `test_contacts_parity_1dc7_gly_pro`: 219 contact pairs involving GLY or PRO

### Common module additions
`crates/proxide-confind/tests/common/mod.rs`:
- `real_pdb_path_1dc7()` — returns path to 1DC7.pdb (env var `DC7_PDB_PATH` or default)
- `load_1dc7_backbone()` — loads 1DC7.pdb backbone, returns None if file not found

### Key facts
- `contact_degree_with_clashes` with `aa_allowed_a = Some(&["ALA"])` corresponds to Mosaist `contactDegree(resi, resj, {ALA}, aaNames)` (position A constrained to single AA, B unconstrained)
- 1DC7.pdb GLY residues: A4, A25, A27, A36, A59, A62, A97; PRO: A48, A58, A74, A77, A103, A105
- 1DC7 tests take ~160s total (3× contacts() pipeline over 124 residues)

---

## Key files

| Path | Lines | Relevance |
|------|-------|-----------|
| `crates/proxide-confind/tests/test_parity_small_pdb.rs` | 316–1693 | REF_SEQ_CONST + test_seq_const_contacts_small_pdb |
| `crates/proxide-confind/tests/test_parity_1dc7.rs` | 1–374 | New file: 3 GLY/PRO parity tests |
| `crates/proxide-confind/tests/common/mod.rs` | 231–248 | 1DC7 path helpers |

---

## Deferred work

None. All items from proxconfind2 handoff are closed.

---

## Context refs

- Prior handoff: `.praxia/docs/handoffs/260601_confind-parity-merge.md` — fixed 3 parity bugs, small.pdb tests pass, main at 8e0cf7c
