---
title: proxide-confind — parity test merge to main
task_id: 260601_rotlib-contact-tests
session_id: 260601-confind-merge-cleanup
status: complete
date: 260601
branch: main
head: 8e0cf7c
suite: "48 passed, 36 ignored (cargo test -p proxide-confind)"
---

## Session summary

Short cleanup session: restored context from `worktree-confind-parity-tests` handoff (status: complete, all 4 parity tests passing). Fast-forward merged `worktree-confind-parity-tests` → `main` (3 commits). Verified 48 confind tests pass. Cleaned up worktree and branch.

### Commits merged

| SHA | Message |
|-----|---------|
| `8e0cf7c` | fix(confind): three parity bugs vs Mosaist — all small.pdb tests pass |
| `d18c7e3` | test(confind): convert parity tests to real Mosaist assertions |
| `c913f4b` | test(confind): add parity tests for small.pdb against real fixtures |

### Bugs fixed (prior session, now on main)

1. **`parallel.rs` — rotamer-pair dedup in `contact_degree_raw`**: cd accumulated once per atom-atom pair; fixed to accumulate once per unique (rot_a, rot_b) pair via `HashMap<Arc<RotamerId>, HashSet<Arc<RotamerId>>>`. Effect: cd values were 3–23× too high.
2. **`cache.rs` — interference accumulation**: `break 'atom_loop` was inside inner `hits` loop; fixed to run after all atoms are processed. Effect: 3 missing interference pairs (44→47).
3. **`coords.rs` — phi/psi sign convention**: `dihedral_angle_f64` returns opposite sign from Mosaist. Fixed by negating phi/psi in `fill_dihedrals` (sentinel 9999.0 preserved). Effect: wrong backbone bins → wrong rotamer placement.

---

## Deferred work

| Item | Rationale | When |
|------|-----------|------|
| `constrained_contacts` parity tests (Mosaist `--seq_const`) | API exists (`88d6bcc`), no parity test | Next session |
| `1DC7.pdb` GLY/PRO contact parity | GLY (na=0) and PRO (CD atom) contact paths untested against Mosaist | Next session |

---

## Key files for next session

| Path | Lines | Relevance |
|------|-------|-----------|
| `crates/proxide-confind/tests/test_parity_small_pdb.rs` | 1–315 | All 4 parity tests; add `constrained_contacts` test here |
| `crates/proxide-confind/src/parallel.rs` | 1–80 | `contact_degree_raw` + `constrained_contacts` implementation |
| `crates/proxide-confind/tests/common/mod.rs` | — | Shared helpers: `mosaist_path()`, `rotlib_path()`, PDB loading |

---

## Context refs

- Prior session handoff: `.claude/worktrees/confind-parity-tests/.praxia/docs/handoffs/260601_confind-parity-tests.md` (worktree removed; content preserved above in "Bugs fixed")
- Rotlib fixture work: `.praxia/docs/handoffs/260601_rotlib-fixture-expansion.md`
