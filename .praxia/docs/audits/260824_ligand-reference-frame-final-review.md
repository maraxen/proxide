---
title: "ligand-reference-frame final whole-branch review"
description: "Final review verdict and disposition for the 10-task proxide-ligand-frame implementation plan (branch feat/ligand-reference-frame-spec)"
category: audits
task_id: 260824_ligand_ref_frame_sdd
status: resolved
---

# ligand-reference-frame final whole-branch review

## Context

The `proxide-ligand-frame` crate (canonical ligand reference frame for
demistify's ligand-extension gate, idea-002) was implemented via
`superpowers:subagent-driven-development` across 10 tasks, each with its
own scoped implementer + reviewer (Task 4 and Task 8 each required one
fix round for task-local findings — see
`.superpowers/sdd/mutable-drifting-seahorse/progress.md`, deleted after
this plan closed; commit history on `feat/ligand-reference-frame-spec`
is the durable record of what changed).

After all 10 tasks passed their own reviews, a final whole-branch review
(dispatched on the most capable model, per the SDD skill's requirement
for this gate) read the branch as a whole rather than task-by-task, and
found 3 Important, cross-task-visible-only defects that no single task's
narrower review could have caught.

## Findings fixed (commit `5fcca25`, one fix round + one scoped re-review, both confirmed addressed independently)

1. **Explicit-hydrogen contract violated** — `build_ligand_reference_frame`
   didn't validate that the caller-supplied `Molecule` has explicit
   hydrogens before calling gaff2 typing, which requires them. Real
   benzene (H-free test fixture) was silently mistyped `c1` (sp1) instead
   of `ca` (aromatic) — no test asserted `gaff2_types`, so this passed
   green. Fixed via `_assert_explicit_hydrogens` (a separate,
   non-`NoImplicit` RDKit mol used only for the atom-count check; the
   real typing mol is never mutated, so no desync with
   `trajectory_positions` is introduced). Test fixture reworked to an
   explicit-H benzene; added a rejection test for the old H-free fixture.

2. **`bond_in_ring` used any-ring membership, not same-ring** —
   contradicted its own doc comment ("common SSSR ring"), silently
   dropping real inter-ring torsion DOF for any bridged bicyclic ligand
   (reproduced: bicyclohexyl gave 0 torsion_definitions, expected 1).
   Fixed via a genuine per-ring `.any(|ring| ring.contains(&i) &&
   ring.contains(&j))` check.

3. **Reachable `panic!` across the FFI boundary** on a malformed bond
   order (e.g. TRIPOS `"nc"` → order 0) — crashed the whole Python
   process instead of raising a catchable exception, unlike the sibling
   `bond_order_from_u8` helper 200 lines away in the same PyO3 crate.
   Fixed via a new `LigandFrameError::InvalidBondOrder` variant.

## Findings deliberately NOT fixed in this round (tracked debt, not silently dropped)

Explicitly out of scope for the final-review fix round (which was scoped
to the 3 Important findings only, per SDD's "ONE fix dispatch" rule).
None are blocking; listed here so they aren't lost once the SDD scratch
workspace is deleted.

- **`ring_traversal_order`'s second `.expect()`** (`pucker.rs`, "no
  unvisited ring neighbor found") — not reachable through the real
  `build_ligand_reference_frame` path (RDKit's ring perception and bond
  graph are mutually consistent by construction), but IS reachable
  through the raw `canonicalize_ligand_topology` PyO3 binding if a caller
  passes an in-bounds-but-not-actually-a-cycle `rings` entry —
  `proxide_gaff2::mol::MolGraph::new`'s ring validation only checks index
  bounds, not that a ring's atoms form a real cycle in `bonds_in`. Same
  defect class as the fixed Finding 3, narrower/lower-likelihood. Worth a
  small follow-up: either validate ring-is-a-cycle at construction, or
  convert this `.expect()` to a typed error.
- **Dead `charge_cache_key`/`compute_graph_fingerprint`/
  `compute_ref_frame_geometry_hash`/`espaloma_weights_version` code**
  (`charges.rs`) — implements the spec §3 cache-key design correctly
  (own unit tests pass) but nothing calls it; no cache store exists
  anywhere in the crate. This matches the plan's own explicitly-scoped
  "Scope note: partial-charge caching" (key design only, no store), but
  should be stated as a scope note in the crate's own docs rather than
  left implicit as apparently-dead code to a future reader.
- **`frame_validity`/torsion-loop ordering inconsistency** (`frame.rs`) —
  `frame_validity[f] == false` does not imply every torsion at frame `f`
  is NaN (a later torsion in the iteration order can retain a finite
  value written before an earlier torsion's collinearity check
  invalidated the frame). Spec §2b already treats `frame_validity` as
  authoritative and torsions as "may still be NaN", so this isn't a spec
  violation, but no test currently exercises a multi-torsion topology to
  confirm downstream behavior is safe under this asymmetry.
- **`bond_angles` has no documented/exposed index space** — a consumer
  gets `(n_angles, n_frames)` with no way to know which `(neighbor,
  center, neighbor)` triple each row corresponds to. Neither
  `LigandTopology` nor `LigandFrameCoordinates` exposes the triple list.
- **Empty-array shape inconsistency at the Python boundary** — a
  zero-torsion ligand's `raw["torsions"]` comes back as numpy shape
  `(0,)`, not `(0, n_frames)` as spec §2b implies for the 2D contract.
  Same for `pucker_phase` with zero rings.
- **Redundant/divergible aromaticity input** at the PyO3 boundary —
  `canonicalize_ligand_topology`'s `bonds: Vec<(usize,usize,u8,bool)>`
  4th field (aromaticity) is silently discarded in favor of the separate
  `bond_is_aromatic` parameter; the two can disagree with no error.
- **`feature_mask` is unconditionally all-`true`** (`frame.rs`) — a v1
  placeholder carrying no real information yet; worth a code comment
  saying so explicitly.
- **License packaging asymmetry** — `proxide-ligand-frame` correctly sets
  `license = "GPL-2.0-or-later"` in `Cargo.toml` (no MIT-workspace-default
  leakage), but unlike `proxide-gaff2` it ships no `LICENSE`/`NOTICE`
  file (mitigated by `publish = false`, but worth parity with the sibling
  crate).
- **Unused `approx` dev-dependency** in `proxide-ligand-frame/Cargo.toml`
  (zero `approx::` references anywhere in the crate).
- **`canon.rs`'s `atomic_number` silently returns 0** for any element
  outside its 10-entry table, collapsing every exotic element into one
  canonical-rank class — unreachable today only because
  `geometry_gate::covalent_radius` rejects the same element set earlier
  in `canonicalize_ligand_topology`'s call order, which is load-bearing
  and currently undocumented as such.
- **Misleading test name** in `proxide-geometry`:
  `test_bond_angle_f64_matches_f32_precision` never actually compares
  against the f32 `bond_angle` — it only checks a known angle to 1e-9.
- **No test exercises a real per-frame torsion value surviving the full
  Python→Rust→Python round trip** — the only Python integration test
  (benzene) has zero torsions by construction; every Rust-side torsion
  test uses synthetic hand-built `LigandTopology` fixtures, never one
  produced by the real `canonicalize_ligand_topology`. Worth a follow-up
  integration test using a molecule with at least one real rotatable
  bond (e.g. the dimethylbutane-shaped fixture already used in
  `typing.rs`'s own unit tests) run through the full wrapper.

## Verdict

All 3 Important findings confirmed fixed and independently re-verified
(both the fix-round implementer and a separate scoped re-reviewer ran
real reproduction probes, not just read the diff). No new issues
introduced. Branch is clear to proceed to
`superpowers:finishing-a-development-branch`.
