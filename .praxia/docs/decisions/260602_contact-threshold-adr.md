---
name: 260602_contact-threshold-adr
description: Architecture decision record for CONTACT_THRESHOLD in proxide-confind — const vs. public arg with cited default
metadata:
  type: decision
  task_id: 260602_rotlib-confind-actions
  status: accepted
---

# ADR: CONTACT_THRESHOLD — pub const with cited default, exposed as filter parameter

**Date:** 2026-06-02
**Status:** Accepted
**Task:** 260602_rotlib-confind-actions

---

## Context

`proxide-confind` computes the contact-degree (CD) metric for all residue pairs in a
protein backbone. The algorithm produces a real-valued CD for each pair `(i, j)`.
Downstream consumers need to select the pairs that are "poised to interact" — i.e. those
with sufficiently high CD — to define TERM neighborhoods for dTERMen sequence optimisation.

The canonical threshold from the Grigoryan lab is:

> **c(i,j) > 0.02** defines positions as "poised to interact."

Source: Zheng & Grigoryan (2017) *PLoS ONE* 12(5): e0178272, equation 9.

This value (0.02) currently appears in two ways in the codebase:

1. As the `cd_cut` parameter to `run_phases_b_c` in `parallel.rs` (line 113), applied
   **at construction time** during Phase B: pairs with `cd <= cd_cut` are never added
   to the `ContactList`.
2. As an implicit magic number chosen by callers who pass `0.0` (no filtering) or `0.02`
   (canonical threshold) without a named constant.

The key architectural question is: should `CONTACT_THRESHOLD` be:

- **(A)** A private or hard-coded constant, forcing all callers through canonical behavior, or
- **(B)** A `pub const` serving as the cited default, plus a `ContactList::filter(threshold)`
  method that accepts any value?

---

## Options Considered

### Option A — Hard const only

Define `CONTACT_THRESHOLD: f64 = 0.02` as an internal constant (not pub) and always
apply it inside ConFind computation. Callers cannot vary the threshold without modifying
library internals.

**Pros:**
- Simple; callers cannot misuse the threshold.
- Forces canonical dTERMen behavior.

**Cons:**
- Callers with different TERM-density requirements (tighter or looser neighborhoods for
  non-dTERMen applications) must fork or patch the library.
- The Grigoryan lab citation is buried inside the library, not visible at the call site.
- Re-running ConFind (the expensive O(N²) Phase B) just to vary TERM tightness is wasteful;
  the threshold is applied at output, not in the inner loop.
- `CLASH_DIST` (2.0 Å) and `CONT_DIST` (3.0 Å) are **not** analogous: changing those
  invalidates the cached proximity grids (the residue cache must be rebuilt). Conflating
  all constants as "not public" would incorrectly treat `CONTACT_THRESHOLD` as having the
  same cache coupling, which it does not.

### Option B — pub const with cited default, parameter to filter()

Define:

```rust
/// Contact-degree threshold defining "poised to interact" residue pairs, per
/// Zheng & Grigoryan (2017) PLoS ONE 12(5): e0178272, eq. 9.
pub const CONTACT_THRESHOLD: f64 = 0.02;
```

Expose a `ContactList::filter(threshold: f64) -> ContactList` method that callers invoke
post-hoc. Callers wanting canonical dTERMen behavior pass `CONTACT_THRESHOLD`; callers
with non-standard requirements pass their own value.

**Pros:**
- The citation is visible at the call site; callers know where `0.02` comes from.
- Downstream consumers (different TERM-density regimes, non-dTERMen uses) can vary the
  threshold without re-running ConFind.
- `CONTACT_THRESHOLD` has no coupling to the residue cache; it is applied once at output
  filtering. Varying it never invalidates cached data.
- `proxide-frag` (the forthcoming TERM construction crate) can import `CONTACT_THRESHOLD`
  from `proxide-confind` and use it as its default, establishing a single source of truth.
- Consistent with how the Grigoryan lab tools expose the threshold: it is a named parameter
  in the MSL CLI, not a sealed constant.

**Cons:**
- Callers could (accidentally or deliberately) pass non-canonical values. This is acceptable
  because: (1) the threshold has no cache implications, so wrong values cannot corrupt
  intermediate state; (2) the API documentation clearly marks the canonical value.

---

## Decision

**Option B is accepted.**

`pub const CONTACT_THRESHOLD: f64 = 0.02` is added to `contact_list.rs` (or re-exported
from `lib.rs`), with the full Zheng & Grigoryan (2017) citation. `ContactList` gains a
`filter(threshold: f64) -> ContactList` method. The `cd_cut` parameter to `run_phases_b_c`
may remain as a construction-time filter for performance (avoiding storing pairs that will
always be discarded), but its default usage in callers should pass `0.0` and rely on
`filter(CONTACT_THRESHOLD)` for semantic clarity, or pass `CONTACT_THRESHOLD` directly.

---

## Rationale

### CLASH_DIST/CONT_DIST are categorically different from CONTACT_THRESHOLD

`CLASH_DIST = 2.0 Å` and `CONT_DIST = 3.0 Å` are **inner-loop constants** used during
Phase A (backbone clash pruning) and Phase B (SC–SC contact accumulation). They determine
which rotamers are retained in the proximity grid (the residue cache). Changing either of
these requires rebuilding the entire residue cache — all cached grids are invalidated.

`CONTACT_THRESHOLD = 0.02` is an **output filter**. It is applied once, after all CD
values have been computed, to select which pairs to include in the `ContactList`. It has
zero coupling to the residue cache: changing it does not invalidate any intermediate
computation, and the same cached ConFind output can be re-filtered with different threshold
values at negligible cost.

Treating all three constants as "sealed implementation details" would be architecturally
incorrect because it conflates two fundamentally different roles: cache-invalidating geometry
parameters vs. output-selection parameters.

### Downstream consumers have legitimate reasons to vary the threshold

Different uses of contact degree in the dTERMen pipeline use different TERM-density
requirements:

- **Tight TERMs** (high-specificity design contexts): callers may prefer `cd > 0.05`
  to include only the most strongly coupled positions.
- **Loose TERMs** (broad sampling): callers may prefer `cd > 0.01`.
- **Non-dTERMen uses** of ConFind (e.g. coiled-coil specificity analysis, protein–protein
  interface mapping): the 0.02 threshold is a dTERMen convention, not a universal law.

The API should express canonical behavior via a named constant, not enforce it via
encapsulation.

### Single source of truth for proxide-frag

`proxide-frag` (the forthcoming TERM construction crate, backlog item B5) will need to
know the canonical contact threshold to define TERM segments from the ConFind contact graph.
If `CONTACT_THRESHOLD` is a `pub const` in `proxide-confind`, `proxide-frag` imports it:

```rust
use proxide_confind::CONTACT_THRESHOLD;

fn build_term(contact_list: &ContactList) -> Vec<Segment> {
    contact_list.filter(CONTACT_THRESHOLD)
        .pairs
        .iter()
        // ...
}
```

This establishes a single source of truth: the Zheng & Grigoryan (2017) citation lives in
one place and both crates reference it without duplication.

---

## Consequences

### Callers wanting canonical dTERMen behavior

Pass `CONTACT_THRESHOLD` to `ContactList::filter`:

```rust
use proxide_confind::{ConFind, CONTACT_THRESHOLD};

let contacts = cf.contacts(&residues, 0.0)?;   // collect all pairs
let term_contacts = contacts.filter(CONTACT_THRESHOLD);  // canonical threshold
```

Or, for performance, pass `CONTACT_THRESHOLD` directly at construction:

```rust
let contacts = cf.contacts(&residues, CONTACT_THRESHOLD)?;
```

Both are valid; the second avoids storing sub-threshold pairs.

### Callers varying TERM density

Pass their own threshold:

```rust
let tight_contacts = contacts.filter(0.05);   // high-specificity regime
let loose_contacts = contacts.filter(0.01);   // broad sampling
```

No recomputation of ConFind is required; the same `ContactList` is re-filtered.

### proxide-frag TERM construction

Imports `CONTACT_THRESHOLD` from `proxide-confind` and uses it as the default threshold
for TERM neighborhood construction. Researchers may override it for non-standard TERM
density experiments.

### Documentation

All uses of `0.02` as a literal in `proxide-confind` and `proxide-frag` tests are
replaced with `CONTACT_THRESHOLD`. The citation (Zheng & Grigoryan 2017, eq. 9) appears
in the const doc comment and is rendered by `cargo doc`.

---

## Citation

Zheng W, Grigoryan G. "Frequency and character of protein–protein interactions observed
in experimentally determined 3D structures: fundamental insights and crystal packing
artifacts." *PLoS ONE.* 2017;12(5):e0178272. https://doi.org/10.1371/journal.pone.0178272

Equation 9 defines the contact-degree formula; the supplementary methods specify
`c(i,j) > 0.02` as the threshold for positions "poised to interact" in dTERMen TERM
neighborhood construction.

---

## Related Decisions

- synthesis:B1 — canonical source for this decision
- synthesis:B6 — RotlibRegistry closed; single-library architecture confirmed
- Plan: `.praxia/docs/plans/260602_rotlib-confind-actions.md`, actions B1, B5
- Future crate: `proxide-frag` (backlog item B5) — principal downstream consumer
