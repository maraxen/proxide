---
name: 260602_rotlib-confind-actions
description: Implementation plan for post-NLM-synthesis code and doc actions in proxide-rotlib and proxide-confind
metadata:
  type: plan
  task_id: 260602_rotlib-confind-actions
  status: draft
---

# Post-NLM-Synthesis Action Plan: proxide-rotlib and proxide-confind

This plan distills the 12 synthesis decisions (ids A1–A6, B1–B6) from `.praxia/synthesis.jsonl`
into concrete, assigned actions. No code is implemented here; each action is advisory/spec only,
ready for fixer dispatch.

---

## Section 1 — proxide-rotlib Code and Doc Actions

### A1 — API-doc warning for φ≥−30° sparse region

**What.** In `crates/proxide-rotlib/src/rotlib.rs`, the `backbone_bin` method (lines 260–272)
contains no documentation about statistical confidence degradation at high positive phi.
Add a `/// # Warning` section to the `backbone_bin` doc comment directly above the `pub fn
backbone_bin` signature (line 260):

```
/// # Warning: sparse φ region
///
/// Bins with φ ≥ −30° have fewer than 3 crystallographic observations in
/// the Dunbrack BBdep library. The nearest-neighbour mapping is correct,
/// but the resulting rotamer probability is statistically unreliable.
/// Placement in this region should be treated as a low-confidence estimate.
```

**Why.** synthesis:A1 — Dunbrack BBdep library uses 20°×20° bins at 10° centers; sparse region
begins at φ ≥ −40°, very sparse at φ ≥ −30°, often only 1–2 observations per bin.
`find_closest_angle` nearest-neighbour is sufficient for v1 but the reliability degradation
must be documented at the call site. Source: Dunbrack & Cohen (1997) *Protein Sci* 6(8):1661–1681;
Dunbrack (2002) *Curr Opin Struct Biol* 12:431–440.

**Success criteria.**
- `cargo doc --no-deps -p proxide-rotlib` renders a "Warning: sparse φ region" section
  under `backbone_bin`.
- The warning is present in `rustdoc` HTML output for `RotamerLibrary::backbone_bin`.
- A new integration test exists that calls `backbone_bin("LEU", 20.0, -40.0)` and asserts
  it returns `Ok` (test documents the sparse-region path without asserting on the specific
  bin value, since the bin is library-data-dependent).

---

### A2 — Debt #67 spec: cis-PRO branch for backbone_bin

**What.** `backbone_bin` (rotlib.rs line 260) currently accepts `phi` and `psi` with no
amino-acid-specific branching. Debt #67 elevates cis-proline handling from cosmetic to
required. Write a concrete spec and commit it as a **new file**
`.praxia/docs/specs/260529_rotlib.md` (the `specs/` subdirectory does not yet exist in
the worktree; create it). The spec doc must cover:

1. **New parameter:** `backbone_bin(aa: &str, phi: f64, psi: f64, cis_proline: bool)`.
   `cis_proline` is only meaningful when `aa == "PRO"`; it is ignored (with no cost) for
   all other amino acids. Default for callers that do not detect ω should be `false`.

2. **Library lookup:** When `aa == "PRO" && cis_proline == true`, look up a separate
   cis-PRO bin set. The MSL BBdep binary must contain a distinct key for cis-PRO (likely
   `"CPRO"` or a library-specific tag — confirm by inspecting the binary header in the
   rotlib fixture). When `cis_proline == false` (default), look up standard `"PRO"`.

3. **Backbone parser extension** (`backbone_parser` or `coords.rs` in proxide-confind):
   - Extract ω dihedral from `C(i-1)–N(i)–CA(i)–C(i)` for each residue.
   - Flag cis if `|ω| < 30°`. Store as `is_cis_peptide: bool` in `ResidueBackbone`.
   - Pass to `backbone_bin` at placement time.

4. **API surface sketch:**
   ```rust
   // rotlib.rs (signature only, no implementation here)
   pub fn backbone_bin(
       &self,
       aa: &str,
       phi: f64,
       psi: f64,
       cis_proline: bool,   // <-- new; ignored unless aa == "PRO"
   ) -> Result<u32, RotlibError>

   // coords.rs / ResidueBackbone (field addition only)
   pub struct ResidueBackbone {
       pub phi: f64,
       pub psi: f64,
       pub omega: Option<f64>,      // <-- new; None for N-terminal residue
       pub is_cis_peptide: bool,    // <-- new; true iff |omega| < 30°
       // ... existing fields ...
   }
   ```

5. **Cγ-endo/Cγ-exo ring-pucker split:** the Dunbrack library separates cis-PRO at ψ=65°
   into Cγ-endo (ψ < 65°) and Cγ-exo (ψ ≥ 65°) sub-populations. This is a v2 refinement;
   the v1 cis-PRO branch need only select the correct top-level library bin.

**Why.** synthesis:A2 — Cis-PRO occupies ψ=20°–100° exclusively, rarely populated by
trans-PRO. The −75°<φ<−50° band has kinetic equilibrium between both forms. Single-form
approximation blends populations and produces incorrect Cγ-endo/exo ring puckers in
cis-PRO-specific ψ regions. This is materially important for design accuracy, not cosmetic.
Source: Dunbrack & Karplus (1994) *J Mol Biol* 230:543–574; Richardson et al. (2003)
*Biochemistry* 42:8603.

**Success criteria.**
- `.praxia/docs/specs/` directory created (if absent) and `.praxia/docs/specs/260529_rotlib.md`
  created as a **new file** containing the cis-PRO section. (This file does not pre-exist
  in the worktree; the fixer must create it, not update it.)
- `.praxia/docs/INDEX.md` updated to list the new spec under the `## Specs` section.
- Debt #67 tracking record updated to `status: planned`.
- No code changes in this action; spec only.

---

### A3 — API docs for GLY na=0 and ALA CB semantics

**What.** Add explicit documentation to two locations in proxide-rotlib:

1. `rotlib.rs` — `contains_aa`, `num_rotamers`, or `sidechain_atom_names` doc comment:

```
/// # GLY and ALA edge cases
///
/// **GLY:** not present in the rotamer library (no χ angles, `na=0`).
/// `contains_aa("GLY")` returns `false`; `place_rotamer("GLY", …)` returns
/// `Err(UnknownAa)`. Callers should check `contains_aa` before placement.
///
/// **ALA:** CB is treated as a sidechain atom in ConFind clash detection
/// (`counts_as_sidechain("CB", "ALA") == true`).
/// For all other amino acids, CB is excluded (`doNotCountCB` semantics).
/// `place_rotamer("ALA", …)` returns a single-atom list containing CB.
```

2. `sidechain.rs` — `counts_as_sidechain` doc comment: add a cross-reference to
   the ALA-CB note in `rotlib.rs`.

**Why.** synthesis:A3 — GLY has no χ1 angle and is excluded from all rotamer populations
(na=0). ALA's CB is the only atom placed as a sidechain atom and is explicitly counted in
ConFind clash detection. Both are correct per spec; documentation prevents future callers
from misinterpreting the behavior. Source: Dunbrack (2002); confirmed in MSL source
`mstrotlib.cpp` (`doNotCountCB` logic).

**Success criteria.**
- `cargo doc` renders the GLY and ALA edge-case notes on at least one public method.
- The `counts_as_sidechain` doc comment cross-references ALA CB behavior.
- No code change; doc only.

---

### A4 — Safety doc for sentinel check and divergence test for LEU

**What.** Two distinct changes in `rotlib.rs`:

1. **`// Safety` comment** at the sentinel check (lines 265–267):
   ```rust
   // Safety: φ=9999.0 or ψ=9999.0 is a sentinel for "no backbone available".
   // Returns default_bin (global mode, argmax over all bin frequencies).
   // Must NOT be called with these sentinel values when real backbone angles are known:
   // default_bin ignores backbone-specific steric constraints (syn-pentane interactions
   // with Ci-1, Ni+1, Oi) and will produce potentially clashing placements.
   if phi == 9999.0 || psi == 9999.0 {
       return Ok(entry.default_bin);
   }
   ```

2. **New test** in the `#[cfg(test)]` block (after line 411):
   ```rust
   #[test]
   fn test_backbone_bin_diverges_from_default_for_leu() {
       // For a real library, backbone_bin at a valid helix backbone must differ from
       // default_bin (which is the global maximum, not the helix-specific maximum).
       // This test is integration-only and requires a real rotlib binary.
       // Add to the integration test suite (tests/integration.rs) rather than here.
       // See ROTLIB_FIXTURE_DIR wiring in A6.
   }
   ```
   The actual test — asserting `backbone_bin("LEU", -60.0, -40.0) != default_bin` and that
   the result varies across φ/ψ — should live in the integration test suite gated on the
   fixture env var, not in `tests` within `rotlib.rs`. The inline test above is a placeholder
   indicating the contract.

**Why.** synthesis:A4 — `default_bin` is argmax over all bin frequencies globally, not a
backbone-specific fallback. Using it with valid backbone angles ignores syn-pentane
interactions and produces incorrect, potentially clashing placements. The sentinel path
(φ=9999.0 || ψ=9999.0) is correct for callers that lack backbone data, but must be
documented as a deliberate fallback, not a general default.

**Success criteria.**
- The sentinel check block in `backbone_bin` has a `// Safety` doc comment explaining
  the contract.
- An integration test asserts that `backbone_bin("LEU", -60.0, -40.0)` returns a value
  different from `default_bin` (requires real library binary; gated on env var from A6).
- `cargo test --test integration` passes with the new test when fixture is present.

---

### A5 — PlacedRotamer chi-angle extension note

**What.** This action is advisory only — no code change now. When `PlacedRotamer`
(rotamer_id.rs, line 11) gains a `chi_angles: Vec<f64>` field, add the following
doc comment to the struct:

```rust
/// # Chi-angle conventions and symmetry
///
/// χ1 = N–Cα–Cβ–Xγ dihedral. Three canonical wells: g+ (~+60°), t (~180°), g- (~−60°).
///
/// **Non-rotameric terminal χ (semirotameric AAs):** PHE and TYR χ2 is non-rotameric;
/// it clusters near ±90° due to sp2 Cδ atoms. Do not apply g+/t/g- labels to PHE/TYR χ2.
///
/// **Carboxylate degeneracy:** ASP and GLU terminal χ has 180° symmetry
/// (χ2=−60° ≡ +120° for ASP). Normalize before comparison or use unsigned angle distance.
///
/// **Amide/imidazole misorientation:** ASN, GLN, and HIS terminal groups are frequently
/// modeled ±180° wrong in raw PDB data; normalize using H-bond network or MolProbity
/// Reduce before comparing to library values.
///
/// Raw library dihedral values are emitted as-is; normalization to canonical wells
/// is a v2 concern.
```

**Why.** synthesis:A5 — Dunbrack 2010 treats 8 semirotameric AAs (Asp, Asn, Gln, Glu, His,
Phe, Tyr, Trp) with continuous terminal χ distributions. PHE/TYR χ2 is genuinely
non-rotameric (sp2 terminal group). ASN/GLN/HIS misorientation is a known PDB data issue.
Documenting these at the PlacedRotamer level prevents downstream consumers from applying
wrong symmetry corrections. Source: Dunbrack (2010) *Structure* 18:1456–1467;
Bhagavan & Ha-eun Oh (2013) *Biochemistry, 5th ed.*

**Success criteria.**
- When `chi_angles: Vec<f64>` is added to `PlacedRotamer`, the doc comment above is
  included in the same commit.
- `cargo doc` renders the chi-angle conventions section.
- No code change in this action; doc placeholder only.

---

### A6 — Fixture expansion plan

**What.** Expand the integration test fixture set for proxide-rotlib (and secondarily confind).
Actions:

1. **Inspect current fixtures.** Run:
   ```
   grep -c 'PRO\|GLY' /home/marielle/repos/mosaist/testfiles/fuserinput.pdb
   ```
   If fuserinput.pdb contains both PRO and GLY, it is immediately usable. If not, proceed
   to step 2.

2. **Add 2ZTA** (leucine zipper, helical). Trim to ≤50 residues via a reproducible script:
   ```
   scripts/fixtures/trim_pdb.py --pdb 2ZTA --chain A --nres 50 \
       --out crates/proxide-rotlib/tests/fixtures/2ZTA_trim50.pdb
   ```
   2ZTA is helical backbone with high likelihood of PRO occurrences.

3. **Add 1DC7** (broad AA/secondary structure, 124 residues, all 20 AAs confirmed in
   confind integration tests). Copy to:
   ```
   crates/proxide-rotlib/tests/fixtures/1DC7.pdb
   ```

4. **Fetch 1VII or 1L2Y** from RCSB for turn/loop coverage (positive-φ backbone, SER/ASN
   placement challenge). Script:
   ```
   scripts/fixtures/fetch_pdb.py --pdb 1VII \
       --out crates/proxide-rotlib/tests/fixtures/1VII.pdb
   ```
   Fetch locally (all external data must be pre-fetched; compute nodes have no internet).

5. **Wire via env var.** Integration tests gated on `ROTLIB_FIXTURE_DIR`:
   ```rust
   // in tests/integration.rs
   fn fixture_dir() -> Option<PathBuf> {
       std::env::var("ROTLIB_FIXTURE_DIR").ok().map(PathBuf::from)
   }
   ```
   CI `.github/workflows/rotlib.yml` sets `ROTLIB_FIXTURE_DIR: ${{ github.workspace }}/...`.

**Why.** synthesis:A6 — SER has only 57% rotamer accuracy even with BBdep; CYS, THR, ASN
are also challenging. Loop/turn traversal hits sparse φ/ψ space (A1 warning applies here).
1DC7 and 2ZTA are the highest-priority additions; 1VII/1L2Y add the loop coverage not
present in helical fixtures. Source: Dunbrack (2002); DePristo et al. (2003)
*Structure* 11:981–990.

**Success criteria.**
- Fixture directory `crates/proxide-rotlib/tests/fixtures/` contains at least:
  `fuserinput.pdb` (existing), `2ZTA_trim50.pdb`, `1DC7.pdb`, `1VII.pdb` (or `1L2Y.pdb`).
- Integration tests parameterised over all fixtures pass when `ROTLIB_FIXTURE_DIR` is set.
- CI workflow file updated to export `ROTLIB_FIXTURE_DIR` and run integration tests.

---

## Section 2 — proxide-confind Code and Doc Actions

### B1 — CONTACT_THRESHOLD pub const

**What.** Add to `crates/proxide-confind/src/contact_list.rs` (or `params.rs`):

```rust
/// Contact-degree threshold defining "poised to interact" residue pairs.
///
/// Residue pairs with `cd > CONTACT_THRESHOLD` are included in TERM
/// neighbourhoods for dTERMen sequence optimisation. This is the canonical
/// published value from Zheng & Grigoryan (2017) PLoS ONE 12(5): e0178272, eq. 9.
///
/// # Design decision
///
/// `CONTACT_THRESHOLD` is applied once at output filtering and has no
/// coupling to the residue cache (unlike `CLASH_DIST`/`CONT_DIST` which
/// invalidate cached sidechain grids if changed). Downstream consumers
/// with different TERM-density requirements (or non-dTERMen uses of
/// contact degree) may pass a custom threshold to `ContactList::filter()`.
/// For canonical dTERMen behaviour, pass `CONTACT_THRESHOLD`.
///
/// See ADR: `.praxia/docs/decisions/260602_contact-threshold-adr.md`
pub const CONTACT_THRESHOLD: f64 = 0.02;
```

Expose through `pub use` in `lib.rs`. Expose as a parameter in a new `ContactList::filter`
method:

```rust
impl ContactList {
    /// Filter to pairs with `cd > threshold`. Pass `CONTACT_THRESHOLD` for
    /// canonical dTERMen behaviour.
    pub fn filter(&self, threshold: f64) -> ContactList {
        let mut out = ContactList::default();
        for (i, &(ri, rj)) in self.pairs.iter().enumerate() {
            if self.degrees[i] > threshold {
                out.pairs.push((ri, rj));
                out.degrees.push(self.degrees[i]);
            }
        }
        out
    }
}
```

**Why.** synthesis:B1 — CD threshold 0.02 defines "poised to interact" in dTERMen;
the current `cd_cut` parameter in `run_phases_b_c` (parallel.rs line 113) applies it at
construction time, coupling it to the computation. Exposing it as a named const enables
downstream callers (proxide-master and others) to vary TERM tightness without re-running
ConFind. See also ADR: `260602_contact-threshold-adr.md`.

**Success criteria.**
- `pub const CONTACT_THRESHOLD: f64 = 0.02` present in `contact_list.rs` or `params.rs`
  and re-exported from `lib.rs`.
- `ContactList::filter(threshold: f64)` method added.
- `cargo doc` renders the Zheng & Grigoryan (2017) citation and ADR cross-reference.
- Existing tests updated to use `CONTACT_THRESHOLD` where they previously used a literal `0.02`.

---

### B2 — LO_COLL_PROB_CUT and HI_COLL_PROB_CUT as named pub consts

**What.** In `crates/proxide-confind/src/params.rs`, the constants `LO_COLL_PROB` and
`HI_COLL_PROB` already exist as pub consts (lines 15, 19) but their names do not match the
synthesis record's preferred naming `LO_COLL_PROB_CUT` / `HI_COLL_PROB_CUT`. Additionally,
the existing doc comments do not state their empirical origin.

Actions:
1. If renaming is safe (no downstream external crate uses the old names yet), rename to
   `LO_COLL_PROB_CUT` and `HI_COLL_PROB_CUT` with deprecation aliases.
2. Update doc comments:
   ```rust
   /// Lower bound on the collision-probability weight used in freedom computation.
   ///
   /// Empirically tuned in the Grigoryan lab MSL ConFind implementation.
   /// **Not calibrated against experimental flexibility measures (B-factors,
   /// NMR order parameters S²).** A --freeB mode is deferred to v2.
   pub const LO_COLL_PROB_CUT: f64 = 0.5;

   /// Upper bound on the collision-probability weight (applied at or below `CLASH_DIST`).
   ///
   /// Empirically tuned; see `LO_COLL_PROB_CUT` note.
   pub const HI_COLL_PROB_CUT: f64 = 2.0;
   ```
3. Update all internal callers (`parallel.rs` `run_phases_b_c` parameter `lo_cut`/`hi_cut`
   call sites) to use the new const names.

**Why.** synthesis:B2 — `loCollProbCut=0.5` and `hiCollProbCut=2.0` are empirically tuned
geometric constants from MSL, not calibrated against B-factors or NMR order parameters.
`--freeB` mode is explicitly deferred to v2. Magic numbers must be replaced with named
constants with documented empirical origin to prevent future "calibration" attempts that
misinterpret them as physics-grounded thresholds. Source: Grigoryan & Keating (2008)
*Curr Opin Struct Biol* 18:477–483; Zheng & Grigoryan (2017).

**Success criteria.**
- `LO_COLL_PROB_CUT` and `HI_COLL_PROB_CUT` (or the existing names with updated doc
  comments) are `pub const` with the empirical-origin note.
- No magic numbers `0.5` or `2.0` in `parallel.rs` freedom computation; all uses reference
  the named consts.
- `cargo doc` renders the empirical-origin note and v2 deferral.

---

### B3 — Doc comment in cache_residue_impl: Phase A non-skippable

**What.** Add a doc comment block to `cache_residue_impl` in `cache.rs` (above line 69)
explaining why Phase A backbone pruning is non-skippable:

```rust
/// Cache one residue: place all rotamers, prune backbone clashes, accumulate interference.
///
/// # Why Phase A pruning is non-skippable
///
/// Backbone-clashing rotamers are excluded from the surviving set, which is the
/// denominator of the contact-degree (CD) formula:
///
/// ```text
/// CD(i,j) = Σ aaProp_a * rotP_a * aaProp_b * rotP_b  /  W_a * W_b
/// ```
///
/// where `W_a = Σ aaProp * rotP` over **surviving** rotamers of residue i.
///
/// If backbone-clashing rotamers were retained (Phase A skipped):
/// 1. `W_a` would be inflated — physically impossible states would dilute all
///    CD values, making every position appear more exposed than it is.
/// 2. False sidechain–sidechain contacts would arise from clashing rotamers that
///    can never coexist with the backbone.
///
/// `CLASH_DIST = 2.0 Å` is a hard vdW exclusion threshold, not a tunable approximation.
/// `fraction_pruned` (crowdedness) is a free byproduct with zero additional compute cost.
///
/// Thread-safe: reads from shared grids; writes only into `cache_out` and `interf_out`
/// which are per-residue slots.
```

**Why.** synthesis:B3 — CLASH_DIST=2.0 Å is physically grounded in vdW repulsion radii.
The crowdedness metric is a byproduct of Phase A. The non-skippability constraint is not
obvious from reading the code; without the comment, a future implementer optimising for
speed might attempt to skip Phase A, breaking CD normalization. Source: Zheng & Grigoryan
(2017); Grigoryan & Keating (2008).

**Success criteria.**
- `cache_residue_impl` has the non-skippability doc comment (or a version equivalent to it).
- `cargo doc` renders the formula and the two consequences of skipping.
- No code change beyond the doc comment.

---

### B4 — Document MSL C++ bug and add fast-path replication stub for seq_const

**What.** The `contact_degree_raw` function in `parallel.rs` (lines 28–end) currently
handles `aa_allowed_a`/`aa_allowed_b` purely via `None`/`Some` expansion into a `HashSet`
(lines 37–44). There is no separate fast-path block analogous to the C++ bug at
`contactDegree.cpp` line 209 (`aaAllowedA.empty() && aaAllowedA.empty()`).

This action has two parts:

**Part 1 — Add a doc comment to `contact_degree_raw` noting the future parity obligation.**
Add the following comment directly above the `aa_set_a`/`aa_set_b` expansion block
(before `let aa_set_a: HashSet<&str> = match aa_allowed_a {`, approximately line 37):

```rust
// MSL C++ bug note (contactDegree.cpp line 209):
// The C++ implementation checks `aaAllowedA.empty() && aaAllowedA.empty()` as a
// fast-path for the unconstrained (no restriction) CD case — checking aaAllowedA
// twice instead of aaAllowedA && aaAllowedB.
//
// In the current Rust port, seq_const is not yet implemented: aa_allowed_a and
// aa_allowed_b are always None (see run_phases_b_c line 149), so no fast-path
// branch is needed here. When seq_const is implemented in v2, a fast-path of the form:
//
//   if aa_allowed_a.is_none() && aa_allowed_a.is_none() { /* fast path */ }
//                                ^^^^^^^^^^^^^^^^^^
//   (intentional: both checks are 'a' — replicates C++ bug for numerical parity)
//
// must be considered. At that point, either replicate the bug exactly for full
// parity, or intentionally diverge and document as a parity exception in the
// parity test. See backlog: seq_const v2.
```

**Part 2 — Add a `// TODO(seq_const v2)` cross-reference comment** at the call site in
`run_phases_b_c` where `contact_degree_raw` is called with `None, None` (line 149):

```rust
// TODO(seq_const v2): pass aa_allowed_a/aa_allowed_b here for constrained CD.
// When implemented, see MSL C++ parity note in contact_degree_raw.
contact_degree_raw(ri, rj, &ca, &cb, rotlib, None, None)
```

Additionally, open a backlog item (see Section 3) for seq_const v2 implementation, noting
that the C++ bug must be either replicated exactly or intentionally diverged-and-documented.

**Why.** synthesis:B4 — The bug affects the no-restriction fast path, not the constrained
path itself. The current Rust port calls `contact_degree_raw` with `None, None` always
(unconstrained), so the bug's fast-path simply does not exist yet as executable code. The
parity obligation must be documented at the function site (Part 1) and at the call site
(Part 2) so a future seq_const implementer cannot miss it. Source: MSL contactDegree.cpp;
Zheng & Grigoryan (2017).

**Success criteria.**
- `contact_degree_raw` in `parallel.rs` has the MSL C++ bug note comment above the
  `aa_set_a`/`aa_set_b` expansion block.
- The `run_phases_b_c` call site for `contact_degree_raw(ri, rj, &ca, &cb, rotlib, None, None)`
  has a `// TODO(seq_const v2)` cross-reference comment.
- No functional code change in this action; comments only.
- A backlog item for seq_const v2 is registered.

---

### B5 — Backlog item: proxide-master crate

**What.** Register a new backlog item in the project backlog system:

```
[P2] proxide-master — backbone RMSD substructure search over a curated PDB fragment library

Category: research
Difficulty: extended
Depends on: proxide-confind (TERM segment set definition)

Description:
  MASTER-style backbone substructure search. Given a TERM (a set of residue
  segments defined by the ConFind contact graph, i.e. {j : cd(i,j) > 0.02}),
  search a pre-built PDB fragment library for structural matches by backbone RMSD.

  Interface: TERM → Vec<PdbMatch { pdb_id, chain, rmsd, sequence }>

  Layer 3 of the proxide pipeline (after Layer 2 ConFind). Feeds Layer 4
  dTERMen sequence optimisation from PDB statistics.

  Note: NOT a Rosetta packer clone. dTERMen's higher-order sub-TERMs capture
  multi-body coupling that pairwise-additive models miss. This is an entirely
  different optimisation paradigm.
```

**Why.** synthesis:B5 — Rosetta packer uses physics-based pairwise-decomposable energies.
dTERMen uses ConFind contact degree to define TERM neighborhoods, then MASTER for backbone
RMSD matching, then sequence statistics from matched fragments. The next crate after
proxide-confind is proxide-master, not a packer. Source: Zheng & Grigoryan (2017) for
dTERMen; Zhou et al. (2015) *Structure* 23:2376 for MASTER.

**Success criteria.**
- Backlog item registered and visible via `praxia backlog` or equivalent.
- Item notes dependency on proxide-confind and the CONTACT_THRESHOLD interface (B1).
- No implementation work in this action; registration only.

---

### B6 — RotlibRegistry architecture: close and document the decision

**What.** The RotlibRegistry abstraction is definitively deferred. Add a note to the
Debt #67 spec doc created in A2 (`.praxia/docs/specs/260529_rotlib.md`, which A2 creates
as a new file) stating:

```
## RotlibRegistry architecture — closed

Architecture decision: single `RotamerLibrary::load()` (BBdep) plus optional
`RotamerLibrary::load_independent()` (BBind, v2). RotlibRegistry deferred indefinitely.

Rationale from literature: modern tools (Rosetta, MSL) use one high-quality BBdep
library (Dunbrack 2010/2011), not a registry of swappable libraries. Resolution
adjustment is done at runtime via "extra rotamers" (-ex1, -ex2 flag expansion of
rotamer well centers), not by library swapping. The Dunbrack 2002→2010 upgrade was
a one-time historical event, not a recurring operation.

Extra-rotamer expansion is a post-load rotamer augmentation step (v3 feature),
not a second library. No RotlibRegistry needed for v1 or v2.

See: ADR 260602_contact-threshold-adr.md for related architecture decisions.
```

**Why.** synthesis:B6 — RotlibRegistry is not supported by literature precedent.
Dunbrack 2010 adds semirotameric treatment for 8 AAs with continuous terminal χ
distributions, but this is handled within a single library, not via multiple swappable
libraries. BBdep vs. BBind selection is path-based in MSL (directory→BBdep, file→BBind).
Source: Dunbrack (2010) *Structure* 18:1456–1467; Shapovalov & Dunbrack (2011)
*Structure* 19:844–858.

**Success criteria.**
- Debt #67 spec doc contains a "RotlibRegistry architecture — closed" section.
- No open design tickets for RotlibRegistry after this action.
- ADR cross-reference is present.

---

## Section 3 — Backlog Items to Register

Two new backlog items are produced by this synthesis:

### Backlog Item 1: proxide-master crate

| Field       | Value                                           |
|-------------|------------------------------------------------|
| Priority    | P2                                              |
| Category    | research                                        |
| Difficulty  | extended                                        |
| Depends on  | proxide-confind (TERM segment set + B1 const)   |
| Source      | synthesis:B5                                    |
| Summary     | Backbone RMSD substructure search (MASTER-style) over a curated PDB fragment library; Layer 3 of the proxide pipeline |

### Backlog Item 2: Debt #67 spec update (cis-PRO branch)

| Field       | Value                                           |
|-------------|------------------------------------------------|
| Priority    | P1                                              |
| Category    | debt                                            |
| Difficulty  | standard                                        |
| Depends on  | A2 spec doc (this plan)                         |
| Source      | synthesis:A2, synthesis:B6                      |
| Summary     | Elevate Debt #67 to planned: backbone_bin cis_proline param, omega extraction in backbone parser, cis-PRO library bin set, RotlibRegistry closure |

---

## Action Summary Table

| ID | Crate           | File(s)                            | Type       | Priority |
|----|-----------------|-------------------------------------|------------|----------|
| A1 | proxide-rotlib  | rotlib.rs:260                       | doc        | P1       |
| A2 | proxide-rotlib  | rotlib.rs, coords.rs, specs/ (new)  | spec       | P1       |
| A3 | proxide-rotlib  | rotlib.rs, sidechain.rs             | doc        | P2       |
| A4 | proxide-rotlib  | rotlib.rs:265, tests/integration.rs | doc + test | P1       |
| A5 | proxide-rotlib  | rotamer_id.rs (future)              | doc note   | P3       |
| A6 | proxide-rotlib  | tests/fixtures/, .github/           | infra      | P2       |
| B1 | proxide-confind | contact_list.rs, params.rs, lib.rs  | code + doc | P1       |
| B2 | proxide-confind | params.rs, parallel.rs              | doc        | P1       |
| B3 | proxide-confind | cache.rs:69                         | doc        | P2       |
| B4 | proxide-confind | parallel.rs (contact_degree_raw)    | comment    | P2       |
| B5 | backlog         | —                                   | backlog    | P2       |
| B6 | proxide-rotlib  | specs/260529_rotlib.md (new, via A2) + debt #67 | doc | P2 |
