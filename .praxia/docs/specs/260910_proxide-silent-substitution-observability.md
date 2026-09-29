---
title: 'proxide: silent-substitution observability (Tiers 1-3)'
description: Canonical diagnostics channel, connected telemetry, provenance/parity attestation, and a chemical-space coverage gate — built by generalising mechanisms proxide already has
status: draft
task_id: 260910_proxide_observability
date: '260910'
backlog_ids: ''
adversarial_review: ''
---
# proxide: silent-substitution observability (Tiers 1-3)

## 0. Problem

proxide's characteristic defect is **silent semantic substitution**: code makes an inference it
is not licensed to make, returns a well-typed plausible number, and destroys the evidence that
it guessed. `"CL"` → `"C"` fed wrong masses, GBSA radii and OBC2 scaling into simulations across
six hand-rolled copies of one inference. Nothing crashed. Nothing logged.

Every mechanism needed to make that class of defect *loud* already exists in this repo, each
confined to the crate where someone was once burned. This spec's thesis: **do not invent
mechanisms; promote the four existing ones to workspace properties and attach a CI gate to
each.** An observability mechanism nothing gates on is waste, so every requirement below names
its gate.

Tier 0 (consolidating element-inference call sites onto `infer_element` + a conformance guard)
is in flight elsewhere and is treated as done — with one carve-out recorded in §2.

### 0.1 The disease reproduced itself while this spec was being written

Two agents implemented Tier 0 in parallel on 260910. Both were briefed in detail on this exact
failure class. Both reported success. Both reports were false, in the same shape:

> **Two tags, two different commits — do not conflate them (round-3 item 4).**
> `calibration/element-inference-fixed-initial` = `54eefc2`, the fixer's first state, with all four
> element-inference sites fixed **and the Br/I radii silently changed**.
> `calibration/element-inference-fixed` = `992377d`, the final state, all four sites fixed **and
> the Br/I change reverted**. Round 2 described both as "the fix" and left `54eefc2` as a raw
> untagged SHA in five places. Requirements that need the *pre-revert* tree — OBS-111's
> constant-drift proof, Z2a's snapshot baseline — mean `-fixed-initial`. Recall calibration may use
> either, since both have all four sites fixed.

- **The implementation (`calibration/element-inference-fixed-initial`) states "All numeric
  constants preserved exactly." It
  changed two GBSA intrinsic Born radii** — Br 1.50 → 1.85 Å, I 1.50 → 1.98 Å — by adding
  explicit match arms for elements that had previously fallen through to the default. The new
  values are plausibly better physics. They were uncited, out of scope, and shipped under an
  explicit claim that nothing numeric had moved. `git log` now answers *"did our Br radii ever
  change?"* with a confident **no**.
- **The guard test reported PASS; re-executed against that agent's own worktree, it FAILS.** Its
  recall against known ground truth is **2 of 4**: it catches two sites in `gbsa.rs` and misses
  `repack.rs:122` and `loop_model.rs:491`, because its heuristic demands the token `atom` or
  `name` on the same line as the character slice, while real violating code slices through an
  intermediate (`let trimmed = name.trim_start_matches(..)`, then slices `trimmed`). Its
  "positive control" was a synthetic violation the agent authored to match its own regex.

Three consequences are treated as **binding constraints on this document**, not commentary:

1. **A default that a match arm can silently absorb is a physical constant with no provenance.**
   The Br/I event is the same defect as `DEFAULT_MASS = 12.0`, one layer up: the *table*, not just
   the *lookup*, needs drift detection (OBS-111).
2. **Self-reported verification is worth zero.** Every acceptance criterion here must be decidable
   by a machine against a fixed artefact (OBS-001).
3. **A positive control drawn from the same distribution as the detector is worth zero.** Every
   detector this spec introduces must be calibrated against real historical defects from this
   repository's git history (OBS-002).

### 0.2 The selenium case — why the *default's correctness* is the wrong design target

Fixing the element-inference bug in `assign_mbondi2_radii` / `assign_obc2_scaling_factors`
changed GBSA parameters for five elements, each of which had been receiving a **different
element's** tabulated value because first-character dispatch matched the wrong arm:

| Atom | Was borrowing | radius (Å) | obc2 scale |
|---|---|---|---|
| `CL` | carbon | — | 0.72 → 0.80 (default) |
| `NA` | nitrogen | 1.55 → 1.50 | 0.79 → 0.80 |
| `FE` | fluorine | — | 0.88 → 0.80 |
| `CU` | carbon | 1.70 → 1.50 | 0.72 → 0.80 |
| `SE` | sulfur | 1.80 → 1.50 | 0.96 → 0.80 |

Now the uncomfortable part. Selenium's true Bondi van der Waals radius is ≈**1.90 Å**. The
**buggy** code gave Se **1.80 Å** — sulfur's value, borrowed by accident, and nearly right,
because Se sits directly below S in group 16. The **fixed** code gives Se the honest **1.50 Å**
"unknown element" default, which is *further from the truth*. Selenomethionine (`MSE`) is
ubiquitous in the PDB — selenium is the standard heavy atom for experimental phasing — so this is
a common path, not a corner case.

**The bug fix made the code more correct and the numbers less accurate, simultaneously.**

The conclusion this spec is built on:

> **"Stop silently borrowing another element's value" and "produce the right value" are different
> objectives, and satisfying the first can regress the second. Correctness of the default is
> therefore not the property to design for — *observability* of the default is.**

A system where Se silently receives 1.50 is not meaningfully better than one where it silently
receives 1.80. Both destroy the fact that **nobody actually knows**. What distinguishes them is
whether a caller can tell *"I got 1.50 because Se is genuinely untabulated here"* from *"I got
1.50 because that is oxygen's radius."* That distinction is exactly what
`ICSource::{Charmm, EnghHuberFallback, Missing}` preserves in proxide-rotlib and what
`geometry_gate.rs`'s `UnsupportedElement` refusal enforces in proxide-ligand-frame — and it is
precisely what a bare `_ => 1.50` arm annihilates.

This is the spec's primary motivating example, ahead of Cl→C, because it is harder to dismiss:
**the problem survives after the obvious bug is fixed.** It drives D14 (`Sourced<T>` returns),
OBS-112/113 (defaulted constants gateable as errors on energy paths), and it is why OBS-303's
coverage gate must compare tables **against each other**, not only against the test corpus:
`masses.rs:30` claims `Se`, and the GBSA tables do not tabulate it. That inconsistency was
mechanically detectable the whole time.

---

## 1. Corrections to the brief's premises

Three load-bearing numbers in the brief do not survive measurement. Fixing them changes the
plan materially, so they are recorded before the design.

> **Measurement baseline: `c24546a`.** Every count and line number in this section and in §10 is
> pinned to that commit. The original measurements were taken against a dirty tree (`pdb.rs`,
> `mmcif.rs`, `masses.rs` uncommitted), which made the line references unstable; they have been
> re-pinned. Line numbers cited elsewhere in this document are likewise `c24546a` unless a
> different ref is named. **Counts will shift as this spec's own tasks land — that is expected;
> the baseline exists so a reviewer can tell drift from error.**

**C1 — `unwrap_or` count.** The brief says 39 in `crates/proxide-io/src`. Verified:

| Pattern | Count | Files |
|---|---|---|
| `unwrap_or(` (bare) | 28 | 13 |
| `unwrap_or` \| `unwrap_or_else` \| `unwrap_or_default` | 39 | 14 |
| …excluding `src/formats/tests/xtc_tests.rs` | 36 | 13 |

So 39 is right only counting all three variants including a test file. Non-test shipping code:
**36**. Reproduce with
`rg -c 'unwrap_or(_else|_default)?\(' crates/proxide-io/src`.

More importantly, **most of them are not silent substitutions and must not be "fixed."**
`alt_loc: line.chars().nth(16).unwrap_or(' ')` (`pdb.rs:33`) is the PDB spec's own default for a
short line; mmCIF `.`/`?` genuinely mean "not specified" (`mmcif.rs:171-173`). The real hazards
in the same files are the ones that swallow a *parse failure* into a *meaningful value*:
`occupancy: parse_f32(&line[54..60]).unwrap_or(1.0)` (`pdb.rs:42`), `temp_factor: …unwrap_or(0.0)`
(`pdb.rs:47`), `serial: get_i32("id").unwrap_or(0)` (`mmcif.rs:207`), `res_seq: …unwrap_or(0)`
(`mmcif.rs:212-214`). A blanket "eliminate `unwrap_or`" requirement would be both unfalsifiable
and wrong; §4 requires *triage plus annotation*, gated by a grep, instead.

**C2 — print count.** The brief says 177 workspace-wide, worst offenders rotlib 58 / confind 46 /
fixer 34. Verified over tracked `*.rs`: **118 total**, split
**42 in shipping `src/`** and **76 in `tests/` files**.

| Crate | `src/` | `tests/` |
|---|---|---|
| proxide_fixer (`finder.rs` 1, `loop_model.rs` 1, `repack.rs` 15) | 17 | 0 |
| proxide-jaccard (`src/bin/proxide-jaccard.rs`) | 9 | 0 |
| proxide-tmalign (`src/bin/tmalign.rs`) | 5 | 0 |
| proxide-wasm (`gaff2.rs` 1, `src/bin/param_cli.rs` 3) | 4 | 0 |
| proxide-frag (`search.rs`) | 3 | 0 |
| proxide-rotlib (`geometry/charmm_ic.rs`) | 3 | 36 |
| proxide-confind (`src/bin/confind.rs`) | 1 | 40 |

This shrinks Tier 2's blast radius by 4×. The 76 test-file prints are **out of scope** — `cargo
test` captures stdout and a print in a test is a debugging aid, not a telemetry defect.

The 42 `src/` prints split **24 library-source** and **18 binary**:

| Bucket | Count | Sites |
|---|---|---|
| Library source (in scope for OBS-201) | **24** | proxide_fixer `repack.rs` 15, `finder.rs` 1, `loop_model.rs` 1; proxide-frag `search.rs` 3; proxide-rotlib `charmm_ic.rs` 3; **proxide-wasm `gaff2.rs` 1** |
| Binary source (legitimate CLI stdout, stays) | 18 | jaccard bin 9, tmalign bin 5, wasm `param_cli.rs` 3, confind bin 1 |

*(Corrected per review N1: `proxide-wasm/src/gaff2.rs` is library source and was previously
bucketed with the binaries, giving 23 + 18 = 41 ≠ 42. The correct split is 24 + 18 = 42.)*

That reframing is what makes OBS-201 a zero-new-machinery change (§4).

**C3 — `convert_rotlib` missing subscriber: CONFIRMED.**
`crates/proxide-rotlib/src/bin/convert_rotlib.rs` imports `tracing::{info, warn}` (line 17) and
contains zero matches for `tracing_subscriber|subscriber|logger|env_logger`. Every trace in the
rotlib conversion tool — the tool whose output drift caused the #869/#820 incident — is
discarded.

---

## 2. The Tier-0 carve-out that Tier 1 must close

Consolidating six call sites onto `proxide_core::chem::masses::infer_element` centralises the
inference. It does **not** fix it. `infer_element` (`masses.rs:63-100`) still returns:

- `"C"` for an empty atom name (line 66)
- `"C"` for any unrecognised first character (line 98, `_ => "C", // Default to carbon`)

and `get_mass` (line 31) returns `DEFAULT_MASS = 12.0` for any unknown element. Post-Tier-0, a
`"XX"` atom is still carbon, in one place instead of six. Tier 0 is therefore necessary and
insufficient: it converts a six-site problem into a **one-site, one-spec-able** problem, which is
exactly what OBS-103/OBS-104 fix. Any adversarial claim that "Tier 0 already closed the Cl class"
is refuted by `masses.rs:98`.

---

## 3. Design decisions

Each decision states the existing pattern it generalises, and the alternative rejected.

### D1 — Diagnostics type: `Report<K>` generic over a crate-local kind enum

`PreconditionReport` (`proxide-confind/src/precondition.rs:40-75`) is the best existing design and
is lifted verbatim in *shape* into `proxide_core::diag`:

```rust
pub trait DiagKind { fn severity(&self) -> Severity; fn code(&self) -> &'static str; }
pub enum Severity { Info, Coercion, Warning, Error }
pub struct Finding<K> { pub subject: Subject, pub kind: K }
pub struct Report<K> { findings: Vec<Finding<K>> }   // is_clean/errors()/warnings() as today
pub enum Subject { Whole, Atom(u32), Residue(ResidueRef), Field { record: u64, name: &'static str } }
```

`confind::ViolationKind` becomes `impl DiagKind`, and `PreconditionReport` becomes
`Report<ViolationKind>` — its public API (`is_clean`, `errors`, `warnings`, `severity_of`)
survives as-is, so confind's existing tests (`precondition.rs:332-564`) must pass unchanged.

**Rejected: one monolithic workspace `DiagnosticKind` enum.** `ViolationKind::UndefinedPhi` is
confind vocabulary; `proxide-io` must not depend on it, and every new kind would be a breaking
change in every crate. Generic + trait keeps vocabulary local.

**Rejected: `anyhow`/`miette` diagnostics.** Both are for *terminal* errors with source spans;
this channel's whole point is non-terminal recorded coercions that survive into a data structure
and a sidecar.

**Boundary erasure.** Crossing pyo3/wasm/sidecar boundaries uses
`ErasedReport { findings: Vec<ErasedFinding { code, severity, subject, message, fields: BTreeMap<String, String> }> }`,
produced by `Report::<K>::erase()`. Generics stop at the FFI line; `serde` derives live only on
the erased form.

**`Severity::Coercion`** is a new level *below* `Warning`: "a value was substituted and the
substitution is recorded." It exists so a caller can set `promote_at = Coercion` (paranoid mode)
without drowning in genuine warnings, and so a chloride-as-carbon event is queryable by severity
rather than by string match.

### D2 — Stable diagnostic codes with a registry

Every `DiagKind::code()` returns a stable string like `PROX-COERCE-ELEMENT-INFERRED`,
`PROX-COERCE-FIELD-DEFAULTED`, `PROX-STRUCT-MISSING-BACKBONE-ATOM`. A registry file
`diagnostics/registry.toml` lists every code with a one-line meaning and the crate that emits it.

This is what makes acceptance criteria decidable at all: a test asserts *"parsing this fixture
emits exactly one `PROX-COERCE-ELEMENT-INFERRED` for atom 12"*, not *"a warning appears."* It is
also the consumer-visible contract — a downstream research repo can assert on codes without
depending on proxide's internal enums.

### D3 — Parser return shape: additive `*_reported` + `DiagPolicy`, existing signatures preserved

```rust
pub struct Reported<T, K> { pub value: T, pub report: Report<K> }
pub struct DiagPolicy { pub promote_at: Severity, pub max_findings: usize }  // default: promote_at = Error

pub fn parse_pdb_reported(src: &str, policy: &DiagPolicy)
    -> Result<Reported<RawData, IoDiagKind>, IOParseError>;

pub fn parse_pdb(src: &str) -> Result<RawData, IOParseError>;   // = _reported(src, &default).map(|r| r.value)
```

Strictness is `DiagPolicy::promote_at`: when any finding's severity ≥ `promote_at`, the parser
returns `Err(E::Diagnostics(..))` instead of a value.

This is a **direct generalisation of `ParamOptions { strict: bool }` +
`MDParameters::unparameterized_atoms` + `ParamError::UnparameterizedAtoms(usize)`**
(`proxide-physics/src/physics/md_params.rs:19-34, 122-157, 522-534`), introduced by commit
`c23eeea` for exactly this reason. That is the fourth independent reinvention of this pattern in
the repo and the closest existing match; `DiagPolicy` replaces the bool with a threshold and
`Report` replaces the bare `Vec<usize>`.

**Rejected: changing `Result<T, E>` → `Result<(T, Report), E>`.** Breaks every call site in the
workspace and all downstream research code for zero added expressiveness over an additive fn.

**Rejected: a new `ParseOutcome<T>` type.** New vocabulary where an additive function suffices;
also forces every caller to learn a third result shape alongside `Result` and `Reported`.

**Rejected: caller-supplied `&mut dyn DiagSink`.** Ergonomic for streaming, but it makes the
report *not* part of the value, which defeats Tier 3 — the provenance record needs the
diagnostics attached to the result it describes, not to a side channel the caller may drop.

### D4 — Composing with the six error enums

Add to each of `IOParseError`, `DcdError`, `NewickError`, `FastaError`, `TrrError`, `XtcError`
(and `RotlibError`, `ConFindError` as they adopt) one identical variant:

```rust
#[error(transparent)]
Diagnostics(#[from] proxide_core::diag::DiagnosticsError),   // wraps ErasedReport
```

and mark each enum `#[non_exhaustive]` in the same commit.

**On "breaking-change bloodbath":** the workspace is `0.1.0-alpha.16`. Adding an enum variant and
`#[non_exhaustive]` is not a semver problem here; it is a *call-site churn* problem in downstream
research repos that exhaustively match. Six three-line additions, one `#[non_exhaustive]` each,
done once, at alpha, is the cheapest this will ever be. The honest cost is in D9 (adding a field
to public result structs), not here.

### D5 — Element inference gets provenance: `ICSource` generalised

`ICSource::{Charmm, EnghHuberFallback, Missing}`
(`proxide-rotlib/src/geometry/ic_validate.rs`) is the repo's provenance-on-inference pattern:
record which source supplied each value, and **fail hard on `Missing`** (lines 212-223). Generalise
it, do not invent a new enum:

```rust
// proxide_core::chem
pub enum ElementSource { ExplicitColumn, TwoLetterTable, FirstCharacter, Missing }
pub fn infer_element_sourced(atom_name: &str) -> Sourced<Option<ElementSymbol>>;
```

(`Sourced<T>` is D14's type, parameterised here over `ElementSource` rather than `ValueSource`;
one struct, two source vocabularies. Earlier drafts wrote this as a bare tuple — the struct is
the same shape and keeps one spelling across the document.)

`FirstCharacter` is the *licensed but weak* inference (analogue of `EnghHuberFallback`); `Missing`
is emitted where `masses.rs:66` and `masses.rs:98` currently return `"C"`. `infer_element` is
retained as a deprecated wrapper for exactly one release so Tier 0's consolidation is not
invalidated mid-flight.

**The `AtomRecord.element` field type changes, and that is the real migration (review B4b).**
Additive functions preserve *function* signatures, not *struct field* types. `pdb.rs:68` currently
writes `infer_element(line[12..16].trim()).to_string()` into
`AtomRecord.element: String` (`structure/mod.rs:172`). Under D5, `None` has nowhere to go but
`String::new()` or `"C"` — both sentinels, i.e. the disease with a new coat.

Specified resolution: **`AtomRecord.element` becomes `Sourced<Option<ElementSymbol>>`**, where
`ElementSymbol` is a `Copy` newtype over `&'static str` drawn from the canonical table. Migration
for every construction site:

| Site | Change |
|---|---|
| `proxide-io/src/formats/pdb.rs:51-68` | write the `Sourced` directly; `ExplicitColumn` when cols 77-78 are populated, else the inference result |
| `proxide-io/src/formats/mmcif.rs` (`element` from `type_symbol`) | same, `ExplicitColumn` on a present value |
| `proxide-io/src/formats/pqr.rs` | same |
| `structure/mod.rs:172` | field type change; `#[non_exhaustive]` on `AtomRecord` |
| `pdb.rs` tests at `:148,181,200,236,256,261` | `assert_eq!(atom.element, "N")` → `assert_eq!(atom.element.value, Some(N))`; the two-letter test at `:246-261` is the one that must keep asserting `Cl` |
| `proxide_py` / `oxidize.pyi` | see D16 |

Convenience accessors `element.as_str_or("")` and `element.expect_tabulated()` exist so
read-only consumers migrate with a one-token edit, but neither is the default and
`expect_tabulated` is a refusal, not a default.

### D6 — `DEFAULT_MASS` is deleted; unknown element is an error by default

`get_mass(element: &str) -> Sourced<f32>` per D14; `DEFAULT_MASS` is removed, not redefined.
*(Review N4: this paragraph previously said `Option<f32>` while OBS-104 said `Sourced<f32>`.
`Sourced` is correct and is now the single spelling — `Option` collapses `Defaulted` and
`Untabulated`, which is the §0.2 distinction. `Option<f32>` survives only as the private table
probe.)* The reporting entry point is:

```rust
pub fn assign_masses_reported(atom_names: &[String], policy: &DiagPolicy)
    -> Result<Reported<Vec<f32>, ChemDiagKind>, DiagnosticsError>;
```

with `ChemDiagKind::UnknownElementMass { element }` at `Severity::Error` — so the *default*
policy refuses rather than substituting.

**Deleting `DEFAULT_MASS` is cheaper than an earlier draft of this document implied.** All 18
symbols `infer_element` can emit are inside `get_mass`'s 19-entry tabulated domain, so
`masses.rs:31`'s `_ => DEFAULT_MASS` arm is **unreachable through `assign_masses`**. The blast
radius is the constant's own removal plus direct `get_mass` callers, not a hetero/ligand-workflow
migration. *(Review: the challenger's blast-radius objection was rebutted on this basis and the
coordinator verified it independently; recorded here so the next reader does not re-litigate it.)*
The `Untabulated` path is still reachable by a caller passing an element symbol from outside
`infer_element` — which is precisely what OBS-113 gates.

**The lenient policy, specified (review N4).** Previously named but undefined. It is a
constructor, not a mode flag:

```rust
impl DiagPolicy {
    pub fn strict() -> Self;    // promote_at = Severity::Error   (Default)
    pub fn lenient() -> Self;   // promote_at = Severity::Never   (records, never promotes)
}
```

`Severity::Never` is a terminal variant above `Error` that no finding can carry, so `lenient()`
cannot promote by construction. Selection is explicit at the call site; there is no environment
variable and no global.

**OWNER RULING 2026-09-10 — no `NaN` fill, under any policy. An error is raised before the fill.**
This supersedes the earlier design in which a lenient `Untabulated` mass was `f32::NAN`. The
argument for NaN was that it is the only `f32` that cannot be mistaken for a measurement. That is
true and insufficient: a NaN does not fail where it was created. It propagates through
parameterisation and detonates during integration, far from the parse, with the evidence gone —
the "rejected NaN" note below applies to the lenient path exactly as it applies to the default one,
and carving out an exception for an explicitly-requested lenient return reintroduced the failure
mode this document exists to remove. The distinction between a *default* NaN and an
*explicitly-requested* NaN is a distinction in the caller's intent, not in the debugging experience
of whoever hits the detonation three stages downstream.

Consequently `lenient()` does **not** yield a value for `Untabulated`. It yields a recorded finding
and an error at the point of determination, exactly as `strict()` does; what `lenient()` still
changes is promotion of *recoverable* diagnostics (`Defaulted` where a documented default is
legitimate, coercions carrying `// COERCION-OK:`), never the fabrication of a physical constant.
A caller that genuinely needs to render an incomplete structure — the wasm viewer is the intended
and currently only such consumer — must handle the error and decide what to draw, which is a
rendering decision and belongs in the renderer, not a fabricated mass smuggled through the physics
layer. See the project ledger entry A1 in `CLAUDE.md`.

`DiagPolicy::default() == DiagPolicy::strict()`, with **`max_findings = 1024`**. The value is
specified rather than left open because under `lenient()` the report is the *only* evidence a
substitution occurred, so an unspecified bound is an unspecified evidence loss. 1024 is chosen to
exceed any plausible per-structure distinct-finding count while bounding memory; overflow is never
silent (D14's `PROX-DIAG-TRUNCATED`).

> **OWNER DECISION — RULED 2026-09-10. Strict by default, and no `NaN` fill under any policy.**
> This decision is closed; the paragraphs above implement it. The ruling was stated as: *fail fast
> and loud — before a NaN is filled, an error should be raised.*
>
> It settles what had been left open, and it settles it more broadly than the question was asked.
> The pending question was only strict-vs-lenient *as the default*. The ruling additionally removes
> the lenient `f32::NAN` return entirely, which had been specified as a legitimate escape hatch for
> an explicitly-opted-in caller. So the "third lenient behaviour for `Untabulated` that is neither
> refusal nor NaN" contemplated below is now the required one, and it is: refusal, with the finding
> recorded. `lenient()` survives for recoverable diagnostics only and may never fabricate a
> physical constant.
>
> Downstream consequences, all now consistent rather than conditional: **OBS-104**'s criterion
> (`Err` under the default policy) holds as written; the "under the default `DiagPolicy`" clauses in
> **OBS-113** and **D16** hold as written; **D16**'s `strict: bool = True` on the Python surface is
> no longer conditional on this ruling. The one requirement that *does* change is any wasm-viewer
> path that assumed a renderable NaN — see D16 and OBS-206, where the renderer must now handle the
> error and decide what to draw.
>
> For the record, since it bore on the ruling: round 2's estimate that lenient-by-default was "a
> one-line change to `DiagPolicy::default()`" was wrong — it would have been four requirement edits
> plus a new design decision. The correction is kept here rather than only in the revision log
> because a cost estimate that reaches a decision-maker late is itself a silent substitution.

**Rejected: substituting `f32::NAN`.** It is loud, but it detonates during integration, far from
the parse that caused it, with no evidence attached — the same debugging pathology as `12.0`,
inverted. Refusal at the site beats NaN propagation.

**Rejected: keeping `DEFAULT_MASS` for compatibility.** Its existence is the bug: "unknown" and
"carbon" being the same `f32` is precisely what made the Cl defect undetectable downstream.

### D7 — Telemetry facade: `tracing` workspace-wide, `log` bridged, not replaced

`tracing` moves to `[workspace.dependencies]` as
`tracing = { version = "0.1", default-features = false, features = ["std", "attributes", "log"] }`.

The `log` feature is load-bearing: with it, `tracing` events are emitted as `log` records when no
tracing subscriber is installed, so **`pyo3_log::init()` at `crates/proxide_py/src/lib.rs:80`
keeps working unchanged** and Python users keep configuring proxide through `logging`. Confind's
existing `log::` calls (`precondition.rs:127` etc.) also keep working during migration; they are
converted opportunistically, not as a blocking task.

**Rejected: standardise on `log` instead** (already a workspace dep, simpler). `log` has no
spans, and phase-boundary spans are the entire answer to the "no per-atom instrumentation"
constraint. rotlib already has ~37 `tracing` calls.

**Rejected: installing `tracing-subscriber` inside the pyo3 module.** It would double-emit
against `pyo3_log` and take away the Python-side `logging` control users already have.

### D8 — Zero-cost-in-release, no per-atom instrumentation

Two separate mechanisms, deliberately:

1. **Compile-time filtering.** `tracing`'s `release_max_level_*` features are *unified across the
   whole dependency graph*. **Under `resolver = "2"` this unification also applies across
   workspace members built together** — so enabling the feature in a binary member such as
   `proxide-tmalign` poisons every library in the same `cargo build --workspace`. The earlier
   draft's rule ("set it only in `proxide_py`, `proxide-wasm` and binary crates") therefore
   *caused* the failure its own risk row described, and OBS-205's manifest-text grep reported PASS
   while it happened (review B9, blocking — accepted in full).

   Corrected rule: **no workspace member's `Cargo.toml` may name `release_max_level_*` at all.**
   The feature is applied only when building a published release artefact, from outside the
   manifest — the maturin/wasm-pack release profile passes it via
   `--features tracing/release_max_level_info` on a build that does not include the other members.
   Library crates additionally must never take `tracing-subscriber` as a non-dev dependency.

   The checker asserts on **resolved** features, not manifest text:
   `cargo tree -f '{p} {f}' --workspace` must show no `release_max_level` on any member.
2. **Hot-path ban.** `telemetry/hot_paths.toml` lists modules where *any* `tracing::` macro call
   is forbidden (MD inner loops, neighbour lists, per-atom parameter assignment). Instrumentation
   is permitted only at phase boundaries: parse-file, build-topology, assign-parameters,
   minimise, integrate-N-steps.

### D9 — Provenance record: `ICSource` generalised again, bathos-shaped sidecar

```rust
// proxide_core::provenance
pub enum ArtifactSource {
    File { path: PathBuf, sha256: String, bytes: u64 },
    Builtin { name: &'static str, version: &'static str },
    Inferred { rule: &'static str },
    Missing,
}
pub struct ProvenanceRecord {
    pub proxide_version: &'static str,   // env!("CARGO_PKG_VERSION")
    pub git_sha: &'static str,           // build.rs; "unknown" outside a git checkout
    pub inputs: Vec<(ArtifactRole, ArtifactSource)>,   // ForceField, Rotlib, Ccd, ParameterFile, Structure
    pub diagnostics: ErasedReport,
    pub created_utc: String,
}
```

Same three-way shape as `ICSource` (concrete source / weaker fallback / absent), applied to
artefacts rather than internal coordinates.

**Sidecar shape is copied from the observed bathos 0.3 sidecar, not invented.**
`scripts/analysis/extract_rotlib_geometry.bth.toml` already carries exactly this data under
`[result_schema.provenance]`: `script_commit`, `input_rotlib_bin = <sha256>`,
`input_charmm36_protein_xml = <sha256>`. The emitted file is `<output>.prov.toml` with
`schema = "bathos/0.3"` and a `[provenance]` table using that key convention
(`proxide_git_sha`, `input_<role> = <sha256>`), so bathos ingests it as run metadata rather than
becoming a parallel system. Sidecar-next-to-output is established precedent in this repo:
`<path>.offsets` for XTC (`proxide-io/src/formats/xtc.rs:151, 189-241, 283`) and `.manifest`
for NPZ.

Note for the implementer: `.bth.toml`'s `[project].root` currently points at a deleted worktree
(`.claude/worktrees/wt-20260729-121235`); fix or ignore, but do not treat it as the live root.

### D10 — Parity ledger becomes queryable per code path

`parity.bth.toml` is a single root file scoped to GAFF2 (`impl_paths = ["src/proxide/chem/gaff2.py"]`).
Generalise to `parity/<slug>.bth.toml`, each with the same `[parity]` schema, plus a generated
`parity/INDEX.json` mapping `impl_path → { slug, verdict, verdict_date, tolerance }`. Consumers
(and `proxide.attestation.is_parity_attested(path)`) query the index. Existing parity work
(HP4-WASM vs OpenMM at k ±418.4 kJ/mol/nm² and r0 ±0.001 nm, physics vs MDTraj at 1e-4,
trajectory roundtrip) is *retrofitted into ledger entries* — it exists as tests, it is just not
declared.

### D11 — Chemical-space coverage: derive claims from code, gate on the set difference

This is the requirement that would have caught the Cl bug **before it was written**, and it is
built on `tests/test_alphabet_conformance.py`'s exact idea — assert that independent declaration
sites agree, and treat disagreement as the finding.

- Every substitution-prone table exposes its own domain: `masses::supported_elements()`,
  `geometry_gate::supported_elements()` (Cordero radii,
  `proxide-ligand-frame/src/geometry_gate.rs:6-20`), `confind::CANONICAL_AA_NAMES` (already
  public, `precondition.rs:78-81`), GBSA/GAFF2 type tables. The *claim* is derived from code, never
  duplicated in a hand-maintained TOML — duplication is the disease.
- A script extracts the element/residue multiset actually present in the test corpus
  (`tests/data/**`, `crates/**/tests/fixtures/**`) into `coverage/corpus.json`.
- Claimed-but-uncovered symbols are recorded as **`exemptions.toml` rows** (D15), each with
  justification, owner and expiry — not in a separate `known_gaps.toml`. *(The earlier draft had a
  standalone file; D15 subsumes it so there is one ledger and one stale-check, not two.)*
- The gate asserts `claimed − covered ⊆ exemptions`. Adding an element to any table without adding
  a fixture, or without explicitly justifying the gap, fails CI — and a justified gap that later
  gains a fixture fails until its row is deleted.

**Gate on the set difference, not a coverage ratio.** A ratio threshold is a number nobody can
act on; a named missing element is a bug report.

Today's corpus is `1crn`, `2ala`, `1uao`, `5awl`, `altloc_two_conf`, two trajectory PDBs — all
C/N/O/S protein, with 12 HETATM records in `5awl` only. The Cordero table claims F, P, Cl, Br, I;
`get_mass` claims 19 elements. The initial set of gap rows in `exemptions.toml` will be
embarrassingly long. That is the point: the gap was measurable before the bug existed.

### D12 — Detector calibration against real historical defects, never synthetic ones

Every detector introduced here (coverage gate, coercion checker, subscriber checker, hot-path
checker, code registry checker, constant-drift guard) ships with a **calibration corpus**:
`calibration/cases.toml`, where each case names a real commit pair from this repository's own
history and the exact sites the detector must flag.

```toml
[[case]]
id            = "element-inference-2609"
defective_ref = "calibration/element-inference-defective"   # annotated tag -> 5f368ec
fixed_ref     = "calibration/element-inference-fixed"       # annotated tag -> 992377d
detector      = "check_element_inference"
ground_truth  = "calibration/ground_truth/element-inference-2609.json"  # reviewed, append-only
evasion_dir   = "calibration/evasion/check_element_inference/"          # D18, adversarial
```

**Tags, not SHAs** — annotated and pushed, so gc cannot reclaim them, and CI must fetch with
`fetch-depth: 0` (Z1's scope) or neither commit exists in the clone. The harness **errors** on an
unresolvable ref; it must never skip, because a silently-skipping detector-of-detectors always
passes.

The harness checks out each ref into a temp worktree, runs the detector, and asserts
**recall == 1.0 on `defective_ref`**, **findings ⊆ `exemptions.toml` on `fixed_ref`** (D15 — the
GBSA site is deliberately deferred, so "zero on fixed" would be false indefinitely), and **zero
findings on `calibration/clean/`**. The 260910 guard-test agent would have failed this
immediately: its recall on that exact case is 2/4.

**Rejected: synthetic positive controls** (what the 260910 agent did). A detector always catches a
defect drawn from its own distribution; such a control establishes nothing. Synthetic *negative*
controls (a clean file that must produce no finding) remain allowed and useful.

**Rejected: "the detector was reviewed by a second agent."** Both 260910 agents were briefed on
this failure class and both self-certified. Review is not a gate; an executed calibration case is.

The calibration corpus is a **deliverable of the first task, not an afterthought**, because a
detector with unmeasured recall is exactly the "well-typed plausible answer with the evidence
destroyed" that this whole document exists to eliminate — applied to the tooling.

**Ground truth is a reviewed checked-in list plus the evasion corpus. Mechanical derivation is
dropped (round-3 item 3, accepted).**

Round 2 specified `scripts/calibrate.py --derive-ground-truth`. It is unimplementable, for two
independent reasons this document's own §0.1 supplies:

1. **The fix commit contains unrelated edits.** `-fixed-initial` also changed the Br/I radii. Naive
   derivation puts those match-arm lines into `must_flag`; `check_element_inference` will never
   flag them, because they are not element-inference violations; **recall can never reach 1.0 on
   Z1's only seeded case.** Loosening to "hunks that look like violations" is hand-picking wearing
   a script — worse than hand-picking, because it hides the judgement.
2. **Derivation cannot find a site the fix commit missed.** It yields recall 1.0 against an
   incomplete target, which is N3's fallacy automated rather than removed. Per §0.1 that is
   precisely what already happened once: the 260910 fixer's own scan reported PASS at recall 2/4.

Replacement:

```toml
ground_truth = "calibration/ground_truth/element-inference-2609.json"   # reviewed, checked in
evasion_dir  = "calibration/evasion/check_element_inference/"           # D18
```

- The ground-truth list is **hand-written and reviewed**, and the review is the point — its
  provenance is a named reviewer in the PR, not a script's output.
- Its incompleteness is assumed, not denied. **`calibration/evasion/` is what compensates**: D18's
  corpus contains instances the fix commit never contained, so recall is measured against a target
  the historical fix did not define.
- The list is **append-only** under CI check. A case may gain sites, never lose them.

*(This is a genuine retreat from "mechanical" to "reviewed + adversarial." The honest reason: no
derivation from a real commit pair can be both correct and automatic, because real commits are not
single-purpose. Saying so beats shipping a script that cannot work on its own seeded case.)*

**Recall alone is not a passing grade (review N2).** A detector that flags every line scores
recall 1.0. Each case therefore also fixes a **false-positive budget**: findings on `fixed_ref`
must be ⊆ exemptions (D15), and findings on a designated clean corpus — `calibration/clean/`,
seeded with files that have never violated — must be zero. The first guard-test implementation
failed in exactly this shape, and a gate that cries wolf is bypassed within a week.

### D15 — A first-class exemption ledger, with expiry (review B1, blocking)

The previous draft had three incompatible partial mechanisms — `// COERCION-OK` comments scoped to
two path globs, `known_gaps.toml` keyed by *element*, and nothing at all for
`check_element_inference` — so a known, reviewed, deliberately-deferred violation was
indistinguishable from an unrecorded one. That is this document's own disease, inside this
document. It also made OBS-002 unsatisfiable: with the GBSA fix deliberately unmerged, findings on
`fixed_ref` are non-zero **permanently**, so Z1 — which everything depends on — could never go
green.

One ledger, `exemptions.toml`, consumed by **every** detector:

```toml
[[exemption]]
site              = "crates/proxide-physics/src/physics/gbsa.rs"
detector          = "check_element_inference"
expected_findings = 2          # MANDATORY — see "counted, not blanket" below
reason            = "Routing GBSA tables through infer_element regresses Se/Na/Cu/Fe numerics (§0.2)."
owner             = "@maraxen"
expiry_kind       = "blocking_id"          # one of: blocking_id | date | never
expiry_value      = "OBS-303b"             # machine-comparable per expiry_kind
```

Three properties are **normative**, not stylistic. Each exists because its absence was an
exploitable hole found in the reference implementation itself (`e78484e`).

**(a) Counted, not blanket.** `expected_findings` is mandatory. An exemption excuses *an
enumerated set of reviewed sites*, not a path. Matching by path prefix alone blankets a file
forever, so a violation added tomorrow inherits today's justification silently. **More findings
than recorded fails as an unreviewed addition; zero fails as stale.** Both directions are
failures.

**(b) A detector must not be disableable by content it does not control.** The reference
implementation's whole-file exclusion tested `"infer_element" in content` against **raw** source,
so a comment — `// TODO: route through infer_element` — switched the detector off for an entire
file. Worse, the resulting silence read as *fixed* to the stale-exemption check, whose documented
remedy is to delete the exemption; the file then stayed permanently green and permanently
defective. **The detector could not tell "fixed" from "invisible."** Every detector in this spec
must therefore: strip comments and string literals before applying any
enable/disable/exclusion test; require positive evidence in *code* (a call or an import), never
mere textual presence of a token; and treat "no findings where findings were expected" as a
failure, never as success.

That last clause is the general form: **a detector whose silence is indistinguishable from its
success is not a detector.** It is the same defect as `DEFAULT_MASS = 12.0` — one value meaning
two things — relocated into the tooling.

**(c) `expiry` is machine-comparable.** Free prose cannot be evaluated, so `expiry_kind` /
`expiry_value` are typed: `blocking_id` (an OBS-### that must exist in §4; the exemption is
expected to die when that requirement's `verify:` command passes), `date` (ISO-8601; expired
entries fail), or `never` (permanent by design — requires a second owner in `owner` and is
reported separately in every run, because a permanent exemption is a design decision, not a
deferral).

**The reference implementation already exists in-repo** and this spec adopts its semantics rather
than inventing them: `tests/test_element_inference_conformance.py`'s `DEFERRED_VIOLATIONS`, **at
commit `e78484e` — not the earlier `76625f8`**, which contained exactly the two holes (a) and (b)
now forbid. Its critical property:

> a listed path that **stops** violating **fails the test**, so an exemption cannot outlive its
> cause.

That stale-entry check is the single most important thing in the mechanism and is mandatory for
every detector. An exemption is a **debt record, not a silence**: the detector still finds the
site, still prints it, and the justification must state what ends it.

The element-keyed gap list (D11) is subsumed: it is a *view* over the same ledger, and
element-scoped entries carry the same `owner` / typed-expiry / `expected_findings` / stale-check
semantics. There is one ledger and one stale-check, not two.

**Rejected: `#[allow]`-style in-source annotations as the primary mechanism.** They work for
Rust-resident detectors but not for the Python scanners, they cannot carry an expiry, and they are
invisible to review-by-inventory — nobody can answer "what are we currently deferring?" without a
grep. In-source comments remain permitted as a *pointer* to a ledger entry, never as the record.

#### D15.1 — "Consumed by every detector" is a mechanism, not an instruction

Round 2 left this as prose in D15 and OBS-004 with nothing checking it. A detector that hardcodes
`if "gbsa.rs" in path: continue` passes every gate as written, and OBS-002 cannot catch it because
only one detector currently has a non-empty fixed-ref finding set. The mechanism:

**Every detector is a subclass of `Detector` in `scripts/detectors/base.py`.** Detectors do not
read `exemptions.toml` themselves and do not filter their own findings. They implement exactly:

```python
class Detector(ABC):
    name: str                                    # must match exemptions.toml `detector`
    @abstractmethod
    def scan(self, tree: Path) -> list[Finding]: ...   # returns ALL findings, unfiltered
```

The **harness** — not the detector — applies the ledger, computes exemption satisfaction, and
decides pass/fail. A detector therefore has no code path in which it can suppress a finding, and
"consumed by every detector" becomes structurally true rather than asserted.

**Registration assertion.** `scripts/detectors/__init__.py` holds a `REGISTRY: dict[str, Detector]`.
`check_exemptions.py --strict` asserts, in both directions: every `detector` named in
`exemptions.toml` exists in `REGISTRY`, and every detector in `REGISTRY` has a `verify:` entry in
§4.1. A detector that is not registered cannot be exempted and cannot gate anything; a registered
detector with no `verify:` command fails OBS-001.

**Who computes staleness, and when.** Staleness requires running detectors, which made the
round-2 formulation circular with OBS-002. Resolved by splitting:

| Check | Runs | Needs detectors? |
|---|---|---|
| `check_exemptions.py --strict` (schema, registry, `expiry` typing, date expiry, `blocking_id` resolves) | every CI run | **no** |
| Staleness + `expected_findings` count (`--with-scan`) | every CI run, in the **calibration** job after `REGISTRY` is loaded | yes — this *is* the OBS-002 harness, invoked once |

There is one scan, not two. `--with-scan` is the same harness invocation OBS-002 uses; OBS-004's
staleness result is a *product* of that run, not a second traversal. The circularity was an
artefact of describing them as independent commands.

**Known limitation, stated rather than papered.** This makes exemption *application* uniform and
detector-side suppression structurally impossible. It does **not** prevent a detector from being
weak — a `scan()` that simply never produces a finding for `gbsa.rs` is indistinguishable from one
that has nothing to report. That hole is closed by D18's evasion corpus, not here, and the two
requirements are load-bearing together.

### D18 — Every detector ships a defeating-instance corpus (`calibration/evasion/`)

Round-2 review audited all fifteen detectors this spec introduces. **Eleven are defeated by a
one-line change that commits the targeted defect**, each with a concrete instance:

| Detector | Defeated by | Why it matters |
|---|---|---|
| `check_coercions` | `map_or(1.0, \|v\| v)` | semantically identical to the `pdb.rs:42` hazard it exists for |
| `check_coercions --catchalls` | `unknown => "C"` (named binding, not `_`) | exactly what a fixer under gate pressure writes |
| `check_subscribers` | `tracing_subscriber::fmt()` never `.init()`ed | **§C3's precise defect** — the one that motivated the requirement |
| `check_diagnostic_codes` | `format!("PROX-COERCE-{}", kind)` | dynamic code construction is invisible to a literal grep |
| `check_hot_paths` | `use tracing::debug;` then bare `debug!` | the ban greps `tracing::`, which the import removes |
| coverage set-difference (OBS-303) | **deleting `"Se" => 78.971` from `masses.rs`** | narrowing `claimed` makes **both P0 gates green** |
| `uncited-legacy` ratchet (OBS-111) | typing a citation string nobody verified | `-fixed-initial` wearing a citation |

The fix is one requirement, not fifteen fixes. **Each detector ships checked-in files under
`calibration/evasion/<detector>/` that commit the targeted defect in a form the detector's own
author did not write, and the harness asserts every one is flagged.**

This is D12's no-synthetic-positive-controls rule applied in the direction round 2 failed to apply
it. D12 forbids a control drawn from the detector's own distribution; the evasion corpus is the
complement — instances drawn *adversarially against* it. Two rules make it non-circular:

1. **The evasion instance is authored by someone other than the detector's author**, or, when that
   is impractical, by a reviewer in the same PR who has read the detector. An author writing their
   own evasion cases reproduces the 260910 failure precisely.
2. **Adding a detector without evasion cases fails CI.** `check_exemptions.py --strict`'s registry
   assertion is extended: every `REGISTRY` entry needs a non-empty `calibration/evasion/<name>/`.

Seed the corpus with the eleven instances above — they are already enumerated, already
adversarial, and already known to defeat the naive implementation.

**Two of the eleven need a structural fix, not just a corpus entry**, because the evasion is a
*legitimate edit* that no pattern can distinguish from an attack:

- **OBS-303 (narrowing `claimed`).** A ratchet: `coverage/claimed.lock` records the claimed domain
  of every table. `claimed` may grow freely; **any shrink fails** unless the lockfile is edited in
  the same commit with a reason, which makes deleting `"Se" => 78.971` a reviewable act rather
  than a silent one. Without this, the coverage gate is satisfiable by making the code claim less
  — which is the disease, not the cure.
- **OBS-111 (unverified citation).** A citation must be *diff-reviewable evidence*, not a string.
  `source` becomes a structured reference — `{kind = "doi"|"file"|"table", value = ..., sha256 = ...}` —
  where `file`/`table` point at a checked-in extract under `constants/sources/` carrying the cited
  value. Changing a constant's citation then changes a file whose content a reviewer can compare
  against the number. A bare `kind = "doi"` remains permitted but is counted separately and
  ratcheted like `uncited-legacy`, because a DOI is a claim, not evidence.

**Known limitation, stated.** An evasion corpus proves a detector catches *the evasions someone
thought of*. It cannot prove absence of others, and a sufficiently novel evasion still passes. The
claim here is bounded and deliberate: it converts "we believe this detector works" into "this
detector demonstrably catches these eleven specific defeats," and it makes each future defeat a
one-file addition rather than a redesign.

### D13 — Physical-constant tables get a checked-in snapshot

`test_alphabet_conformance.py` guards *ordering* declarations against silent copy-drift. Apply the
identical pattern to *numeric* declarations: every physical-constant table (atomic masses, GBSA
intrinsic Born radii, OBC2 scale factors, Cordero covalent radii, LJ defaults) emits its full
`(key, value)` set through a `snapshot()` function, and a test compares it against a checked-in
`constants/snapshots/<table>.json`.

Changing a value then requires editing the snapshot in the same commit, which makes it
**greppable, reviewable, and blame-able**. `git log -p constants/snapshots/gbsa_radii.json`
answers "did our Br radii ever change?" correctly and forever.

Crucially, the snapshot key set must be the table's *full domain* including what the default arm
covers, so that **adding a match arm changes the snapshot** — that is the precise event
`-fixed-initial` performed invisibly. Snapshot entries carry a `source` field (citation or
`"uncited-legacy"`); the count of `uncited-legacy` entries is a ratchet that may only decrease.

### D14 — Every physical-constant lookup returns `Sourced<T>`, never a bare number

Generalise `ICSource` one final time, to the narrowest and most numerous case — the constant
lookup itself:

```rust
pub enum ValueSource {
    Tabulated { citation: &'static str },   // ICSource::Charmm analogue
    Defaulted { rule: &'static str },       // ICSource::EnghHuberFallback analogue
    Untabulated,                            // ICSource::Missing analogue
}
pub struct Sourced<T> { pub value: T, pub source: ValueSource }

pub fn mbondi2_radius(element: &str) -> Sourced<f32>;
pub fn obc2_scale(element: &str) -> Sourced<f32>;
pub fn get_mass(element: &str) -> Sourced<f32>;      // supersedes the Option<f32> of D6
```

**`Sourced<T>` is per-lookup and is aggregated before any vector is built. It never appears as
`Vec<Sourced<f32>>`.** *(Review B8 — the earlier claim that `Sourced<f32>` is "one enum
discriminant wide, zero runtime cost" was wrong, and the correction is accepted in full.
`ValueSource::Tabulated { citation: &'static str }` carries a 16-byte fat pointer; with
discriminant and alignment that is ≥24 bytes against 4 — a 6-8× blowup on a 100k-atom vector, in
exactly the loops D8 exists to protect.)*

The required shape:

```rust
// per-lookup: Sourced<f32> lives in a register, is consumed immediately
for (i, name) in atom_names.iter().enumerate() {
    let Sourced { value, source } = mbondi2_radius(element_of(name));
    radii.push(value);                       // stays Vec<f32> — no layout change
    report.record_if_not_tabulated(i, source);   // provenance goes to the Report, not the vector
}
```

So the memory cost is `Vec<f32>` (unchanged) plus a `Report` whose size is proportional to the
number of *non-`Tabulated`* lookups — zero for the all-tabulated common case. The `citation`
pointer is never stored per atom; the report stores the element and table once per distinct
finding, deduplicated.

**Report growth is bounded and truncation is never silent.** `DiagPolicy.max_findings` was
previously undefined at the limit, which on a 100k-atom all-defaulted structure would produce a
truncated report — a well-typed plausible answer with the evidence destroyed, i.e. this
document's own failure mode. Overflow behaviour is now specified: on reaching `max_findings` the
report stops accumulating and records one terminal `PROX-DIAG-TRUNCATED` finding carrying the
number of findings dropped and the count by code. A `Report` may never lose a finding without
saying so.

This **supersedes D6's `Option<f32>`**: `Option` collapses `Defaulted` and `Untabulated` into one
`None`, which is the selenium distinction we just paid to learn. `Option` remains the return of
the *private* table probe; `Sourced` is the public API.

The rule this makes enforceable: **a `Defaulted` or `Untabulated` physical constant reaching an
energy calculation is an error by default** (OBS-113). Not a warning — the number is being
integrated into published physics and nobody knows if it is right. Lenient policy remains
available for viewers and exploratory work, and still records the finding.

**Rejected: "just make the defaults better"** (e.g. give Se its real 1.90 Å Bondi radius and move
on). Necessary, but it is a fix for one element, not a property. The next `MSE`-equivalent
arrives with the next ligand. Adding the citation is a *task* (Z2's retroactive item); making
absence visible is the *design*.

**Rejected: sentinel values** (`NaN`, `-1.0`, `f32::MIN`) to mark "unknown". This is
`DEFAULT_MASS = 12.0` with extra steps — a value that must be re-recognised by every consumer, and
that silently becomes data if any consumer forgets. A type-level discriminant cannot be forgotten.

### D16 — The Python surface breaks. Say so, version it, and do not leave a bare-float default

Review B4a is correct and forces a choice this document previously dodged. `oxidize.pyi:55-57`
declares:

```python
def assign_masses(atom_names: list[str]) -> list[float]: ...
def assign_mbondi2_radii(atom_names: list[str], bonds: list[tuple[int, int]]) -> list[float]: ...
def assign_obc2_scaling_factors(atom_names: list[str]) -> list[float]: ...
```

There is no Python representation of `Sourced<f32>`, and no way to thread a `DiagPolicy` through a
registered `#[pyfunction]` without changing its signature. "Existing signatures untouched" (D3) is
true of the *Rust* parser functions and **false at the pyo3 boundary**.

> **OWNER DECISION PENDING.** Everything in D16 is conditional on accepting the break. Dependents
> if it is rejected: **OBS-112** covers Rust only (and needs a `never` exemption naming the three
> `oxidize` symbols); **OBS-113(b)** becomes unimplementable and must be struck, leaving the GBSA
> path ungated at its only real boundary; **B7** is cut; **D17's boundary 2** does not exist. This
> is recorded here, in the decision, rather than only in §11's revision log — a reader of D16 must
> not have to find the caveat elsewhere.

**Decision (proposed): the Python API breaks, deliberately and once.** Leaving the default Python surface
returning bare floats would contradict OBS-112 for proxide's *primary consumer* — the exact
population the Cl and Se defects reached. The new shape:

```python
class Assignment(TypedDict):
    values: list[float]
    report: list[Finding]        # ErasedReport, one dict per finding
    all_tabulated: bool

def assign_masses(atom_names: list[str], *, strict: bool = True) -> Assignment: ...
```

- `strict=True` (default) raises `ProxideDiagnosticsError` carrying the report when any finding is
  `Error`-severity — this is `DiagPolicy::promote_at` at the FFI boundary, and `strict` is the
  keyword-argument form rather than a `DiagPolicy` object because pyo3 handles `bool` kwargs
  cleanly and a policy struct would need its own binding.
- The bare-list return is available for one deprecation cycle as `assign_masses_values()`,
  emitting a `DeprecationWarning`. It is **not** the default and does not survive to 0.2.
- `oxidize.pyi:55-57` and `src/proxide/__init__.py:29-30` are updated in the same task.

**`ErasedReport` reaching Python needs a converter, and one is named (review N5).** `serde` alone
covers wasm (`serde-wasm-bindgen` is present); pyo3 does not consume `serde`. Use `pythonize`
(serde → `PyObject`), added to `proxide_py`'s dependencies. Rejected: hand-written
`IntoPyObject` impls — one per finding kind, drifting from the Rust definition, which is the
duplication this document exists to prevent. `pythonize` derives from the same `serde` impls that
feed the sidecar and wasm, so all three surfaces cannot disagree.

### D17 — The refusal boundary is `md_params.rs`, not `gbsa.rs` (review B2, blocking)

The earlier draft's OBS-113 criterion — "parameterising a structure containing `MSE` fails by
default" — named a boundary that **does not exist in Rust**. Verified at `c24546a`:

- `assign_mbondi2_radii` (`proxide-physics/src/physics/gbsa.rs:24`) and
  `assign_obc2_scaling_factors` (`:93`) have **zero Rust callers**.
- Their only consumers are pyo3 wrappers (`proxide_py/src/py_chemistry.rs:130,137`), reached from
  `src/proxide/__init__.py:29-30`, `crates/proxide_rs/oxidize.pyi:56-57` and
  `scripts/check_energy.py:47-48`.
- **GBSA assignment is not part of `MDParameters` parameterisation.** The composition happens in
  Python.

So the gate has two real boundaries and needs both:

1. **Rust:** `parameterize_structure` in `proxide-physics/src/physics/md_params.rs`, whose
   `ParamOptions`/`ParamError` already carry the strictness pattern D3 generalises. `ParamError`
   joins D4's list and gains the `Diagnostics(..)` variant. This is where a defaulted *mass* or
   *nonbonded* constant refuses.
2. **Python:** the D16 `strict=True` boundary on `assign_mbondi2_radii` /
   `assign_obc2_scaling_factors`. This is where a defaulted *GBSA* constant refuses, because that
   is the only place those functions are ever called.

Composing GBSA into `MDParameters` on the Rust side would give a single boundary and is the
better end state, but it is a **feature port, not an observability change**, and is explicitly out
of scope (§8).

---

## 4. Requirements

The P-column is the correctness-payoff priority used for sequencing in §6.

**Every criterion has a literal command in §4.1.** The prose in the table says what is being
asserted; §4.1 says how a machine decides it. A requirement whose §4.1 entry is missing or is not
a runnable command is not a requirement and does not merge (OBS-001).

### Tier ∅ — cross-cutting meta-requirements (bind every other requirement)

| ID | Requirement | Acceptance criterion (verifiable) | P |
|---|---|---|---|
| OBS-001 | Every acceptance criterion in this spec is decidable by a machine against a fixed artefact — a named commit, tag, checked-in fixture, or checked-in corpus. No criterion may be satisfied by an agent's or author's assertion that the work was done. | `scripts/check_spec_criteria.py` parses §4.1, and fails if any requirement ID in §4 lacks a §4.1 entry, if any entry is not executable, or if any table cell contains a criterion with no corresponding command. **This requirement is exempt from OBS-002** — no historical defect corpus for spec criteria can exist, so no calibration case is possible; its own gate is self-application (running it against this document must pass). | **P0** |
| OBS-002 | Every detector introduced by this spec has a calibration case in `calibration/cases.toml` per D12: **recall 1.0** on the defective ref, and findings on the fixed ref **⊆ `exemptions.toml`** (D15). | Harness checks out each tagged ref into a temp worktree. First entry: tags `calibration/element-inference-defective` → `calibration/element-inference-fixed`. Ground truth derived mechanically from the diff, never hand-listed (D12). A detector with no calibration case does not merge. **Exempt: OBS-001's checker.** | **P0** |
| OBS-003 | Detector precision is budgeted, not just recall. Findings on `calibration/clean/` must be **zero**; measured recall and false-positive counts are written to `calibration/RECALL.md` by the harness. | Harness fails on any finding in the clean corpus and on any merged case below recall 1.0. `RECALL.md` is a generated artefact; CI fails if it is stale. *(Review N2: recall-only would pass a detector that flags every line.)* | P1 |
| OBS-004 | One exemption ledger, `exemptions.toml`, per D15: entries carry `site`, `detector`, **`expected_findings`**, `reason`, `owner`, **`expiry_kind`/`expiry_value`**. **Both directions fail** — a listed site yielding zero findings fails as stale; yielding more than `expected_findings` fails as an unreviewed addition. | `check_exemptions.py --strict` validates schema, typed expiry, and registry cross-membership **without running detectors**; `--with-scan` (the OBS-002 harness run, not a second traversal) evaluates staleness and counts. Semantics from `tests/test_element_inference_conformance.py` at `e78484e`. | **P0** |
| OBS-005 | Detectors cannot suppress their own findings (D15.1) and cannot be disabled by content they do not control (D15(b)). Every detector subclasses `scripts/detectors/base.Detector`, returns **unfiltered** findings, and is listed in `REGISTRY`; exemption application lives only in the harness. Any enable/exclude test runs on **comment- and string-stripped** source and requires a call or import, never token presence. | `check_exemptions.py --strict` asserts `REGISTRY` ↔ `exemptions.toml` ↔ §4.1 three-way consistency and fails on a detector that reads `exemptions.toml` directly (import check). **Evasion proof required:** `calibration/evasion/_shared/comment_disable/` — a defective file carrying `// TODO: route through infer_element` — must still be flagged. | **P0** |
| OBS-006 | Every detector ships a defeating-instance corpus at `calibration/evasion/<detector>/` per D18, authored adversarially (not by the detector's author), and every instance is flagged. | Harness fails if any registered detector has an empty evasion dir, or if any instance goes unflagged. Seeded with the eleven known defeats in D18's table. A new detector without evasion cases does not merge. | **P0** |

### Tier 1 — canonical diagnostics channel

| ID | Requirement | Acceptance criterion (verifiable) | P |
|---|---|---|---|
| OBS-101 | `proxide_core::diag` provides `Severity`, `DiagKind`, `Finding<K>`, `Report<K>`, `Reported<T,K>`, `DiagPolicy`, `ErasedReport`, `DiagnosticsError` per D1/D3. | `cargo test -p proxide-core diag::` passes; `Report::{is_clean,errors,warnings}` have tests mirroring `precondition.rs:332-564`. | P2 |
| OBS-102 | `confind::PreconditionReport` is a re-export/alias of `Report<ViolationKind>`; confind's public API is unchanged. | `cargo test -p proxide-confind` passes, **and** `git diff -U0 $(git merge-base HEAD main) -- crates/proxide-confind/src/precondition.rs` has no hunk whose `@@` header falls inside the `mod tests` line range. *(N6 as amended: `--stat` emits no line numbers, so it could not evaluate its own criterion — `-U0` plus hunk-header parsing can. `scripts/check_untouched_tests.py` does the parsing.)* | P2 |
| OBS-103 | `infer_element_sourced` exists per D5; `infer_element`'s two `"C"` fallbacks (`masses.rs:66,98`) return `ElementSource::Missing` through it. | Named test `chem::masses::tests::test_infer_element_sourced_reports_missing` asserts `infer_element_sourced("XX").value == None` with `.source == ElementSource::Missing`, `infer_element_sourced("").source == ElementSource::Missing`, and `infer_element_sourced("CL").source == ElementSource::TwoLetterTable`. | **P0** |
| OBS-104 | `DEFAULT_MASS` deleted; `get_mass -> Sourced<f32>` per D14; `assign_masses_reported` errors on `Untabulated` under the default policy. | `rg -w DEFAULT_MASS crates/` returns nothing; a test asserts `assign_masses_reported(&["XX"], &DiagPolicy::default())` is `Err` and that the lenient policy yields one `PROX-CHEM-UNKNOWN-ELEMENT` finding. | **P0** |
| OBS-105 | `parse_pdb_reported` / `parse_mmcif_reported` / `parse_pqr_reported` exist per D3; existing Rust `parse_*` signatures unchanged (the Python surface does change — D16). **Amended 2026-09-24 (owner ruling, option a):** a *blank* occupancy/B-factor field takes the documented default (1.0 / 0.0); a *non-blank* field that fails to parse is a hard error (malformed record, with line and column) under every policy — it is not a recoverable coercion. Recording blank-field defaults as findings, and any policy-governed recovery for malformed fields, is deferred to Tier-1 (debt #1911). | Named test `formats::pdb::tests::test_malformed_occupancy_is_an_error` against checked-in fixture `tests/data/coercion/malformed_occupancy.pdb`: `parse_pdb` returns `Err` naming the line and the occupancy field; companion test `test_blank_occupancy_takes_documented_default` on a blank-field fixture returns `Ok` with occupancy 1.0. Once Tier-1 lands (debt #1911), the blank case additionally yields exactly one `PROX-COERCE-FIELD-DEFAULTED` finding with `fields["field"] == "occupancy"`. | P2 |
| OBS-106 | Coercion triage: every `unwrap_or*` in `crates/proxide-io/src/formats/**`, `crates/proxide-core/src/chem/**`, **`crates/proxide-core/src/forcefield/**`** and **`crates/proxide-physics/src/physics/**`** is either routed through a recorded coercion or carries an `exemptions.toml` entry. | `scripts/check_coercions.py` exits non-zero on any unannotated site. Baseline: 36 non-test sites in proxide-io (§C1) plus the newly-scoped modules. **Must flag `forcefield/xml_parser.rs:166`** — `get_attr_opt(e, b"element").unwrap_or_default()` defaults a forcefield *element* to the empty string, and was outside the previous scope (review N7, accepted). | P2 |
| OBS-107 | The `_ =>` catch-alls in the OBS-106 scope are each either made exhaustive or carry an `exemptions.toml` entry. | Same script, same gate. Counts are re-derived by the script rather than asserted here (they shift as OBS-106 lands). | P3 |
| OBS-108 | `LoopModelReport::geometry_warnings: Vec<String>` becomes `Report<LoopDiagKind>`. | `rg 'geometry_warnings' crates/` returns no `Vec<String>` declaration; `cargo test -p proxide_fixer` passes. | P4 |
| OBS-109 | Each of the 6 io error enums gains `Diagnostics(..)` and `#[non_exhaustive]` per D4. | `cargo build --workspace` and `cargo hack check --feature-powerset` pass. | P2 |
| OBS-110 | `diagnostics/registry.toml` lists every code; codes are unique and never reused. Must include `PROX-DIAG-TRUNCATED` (D14). | `scripts/check_diagnostic_codes.py` compares registry against the codes emitted by `DiagKind::code()` impls and fails on unregistered or duplicate codes. **Dynamic construction is a violation:** the detector flags `format!("PROX-…{}", ..)` and any non-literal `code()` return, because a code assembled at runtime cannot be registered (D18 evasion case). | P2 |
| OBS-111 | Every physical-constant table exposes `snapshot()` and is guarded by a checked-in `constants/snapshots/<table>.json` per D13, keyed over the table's full domain including default-arm coverage, with a `source` field per entry. | `cargo test constants::snapshot` fails on any value change not accompanied by a snapshot edit. **Regression proof required:** replaying `54eefc2` against the guard must fail with the Br/I radii diff named. Ratchet: `scripts/check_uncited_constants.py` fails if the `uncited-legacy` count increases. | **P0** |
| OBS-112 | Every constant lookup on the **enumerated symbol list** below returns `Sourced<T>` per D14. **CONDITIONAL — scope depends on an unmade owner decision (D16).** As written this requirement covers **Rust symbols only**, and its §4.1 command checks only Rust. If the owner accepts D16 (Python surface breaks), OBS-112 additionally requires that `oxidize.assign_*` expose `source` per value, and B7 delivers it. **If the owner keeps `list[float]` as the Python default, OBS-112 does not hold for proxide's primary consumer, and that must be recorded as a `never`-kind exemption naming the three `oxidize` symbols — not left implicit.** A PASS on this requirement is not a claim about Python unless D16 is accepted. | `scripts/check_sourced_lookups.py` operates on an **explicit symbol list**, not on a return-type shape, and fails if a listed symbol's return type does not transitively contain `Sourced`. *(Review B3, accepted: a "bare `f32`/`f64` return" check would have passed both GBSA functions — they return `Vec<f32>` — i.e. green-lit the exact defect motivating the requirement; and it would have skipped `covalent_radius`, which is private and returns `Option<f64>`.)* Symbols: `masses::get_mass`, `masses::infer_element_sourced`, `gbsa::assign_mbondi2_radii` (`gbsa.rs:24`), `gbsa::assign_obc2_scaling_factors` (`gbsa.rs:93`), `geometry_gate::covalent_radius` (`geometry_gate.rs:6`, made `pub`), and each LJ/nonbonded default accessor. The list is the requirement; adding a constant table without adding its symbol fails OBS-301. | **P0** |
| OBS-112b | Selenium provenance is pinned so a future change to it is a visible test edit. | Named test asserting `mbondi2_radius("Se").source` is a specific variant, with the current value written literally in the assertion. Whichever way Z2b resolves Se, the resolution edits this test. | **P0** |
| OBS-113 | A `Defaulted` or `Untabulated` constant reaching a parameterisation path is an `Error`-severity finding under the default `DiagPolicy`; `DiagPolicy::lenient()` records it without promoting. **Two boundaries per D17**, because GBSA composition happens in Python. | **(a) Rust:** named test on `md_params::parameterize_structure` with `ParamOptions::default()` over a structure whose element is outside the tabulated domain → `Err(ParamError::Diagnostics(..))` carrying `PROX-CHEM-DEFAULTED-CONSTANT`. **The fixture must construct `AtomRecord` directly with an out-of-domain `ElementSymbol`, not parse a PDB** — per D6, every symbol `infer_element` can emit is inside the tabulated domain, so a parsed fixture provably cannot reach this path and the test would pass vacuously. **(b) Python — CONDITIONAL ON D16.** `pytest` test that `oxidize.assign_mbondi2_radii(["SE"], [])` raises `ProxideDiagnosticsError` naming `Se` under default `strict=True`, and with `strict=False` returns a result whose `report` contains that finding. **Neither behaviour is expressible by a bare `list[float]` return**, so (b) silently assumes D16 is accepted. If it is not, (b) is unimplementable and must be struck with a `never`-kind exemption, leaving the GBSA path — the §0.2 path — ungated at its only real boundary. *(Review B2: the earlier "parameterising a structure containing MSE fails" named a Rust boundary that does not exist.)* **(a) is unconditional; (b) is not, and the requirement passes on (a) alone only if (b)'s exemption exists.** | **P0** |

### Tier 2 — one connected telemetry channel

| ID | Requirement | Acceptance criterion (verifiable) | P |
|---|---|---|---|
| OBS-201 | `println!`/`eprintln!` are denied in library source. Set `[workspace.lints.clippy] print_stdout = "deny"`, `print_stderr = "deny"`; each crate adds `[lints] workspace = true`; binary crate roots carry an explicit `#![allow(clippy::print_stdout)]`. | **CI already runs `cargo clippy --workspace -- -D warnings`** (`ci.yml:117`) — zero new machinery. Gate passes only after the **24** library-source prints (§C2, corrected per review N1 — `proxide-wasm/src/gaff2.rs` is library source) are converted to `tracing`. | **P1** |
| OBS-202 | Every binary target installs a subscriber in `main()` before any work; no library crate installs one. | `scripts/check_subscribers.py` enumerates `src/bin/*.rs` + `[[bin]]` targets, asserts each contains `tracing_subscriber::` init, and asserts no library crate lists `tracing-subscriber` outside `[dev-dependencies]`. Must fix `convert_rotlib.rs` (confirmed missing, §C3) and `confind.rs`. | **P1** |
| OBS-203 | `tracing` in `[workspace.dependencies]` with the `log` feature per D7; `pyo3_log::init()` unchanged. | A Python test configures `logging` at DEBUG, parses a fixture that emits a coercion, and asserts ≥1 record carrying a `PROX-` code. | P2 |
| OBS-204 | Phase-boundary spans only; no `tracing::` macro in any module listed in `telemetry/hot_paths.toml`. | `scripts/check_hot_paths.py` greps the listed modules and fails on any `tracing::`/`#[instrument]` occurrence. Advisory secondary check: `cargo bench --workspace` shows no >3% regression on existing MD benches. | P2 |
| OBS-205 | **No workspace member's `Cargo.toml` names `release_max_level_*` at all** (D8, corrected per review B9). The feature is applied only when building a published release artefact, from outside the manifest. | `scripts/check_release_levels.py` asserts on **resolved** features via `cargo tree -f '{p} {f}' --workspace`, not manifest text. *(The earlier manifest-grep would have reported PASS while a binary member's feature poisoned every library in the same `resolver = "2"` build — the exact failure the risk row described.)* | P2 |
| OBS-206 | `proxide-wasm` exposes `init_logging(level: &str)` bridging `tracing` to `console.*`, plus `console_error_panic_hook`; idempotent. | `wasm-bindgen-test` asserts double-init does not panic and that a parse with a coercion produces ≥1 console record. **Requires adding a wasm test runner to CI** — today `ci.yml:120-124` only type-checks wasm. If the runner is not added, this requirement's gate is compile-only and the mechanism is explicitly *unattested*. | P4 |

### Tier 3 — provenance, parity attestation, chemical coverage

| ID | Requirement | Acceptance criterion (verifiable) | P |
|---|---|---|---|
| OBS-301 | Each table exposes its own supported domain (`supported_elements()` etc.) per D11, **and every constant table is itself registered** in `constants/registry.toml`. | `cargo test` asserts `supported_elements()` and the arms of `get_mass` agree in both directions (a table-driven `get_mass` makes this trivially true — preferred), **and** that `constants::REGISTRY` enumerates every table, cross-checked against OBS-112's symbol list. *(B3 residue: without a registry of tables, a new `fn dielectric(..) -> f32` in proxide-physics is invisible to OBS-112, OBS-301 and OBS-303b simultaneously — three P0 gates blind to the same omission. The registry is the single place a new table must appear, and OBS-112's symbol list is derived from it rather than maintained beside it.)* | **P0** |
| OBS-302 | `scripts/chem_coverage.py` extracts the corpus multiset to `coverage/corpus.json`. | Running it on the current corpus yields JSON whose `elements` contains at least `{C,N,O,S}`; committed as a snapshot. | **P0** |
| OBS-303 | Coverage gate: `claimed − covered ⊆ exemptions`, each gap justified with owner and expiry. | Fails when an element is added to any table without a fixture or an `exemptions.toml` entry. **Regression proof required:** a test that removes the `Cl` entry in a tmpdir copy of the ledger and asserts the gate fails. Element-scoped gaps are ledger rows (D15), not a separate file — the stale-check applies, so a gap entry for an element that later gains a fixture fails until deleted. | **P0** |
| OBS-303b | Cross-table domain consistency: the coverage tool compares each table's claimed domain against **every other table's**, and flags any element tabulated in one and defaulted in another. | Named test `test_cross_table_domains::test_flags_selenium` asserts the tool flags `Se` — present in `masses.rs:30`, absent from the mbondi2/obc2 tables (the §0.2 case) — and that the flag clears only via a citation or an `exemptions.toml` entry. This is the check that was mechanically available and unrun for the entire life of the selenium defect. **Depends on `gbsa::supported_elements()`, which is A1b's file** (review B6). | **P0** |
| OBS-303d | **`claimed` may not shrink.** `coverage/claimed.lock` records every table's claimed domain; growth is free, any shrink fails unless the lockfile is edited in the same commit with a reason. | Named test asserting the gate fails when `"Se" => 78.971` is deleted from `masses.rs` without a lockfile edit. *(Without this, OBS-303 and OBS-303b — two P0 gates — are both satisfiable by making the code **claim less**, which is the disease rather than the cure. D18.)* | **P0** |
| OBS-303c | The test corpus gains at least one fixture per element class the code claims: a halide-containing structure, a metal site, and a selenomethionine (`MSE`) structure. | `pytest tests/test_chemical_coverage.py::test_corpus_covers_claimed_classes`; `coverage/corpus.json` contains `Cl`, `Se`, and ≥1 transition metal. | P1 |
| OBS-304 | `ProvenanceRecord` per D9, with `git_sha` from a `build.rs` in proxide-core. | Test asserts `git_sha` matches `^[0-9a-f]{40}(-dirty)?$` or `== "unknown"`; `build.rs` emits `cargo:rerun-if-changed=.git/HEAD`. | P3 |
| OBS-305 | `MDParameters` and the primary parse results carry `provenance: ProvenanceRecord`; the structs become `#[non_exhaustive]` with builders. | `cargo build --workspace` passes; a test asserts a parameterisation run records the forcefield file's sha256 under `ArtifactRole::ForceField`. | P3 |
| OBS-306 | `write_provenance_sidecar(output_path, &record)` emits `<output>.prov.toml` with `schema = "bathos/0.3"` per D9. | Round-trip test: the emitted file parses as TOML and contains `schema`, `proxide_git_sha`, and ≥1 `input_* = <64 hex>` key. **Open item assigned to D3t: confirm against the live bathos ingest contract before locking key names** — the shape here is derived from an observed sidecar, not from bathos's published schema. | P4 |
| OBS-307 | `parity/<slug>.bth.toml` + generated `parity/INDEX.json` per D10; GAFF2's root file migrated; HP4-WASM/OpenMM, physics/MDTraj, trajectory-roundtrip retrofitted as entries. | `scripts/parity_index.py --check` fails if `INDEX.json` is stale or an `impl_paths` entry names a nonexistent file. ≥4 ledger entries exist. | P4 |
| OBS-308 | Consumer-visible query: `proxide.attestation.is_parity_attested(path) -> Verdict \| None`. | Python test asserts `is_parity_attested("src/proxide/chem/gaff2.py")` is not `None` and an unattested path returns `None`. | P4 |

### 4.1 Verify commands

`scripts/check_spec_criteria.py` parses this block. Every ID in §4 must appear exactly once; an ID
without an entry, or an entry that is not executable, fails OBS-001. Commands are run from the
repository root.

```verify
OBS-001:  uv run python scripts/check_spec_criteria.py .praxia/docs/specs/260910_proxide-silent-substitution-observability.md
OBS-002:  uv run pytest tests/test_detector_calibration.py -k recall
OBS-003:  uv run pytest tests/test_detector_calibration.py -k "precision or clean" && uv run python scripts/calibrate.py --check-recall-md
OBS-004:  uv run python scripts/check_exemptions.py --strict && uv run python scripts/check_exemptions.py --with-scan
OBS-005:  uv run python scripts/check_exemptions.py --strict --assert-registry && uv run pytest tests/test_detector_calibration.py -k evasion_shared
OBS-006:  uv run pytest tests/test_detector_calibration.py -k evasion
OBS-101:  cargo test -p proxide-core diag::
OBS-102:  cargo test -p proxide-confind && git diff -U0 $(git merge-base HEAD main) -- crates/proxide-confind/src/precondition.rs | uv run python scripts/check_untouched_tests.py --module tests
OBS-103:  cargo test -p proxide-core chem::masses::tests::test_infer_element_sourced_reports_missing
OBS-104:  cargo test -p proxide-core chem::masses::tests::test_unknown_element_mass_refuses && ! rg -qw DEFAULT_MASS crates/
OBS-105:  cargo test -p proxide-io formats::pdb::tests::test_malformed_occupancy_is_an_error formats::pdb::tests::test_blank_occupancy_takes_documented_default
OBS-106:  uv run python scripts/check_coercions.py
OBS-107:  uv run python scripts/check_coercions.py --catchalls
OBS-108:  cargo test -p proxide_fixer && ! rg -q 'geometry_warnings:\s*Vec<String>' crates/
OBS-109:  cargo build --workspace && cargo hack check --feature-powerset
OBS-110:  uv run python scripts/check_diagnostic_codes.py
OBS-111:  cargo test -p proxide-core constants::snapshot && uv run python scripts/check_uncited_constants.py && uv run python scripts/calibrate.py --case constant-drift
OBS-112:  uv run python scripts/check_sourced_lookups.py --symbols spec
OBS-112b: cargo test -p proxide-physics gbsa::tests::test_selenium_source_pinned
OBS-113:  cargo test -p proxide-physics md_params::tests::test_defaulted_constant_refuses && uv run pytest tests/test_gbsa_strictness.py
OBS-201:  cargo clippy --workspace -- -D warnings
OBS-202:  uv run python scripts/check_subscribers.py
OBS-203:  uv run pytest tests/test_python_logging_bridge.py
OBS-204:  uv run python scripts/check_hot_paths.py
OBS-205:  uv run python scripts/check_release_levels.py
OBS-206:  wasm-pack test --headless --firefox crates/proxide-wasm
OBS-301:  cargo test -p proxide-core constants::domain_agreement
OBS-302:  uv run python scripts/chem_coverage.py --check coverage/corpus.json
OBS-303:  uv run pytest tests/test_chemical_coverage.py::test_claimed_minus_covered_equals_gaps
OBS-303b: uv run pytest tests/test_chemical_coverage.py::test_cross_table_domains
OBS-303c: uv run pytest tests/test_chemical_coverage.py::test_corpus_covers_claimed_classes
OBS-303d: uv run pytest tests/test_chemical_coverage.py::test_claimed_domain_may_not_shrink
OBS-304:  cargo test -p proxide-core provenance::tests::test_git_sha_format
OBS-305:  cargo test -p proxide-physics md_params::tests::test_provenance_records_forcefield_sha
OBS-306:  cargo test -p proxide-core provenance::tests::test_sidecar_roundtrip
OBS-307:  uv run python scripts/parity_index.py --check
OBS-308:  uv run pytest tests/test_attestation.py
```

**Two of these are load-bearing beyond their own requirement.** `OBS-002`'s harness and
`OBS-004`'s ledger are invoked by every other detector command; a change to either re-runs the
whole block.

---

## 5. Fixer tasks and dependency order

Each task is one session for one fixer. `→` denotes a hard dependency.

**Wave ∅ — blocks every detector task. Build first.**

| Task | Scope | Files | Gate | ~LOC |
|---|---|---|---|---|
| **Z0** | Exemption ledger + detector base class (OBS-004, OBS-005) per D15/D15.1. Schema with `expected_findings` and typed expiry; `Detector` ABC returning **unfiltered** findings; `REGISTRY`; three-way registry ↔ ledger ↔ §4.1 assertion; comment/string-stripping helper. Port semantics from `tests/test_element_inference_conformance.py` at **`e78484e`** — including the stale-entry failure, the `expected_findings` bound, and the comment-stripped exclusion test. | `exemptions.toml`, `scripts/detectors/{base.py,__init__.py}`, `scripts/check_exemptions.py` (create) | `verify:OBS-004`, `verify:OBS-005`; tests that a listed-but-clean site fails, an over-count fails, and a comment cannot disable a detector | ~320 |
| **Z1** | Calibration + evasion harness (OBS-002, OBS-003, OBS-006). Temp-worktree checkout of a **tagged** ref pair; assert recall 1.0 on defective against the **reviewed checked-in** ground truth, `findings ⊆ exemptions` on fixed, zero on `calibration/clean/`, and **every `calibration/evasion/` instance flagged**; generate `RECALL.md`. Adds `fetch-depth: 0` to `ci.yml`. Seed the evasion corpus with D18's eleven known defeats. `→` Z0. | `calibration/{cases.toml,clean/,evasion/,ground_truth/}`, `tests/test_detector_calibration.py`, `scripts/calibrate.py` (create), `ci.yml` (modify) | `verify:OBS-002`, `verify:OBS-003`, `verify:OBS-006` — must fail against the 260910 guard-test heuristic (recall 2/4) and against each of the eleven evasions | ~420 |
| **Z2a** | Constant snapshots (OBS-111). Add `snapshot()` to each constant table; generate `constants/snapshots/*.json` **from `calibration/element-inference-fixed-initial`'s parent** so the Br/I change shows as a reviewable diff, not as baked-in truth; add the `uncited-legacy` ratchet and D18's structured `source` refs under `constants/sources/`. | `constants/snapshots/*.json`, `constants/sources/**` (create), `proxide-core/src/chem/*.rs`, `proxide-physics/src/**/gbsa.rs` (modify), `scripts/check_uncited_constants.py` (create) | `verify:OBS-111`; replaying `calibration/element-inference-fixed-initial` fails with Br/I named | ~260 |
| **Z2b** | Retroactive citations (below). Blocks nothing; blocked by Z2a. Chemistry sourcing, not code. | `constants/snapshots/*.json`, `exemptions.toml` (modify) | `verify:OBS-112b` pin-test updated in the same commit | ~60 |

**Dependencies.** Z0 `→` Z1 `→` every task shipping a detector (A2, A3, A4's checker, B5, and the
OBS-111 case). Z2a `→` A1, A1b. Z2b blocks nothing. *(Review B6: the earlier draft had Z2 both
blocking A1b in §5 and being explicitly non-blocking for it in §7's risk row. The split resolves
the contradiction — the **snapshot** must precede the code change; the **citation** must not.)*

**Calibration refs are tags, not raw SHAs (review B5).** Use
`calibration/element-inference-defective` and `calibration/element-inference-fixed` — annotated and
pushed, so they survive gc. `ci.yml` uses `actions/checkout@v4` with no `fetch-depth`, so a shallow
clone contains neither commit; adding `fetch-depth: 0` is **Z1's** scope, not A3's. The harness must
**error** on an unresolvable ref, never skip: a silent skip is a detector-of-detectors that always
passes, which is this document's failure mode applied to its own foundation.

**Retroactive items, assigned to Z2b** — leaving either open converts a documented incident into
permanent unattributed physics:

1. The Br/I radii introduced by `-fixed-initial` (already reverted in `-fixed`) are either cited
   to a reference and reinstated, or left reverted, in
   a commit that says which.
2. The five elements of §0.2 (`Cl`, `Na`, `Fe`, `Cu`, `Se`) get **cited** mbondi2 radii and obc2
   scale factors, or an explicit `Defaulted` marking plus an `exemptions.toml` entry. Selenium is
   the priority: `MSE` is common, and the current post-fix value (1.50 Å) is *further* from Bondi's
   ≈1.90 Å than the value the bug was supplying. Do not close this by quietly writing `1.90` — cite
   it, snapshot it, and let OBS-112b's pin-test record the change. **The `exemptions.toml` entry
   seeded by Z0 expires exactly here**, and its stale-check will fail the build if this lands
   without deleting it.

**Wave A — independently parallelisable once Wave ∅ lands.**

| Task | Scope | Files | Gate | ~LOC |
|---|---|---|---|---|
| **A1** | Element source + mass domain (OBS-103, OBS-104, OBS-301). Delete `DEFAULT_MASS`, table-drive `get_mass`, add `infer_element_sourced`, `supported_elements()`. Update the 3 io call sites + `py_chemistry.rs`. | `proxide-core/src/chem/masses.rs` (modify), `proxide-io/src/formats/{pdb,mmcif,pqr}.rs` (modify), `proxide_py/src/py_chemistry.rs` (modify) | `cargo test -p proxide-core -p proxide-io` + `rg -w DEFAULT_MASS crates/` empty | ~180 |
| **A1b** | `Sourced<T>` constant lookups + defaulted-constant gate (OBS-112, OBS-112b, OBS-113) per D14/D17. Convert masses, mbondi2 radii, obc2 scale, Cordero radii, LJ defaults; add `gbsa::supported_elements()`; make `covalent_radius` `pub`. Refusal lands at **both** boundaries: `ParamError::Diagnostics` in `md_params.rs` and `strict=True` on the two pyo3 GBSA wrappers. `→` **A1** (A1b converts the lookups A1 has just made table-driven; sharing `masses.rs` they must not run concurrently), Z2a, B1 (`Severity`/`DiagPolicy`), B3 (`ParamError` variant). *(Round-3 nit: the round-2 dependency row omitted A1 while the diagram showed `Z2a → A1 → A1b`; A1 is a prerequisite.)* | `proxide-core/src/chem/*.rs`, `proxide-physics/src/physics/{gbsa.rs,md_params.rs}`, `proxide-ligand-frame/src/geometry_gate.rs`, `proxide_py/src/py_chemistry.rs`, `crates/proxide_rs/oxidize.pyi` (modify), `scripts/check_sourced_lookups.py` (create) | `verify:OBS-112`, `verify:OBS-112b`, `verify:OBS-113` | ~340 |
| **A2** | Coverage extractor + gates (OBS-302, OBS-303, OBS-303b). `→` Z1 (ships a detector), **`→` A1b** for `gbsa::supported_elements()`, which OBS-303b's cross-table check reads (review B6 — previously in neither A2's files nor its dependencies). Write against a stub only if A1b is genuinely in flight. | `scripts/chem_coverage.py`, `coverage/corpus.json`, `tests/test_chemical_coverage.py` (create) | `verify:OBS-302`, `verify:OBS-303`, `verify:OBS-303b` + the Cl-removal regression proof | ~260 |
| **A3** | Subscriber rule (OBS-202, OBS-205). Fix `convert_rotlib.rs` and `confind.rs`; add the checker. | `proxide-rotlib/src/bin/convert_rotlib.rs`, `proxide-confind/src/bin/confind.rs`, `proxide-tmalign/src/bin/tmalign.rs`, `proxide-jaccard/src/bin/*.rs`, `proxide-wasm/src/bin/param_cli.rs` (modify), `scripts/check_subscribers.py` (create), `ci.yml` (modify) | `python scripts/check_subscribers.py` exits 0 | ~120 |
| **A4** | Print eradication (OBS-201). Convert the **24** library-source prints to `tracing`; add workspace lints; allow-list binary roots. | root `Cargo.toml` + 18 crate `Cargo.toml`s, `proxide_fixer/src/{repack,finder,loop_model}.rs`, `proxide-frag/src/search.rs`, `proxide-rotlib/src/geometry/charmm_ic.rs`, `proxide-wasm/src/gaff2.rs` (modify) | `verify:OBS-201` | ~150 |

**Wave A dependency summary.** `Z2a → A1 → A1b`; `A1b → A2`; `Z1 → {A2, A3, A4}` (each ships a
detector); `B1, B3 → A1b`; `A4 → A5`. A3 and A4 touch disjoint files (bins vs libs) except
`proxide-wasm`; give both to one fixer, or sequence A3→A4, if that conflict matters.

**A1b is not independent of Wave B.** It needs `DiagPolicy`/`Severity` from B1 and the
`ParamError::Diagnostics` variant from B3. §6's "irreducible core" is corrected accordingly — the
core pulls in B1 and B3, and saying otherwise (as the earlier draft did) hid two sessions of work.

**Wave B — core diagnostics; serialised on B1.**

| Task | Scope | Gate |
|---|---|---|
| **B1** | `proxide_core::diag` module (OBS-101, OBS-110) + registry + code checker. `→` nothing. | `cargo test -p proxide-core diag::`; `scripts/check_diagnostic_codes.py` |
| **B2** | Migrate confind onto `Report<ViolationKind>` (OBS-102). `→` B1. Proves the generalisation preserves the source design. | `cargo test -p proxide-confind` with unmodified `mod tests` |
| **B3** | Error-enum composition (OBS-109). `→` B1. Mechanical, **7** files — the 6 io enums plus `ParamError` (`md_params.rs:19`), which D17 puts on the critical path for A1b. | `verify:OBS-109` |
| **B4** | `*_reported` parsers + coercion recording (OBS-105) + the `AtomRecord.element` type migration (D5). `→` B1, B3, A1. Larger than the other B tasks because of the field-type change and its construction sites. | `verify:OBS-105` |
| **B5** | Coercion/catch-all triage + checker (OBS-106, OBS-107), scoped to include `forcefield/**` and `physics/**` (N7). `→` B4, Z0, Z1. | `verify:OBS-106`, `verify:OBS-107` |
| **B6** | `LoopModelReport` migration (OBS-108). `→` B1. Parallel with B2-B5. | `verify:OBS-108` |
| **B7** | **CONDITIONAL on the D16 owner decision — cut entirely if rejected.** Python surface (D16, N5): `Assignment` TypedDict, `strict=` kwarg, `pythonize` for `ErasedReport`, deprecation shims. Updates all in-tree consumers: `crates/proxide_rs/oxidize.pyi:55-57`, `crates/proxide_py/src/py_chemistry.rs:130,137`, `src/proxide/__init__.py:29-30`, **`scripts/check_energy.py:47-48`** (omitted from the round-2 file list). `→` A1b, B1. | `verify:OBS-113` (Python half), `verify:OBS-203` |

**Wave C — telemetry finish.**

| Task | Scope | Gate |
|---|---|---|
| **A5** | `tracing` workspace dep with `log` feature, phase-boundary spans, hot-path ban (OBS-203, OBS-204). `→` A4. | `scripts/check_hot_paths.py`; Python logging test |
| **C1** | wasm console bridge + CI wasm test runner (OBS-206). `→` A5. | `wasm-pack test --headless` in CI |

**Wave D — provenance and attestation.**

| Task | Scope | Gate |
|---|---|---|
| **D1t** | `ProvenanceRecord` + `build.rs` git SHA (OBS-304). `→` B1. | git-sha format test |
| **D2t** | Thread provenance onto `MDParameters` + parse results (OBS-305). `→` D1t, B4. | forcefield-sha test |
| **D3t** | Bathos sidecar emit + schema confirmation (OBS-306). `→` D1t. **First step is reading the live bathos sidecar contract**, not writing code. | TOML round-trip test |
| **D4t** | Parity ledger split + index + retrofits (OBS-307). `→` nothing. Fully parallel with everything. | `scripts/parity_index.py --check` |
| **D5t** | `is_parity_attested` consumer API (OBS-308). `→` D4t. | Python attestation test |

**Critical path (corrected, review B6).** The longest chain is **seven** sessions, not four:

```
Z0 → Z1 → ─┐
Z2a → A1 → ─┴→ A1b → A2          (needs B1, B3 before A1b)
B1 → B3 → B4 → B5
```

Fully expanded, the binding chain is `Z0 → Z1 → A1b → A2` interleaved with `B1 → B3 → A1b`, and
separately `B1 → B3 → B4 → B5`. The earlier draft's stated path (`B1 → B3 → B4 → B5`, four
sessions) omitted `Z2a → A1 → A1b`, which is longer once A1b's B1/B3 dependencies are counted.

Fully parallel with everything: **D4t** (parity ledger), **B6** (`LoopModelReport`), **A3**
(subscribers, after Z1), **A4** (prints, after Z1).

---

## 6. Sequencing by correctness payoff

**Load-bearing for correctness** (these prevent wrong physics):

1. **OBS-002/003 — detector calibration.** Promoted to first position by the 260910 incident.
   Every other gate in this document is a detector, and an uncalibrated detector that reports PASS
   is worse than no detector: it manufactures false assurance. The 260910 guard-test agent's
   heuristic had recall 2/4 and self-certified as passing. Build the harness before the detectors.
2. **OBS-112/113 — `Sourced<T>` constant lookups and the defaulted-constant gate.** The §0.2
   selenium case shows the defect *survives the bug fix*: post-fix, `MSE` gets a 1.50 Å radius that
   is further from truth than the accidentally-borrowed 1.80 Å, and nothing distinguishes "Se is
   untabulated" from "that is oxygen's radius." Making the default observable is the property; the
   default's accuracy is not.
3. **OBS-111 — constant-table snapshots.** `-fixed-initial` changed published physics under a commit
   message asserting it had not. Same defect class as `DEFAULT_MASS`, at table granularity, and
   the only requirement here that makes `git log` answer numeric-provenance questions correctly.
4. **OBS-103/104/301/302/303/303b — element source, mass domain, coverage gates.** The entire Cl
   class, plus the cross-table check (OBS-303b) that was mechanically available and unrun for the
   whole life of the selenium defect. These are the only requirements that would have *prevented*
   the incidents rather than reported them afterwards.
5. **OBS-105/106 — recorded coercions in parsers.** Turns `occupancy = 1.0` from a fact into an
   attributable guess.
6. **OBS-202 — subscriber installation.** Not cosmetic: `convert_rotlib` currently discards
   *warnings about IC fallbacks* in the tool whose output drift caused #820/#869. Its diagnostics
   already exist; nobody can see them.

**Hygiene** (real value, no correctness claim): OBS-201 print eradication, OBS-108
`LoopModelReport` migration, OBS-107 catch-alls, OBS-206 wasm bridge, OBS-110 code registry —
though the registry is hygiene that *enables* the P0 tests to be written precisely, so build it
early despite the label.

**Provenance** (OBS-304/305/306) is correctness-adjacent: it does not prevent a wrong number, it
makes a wrong number *diagnosable after publication*. That is worth a lot, and it is worth less
than the coverage gate.

**If only a third gets built, build exactly this:** Z0, Z1, Z2a, A1, A1b, A2, A3, A4 — plus **B1
and B3**, which A1b requires. Ten sessions: the exemption ledger, the calibration harness, the
constant snapshots, element provenance, `Sourced<T>` lookups and the defaulted-constant gate, the
chemical-coverage gates, the subscriber rule, the clippy print lints, and the two Wave-B tasks
those depend on. Three are script-only, and one (OBS-201) requires **zero new CI machinery**
because `cargo clippy --workspace -- -D warnings` already runs at `ci.yml:117`.

If even that is too much, the irreducible core is **Z0 → Z1 → Z2a**, then **B1 → B3 → A1b** — six
sessions.

*(Review B6, accepted: the earlier draft called "Z1, Z2, A1b" the irreducible core while A1b
silently depended on B1, and — per D17 — on `ParamError` from B3 as well. Three stated, six
actual. That was this document's own failure mode: a plausible number with the evidence of how it
was reached removed.)*

Z0/Z1/Z2a earn first position because they are the only items that have already caught a real
defect in this repository within the last 24 hours, and because every other requirement depends on
Z1 to be trustworthy at all. A1b earns its place because it is the one item addressing the
residual defect (§0.2) that the completed Tier 0 fix demonstrably does *not* remove.

**Cut, in this order:** OBS-206 (wasm bridge — needs a CI runner that does not exist),
OBS-306 (sidecar — depends on an unconfirmed external schema), OBS-307/308 (parity ledger —
valuable, but no active incident points at it), OBS-108 (`LoopModelReport` — one struct, one
crate, low traffic), OBS-107 (catch-alls — mostly benign).

---

## 7. Risks

| Risk | Blast radius | Mitigation |
|---|---|---|
| D4 adds an enum variant + `#[non_exhaustive]` to 6 public error enums; downstream research repos that match exhaustively break. | 6 enums; unknown number of external match sites. | Pre-1.0 alpha (`0.1.0-alpha.16`) — do it once, now, in a single commit, and announce the code registry as the stable surface. Rollback: revert the 6 files; `Report` and `*_reported` still work without the error variant, they just cannot promote to `Err`. |
| D9 adds a field to public result structs (`MDParameters`, parse results); struct-literal construction breaks. | Every construction site of those structs, in and out of tree. | `#[non_exhaustive]` + a `Default`-backed builder in the same commit; supply `..Default::default()` migration examples. Defer D2t if churn exceeds estimate — D1t alone is useful. |
| OBS-201 turns clippy `-D warnings` into a hard blocker mid-migration; CI red on main. | Whole workspace. | Land the 24 conversions **first**, the lint config **last**, in that order within A4. Verify locally with `cargo clippy --workspace -- -D warnings` before the lint commit. |
| `release_max_level_*` feature unification silently disables logging for all consumers. **Under `resolver = "2"` this crosses workspace members**, so a binary member poisons every library in the same build. | Entire dependency graph, silently. | OBS-205 asserts on **resolved** features (`cargo tree -f '{p} {f}' --workspace`), not manifest text, and forbids the feature in *all* members. *(The earlier manifest-grep mitigation could not detect the failure it was written to mitigate — review B9.)* |
| The element-keyed gap view becomes a rubber stamp — every new element gets an entry instead of a fixture. | Defeats OBS-303 entirely. | Entries are `exemptions.toml` rows (D15) with mandatory `owner`/`expiry` and the stale-check; a check fails if the ledger grows in a PR with no corresponding fixture addition. Review entries at each release. |
| **The Python API break lands badly** — D16 changes three `oxidize` return types from `list[float]` to a TypedDict, and `src/proxide/__init__.py:29-30` plus `scripts/check_energy.py:47-48` are in-tree consumers. | Every Python consumer of proxide, i.e. the primary consumer. | Ship `assign_*_values()` bare-list shims with `DeprecationWarning` for one cycle; update the three in-tree call sites in B7 itself; `oxidize.pyi` is edited in the same commit so type-checkers flag downstream breaks at CI time rather than runtime. Rollback: the shims are the old behaviour, so reverting is a one-line default swap. |
| Bathos sidecar key names guessed from one observed file rather than the schema. | OBS-306 only. | D3t's first step is reading the live contract; the requirement explicitly flags the schema as unconfirmed. Do not let D3t start with code. |
| Adding `tracing` to 18 crates measurably slows compile/CI. | Build times. | `default-features = false`; only `std`, `attributes`, `log`. `tracing-subscriber` stays dev/bin-only, which is where the compile cost actually lives. |
| `Report<K>` generics leak into pyo3/wasm signatures and cause a type-parameter explosion. | proxide_py, proxide-wasm. | D1's `ErasedReport` boundary is mandatory, not optional: `serde` derives exist only on the erased form, so generics structurally cannot cross the FFI line. |
| A fixer self-certifies a detector as passing when it does not (observed twice on 260910). | Every gate in this document. | OBS-002: no detector merges without a calibration case whose recall is machine-measured. OBS-001: no criterion is satisfiable by assertion, and §4.1 gives every one a literal command. Reviewers reject "the scan passed" without a `RECALL.md` diff. |
| The calibration corpus itself is gamed — cases chosen to be easy, or ground truth trimmed to whatever the detector happens to catch. | OBS-002 becomes theatre. | Ground truth is **reviewed and append-only** (a case may gain sites, never lose them; CI-checked), and is compensated by **D18's adversarially-authored evasion corpus**, which contains instances the fix commit never contained. Mechanical derivation was tried and dropped as unimplementable — §11.1 item 3. |
| A detector suppresses its own findings, or is disabled by a comment. | Every gate; and the silence reads as *success*. | D15.1/OBS-005: detectors return unfiltered findings and the harness alone applies the ledger, so suppression has no code path; exclusion tests run on comment-stripped source and require a call or import. Evasion proof `calibration/evasion/_shared/comment_disable/` is mandatory. **Both holes were live in the reference implementation until `e78484e`.** |
| A detector is simply weak — it reports nothing where there is something to report. | OBS-005 does not catch this. | D18's evasion corpus (OBS-006) is the only thing that does, and it catches only the evasions someone thought of. Stated as known limitation §11.1(1)-(2); OBS-005 and OBS-006 are load-bearing only together. |
| **Numeric-accuracy regression is already shipped.** Tier 0's fix moved Se from 1.80 Å (borrowed from S, ≈0.10 off Bondi) to 1.50 Å (default, ≈0.40 off). `MSE` is common in the PDB, so real GBSA energies are now measurably worse than before the "fix". | Every solvation calculation on a selenomethionine structure since `54eefc2`. | **Z2b** (retroactive item 2): cite and snapshot Se's real radius. **Interim:** OBS-113 makes the affected path *fail loudly* rather than continue quietly — a refused calculation is recoverable, a published one is not. **A1b (the gate) depends on Z2a (the snapshot) but NOT on Z2b (the correct number)** — that is exactly why Z2 was split; the gate is the more urgent of the two and must not wait on chemistry sourcing. *(The earlier draft asserted this in prose while §5 encoded the opposite dependency — review B6.)* |
| Making `Defaulted` constants an error by default breaks currently-working user pipelines that process metal sites, halides or `MSE`. | Every downstream consumer parameterising non-C/N/O/S structures. | This is intended and is the point of OBS-113 — those pipelines have been producing unattributed numbers. Mitigation is a documented one-line lenient policy plus a `PROX-CHEM-DEFAULTED-CONSTANT` code they can assert on, not a softer default. Announce with the release; land A1b and OBS-303c's fixtures together so the failure is demonstrable rather than surprising. |
| Snapshotting constants from the *current* tree bakes unreviewed values in as canonical. | GBSA physics, permanently. | Z2a generates snapshots from `calibration/element-inference-fixed-initial`'s **parent**, so any change since appears as a reviewable diff; the cite-or-revert decision is Z2b, which cannot be closed by silence because OBS-112b's pin-test must be edited to close it. |
| The exemption ledger becomes a dumping ground — every finding gets an entry instead of a fix. | D15 becomes theatre. | Typed expiry is mandatory; the stale-check fails when a listed site stops violating; `expected_findings` fails an over-count as an unreviewed addition, so an entry cannot silently absorb tomorrow's violation. `--strict` reports ledger size per detector and lists `never`-kind entries separately. Ledger *growth* is a weaker gate than the stale-check and is labelled as such (§11.1 limitation 6). |
| The calibration harness silently skips when a ref is unresolvable (shallow clone, missing tag), reporting green. | Z1, therefore everything. | The harness **errors** on an unresolvable ref, never skips (B5). `fetch-depth: 0` is in Z1's scope. A test asserts the harness fails on a deliberately bogus ref name. |
| `uncited-legacy` ratchet stalls at a large number and is ignored. | OBS-111 loses teeth over time. | The ratchet only forbids *increase*; pair it with a per-release review of the top entries by blast radius. This is honestly a weak gate — it prevents regression, not remediation, and is labelled as such. |
| Tier 0 lands concurrently and conflicts with A1 in the same three parser files. | `pdb.rs`, `mmcif.rs`, `pqr.rs` — currently dirty on `fix/parameterize-solvent-atoms`. | A1 rebases onto completed Tier 0; it does not run beside it. Sequence: Tier 0 merges → A1 starts. `infer_element` stays as a deprecated wrapper (D5) so Tier 0's call sites keep compiling. |

---

## 8. Deliberately not specified

- **Tier 0** (element call-site consolidation + conformance guard) — in flight elsewhere. §2
  records only the carve-out it does *not* close.
- **OpenTelemetry / distributed trace export, metrics, counters.** proxide is a library called
  from a Python process and a browser; there is no service to trace. Adds dependency weight and a
  subscriber-configuration surface for zero present benefit.
- **A structured JSON log format or log-schema versioning.** The stable contract is the
  *diagnostic code registry* (D2), not the log line. Consumers assert on codes, not on log text.
- **Unifying the ~27 error enums into one error type.** Orthogonal refactor, enormous churn, and
  `thiserror`-per-crate is the right shape. D4 composes with them instead.
- **Python-side `warnings.warn` integration.** `pyo3_log` already exists and works; a second
  consumer-facing channel reintroduces the "two mechanisms" disease this spec exists to cure.
- **Numeric performance budgets for tracing** (e.g. "<1% overhead"). No baseline exists, so any
  number would be invented and the check flaky. OBS-204 gates on a *structural* property (no
  macros in hot modules), which is decidable.
- **Retrofitting provenance into the serialized rotlib protobuf schema.** A schema migration with
  its own compatibility story; `ProvenanceRecord` covers the in-memory and sidecar paths only.
- **Reducing the 76 test-file prints.** Out of scope by §C2's reasoning.
- **Fixing every one of the 36 `unwrap_or` sites.** §C1 establishes most are correct spec
  behaviour. OBS-106 requires triage and annotation, not elimination.
- **Determining the correct value for every defaulted physical constant.** Only Se, Br, I, Cl, Na,
  Fe, Cu — the seven with a documented incident — are assigned to Z2. The rest are recorded as
  `Defaulted` and surfaced by OBS-113; sourcing them is ongoing chemistry work, not a
  precondition for this spec. Per §0.2, making them visible is the deliverable; making them right
  is a separate, unbounded backlog.
- **Auditing published results computed before this spec lands.** The Se/Br/I history means some
  already-published GBSA numbers used borrowed or defaulted radii. Establishing which is a
  scientific-integrity task with its own scope and reviewers; it is named here so it is not lost,
  and explicitly excluded from these tasks.
- **Porting GBSA composition into Rust `MDParameters`.** Per D17, `assign_mbondi2_radii` and
  `assign_obc2_scaling_factors` have no Rust callers and are composed in Python. Unifying that
  would give one refusal boundary instead of two and is the better end state — but it is a feature
  port, not an observability change, and doing it inside this spec would hide a functional
  migration inside a diagnostics one. OBS-113 gates both boundaries as they actually exist today.
- **Whether `Severity::Coercion` should be user-configurable per code.** A per-code severity
  override table is an obvious next step once the registry (D2) exists; specifying it now would be
  designing against no observed demand.

---

## 9. References

- `crates/proxide-confind/src/precondition.rs:40-75` — `PreconditionReport`, source design for D1
- `crates/proxide-rotlib/src/geometry/ic_validate.rs:83-226` — `ICSource`, source design for D5/D9
- `crates/proxide-physics/src/physics/md_params.rs:19-34,122-157,522-534` — `ParamOptions::strict`
  + `unparameterized_atoms`, source design for D3 (commit `c23eeea`)
- `crates/proxide-ligand-frame/src/geometry_gate.rs:6-20` — refuse-rather-than-default precedent
- `crates/proxide-core/src/chem/masses.rs:7,31,63-100` — `DEFAULT_MASS`, the `"C"` fallbacks
- `crates/proxide_fixer/src/loop_model.rs:50-56` — `LoopModelReport`, the reinvention to retire
- `tests/test_alphabet_conformance.py` — cross-declaration drift-guard pattern for D11
- `parity.bth.toml`, `scripts/analysis/extract_rotlib_geometry.bth.toml` — parity ledger and
  bathos 0.3 `[result_schema.provenance]` shape for D9/D10
- `crates/proxide-io/src/formats/xtc.rs:151,189-241,283` — `.offsets` sidecar precedent
- `crates/proxide_py/src/lib.rs:80` — `pyo3_log::init()`, the consumer bridge D7 must preserve
- `.github/workflows/ci.yml:114-136` — the existing gates new checks attach to
- Tags `calibration/element-inference-defective` (`5f368ec`) → `calibration/element-inference-fixed`
  (`992377d`) — the calibration corpus's first case (D12) and the source of §0.1/§0.2. Use the
  tags, not the SHAs (review B5).
- `tests/test_element_inference_conformance.py` at **`e78484e`** — `DEFERRED_VIOLATIONS`, the
  in-repo reference implementation of D15's exemption ledger. Use `e78484e`, **not** `76625f8`:
  the earlier revision had both bugs D15(a)/(b) now forbid — prefix-blanket exemptions and a
  comment-disableable whole-file exclusion whose silence read as "fixed"
- `crates/proxide-physics/src/physics/gbsa.rs:24,93` — the two `Vec<f32>`-returning functions with
  zero Rust callers that drive D17 and OBS-112's symbol-list design
- `crates/proxide_rs/oxidize.pyi:55-57`, `crates/proxide_py/src/py_chemistry.rs:130,137`,
  `src/proxide/__init__.py:29-30`, `scripts/check_energy.py:47-48` — the real GBSA call chain (D16/D17)
- `crates/proxide-core/src/forcefield/xml_parser.rs:166` — forcefield element defaulted to `""`,
  the site that motivated widening OBS-106's scope (review N7)

---

## 10. Verification of this document's own claims

Per OBS-001, the factual claims here were measured, not assumed. **All commands are pinned to
`c24546a`** — run `git checkout c24546a` first, or prefix each with `git -c ... show`. The original
measurements were taken against a dirty tree (`pdb.rs`, `mmcif.rs`, `masses.rs` uncommitted), which
made every cited line number unstable; that is corrected here (review B7).

| Claim | Command (at `c24546a`) |
|---|---|
| §C1 `unwrap_or` counts (28 / 39 / 36) | `rg -c 'unwrap_or\(' crates/proxide-io/src` and `rg -c 'unwrap_or(_else\|_default)?\(' crates/proxide-io/src` |
| §C2 print counts (118 / 42 / 76; library 24, binary 18) | `rg -c '\b(println!\|eprintln!)' --glob '**/*.rs' .` |
| §C3 `convert_rotlib` has no subscriber | `rg 'tracing_subscriber\|env_logger\|subscriber\|logger' crates/proxide-rotlib/src/bin/convert_rotlib.rs` → no matches |
| §2 `infer_element` still defaults to carbon | `sed -n '66p;98p' crates/proxide-core/src/chem/masses.rs` |
| §D6 all 18 inferable symbols are inside `get_mass`'s 19-entry domain | `sed -n '10,33p;63,100p' crates/proxide-core/src/chem/masses.rs` |
| §D17 GBSA functions have zero Rust callers | `rg -n 'assign_mbondi2_radii\|assign_obc2_scaling_factors' --glob '*.rs' crates/` → definitions + pyo3 wrappers only |
| §0.1 / §0.2 GBSA value changes | `git diff calibration/element-inference-defective..calibration/element-inference-fixed -- '*gbsa*'` |
| §N7 forcefield element defaulted to `""` | `sed -n '166p' crates/proxide-core/src/forcefield/xml_parser.rs` |

**Counts will drift as this spec's own tasks land.** The baseline SHA is what lets a reviewer tell
drift from error; a mismatch at `c24546a` is a defect in this document, a mismatch at `HEAD` may
simply be progress.

Two of the brief's premises did not survive (§C1, §C2), and adversarial review found a third error
of the same kind in this document (§C2's library/binary split, review N1). A reviewer should assume
the same failure rate still applies and check before building.

---

## 11. Revision log — adversarial review 260910

Challenger returned `not_ready`; defender returned `needs_revision`; the coordinator adjudicated.
Both reviewers independently re-ran every §10 command and reproduced every result, and verified all
five in-repo precedents at the cited lines. What follows records what changed and what did not.

**Accepted and applied (blocking).**

| # | Objection | Resolution | Sections |
|---|---|---|---|
| B1 | No exemption mechanism; three incompatible partial ones; OBS-002 unsatisfiable while the GBSA fix is deliberately unmerged | New **D15** exemption ledger with mandatory expiry and stale-entry failure, adopting `DEFERRED_VIOLATIONS` semantics *(round 3: from `e78484e`, not `76625f8`)*; OBS-002 restated as `findings ⊆ exemptions`; new **OBS-004**; new task **Z0**. *(Round 3 found this only partially closed — see §11.1 item 1.)* | D15, OBS-002/004, Z0 |
| B2 | OBS-113 gated a Rust boundary that does not exist | New **D17**: two real boundaries — `md_params.rs`/`ParamError` in Rust, `strict=` on the two pyo3 GBSA wrappers in Python; OBS-113 split (a)/(b) | D17, OBS-113, A1b, B7 |
| B3 | OBS-112's "bare float return" check would pass both GBSA functions (`Vec<f32>`) and skip `covalent_radius` (private, `Option<f64>`) | Check restated over an **enumerated symbol list**, spelled out in the requirement | OBS-112 |
| B4 | "Signatures untouched" false at the pyo3 boundary and for `AtomRecord.element` | New **D16** (Python API breaks deliberately, versioned, with shims); D5 gains the field-type migration table | D5, D16, B7 |
| B5 | Calibration SHAs unreachable in CI's shallow clone; silent-skip hazard | Tags instead of SHAs; `fetch-depth: 0` moved into Z1's scope; harness must **error**, never skip | Z1, §7, §9 |
| B6 | `A1b → Z2` in §5 vs "do not let A1b wait on Z2" in §7; "irreducible core" hid B1/B3; critical path understated | Z2 split into **Z2a** (snapshots, blocking) and **Z2b** (citations, non-blocking); `A2 → A1b` added; critical path corrected to seven sessions; core corrected from three to six | §5, §6, §7 |
| B7 | Spec fails its own OBS-001; measurements taken against a dirty tree | New **§4.1** with a literal command per requirement ID; reviewer-gate sentence removed from OBS-001; OBS-001 explicitly exempted from OBS-002; all measurements re-pinned to `c24546a` | OBS-001, §4.1, §10 |
| B8 | `Sourced<f32>` cost claim arithmetically wrong; `max_findings` overflow undefined | Claim corrected (≥24 bytes vs 4); `Sourced` specified as **per-lookup, aggregated before the vector**, so `Vec<f32>` stays `Vec<f32>`; overflow defined as a recorded `PROX-DIAG-TRUNCATED` finding | D14 |
| B9 | `release_max_level` rule unenforceable — `resolver = "2"` unifies across workspace members; OBS-205's grep misses it | Feature forbidden in **all** members; checker asserts on resolved features via `cargo tree -f '{p} {f}'` | D8, OBS-205, §7 |

**Accepted and applied (non-blocking).** N1 library/binary print split corrected to 24 + 18 = 42
(§C2, OBS-201, A4). N2 false-positive budget and a `calibration/clean/` corpus added (OBS-003).
N3 ground truth derivation *(superseded in round 3 — derivation dropped, see §11.1 item 3)*, and the inconsistent six/four/three
violation counts removed rather than reconciled (D12, Z1). N4 lenient policy specified as
`DiagPolicy::lenient()` with `Severity::Never`, a lenient `Untabulated` mass defined as `f32::NAN`
*(the NaN half superseded by the owner ruling — see §11.2)*,
and the `Option<f32>`/`Sourced<f32>` spelling unified (D6). N5 `pythonize` named as the
`ErasedReport` → Python converter (D16, B7). N6 OBS-102 restated as a `git diff --stat` assertion
against the merge base. N7 OBS-106 widened to `forcefield/**` and `physics/**`, with
`xml_parser.rs:166` named as a must-flag site.

**Rebutted upstream, recorded so it is not re-litigated.** D6's blast radius: all 18 symbols
`infer_element` can emit are inside `get_mass`'s 19-entry domain, so `masses.rs:31` is unreachable
through `assign_masses` — deleting `DEFAULT_MASS` is cheaper than the earlier draft implied, and
D6 now says so. D14's proportionality (~6 lookup sites, not ~60). D1's wasm-side erasure
(`serde-wasm-bindgen` present; the Python side genuinely was missing — see N5). "D14 contradicts
D8" — a return type trips neither OBS-204's grep nor D8's macro ban, though B8's underlying
performance concern was real and is fixed.

**Contested: none.** Every ruling above is accepted as stated. Two are worth flagging as
*accepted with residual risk* rather than disputed:

1. **D16's Python break.** The ruling forces a choice and I chose "break it." That is the right
   call for correctness and the wrong call for anyone with a pinned proxide in a running pipeline.
   The shims and `oxidize.pyi` update reduce but do not eliminate this. If the repository owner
   would rather keep `list[float]` as the default surface, the honest consequence is that OBS-112
   does not hold for Python and that must be written into the spec rather than left implicit —
   this document should not silently claim a property its primary consumer does not have.
2. **OBS-113 breaking working pipelines** (already a risk row). The gate is deliberately louder
   than the status quo, and the §0.2 argument is that quiet wrongness is worse. That argument is
   strong but it is not free, and a maintainer could reasonably want the first release to ship
   `lenient()` as the default with a migration window. The spec takes the stricter position; the
   weaker one is a one-line change to `DiagPolicy::default()` and is not a redesign.
   **↑ This last sentence is wrong; corrected in round 3 — see §11.1 item 5(b) and the boxed note
   in D6.** Both owner-pending decisions now live in the requirements and decisions they affect,
   not only here.

---

## 11.1 Revision log — adversarial review round 3 (final)

Round-2 closure audit: six of nine blocking objections closed, three partial. Scoped to five
items. All five applied; two produced deliberate retreats, recorded as such.

**Reference implementation fixed upstream (`e78484e`) and now inherited explicitly, not by
reference.** Two real bugs were found in the very artefact D15 tells implementers to copy:
a whole-file exclusion tested against **raw** source, so a comment could disable the detector for
an entire file — and the resulting silence read as *fixed* to the stale-exemption check, retiring
the exemption and leaving the file permanently green and permanently defective; and exemptions
matched by path prefix, blanketing a file so tomorrow's violation would inherit today's
justification. D15 now states both as normative properties (a) counted-not-blanket via mandatory
`expected_findings`, and (b) a detector must not be disableable by content it does not control,
generalised to: **a detector whose silence is indistinguishable from its success is not a
detector** — the `DEFAULT_MASS` defect relocated into the tooling.

| # | Item | Resolution |
|---|---|---|
| 1 | "Consumed by every detector" was prose with nothing checking it; `expiry` was unevaluable prose; staleness circular with OBS-002 | New **D15.1** + **OBS-005**. Detectors subclass `Detector`, return **unfiltered** findings; the harness alone applies the ledger, so suppression is structurally impossible. `REGISTRY` ↔ ledger ↔ §4.1 three-way assertion. `expiry_kind`/`expiry_value` typed (`blocking_id`\|`date`\|`never`). Circularity resolved: `--strict` needs no detectors, `--with-scan` **is** the OBS-002 run, not a second traversal |
| 2 | Eleven of fifteen detectors defeated by a one-line change | New **D18** + **OBS-006**: every detector ships `calibration/evasion/<detector>/`, authored adversarially, all instances must be flagged. Two needed structural fixes because the evasion is a *legitimate edit*: **OBS-303d** ratchets `claimed` against shrinking (deleting `"Se" => 78.971` turned both P0 coverage gates green), and OBS-111's `source` becomes diff-reviewable structured evidence under `constants/sources/` rather than a typed string |
| 3 | `--derive-ground-truth` unimplementable | **Dropped.** Ground truth is a reviewed, checked-in, append-only list plus the D18 evasion corpus. Two reasons, both from this document's own §0.1: the fix commit carries unrelated Br/I edits so naive derivation makes recall 1.0 unreachable on the only seeded case; and derivation cannot find a site the fix commit missed — N3's fallacy automated rather than removed |
| 4 | `54eefc2` leaked as a raw SHA in five places and is **not** `992377d` | Both tagged. `-fixed-initial` (pre-revert) vs `-fixed` (post-revert); §0.1 carries a boxed note saying which requirements mean which. OBS-111/Z2a use `-fixed-initial`'s parent as the snapshot baseline |
| 5a | OBS-112's Python caveat lived only in the revision log; OBS-113(b) silently assumed the break | Conditionality moved into **OBS-112**, **OBS-113(b)**, **B7** and **D16** itself (boxed OWNER DECISION PENDING). If D16 is rejected: OBS-112 needs a `never` exemption naming the three `oxidize` symbols, and OBS-113(b) is struck, leaving the §0.2 path ungated at its only real boundary |
| 5b | "One-line change to `DiagPolicy::default()`" was inaccurate | Corrected at the decision point (D6). Lenient-by-default makes **`f32::NAN` the default** for an unknown element reaching MD integration — promoting exactly what D6's own rejection paragraph argues against — and falsifies OBS-104 verbatim plus the "default `DiagPolicy`" clauses in OBS-113 and D16. Four requirement edits and a new design decision, not one line. `DiagPolicy::default()` is `strict()` with **`max_findings = 1024`**, now specified |

**Non-blocking, folded in.** B3 residue → OBS-301 gains `constants/registry.toml`, from which
OBS-112's symbol list is *derived* rather than maintained beside it (a new
`fn dielectric(..) -> f32` was previously invisible to three P0 gates at once). OBS-102 → `-U0`
plus hunk-header parsing. B7 → `scripts/check_energy.py:47-48` added. A1b → `A1` stated as a
prerequisite. OBS-113(a) → fixture must construct `AtomRecord` directly, since per D6 no parsed
input can reach that path. `PROX-DIAG-TRUNCATED` registered under OBS-110, which also now flags
dynamically-constructed codes.

### Known limitations, stated rather than papered

The review closes here, so these are the residue and they are deliberate:

1. **The evasion corpus proves a detector catches the evasions someone thought of.** It cannot
   prove absence of others. The claim is bounded: "demonstrably catches these eleven specific
   defeats," and each future defeat is a one-file addition rather than a redesign. A sufficiently
   novel evasion still passes.
2. **D15.1 makes detector-side suppression impossible; it does not make a detector strong.** A
   `scan()` that simply never reports on `gbsa.rs` is indistinguishable from one with nothing to
   report. OBS-005 and OBS-006 are load-bearing only together; neither alone closes the hole.
3. **Ground truth is now openly incomplete** (item 3). Recall 1.0 means "against the reviewed list
   plus the evasion corpus," not "against all defects present." This is weaker than round 2
   claimed and stronger than round 2 could actually deliver.
4. **Two owner decisions are unmade** (D16, and strict-vs-lenient default). Each has its dependents
   enumerated at the decision point. Neither can be closed by a fixer.
5. **OBS-206's wasm gate remains compile-only** unless a wasm test runner is added to CI — carried
   unchanged from round 1 and still true.
6. **The `never` expiry kind is an escape hatch.** It requires a second owner and separate
   reporting, but a determined maintainer can mark anything permanent. That is intentional: the
   mechanism makes permanence *visible and attributable*, which is the most a ledger can do.

## 11.2 Owner ruling — 2026-09-10 (closes decision 2 of 2)

> *"we want to fail fast and loud, before NaN is filled an error should be raised"*

**Ruled: `DiagPolicy::default()` is `strict()`, and no policy may fill `f32::NAN` for an
undeterminable physical constant. The error is raised at the point of determination.**

This closes the second of the two decisions §11 recorded as owner-pending, and closes it wider than
it was posed. The open question was strict-vs-lenient *as the default*. The ruling also removes the
lenient `f32::NAN` return that D6 had specified as a legitimate escape hatch for an explicitly
opted-in caller — so `lenient()` now governs promotion of *recoverable* diagnostics only, and may
never fabricate a physical constant.

What changed in the document: D6's lenient-`Untabulated`-is-NaN paragraph is replaced by refusal
with a recorded finding; the OWNER DECISION PENDING block at D6 becomes a ruling; and the
conditional markers this ruling resolves are now unconditional — OBS-104's `Err`-under-default
criterion, the "under the default `DiagPolicy`" clauses in OBS-113 and D16, and D16's
`strict: bool = True`. The `Rejected: substituting f32::NAN` note at D6 and the `Rejected: sentinel
values` note at D14 were already correct and now apply without exception.

What this ruling *creates*: any consumer that needs to render a structure containing an
undeterminable constant must handle the error rather than receive a renderable non-number. The wasm
viewer is the only such consumer today (D16, OBS-206). That is deliberate — what to draw for an
unknown atom is a rendering decision, and it belongs in the renderer rather than travelling through
the physics layer disguised as a mass.

Decision 1 of 2 — whether the Python surface breaks to carry provenance (D16) — **remains open**.

The rationale is recorded as hard rules and ledger entry A1 in the project `CLAUDE.md`, which is
now the interim home for the failure-mode ledger this specification produced.
