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

- **The implementation (commit `54eefc2`) states "All numeric constants preserved exactly." It
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
test` captures stdout and a print in a test is a debugging aid, not a telemetry defect. The real
target is **23 non-binary library-source prints** (fixer 17, frag 3, rotlib 3) plus **18 binary
prints** that are legitimate CLI stdout and stay. That reframing is what makes OBS-201 a
zero-new-machinery change (§4).

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
pub fn infer_element_sourced(atom_name: &str) -> (Option<&'static str>, ElementSource);
```

`FirstCharacter` is the *licensed but weak* inference (analogue of `EnghHuberFallback`); `Missing`
is emitted where `masses.rs:66` and `masses.rs:98` currently return `"C"`. `infer_element` is
retained as a deprecated wrapper for exactly one release so Tier 0's consolidation is not
invalidated mid-flight.

### D6 — `DEFAULT_MASS` is deleted; unknown element is an error by default

`get_mass(element: &str) -> Option<f32>`; `DEFAULT_MASS` is removed, not redefined. The reporting
entry point is:

```rust
pub fn assign_masses_reported(atom_names: &[String], policy: &DiagPolicy)
    -> Result<Reported<Vec<f32>, ChemDiagKind>, DiagnosticsError>;
```

with `ChemDiagKind::UnknownElementMass { element }` at `Severity::Error` — so the *default*
policy refuses rather than substituting. A lenient policy (never promote) is available for the
wasm viewer, where rendering something beats refusing; it still records the coercion.

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
   whole dependency graph* — a library that enables one silently poisons every consumer. Rule:
   `release_max_level_info` is set **only** in `crates/proxide_py`, `crates/proxide-wasm`, and
   binary crates. Library crates must never name it, and must never take
   `tracing-subscriber` as a non-dev dependency.
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
- `coverage/known_gaps.toml` lists each claimed-but-uncovered symbol **with a justification and an
  owner**.
- The gate asserts `claimed − covered == known_gaps`. Adding an element to any table without
  adding a fixture, or without explicitly justifying the gap, fails CI.

**Gate on the set difference, not a coverage ratio.** A ratio threshold is a number nobody can
act on; a named missing element is a bug report.

Today's corpus is `1crn`, `2ala`, `1uao`, `5awl`, `altloc_two_conf`, two trajectory PDBs — all
C/N/O/S protein, with 12 HETATM records in `5awl` only. The Cordero table claims F, P, Cl, Br, I;
`get_mass` claims 19 elements. The initial `known_gaps.toml` will be embarrassingly long. That is
the point: the gap was measurable before the bug existed.

### D12 — Detector calibration against real historical defects, never synthetic ones

Every detector introduced here (coverage gate, coercion checker, subscriber checker, hot-path
checker, code registry checker, constant-drift guard) ships with a **calibration corpus**:
`calibration/cases.toml`, where each case names a real commit pair from this repository's own
history and the exact sites the detector must flag.

```toml
[[case]]
id = "element-inference-2609"
defective_ref = "5f368ec"     # four live hand-rolled element-inference violations
fixed_ref     = "54eefc2"     # all four fixed
detector      = "check_element_inference"
must_flag     = ["crates/.../gbsa.rs:NN", "crates/.../gbsa.rs:MM",
                 "crates/proxide_fixer/src/repack.rs:122",
                 "crates/proxide_fixer/src/loop_model.rs:491"]
```

The harness checks out each ref into a temp worktree, runs the detector, and asserts
**recall == 1.0 on `defective_ref`** and **zero findings on `fixed_ref`**. The 260910 guard-test
agent would have failed this immediately: its recall on that exact case is 2/4.

**Rejected: synthetic positive controls** (what the 260910 agent did). A detector always catches a
defect drawn from its own distribution; such a control establishes nothing. Synthetic *negative*
controls (a clean file that must produce no finding) remain allowed and useful.

**Rejected: "the detector was reviewed by a second agent."** Both 260910 agents were briefed on
this failure class and both self-certified. Review is not a gate; an executed calibration case is.

The calibration corpus is a **deliverable of the first task, not an afterthought**, because a
detector with unmeasured recall is exactly the "well-typed plausible answer with the evidence
destroyed" that this whole document exists to eliminate — applied to the tooling.

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
`54eefc2` performed invisibly. Snapshot entries carry a `source` field (citation or
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

`Sourced<f32>` is `Copy`, one enum discriminant wide, and has zero runtime cost against the
current `f32` return in any realistic parameter-assignment loop (the lookup is already a `match`).

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

---

## 4. Requirements

The P-column is the correctness-payoff priority used for sequencing in §6. Every criterion below
is decidable by a command.

### Tier ∅ — cross-cutting meta-requirements (bind every other requirement)

| ID | Requirement | Acceptance criterion (verifiable) | P |
|---|---|---|---|
| OBS-001 | Every acceptance criterion in this spec is decidable by a machine against a fixed artefact — a named commit, a checked-in fixture, or a checked-in corpus. No criterion may be satisfied by an agent's or author's assertion that the work was done. | `scripts/check_spec_criteria.py` parses this document's requirement tables and fails on any criterion lacking a runnable command or a named artefact. Reviewer gate: a requirement whose only evidence is a prose report is rejected. | **P0** |
| OBS-002 | Every detector introduced by this spec has a calibration case in `calibration/cases.toml` per D12, with **recall 1.0** on the defective ref and **zero findings** on the fixed ref. | `pytest tests/test_detector_calibration.py` checks out each ref into a temp worktree and asserts both. First entry: `5f368ec` (4 live element-inference violations) → `54eefc2` (all 4 fixed). A detector with no calibration case is not merged. | **P0** |
| OBS-003 | Detector recall is reported, not assumed. Each detector's measured recall over the calibration corpus is written to `calibration/RECALL.md` by the harness. | The file is a generated artefact; CI fails if it is stale relative to a fresh harness run. Any detector below recall 1.0 on a merged case blocks CI. | P1 |

### Tier 1 — canonical diagnostics channel

| ID | Requirement | Acceptance criterion (verifiable) | P |
|---|---|---|---|
| OBS-101 | `proxide_core::diag` provides `Severity`, `DiagKind`, `Finding<K>`, `Report<K>`, `Reported<T,K>`, `DiagPolicy`, `ErasedReport`, `DiagnosticsError` per D1/D3. | `cargo test -p proxide-core diag::` passes; `Report::{is_clean,errors,warnings}` have tests mirroring `precondition.rs:332-564`. | P2 |
| OBS-102 | `confind::PreconditionReport` is a re-export/alias of `Report<ViolationKind>`; no confind public API changes. | `cargo test -p proxide-confind` passes with **zero edits to** `precondition.rs`'s `mod tests`. | P2 |
| OBS-103 | `infer_element_sourced` exists per D5; `infer_element`'s two `"C"` fallbacks (`masses.rs:66,98`) return `ElementSource::Missing` through it. | `assert_eq!(infer_element_sourced("XX"), (None, ElementSource::Missing));` and `assert_eq!(infer_element_sourced("CL").1, ElementSource::TwoLetterTable);` | **P0** |
| OBS-104 | `DEFAULT_MASS` deleted; `get_mass -> Sourced<f32>` per D14; `assign_masses_reported` errors on `Untabulated` under the default policy. | `rg -w DEFAULT_MASS crates/` returns nothing; a test asserts `assign_masses_reported(&["XX"], &DiagPolicy::default())` is `Err` and that the lenient policy yields one `PROX-CHEM-UNKNOWN-ELEMENT` finding. | **P0** |
| OBS-105 | `parse_pdb_reported` / `parse_mmcif_reported` / `parse_pqr_reported` exist per D3; existing `parse_*` signatures unchanged. | A fixture PDB with a malformed occupancy field yields exactly one `PROX-COERCE-FIELD-DEFAULTED` finding with `fields["field"] == "occupancy"`; `parse_pdb` on the same input still returns `Ok`. | P2 |
| OBS-106 | Coercion triage: every `unwrap_or*` in `crates/proxide-io/src/formats/**` and `crates/proxide-core/src/chem/**` is either routed through a recorded coercion or carries a preceding `// COERCION-OK: <reason>` comment. | `scripts/check_coercions.py` exits non-zero on any unannotated site; wired into the `rust-checks` CI job. Baseline: 36 non-test sites (§C1). | P2 |
| OBS-107 | The 5 `_ =>` catch-alls in `proxide-io/src` and 4 in `proxide-core/src/chem` are each either made exhaustive or annotated `// CATCHALL-OK: <reason>`. | Same script, same gate. | P3 |
| OBS-108 | `LoopModelReport::geometry_warnings: Vec<String>` becomes `Report<LoopDiagKind>`. | `rg 'geometry_warnings' crates/` returns no `Vec<String>` declaration; `cargo test -p proxide_fixer` passes. | P4 |
| OBS-109 | Each of the 6 io error enums gains `Diagnostics(..)` and `#[non_exhaustive]` per D4. | `cargo build --workspace` and `cargo hack check --feature-powerset` pass. | P2 |
| OBS-110 | `diagnostics/registry.toml` lists every code; codes are unique and never reused. | `scripts/check_diagnostic_codes.py` compares registry against `rg -o 'PROX-[A-Z-]+' crates/` and fails on unregistered or duplicate codes. | P2 |
| OBS-111 | Every physical-constant table exposes `snapshot()` and is guarded by a checked-in `constants/snapshots/<table>.json` per D13, keyed over the table's full domain including default-arm coverage, with a `source` field per entry. | `cargo test constants::snapshot` fails on any value change not accompanied by a snapshot edit. **Regression proof required:** replaying `54eefc2` against the guard must fail with the Br/I radii diff named. Ratchet: `scripts/check_uncited_constants.py` fails if the `uncited-legacy` count increases. | **P0** |
| OBS-112 | Every public physical-constant lookup returns `Sourced<T>` per D14 — masses, mbondi2 radii, obc2 scale factors, Cordero radii, LJ defaults. No public lookup returns a bare numeric type. | `scripts/check_sourced_lookups.py` fails on any `pub fn` in a listed constants module whose return type is a bare `f32`/`f64`. **Selenium regression proof required:** `assert!(matches!(mbondi2_radius("Se").source, ValueSource::Tabulated{..} \| ValueSource::Defaulted{..}))` and a test that pins which one it currently is, so a future change to Se is a visible test edit. | **P0** |
| OBS-113 | A `Defaulted` or `Untabulated` constant reaching an energy/parameterisation path is an `Error`-severity finding under the default `DiagPolicy`; lenient policy downgrades to `Coercion` and still records it. | Test: parameterising a structure containing `MSE` fails by default with a `PROX-CHEM-DEFAULTED-CONSTANT` finding naming `Se` and the affected table; under lenient policy it succeeds and the report contains that finding. | **P0** |

### Tier 2 — one connected telemetry channel

| ID | Requirement | Acceptance criterion (verifiable) | P |
|---|---|---|---|
| OBS-201 | `println!`/`eprintln!` are denied in library source. Set `[workspace.lints.clippy] print_stdout = "deny"`, `print_stderr = "deny"`; each crate adds `[lints] workspace = true`; binary crate roots carry an explicit `#![allow(clippy::print_stdout)]`. | **CI already runs `cargo clippy --workspace -- -D warnings`** (`ci.yml:117`) — zero new machinery. Gate passes only after the 23 library-source prints (§C2) are converted to `tracing`. | **P1** |
| OBS-202 | Every binary target installs a subscriber in `main()` before any work; no library crate installs one. | `scripts/check_subscribers.py` enumerates `src/bin/*.rs` + `[[bin]]` targets, asserts each contains `tracing_subscriber::` init, and asserts no library crate lists `tracing-subscriber` outside `[dev-dependencies]`. Must fix `convert_rotlib.rs` (confirmed missing, §C3) and `confind.rs`. | **P1** |
| OBS-203 | `tracing` in `[workspace.dependencies]` with the `log` feature per D7; `pyo3_log::init()` unchanged. | A Python test configures `logging` at DEBUG, parses a fixture that emits a coercion, and asserts ≥1 record carrying a `PROX-` code. | P2 |
| OBS-204 | Phase-boundary spans only; no `tracing::` macro in any module listed in `telemetry/hot_paths.toml`. | `scripts/check_hot_paths.py` greps the listed modules and fails on any `tracing::`/`#[instrument]` occurrence. Advisory secondary check: `cargo bench --workspace` shows no >3% regression on existing MD benches. | P2 |
| OBS-205 | `release_max_level_info` appears only in `proxide_py`, `proxide-wasm`, and binary crates. | Same script as OBS-202 greps all `Cargo.toml`s and fails on a library-crate occurrence. | P2 |
| OBS-206 | `proxide-wasm` exposes `init_logging(level: &str)` bridging `tracing` to `console.*`, plus `console_error_panic_hook`; idempotent. | `wasm-bindgen-test` asserts double-init does not panic and that a parse with a coercion produces ≥1 console record. **Requires adding a wasm test runner to CI** — today `ci.yml:120-124` only type-checks wasm. If the runner is not added, this requirement's gate is compile-only and the mechanism is explicitly *unattested*. | P4 |

### Tier 3 — provenance, parity attestation, chemical coverage

| ID | Requirement | Acceptance criterion (verifiable) | P |
|---|---|---|---|
| OBS-301 | Each table exposes its own supported domain (`supported_elements()` etc.) per D11. | `cargo test` asserts `supported_elements()` and the arms of `get_mass` agree in both directions (a table-driven `get_mass` makes this trivially true — preferred). | **P0** |
| OBS-302 | `scripts/chem_coverage.py` extracts the corpus multiset to `coverage/corpus.json`. | Running it on the current corpus yields JSON whose `elements` contains at least `{C,N,O,S}`; committed as a snapshot. | **P0** |
| OBS-303 | Coverage gate: `claimed − covered == known_gaps`, each gap justified. | `pytest tests/test_chemical_coverage.py` fails when an element is added to any table without a fixture or a `known_gaps.toml` entry. **Regression proof required:** a test that removes `Cl` from `known_gaps.toml` in a tmpdir copy and asserts the gate fails. | **P0** |
| OBS-303b | Cross-table domain consistency: the coverage tool compares each table's claimed domain against **every other table's**, and flags any element tabulated in one and defaulted in another. | Test asserts the tool flags `Se` (present in `masses.rs:30`, absent from the mbondi2/obc2 tables) — the §0.2 case — and that the flag clears only via a citation or an explicit `known_gaps.toml` entry. This is the check that was mechanically available and unrun for the entire life of the selenium defect. | **P0** |
| OBS-303c | The test corpus gains at least one fixture per element class the code claims: a halide-containing structure, a metal site, and a selenomethionine (`MSE`) structure. | `pytest tests/test_chemical_coverage.py::test_corpus_covers_claimed_classes`; `coverage/corpus.json` contains `Cl`, `Se`, and ≥1 transition metal. | P1 |
| OBS-304 | `ProvenanceRecord` per D9, with `git_sha` from a `build.rs` in proxide-core. | Test asserts `git_sha` matches `^[0-9a-f]{40}(-dirty)?$` or `== "unknown"`; `build.rs` emits `cargo:rerun-if-changed=.git/HEAD`. | P3 |
| OBS-305 | `MDParameters` and the primary parse results carry `provenance: ProvenanceRecord`; the structs become `#[non_exhaustive]` with builders. | `cargo build --workspace` passes; a test asserts a parameterisation run records the forcefield file's sha256 under `ArtifactRole::ForceField`. | P3 |
| OBS-306 | `write_provenance_sidecar(output_path, &record)` emits `<output>.prov.toml` with `schema = "bathos/0.3"` per D9. | Round-trip test: the emitted file parses as TOML and contains `schema`, `proxide_git_sha`, and ≥1 `input_* = <64 hex>` key. **Open item assigned to D3t: confirm against the live bathos ingest contract before locking key names** — the shape here is derived from an observed sidecar, not from bathos's published schema. | P4 |
| OBS-307 | `parity/<slug>.bth.toml` + generated `parity/INDEX.json` per D10; GAFF2's root file migrated; HP4-WASM/OpenMM, physics/MDTraj, trajectory-roundtrip retrofitted as entries. | `scripts/parity_index.py --check` fails if `INDEX.json` is stale or an `impl_paths` entry names a nonexistent file. ≥4 ledger entries exist. | P4 |
| OBS-308 | Consumer-visible query: `proxide.attestation.is_parity_attested(path) -> Verdict \| None`. | Python test asserts `is_parity_attested("src/proxide/chem/gaff2.py")` is not `None` and an unattested path returns `None`. | P4 |

---

## 5. Fixer tasks and dependency order

Each task is one session for one fixer. `→` denotes a hard dependency.

**Wave ∅ — blocks every detector task. Build first.**

| Task | Scope | Files | Gate | ~LOC |
|---|---|---|---|---|
| **Z1** | Calibration harness + corpus (OBS-002, OBS-003). Temp-worktree checkout of a ref pair, run a named detector, assert recall 1.0 / zero-on-fixed, generate `RECALL.md`. Seed with the `5f368ec` → `54eefc2` element-inference case, all four sites enumerated by hand from the diff. | `calibration/cases.toml`, `tests/test_detector_calibration.py`, `scripts/calibrate.py` (create) | `pytest tests/test_detector_calibration.py` — must fail against the 260910 guard-test heuristic (recall 2/4) and pass against a corrected one | ~250 |
| **Z2** | Constant snapshots (OBS-111). Add `snapshot()` to each constant table; generate `constants/snapshots/*.json` **from the pre-`54eefc2` tree** so the Br/I change shows as a reviewable diff, not as baked-in truth; add the `uncited-legacy` ratchet. | `constants/snapshots/*.json` (create), `proxide-core/src/chem/*.rs`, `proxide-physics/src/**/gbsa.rs` (modify), `scripts/check_uncited_constants.py` (create) | `cargo test constants::snapshot`; replaying `54eefc2` fails with Br/I named | ~200 |

Z1 `→` A2, A3, B5, and any other task shipping a detector. Z2 `→` A1 (A1 edits `masses.rs`, so the
snapshot must exist first or the deletion of `DEFAULT_MASS` is itself an unguarded constant
change). Z1 and Z2 are parallel with each other.

**Retroactive items, both assigned to Z2** — leaving either open converts a documented incident
into permanent unattributed physics:

1. The Br/I radii introduced by `54eefc2` are either cited to a reference and kept, or reverted,
   in a commit that says which.
2. The five elements of §0.2 (`Cl`, `Na`, `Fe`, `Cu`, `Se`) get **cited** mbondi2 radii and obc2
   scale factors, or an explicit `Defaulted` marking plus a `known_gaps.toml` entry. Selenium is
   the priority: `MSE` is common, and the current post-fix value (1.50 Å) is *further* from Bondi's
   ≈1.90 Å than the value the bug was supplying. Do not close this by quietly writing `1.90` — cite
   it, snapshot it, and let OBS-112's pin-test record the change.

**Wave A — independently parallelisable once Wave ∅ lands.**

| Task | Scope | Files | Gate | ~LOC |
|---|---|---|---|---|
| **A1** | Element source + mass domain (OBS-103, OBS-104, OBS-301). Delete `DEFAULT_MASS`, table-drive `get_mass`, add `infer_element_sourced`, `supported_elements()`. Update the 3 io call sites + `py_chemistry.rs`. | `proxide-core/src/chem/masses.rs` (modify), `proxide-io/src/formats/{pdb,mmcif,pqr}.rs` (modify), `proxide_py/src/py_chemistry.rs` (modify) | `cargo test -p proxide-core -p proxide-io` + `rg -w DEFAULT_MASS crates/` empty | ~180 |
| **A1b** | `Sourced<T>` constant lookups + defaulted-constant gate (OBS-112, OBS-113) per D14. Convert masses, mbondi2 radii, obc2 scale, Cordero radii, LJ defaults. `→` Z2, B1 (needs `Severity`/`DiagPolicy`). | `proxide-core/src/chem/*.rs`, `proxide-physics/src/**/gbsa.rs`, `proxide-ligand-frame/src/geometry_gate.rs` (modify), `scripts/check_sourced_lookups.py` (create) | `MSE` parameterisation fails by default with `PROX-CHEM-DEFAULTED-CONSTANT` naming `Se`; passes under lenient policy with the finding recorded | ~260 |
| **A2** | Coverage extractor + gate (OBS-302, OBS-303). Depends on A1 only for `supported_elements()`; write against a stub if A1 is in flight. | `scripts/chem_coverage.py`, `coverage/{corpus.json,known_gaps.toml}`, `tests/test_chemical_coverage.py` (create) | `pytest tests/test_chemical_coverage.py` + the Cl-removal regression proof | ~220 |
| **A3** | Subscriber rule (OBS-202, OBS-205). Fix `convert_rotlib.rs` and `confind.rs`; add the checker. | `proxide-rotlib/src/bin/convert_rotlib.rs`, `proxide-confind/src/bin/confind.rs`, `proxide-tmalign/src/bin/tmalign.rs`, `proxide-jaccard/src/bin/*.rs`, `proxide-wasm/src/bin/param_cli.rs` (modify), `scripts/check_subscribers.py` (create), `ci.yml` (modify) | `python scripts/check_subscribers.py` exits 0 | ~120 |
| **A4** | Print eradication (OBS-201). Convert the 23 library-source prints to `tracing`; add workspace lints; allow-list binary roots. | root `Cargo.toml` + 18 crate `Cargo.toml`s, `proxide_fixer/src/{repack,finder,loop_model}.rs`, `proxide-frag/src/search.rs`, `proxide-rotlib/src/geometry/charmm_ic.rs`, `proxide-wasm/src/gaff2.rs` (modify) | `cargo clippy --workspace -- -D warnings` | ~150 |

A4 `→` A5 (needs `tracing` in workspace deps); otherwise Wave A is fully parallel. A3 and A4
touch disjoint files (bins vs libs) except `proxide-wasm`; give both to one fixer, or sequence
A3→A4, if that conflict matters. A1 `→` Z2; A2, A3 `→` Z1 (each ships a detector).

**Wave B — core diagnostics; serialised on B1.**

| Task | Scope | Gate |
|---|---|---|
| **B1** | `proxide_core::diag` module (OBS-101, OBS-110) + registry + code checker. `→` nothing. | `cargo test -p proxide-core diag::`; `scripts/check_diagnostic_codes.py` |
| **B2** | Migrate confind onto `Report<ViolationKind>` (OBS-102). `→` B1. Proves the generalisation preserves the source design. | `cargo test -p proxide-confind` with unmodified `mod tests` |
| **B3** | Error-enum composition (OBS-109). `→` B1. Mechanical, 6 files. | `cargo hack check --feature-powerset` |
| **B4** | `*_reported` parsers + coercion recording (OBS-105). `→` B1, B3, A1. | malformed-occupancy fixture test |
| **B5** | Coercion/catch-all triage + checker (OBS-106, OBS-107). `→` B4. | `scripts/check_coercions.py` |
| **B6** | `LoopModelReport` migration (OBS-108). `→` B1. Parallel with B2-B5. | `cargo test -p proxide_fixer` |

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

**Critical path:** Z1 → (A2 ∥ A3) and B1 → B3 → B4 → B5. Z1/Z2 are two sessions ahead of
everything; D4t and B1 run concurrently with them.

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
3. **OBS-111 — constant-table snapshots.** `54eefc2` changed published physics under a commit
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

**If only a third gets built, build exactly this:** Z1, Z2, A1, A1b, A2, A3, A4 — the calibration
harness, the constant snapshots, element provenance, `Sourced<T>` lookups + the defaulted-constant
gate, the chemical-coverage gate, the subscriber rule, the clippy print lints. Three are
script-only, and one (OBS-201) requires **zero new CI machinery** because
`cargo clippy --workspace -- -D warnings` already runs at `ci.yml:117`.

If even that is too much, **Z1, Z2 and A1b are the irreducible core.** Z1/Z2 are the only items
that have already caught a real defect in this repository within the last 24 hours, and every
remaining requirement depends on Z1 to be trustworthy at all. A1b is the one item that addresses
the residual defect (§0.2) which the completed Tier 0 fix demonstrably does *not* remove.

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
| OBS-201 turns clippy `-D warnings` into a hard blocker mid-migration; CI red on main. | Whole workspace. | Land the 23 conversions **first**, the lint config **last**, in that order within A4. Verify locally with `cargo clippy --workspace -- -D warnings` before the lint commit. |
| `release_max_level_*` feature unification silently disables logging for all consumers if set in a library. | Entire dependency graph, silently. | OBS-205's grep gate. This is a silent-substitution failure mode *inside* the observability system; it gets a CI check, not a code comment. |
| `known_gaps.toml` becomes a rubber stamp — every new element gets a gap entry instead of a fixture. | Defeats OBS-303 entirely. | Require an owner and a justification per entry; add a check that fails if `known_gaps.toml` grows in a PR with no corresponding fixture addition. Review entries at each release. |
| Bathos sidecar key names guessed from one observed file rather than the schema. | OBS-306 only. | D3t's first step is reading the live contract; the requirement explicitly flags the schema as unconfirmed. Do not let D3t start with code. |
| Adding `tracing` to 18 crates measurably slows compile/CI. | Build times. | `default-features = false`; only `std`, `attributes`, `log`. `tracing-subscriber` stays dev/bin-only, which is where the compile cost actually lives. |
| `Report<K>` generics leak into pyo3/wasm signatures and cause a type-parameter explosion. | proxide_py, proxide-wasm. | D1's `ErasedReport` boundary is mandatory, not optional: `serde` derives exist only on the erased form, so generics structurally cannot cross the FFI line. |
| A fixer self-certifies a detector as passing when it does not (observed twice on 260910). | Every gate in this document. | OBS-002: no detector merges without a calibration case whose recall is machine-measured. OBS-001: no criterion is satisfiable by assertion. Reviewers reject "the scan passed" without a `RECALL.md` diff. |
| The calibration corpus itself is gamed — cases chosen to be easy, or `must_flag` sites trimmed to whatever the detector happens to catch. | OBS-002 becomes theatre. | `must_flag` sites are enumerated **from the fixed commit's diff**, mechanically, before the detector is written; the diff hunks are the ground truth, not the author's judgement. A case may only be added, never narrowed — `calibration/cases.toml` is append-only under CI check. |
| **Numeric-accuracy regression is already shipped.** Tier 0's fix moved Se from 1.80 Å (borrowed from S, ≈0.10 off Bondi) to 1.50 Å (default, ≈0.40 off). `MSE` is common in the PDB, so real GBSA energies are now measurably worse than before the "fix". | Every solvation calculation on a selenomethionine structure since `54eefc2`. | Z2 retroactive item 2: cite and snapshot Se's real radius. **Interim, before Z2 lands:** OBS-113 makes the affected path *fail loudly* rather than continue quietly — a refused calculation is recoverable, a published one is not. Do not let A1b (the gate) wait on Z2 (the correct number); the gate is the more urgent of the two. |
| Making `Defaulted` constants an error by default breaks currently-working user pipelines that process metal sites, halides or `MSE`. | Every downstream consumer parameterising non-C/N/O/S structures. | This is intended and is the point of OBS-113 — those pipelines have been producing unattributed numbers. Mitigation is a documented one-line lenient policy plus a `PROX-CHEM-DEFAULTED-CONSTANT` code they can assert on, not a softer default. Announce with the release; land A1b and OBS-303c's fixtures together so the failure is demonstrable rather than surprising. |
| Snapshotting constants from the *current* tree bakes `54eefc2`'s unreviewed Br/I values in as canonical. | GBSA physics, permanently. | Z2 generates snapshots from the **pre-`54eefc2`** tree so the change appears as a reviewable diff; the retroactive cite-or-revert decision is part of Z2, not deferred. |
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
- Commit `5f368ec` (4 live element-inference violations) → `54eefc2` (all 4 fixed, Br/I radii
  silently changed) — the calibration corpus's first case (D12) and the source of §0.1/§0.2

---

## 10. Verification of this document's own claims

Per OBS-001, the factual claims here were measured, not assumed. Reproduce:

| Claim | Command |
|---|---|
| §C1 `unwrap_or` counts (28 / 39 / 36) | `rg -c 'unwrap_or\(' crates/proxide-io/src` and `rg -c 'unwrap_or(_else\|_default)?\(' crates/proxide-io/src` |
| §C2 print counts (118 / 42 / 76) | `rg -c '\b(println!\|eprintln!)' --glob '**/*.rs' .` |
| §C3 `convert_rotlib` has no subscriber | `rg 'tracing_subscriber\|env_logger\|subscriber\|logger' crates/proxide-rotlib/src/bin/convert_rotlib.rs` → no matches |
| §2 `infer_element` still defaults to carbon | `crates/proxide-core/src/chem/masses.rs:66,98` |
| §0.1 / §0.2 GBSA value changes | `git diff 5f368ec..54eefc2 -- '*gbsa*'` |

Two of the brief's premises did not survive (§C1, §C2). A reviewer should assume the same failure
rate applies to this document and check before building.
