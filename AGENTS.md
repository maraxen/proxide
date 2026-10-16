# proxide — project rules

## Hard rules

**Fail fast and loud. Never fill a sentinel where you mean "unknown".**

- **No `NaN` as an unknown value. Raise before the fill, not after.** A `NaN` mass or radius does not fail where it was created — it propagates silently through parameterisation and detonates during integration, far from the parse, with the evidence gone. If a value cannot be determined, return an error at the point of determination. This applies to every physical constant: masses, radii, scaling factors, charges, force-field parameters.
- The same prohibition covers every other quiet stand-in: `0.0`, `-1`, `""`, a neighbouring element's tabulated value, or a "reasonable" default that happens to be a real measurement.
- Unknown is its own state. It must be representable, and it must be distinguishable from every legitimate value.
- Changing a physical constant is a physics change even when it arrives inside a refactor. It needs a citation and its own commit. Uncited is worse than conservative — if the correct value can't be sourced, say so and leave the conservative one.

## Failure-mode ledger

Recorded 2026-09-10 from a real incident sweep (`task 260910_proxide_observability`). Interim location — this moves to `.praxia/docs/` once the observability spec lands. Full analysis: `.praxia/docs/specs/260910_proxide-silent-substitution-observability.md`.

Nothing here is hypothetical. Every entry names the instance it came from.

### A. Silent substitution in code

The class: the code makes an inference or substitutes a default it isn't licensed to make, returns a well-typed plausible number, and destroys the evidence that it guessed. Nothing crashes. Nothing logs.

- **A1 — Unknown encoded as a real value.** `DEFAULT_MASS = 12.0` is carbon's mass, so "unknown element" and "carbon" are literally the same `f32`; `infer_element`'s `_ => "C"` does it for symbols. Once assigned, no downstream consumer can tell a real carbon from a shrug. → Unknown gets its own representation, always.
- **A2 — Prefix dispatch silently borrows another entity's parameters.** Matching on an atom name's first character gave `"CL"` carbon's values, `"NA"` nitrogen's, `"FE"` fluorine's, `"CU"` carbon's, `"SE"` sulfur's. Each was a confident, plausible, wrong number. → Dispatch on the resolved thing, never on a prefix of its name.
- **A3 — The honest default can be *less accurate* than the accidental borrow.** Fixing A2 moved selenium from sulfur's 1.80 Å (true Bondi ≈1.90 Å) to the 1.50 Å unknown default — worse numbers, on the ubiquitous selenomethionine path. → Correctness of a default is the wrong design target; **observability** of it is the right one. Both 1.50 and 1.80 destroy the fact that nobody knows.
- **A4 — Silent zeroing.** `c23eeea`: solvent atoms were being zeroed rather than parameterised. A zero is a measurement.
- **A5 — Duplicated derivation logic, every copy wrong.** Element-from-atom-name was hand-rolled in six modules; all six were wrong, and two prior fixes each stopped at the crate where the incident surfaced. → One canonical implementation, plus a guard that fails on the seventh copy. See `tests/test_element_inference_conformance.py`.
- **A6 — Deferred fixes decay into silence.** A known violation left in place is indistinguishable from an unnoticed one within a week. → Record it with an owner, a justification, a bounded site count, and an expiry condition that the build checks.

### B. Silent failure in verification

The same class, one level up: the thing that was supposed to catch the failure reports success.

- **B1 — Self-reported verification is worthless.** Two agents briefed in detail on this exact failure class both reported PASS on work that failed: one commit stated "All numeric constants preserved exactly" while moving two GBSA radii; one guard reported its scan passing when the scan failed. → Acceptance criteria must be decidable by a machine against a fixed artifact, never by an assertion that the work was done.
- **B2 — A positive control drawn from the detector's own distribution proves nothing.** A guard "verified" itself by planting a violation shaped like its own regex. It caught it, and had 50% recall on the real bugs. → Calibrate against real historical defects from git history. Pinned refs: `calibration/element-inference-{defective,fixed-initial,fixed}`.
- **B3 — A detector whose silence is indistinguishable from its success is not a detector.** A file-level exclusion keyed on raw source text meant a *comment* mentioning `infer_element` switched off scanning for the whole file — and the resulting silence then read as "fixed". This is A1 relocated into tooling: one signal meaning two things. → "No findings" where findings were expected must fail.
- **B4 — A check written against a *shape* rather than the *thing*.** Three instances: a gate testing for `-> f32` passed both functions returning `Vec<f32>` that caused the incident; a gate on `Cargo.toml` text missed resolver-2 feature unification; a gate on Rust-side parameterisation named a boundary with zero Rust callers. Each green-lit its own motivating defect. → Enumerate ground truth; assert on resolved state, not on text.
- **B5 — A gate satisfiable by narrowing the claim.** A coverage gate comparing claimed vs tested elements goes green if you *delete* `"Se" => 78.971`. → Ratchet the claim against shrinking.
- **B6 — A ratchet satisfiable by asserting.** An "uncited constant" counter decreases when someone types a citation string nobody checked. → Require diff-reviewable evidence, not a string edit.
- **B7 — Ground truth derived from a real commit is unreliable.** Real commits are not single-purpose: the element-inference fix also carried an unrelated Br/I radius change, so naive derivation makes recall 1.0 unreachable. Derivation also cannot find a site the fix commit missed. → Reviewed, checked-in, append-only ground truth.
- **B8 — Recall without precision gets bypassed.** A gate that cries wolf is `--no-verify`'d within a week. Budget false positives explicitly.
- **B9 — Truncated evidence is lost evidence.** A diagnostics report that silently caps its findings is a well-typed plausible answer with the evidence destroyed — A1 again. → Truncation must itself be a recorded finding.

### How to use this

When adding a lookup, a parser fallback, a default, or a check: find the entry above it resembles and satisfy the rule. When you catch a new instance, add an entry with the file or commit that produced it — an unattributed ledger entry rots into folklore.
