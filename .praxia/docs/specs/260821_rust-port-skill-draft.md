---
title: 'Draft: a generalized `rust-port` skill'
description: Proposal for a reusable skill covering Python→Rust ports of validated algorithmic code, distilled from the GAFF2 atom-typer port (260821)
status: draft
task_id: 260821_gaff2-rust-port
date: '260821'
---

# Draft: a generalized `rust-port` skill

**This is a draft for review. It is not installed anywhere and should not be until the
open questions at the end are answered.**

Distilled from one run: the Python→Rust port of the GAFF2 DEF-grammar atom typer
(`crates/proxide-gaff2`, 2026-08-21). Full evidence in
[`260821_gaff2-rust-port-lessons.md`](../reference/260821_gaff2-rust-port-lessons.md).
Every claim below is anchored to something that actually happened in that run; where I am
extrapolating from a single instance, I say so.

---

## Scope: what this skill would be for

**Trigger**: porting a *validated* algorithmic implementation from Python to Rust, where a
parity figure already exists against an external ground truth and the port's job is to
**reproduce behavior exactly**, bugs included — not to improve it, not to modernize it.

Sibling to `jax-port` (which handles NumPy/PyTorch→JAX numerical ports) and downstream of
`customizing-tools`, which already owns the *should we rewrite this at all* decision. This
skill starts after that decision says yes.

**Not for**: greenfield Rust, performance rewrites with no reference implementation, or
ports where divergence from the source is acceptable. Those are different problems with
different gates.

**The defining constraint** that makes this its own skill: for a behavior-preservation
port, **a bug you accidentally fix is a defect**. Every ordinary engineering instinct —
clean up the weird branch, use the right constant, make the API safer — is actively
wrong here, and needs a process that resists it rather than relying on discipline.

---

## Phase structure

Seven phases. The ordering is not arbitrary; each exists because skipping it in this run
would have produced a specific failure that did in fact almost happen.

### Phase 0 — Baseline and existing-implementation audit

Establish that the reference suite passes and the Rust workspace builds *before* touching
anything, so any later breakage is attributable.

Then, and this is the phase most likely to be skipped: **grep the target workspace for an
existing implementation of the thing you are about to port.**

> **This run's evidence.** The entire premise of the GAFF2 port came from discovering
> that a Rust "GAFF typer" had already shipped, been exposed through pyo3, and been
> consumed at four call sites for months — and was not a port of anything.
> `crates/proxide-gaff/src/gaff.rs:312` is a 42-line element-plus-neighbor-count `match`
> with no DEF grammar at all, authored from intuition against no oracle. It never failed
> loudly. It returned plausible-looking atom type strings that were simply wrong, and
> downstream force-field parameter selection consumed them silently.

The generalizable rule: **the most dangerous artifact in a port project is not the
untranslated Python; it is confident-looking Rust already sitting in the repo under the
name you were about to use.** A heuristic that returns the right *type* is invisible to
type checking, invisible to "does it compile", and invisible to any test written by the
same person who wrote the heuristic.

Phase 0 outputs, all of which later phases consume:
- Reference suite green, with counts.
- For each existing candidate implementation found: is it a real port (cite the source
  lines it tracks) or an unvalidated heuristic? What are its live call sites?
- A decision on whether the existing implementation gets deprecated **before** the port
  lands. In this run the jury said "before"; it was not done, and the wrong typer is
  still live at four call sites today. Make this a gate, not a recommendation.

### Phase 1 — Decide, via a fresh jury

A multi-juror decision on port *method* — transpile / hybrid / hand-write — with each
juror reaching its own verdict independently before synthesis.

> **This run's evidence.** Three jurors: two for `reimplement_manual` at 0.85, one hybrid
> dissent at 0.72. The value was not the majority verdict; it was the structure of the
> agreement. All three converged on facts that then constrained everything downstream:
> that 68–80% of the module sits inside RDKit-touching functions (measured by two
> independent AST splits, 109–129 call sites), that the transpiler has no C-extension
> story so `Chem.Mol` in a signature is a category error rather than a risk tradeoff, and
> — most importantly — that the parity figure is a *joint* property of the Python and
> RDKit's perception conventions, so **the port must be scoped at the RDKit boundary.**
>
> The dissent was also load-bearing in a way a consensus process would have lost: it was
> the only juror to raise licensing, and its blocking conditions were adopted wholesale
> even though its method recommendation lost.

Generalizable: the jury's real output is not the verdict, it is (a) the **boundary** — the
exact line where the port stops and the host language keeps its C-extension-dependent
work, and (b) any **blocking conditions orthogonal to method** that a method-focused
process would drop on the floor.

A verdict that says "port X to Rust" without naming the boundary has not decided anything.

### Phase 2 — Assess / recon

Inventory the module: functions, line ranges, cross-references, external-library call
sites, existing test fixtures.

Two hazards this run hit, both worth baking into the skill as required checks:

- **Pin the checkout.** Three separate agents lost time to line-number citations that
  resolved into a different file — the worktree's `gaff2.py` is 1,905 lines, the main
  checkout's is 1,259, and the dispatch prompts' ranges were valid only in the worktree.
  One agent found a doc comment citing ranges that exist in neither. **A line citation is
  not an address unless the checkout is named alongside it.** Recon output should carry
  the branch and worktree path in every citation.

- **Recon is stale by construction.** This run's recon caller list was gathered on a
  different branch; the architecture pass had to correct it, finding three callers recon
  missed (including two that made the signature change a *public API break* rather than
  an internal refactor) and two it listed that had already been deleted. Recon should be
  treated as a hypothesis the architecture phase verifies, not as input the port phase
  trusts.

### Phase 3 — Architecture, then scaffold

Architecture decides: crate identity and dependency graph, the input data type, the asset
embedding strategy, the module list, and the pyo3 contract.

The single highest-value output in this run was the **input type**, decided before a line
was written:

> The architecture pass rejected the workspace's existing `Topology` type on two
> parity-fatal grounds — it carries no bond orders at all, and its `adjacency` is a
> `HashMap`, so bond iteration order is nondeterministic. It mandated an owned
> `MolGraph { bonds: Vec<Bond>, ... }`, and made `BondOrder` an enum with **no `Aromatic`
> variant** so that an un-Kekulized aromatic bond is un-representable rather than silently
> mis-coloring. That is fail-loud encoded structurally rather than as a runtime check.

**Scaffold** then creates the crate, wires workspace membership, embeds assets, and — this
is where this run failed — **implements the guards**.

> **This run's evidence, negative.** The scaffold promised three drift guards: a
> `include_str!` embedding of the canonical asset, a SHA-256 content-digest pin so an
> upstream asset bump becomes a reviewed change that forces re-running the parity
> campaign, and a cross-language digest test that survives the maturin wheel boundary.
> The digest pin was marked TODO. Neither digest guard was ever implemented, because
> **once the per-module fan-out starts, nobody owns the scaffold's promises** — each
> module agent's scope is its own file. Worse, the `include_str!` embedding was silently
> *reverted* by the `rules_loader` porter, who replaced it with an
> `env!("CARGO_MANIFEST_DIR")` path plus a silent empty-ruleset fallback, reintroducing
> exactly the cwd-dependence the scaffold had explicitly rejected. The verifier caught it;
> it is still unfixed.

Rule for the skill: **a scaffold guard marked TODO will not be implemented.** Guards
either land in the scaffold phase or are dispatched as their own module with their own
verifier.

Second rule, also learned negatively: **generate the module fan-out from the architecture
verdict's scoped list, not from the recon inventory.**

> The architecture verdict named 9 modules and explicitly instructed that six others be
> **dropped** from the crate as out-of-scope parameter-generation work already owned by a
> sibling crate. All six were ported anyway, by six subagents each of which correctly
> judged its module in-scope because its own dispatch prompt said so. Result: 17 shipped
> files against 9 planned; one module re-implementing four already-ported siblings in-file
> (with the binary embedding the same 878 KB asset twice, and the twins already diverged
> on map type); and **three of the four unresolved behavioral divergences landed in the
> dropped six** — where the regression gate cannot see them.

The scoping decision has to be applied to the *dispatch list*. A per-module pipeline
faithfully ports whatever list it is handed and will not re-litigate scope.

### Phase 4 — Per-module: port → adversarial-verify → record lesson

The core loop, run in parallel across modules. Three distinct roles, and the separation
between the first two is the most valuable single element of this whole structure.

**Port.** One agent per module. Given: the Python source range (with checkout named), the
established crate conventions, and the instruction to preserve bugs. Produces the module,
its unit tests (ported from the Python suite's fixtures where they exist), and an explicit
list of deviations classified as *structural* (forced by the target language) vs.
*behavioral* (never acceptable).

**Adversarial verify.** A *different* agent, **given the Python as ground truth and not
given the porter's rationale**, instructed to refute. This blindness is the mechanism —
see the next section.

**Record lesson.** Append the module's outcome — both voices, disagreements intact — to a
running lessons log. Cheap-tier work, but with a constraint learned the hard way (see
"Model tiering", below).

Sequencing note this run got wrong: **shared-type modules must land before the fan-out,
not inside it.** Porters reported sibling files changing shape under them mid-task, a
duplicate `[dev-dependencies]` table producing invalid TOML, one agent's compile fix
silently reverted by another, and — twice — discovering mid-task that a sibling had already
ported the exact function they were writing. One porter had to throw away and rewrite an
entire matching-function design after discovering the sibling convention had changed under
them. Two disciplines that did work, worth encoding: read a sibling's *current* content
immediately before designing any cross-module interface (not once at task start), and when
you must patch a file you do not own to get a build, revert it byte-for-byte afterward and
say so.

**Add a crate-level build gate between port and verify.** This run had none:

> For much of the port phase the crate did not compile — six type errors in one module,
> one in another. **Five separate verifiers independently rediscovered this** and reported
> it as a blocking defect, and every porter's "N/N tests pass" claim was true only of a
> scratch crate extracted to `$TMPDIR`. That is five duplicated investigations of one fact
> a single `cargo test -p <crate>` between stages would have surfaced once.

(The porters' workaround — extracting the module plus its stable dependencies into a
disposable crate to get evidence-backed pass/fail on their own logic — was individually
correct and worth documenting as a technique. It was collectively wasteful.)

### Phase 5 — Corpus-scale regression gate, requiring EXACT reproduction

See its own section below. This is the phase that distinguishes a behavior-preservation
port from a rewrite.

### Phase 6 — Conditional cutover

Land the crate with **no caller rewired**, prove the gate green, *then* flip the call
sites — and only if the blocking conditions from Phase 1 are met.

> **This run followed the ordering and correctly declined the cutover.** The architecture
> pass had reasoned: if the swap and the engine land together, any corpus disagreement is
> ambiguous between "port bug" and "input-contract bug." So the harness landed first, went
> green at 100%, and no caller was flipped. Cutover was then withheld because the Phase-1
> licensing conditions were unmet.
>
> The cost of being right here: the port is currently dead code, and the wrong heuristic
> it was built to replace is *still live at all four call sites*. A conditional cutover
> must be paired with an explicit owner and deadline for the conditions, or "correctly
> declined" silently becomes "abandoned."

### Phase 7 — Synthesis

Consolidate the progressively-written log into a coherent document with an executive
summary, reconcile in-flight claims against the delivered artifact, and extract
cross-cutting lessons.

This is not ceremony. Synthesis in this run found three things no phase had noticed,
because each was only visible by comparing across phases: the scaffold's abandoned guards,
the 9-vs-17 scope drift, and the fact that the Decide-phase log section was **largely
confabulated** (see "Model tiering").

---

## The hazard class this port was designed around

**Order-dependent algorithms are the silent-port killer, and Rust's `HashMap` is the trap.**

The mechanism, stated generally: Python code frequently inherits iteration order *for
free* — from dict insertion order (guaranteed since 3.7), from a library's internal
storage (RDKit's `GetBonds()` returns file order), from list construction. When that
inherited order is load-bearing for correctness, it is usually **undocumented**, because
the Python author never had to think about it. Translate it into Rust with the obvious
`HashMap`/`HashSet` and you get a program that:

- compiles cleanly,
- returns a well-formed, plausible answer for every input,
- passes any test whose input is small enough that order does not bite,
- and is **run-to-run unstable**, so it may pass CI and fail in production, or vice versa.

There is no type error, no panic, no `unwrap` on `None`. It is the purest form of silent
wrong answer.

> **This run's instance.** The reference algorithm (`atadjust()`, itself a C
> transliteration) 2-colors conjugated systems by sweeping bonds in molecule/input-file
> order, with exactly one reseed permitted per pass. A `HashMap`-ordered sweep produces a
> different — and unstable — `cc`/`cd` atom-type coloring. Every affected atom still gets
> a valid GAFF type string; nothing errors.

**What worked: design it out, don't test it out.** The hazard was named in the
architecture phase before any code existed, the `HashMap`-carrying input type was
rejected on those grounds, an ordered `Vec<Bond>` was mandated, and a comment in the
crate's `Cargo.toml` explicitly forbids `HashMap` for the order-bearing tables. Then every
module verifier hunted it independently as a named target. **Result: zero order-dependence
defects across 13 modules.**

**The counterweight — don't over-apply it.** Two porters reached for `IndexMap` where the
Python dict is only ever `.get()`-ed and never iterated; one then had to rewrite a public
struct because a sibling had already been verified against a `HashMap`-shaped contract.
The discriminator is one grep: **does any consumer iterate this map?** If not, `HashMap`
is behavior-preserving and simpler. Blanket "IndexMap everywhere" is noise that hides the
two places where it genuinely matters.

### Why the verifier must be blind to the porter's rationale

An adversarial verifier that has read the porter's justification reads a
*checked-and-dismissed concern* and moves on. A blind one re-derives from the source and
runs the thing.

> **This run's decisive case.** The `parameterize` porter wrote a doc comment explaining
> why fractional aromatic bond orders were unreachable in the Rust, and backed it with two
> passing unit tests **whose names asserted that claim**
> (`db_bucket_is_reachable_only_for_a_fractional_bond_order_in_1_4_to_1_9`). Any reviewer
> handed that rationale sees verified parity work. The blind verifier instead ran real
> phenol through the real Python, found the reference emits `db`/1.5 on six ring bonds
> where the Rust gives `sb`/1.0 and `tb`/2.0, and refuted — noting that the tests and doc
> "read as verified parity work but encode the divergence, actively misleading future
> reviewers."
>
> The same pattern recurred in `atom_bond_facts`, where a doc block reassuringly stated
> two copies of a function "cannot diverge in practice"; the verifier proved by building
> that one copy was dead code and the live call site was the one failing to compile.

Generalizable: **a confident doc comment near a subtle divergence is an amplifier, not a
mitigation.** It converts a findable bug into one that survives review. The verifier's
blindness is what breaks that.

**Verifier standards that produced real findings in this run**, worth encoding as
requirements rather than suggestions:

- **Differential execution against the real reference, not transcription.** Verifiers
  imported the actual Python functions and drove both implementations with identical
  inputs. Scale reached: 155,292 cases (`atomic_prop`), 37,800 (`param_lookup`), 600k
  parser-fuzz plus 10,868 real-molecule match checks (`chem_env`), 1,527 assertions over
  84 molecules (`atom_bond_facts`), 322/322 byte-identical corpus lines (`def_parser`).
- **Mutation power-checks.** The `alternation` verifier, having found zero mismatches over
  4,000 cases, then *injected* two plausible mis-scopings of the reseed flag and re-ran
  the same corpus, getting 85 and 121 mismatches. This proves the corpus discriminates the
  exact bug class rather than merely being large. **A clean differential with no power
  check is weak evidence.** This should be mandatory for any module carrying an
  order-dependent or iterative algorithm.
- **A fixed four-item hunt list**, reported on explicitly even when N/A: (1) `HashMap`/
  `HashSet` where the Python relies on order; (2) the algorithm's specific known-subtle
  property — here per-pass vs. per-component reseed scoping; (3) off-by-one and precedence
  in first-match logic; (4) **silent improvements**. Requiring "NOT APPLICABLE, and here is
  why" prevents a clean report from being read as coverage it does not have — one verifier
  explicitly flagged this, noting the orchestrator should not read its clean item-2 result
  as covering the module where that logic actually lives.
- **Report defect class separately from logic verdict.** Three of this run's seven
  refutations were "the crate does not build, so my tests never ran" — nothing to do with
  translation fidelity, and all resolved before the gate. Conflating them with the four
  genuine behavioral divergences would have made the port look far worse than it was, or
  (worse) let the real four hide in the noise.

---

## "Port bugs faithfully, triage after"

The discipline: **translate the reference including its bugs, and fix nothing.**

This is counterintuitive enough that it needs stating in the dispatch prompt of every
porter and every verifier, because otherwise competent engineers will quietly do the right
thing and break the port.

> **Bugs deliberately preserved in this run**, each confirmed preserved by a verifier:
> `AR2` and `AR3` collapsing into a single class; count prefixes on certain bare tokens
> silently ignored; a constraint body whose tokens all fail to parse becoming an
> always-true wildcard (fail-*open*); one grammar field lacking a `*` branch its siblings
> have, so `*(C4)` silently loses its constraint; the improper-torsion table coming out
> **entirely empty** against the real data file because a presumed-integer periodicity
> column actually holds fractional force constants; a threshold ladder testing `>= 1.9`
> before `>= 1.4`, so a double bond classifies as triple; a `predecessor` parameter
> accepted and never read; a function whose docstring contradicts its own code (the port
> preserved the code, not the docstring).

**Why**: the parity figure the port is measured against is a property of the reference
*including* its bugs. Fix one during translation and you have (a) failed the gate, and (b)
destroyed your ability to attribute any later divergence — every future mismatch is now
ambiguous between "port bug" and "the fix I made."

**Triage after** means: bugs get fixed in a separate, separately-gated change, where **the
diff in the signature set is the deliverable** rather than an accident. That is also the
only way the fix gets validated — you can see exactly which corpus signatures moved.

Two supporting techniques worth encoding:

- **Write the assertion before trusting your paraphrase.** One porter read
  `if bo >= 1.9: "tb" elif bo >= 1.4: "db"`, concluded a double bond (2.0) maps to `db`,
  and only a failing unit test forced the re-read revealing `2.0 >= 1.9` is true. A
  genuine reference bug, correctly preserved — but it would have shipped silently wrong
  without an executed assertion.
- **Verify a ported fixture's expected values before asserting them.** One porter found
  the Python suite's `known_masses` table was comparing masses against van-der-Waals
  sigma values, invisible for years because the test ended in `return failed == 0` rather
  than `assert` and therefore could never fail. Transcribing that into a real Rust
  `assert_eq!` would have made a *correct* port fail. **A Python test that cannot fail is
  not a fixture; it is a comment.**

---

## The regression gate: exact signature-set reproduction

**A behavior-preservation port must not be gated on match rate.** It must be gated on
reproducing the reference's divergence *signature set* exactly.

The construction that worked:

1. Run reference and port over a corpus sample, from **identical parsed inputs** — same
   molecule object, same preprocessing. (This run needed a helper mirroring the Python's
   internal preprocessing exactly, because the Python Kekulizes a local copy and the Rust
   binding has no equivalent internal step. Getting this wrong makes every result
   ambiguous between engine bug and input-contract bug.)
2. Report direct port-vs-reference match rate. **Require 100%.**
3. Independently compute *each* engine's mismatch signature set against the external
   ground truth, and require the two sets to be **identical** — `only_in_reference` and
   `only_in_port` both empty.

> **This run's result**: 3,000-ligand sample (seed 42), 2,923 typeable, **100.00% match,
> 0 mismatches**. Both engines showed exactly the same 10 signatures against ground truth
> — `(c2,c)`, `(cc,cd)`, `(cd,cc)`, `(ce,cc)`, `(ce,cd)`, `(cf,cd)`, `(cq,cp)`, `(n2,nc)`,
> `(n2,nd)`, `(nd,nc)` — with both difference sets empty. Those 10 are the reference's
> *known remaining bugs*, documented in a prior audit.

**Why step 3 is the real gate.** Step 2 answers "does the port agree with the reference
here." Step 3 answers "does the port diverge from truth in exactly and only the same ways
the reference does." A new signature in `only_in_port` means a novel divergence. A missing
signature in `only_in_reference` means **the port silently fixed a bug the reference still
has** — and *that is a failure*, and it is the one a naive "match rate went up, ship it"
gate waves straight through.

The `parameterize` refutation is this exact class made concrete: the port produced
chemically *more correct* aromatic bond orders than the reference, and pinned the
improvement with two passing tests and a doc claim that the divergence had been checked.

**Failure modes to design against, all observed here:**

- **The gate covers less than the crate.** This gate exercises the atom-typing entry point
  only, so the six out-of-scope modules — carrying three of the four unresolved
  behavioral divergences — are entirely invisible to the 100% figure. State the gate's
  coverage boundary explicitly next to its result, or the number will be read as broader
  than it is.
- **The gate cannot see build-machine-dependent defects.** One module resolves an asset
  path from `env!("CARGO_MANIFEST_DIR")` and falls back to an empty ruleset if absent. The
  gate ran where the crate was built, so the file existed. The silent-empty failure mode
  is structurally invisible to any run performed on the build machine.
- **Sampling erodes the criterion.** The acceptance gate specified a full-corpus re-run;
  8% ran, for time and network budget. Reproducible (fixed seed) and probably sufficient,
  but it is not what was agreed, and the difference should be recorded rather than
  rounded off.

---

## Model tiering

What this run used, and what the evidence says about it.

| Phase | Tier | Evidence from this run |
|---|---|---|
| Recon / assess | Haiku | Adequate for inventory. **Produced a stale caller list** (wrong branch) that architecture had to correct — acceptable if recon is explicitly a hypothesis, not if downstream trusts it. |
| Scaffold | Haiku | Adequate for mechanical crate setup. **Deferred both drift guards as TODO**, and nothing later owned them. Symptom of tier *or* of phase design — see below. |
| Port (per module) | Sonnet | Right-sized. Produced faithful translations under genuinely awkward conditions (churning siblings, stubbed dependencies), and several porters independently invented good techniques — scratch-crate isolation, dependency-injected test seams, byte-for-byte revert of borrowed patches. |
| Adversarial verify | Opus | **Clearly right-sized, and the run's best value-for-tier.** Verifiers built differential harnesses driving the real Python, ran 10⁴–10⁵-case fuzz campaigns, performed mutation power-checks, and caught four genuine behavioral divergences plus a systemic build failure. This is not review, it is independent re-derivation plus experiment design; it does not degrade gracefully. |
| Jury / architecture | Opus | Right-sized. The architecture pass's input-type decision (rejecting the `HashMap`-carrying type, forbidding an `Aromatic` enum variant) is plausibly the single highest-leverage output of the run — it eliminated the primary hazard class by construction. It also corrected recon's caller list and caught the public-API-break implication recon missed. |
| Regression gate | Sonnet | Right-sized. Wrote a new validation script following existing conventions, correctly reused sibling helpers rather than copy-pasting, diagnosed a GitHub rate limit, and — notably — **designed the signature-set cross-check itself**, correctly reasoning about why transitivity makes it the right gate. |
| Cutover | Sonnet | Not exercised (correctly declined). |
| Lesson recording | Haiku | **The one clear mis-fit. See below.** |
| Synthesis | Opus | Right-sized; found three cross-phase problems no single phase could see. |

### The concrete tiering finding: cheap-tier recording of work it did not do

> The Decide-phase section of the lessons log had to be **replaced wholesale** at
> synthesis. The cheap-tier recorder, asked to summarize a decision it did not
> participate in, invented: a caller list drawn from an unrelated project (three of the
> eight named functions do not exist in this repo), a crate name never adopted, and an
> architecture involving "distance computation kernels" that has nothing to do with the
> algorithm. It was fluent, well-structured, confidently specific, and almost entirely
> wrong. Nothing downstream depended on it, so nothing caught it for the whole run.

Two readings, and I think both are true:

1. **Tier mis-fit.** Summarizing a dense technical decision is not the churn task it looks
   like. It requires holding the artifact and the summary side by side and resisting
   plausible completion.
2. **Task design failure, which is the more portable lesson.** The recorder was asked to
   *paraphrase from context* rather than to *quote from an artifact*. Even a strong model
   given that instruction produces confabulation risk. **Lesson capture should quote or
   link the decision artifact, and any phase summary written by a tier that did not
   participate in the phase should be marked unverified until reconciled at synthesis.**

Per-module lesson recording — where the recorder summarizes a report it was handed
verbatim — worked fine at Haiku. The failure is specific to summarizing work the recorder
had no artifact for.

**Where I would change the tiering next time:** none of the assignments look wrong on
this evidence. What I would change is the *insertion of a mechanical gate* (crate-level
`cargo test` between port and verify), which costs no model tier at all and would have
saved five Opus verifiers from independently rediscovering the same broken build. The
cheapest available intervention in this run was not a model upgrade; it was a build check.

---

## Proposed skill structure

If this becomes a real skill:

```
rust-port/
  SKILL.md                    # trigger, phase overview, the three non-negotiables
  references/
    hazard-order-dependence.md   # HashMap trap, detection, design-it-out patterns
    faithful-porting.md          # bug preservation, structural-vs-behavioral deviations
    adversarial-verify.md        # verifier brief template, hunt list, power-check requirement
    regression-gate.md           # signature-set construction, coverage-boundary reporting
    dispatch-templates.md        # port / verify / record prompt templates
```

**Three non-negotiables** for `SKILL.md`, each earning its place from a specific near-miss
above:

1. **Audit for an existing divergent implementation before assuming greenfield.**
2. **The verifier is blind to the porter's rationale, and refutes rather than reviews.**
3. **The gate is exact signature-set reproduction, not match rate. An accidental fix is a
   defect.**

---

## Open questions for review

1. **Is one run enough to generalize?** The phase structure is derived from a single port
   of a rule-matching grammar engine with an unusually good external oracle (~37k
   validated ligands, plus reference C compiled standalone). Ports without a corpus-scale
   oracle cannot run Phase 5 as described, and I do not know what substitutes. Possibly
   the skill should refuse to start without one, and say so.

2. **Where does this end and `jax-port` / `customizing-tools` begin?** `customizing-tools`
   already owns the rewrite-vs-depend decision and explicitly mentions Python→Rust
   migration. Is `rust-port` a downstream skill, a reference file inside
   `customizing-tools`, or does the jury phase belong to that skill and only Phases 2–7
   here?

3. **Should parallel per-module fan-out be the default, or the exception?** It bought real
   wall-clock speed and 13 independent verifications. It also cost: two duplicated ports,
   one thrown-away design, a broken shared `Cargo.toml`, a silently-reverted fix, and five
   duplicated build investigations. A sequenced shared-types-first phase plus a narrower
   parallel fan-out might dominate. This run has no counterfactual.

4. **How is the conditional cutover kept alive?** Declining cutover was correct here, and
   the result is that a verified-exact port is dead code while the wrong implementation it
   replaces stays live at four call sites. "Correctly declined" needs a named owner and a
   deadline or it is indistinguishable from abandoned.

5. **Should the "drop these modules" decision be mechanically enforced?** Writing it in the
   architecture document did not work — six subagents ported the dropped modules because
   their dispatch prompts said to. This suggests the dispatch list should be *generated
   from* the architecture verdict rather than authored alongside it.
