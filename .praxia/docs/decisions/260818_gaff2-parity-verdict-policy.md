---
name: 260818_gaff2-parity-verdict-policy
description: Verdict policy and tolerances for bathos-literature-parity validation of GAFF2 atom typing implementation
metadata:
  type: decision
  task_id: 260818_ligand_extension_scope
  status: draft
---

# Decision: GAFF2 Parity Verdict Policy

**Date:** 2026-08-18  
**Status:** Draft  
**Task:** 260818_ligand_extension_scope  
**Backlog Item:** #209 (gaff2-verdict-policy)

---

## Context

proxide's GAFF2 atom typing implementation (`proxide/chem/gaff2.py`) is a port of AMBER's GAFF2 force field rules. The implementation was not developed under formal parity validation against a published reference (AmberTools/antechamber or OpenFF GAFF2 plugin) — it has been validated informally on the 17-OHP steroid and integrated into the naurmalade biosensors pipeline. However, three targeted bugs were discovered and fixed (detailed in `.praxia/docs/misc/260623_gaff2-typing-debt.md`). Before GAFF2 can be used in a Core-tier campaign (a high-confidence, claim-tier experiment), a formal literature-parity validation must establish that the implementation faithfully reproduces GAFF2 atom type assignment as described in the AMBER/OpenFF specifications.

This decision document sets the operational definitions and tolerances for that validation, using the bathos-literature-parity protocol defined in `/using-bathos` skill.

---

## Tolerance Knobs for bathos-literature-parity

### Equivalence Bound (`equivalence_bound`)

**Adoption: Explicit value = 0.0 (no numeric tolerance; exact atom-type match required)**

**Rationale:**  
GAFF2 atom type assignment is a discrete categorization task, not a continuous metric. Each atom must be assigned to exactly one of the ~100 GAFF2 atom types (e.g., `ca`, `c3`, `oh`, `hs`). There is no notion of "approximately correct" — an atom typed as `ca` (aromatic carbon) is categorically different from `c3` (sp³ carbon), and partial credit is meaningless. The equivalence bound of 0.0 means:
- Every heavy atom in the test set must be assigned the **exact** GAFF2 type
- No fuzzy matching; no tolerance bands
- Disagreement on a single atom = **FAIL** verdict (see below)

This strictness is appropriate because:
1. Incorrect typing propagates downstream to force-field parameterization (bond, angle, torsion, LJ assignments), affecting energy and dynamics
2. The test set is deliberately small and carefully curated (17-OHP, acetamide, furan, etc.), not a random ensemble where noise averaging applies
3. The reference implementations (antechamber, OpenFF) produce deterministic output for these molecules

### Reconstruction Lenses (`recon_lenses`)

**Adoption: Default set = ["math", "algo", "protocol"] with N=3**

**Rationale:**  
GAFF2 is specified via:
- **Math lens:** The GAFF2 DEF file encoding and atom-property precedence rules (e.g., ring constraints, hybridization priority) — how properties combine to define an atom class
- **Algo lens:** The sequential rule matching algorithm — which rule wins when multiple rules match, and how DEF file line order breaks ties
- **Protocol lens:** The full workflow — molecular graph construction, aromaticity detection, constraint parsing, and the order of rule evaluation phases

All three are needed because an error in any one lens can produce incorrect types. N=3 reconstructors ensures independent discovery of ambiguities.

### Attack Lenses (`attack_lenses`)

**Adoption: Default set = ["stats", "struct"] with M=3**

**Rationale:**  
GAFF2 typing is sensitive to:
- **Stats lens:** Empirical validation — do our types match the reference output on a held-out test set? Statistical disagreement signals mechanism defects
- **Struct lens:** Structural/algorithmic validity — does the implementation correctly parse DEF rules, handle priority precedence, and apply aromatic bond detection? Structural bugs are mechanism-nullifying

We **omit** "hyper" (hyperparameter tuning) because GAFF2 has no learned parameters — it is a rule-based system with no hyperparameters to tune.

M=3 attackers ensures independent refutation attempts.

---

## Operational Definitions: PARITY / PARTIAL / FAIL

### PARITY (Verdict)

**Condition:** The GAFF2 implementation faithfully reproduces the published GAFF2 specification.

**Evidence triggers PARITY:**
- All N=3 reconstruction lenses independently converge on the same interpretation of the GAFF2 method (consensus on math, algo, and protocol)
- All M=3 adversarial attacks fail to find mechanism-nullifying defects (i.e., attacks can only claim "minor deviation" or "unresolved ambiguity," not "incorrect")
- Invariant tests (synthetic ground truth) confirm all core properties:
  - Every test molecule is typed identically to the reference implementation (0.0 equivalence bound)
  - Ambiguous cases (e.g., delocalized carbons in amides) are handled per the DEF file precedence rules
  - Edge cases (exocyclic double bonds, three-membered rings) are typed correctly
- No unresolved ambiguities in the paper text that would justify deviations

**Downstream impact:**  
- `[confounds.reference_parity]` marked as **controlled**
- Core-tier campaigns may proceed
- GAFF2 is eligible for production use in naurmalade and other pipelines

### PARTIAL (Verdict)

**Condition:** The GAFF2 implementation reproduces the core mechanism faithfully, but contains controlled deviations documented as acceptable trade-offs or unresolved paper ambiguities.

**Evidence triggers PARTIAL:**
- Reconstructions mostly agree (≥2 of N=3 converge); minority lens reports an unresolved ambiguity in the paper text
- Adversarial attackers find deviations but:
  - All deviations are mapped to a specific known issue (e.g., the h_ew bond-type checking bug from item 1 in 260623_gaff2-typing-debt.md)
  - Deviations do not affect the core typing logic for the test set (e.g., only affect rare edge cases not in the test molecules)
  - A documented fix exists (code change or explicit decision to accept the deviation)
- Invariant tests confirm the core properties for all test molecules where the deviation does not apply
- The deviation is documented in the sidecar postmortem with a clear remediation plan

**Downstream impact:**  
- `[confounds.reference_parity]` marked as **partially controlled**
- Core-tier campaigns **may proceed only if**:
  - The documented deviation is acknowledged in the campaign's claim block and confirmed by human sign-off
  - Proof is provided that the deviation does not affect the molecules/properties the campaign depends on
  - A time-bound remediation plan is recorded (e.g., "fix item 1 from 260623_gaff2-typing-debt.md before Q3 2026 campaign release")
- GAFF2 is eligible for production use **with documented caveats**

### FAIL (Verdict)

**Condition:** The GAFF2 implementation diverges significantly from the published specification, in a way that affects the core typing logic and is not remediable without substantial code changes.

**Evidence triggers FAIL:**
- ≥2 of N=3 reconstructors disagree on the core interpretation of GAFF2
- ≥2 of M=3 adversarial attacks find mechanism-nullifying defects (e.g., "aromatic bonds are not detected," "DEF precedence is inverted," "h_ew constraints are silently ignored")
- Invariant tests fail on one or more test molecules, indicating incorrect atom types
- The failure is in a critical path of the algorithm (not an edge case)

**Downstream impact:**  
- `[confounds.reference_parity]` marked as **uncontrolled**
- Core-tier campaigns **are blocked** until GAFF2 is remediated
- A time-bound remediation plan must be recorded (see FAIL remediation path below)

---

## Approver

[NEEDS HUMAN SIGN-OFF: name the approver for an accepted-PARTIAL verdict]

---

## FAIL Remediation Path

If the parity run returns a FAIL verdict:

1. **Root-cause analysis** — The postmortem from Phase 5 identifies the mechanism-nullifying defect(s). Work with the investigative agents to trace the defect into the code (`proxide/chem/gaff2.py`).

2. **Code fix or explicit re-derivation** — Either:
   - **Option A (Code fix):** Patch `proxide/chem/gaff2.py` to restore fidelity to the specification. Examples:
     - Implement h_ew bond-type checking (item 1 from 260623_gaff2-typing-debt.md)
     - Fix aromatic bond detection if it was identified as faulty
     - Correct DEF rule precedence logic
   - **Option B (Specification re-derivation):** If the "specification" in the paper is genuinely ambiguous or incompletely specified, consult the original AMBER/OpenFF source code or authors. Document the ambiguity in `.praxia/docs/decisions/` and update the parity test to reflect the **actual** GAFF2 behavior (not the paper's description).

3. **Re-run the parity validation** — Once the fix lands, re-run the full 5-phase literature-parity protocol to confirm the defect is resolved. This is not a simple re-test of invariant cases; the full protocol must run.

4. **Owner and time-bound** — Assign this work to a specific person (default: the person who authored the naurmalade GAFF2 integration, or the proxide maintainer if no one else is available). Set a time-bound:
   - **P1 (blocks Core-tier campaigns):** Fix by 2026-09-30 (6 weeks from now)
   - **P2 (defers Core-tier campaigns but does not block infrastructure):** Fix by 2026-10-31

5. **Record the fix in a new decision document** — Once the code fix and re-validation are complete, write a follow-up decision document (e.g., `260830_gaff2-remediation-complete.md`) summarizing the defect, the fix, and the re-validation results.

---

## Recommendation: h_ew Typing Fix and Core-Tier Campaign Prerequisite

[NEEDS HUMAN SIGN-OFF: decision on whether item 1 (h_ew bond-type checking) from `.praxia/docs/misc/260623_gaff2-typing-debt.md` must land before Core-tier campaign execution]

The h_ew field encodes bond-type patterns (2DL = delocalized double bonds, 1DB,0DL = one double + zero delocalized, 3sb = three single bonds) that disambiguate otherwise identical GAFF2 rules. The current implementation in `proxide/chem/gaff2.py` **does not check h_ew** — all matching rules are treated as equivalent from a type-assignment standpoint.

**Impact assessment:**
- **Scope of molecules affected:** Molecules with delocalized bonding (amides, enolates, resonance-stabilized carbanions) are at risk. Small drug-like molecules and proteins typically avoid these extreme cases.
- **Current test set:** 17-OHP (steroid with conjugated enone), acetamide (amide C=O), furan (aromatic), and acetamide all contain delocalized or special bonding. If the parity run on this set passes with h_ew **unchecked**, it means either:
  1. The DEF file's rule ordering happens to give correct types even without h_ew checks (luck, or the first matching rule is the correct one)
  2. The test molecules don't exercise h_ew disambiguations (they're too simple)
- **Risk to Core-tier campaigns:** If a later pipeline uses GAFF2 on a molecule with significant delocalization (e.g., a ligand with an amide or enolate), the lack of h_ew checking could produce silent mistyping.

**Recommendation:**

**Option A (Conservative — h_ew fix is mandatory prerequisite):**  
Item 1 from 260623_gaff2-typing-debt.md (implement h_ew bond-type matching) **must land and be validated** before Core-tier campaign execution. This ensures the typing is robust against delocalized bonding patterns. Estimated effort: 2–3 days. Time-bound: 2026-09-15.

**Option B (Permissive — h_ew fix deferred, FAIL treated as expected signal):**  
If the parity run flags h_ew-related mistyping as a FAIL, treat it as an **expected, non-blocking signal** that documents a known limitation. Core-tier campaigns may proceed with an explicit caveat: "GAFF2 has not been validated on molecules with delocalized bonding; ligands with amides or enolates should use SMIRNOFF instead." This is a trade-off: faster campaign startup vs. later discovery of typing bugs on complex ligands.

**[NEEDS HUMAN SIGN-OFF: choose Option A or Option B and provide a name/email of who made the decision]**

---

## Summary

This decision document establishes:

1. **Tolerance policy:** Exact atom-type matching (equivalence_bound = 0.0), N=3 reconstructors, M=3 attackers
2. **Verdict criteria:** PARITY (faithful reproduction), PARTIAL (controlled deviations with fix plan), FAIL (mechanism-nullifying defects → blocked campaigns)
3. **Remediation:** Assign owner, 6-week time-bound, re-run full protocol on fix
4. **h_ew decision:** Explicit recommendation pending human sign-off (Option A: mandatory, Option B: deferred)

The parity run is scheduled to begin 2026-08-20 (T+2 days from this decision) and conclude by 2026-08-30 (T+12 days), feeding into the F3 campaign-submit gate.

---

## Related Documents

- `.praxia/docs/misc/260623_gaff2-typing-debt.md` — Technical debt and the three bugs fixed to date
- `using-bathos` skill — Full bathos-literature-parity protocol and sidecar format
- `/using-bathos`: Literature-parity validation workflow (Phase 1–5, evidence channels, grading cap-lattice)
- Claim-tier gate documentation — How `[confounds.reference_parity]` verdicts control downstream campaign progression
