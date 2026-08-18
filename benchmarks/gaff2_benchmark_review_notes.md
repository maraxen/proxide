# GAFF2 Benchmark Curation - Self-Critique Pass

**Task ID:** 260818_ligand_extension_scope  
**Date:** 2026-08-18  
**Tier:** Core Tier (10 molecules)  
**Author:** Claude Code (self-critique, not independent review)

## Summary

This document records a **self-critique pass** on the curated core tier molecules against the 4 known GAFF2 gaps plus explicitly required bridgehead aromatic and cross-conjugated polyene coverage. 

**Critical caveat:** This self-critique does NOT substitute for genuinely independent review. A separate reviewer (human or independent agent) MUST re-evaluate this list before the actual parity validation campaign runs. Self-evaluation has inherent blind spots; this pass documents what I could identify, not what I missed.

---

## Gap Coverage Analysis

### Gap 1: h_ew (Hydrogen in Electron-Withdrawing Environment)

**Target:** Hydrogen on nitrogen/oxygen in electron-withdrawing contexts (amides, enolates, etc.)

**Molecules claiming to target this gap:**
- **Formamide** (NC=O): ✓ Minimal amide; H on N bonded to C=O
- **N-methylacetamide** (CC(=O)NC): ✓ Secondary amide variant; H on N adjacent to C=O

**Self-critique confidence: MODERATE**
- Both molecules have hydrogen on nitrogen bonded to carbonyl (C=O), which is the classic EW environment
- Formamide is minimal/pure; N-methylacetamide adds methyl substitution on both N and C=O sides
- **Potential gaps I'm not confident about:**
  - Enolate forms are not represented (e.g., keto-enol tautomerism). If the gap is specifically about enolate H, these two amides may not fully exercise it.
  - No carbamate or urea structures (N bonded to C(=O)O or C(=O)N), which might have different h_ew typing
  - Only primary/secondary amides; no tertiary amides (but tertiary have no N-H)

**Flag for independent review:** Verify that amides are the intended h_ew test case; if enolates or other EW environments are the main gap, supplement with additional structures.

---

### Gap 2: cc/cd/ce/cf Conjugated Carbon Systems

**Target:** sp2 carbons in conjugated systems; GAFF2 spec defines cc (ring conjugation), cd (terminal ring), ce (chain conjugation), cf (other conjugation contexts).

**Molecules claiming to target this gap:**
- **1,3-Butadiene** (C=CC=C): ✓ Conjugated diene chain; central C's will be cc or ce, terminal C's may be cc or cd depending on implementation
- **Acrolein** (C=CC=O): ✓ α,β-unsaturated aldehyde; combines C=C-C=O conjugation with carbonyl
- **Thiophene** (c1ccsc1): ~ Aromatic heterocycle (conjugation within aromatic system, but aromatic typing may mask conjugation gap)
- **Naphthalene** (c1ccc2ccccc2c1), **Indene**, **Anthracene**: ~ Aromatic fused systems (again, aromatic typing may dominate)

**Self-critique confidence: MODERATE-LOW**
- 1,3-Butadiene is a strong, unambiguous test case for conjugated diene (ce or cc carbons)
- Acrolein combines conjugation with carbonyl, which is good for stress-testing but may conflate two gaps
- Aromatic molecules (naphthalene, indene, thiophene, anthracene) have internal sp2 carbons that are conjugated, but they are classified under aromatic rules (cp, ca). This may NOT exercise the cc/cd/ce/cf gap if the implementation treats aromatic as a separate typing category that doesn't use cc/cd/ce/cf at all.

**Flag for independent review:** 
- Verify that aromatic fused systems actually use cc/cd/ce/cf typing or if they use separate aromatic rules (ca, cp). If the latter, these molecules do NOT test the gap.
- 1,3-Butadiene alone may be insufficient. Consider adding **1,3-cyclohexadiene** (C1=CC=CCC1) to test conjugation within a non-aromatic context (alicyclic conjugation).

---

### Gap 3: cp (Aromatic Carbon - Bridgehead)

**Target:** Aromatic carbon in bridgehead/fused-ring aromatic systems.

**Molecules claiming to target this gap:**
- **Naphthalene** (c1ccc2ccccc2c1): ✓✓ Two fused 6-rings; clear bridgehead carbons at positions 4a/8a (junction points)
- **Indene** (C1=CC=C2C(=C1)C=C2): ✓ Fused 5+6 rings; has bridgehead carbons in the 5-ring portion
- **Benzene** (c1ccccc1): ✓ Single aromatic ring; baseline for cp (all carbons are cp)
- **Thiophene** (c1ccsc1): ✓ Aromatic heterocycle; cp in non-6-membered context
- **Anthracene** (c1cc2ccccc2cc1): ✓✓ Three fused 6-rings; multiple bridgehead carbons

**Self-critique confidence: HIGH**
- Naphthalene and Anthracene have unambiguous bridgehead aromatic carbons (the junction points between fused rings)
- All molecules are confidently aromatic and should have cp (or aromatic-variant) carbons
- This gap appears well-covered

**No flags for independent review on cp gap coverage.**

---

### Gap 4: Multi-Ring (Multi-Wildcard ATD Rules)

**Target:** Complex topologies stressing multi-wildcard ATD rules; fused ring systems with complex neighbor patterns.

**Molecules claiming to target this gap:**
- **Naphthalene** (c1ccc2ccccc2c1): ✓ Two fused 6-rings
- **Indene** (C1=CC=C2C(=C1)C=C2): ✓ Fused 5+6 rings (asymmetric)
- **Anthracene** (c1cc2ccccc2cc1): ✓✓ Three fused 6-rings (larger, more complex)
- **Thiophene** (c1ccsc1): ~ 5-membered aromatic (less multi-ring stress than fused systems)

**Self-critique confidence: MODERATE**
- Naphthalene and Anthracene are good test cases for multi-ring topology
- Indene adds asymmetry (5-membered + 6-membered fusion) which may stress ATD differently
- **Potential gaps I'm not confident about:**
  - No tricyclic non-aromatic systems (e.g., norbornane derivatives, steroid skeletons). If the gap is specifically in sp3-dominated multi-ring systems, this list is weak.
  - No cross-ring bonds (e.g., adamantane, which has bridging bonds between rings). These might stress ATD rules differently than simple fused rings.
  - All multi-ring molecules in this list are aromatic; alicyclic polycycles might be different.

**Flag for independent review:** Consider adding a non-aromatic multi-ring molecule (e.g., **norbornane** C1CC2CCC1C2) to test sp3-dominated topology if the gap is not aromatic-specific.

---

## Adversarial Requirements

### Requirement 1: Bridgehead Aromatic

**Definition:** An aromatic carbon that is part of a bicyclic/fused system at the ring junction point.

**Status:** ✓✓ WELL COVERED
- **Naphthalene** and **Anthracene** have unambiguous bridgehead aromatic carbons
- **Indene** has bridgehead carbons (though the 5-membered ring bridgehead is sp2, not sp3)

### Requirement 2: Cross-Conjugated Polyene

**Definition:** Multiple C=C bonds where conjugation is not continuous (separated by sp3 carbons or in branched arrangements).

**Status:** ✓ COVERED
- **1,5-Hexadiene** (C=CCCC=C) explicitly has two C=C bonds separated by an sp3 linker (C-C-C)
- This is unambiguous cross-conjugation (not continuous)

---

## Summary of Confidence Levels

| Gap | Coverage | Confidence | Notes |
|-----|----------|------------|-------|
| h_ew | 2 molecules (formamide, N-methylacetamide) | MODERATE | Strong for amides; may miss enolates or other EW contexts |
| cc/cd/ce/cf | 4-5 molecules (1,3-butadiene strong; aromatic weak) | MODERATE-LOW | Non-aromatic conjugation well-tested; aromatic conjugation unclear |
| cp (bridgehead) | 5 molecules (naphthalene, indene, anthracene, thiophene, benzene) | HIGH | Excellent coverage; aromatic is well-defined |
| multi-ring | 4 molecules (naphthalene, indene, anthracene, thiophene) | MODERATE | Aromatic fused systems well-covered; sp3 polycycles missing |

---

## Gaps in This Self-Critique (Known Unknowns)

1. **No empirical validation against proxide yet.** This curation is theoretical; actual GAFF2 typing results may reveal issues I didn't anticipate.

2. **Unknown: Implementation details of proxide's GAFF2 module.** If proxide uses simplified ATD rules or deviates from the ATOMTYPE_GFF2.DEF spec in ways I don't know, this list might miss key stress points.

3. **Unknown: Exact definition of cc/cd/ce/cf gap.** I assumed it's about typing conjugated sp2 carbons; if it's something else, my coverage is wrong.

4. **Unknown: Whether h_ew is specifically amide or broader.** Amides are strong candidates, but enolates, ynamides, or other EW environments might be the real gap.

5. **Unknown: RDKit SMILES rendering.** I assumed RDKit would render these SMILES correctly; if any SMILES is malformed or renders to wrong structure, the test fails silently.

---

## Recommendations for Independent Review

**Before running the actual parity validation campaign, an independent reviewer MUST:**

1. **Verify SMILES correctness:** Render each molecule in RDKit and visually confirm structure (not just valence check).

2. **Cross-check against proxide's known issues:** If proxide has GitHub issues or PRs mentioning GAFF2 gaps, verify this list actually targets those issues.

3. **Run preliminary GAFF2 typing:** Use proxide to parameterize each molecule and confirm:
   - Formamide / N-methylacetamide get correct h_ew handling
   - 1,3-Butadiene uses cc/cd/ce/cf typing (or whatever the gap is)
   - Naphthalene / Indene / Anthracene get cp bridgehead typing

4. **Consider adding supplementary molecules** if any gap is under-covered:
   - h_ew: Add an enolate or carbamate if needed
   - cc/cd/ce/cf: Add 1,3-cyclohexadiene or a non-aromatic conjugated system
   - multi-ring: Add norbornane or adamantane for sp3-dominated topology

5. **Sanity-check against literature:** Cross-reference this curation against published GAFF2 benchmark molecules (if they exist) to ensure real-world relevance.

---

## Conclusion

This core tier of 10 molecules provides **reasonable coverage** of the 4 GAFF2 gaps and explicitly includes bridgehead aromatic and cross-conjugated polyene molecules. However, this is a **self-critique only** and has known blind spots (see "Gaps in This Self-Critique" above).

**This list is NOT FINAL for the parity campaign.** An independent review MUST complete before actual validation runs.

---

## Supplement Tier Notes

The core tier is supplemented by:

1. **AmberMD GAFF2 geostd tarball:** ~29,000 pre-parameterized ligands (download and documentation TBD; see gaff2_parity_molecules.yaml)

2. **Project-specific molecules:**
   - **17-OHP (17-hydroxyprogesterone):** From biosensors idea-019 (steroid with h_ew and multi-ring stress)
   - **naurmalade ligands:** From gaff2-paper-inputs branch (if relevant to gaps)

These supplement the core tier for cross-validation and real-world relevance testing.

---

**Generated:** 2026-08-18  
**Task:** 260818_ligand_extension_scope (backlog item #211)  
**Status:** Ready for independent review
