---
name: 260602_rotlib-notebook-plan
description: NotebookLM research notebook plan for rotamer library and confind — sources, prompts, and expected outputs to ground project direction
metadata:
  type: reference
  task_id: 260602_rotlib-notebook-plan
  status: draft-v2 (post oracle critique #1 — pending oracle critique #2)
---

# NotebookLM Research Notebook Plan: Rotamer Libraries & Contact Degree

**Date**: 2026-06-02
**Status**: COMPLETE — NLM queries run and synthesis written (2026-06-02)
**Feeds into**: Project direction decisions for proxide-rotlib + proxide-confind extensions

**Changes from v1** (oracle critique #1):
- Split into two notebooks (A: rotlib theory, B: ConFind + design)
- Added ConFind primary citation (Zheng/Zhang/Grigoryan 2014) to Notebook B
- Added chi-angle convention source to Notebook A
- Fixed Mosaist GitHub URL to specific blob path
- Reformulated Prompts 4, 6, 8, 11

**Changes from v2** (oracle critique #2):
- Added Mackenzie et al. 2016 PNAS as fallback P1 ConFind citation alongside Zheng 2014
- Tightened Prompt A5 to name symmetric residues (PHE, TYR, ASP, GLU) explicitly
- Narrowed Prompt B5 to Rosetta-packer vs. dTERMen comparison framing

**Citation correction** (web search verification):
- ConFind primary cite is **Zheng & Grigoryan (2017) PLoS ONE 12(5): e0178272** — NOT 2014 Structure
- Confirmed directly from grigoryanlab.org/confind/ tool page
- Mackenzie 2016 PNAS confirmed at pnas.org/doi/10.1073/pnas.1607178113
- dTERMen paper: Zhou, Panaitiu & Grigoryan (2020) PNAS 117(2):1059–1068 added as Notebook B source

**Execution** (2026-06-02):
- Notebook A ID: 171c5c8b-8bae-48c1-9e6b-6cb3a45b7a8a
- Notebook B ID: a2302b01-05b9-44b3-af62-58ea2e892298
- Notebook A sources: 8 URL + 2 file + 1 text (rotlib code) = 11 sources
- Notebook B sources: 9 URL + 2 file + 1 text (confind code) = 12 sources

---

## 0. Purpose

This plan grounds our implementation work in the scientific and algorithmic literature.
Both proxide-rotlib and proxide-confind are ports of Mosaist (MSL) — but Mosaist derives
from decades of published protein biophysics. Understanding the original papers lets us make
better decisions on open questions (PRO cis-trans, bin resolution, downstream design
applications) without re-deriving them from C++ source alone.

Two notebooks, not one: the source sets and prompt themes are distinct enough that a single
large notebook would dilute the synthesis quality.

---

## Notebook A: "proxide: Rotamer Library Theory"

Covers backbone-dependent rotamer library science — how libraries are built, how phi/psi
binning works, how sidechain geometry is represented, and what edge cases (GLY, PRO) require
special handling.

### A.1 Sources

**Papers (add as DOI links or PDFs):**

| Priority | Citation | Why |
|----------|----------|-----|
| P1 | **Shapovalov & Dunbrack (2011)** — "A Smoothed Backbone-Dependent Rotamer Library for Proteins Derived from Adaptive Kernel Density Estimates and Regressions." *Structure* 19(6):844–858. DOI: 10.1016/j.str.2011.03.019 | Current standard BBdep library; defines the .lib format we load; smoothing over sparse phi/psi regions |
| P1 | **Dunbrack & Cohen (1997)** — "Bayesian statistical analysis of protein side-chain rotamer preferences." *Protein Science* 6(8):1661–1681. DOI: 10.1002/pro.5560060807 | Foundational theory: phi/psi bin definition, rotamer population estimation, why BB-dependent matters |
| P1 | **Lovell et al. (2000)** — "The Penultimate Rotamer Library." *Proteins* 40(3):389–408. DOI: 10.1002/1097-0134(20000815)40:3<389::AID-PROT50>3.0.CO;2-2 | Alternative library; contrast on grid resolution choices and coverage |
| P2 | **Dunbrack (2002)** — "Rotamer Libraries in the 21st Century." *Current Opinion in Structural Biology* 12(4):431–440. DOI: 10.1016/S0959-440X(02)00344-5 | Review; GLY/PRO special handling rationale; backbone-independent vs. backbone-dependent tradeoffs |
| P2 | **Kulp et al. (2012)** — "Structural Informatics, Modeling, and Design with an Open-Source Molecular Software Library (MSL)." *J Comp Chem* 33(20):1645–1661. DOI: 10.1002/jcc.22968 | MSL paper — RotamerLibrary API, Frame/Transform conventions; explains what proxide-rotlib ports |

**Internal docs (add as text):**

| File | Why |
|------|-----|
| `.praxia/docs/specs/260529_rotlib.md` | Binary format spec, Frame/Transform algorithm, backbone_bin logic |
| `.praxia/docs/handoffs/260601_rotlib-fixture-expansion.md` | Known gaps and deferred work |

**Source code (add as text):**

| File | Why |
|------|-----|
| `crates/proxide-rotlib/src/rotlib.rs` | Binary parser + AaEntry data model |
| `crates/proxide-rotlib/src/binning.rs` | Circular grid nearest-neighbor lookup |
| `crates/proxide-rotlib/src/frame.rs` | Frame/Transform rigid-body placement math |
| `crates/proxide-rotlib/src/sidechain.rs` | doNotCountCB rule |

**Web sources:**

| URL | Why |
|-----|-----|
| `https://dunbrack.fccc.edu/bbdep2010/` | BBdep2010 library page — file format docs, version notes |
| `https://github.com/Grigoryanlab/Mosaist/blob/master/README.md` | Mosaist README — library usage, ConFind CLI; specific blob path to ensure indexing |
| `https://www.chem.qmul.ac.uk/iupac/misc/ppep.html` | IUPAC 1970 chi-angle nomenclature — canonical definition of χ1/χ2/χ3/χ4 dihedral conventions |

### A.2 Source Addition Order

1. Shapovalov & Dunbrack 2011 (defines the library we load)
2. Dunbrack & Cohen 1997 (foundational theory)
3. `.praxia/docs/specs/260529_rotlib.md` (implementation grounding)
4. `crates/proxide-rotlib/src/rotlib.rs` + `binning.rs` + `frame.rs` + `sidechain.rs`
5. Lovell 2000 (comparative context)
6. Dunbrack 2002 review (strategic context)
7. Kulp 2012 (MSL architecture)
8. Dunbrack lab BBdep2010 website
9. IUPAC chi-angle page
10. Mosaist README
11. `.praxia/docs/handoffs/260601_rotlib-fixture-expansion.md`

### A.3 Research Prompts

**Prompt A1 — Grid resolution and sparse regions**:
> "The Dunbrack backbone-dependent rotamer library uses a phi/psi grid. What resolution does
> the Shapovalov 2011 library use, and how does kernel-density smoothing handle bins with
> few crystallographic observations? At what phi/psi values does sparse-region accuracy
> become a practical concern for nearest-neighbor bin lookup?"

*Feeds*: Decision on whether `find_closest_angle` is sufficient or whether edge cases
near ±180° or in rarely-sampled regions need interpolation; informs fixture selection.

**Prompt A2 — PRO cis-trans handling (Debt #67)**:
> "How do backbone-dependent rotamer libraries handle proline's cis and trans peptide bond
> isomerism? Does the Dunbrack library include separate rotamer populations for cis-PRO vs.
> trans-PRO, and when does omitting this distinction introduce meaningful error in sidechain
> placement accuracy?"

*Feeds*: Concrete spec for Debt #67 — is single-form approximation acceptable?

**Prompt A3 — GLY and ALA special cases**:
> "Why is glycine excluded from backbone-dependent rotamer libraries, and what is the
> conventional treatment for alanine's CB atom in sidechain enumeration? How do MSL and the
> Dunbrack library specify these edge cases?"

*Feeds*: Documentation anchor for `doNotCountCB` rule and GLY `na=0` sentinel.

**Prompt A4 — How MSL selects the default rotamer**:
> "In MSL's RotamerLibrary, how is the 'default' rotamer for a given amino acid and phi/psi
> bin selected — is it the highest-probability rotamer, a frequency-weighted mean, or
> something else? What statistical assumption does this selection make, and when would it
> be inappropriate to use the default rotamer for placement?"

*Feeds*: API doc for our default-bin logic; clarifies when `rot_index=0` is appropriate
vs. when a caller should enumerate all rotamers.

**Prompt A5 — Chi-angle dihedral conventions**:
> "What are the IUPAC conventions for chi-angle (χ1, χ2, χ3, χ4) measurement in protein
> sidechains? How do these relate to the rotamer dihedral angles stored in the Dunbrack
> library, and how are symmetric chi angles handled for residues like PHE, TYR, ASP, and
> GLU where two sidechain conformations are physically equivalent?"

*Feeds*: Grounds chi-angle coordinate representation before we add chi-angle output to
`PlacedRotamer`.

**Prompt A6 — Fixture coverage for secondary structure**:
> "Which secondary structure types produce the most challenging phi/psi distributions for
> rotamer placement — beta turns, coiled coils, 3-10 helices, or other motifs? Where does
> backbone-dependent library accuracy degrade most, and what crystallographic examples
> would cover these cases?"

*Feeds*: Concrete PDB fixture list for `rotlib-fixture-expansion`.

---

## Notebook B: "proxide: ConFind, Contact Degree & Protein Design"

Covers the ConFind contact-degree algorithm, its algorithmic basis in the literature, and
the protein design applications that motivate implementing it.

### B.1 Sources

**Papers (add as DOI links or PDFs):**

| Priority | Citation | Why |
|----------|----------|-----|
| P1 | **Zheng & Grigoryan (2017)** — "Sequence statistics of tertiary structural motifs reflect protein stability." *PLoS ONE* 12(5): e0178272. DOI: 10.1371/journal.pone.0178272. **CONFIRMED** as primary ConFind cite by grigoryanlab.org/confind/ tool page. | Primary citation for contact-degree metric; grounds the CD formula and freedom scoring used in ConFind |
| P1 | **Mackenzie, Zhou & Grigoryan (2016)** — "Tertiary alphabet for the observable protein structural universe." *PNAS* 113(47):E7438–E7447. DOI: 10.1073/pnas.1607178113 | Fallback P1 if Zheng 2014 cannot be verified; TERMs framework where contact degree is a core building block; high-confidence Grigoryan lab citation |
| P1 | **Kulp et al. (2012)** — same MSL paper as Notebook A | ConFind is part of MSL; describes API surface and algorithmic motivation |
| P2 | **Grigoryan & Keating (2008)** — "Structural specificity in coiled-coil interactions." *Current Opinion in Structural Biology* 18(4):477–483. DOI: 10.1016/j.sbi.2008.04.008 | Grigoryan lab context — where ConFind-style contact analysis is applied |
| P2 | **Bhardwaj et al. (2016)** — "Accurate de novo design of hyperstable constrained peptides." *Nature* 538:329–335. DOI: 10.1038/nature19791 | Exemplar downstream application using backbone-dependent rotamers in de novo design |
| P3 | **Grigoryan & DeGrado (2011)** — "Probing Designability via a Generalized Model of Helical Bundle Geometry." *J Mol Biol* 405(4):1079–1100. DOI: 10.1016/j.jmb.2010.08.058 | Grigoryan lab design methodology; shows how contact-based metrics feed into designability scoring |

**Internal docs (add as text):**

| File | Why |
|------|-----|
| `.praxia/docs/specs/260529_confind.md` | Three-phase ConFind spec, crate boundaries, parity-test semantics |
| `.praxia/docs/notes/260529_confind_model.md` | Formal mstcondeg.cpp derivation — parameters, CD formula, freedom types, interference |

**Source code (add as text):**

| File | Why |
|------|-----|
| `crates/proxide-confind/src/cache.rs` | Phase A: backbone pruning + proximity grid |
| `crates/proxide-confind/src/parallel.rs` | Phase B/C: cross-rotamer enumeration + freedom |

**Note on Mosaist C++ sources**: `mstcondeg.cpp` (~634 lines) is algorithm provenance but
is better referenced as a DOI/GitHub link than ingested as a text source — NotebookLM
will synthesize it poorly vs. the prose model notes. Do not add the raw C++.

### B.2 Source Addition Order

1. Zheng/Zhang/Grigoryan 2014 (primary algorithm citation — if verified)
2. Kulp 2012 (MSL context)
3. `.praxia/docs/notes/260529_confind_model.md` (algorithmic model)
4. `.praxia/docs/specs/260529_confind.md` (implementation spec)
5. `crates/proxide-confind/src/cache.rs` + `parallel.rs`
6. Grigoryan & Keating 2008 (application context)
7. Grigoryan & DeGrado 2011 (design methodology)
8. Bhardwaj 2016 (exemplar de novo design application)

### B.3 Research Prompts

**Prompt B1 — Contact degree vs. alternative metrics**:
> "The ConFind algorithm defines contact degree between two residues as the probability of
> sidechain-sidechain atom clashes across all rotamer pairs. How does this metric compare to
> other residue-contact measures (buried surface area, van der Waals energy, heavy-atom
> distance)? What information does contact degree capture that distance-based metrics miss,
> and in what design contexts is the difference meaningful?"

*Feeds*: Articulates when confind output is most useful; grounds API documentation.

**Prompt B2 — Freedom score calibration**:
> "In the ConFind algorithm, freedom scores are computed using the loCollProbCut and
> hiCollProbCut thresholds. Is this threshold scheme calibrated against any experimental
> measure of residue conformational flexibility (B-factors, NMR order parameters), or is
> it empirically tuned? What does a freedom near 0 vs. near 1 predict about a residue's
> behavior in protein design?"

*Feeds*: Anchors `confind::Freedom` documentation to experimental basis; answers whether
the constants are tunable.

**Prompt B3 — Backbone pruning rationale**:
> "Phase A of ConFind prunes rotamers that clash with the backbone before computing
> inter-residue contacts. What is the physical justification for this, and how does
> backbone-clash pruning at 2.0 Å affect contact degree accuracy vs. a brute-force
> all-rotamer approach?"

*Feeds*: Grounds `CLASH_DIST = 2.0 Å` as tunable vs. fixed; clarifies if Phase A can
be skipped for faster approximations.

**Prompt B4 — Sequence constraint propagation**:
> "How do protein design tools propagate sequence constraints across a protein structure?
> Specifically, when one position is fixed to a specific amino acid, how does that constraint
> propagate to neighboring positions via contact-degree information, and what is the
> algorithmic basis for the constrained-contacts computation?"

*Feeds*: ConFind's `seq_const_contacts` mode is tested but not fully documented; grounds
the constraint semantics before extension.

**Prompt B5 — Repacking and sequence design pipelines**:
> "What does the Rosetta packer do with backbone-dependent rotamer libraries and pairwise
> contact information, and how does the dTERMen / MASTER design loop (Grigoryan lab) use
> the same inputs differently? Compare the two pipelines from rotamer enumeration through
> contact scoring to sequence selection, highlighting where they diverge."

*Feeds*: Concrete roadmap input for the next crate after confind — the Grigoryan-lab
pipeline (dTERMen) is the more relevant downstream consumer since ConFind lives in Mosaist.

**Prompt B6 — Multi-rotamer library strategies**:
> "In protein design software, is it common to use a single backbone-dependent rotamer
> library or multiple libraries with different resolution or smoothing strategies? When would
> a design tool switch between Dunbrack 2002 and Dunbrack 2011 libraries, or between
> backbone-dependent and backbone-independent representations?"

*Feeds*: Architecture decision: `RotlibRegistry` abstraction now vs. deferred (Debt #67
multi-rotlib note).

---

## 4. Expected Outputs from Both Notebooks

| Prompt | Expected project output |
|--------|------------------------|
| A1 (grid resolution) | Decision: nearest-neighbor sufficient or sparse-region interpolation needed? |
| A2 (PRO cis-trans) | Spec for Debt #67: single-form acceptable or must fix; what data? |
| A4 (default rotamer) | API doc for default-bin selection semantics |
| A5 (chi-angle) | Grounded chi-angle representation before adding chi output to PlacedRotamer |
| A6 (fixture coverage) | Concrete PDB fixture list for rotlib-fixture-expansion |
| B2 (freedom calibration) | Freedom score documentation and whether thresholds are tunable |
| B5 (repacking applications) | Roadmap: next crate direction |
| B6 (multi-rotlib) | Architecture decision on RotlibRegistry |

---

## 5. Open Questions for Oracle Critique #2

1. **ConFind citation confidence**: The Zheng/Zhang/Grigoryan 2014 *Structure* 23:961
   candidate is medium-confidence from oracle critique #1. Should we add it as P1 with a
   verification note, or hold Notebook B until the citation is confirmed?

2. **Grigoryan & Keating 2008** — is this the right "context" paper for ConFind, or is
   there a more direct citation where the contact-degree formula first appears?

3. **Prompt B5 specificity**: Is "what algorithms use rotamer enumeration + contact scoring"
   the right framing, or should it be narrowed to "what does Rosetta's packer do with these
   inputs" to get a more actionable answer?
