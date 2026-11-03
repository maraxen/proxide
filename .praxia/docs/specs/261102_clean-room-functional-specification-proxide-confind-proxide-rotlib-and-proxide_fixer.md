---
title: 'Clean-Room Functional Specification: proxide-confind, proxide-rotlib, and proxide_fixer'
description: Clean-room behavioral specification for proxide-confind, proxide-rotlib, and proxide_fixer under MIT license, covering mathematical formulas with citations, a typed configuration section (Cβ clash policy, crowdedness mode, freedom mode, scope, cutoffs), Option<f64> dihedrals (#1897), IUPAC omega (#1903), chain semantics (#1904), fixer repack tests (#1891), and CCD parser fallbacks (#1933).
status: draft
task_id: 261102_proxide-mosaist-cleanroom
date: '261102'
backlog_ids: ''
adversarial_review: ''
---
# Clean-Room Functional Specification: `proxide-confind`, `proxide-rotlib`, and `proxide_fixer`

**Document ID:** `SPEC-261102-CONFIND-CLEANROOM`  
**Author:** Spec Team Lead (`spec_author`) & Spec Defender (`spec_defender_cleanroom`)  
**Date:** 2026-11-02 (Revision 2.0: 2026-11-03)  
**Status:** REVISED SPECIFICATION (Revision 2.0 — owner decisions incorporated; composable configuration)  
**Target Crates:** `crates/proxide-confind`, `crates/proxide-rotlib`, `crates/proxide_fixer`  

---

## 1. Overview & Architecture

### 1.1 Objective & Principles
This specification defines, from peer-reviewed scientific literature and public Rust type signatures only, the behavior of `proxide-confind`, `proxide-rotlib`, and `proxide_fixer`. The crates are pure MIT / Apache-2.0 licensed software suitable for distribution on `crates.io`.

Principles:
1. **Behavioral Specification Only**: this document gives mathematical equations, geometric transformations, physical invariants, data schemas, public APIs, input preconditions, and postconditions. Implementation structure is the implementer's choice.
2. **Literature-Only Sources**: every constant and formula cites peer-reviewed literature (Zheng et al. 2015, Zheng & Grigoryan 2017, Dunbrack & Cohen 1997, Shapovalov & Dunbrack 2011, Engh & Huber 1991, Bondi 1964) or is defined here as pure mathematics.
3. **Single Rotamer Format**: `proxide-rotlib` loads the Protobuf rotamer library (`load_pb`, `load_pb_zst`) only.
4. **Independent Test Invariants**: tests use physical conservation laws, geometric invariants, synthetic structural generators, committed public-domain structures, and the committed Dunbrack Protobuf asset. No externally generated golden tables are embedded.
5. **Composability & Configurability**: behaviors that admit more than one physically defensible definition are exposed as typed, documented options (§3.9) that compose freely, with sensible defaults.

---

### 1.2 Crate Boundaries

```
+-----------------------------------------------------------------------------------------+
|                                  PROXIDE WORKSPACE                                      |
+-----------------------------------------------------------------------------------------+
|  [ crates/proxide-confind ] (MIT)                                                       |
|  * Contact degree, crowdedness, interference, structural freedom                        |
|  * Typed configuration (§3.9) and `confind` binary flags                                |
|  * Option<f64> dihedral angles (Debt #1897)                                             |
|  * IUPAC omega & segmented chain semantics (Debts #1903, #1904)                         |
|  * Depends on: proxide-rotlib, proxide-core, proxide-geometry, orx-parallel, dashmap    |
|                                                                                         |
|  [ crates/proxide-rotlib ] (MIT)                                                        |
|  * Protobuf-backed Dunbrack rotamer library engine (load_pb, load_pb_zst)               |
|  * Gram-Schmidt orthonormal coordinate frame (Dunbrack & Cohen 1997)                    |
|  * CCD parser with explicit token validation & cited fallbacks (Debt #1933)             |
|                                                                                         |
|  [ crates/proxide_fixer ] (MIT)                                                         |
|  * Atomic model prep, sidechain repacking, point mutagenesis                            |
|  * Inlined canonical amino-acid set (no dependency on proxide-confind)                  |
|  * Repack tests run against the committed Dunbrack Protobuf asset (Debt #1891)          |
+-----------------------------------------------------------------------------------------+
```

---

### 1.3 Module Requirements

| Crate | Module | Requirement |
|---|---|---|
| `proxide-confind` | `src/coords.rs` | `phi`, `psi`, `omega` are `Option<f64>`; segmented chain semantics (§3.4–§3.6). |
| `proxide-confind` | `src/precondition.rs` | Dihedral validation on `Option<f64>`; precondition reporting. |
| `proxide-confind` | `src/grid.rs` | 3D spatial hash index `ProximityGrid<T>` in f64. |
| `proxide-confind` | `src/contact_list.rs` | `ContactList` container and `CONTACT_THRESHOLD = 0.02`. |
| `proxide-confind` | `src/error.rs` | `ConFindError` enum (§3.2). |
| `proxide-confind` | `src/config.rs` | Typed configuration, validation, string parsing (§3.9). |
| `proxide-confind` | `src/cache.rs` | Phase A: sidechain placement, backbone clash pruning, residue caching (§2.4, §3.8.3). |
| `proxide-confind` | `src/parallel.rs` | Phase B: pairwise contact degree with parallel batch processing (§2.1). |
| `proxide-confind` | `src/freedom.rs` | Phase C: freedom from collision probabilities (§2.5, §3.8.2). |
| `proxide-confind` | `src/confind.rs` | Pipeline orchestration for caching, contact queries, crowdedness, freedom, interference. |
| `proxide-confind` | `src/params.rs` | Background frequencies and default cutoffs with literature citations. |
| `proxide-rotlib` | `src/rotlib.rs` | Protobuf loading (`load_pb`, `load_pb_zst`) and query APIs. |
| `proxide-rotlib` | `src/frame.rs` | Standard Gram-Schmidt orthonormalization (§2.7). |
| `proxide-rotlib` | `src/sidechain.rs` | IUPAC sidechain heavy-atom predicates (§4.5). |
| `proxide-rotlib` | `src/geometry/ccd_parser.rs` | Exact CIF token validation; recorded, cited fallbacks (Debt #1933). |
| `proxide_fixer` | `src/mutate.rs` | Inline `CANONICAL_AA_NAMES`; no `proxide-confind` dependency. |
| `proxide_fixer` | `src/repack.rs` | Repack tests use workspace Dunbrack Protobuf asset; fail loudly on corrupt load (Debt #1891). |

---

## 2. Mathematical Definitions & Equations with Academic Citations

All algorithms in `proxide-confind` and `proxide-rotlib` are derived directly from the fundamental biophysical models published in peer-reviewed scientific literature.

### 2.1 Contact Degree Formulation & Ensemble Coupling
The contact degree $c(i, j)$ quantifies the statistical probability that residue position $i$ and residue position $j$ in a fixed protein backbone conformation form steric contact across their conformational sidechain ensembles, assuming no sequence information is known a priori.

#### Theoretical Formulation
From Zheng, Zhang, & Grigoryan (2015) *Structure* (Eq. 1) and Zheng & Grigoryan (2017) *PLoS ONE* (Eq. 9):
Let $\mathcal{A}_{18}$ denote the set of 18 canonical amino acids possessing discrete rotameric sidechain degrees of freedom (excluding Glycine and Proline). For a residue position $i$ with backbone dihedrals $(\phi_i, \psi_i)$, the probability of observing amino acid identity $a \in \mathcal{A}_{18}$ in rotamer conformation $r_a \in \mathcal{R}_i(a)$ is given by:
$$P(a, r_a \mid \phi_i, \psi_i) = P(a) \cdot P(r_a \mid a, \phi_i, \psi_i)$$
where:
- $P(a)$ is the empirical background propensity of amino acid $a$ across non-redundant structural databases (Section 2.2).
- $P(r_a \mid a, \phi_i, \psi_i)$ is the backbone-dependent rotamer probability from the smoothed Dunbrack library (Dunbrack & Cohen 1997; Shapovalov & Dunbrack 2011).
- $\mathcal{R}_i(a)$ is the set of rotamers for amino acid $a$ that are sterically compatible with the local backbone environment at position $i$ (Phase A pruning, Section 2.4).

The sequence-independent pairwise contact degree $c(i, j)$ between residue positions $i$ and $j$ is the ensemble expectation of contact:
$$c(i, j) = \frac{1}{\Omega_i \, \Omega_j} \sum_{a \in \mathcal{A}_{18}} P(a) \sum_{b \in \mathcal{A}_{18}} P(b) \sum_{r_a \in \mathcal{R}_i(a)} P(r_a \mid a, \phi_i, \psi_i) \sum_{r_b \in \mathcal{R}_j(b)} P(r_b \mid b, \phi_j, \psi_j) \, w(r_a, r_b)$$
where:
- $w(r_a, r_b)$ is the binary steric contact indicator function between rotamer $r_a$ at position $i$ and rotamer $r_b$ at position $j$:
  $$w(r_a, r_b) = \begin{cases} 1 & \text{if } \exists \, k \in S_{\text{cont}}(r_a), \, l \in S_{\text{cont}}(r_b) \text{ such that } \|\vec{x}_{ik} - \vec{x}_{jl}\| \le d_{\text{cont}} \\ 0 & \text{otherwise} \end{cases}$$
- $d_{\text{cont}} = 3.0$ Å is the standard heavy-atom contact threshold for non-bonded sidechain interaction.
- $\Omega_i$ is the surviving rotamer partition function (normalization weight) at position $i$:
  $$\Omega_i = \sum_{a \in \mathcal{A}_{18}} P(a) \sum_{r_a \in \mathcal{R}_i(a)} P(r_a \mid a, \phi_i, \psi_i)$$

#### Zero Partition Function Invariant (Edge Case Handling)
If a residue position is completely occluded by the local backbone such that no rotamers survive Phase A clash pruning ($\mathcal{R}_i(a) = \emptyset$ for all $a$), its partition function is $\Omega_i = 0.0$ (crowdedness $C(i) = 1.0$). To prevent IEEE-754 division-by-zero (`NaN`), the pairwise contact degree is analytically defined as:
$$\text{If } \Omega_i = 0.0 \text{ or } \Omega_j = 0.0 \implies c(i, j) \equiv 0.0$$
The implementation must short-circuit to $0.0$ without performing float division.

#### Canonical Interaction Threshold
In the dTERMen framework for protein stability and design (Zheng & Grigoryan 2017; Mackenzie et al. 2016):
- Residue pairs satisfying:
  $$c(i, j) \ge c_{\text{threshold}} = 0.02$$
  are formally designated as **poised to interact**. Positions satisfying this threshold are included in tertiary structural motif (TERM) definitions.

---

### 2.2 Background Amino Acid Propensities ($P(a)$)
Background frequencies $P(a)$ represent the baseline marginal probability of encountering amino acid $a$ in non-redundant globular protein crystal structures.

#### Numerical Frequency Table
From Zheng, Zhang, & Grigoryan (2015) *Structure* 23(5): 961–971 (Table S1):

| Amino Acid ($a$) | Code | Propensity $P(a)$ (%) | Amino Acid ($a$) | Code | Propensity $P(a)$ (%) |
|---|---|---|---|---|---|
| **Alanine** | `ALA` | 7.73 | **Leucine** | `LEU` | 8.83 |
| **Arginine** | `ARG` | 5.03 | **Lysine** | `LYS` | 6.27 |
| **Asparagine** | `ASN` | 4.50 | **Methionine** | `MET` | 2.08 |
| **Aspartate** | `ASP` | 5.82 | **Phenylalanine** | `PHE` | 4.05 |
| **Cysteine** | `CYS` | 1.84 | **Proline** | `PRO` | 4.52 *(excluded from $\mathcal{A}_{18}$)* |
| **Glutamine** | `GLN` | 3.94 | **Serine** | `SER` | 6.13 |
| **Glutamate** | `GLU` | 6.61 | **Threonine** | `THR` | 5.53 |
| **Glycine** | `GLY` | 7.11 *(excluded from $\mathcal{A}_{18}$)* | **Tryptophan** | `TRP` | 1.51 |
| **Histidine** | `HIS` / `HSD` | 2.35 | **Tyrosine** | `TYR` | 3.54 |
| **Isoleucine** | `ILE` | 5.66 | **Valine** | `VAL` | 6.91 |

*(Total across 20 amino acids = 100.06% ≈ 100%).*

#### Exclusion Rationale for GLY and PRO in $\mathcal{A}_{18}$
1. **Glycine (`GLY`)**: Lacks a $C_\beta$ atom and sidechain torsional degrees of freedom ($\chi$). Glycine cannot form sidechain-sidechain steric contacts; including it in the rotamer summation would arbitrarily dilute contact degrees.
2. **Proline (`PRO`)**: Features a rigid five-membered pyrrolidine ring covalently linked to the backbone nitrogen ($N$), strictly constraining both $\phi$ and sidechain conformations. Proline sidechain placement requires cyclic coordinate descent ring closure rather than standard unconstrained rotamer placement.

---

### 2.3 Distance Metrics & Heavy-Atom Steric Cutoffs
All spatial calculations in `proxide-confind` are parameterized by the following physical and geometric cutoffs:

1. **Backbone Steric Clash Cutoff ($d_{\text{clash}} = 2.0$ Å)**:
   Any sidechain heavy atom placed closer than $2.0$ Å to any backbone heavy atom ($N, C_\alpha, C, O$) of any other residue (tested atom set per `CbClashPolicy`, §2.4) represents an unphysical hard steric overlap (van der Waals overlap $> 0.8$ Å) and causes that rotamer to be pruned in Phase A.
2. **Sidechain Contact Cutoff ($d_{\text{cont}} = 3.0$ Å)**:
   Two sidechain heavy atoms approaching within $3.0$ Å are defined as being in physical contact (van der Waals contact distance).
3. **Backbone Neighbor Search Cutoff ($D_{\text{cut}} = 25.0$ Å)**:
   The maximum physical span of two extended sidechains (e.g. dual Arginine residues with fully extended sidechains of length $\approx 6.5$ Å each) plus the contact threshold ($3.0$ Å) is strictly bounded by $< 20.0$ Å. Pairs of residues with $C_\alpha - C_\alpha$ Euclidean distance:
   $$\|\vec{C}\alpha_i - \vec{C}\alpha_j\| > 25.0 \text{ Å}$$
   cannot physically come into contact under any valid rotamer state. Their contact degree is analytically $0.0$ and their pairwise evaluation is omitted.
4. **Peptide Chain Break Cutoff ($d_{\text{break}} = 4.5$ Å)**:
   In a trans peptide bond, the standard distance between consecutive $C_\alpha$ atoms is approximately $3.8$ Å (Engh & Huber 1991). If consecutive $C_\alpha$ atoms in the input file exceed $4.5$ Å, a physical chain break is registered.

---


### 2.4 Residue Crowdedness ($C(i)$)
Residue crowdedness measures the degree of steric confinement imposed on residue position $i$ by the surrounding fixed protein backbone (Zheng & Grigoryan 2017, S1 Text).

**Atom sets.** For rotamer $r$ of amino acid $a$:
- $S_{\text{cont}}(r)$: sidechain heavy atoms that count for sidechain–sidechain contact, per `counts_as_sidechain` (§4.5): all non-hydrogen atoms other than $N, C_\alpha, C, O$, with $C_\beta$ retained only for alanine.
- $S_{\text{clash}}(r)$: atoms tested against backbone, selected by the `CbClashPolicy` option (§3.9):
  - `AlaOnly`: $S_{\text{clash}}(r)=S_{\text{cont}}(r)$ ($C_\beta$ tested for alanine only).
  - `AllResidues` (default): $S_{\text{clash}}(r)=S_{\text{cont}}(r)\cup\{C_\beta\}$ for every amino acid.
- $B_j=\{N,C_\alpha,C,O\}$ present for residue $j$. The own-residue backbone $B_i$ is never tested for rotamers of $i$ (bonded $C_\beta$–$N$ and $C_\beta$–$C$ distances of $\approx1.5$ Å are below $d_{\text{clash}}$).

Distance indicator, for atom sets $X,Y$: $\operatorname{near}(X,Y;d)=\mathbf 1[\exists x\in X,y\in Y:\|x-y\|\le d]$. A rotamer–rotamer or rotamer–backbone pair contributes **at most once**, however many atom pairs are within $d$.

Let $\mathcal{R}_i^{\text{lib}}(a)$ be the complete set of rotamers for amino acid $a$ tabulated at $(\phi_i,\psi_i)$. A rotamer is pruned if it clashes with the backbone of any other residue:
$$\text{clash}(r)=\max_{j\ne i}\ \operatorname{near}\big(S_{\text{clash}}(r),B_j;\,d_{\text{clash}}\big),\qquad d_{\text{clash}}=2.0\text{ Å (inclusive)}$$
$$\mathcal{R}_i(a)=\{\,r\in\mathcal{R}_i^{\text{lib}}(a)\mid \text{clash}(r)=0\,\}$$

Counts and masses (pooled over $\mathcal A_{18}$): $N_{\text{lib}}(i)=\sum_a|\mathcal R^{\text{lib}}_i(a)|$, $N_{\text{surv}}(i)=\sum_a|\mathcal R_i(a)|$, and
$$W^{\text{lib}}_i=\sum_{a}P(a)\!\!\sum_{r\in\mathcal R^{\text{lib}}_i(a)}\!\!P(r\mid a,\phi_i,\psi_i),\qquad W^{\text{surv}}_i=\sum_{a}P(a)\!\!\sum_{r\in\mathcal R_i(a)}\!\!P(r\mid a,\phi_i,\psi_i).$$

**Crowdedness modes** (`CrowdednessMode`, §3.9):

- `ProbabilityWeighted` (default): the pruned fraction of prior probability mass,
  $$C(i)=1-\frac{W^{\text{surv}}_i}{W^{\text{lib}}_i}\qquad(W^{\text{lib}}_i=0\Rightarrow C(i)=0).$$
- `PrunedCountFraction`: the pruned fraction of rotamer count,
  $$C(i)=1-\frac{N_{\text{surv}}(i)}{N_{\text{lib}}(i)}\qquad(N_{\text{lib}}(i)=0\Rightarrow C(i)=0).$$

**Properties (both modes):**
- $C(i)\in[0,1]$; $C(i)=0$: no backbone occlusion; $C(i)=1$: no rotamer fits.
- $C(i)$ does not depend on the analysis scope (§2.5) or on freedom or interference options; the backbone of every residue in the structure is tested.
- The mode changes only the value returned by `ConFind::crowdedness`. The surviving sets $\mathcal R_i(a)$ — and therefore contact degree, freedom and interference — are identical in both modes.

---

### 2.5 Residue Structural Freedom ($F(i)$)
Structural freedom quantifies the residual conformational flexibility of residue $i$ given backbone pruning and mutual sidechain crowding from neighbouring residues (Zheng & Grigoryan 2017, S1 Text). It is defined with the quantities of §2.4 and §3.8.2.

**Neighbour set.** For analysis scope $\mathcal S$ (a set of residues; default all residues of the structure): $\mathcal N(i)=\{j\in\mathcal S,\ j\ne i:\ \|C\alpha_i-C\alpha_j\|\le D_{\text{cut}}\}$.

**Collision probability** of a surviving rotamer $r\in\mathcal R_i=\bigsqcup_a\mathcal R_i(a)$ ($P(b)$ is in percent, so it is divided by $100$ to give fractional probability):
$$P_{\text{coll}}(r)=\sum_{j\in\mathcal N(i)}\ \sum_{b\in\mathcal A_{18}}\frac{P(b)}{100}\sum_{s\in\mathcal R_j(b)}P(s\mid b,\phi_j,\psi_j)\ \operatorname{near}\big(S_{\text{cont}}(r),S_{\text{cont}}(s);\,d_{\text{cont}}\big)$$
$P_{\text{coll}}$ is an unnormalised mass (not divided by the neighbour's surviving mass) and may exceed $1$.

**Threshold counts.** With $\tau_{\text{lo}}=0.5$, $\tau_{\text{hi}}=2.0$ (dimensionless) and strict inequality,
$$N_\tau(i)=\big|\{\,r\in\mathcal R_i:\ P_{\text{coll}}(r)<\tau\,\}\big|,\qquad \tau\in\{\tau_{\text{lo}},\tau_{\text{hi}}\}.$$
Uncontested rotamers ($P_{\text{coll}}=0$) are counted. Hence $0\le N_{\tau_{\text{lo}}}\le N_{\tau_{\text{hi}}}\le N_{\text{surv}}\le N_{\text{lib}}$.

**Freedom modes** (`FreedomMode`, §3.9). Freedom is count-based: every rotamer counts once irrespective of $P(a)$ and $P(r\mid a,\phi,\psi)$.
- `Rms` (default): $$F(i)=\frac{1}{N_{\text{lib}}(i)}\sqrt{\frac{N_{\tau_{\text{lo}}}(i)^2+N_{\tau_{\text{hi}}}(i)^2}{2}}$$
- `SingleThreshold`: $$F^{(1)}(i)=\frac{N_{\tau_{\text{lo}}}(i)}{N_{\text{lib}}(i)}$$

Both lie in $[0,1]$; $F^{(1)}(i)\le F(i)$. $N_{\text{lib}}(i)=0\Rightarrow F=0$ (convention; no error). The freedom modes have no effect on any quantity other than `ConFind::freedom`.

---

### 2.6 Mutual Interference ($I(i,j)$)
Mutual interference is the prior probability mass of rotamers of residue $i$ that hit the backbone of residue $j\ne i$. The sum runs over the **unpruned library** set (summing over survivors would give $I\equiv0$, since survivors clash with no backbone by construction):
$$I(i,j)=\sum_{a\in\mathcal A_{18}}\frac{P(a)}{100}\sum_{r\in\mathcal R^{\text{lib}}_i(a)}P(r\mid a,\phi_i,\psi_i)\ \operatorname{near}\big(S_{\text{clash}}(r),B_j;\,d_{\text{clash}}\big),\qquad d_{\text{clash}}=2.0\text{ Å},\ j\ne i$$
$S_{\text{clash}}$ follows `CbClashPolicy` (§2.4). A rotamer hitting several residues' backbones counts fully toward **each** of them (every clashed residue is credited), so $\sum_jI(i,j)$ may exceed the pruned mass. The full definition, optional normalised form, edge cases and tests are in §3.8.3.

---

### 2.7 Backbone Coordinate Frame & Local-to-Lab Rigid Transformation
To place rotamers onto an arbitrary protein backbone without empirical heuristics, a standard right-handed orthonormal coordinate frame is constructed at each residue position from backbone atoms $N, C_\alpha, C$ (Dunbrack & Cohen 1997; Shapovalov & Dunbrack 2011).

#### Gram-Schmidt Orthonormalization Formulation
Given Cartesian coordinates $\vec{N}, \vec{C}_\alpha, \vec{C} \in \mathbb{R}^3$:
1. **Origin**:
   $$\vec{O} = \vec{C}_\alpha$$
2. **Primary X-Axis** (aligned along the $N \to C_\alpha$ bond direction):
   $$\vec{v}_x = \vec{C}_\alpha - \vec{N}, \quad \hat{u}_x = \frac{\vec{v}_x}{\|\vec{v}_x\|}$$
3. **Out-of-Plane Z-Axis** (perpendicular to the $N-C_\alpha-C$ peptide plane):
   $$\vec{v}_c = \vec{C} - \vec{C}_\alpha$$
   $$\vec{v}_z = \vec{v}_x \times \vec{v}_c, \quad \hat{u}_z = \frac{\vec{v}_z}{\|\vec{v}_z\|}$$
4. **Secondary Y-Axis** (completing the right-handed triad):
   $$\hat{u}_y = \hat{u}_z \times \hat{u}_x$$

The rotation matrix $R \in \text{SO}(3)$ and translation vector $\vec{T} \in \mathbb{R}^3$ mapping canonical local rotamer coordinates $\vec{p}_{\text{local}} = (x, y, z)^T$ to global lab coordinates $\vec{P}_{\text{lab}}$ are:
$$R = \begin{bmatrix} \hat{u}_x & \hat{u}_y & \hat{u}_z \end{bmatrix}, \quad \vec{T} = \vec{C}_\alpha$$
$$\vec{P}_{\text{lab}} = R \, \vec{p}_{\text{local}} + \vec{T} = \vec{C}_\alpha + x \hat{u}_x + y \hat{u}_y + z \hat{u}_z$$

---

### 2.8 Rotamer Library Distributions & Dihedral Space Discretization
From Dunbrack & Cohen (1997) and Shapovalov & Dunbrack (2011):
Backbone dihedral space $(\phi, \psi)$ is discretized into a toroidal grid:
- Grid intervals: $\Delta\phi = 10.0^\circ, \Delta\psi = 10.0^\circ$ (or $20.0^\circ$).
- Domain: $\phi, \psi \in [-180.0^\circ, +180.0^\circ)$.
- Periodic distance metric between target dihedral $\theta$ and bin center $\theta_k$:
  $$\Delta\theta = \min\left(|\theta - \theta_k| \pmod{360^\circ}, \, 360^\circ - (|\theta - \theta_k| \pmod{360^\circ})\right)$$
The bin index corresponding to query $(\phi, \psi)$ is selected by independent circular nearest-neighbor minimization on $\phi$ and $\psi$.

---

### 2.9 Ideal Geometry & Molecular Stereochemistry
From Engh & Huber (1991) *Acta Cryst.* A47: 392–400:
Ideal bond lengths and valence angles for amino-acid residues are derived from small-molecule crystal structures in the Cambridge Structural Database (CSD):

| Parameter | Engh & Huber Ideal Value | Parameter | Ideal Value |
|---|---|---|---|
| $N - C_\alpha$ bond length | $1.458 \pm 0.019$ Å | $C_\alpha - C$ bond length | $1.525 \pm 0.021$ Å |
| $C - N$ (peptide) bond length | $1.329 \pm 0.014$ Å | $C = O$ bond length | $1.231 \pm 0.020$ Å |
| $C_\alpha - C_\beta$ bond length | $1.530 \pm 0.020$ Å | Tetrahedral valence angle ($sp^3$) | $\arccos(-1/3) \approx 109.471^\circ$ |
| $N - C_\alpha - C$ valence angle | $111.2^\circ \pm 2.8^\circ$ | Planar peptide valence angle ($sp^2$) | $120.0^\circ$ |

---

## 3. `proxide-confind` Functional Specification

### 3.1 Public Types and API Contracts

#### Core Data Structures (`src/coords.rs`)
```rust
/// Flat 0-based dense index for protein residues.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ResidueIndex(pub u32);

/// Full protein backbone container in f64 precision.
#[derive(Debug, Clone)]
pub struct ProteinBackbone {
    pub bb: Vec<ResidueBackbone>,
    pub ids: Vec<ResidueId>,
    pub chain_map: Vec<usize>,
}

/// Backbone geometry for a single protein residue.
///
/// Under Debt #1897, undefined phi/psi angles (at chain termini or breaks)
/// are represented strictly as `None`. Sentinel floats (e.g. 9999.0) are prohibited.
#[derive(Debug, Clone, PartialEq)]
pub struct ResidueBackbone {
    pub res_name: String,
    pub n: Option<[f64; 3]>,
    pub ca: Option<[f64; 3]>,
    pub c: Option<[f64; 3]>,
    pub o: Option<[f64; 3]>,
    pub phi: Option<f64>,
    pub psi: Option<f64>,
    pub omega: Option<f64>,
    pub is_cis_peptide: bool,
}

/// Load protein backbone from PDB file in f64 precision.
pub fn load_pdb_f64<P: AsRef<Path>>(path: P) -> Result<ProteinBackbone, ConFindError>;

/// Extract protein backbone from an already-parsed ProcessedStructure in f64 precision.
pub fn extract_f64_backbone(s: &ProcessedStructure) -> Result<ProteinBackbone, ConFindError>;
```

#### Configuration Types (`src/config.rs`) — full definitions in §3.9
```rust
pub struct ConFindConfig { /* cutoffs, scope, cb_clash_policy, crowdedness_mode, freedom_mode */ }
pub enum CbClashPolicy { AlaOnly, AllResidues }
pub enum CrowdednessMode { ProbabilityWeighted, PrunedCountFraction }
pub enum FreedomMode { Rms, SingleThreshold }
pub enum ResidueScope { All, Subset(Vec<ResidueIndex>) }
pub struct Cutoffs { pub d_clash: f64, pub d_cont: f64, pub d_cut: f64, pub tau_lo: f64, pub tau_hi: f64 }
```

#### Pipeline Interface (`src/confind.rs`)
```rust
pub struct ConFind {
    pub rotlib: Arc<RotamerLibrary>,
    pub strict: bool,
    pub backbone: Arc<ProteinBackbone>,
    // Private spatial acceleration structures, concurrent caches and the validated config
}

impl ConFind {
    /// Construct with `ConFindConfig::default()`. Equivalent to `with_config(.., ConFindConfig::default())`.
    pub fn new(
        rotlib: Arc<RotamerLibrary>,
        backbone: Arc<ProteinBackbone>,
        strict: bool,
    ) -> Result<Self, ConFindError>;

    /// Construct with an explicit configuration. Validates the configuration (§3.9.3),
    /// computes R_max from `rotlib` (§3.8.5) and checks D_cut against it, then builds spatial indices.
    pub fn with_config(
        rotlib: Arc<RotamerLibrary>,
        backbone: Arc<ProteinBackbone>,
        strict: bool,
        config: ConFindConfig,
    ) -> Result<Self, ConFindError>;

    /// The validated configuration in force.
    pub fn config(&self) -> &ConFindConfig;

    /// Number of protein residues in backbone.
    pub fn n_residues(&self) -> usize;

    /// Query original PDB ResidueId (chain, sequence number, insertion code) from internal ResidueIndex.
    pub fn residue_id(&self, ri: ResidueIndex) -> &ResidueId;

    /// Return sorted list of residue indices that form constrained contacts (contact degree > 0.0) with residue `res`.
    pub fn constrained_contacts(&self, res: ResidueIndex) -> Vec<ResidueIndex>;

    /// Execute Phase A rotamer placement and backbone clash pruning for residue `ri`.
    pub fn cache_residue(&self, ri: ResidueIndex) -> Result<(), ConFindError>;

    /// Concurrently execute Phase A caching for all residues in structure.
    pub fn cache_all(&self) -> Result<(), ConFindError>;

    /// Query residues within `config.cutoffs.d_cut` Cα–Cα distance (inclusive).
    pub fn neighbors(&self, ri: ResidueIndex) -> Vec<ResidueIndex>;

    /// Compute pairwise contact degrees across query residue set.
    /// Returns pairs with contact degree > cd_cut.
    pub fn contacts(
        &self,
        residues: &[ResidueIndex],
        cd_cut: f64,
    ) -> Result<ContactList, ConFindError>;

    /// Interference I(i,j) (§2.6, §3.8.3): pairs where BOTH endpoints are present in
    /// `residues` and the value is >= in_cut. Interference is directed (i -> backbone of j).
    pub fn interference(
        &self,
        residues: &[ResidueIndex],
        in_cut: f64,
    ) -> Result<ContactList, ConFindError>;

    /// Minimum heavy-atom distance between backbone residues.
    pub fn bb_interaction(
        &self,
        residues: &[ResidueIndex],
        dcut_bb: f64,
        ignore_flanking: usize,
    ) -> Result<ContactList, ConFindError>;

    /// Crowdedness C(i) per `config.crowdedness_mode` (§2.4).
    pub fn crowdedness(&self, ri: ResidueIndex) -> Result<f64, ConFindError>;

    /// Freedom per `config.freedom_mode` (§2.5) with neighbours drawn from `config.scope`.
    /// Returns `ConFindError::OutOfScope` if `ri` is not in the scope.
    pub fn freedom(&self, ri: ResidueIndex) -> Result<f64, ConFindError>;
}
```

#### Freedom Kernel (`src/freedom.rs`)
```rust
/// Compute residue structural freedom (§2.5) from surviving-rotamer collision probabilities.
///
/// - `p_coll`: P_coll of the *contested* surviving rotamers only (those with P_coll > 0).
/// - uncontested = n_surviving.saturating_sub(p_coll.len())   (P_coll = 0 < tau for any tau > 0)
/// - N_lo = uncontested + count(p < tau_lo);  N_hi = uncontested + count(p < tau_hi)
/// - Rms:             sqrt((N_lo^2 + N_hi^2) / 2) / n_library
/// - SingleThreshold: N_lo / n_library
/// - n_library == 0 returns 0.0.
pub fn compute_freedom(
    mode: FreedomMode,
    p_coll: &[f64],
    n_surviving: usize,
    n_library: usize,
    tau_lo: f64,
    tau_hi: f64,
) -> f64;
```

#### Precondition Validation (`src/precondition.rs`)
```rust
#[derive(Debug, Clone, PartialEq)]
pub struct PreconditionViolation {
    pub residue: ResidueIndex,
    pub id: ResidueId,
    pub res_name: String,
    pub kind: ViolationKind,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ViolationKind {
    /// Backbone atom (N, CA, or C) is missing.
    MissingBackboneAtom { atom: &'static str },
    /// φ dihedral is None for a non-terminal residue (Debt #1897).
    UndefinedPhi,
    /// ψ dihedral is None for a non-terminal residue (Debt #1897).
    UndefinedPsi,
    /// Residue name not in canonical set (20 standard AAs + variants).
    UnknownResidueType { res_name: String },
    /// Consecutive same-chain residues with CA atoms separated by > 4.5 Å (Debt #1904).
    ChainBreak { gap_to_next_ca: f64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    /// Structural integrity compromised; ConFind cannot run.
    Error,
    /// Minor structural irregularity; ConFind may still run.
    Warning,
}

#[derive(Debug, Clone, Default)]
pub struct PreconditionReport {
    pub violations: Vec<PreconditionViolation>,
}

impl PreconditionReport {
    pub fn is_clean(&self) -> bool;
    pub fn errors(&self) -> impl Iterator<Item = &PreconditionViolation>;
    pub fn warnings(&self) -> impl Iterator<Item = &PreconditionViolation>;
    pub fn severity_of(kind: &ViolationKind) -> Severity;
}

/// Inspect structural preconditions on a protein backbone.
pub fn check_preconditions(bb: &ProteinBackbone) -> PreconditionReport;

/// Ensure structural preconditions pass without errors; returns ConFindError::PreconditionsFailed on error.
pub fn require_preconditions(bb: &ProteinBackbone) -> Result<(), ConFindError>;
```

#### Results Container (`src/contact_list.rs`)
```rust
pub const CONTACT_THRESHOLD: f64 = 0.02;

#[derive(Debug, Default, Clone)]
pub struct ContactList {
    pub pairs: Vec<(ResidueIndex, ResidueIndex)>,
    pub degrees: Vec<f64>,
}

impl ContactList {
    pub fn degree(&self, a: ResidueIndex, b: ResidueIndex) -> Option<f64>;
    pub fn ordered_pairs(&self) -> Vec<(ResidueIndex, ResidueIndex)>;
    pub fn filter(&self, threshold: f64) -> ContactList;
}
```

---

### 3.2 Error Taxonomy (`src/error.rs`)
```rust
#[derive(Debug, thiserror::Error)]
pub enum ConFindError {
    #[error("residue {0:?} not cached — call cache_residue first")]
    NotCached(ResidueIndex),

    #[error("residue {0:?} missing required backbone atom {1}")]
    MissingBackbone(ResidueIndex, &'static str),

    #[error("residue {0:?} is not in the configured analysis scope")]
    OutOfScope(ResidueIndex),

    #[error("structure failed {0} structural precondition check(s); inspect report for diagnostics")]
    PreconditionsFailed(usize),

    #[error("invalid configuration: {0}")]
    InvalidConfig(#[from] ConfigError),

    #[error("rotamer library error: {0}")]
    RotlibError(#[from] proxide_rotlib::RotlibError),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}
```
`ConfigError` is specified in §3.9.3.

---

### 3.3 Module Decomposition

1. **`src/cache.rs`** (Phase A):
   - Places rotamers using `rotlib.place_rotamer(aa, phi, psi, rot_idx, cis_proline, ...)`.
   - Queries the backbone spatial index (`ProximityGrid`) for heavy-atom clashes $\le d_{\text{clash}}$ using $S_{\text{clash}}$ per `CbClashPolicy` (§2.4), excluding the own-residue backbone.
   - Accumulates $W^{\text{lib}}$, $W^{\text{surv}}$, $N_{\text{lib}}$, $N_{\text{surv}}$ (both crowdedness modes are derivable from the same cache) and the per-neighbour-residue interference masses $I(i,j)$ (§3.8.3).
   - Alanine: $C_\beta$ is in $S_{\text{cont}}$ via `counts_as_sidechain`. Under `AllResidues`, $C_\beta$ of every amino acid is additionally in $S_{\text{clash}}$.
   - Glycine residue: zero sidechain atoms placed; crowdedness defined as $0.0$.
2. **`src/parallel.rs`** (Phase B):
   - Evaluates pairwise contact degrees for all candidate pairs $(i,j)$ with $\|C\alpha_i-C\alpha_j\|\le D_{\text{cut}}$.
   - Parallelised across chunks using `orx-parallel` or `rayon`.
   - Accumulates cross-rotamer heavy-atom contacts ($S_{\text{cont}}$, $\le d_{\text{cont}}$) weighted by $P(a)P(b)P(r_a)P(r_b)$ and normalises by $\Omega_i\Omega_j$.
   - Short-circuits to $0.0$ if $\Omega_i=0$ or $\Omega_j=0$ (no float division).
3. **`src/freedom.rs`** (Phase C): `compute_freedom` (§3.1) implementing §2.5 and §3.8.2. Returns $0.0$ if $n_{\text{library}}=0$.
4. **`src/params.rs`**:
   - Public default constants `CLASH_DIST = 2.0`, `CONT_DIST = 3.0`, `DCUT = 25.0`, `LO_COLL_PROB_CUT = 0.5`, `HI_COLL_PROB_CUT = 2.0` (the defaults of `Cutoffs`).
   - `aa_propensity(aa: &str) -> f64` returning percentage values $P(a)\in[1.5,8.8]$ with citations to Zheng et al. (2015).
   - In collision-probability and interference summation, $P(\cdot)$ is divided by $100.0$ (§2.5).
5. **`src/config.rs`**: the types, defaults, validation and string parsing of §3.9.
6. **`src/confind.rs`**: orchestrates caching, spatial indexing, pairwise dispatch, residue ID resolution and metric extraction under one validated `ConFindConfig`.

---

### 3.4 Elimination of Magic Sentinels: `Option<f64>` Dihedrals (Debt #1897)
- **Defect Description**: The existing codebase uses `9999.0` (or `999.0`) as an in-band float sentinel for missing dihedral angles at terminal residues or chain breaks.
- **Specification**:
  - `ResidueBackbone.phi` is typed as `Option<f64>`.
  - `ResidueBackbone.psi` is typed as `Option<f64>`.
  - `ResidueBackbone.omega` is typed as `Option<f64>`.
  - At the N-terminal residue of a chain: `phi = None`, `omega = None`.
  - At the C-terminal residue of a chain: `psi = None`.
  - In `precondition.rs`: preconditions are checked using `res.phi.is_none()` and `res.psi.is_none()`. No float tolerance checks against `9999.0` are permitted.
  - Across the `proxide-confind` $\to$ `proxide-rotlib` boundary: `RotamerLibrary::backbone_bin` accepts `phi: Option<f64>, psi: Option<f64>`. If either angle is `None`, it routes to the unconditioned fallback default bin (`AaEntry.default_bin`).

---

### 3.5 IUPAC Omega Dihedral Angle Convention (Debt #1903)
- **Defect Description**: A sign-inverted $\omega$ (from vector inversion) is a defect.
- **Specification**:
  - Standard IUPAC-IUBMB peptide torsion convention is enforced:
    $$\omega_i = \text{torsion}(C\alpha_{i-1}, C_{i-1}, N_i, C\alpha_i)$$
  - Range: $[-180.0^\circ, +180.0^\circ]$.
  - Classification:
    - Trans peptide bond: $|\omega_i| \ge 150.0^\circ$.
    - Cis peptide bond: $|\omega_i| < 30.0^\circ$ (`is_cis_peptide = true`).
  - Implemented cleanly using standard vector cross-product torsion angle formulas in `proxide-geometry`.

---

### 3.6 Segmented Chain Semantics, Distance Breaks, and TER Handling (Debt #1904)
- **Defect Description**: Grouping by chain identifier caused disconnected protein segments sharing a chain ID to be treated as continuous chains across breaks.
- **Specification**:
  1. **Ordering Invariant**: PDB residue order must be strictly preserved as encountered in the file; residues must not be re-sorted by chain ID.
  2. **Chain Boundary & Break Detection**: A chain boundary is triggered if ANY of the following conditions hold:
     - (a) `res_i.chain_id != res_{i-1}.chain_id` (chain transition without intervening TER record);
     - (b) An explicit `TER` record is encountered in the coordinate stream;
     - (c) Consecutive residues within the same chain have $\|\vec{C}\alpha_i - \vec{C}\alpha_{i-1}\| > 4.5$ Å (geometric distance break).
  3. **Break Boundary Semantics**:
     - Residue $i-1$ is treated as a segment C-terminus: `psi = None`.
     - Residue $i$ is treated as a segment N-terminus: `phi = None`, `omega = None`, `is_cis_peptide = false`.
  4. **Capping Groups**: Residues matching `ACE`, `NME`, or `NH2` are recognized as terminal capping groups and must not cause adjacent protein residues to compute invalid cross-cap dihedrals.

---

### 3.7 Documentation & Package Metadata Requirements
1. **Citations**: crate-level docs cite exactly: Zheng, F., Zhang, J., & Grigoryan, G. (2015). "Tertiary Structural Propensities Reveal Fundamental Sequence/Structure Relationships in Proteins." *Structure*, 23(5), 961–971; and Zheng, W., & Grigoryan, G. (2017). "Sequence statistics of tertiary structural motifs reflect protein stability." *PLoS ONE*, 12(5), e0178272. Helical-bundle parameterization papers are not cited as the source of contact degree.
2. **Interference docstring**: `interference` states: "pairs where BOTH endpoints are present in the query set and value >= in_cut."
3. **Comments**: source comments describe the mathematics in this specification and cite literature; they contain no references to external tools, file names or line numbers.
4. **`Cargo.toml` description** of `proxide-confind`: `Sequence-independent contact degree, crowdedness, and structural freedom algorithm for protein backbones`.
5. **Option documentation**: every public option of §3.9 has rustdoc stating its variants, default, and the quantities it affects.

---

### 3.8 Definitions of Freedom, Interference, Cutoffs and Reach

#### 3.8.1 Notation
$\mathcal{A}_{18}$, $P(a)$ (percent), $P(r\mid a,\phi,\psi)$, $\mathcal{R}^{\text{lib}}_i(a)$, $\mathcal{R}_i(a)$, $N_{\text{lib}}$, $N_{\text{surv}}$, $S_{\text{cont}}$, $S_{\text{clash}}$, $B_j$ and $\operatorname{near}$ are as defined in §2.2 and §2.4. $\mathcal R^{\text{lib}}_i=\bigsqcup_{a}\mathcal R^{\text{lib}}_i(a)$ and $\mathcal R_i=\bigsqcup_{a}\mathcal R_i(a)$. $S_{\text{all}}(r)$ is the set of all non-hydrogen atoms of rotamer $r$ other than $N,C_\alpha,C,O$ (includes $C_\beta$ for every amino acid); it is used only for reach (§3.8.5). Own-residue backbone is never tested when testing rotamers of $i$.

Quantities depending on the configuration (§3.9): $S_{\text{clash}}$ (`CbClashPolicy`), $C(i)$ (`CrowdednessMode`), $F(i)$ (`FreedomMode`), $\mathcal N(i)$ (`ResidueScope`, $D_{\text{cut}}$), and every cutoff.

---

#### 3.8.2 Collision Probability and Freedom

**Inputs**: residue $i$; analysis scope $\mathcal S$ (default: all residues of the structure); $\mathcal R^{\text{lib}}_i$, $\mathcal R_i$, $\mathcal R_j$ for $j\in\mathcal S$; $d_{\text{cont}}=3.0$ Å; $\tau_{\text{lo}}=0.5$, $\tau_{\text{hi}}=2.0$.

**Neighbour set**: $\mathcal N(i)=\{\,j\in\mathcal S,\ j\ne i:\ \|C\alpha_i-C\alpha_j\|\le D_{\text{cut}}\,\}$.

**Collision probability** of a surviving rotamer $r\in\mathcal R_i$:
$$P_{\text{coll}}(r)=\sum_{j\in\mathcal N(i)}\ \sum_{b\in\mathcal A_{18}}\frac{P(b)}{100}\sum_{s\in\mathcal R_j(b)}P(s\mid b,\phi_j,\psi_j)\ \operatorname{near}\big(S_{\text{cont}}(r),S_{\text{cont}}(s);\,d_{\text{cont}}\big)$$
It is a mass, not a conditional probability (it is not divided by the neighbour's surviving mass), and may exceed $1$.

**Threshold counts** ($\tau\in\{\tau_{\text{lo}},\tau_{\text{hi}}\}$), strict inequality:
$$N_\tau(i)=\big|\{\,r\in\mathcal R_i:\ P_{\text{coll}}(r)<\tau\,\}\big|$$
Rotamers with no contacts have $P_{\text{coll}}=0<\tau$ and are counted. Since $\tau_{\text{lo}}<\tau_{\text{hi}}$: $0\le N_{\tau_{\text{lo}}}\le N_{\tau_{\text{hi}}}\le N_{\text{surv}}\le N_{\text{lib}}$.

**Freedom**:
- `Rms` (default): $$F(i)=\frac{1}{N_{\text{lib}}(i)}\sqrt{\frac{N_{\tau_{\text{lo}}}(i)^2+N_{\tau_{\text{hi}}}(i)^2}{2}}\qquad F(i)\in[0,1]$$
- `SingleThreshold`: $$F^{(1)}(i)=N_{\tau_{\text{lo}}}(i)/N_{\text{lib}}(i)$$ using the configured $\tau_{\text{lo}}$.

$F$ is count-based: every rotamer counts once, irrespective of $P(a)$ and $P(r\mid a,\phi,\psi)$. $N_{\text{lib}}$ counts the unpruned library rotamers of all 18 amino acids at $(\phi_i,\psi_i)$, so backbone crowding enters through $N_{\text{surv}}\le N_{\text{lib}}$ and neighbour crowding through $P_{\text{coll}}$.

**Output**: $F(i)\in[0,1]$.

**Error / edge cases**
- $N_{\text{lib}}(i)=0$: $F(i)=0$ by convention (no error).
- $N_{\text{surv}}(i)=0$: $N_\tau=0$, $F=0$.
- $\mathcal N(i)=\emptyset$: every $P_{\text{coll}}=0$, $F(i)=N_{\text{surv}}/N_{\text{lib}}$ (= 1 − pruned count fraction) in both freedom modes.
- A rotamer with $P_{\text{coll}}=\tau$ exactly is *not* counted for that $\tau$.
- Library lookup failure for $i$ or any $j\in\mathcal N(i)$ propagates as an error; it is not silently treated as zero.
- $i\notin\mathcal S$: `ConFindError::OutOfScope`.
- The result depends on $\mathcal S$ only through $\mathcal N(i)$; outputs report the scope used.
- An unknown mode name (string parsing) is an error (§3.9.3); there is no silent fallback.

**Acceptance tests** (kernel level, defaults)
1. $N_{\text{lib}}=4,\ N_{\text{surv}}=2$, no neighbours: $F=\sqrt{(2^2+2^2)/2}/4=0.5$; `SingleThreshold`: $F^{(1)}=0.5$.
2. $N_{\text{lib}}=10,\ N_{\text{surv}}=8$; four rotamers have $P_{\text{coll}}=\{0.3,0.7,1.5,2.5\}$, four have none: $N_{\tau_{\text{lo}}}=5$, $N_{\tau_{\text{hi}}}=7$, $F=\sqrt{37}/10\approx0.6083$ (tolerance $10^{-9}$); `SingleThreshold`: $F^{(1)}=0.5$.
3. Boundary: a rotamer with $P_{\text{coll}}=0.5$ is excluded from $N_{\tau_{\text{lo}}}$ but included in $N_{\tau_{\text{hi}}}$.
4. $N_{\text{lib}}=0\Rightarrow F=0$; $N_{\text{surv}}=0\Rightarrow F=0$.
5. Invariants over random inputs: $0\le F\le1$; $F$ is non-increasing when any $P_{\text{coll}}$ increases; $F\le N_{\text{surv}}/N_{\text{lib}}$; $F^{(1)}\le F$.
6. Contact multiplicity: duplicating atoms of a rotamer pair that already touch does not change $P_{\text{coll}}$.
7. Pair symmetry: $\operatorname{near}(S_{\text{cont}}(r),S_{\text{cont}}(s);d_{\text{cont}})$ is symmetric in $(r,s)$.

---

#### 3.8.3 Interference of Residue $i$ Rotamers with Residue $j$ Backbone

**Definition** (identical to §2.6):
$$I(i,j)=\sum_{a\in\mathcal A_{18}}\frac{P(a)}{100}\sum_{r\in\mathcal R^{\text{lib}}_i(a)}P(r\mid a,\phi_i,\psi_i)\ \operatorname{near}\big(S_{\text{clash}}(r),B_j;\,d_{\text{clash}}\big),\qquad j\ne i$$
with $d_{\text{clash}}=2.0$ Å. The sum runs over the **full library set**, before backbone pruning. The per-amino-acid contribution $I_a(i,j)$ is the inner term; $I(i,j)=\sum_aI_a(i,j)$.

The result is the prior probability mass of rotamers of $i$ that are eliminated, in part, because they hit the backbone of $j$. A rotamer hitting several residues' backbones counts fully toward **each** of them, for every amino acid (order-independent), so $\sum_jI(i,j)$ may exceed the pruned mass.

**Optional normalised form**: $\hat I(i,j)=I(i,j)/Z_i$, $Z_i=\sum_{a}\frac{P(a)}{100}\sum_{r\in\mathcal R^{\text{lib}}_i(a)}P(r\mid a,\phi_i,\psi_i)=W^{\text{lib}}_i/100$; $Z_i=0\Rightarrow\hat I=0$.

**Output**: $I(i,j)\in[0,Z_i]$, with $Z_i\le\sum_a P(a)/100\approx0.8837$ when each conditional distribution sums to $\le1$.

**Edge cases**: $j=i$ is undefined (excluded; return $0$ or reject the query, but never test own backbone); missing atoms in $B_j$ simply remove those atoms from the set; $B_j=\emptyset\Rightarrow I=0$; no library rotamers $\Rightarrow I=0$; library lookup failure is an error.

**Acceptance tests** (kernel level; defaults unless stated)
1. One amino acid with $P(a)=10$, three rotamers with $P=\{0.5,0.3,0.2\}$; rotamer 1 clashes (through a non-$C_\beta$ atom) only with $B_j$; rotamer 2 clashes (non-$C_\beta$ atoms) with $B_j$ and $B_k$; rotamer 3 clashes with nothing: $I(i,j)=0.1(0.5+0.3)=0.08$, $I(i,k)=0.1\cdot0.3=0.03$, $I(i,i)$ not evaluated.
2. Order independence: permuting the order of atoms within rotamers, or of rotamers within the library, leaves every $I(i,j)$ unchanged (to $10^{-12}$).
3. Multiplicity: a rotamer with many atoms clashing with $B_j$ contributes once.
4. Bound: for every $j$, $I(i,j)\le Z_i\,C_w(i)$ where $C_w$ is the probability-weighted crowdedness of §2.4 computed with the same $S_{\text{clash}}$ (regardless of the configured `CrowdednessMode`).
5. Boundary: atom at exactly $2.0$ Å counts as clashing (inclusive), at $2.0+10^{-9}$ Å it does not.
6. Distance cutoff: if $\|C\alpha_i-C\alpha_j\|>R_{\max}(a)+R_{\text{bb}}+d_{\text{clash}}$ for all $a$ ($R_{\text{bb}}$ the largest $C_\alpha$–backbone-atom distance in a residue, $\approx2.4$ Å), then $I(i,j)=0$.

---

#### 3.8.4 Cutoffs and Their Roles

| Symbol | Default | Role | Used in |
|---|---|---|---|
| $d_{\text{clash}}$ | $2.0$ Å (inclusive $\le$) | Hard sidechain–backbone exclusion. A rotamer with any $S_{\text{clash}}(r)$–$B_j$ distance $\le d_{\text{clash}}$ ($j\ne i$) is pruned from $\mathcal R_i$ and contributes to $I(i,j)$. | $C(i)$, $\mathcal R_i$, $I(i,j)$, $\Omega_i$ in the contact-degree denominator |
| $d_{\text{cont}}$ | $3.0$ Å | Sidechain–sidechain contact between surviving rotamers. | $c(i,j)$, $P_{\text{coll}}$ |
| $D_{\text{cut}}$ | $25.0$ Å | Pre-filter on $C_\alpha$–$C_\alpha$ distance; pairs beyond it are not evaluated and have $c=0$ and contribute $0$ to $P_{\text{coll}}$. It is an **exactness-preserving optimisation**, valid iff $D_{\text{cut}}\ge R_{\max}(a)+R_{\max}(b)+d_{\text{cont}}$ for all $a,b$ (§3.8.5). | $\mathcal N(i)$, $c(i,j)$ |
| $\tau_{\text{lo}},\tau_{\text{hi}}$ | $0.5,\,2.0$ (dimensionless) | Collision-mass thresholds in $F(i)$; **unrelated** to $d_{\text{clash}}$ despite $\tau_{\text{hi}}$ and $d_{\text{clash}}$ sharing the number $2.0$. | $F(i)$ |

Provenance of default values: $d_{\text{clash}}=2.0$ Å is a deliberately tight, overlap-only exclusion (the van der Waals sum for C···N is $\approx3.25$ Å, Bondi 1964). $D_{\text{cut}}=25.0$ Å is justified as an upper bound by §3.8.5. $\tau_{\text{lo}},\tau_{\text{hi}}$ are empirical and calibrated only against the unnormalised scale of $P_{\text{coll}}$; a normalised $P_{\text{coll}}$ would require re-deriving them. All five are configurable (§3.9) and validated.

**Acceptance tests**
1. $d_{\text{clash}}$ inclusive: atom–backbone distance $2.0$ Å prunes; $2.0001$ Å does not.
2. $D_{\text{cut}}$ soundness: for a library-derived $R_{\max}$ (§3.8.5), construction checks $D_{\text{cut}}\ge2\max_aR_{\max}(a)+d_{\text{cont}}$ and returns `ConfigError::DcutTooSmall` otherwise.
3. $D_{\text{cut}}$ invariance: recomputing $c(i,j)$ for all pairs with $D_{\text{cut}}'=\infty$ and with $D_{\text{cut}}=25.0$ Å gives identical results (to $10^{-12}$) on a test protein.
4. A pair with $\|C\alpha_i-C\alpha_j\|=25.0$ Å is evaluated (inclusive); $25.0+10^{-9}$ Å is not.

---

#### 3.8.5 Maximum Sidechain Reach per Amino Acid

**Definition** (library-derived):
$$R_{\max}(a)=\max_{\text{bins }(\phi,\psi,\text{cis/trans})}\ \max_{r\in\mathcal R^{\text{lib}}(a)}\ \max_{x\in S_{\text{all}}(r)}\ \|x-C\alpha\|,\qquad a\in\mathcal A_{18}$$
evaluated over every tabulated bin and every rotamer, **before** any backbone pruning, with atoms placed in the standard residue frame (§2.7). Required cutoff:
$$D_{\text{cut}}^{\min}=2\max_{a\in\mathcal A_{18}}R_{\max}(a)+d_{\text{cont}}$$

**Independent geometric upper bound** (library-free): by the triangle inequality $R_{\max}(a)\le L(a)$, with $L(a)$ the sum of bond lengths along the longest bonded path from $C_\alpha$ to a sidechain heavy atom (ideal lengths from Engh & Huber 1991). For arginine the path $C_\alpha\!-\!C_\beta\!-\!C_\gamma\!-\!C_\delta\!-\!N_\epsilon\!-\!C_\zeta\!-\!N_\eta$ gives $L\approx8.7$ Å; lysine $\approx7.6$ Å; tryptophan, tyrosine and the other aromatics $\lesssim8.7$ Å. Hence $2L_{\max}+d_{\text{cont}}\approx20.4$ Å. If contact were instead defined by van der Waals envelopes (Bondi radii, $r_C=1.70$, $r_N=1.55$, $r_O=1.52$, $r_S=1.80$ Å) the additive term would be at most $2\cdot1.80=3.6$ Å, giving $\approx21.0$ Å. Both bounds are $<25.0$ Å.

**Computing it from the loaded library is sufficient** for exactness of the quantities defined in this specification, with three conditions:
1. Every contact and clash is evaluated only among library-placed rotamers, so a reach measured over the same library bounds the same set of states; the library-free bound $L(a)$ is only a sanity ceiling.
2. The maximum is taken over **all** bins, including cis, and over unpruned rotamers; a per-structure maximum over survivors would be unsound.
3. It is recomputed whenever a different library is loaded; a library with additional extended states can raise $R_{\max}$.

$R_{\max}$ is computed from the loaded library at construction. $D_{\text{cut}}$ is **validated** against $D^{\min}_{\text{cut}}$ (error `ConfigError::DcutTooSmall` if $D_{\text{cut}}<2\max_aR_{\max}(a)+d_{\text{cont}}$) and is never silently replaced. A smaller derived cutoff would only speed up neighbour search and must not change results.

**Edge cases**: amino acid with no library entries ($\mathcal R^{\text{lib}}(a)=\emptyset$): $R_{\max}(a)$ undefined, library-load error (all 18 amino acids must be present); non-finite coordinates: error; GLY and PRO excluded ($a\in\mathcal A_{18}$ only).

**Acceptance tests**
1. $R_{\max}(\text{ALA})\in[1.45,1.60]$ Å ($C_\alpha$–$C_\beta$ distance).
2. For every $a$: $R_{\max}(\text{ALA})\le R_{\max}(a)\le L(a)$.
3. $R_{\max}(\text{ARG})\ge5.0$ Å for any library containing an extended $\chi$-state.
4. $2\max_aR_{\max}(a)+3.0\le25.0$ for the shipped library (else load fails).
5. Exactness: for random placed pairs with $\|C\alpha_i-C\alpha_j\|>D_{\text{cut}}^{\min}$, no rotamer pair has any atom pair within $3.0$ Å.

Convention notes: $N_{\text{lib}}=0\Rightarrow F=0$ is a chosen convention; library-derived $R_{\max}$ is sufficient for exactness while $25.0$ Å is retained as a validated default.

---

### 3.9 Typed Configuration

All behavior-selecting parameters of `proxide-confind` are carried by one typed value, `ConFindConfig`, accepted by `ConFind::with_config` and populated by the `confind` binary flags (§3.9.5). Options are orthogonal: each option affects a documented set of outputs (§3.9.4) and any combination of valid values is legal, subject only to the validation rules of §3.9.3.

#### 3.9.1 Types, Variants and Defaults
```rust
/// Which residues' backbone-clash test includes the Cβ atom.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum CbClashPolicy {
    /// Cβ is tested against backbone for alanine only.
    AlaOnly,
    /// Cβ is tested against backbone for every amino acid.
    #[default]
    AllResidues,
}

/// How crowdedness C(i) aggregates pruned rotamers (§2.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum CrowdednessMode {
    /// C = 1 - W_surv / W_lib   (prior-probability weighted).
    #[default]
    ProbabilityWeighted,
    /// C = 1 - N_surv / N_lib   (rotamer count).
    PrunedCountFraction,
}

/// How freedom F(i) combines threshold counts (§2.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum FreedomMode {
    /// sqrt((N_lo^2 + N_hi^2)/2) / N_lib
    #[default]
    Rms,
    /// N_lo / N_lib
    SingleThreshold,
}

/// Residues eligible as neighbours in the collision accumulation of freedom (§3.8.2).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ResidueScope {
    #[default]
    All,
    Subset(Vec<ResidueIndex>),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cutoffs {
    pub d_clash: f64, // default 2.0 Å
    pub d_cont: f64,  // default 3.0 Å
    pub d_cut: f64,   // default 25.0 Å
    pub tau_lo: f64,  // default 0.5
    pub tau_hi: f64,  // default 2.0
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ConFindConfig {
    pub cutoffs: Cutoffs,                 // Default = the defaults above
    pub scope: ResidueScope,              // Default = All
    pub cb_clash_policy: CbClashPolicy,   // Default = AllResidues
    pub crowdedness_mode: CrowdednessMode,// Default = ProbabilityWeighted
    pub freedom_mode: FreedomMode,        // Default = Rms
}

impl ConFindConfig {
    /// Builder-style setters, one per field, each returning `Self`.
    pub fn with_cutoffs(self, c: Cutoffs) -> Self;
    pub fn with_scope(self, s: ResidueScope) -> Self;
    pub fn with_cb_clash_policy(self, p: CbClashPolicy) -> Self;
    pub fn with_crowdedness_mode(self, m: CrowdednessMode) -> Self;
    pub fn with_freedom_mode(self, m: FreedomMode) -> Self;
    /// Context-free validation (§3.9.3 rules V1–V5).
    pub fn validate(&self) -> Result<(), ConfigError>;
}
```
`Cutoffs` implements `Default` with the values in the table. All enums implement `Display` and `FromStr` using the kebab-case names of §3.9.5.

| Option | Variants | Default | Meaning |
|---|---|---|---|
| `CbClashPolicy` | `AlaOnly`, `AllResidues` | `AllResidues` | Defines $S_{\text{clash}}$ (§2.4). `AllResidues` is the physically sensible choice (a clashing $C_\beta$ excludes every rotamer); `AlaOnly` restricts the $C_\beta$ test to alanine. |
| `CrowdednessMode` | `ProbabilityWeighted`, `PrunedCountFraction` | `ProbabilityWeighted` | Defines $C(i)$ (§2.4). |
| `FreedomMode` | `Rms`, `SingleThreshold` | `Rms` | Defines $F(i)$ (§2.5). |
| `ResidueScope` | `All`, `Subset(..)` | `All` | Defines $\mathcal S$ and thus $\mathcal N(i)$ (§3.8.2). |
| `Cutoffs` | five f64 fields | §3.8.4 | $d_{\text{clash}}, d_{\text{cont}}, D_{\text{cut}}, \tau_{\text{lo}}, \tau_{\text{hi}}$. |

#### 3.9.2 Fixed Definitions (not options)
These follow from the owner-accepted definitions and are not configurable: symmetric RMS and single-threshold freedom are the only freedom variants; every clashed residue is credited in $I(i,j)$ (§3.8.3); $P_{\text{coll}}$ is the unnormalised mass (§2.5); freedom is count-based; $R_{\max}$ is computed from the loaded library (§3.8.5); own-residue backbone is never tested; contact atoms $S_{\text{cont}}$ follow §4.5.

#### 3.9.3 Validation and Errors
```rust
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ConfigError {
    #[error("cutoff {name} must be finite and > 0 (got {value})")]
    InvalidCutoff { name: &'static str, value: f64 },
    #[error("tau_lo ({tau_lo}) must be < tau_hi ({tau_hi}) when freedom_mode is Rms")]
    TauOrder { tau_lo: f64, tau_hi: f64 },
    #[error("d_cut {d_cut} Å < required {required} Å (= 2*max R_max + d_cont) for the loaded rotamer library")]
    DcutTooSmall { d_cut: f64, required: f64 },
    #[error("scope subset is empty")]
    EmptyScope,
    #[error("scope contains residue index {index} but structure has {n_residues} residues")]
    ScopeOutOfRange { index: u32, n_residues: usize },
    #[error("unknown {option} value '{value}'; expected one of: {expected}")]
    UnknownVariant { option: &'static str, value: String, expected: &'static str },
}
```
Rules (all reported as `ConFindError::InvalidConfig`; no silent correction or fallback):
- **V1** every cutoff is finite and $>0$ (`InvalidCutoff`).
- **V2** $\tau_{\text{lo}}<\tau_{\text{hi}}$ if `freedom_mode == Rms`; for `SingleThreshold` only $\tau_{\text{lo}}$ is used and $\tau_{\text{hi}}$ need only satisfy V1 (`TauOrder`).
- **V3** `ResidueScope::Subset` is non-empty (`EmptyScope`); duplicates are permitted and ignored.
- **V4** string parsing of a variant name rejects unknown names (`UnknownVariant`).
- **V5** $D_{\text{cut}}\ge2\max_aR_{\max}(a)+d_{\text{cont}}$ is checked at `with_config`/`new` time against the loaded library (`DcutTooSmall`).
- **V6** (at construction) every scope index is $<$ `n_residues` (`ScopeOutOfRange`).
`validate()` performs V1–V4; construction performs V1–V6 in order V1, V2, V3, V6, V5.

#### 3.9.4 Composition Matrix
Each option affects exactly the outputs marked ✔; options never alter outputs marked –.

| Output | `CbClashPolicy` | `CrowdednessMode` | `FreedomMode` | `ResidueScope` | `Cutoffs` |
|---|---|---|---|---|---|
| surviving sets $\mathcal R_i$, $N_{\text{surv}}$ | ✔ | – | – | – | ✔ ($d_{\text{clash}}$) |
| `crowdedness` $C(i)$ | ✔ | ✔ | – | – | ✔ ($d_{\text{clash}}$) |
| `contacts` $c(i,j)$ | ✔ (via $\Omega$) | – | – | – | ✔ ($d_{\text{clash}},d_{\text{cont}},D_{\text{cut}}$) |
| `interference` $I(i,j)$ | ✔ | – | – | – | ✔ ($d_{\text{clash}}$) |
| `freedom` $F(i)$ | ✔ (via $\mathcal R_i$) | – | ✔ | ✔ | ✔ (all five) |

Documented interaction restrictions are exactly V2 (`Rms` needs $\tau_{\text{lo}}<\tau_{\text{hi}}$) and V5 ($D_{\text{cut}}$ vs. $d_{\text{cont}}$ and library reach). No other combination is restricted; in particular, all $2\times2\times2$ combinations of the three enums are legal with any valid cutoffs and any valid scope.

#### 3.9.5 `confind` Binary Flags
Existing flags (`--p`, `--rLib`, `--o`, `--sel`, `--cdcut`, `--incut`, `--threads`) are retained. New flags map one-to-one onto `ConFindConfig`:

| Flag | Values | Default | Field |
|---|---|---|---|
| `--cbclash` | `ala-only` \| `all-residues` | `all-residues` | `cb_clash_policy` |
| `--crowdmode` | `probability-weighted` \| `pruned-count-fraction` | `probability-weighted` | `crowdedness_mode` |
| `--freedommode` | `rms` \| `single-threshold` | `rms` | `freedom_mode` |
| `--scope` | `all` \| `<residue-id list or file>` | `all` | `scope` |
| `--dclash`, `--dcont`, `--dcut` | f64 Å | 2.0, 3.0, 25.0 | `cutoffs.*` |
| `--taulo`, `--tauhi` | f64 | 0.5, 2.0 | `cutoffs.tau_*` |

Behavior: flags may appear in any order and in any combination; an unknown value or a configuration violating §3.9.3 exits with a non-zero status and a message naming the option, the offending value and the allowed values; nothing falls back silently. The output header records every effective option value (including defaults) so that a run is reproducible from its output.

#### 3.9.6 Acceptance Tests for Options
Kernel-level fixture **T1** (injected rotamer sets, non-alanine amino acid, $P(a)=10$): rotamers $r_1,r_2,r_3$ with $P=\{0.5,0.3,0.2\}$; $r_1$ has a non-$C_\beta$ atom at $1.9$ Å from $B_j$ (no other close backbone); $r_2$ has only its $C_\beta$ at $1.9$ Å from $B_k$; $r_3$ is clash-free.

1. **Defaults.** `ConFindConfig::default()` equals `{AllResidues, ProbabilityWeighted, Rms, All, Cutoffs{2.0,3.0,25.0,0.5,2.0}}`; `ConFind::new` behaves identically to `with_config(.., default())`.
2. **Defaults reproduce stated numbers.** Under defaults the tests of §3.8.2 (items 1–2: $F=0.5$, $F=\sqrt{37}/10$), §3.8.3 (item 1: $I(i,j)=0.08$, $I(i,k)=0.03$) and T1 (below, `AllResidues`/`ProbabilityWeighted`) hold to $10^{-9}$.
3. **T1 expected values**:

   | `CbClashPolicy` | pruned | $N_{\text{surv}}/N_{\text{lib}}$ | $C$ `ProbabilityWeighted` | $C$ `PrunedCountFraction` | $I(i,j)$ | $I(i,k)$ |
   |---|---|---|---|---|---|---|
   | `AllResidues` | $r_1,r_2$ | 1/3 | $0.8$ | $2/3$ | $0.05$ | $0.03$ |
   | `AlaOnly` | $r_1$ | 2/3 | $0.5$ | $1/3$ | $0.05$ | $0$ |

   (Tolerance $10^{-12}$.) For an alanine fixture with the same geometry, both policies give the `AllResidues` row, because $C_\beta\in S_{\text{cont}}$ for alanine.
4. **Policy-combination matrix.** On a synthetic helix and on the vendored crystal structure, for each of the $2\times2\times2$ combinations of `CbClashPolicy`, `CrowdednessMode`, `FreedomMode`, crossed with scope in {`All`, a proper `Subset`} (16 runs): construction succeeds; `cache_all`, `contacts`, `interference`, `crowdedness` and `freedom` all return `Ok`; all $C,F,c\in[0,1]$ and finite; $I(i,j)\le Z_iC_w(i)$.
5. **Orthogonality (§3.9.4).** Holding other options fixed: changing `CrowdednessMode` changes only `crowdedness` outputs (bitwise-identical `contacts`, `interference`, `freedom`); changing `FreedomMode` changes only `freedom` outputs; changing `ResidueScope` changes only `freedom` outputs and never any `crowdedness`, `contacts`, `interference` value; changing `CbClashPolicy` leaves all outputs defined (monotonicity: $N_{\text{surv}}^{\text{AllResidues}}(i)\le N_{\text{surv}}^{\text{AlaOnly}}(i)$ and, in either crowdedness mode, $C^{\text{AllResidues}}(i)\ge C^{\text{AlaOnly}}(i)$).
6. **Freedom-mode relation.** For every residue, $F^{(1)}(i)\le F(i)$, and both equal $N_{\text{surv}}/N_{\text{lib}}$ when $\mathcal N(i)=\emptyset$.
7. **Crowdedness-mode relation.** On a single-amino-acid fixture with uniform rotamer probabilities the two modes agree; on T1 they differ as in item 3.
8. **Scope.** `Subset(S)` equals `All` for any residue whose Cα neighbours within $D_{\text{cut}}$ all lie in $S$; `freedom(i)` for $i\notin S$ returns `OutOfScope`; adding a residue to $S$ never increases any $F(i)$ (monotone non-increasing).
9. **Validation.** One test per error: each of V1 (zero, negative, NaN, infinite for each of the five cutoffs), V2 (`tau_lo>=tau_hi` with `Rms` fails; the same values with `SingleThreshold` pass), V3, V4 (each enum), V5 ($D_{\text{cut}}=10.0$ with the shipped library fails with `DcutTooSmall` whose `required` equals $2\max R_{\max}+d_{\text{cont}}$; setting $d_{\text{cont}}=10$ with $D_{\text{cut}}=25$ fails as well), V6.
10. **Cutoff composition.** For each policy combination, $F$ is non-decreasing in $\tau_{\text{lo}}$ and in $\tau_{\text{hi}}$ and stays in $[0,1]$; with $d_{\text{clash}}$ below the smallest atom–backbone distance, $C=0$ and $N_{\text{surv}}=N_{\text{lib}}$.
11. **CLI.** For every flag of §3.9.5: a valid value is accepted and echoed in the output header; an invalid value exits non-zero naming the option; the 8 enum combinations plus a `--scope` list run to completion; a command line with no new flags produces output identical to the same command with every new flag set to its default.
12. **Determinism.** Identical config and input give bitwise-identical outputs for any `--threads` value.

---

## 4. `proxide-rotlib` & `proxide_fixer` Functional Specification

### 4.1 Rotamer Library Loading (`proxide-rotlib`)
`proxide-rotlib` carries `license = "MIT"` and loads rotamer libraries exclusively through the Protobuf API:
```rust
impl RotamerLibrary {
    pub fn load_pb(path: &Path) -> Result<Self, RotlibError>;
    pub fn load_pb_zst(path: &Path) -> Result<Self, RotlibError>;
}
```
No other binary-format reader, converter or conversion CLI is part of this crate. Tests under `crates/proxide-rotlib/tests/` use the committed Protobuf asset `data/rotlibs/proxide-rotlib-bbdep2010.pb.zst` (ODC-BY-1.0 Dunbrack library) or synthetic in-memory libraries.

---

### 4.2 (Reserved)

---

### 4.3 Clean-Room Protobuf Rotamer Library Engine (`proxide-rotlib`)
`proxide-rotlib` exposes the clean public query API:
```rust
impl RotamerLibrary {
    /// True if amino acid is present in library.
    pub fn contains_aa(&self, aa: &str) -> bool;

    /// Number of rotamers available for residue at given dihedral angles.
    ///
    /// For PRO residues with cis_proline = true, routes to the CPR library definition.
    pub fn num_rotamers(
        &self,
        aa: &str,
        phi: Option<f64>,
        psi: Option<f64>,
        cis_proline: bool,
    ) -> Result<usize, RotlibError>;

    /// Probability of specific rotamer index.
    pub fn rotamer_probability(
        &self,
        aa: &str,
        phi: Option<f64>,
        psi: Option<f64>,
        cis_proline: bool,
        rot_index: usize,
    ) -> Result<f64, RotlibError>;

    /// Sidechain heavy atom names in library order.
    pub fn sidechain_atom_names(&self, aa: &str) -> Result<&[String], RotlibError>;

    /// Discrete Ramachandran bin index.
    pub fn backbone_bin(
        &self,
        aa: &str,
        phi: Option<f64>,
        psi: Option<f64>,
        cis_proline: bool,
    ) -> Result<usize, RotlibError>;
}
```

#### Cis-Proline Routing Contract
When an incoming peptide bond is cis ($|\omega| < 30^\circ$, Debt #1903), callers set `cis_proline: true`.
- If `aa == "PRO"` and `cis_proline == true`, query and placement functions route internally to the `CPR` rotamer entry in the Dunbrack library.
- If `aa == "PRO"` and `cis_proline == false`, functions route to `PRO`.
- For all other amino acids ($a \neq \text{PRO}$), the `cis_proline` flag is ignored.

---

### 4.4 Canonical Placement API & Orthonormal Frame Construction
Rotamer atom placement onto a target backbone is specified as:
```rust
pub struct PlacedAtom {
    pub name: String,
    pub coords: [f64; 3],
}

pub struct PlacedRotamer {
    pub id: RotamerId,
    pub atoms: Vec<PlacedAtom>,
}

impl RotamerLibrary {
    pub fn place_rotamer(
        &self,
        aa: &str,
        phi: Option<f64>,
        psi: Option<f64>,
        rot_index: usize,
        cis_proline: bool,
        n: [f64; 3],
        ca: [f64; 3],
        c: [f64; 3],
    ) -> Result<PlacedRotamer, RotlibError>;
}
```
**Behavioral Contract**:
1. Construct right-handed orthonormal triad $\hat{u}_x, \hat{u}_y, \hat{u}_z$ at $\vec{C}_\alpha$ via Gram-Schmidt orthonormalization (Section 2.7).
2. Fetch canonical sidechain atom coordinates $(x_k, y_k, z_k)$ from the library (routing to `CPR` if `aa == "PRO" && cis_proline`).
3. Compute global coordinates: $\vec{x}_k = \vec{C}_\alpha + x_k \hat{u}_x + y_k \hat{u}_y + z_k \hat{u}_z$.
4. Return `PlacedRotamer`. For Glycine (`GLY`), returns an empty atom list.

---

### 4.5 Sidechain Atom Predicates (`src/sidechain.rs`)
```rust
/// True for peptide backbone heavy atoms and hydrogens.
pub fn is_backbone_or_hydrogen(name: &str) -> bool {
    matches!(name, "N" | "CA" | "C" | "O" | "H") || name.starts_with('H')
}

/// Determines whether an atom contributes to sidechain-sidechain contact calculations.
///
/// In standard residue topologies, Cβ is bonded to Cα and shares the rigid residue
/// backbone frame. For Alanine, Cβ is the sole sidechain heavy atom and is retained;
/// for all other amino acids, Cβ is excluded to prevent rigid backbone-level contacts
/// from artificially inflating rotamer contact degree.
///
/// This predicate defines S_cont (the sidechain-sidechain contact atoms). The backbone-clash
/// test uses S_clash, which adds Cβ for every amino acid under `CbClashPolicy::AllResidues` (§2.4).
pub fn counts_as_sidechain(atom_name: &str, aa: &str) -> bool {
    if is_backbone_or_hydrogen(atom_name) {
        return false;
    }
    if atom_name == "CB" && aa != "ALA" {
        return false;
    }
    true
}
```

---

### 4.6 Fixer Repack Test Suite Migration to Dunbrack Protobuf (Debt #1891)
- **Defect Description**: Seven tests in `crates/proxide_fixer/src/repack.rs` attempt to load `testfiles/rotlib.pb.zst`, which does not exist in the repository. When absent, the tests execute `eprintln!("Skipping test...")` and return early, silently passing in CI.
- **Specification**:
  1. Update `load_rotlib()` to resolve the committed repository asset:
     `data/rotlibs/proxide-rotlib-bbdep2010.pb.zst` (ODC-BY-1.0 Dunbrack library).
     Path resolution must be portable across both workspace root and crate directory test executions:
     ```rust
     let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
         .join("../../data/rotlibs/proxide-rotlib-bbdep2010.pb.zst");
     ```
     with fallback checking `CARGO_WORKSPACE_DIR` or parent directory traversal.
  2. Implement strict load status discrimination:
     ```rust
     pub enum RotlibLoadResult {
         Loaded(RotamerLibrary),
         NotConfigured(String),
         Failed(String),
     }
     ```
  3. Replace all `eprintln!` calls with structured `tracing::warn!` and `tracing::info!`.
  4. If the rotamer library file exists on disk but fails to decode or decompress, the test must **fail loudly**:
     ```rust
     RotlibLoadResult::Failed(err) => panic!("CORRUPT ROTAMER LIBRARY ASSET: {}", err),
     ```
  5. The 7 repack tests (`test_truncated_lys_completion`, `force_rebuild_strips_sidechain_and_rebuilds`, `force_rebuild_maintains_backbone_bond_lengths`, `force_rebuild_does_not_process_gly`, `rebuild_residue_calls_force_rebuild`, `rebuild_residue_returns_error_for_missing_residue`, `repack_neighbourhood_empty_targets`) will run and pass unconditionally in CI.

---

### 4.7 CCD Parser Explicit Token Validation & Fallback Provenance (Debt #1933)
- **Defect Description**: In `crates/proxide-rotlib/src/geometry/ccd_parser.rs`:
  1. Extra tokens on CIF table lines beyond field definitions are silently discarded.
  2. Missing bond lengths silently default to 1.5 Å without record or citation.
  3. Missing bond angles silently default to 109.5° without record or citation.
  4. Standard CIF null/missing tokens (`"?"` and `"."`) trigger parse errors that cause entire residue components to be dropped.
- **Specification**:
  1. **Strict Token Validation**:
     Each CIF table row must have an exact token count matching the declared field definition count. If a row contains extra or missing tokens, parsing must fail with `RotlibError::InvalidFormat`.
  2. **CIF Null Token Handling**:
     Field values equal to standard CIF unmeasured/null tokens (`"?"` or `"."`) must be parsed as absent values (`None`) rather than causing fatal parse aborts or dropping valid residue components.
  3. **Mandatory Fallback Recording & Provenance Types**:
     Silent fallbacks are strictly prohibited. When an ideal bond length or valence angle is absent from the Chemical Component Dictionary, the parser must record the fallback into an explicit `fallbacks: Vec<GeometryFallback>` field and emit a structured `tracing::warn!` message citing the literature source:

     ```rust
     #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
     pub enum FallbackKind {
         BondLength,
         ValenceAngle,
     }

     #[derive(Debug, Clone, PartialEq)]
     pub struct GeometryFallback {
         pub component: String,
         pub atom_names: Vec<String>,
         pub parameter_kind: FallbackKind,
         pub fallback_value: f64,
         pub citation: &'static str,
     }
     ```

     The containing geometry table in `proxide-rotlib` must expose:
     ```rust
     pub struct ResidueGeometryTable {
         // Standard residue stereochemical tables...
         pub fallbacks: Vec<GeometryFallback>,
     }
     ```

     Standard Physical Citations for Fallbacks:
     - Bond length fallback: `1.50 Å` — Engh, R. A. & Huber, R. (1991). "Accurate bond and angle parameters for X-ray protein structure refinement." *Acta Cryst.* A47, 392–400.
     - Valence angle fallback: `109.47°` — Ideal tetrahedral $sp^3$ valence angle $\arccos(-1/3)$.

---

### 4.8 Workspace Decoupling: Inlining `CANONICAL_AA_NAMES` into `proxide_fixer`
- **Defect Description**: `crates/proxide_fixer` depends on `proxide-confind` exclusively to import `CANONICAL_AA_NAMES`.
- **Specification**:
  1. In `crates/proxide_fixer/src/models.rs`, define:
     ```rust
     /// 20 canonical amino acids plus standard protonation and cysteine variants.
     pub const CANONICAL_AA_NAMES: &[&str] = &[
         "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "HID", "HIE", "HIP",
         "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "CYX",
     ];
     ```
  2. Update `crates/proxide_fixer/src/mutate.rs` to import from `crate::models::CANONICAL_AA_NAMES`.
  3. Remove `proxide-confind = { workspace = true }` completely from `crates/proxide_fixer/Cargo.toml`.
  4. Verify with `cargo tree -p proxide_fixer -e normal` that `proxide-confind` is absent from dependencies.

---

## 5. Test Acceptance & Golden Replacement Specification

### 5.1 Test Fixtures & Constants
The test suites must contain only data produced by this repository's own generators, public-domain structures, or the committed Protobuf asset:

| Need | Location | Source |
|---|---|---|
| Small backbone | `tests/common/mod.rs` | Synthetic helical generator `make_synthetic_backbone(n, spacing)` |
| Crystal structure | `tests/common/mod.rs` | Clean coordinates vendored from wwPDB into `tests/data/1dc7.pdb` |
| Rotamer library | `tests/common/mod.rs` | `data/rotlibs/proxide-rotlib-bbdep2010.pb.zst` |
| Paths | `tests/common/mod.rs` | Workspace-relative paths (`CARGO_MANIFEST_DIR`) or synthetic fixtures; no absolute or external-checkout paths |
| Distogram checks | `test_distogram.rs` | Mathematical invariant tests (symmetry, triangle inequality, SE(3) invariance); no embedded float tables |
| $C_\beta$ placement | `test_placement.rs` | Exact analytical coordinate derivation; no embedded float tables |
| Contact / crowdedness / freedom / interference | `crates/proxide-confind/tests/` | Invariant tests of §3.8, §3.9.6 and §5.3; no embedded reference tables |

---

### 5.2 Synthetic Structure Generators & Public Domain Coordinates
Unit testing in `proxide-confind` will utilize clean synthetic structure generators:
1. **Ideal $\alpha$-Helix Generator**:
   Generates an $N$-residue peptide with canonical dihedrals $(\phi = -57.0^\circ, \psi = -47.0^\circ)$, $C_\alpha - C_\alpha$ spacing of $3.8$ Å, and helical pitch of $1.5$ Å per residue.
2. **Extended $\beta$-Strand Generator**:
   Generates an $N$-residue peptide with canonical dihedrals $(\phi = -139.0^\circ, \psi = +135.0^\circ)$.
3. **Two-Chain Interacting Complex Generator**:
   Generates two antiparallel synthetic helices separated by distance $D \in [6.0 \text{ Å}, 30.0 \text{ Å}]$.
4. **Crystallographic Reference Structures**:
   Clean PDB structures from wwPDB (Crambin, PDB ID `1DC7`; Ubiquitin, PDB ID `1UBQ`) vendored with open PDB / CC0 attribution.

---

### 5.3 Physical Invariants & Mathematical Consistency Tests
All clean-room implementations must satisfy the following invariant unit tests:

1. **Contact Degree Symmetry**:
   $$\forall \, i, j: \quad c(i, j) = c(j, i)$$
2. **Spatial Cutoff Monotonicity**:
   $$\|\vec{C}\alpha_i - \vec{C}\alpha_j\| > 25.0 \text{ Å} \implies c(i, j) = 0.0$$
3. **Metric Boundedness**:
   $$0.0 \le c(i, j) \le 1.0, \quad 0.0 \le C(i) \le 1.0, \quad 0.0 \le F(i) \le 1.0$$
4. **Rigid-Body SE(3) Invariance**:
   For any rotation matrix $R \in \text{SO}(3)$ and translation vector $\vec{t} \in \mathbb{R}^3$:
   $$c(i, j; \, R \cdot \text{backbone} + \vec{t}) = c(i, j; \, \text{backbone})$$
5. **Crowdedness-Freedom Anti-Correlation**:
   For an isolated residue without spatial obstacles, $C(i) \to 0.0$ and $F(i) \to 1.0$.

---

### 5.4 Configuration-Option Tests
The per-option acceptance tests, policy-combination matrix and CLI tests are specified in §3.9.6 and are mandatory.

---

## 6. Traceability Matrix & Debt Reconciliation

| Debt / Issue ID | Summary | Resolution in Specification | Section |
|---|---|---|---|
| **#3184** | Clean-room rewrite of `proxide-confind` | `cache.rs`, `parallel.rs`, `freedom.rs`, `confind.rs`, `params.rs` specified from Zheng & Grigoryan (2017) | §1.3, §2.1–§2.6, §3.3 |
| **#3185** | Single rotamer format for MIT crates | `proxide-rotlib` loads Protobuf only; `proxide_fixer` decoupled | §1.2, §4.1, §4.3 |
| **#1897** | Magic `9999.0` dihedral sentinel | `Option<f64>` on `ResidueBackbone` across `confind` and `rotlib`; no sentinel floats | §3.1, §3.4, §4.3 |
| **#1891** | Fixer repack tests skip in CI | `load_rotlib()` points to committed `proxide-rotlib-bbdep2010.pb.zst`; fail loudly on corrupt load; portable workspace path resolution; `tracing` instead of `eprintln!` | §4.6 |
| **#1903** | Omega dihedral sign convention | Standard IUPAC $C\alpha_{i-1}-C_{i-1}-N_i-C\alpha_i$ torsion; cis defined by $|\omega|<30^\circ$ | §3.5 |
| **#1904** | Chain semantics & breaks | File order preserved; segmented on `chain_id` change, TER records, or $C\alpha-C\alpha>4.5$ Å; dihedrals cleared at breaks | §3.6 |
| **#1933** | CCD parser fallbacks | Strict token validation; CIF null tokens as `None`; mandatory `GeometryFallback` records on `ResidueGeometryTable` with `tracing::warn!` citing Engh & Huber (1991) and tetrahedral $109.47^\circ$ | §4.7 |
| **Challenge Obj 1** | Rotlib API cis-proline | `is_c_terminal: bool` → `cis_proline: bool` across all `RotamerLibrary` methods with explicit CPR routing | §4.3, §4.4 |
| **Challenge Obj 2** | $P_{\text{coll}}$ factor-of-100 mismatch | $P(b)/100$ fractional probability, aligning with $\tau_{\text{lo}}=0.5$, $\tau_{\text{hi}}=2.0$ | §2.5, §3.3 |
| **Challenge Obj 3** | Terminology & missing `compute_freedom` | "RMS freedom"; public `compute_freedom` (now with `FreedomMode`) | §2.5, §3.1, §3.3 |
| **Challenge Obj 4** | Missing backbone loaders & public APIs | Signatures for `load_pdb_f64`, `extract_f64_backbone`, `residue_id`, `constrained_contacts`, `precondition` | §3.1 |
| **Challenge Obj 5** | Ambiguous #1933 fallback contract | `GeometryFallback` and `FallbackKind` attached to `ResidueGeometryTable` | §4.7 |
| **Challenge Obj 6** | Embedded reference tables in tests | Fixture table: no embedded reference tables; invariant tests only | §5.1 |
| **Challenge Obj 7** | External tool names in metadata | Metadata/docs cite literature only | §1.1, §3.7 |
| **Challenge Obj 8** | Procedural Rust code in spec | Declarative mathematical invariants | §4.2, §4.7 |
| **Challenge Obj 9** | Chain boundaries without TER | `res_i.chain_id != res_{i-1}.chain_id` is a boundary condition | §3.6 |
| **Challenge Obj 10** | Zero partition function division | $\Omega_i=0\lor\Omega_j=0\implies c(i,j)=0.0$ | §2.1, §3.3 |
| **Challenge Obj 12** | Fixer repack portable paths | `CARGO_MANIFEST_DIR` + relative traversal with workspace fallback | §4.6 |
| **Doc Fix** | Contact-degree citation | Zheng et al. (2015) and Zheng & Grigoryan (2017) | §3.7 |
| **Doc Fix** | `interference` endpoint doc | Both endpoints must belong to the query residue set | §3.7 |
| **Owner decision** | Freedom, interference, crowdedness, Cβ clash, scope, cutoffs | Defined in §2.4–§2.6, §3.8 and exposed as typed options with defaults and validation | §3.8, §3.9 |
