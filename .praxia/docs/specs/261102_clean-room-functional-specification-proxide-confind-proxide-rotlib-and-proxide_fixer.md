---
title: 'Clean-Room Functional Specification: proxide-confind, proxide-rotlib, and proxide_fixer'
description: Clean-room behavioral specification for de-deriving proxide-confind, proxide-rotlib, and proxide_fixer under MIT license, covering mathematical formulas with citations, Option<f64> dihedrals (#1897), IUPAC omega (#1903), chain semantics (#1904), Mosaist reader segregation to proxide-rotlib-mosaist, fixer repack tests (#1891), and CCD parser fallbacks (#1933).
status: draft
task_id: 261102_proxide-mosaist-cleanroom
date: '261102'
backlog_ids: ''
adversarial_review: ''
---
# Clean-Room Functional Specification: `proxide-confind`, `proxide-rotlib`, and `proxide_fixer`

**Document ID:** `SPEC-261102-MOSAIST-CLEANROOM`  
**Task ID:** `261102_proxide-mosaist-cleanroom`  
**Author:** Spec Team Lead (`spec_author`) & Spec Defender (`spec_defender_cleanroom`)  
**Date:** 2026-11-02  
**Status:** REVISED SPECIFICATION (Adversarial Defense Remediated — Revision 1.1)  
**Target Crates:** `crates/proxide-confind`, `crates/proxide-rotlib`, `crates/proxide_fixer`, `crates/proxide-rotlib-mosaist`  

---

## 1. Overview & Clean-Room Architecture

### 1.1 Objective & Clean-Room De-Derivation Principles
The objective of this clean-room re-specification is to eliminate all intellectual property taint originating from upstream Mosaist (licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International, CC BY-NC-SA 4.0). By establishing independent functional specifications derived strictly from peer-reviewed scientific literature and public Rust type signatures, `proxide-confind`, `proxide-rotlib`, and `proxide_fixer` will be re-implemented as pure MIT / Apache-2.0 licensed software suitable for commercial distribution on `crates.io`.

The clean-room engineering boundary adheres to the following strict principles:
1. **Behavioral Specification Only**: Specifications describe mathematical equations, geometric transformations, physical invariants, data schemas, public APIs, input preconditions, and postconditions. Internal loop nesting structures, private variable naming conventions, AST topologies, and idiosyncrasies of upstream C++ implementations are strictly excluded.
2. **Strict Licensing Isolation**: Any parsing logic, conversion tooling, or file format specifications specific to the proprietary Mosaist binary format (`rotlib.bin`) are segregated into a standalone, unpublishable companion crate (`proxide-rotlib-mosaist`) marked `publish = false` with an explicit CC BY-NC-SA 4.0 `NOTICE`.
3. **Zero Upstream Ingestion**: Production MIT crates (`proxide-confind`, `proxide-rotlib`, `proxide_fixer`, `proxide-frag`) must have zero direct or indirect compile-time or runtime dependencies on `proxide-rotlib-mosaist`.
4. **Independent Test Invariants**: All hardcoded test golden arrays, upstream PDB coordinate files (`small.pdb`, `1DC7.pdb`), and binary rotamer tables (`rotlib.bin`) are eliminated from the MIT test suites. Correctness is verified through physical conservation laws, geometric invariants, synthetic structural generators, and out-of-tree black-box differential parity testing pre-registered with `bth`.
5. **Upstream Metadata Scrubbing**: All package descriptions in `Cargo.toml`, crate-level docstrings in `lib.rs`, and source comments must be thoroughly scrubbed of upstream tool names (`Mosaist`, `mstcondeg`, `msttransforms.cpp`), citing solely peer-reviewed scientific literature (Zheng et al. 2015, Zheng & Grigoryan 2017, Dunbrack 1997, Engh & Huber 1991).

---

### 1.2 Crate Segregation & Workspace Boundaries

```
+-----------------------------------------------------------------------------------------+
|                                  PROXIDE WORKSPACE                                      |
+-----------------------------------------------------------------------------------------+
|                                                                                         |
|  [ crates/proxide-confind ] (Target: MIT)                                              |
|  * Clean mathematical contact degree, crowdedness, and structural freedom               |
|  * Option<f64> dihedral angles (Debt #1897)                                             |
|  * Standard IUPAC omega torsion & segmented chain semantics (Debts #1903, #1904)         |
|  * Depends on: proxide-rotlib, proxide-core, proxide-geometry, orx-parallel, dashmap    |
|                                                                                         |
|  [ crates/proxide-rotlib ] (Target: MIT)                                                |
|  * Protobuf-backed Dunbrack rotamer library engine (load_pb, load_pb_zst)               |
|  * Standard Gram-Schmidt orthonormal coordinate frame (Dunbrack 1997)                   |
|  * CCD parser with explicit token validation & cited fallbacks (Debt #1933)             |
|  * Clean query & placement APIs; zero Mosaist binary format code                        |
|                                                                                         |
|  [ crates/proxide_fixer ] (Target: MIT)                                                 |
|  * High-throughput atomic model prep, sidechain repacking, point mutagenesis            |
|  * Inlined canonical amino-acid set (decoupled from proxide-confind)                    |
|  * Repack test suite running against committed Dunbrack Protobuf asset (Debt #1891)     |
|                                                                                         |
|  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  |
|  AIR-GAP BOUNDARY (No production dependencies from MIT crates)                          |
|  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  |
|                                                                                         |
|  [ crates/proxide-rotlib-mosaist ] (License: CC BY-NC-SA 4.0, publish = false)          |
|  * Mosaist rotlib.bin binary reader (load, validate_grid with phi-major sort check)     |
|  * Offline converter CLI (parse_master) emitting non-commercial Protobuf blobs          |
|  * Strictly isolated; unpublishable; ignored by cargo tree -e normal from MIT crates    |
|                                                                                         |
+-----------------------------------------------------------------------------------------+
```

---

### 1.3 Module Reconstruction Boundary

| Crate | Module | Current Status | Clean-Room Strategy |
|---|---|---|---|
| `proxide-confind` | `src/coords.rs` | Independent | **Retain**: Update `phi` and `psi` from `f64` (with `9999.0` sentinel) to `Option<f64>`. Implement segmented chain semantics. |
| `proxide-confind` | `src/precondition.rs` | Independent | **Retain**: Adapt dihedral validation to `Option<f64>`. Preserve precondition reporting. |
| `proxide-confind` | `src/grid.rs` | Independent | **Retain**: Pure mathematical 3D spatial hash index (`ProximityGrid<T>`) in f64. |
| `proxide-confind` | `src/contact_list.rs` | Independent | **Retain**: Holds `ContactList` container and `CONTACT_THRESHOLD = 0.02`. |
| `proxide-confind` | `src/error.rs` | Independent | **Retain**: Standard `ConFindError` enum. |
| `proxide-confind` | `src/cache.rs` | Tainted | **Delete & Re-implement**: Clean Phase A sidechain placement, local backbone clash filtering, and residue caching from literature equations. |
| `proxide-confind` | `src/parallel.rs` | Tainted | **Delete & Re-implement**: Clean Phase B pairwise contact degree reduction using parallel batch processing (`orx-parallel` / `rayon`). Remove all C++ line comments. |
| `proxide-confind` | `src/freedom.rs` | Tainted | **Delete & Re-implement**: Clean Phase C structural freedom aggregation from collision probabilities. |
| `proxide-confind` | `src/confind.rs` | Tainted | **Delete & Re-implement**: Clean pipeline orchestrator for caching, contact queries, and crowdedness. Fix doc contradictions. |
| `proxide-confind` | `src/params.rs` | Tainted | **Delete & Re-implement**: Re-specify background amino-acid frequencies and physical cutoffs with peer-reviewed literature citations. |
| `proxide-rotlib` | `src/rotlib.rs` | Mixed | **Segregate**: Extract `load()`, `validate_grid()`, and `to_protobuf()` into `proxide-rotlib-mosaist`. Retain `load_pb()` and clean query APIs in `proxide-rotlib`. |
| `proxide-rotlib` | `src/frame.rs` | Minor Taint | **Re-document**: Remove Mosaist source line comments; specify standard Gram-Schmidt orthonormalization. |
| `proxide-rotlib` | `src/sidechain.rs` | Independent | **Retain**: Formalize IUPAC sidechain heavy-atom predicates and alanine $C_\beta$ handling. |
| `proxide-rotlib` | `src/geometry/ccd_parser.rs` | Independent Bug | **Patch**: Enforce exact CIF token validation; eliminate silent 1.5 Å / 109.5° defaults (Debt #1933). |
| `proxide_fixer` | `src/mutate.rs` | External Coup. | **Decouple**: Inline `CANONICAL_AA_NAMES`; remove `proxide-confind` dependency. |
| `proxide_fixer` | `src/repack.rs` | Test Flaw | **Patch**: Migrate repack tests to workspace Dunbrack Protobuf asset; fail loudly on corrupt load (Debt #1891). |

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
  $$w(r_a, r_b) = \begin{cases} 1 & \text{if } \exists \, k \in \text{atoms}(r_a), \, l \in \text{atoms}(r_b) \text{ such that } \|\vec{x}_{ik} - \vec{x}_{jl}\| \le d_{\text{cont}} \\ 0 & \text{otherwise} \end{cases}$$
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
   Any sidechain heavy atom placed closer than $2.0$ Å to any backbone heavy atom ($N, C_\alpha, C, O$) of the protein represents an unphysical hard steric overlap (van der Waals overlap $> 0.8$ Å) and causes that rotamer to be pruned in Phase A.
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
Residue crowdedness measures the degree of steric confinement imposed on residue $i$ by the surrounding fixed protein backbone (Zheng & Grigoryan 2017, S1 Text).

Let $\mathcal{R}_i^{\text{lib}}(a)$ be the complete set of rotamers for amino acid $a$ tabulated in the rotamer library at dihedrals $(\phi_i, \psi_i)$. A rotamer $r \in \mathcal{R}_i^{\text{lib}}(a)$ is marked as clashing with the backbone if:
$$\exists \, k \in \text{atoms}(r), \, \exists \, l \in \text{backbone atoms} \text{ such that } \|\vec{x}_{ik} - \vec{x}_{\text{bb}, l}\| \le d_{\text{clash}} = 2.0 \text{ Å}$$
The surviving rotamer set is:
$$\mathcal{R}_i(a) = \left\{ r \in \mathcal{R}_i^{\text{lib}}(a) \mid \min_{k, l} \|\vec{x}_{ik} - \vec{x}_{\text{bb}, l}\| > 2.0 \text{ Å} \right\}$$
The residue crowdedness $C(i)$ is the weighted fraction of rotamer probability pruned by steric clash with the backbone:
$$C(i) = 1.0 - \frac{\sum_{a \in \mathcal{A}_{18}} P(a) \sum_{r \in \mathcal{R}_i(a)} P(r \mid a, \phi_i, \psi_i)}{\sum_{a \in \mathcal{A}_{18}} P(a) \sum_{r \in \mathcal{R}_i^{\text{lib}}(a)} P(r \mid a, \phi_i, \psi_i)}$$
**Properties:**
- $C(i) \in [0.0, 1.0]$.
- $C(i) = 0.0$: Completely solvent-exposed, zero backbone occlusion.
- $C(i) = 1.0$: Completely buried or sterically blocked; no sidechains can fit.

---

### 2.5 Residue Structural Freedom ($F(i)$)
Structural freedom quantifies the residual conformational flexibility available to residue $i$ when considering mutual sidechain crowding from neighboring residues (Zheng & Grigoryan 2017, S1 Text).

For each surviving rotamer $r_a \in \mathcal{R}_i(a)$, the cumulative collision probability $P_{\text{coll}}(r_a)$ against all neighboring residues $j \in \text{neighbors}(i)$ is accumulated across the ensemble. Because the background propensities $P(b)$ in Table 2.2 are expressed in percent ($P(b) \in [1.5\%, 8.8\%]$), $P(b)$ is normalized by $\frac{1}{100.0}$ to yield fractional probability:
$$P_{\text{coll}}(r_a) = \sum_{j \in \text{neighbors}(i)} \sum_{b \in \mathcal{A}_{18}} \frac{P(b)}{100.0} \sum_{r_b \in \mathcal{R}_j(b)} P(r_b \mid b, \phi_j, \psi_j) \, w(r_a, r_b)$$
Two empirical collision thresholds define the accessibility of rotamers:
- Low-collision threshold: $\tau_{\text{lo}} = 0.5$
- High-collision threshold: $\tau_{\text{hi}} = 2.0$

Let:
- $n_1$ be the number of rotamers with $P_{\text{coll}}(r) < \tau_{\text{lo}}$ (unhindered rotamers, including uncontested rotamers with $P_{\text{coll}} = 0$).
- $n_2$ be the number of rotamers with $P_{\text{coll}}(r) < \tau_{\text{hi}}$ (moderately accessible rotamers, including uncontested rotamers).
- $N_{\text{lib}}$ be the total number of rotamers in the unpruned library for that residue.

The RMS freedom formulation is defined as:
$$F(i) = \frac{1}{N_{\text{lib}}} \sqrt{\frac{n_1^2 + n_2^2}{2}}$$
*(Alternative formulations and parked gap details are specified in Section 3.8).*

---

### 2.6 Mutual Interference ($I(i, j)$)
Mutual interference measures steric clash between the placed sidechains of residue $i$ and the fixed backbone atoms of residue $j$:
$$I(i, j) = \sum_{a \in \mathcal{A}_{18}} P(a) \sum_{r_a \in \mathcal{R}_i(a)} P(r_a \mid a, \phi_i, \psi_i) \, \mathbf{1}_{\text{clash}}(r_a, \text{backbone}_j)$$
where $\mathbf{1}_{\text{clash}}(r_a, \text{backbone}_j) = 1$ if any sidechain atom of $r_a$ is within $2.0$ Å of any backbone heavy atom of residue $j$.

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

#### Pipeline Interface (`src/confind.rs`)
```rust
pub struct ConFind {
    pub rotlib: Arc<RotamerLibrary>,
    pub strict: bool,
    pub backbone: Arc<ProteinBackbone>,
    // Private spatial acceleration structures and concurrent caches
}

impl ConFind {
    /// Construct ConFind instance and build spatial indices.
    pub fn new(
        rotlib: Arc<RotamerLibrary>,
        backbone: Arc<ProteinBackbone>,
        strict: bool,
    ) -> Result<Self, ConFindError>;

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

    /// Query residues within 25.0 Å Cα–Cα distance cutoff.
    pub fn neighbors(&self, ri: ResidueIndex) -> Vec<ResidueIndex>;

    /// Compute pairwise contact degrees across query residue set.
    /// Returns pairs with contact degree > cd_cut.
    pub fn contacts(
        &self,
        residues: &[ResidueIndex],
        cd_cut: f64,
    ) -> Result<ContactList, ConFindError>;

    /// Compute mutual sidechain-to-backbone interference.
    /// Evaluates pairs where BOTH endpoints are present in `residues`.
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

    /// Query fraction of rotamers pruned by backbone clash for residue `ri`.
    pub fn crowdedness(&self, ri: ResidueIndex) -> Result<f64, ConFindError>;

    /// Query residual conformational freedom for residue `ri`.
    /// Residue must have been included in preceding `contacts()` query set.
    pub fn freedom(&self, ri: ResidueIndex) -> Result<f64, ConFindError>;
}
```

#### Freedom Formulation (`src/freedom.rs`)
```rust
/// Compute residue structural freedom from surviving rotamer collision probabilities.
///
/// Under the RMS freedom formulation:
/// - Uncontested rotamers (zero collision with neighbors) = n_surviving - collision_counts.len()
/// - n_1 = n_uncontested + count(P_coll < lo_cut)
/// - n_2 = n_uncontested + count(P_coll < hi_cut)
/// - Freedom = sqrt((n_1^2 + n_2^2) / 2) / n_library
pub fn compute_freedom(
    collision_counts: &[f64],
    n_surviving: usize,
    n_library: usize,
    lo_cut: f64,
    hi_cut: f64,
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

    #[error("freedom not computed for {0:?} — residue must be included in contacts() query set")]
    FreedomNotComputed(ResidueIndex),

    #[error("structure failed {0} structural precondition check(s); inspect report for diagnostics")]
    PreconditionsFailed(usize),

    #[error("rotamer library error: {0}")]
    RotlibError(#[from] proxide_rotlib::RotlibError),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}
```

---

### 3.3 Module Decomposition & Reconstruction Plan

1. **`src/cache.rs` (Re-implemented Cleanly)**:
   - Evaluates Phase A rotamer compatibility: places rotamers using `rotlib.place_rotamer(aa, phi, psi, rot_idx, cis_proline, ...)`.
   - Queries backbone spatial index (`ProximityGrid`) for heavy-atom clashes $\le 2.0$ Å.
   - Accumulates crowdedness fraction $C(i)$ and sidechain-to-backbone interference.
   - For Alanine: treats $C_\beta$ as permanent contact via `counts_as_sidechain`.
   - For Glycine: zero sidechains placed; crowdedness defined as 0.0.
2. **`src/parallel.rs` (Re-implemented Cleanly)**:
   - Evaluates Phase B pairwise contact degrees for all candidate pairs $(i, j)$ with $d(C\alpha_i, C\alpha_j) \le 25.0$ Å.
   - Parallelized across chunks using `orx-parallel` or `rayon`.
   - Accumulates cross-rotamer heavy-atom contacts $\le 3.0$ Å weighted by $P(a) P(b) P(r_a) P(r_b)$ and normalizes by $\Omega_i \Omega_j$.
   - Short-circuits to $0.0$ if $\Omega_i == 0.0$ or $\Omega_j == 0.0$ to prevent division by zero.
   - Strips all comments referencing C++ line numbers or historical compiler quirks.
3. **`src/freedom.rs` (Re-implemented Cleanly)**:
   - Evaluates Phase C conformational freedom $F(i)$ from collision probabilities.
   - Implements clean public function:
     ```rust
     pub fn compute_freedom(
         collision_counts: &[f64],
         n_surviving: usize,
         n_library: usize,
         lo_cut: f64,
         hi_cut: f64,
     ) -> f64;
     ```
   - Behavior:
     - Returns $0.0$ if $n_{\text{library}} == 0$.
     - Computes uncontested rotamers: $n_{\text{uncontested}} = (n_{\text{surviving}} - \text{collision\_counts.len()})$ (saturating at 0).
     - Counts accessible rotamers using normalized collision probabilities:
       $n_1 = n_{\text{uncontested}} + |\{c \in \text{collision\_counts} \mid c < \text{lo\_cut}\}|$
       $n_2 = n_{\text{uncontested}} + |\{c \in \text{collision\_counts} \mid c < \text{hi\_cut}\}|$
     - Evaluates RMS freedom: $F = \frac{1}{n_{\text{library}}} \sqrt{\frac{n_1^2 + n_2^2}{2}}$.
4. **`src/params.rs` (Re-implemented Cleanly)**:
   - Exposes public physical constants `CLASH_DIST = 2.0`, `CONT_DIST = 3.0`, `DCUT = 25.0`, `LO_COLL_PROB_CUT = 0.5`, `HI_COLL_PROB_CUT = 2.0`.
   - Exposes `aa_propensity(aa: &str) -> f64` returning percentage values $P(a) \in [1.5, 8.8]$ with citations to Zheng et al. (2015).
   - In collision probability summation, $P(b)$ is divided by $100.0$ to ensure proper calibration against collision thresholds $\tau_{\text{lo}} = 0.5$ and $\tau_{\text{hi}} = 2.0$.
5. **`src/confind.rs` (Re-implemented Cleanly)**:
   - Orchestrates caching, spatial indexing, pairwise dispatch, residue ID resolution, and metrics extraction.

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
- **Defect Description**: Prior versions calculated $\omega$ with a sign opposite to upstream Mosaist due to vector inversion.
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

### 3.7 Documentation & Package Metadata Corrections
1. **`lib.rs:36-37` Erroneous Coiled-Coil Citation**:
   - *Current*: Cites Grigoryan & DeGrado (2011) *J Mol Biol* 405(4):1079-1100 (helical bundle parameterization).
   - *Correction*: Replace with:
     - Zheng, F., Zhang, J., & Grigoryan, G. (2015). "Tertiary Structural Propensities Reveal Fundamental Sequence/Structure Relationships in Proteins." *Structure*, 23(5), 961–971.
     - Zheng, W., & Grigoryan, G. (2017). "Sequence statistics of tertiary structural motifs reflect protein stability." *PLoS ONE*, 12(5), e0178272.
2. **`confind.rs:184` Contradictory Endpoint Documentation**:
   - *Current*: `/// Interference: pairs where EITHER endpoint is in residues and value >= incut.`
   - *Correction*: Update docstring to state: `/// Interference: pairs where BOTH endpoints are present in the query set and value >= incut.`
3. **`parallel.rs:37-55` Upstream C++ Bug Comments**:
   - *Current*: Detailed comments regarding `contactDegree.cpp line 209` and upstream C++ variable typos.
   - *Correction*: Purge all upstream file names, line numbers, and bug-replication logic.
4. **`Cargo.toml` and `lib.rs` Upstream Tool Names**:
   - *Current*: `crates/proxide-confind/Cargo.toml` states `description = "... reimplementation of Mosaist mstcondeg"`, and `lib.rs` references `Mosaist's mstcondeg`.
   - *Correction*: Update `Cargo.toml` description to: `Sequence-independent contact degree, crowdedness, and structural freedom algorithm for protein backbones`. Scrub `lib.rs` and all crate docs of `Mosaist` and `mstcondeg`, citing exclusively Zheng et al. (2015) and Zheng & Grigoryan (2017).

---

### 3.8 Parked Gaps & Owner Feedback Beat (`awaiting_feedback`)

```markdown
### Parked Gap: Non-Standardized Empirical Quantities in ConFind (Awaiting Owner Feedback)

**Context**:
While the pairwise contact degree $c(i, j) = \sum w(d)$ and canonical threshold $c > 0.02$ are rigorously derived and published in Zheng, Zhang & Grigoryan (2015) and Zheng & Grigoryan (2017) Eq. 9, several auxiliary metrics and cutoffs in ConFind represent empirical heuristics developed in the Grigoryan laboratory that are not formally standardized in public peer-reviewed literature:
1. **Structural Freedom from Collision Probabilities**: The multi-mode formula for freedom ($F(i) = \frac{1}{N_{\text{lib}}} \sqrt{\frac{n_1^2 + n_2^2}{2}}$) and the specific collision thresholds ($\tau_{\text{lo}} = 0.5$, $\tau_{\text{hi}} = 2.0$) are empirical heuristics.
2. **Mutual Interference Metric**: Pairwise sidechain-to-backbone clash integration ($I(i, j)$) is defined in Mosaist tooling but rarely referenced in tertiary structural motif publications.
3. **Geometric Cutoffs**: The backbone clash distance ($d_{\text{clash}} = 2.0$ Å) and neighbor cutoff ($D_{\text{cut}} = 25.0$ Å).

**Decision Required from Project Owner (`awaiting_feedback`)**:
Please select one of the following paths for these empirical quantities in the clean-room implementation:

- **Option A (Drop / Deprecate Auxiliary Metrics)**:
  Deprecate `compute_freedom`, `ConFind::freedom`, and `ConFind::interference`, retaining strictly the mathematically published contact degree $c(i, j)$ and residue crowdedness $C(i)$.
- **Option B (Independent Definition & Documented Divergence - RECOMMENDED)**:
  Retain the existing public API signatures, implement the clean mathematical definitions specified in Sections 2.4–2.6 with explicit literature-divergence documentation, and provide unit tests verifying expected physical invariants.
- **Option C (Consult Grigoryan Laboratory)**:
  Pause implementation of the freedom and interference submodules to solicit formal clarification or public domain release documentation from the Grigoryan laboratory.
```

---

## 4. `proxide-rotlib` & `proxide_fixer` Functional Specification

### 4.1 Segregation of Mosaist Reader to `crates/proxide-rotlib-mosaist`
All code supporting the parsing and conversion of Mosaist's proprietary binary format (`rotlib.bin`) is segregated into a dedicated crate:

#### Package Metadata (`crates/proxide-rotlib-mosaist/Cargo.toml`)
```toml
[package]
name = "proxide-rotlib-mosaist"
version.workspace = true
edition.workspace = true
license = "CC-BY-NC-SA-4.0"
publish = false
description = "Mosaist binary rotlib.bin format reader and Protobuf converter (Non-Commercial)"

[dependencies]
proxide-rotlib = { path = "../proxide-rotlib" }
prost = "0.13"
zstd = "0.13"
clap = { version = "4.4", features = ["derive"] }
tracing = "0.1"
thiserror = { workspace = true }
```

#### NOTICE Requirements (`crates/proxide-rotlib-mosaist/NOTICE`)
Must state clearly that the crate implements format specifications of the Mosaist/MASTER protein design suite, is licensed under CC BY-NC-SA 4.0, and must not be used commercially or committed into commercial distributions.

#### Clean-Room MIT Crates
- `crates/proxide-rotlib` retains strictly `license = "MIT"` and zero dependencies on `proxide-rotlib-mosaist`.
- Production Protobuf loading is executed exclusively via:
  ```rust
  impl RotamerLibrary {
      pub fn load_pb(path: &Path) -> Result<Self, RotlibError>;
      pub fn load_pb_zst(path: &Path) -> Result<Self, RotlibError>;
  }
  ```
- Existing binary format tests in `crates/proxide-rotlib/tests/test_load.rs` and `crates/proxide-rotlib/tests/test_parse_master.rs` must be relocated to `crates/proxide-rotlib-mosaist/tests/` to maintain clean workspace segregation.

---

### 4.2 Phi-Major Sorting Validation Invariants in `load()`
In the segregated `proxide-rotlib-mosaist` crate, `validate_grid` must enforce strict phi-major ordering as a formal mathematical invariant:
1. **Dimension Match**: The total number of bins must equal the exact product of dihedral centers: $|\Phi| \times |\Psi| = |\text{bins}|$.
2. **Lexicographical Product Invariant**: The grid array must follow phi-major ordering (outer iteration over $\phi$, inner iteration over $\psi$). For every bin index $k \in [0, |\Phi| \cdot |\Psi| - 1]$, stored dihedral centers $(\phi_k, \psi_k)$ must match the expected grid centers within numerical tolerance $\epsilon = 10^{-4}$:
   $$|\phi_k - \phi_{\lfloor k / |\Psi| \rfloor}| \le 10^{-4}, \quad |\psi_k - \psi_{k \pmod{|\Psi|}}| \le 10^{-4}$$
3. **Failure Contract**: Any dimension mismatch or deviation from phi-major ordering returns an explicit `RotlibError::InvalidFormat` describing the mismatched bin index and coordinates.

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

### 5.1 Elimination of Tainted Test Fixtures & Constants
The clean-room test suite must not contain any data derived from upstream Mosaist runs or proprietary binary files:

| Existing Tainted Asset | Location | Elimination & Replacement Strategy |
|---|---|---|
| `small.pdb` | `tests/common/mod.rs` | Replace with synthetic helical generator `make_synthetic_backbone(n, spacing)` |
| `1DC7.pdb` | `tests/common/mod.rs` | Vendor clean crystallographic coordinates directly from wwPDB into `tests/data/1dc7.pdb` |
| `rotlib.bin` | `tests/common/mod.rs` | Replace with `data/rotlibs/proxide-rotlib-bbdep2010.pb.zst` |
| Upstream path defaults | `tests/common/mod.rs:231,240,258` | Purge all `/home/marielle/repos/mosaist/testfiles/` paths; resolve workspace paths or synthetic fixtures |
| `REF_CONTACTS` (60 pairs) | `test_parity_small_pdb.rs` | Purge from MIT crate; verify via out-of-tree black-box parity harness |
| `test_drift_loadpb_small_pdb.rs` | `crates/proxide-confind/tests/` | **Complete removal** from MIT crate; purge hardcoded Mosaist `REF_CONTACTS` table |
| `test_parity_1dc7.rs` | `crates/proxide-confind/tests/` | **Complete removal** from MIT crate; purge hardcoded Mosaist `REF_CROWDEDNESS*`, `REF_FREEDOM*`, `REF_CONTACTS*` |
| `REF_INTERFERENCE` (1,700+ entries) | `test_parity_small_pdb.rs` | Purge from MIT crate |
| `REF_DISTOGRAM*` (13,000+ floats) | `test_distogram.rs` | Replace with mathematical invariant tests (symmetry, triangle inequality, SE(3) invariance) |
| `REF_CB` float array | `test_placement.rs` | Replace with exact analytical coordinate derivation |
| `test_load.rs` & `test_parse_master.rs` | `crates/proxide-rotlib/tests/` | **Relocate** to `crates/proxide-rotlib-mosaist/tests/` to prevent test failures on segregated crate |

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

### 5.4 Out-of-Tree Black-Box Parity Testing Protocol with `bth`
To confirm numerical parity with upstream Mosaist without copyright contamination:

1. **Isolated Execution Environment**:
   - Upstream Mosaist is compiled in `/tmp/mosaist_scratch/` outside the workspace.
   - The parity evaluation script runs strictly in `/tmp/`.
2. **Pre-Registration via `bth`**:
   - Every verification run is pre-registered with a committed `.bth.toml` sidecar and tracked script.
3. **Data Isolation Barrier**:
   - The parity runner compares clean-room output against upstream output:
     $$\max_{i, j} |c_{\text{clean}}(i, j) - c_{\text{upstream}}(i, j)| \le 5 \times 10^{-4}$$
   - The runner emits **only**:
     - Status: `VERDICT: PASS / FAIL`
     - Maximum absolute delta ($\Delta_{\max}$) and root-mean-square delta ($\Delta_{\text{RMS}}$).
   - Zero upstream output tables or C++ code fragments are committed into the `proxide` repository.

---

## 6. Traceability Matrix & Debt Reconciliation

| Debt / Issue ID | Summary | Clean-Room Resolution in Specification | Section |
|---|---|---|---|
| **#3184** | Clean-room rewrite of `proxide-confind` | `cache.rs`, `parallel.rs`, `freedom.rs`, `confind.rs`, `params.rs` deleted and re-specified from Zheng & Grigoryan (2017) | §1.3, §2.1–§2.6, §3.3 |
| **#3185** | Segregate `rotlib` and `fixer` | `load()`, `validate_grid()`, `to_protobuf()`, `parse_master` moved to `proxide-rotlib-mosaist`; MIT crates decoupled | §1.2, §4.1, §4.3 |
| **#1897** | Magic `9999.0` dihedral sentinel | Replaced with `Option<f64>` on `ResidueBackbone` across `confind` and `rotlib`; no sentinel floats | §3.1, §3.4, §4.3 |
| **#1891** | Fixer repack tests skip in CI | Point `load_rotlib()` to committed `proxide-rotlib-bbdep2010.pb.zst`; fail loudly on corrupt load; portable workspace path resolution; replace `eprintln!` with `tracing` | §4.6 |
| **#1903** | Omega dihedral sign convention | Standard IUPAC $C\alpha_{i-1}-C_{i-1}-N_i-C\alpha_i$ torsion adopted; cis defined by $|\omega| < 30^\circ$ | §3.5 |
| **#1904** | Chain semantics & breaks | PDB file order preserved; segmented on `chain_id` change, TER records, or $C\alpha-C\alpha > 4.5$ Å; dihedrals cleared at breaks | §3.6 |
| **#1933** | CCD parser fallbacks | Strict token validation; CIF null tokens handled as `None`; mandatory `GeometryFallback` records on `ResidueGeometryTable` with `tracing::warn!` citing Engh & Huber (1991) and tetrahedral $109.47^\circ$ | §4.7 |
| **Challenge Obj 1** | Rotlib API cis-proline hallucination | Conceded: fixed `is_c_terminal: bool` -> `cis_proline: bool` across all `RotamerLibrary` methods (§4.3, §4.4) with explicit CPR routing | §4.3, §4.4 |
| **Challenge Obj 2** | P_coll factor-of-100 dimensional mismatch | Conceded: normalized Eq. 2.5 with $\frac{P(b)}{100.0}$ fractional probability, aligning with $\tau_{\text{lo}} = 0.5$ and $\tau_{\text{hi}} = 2.0$ | §2.5, §3.3 |
| **Challenge Obj 3** | "Type 2" terminology & missing compute_freedom | Conceded: purged "Type 2" in favor of "RMS freedom formulation"; added public `compute_freedom` signature taking `&[f64]` | §2.5, §3.1, §3.3 |
| **Challenge Obj 4** | Missing backbone loaders & public APIs | Conceded: added public signatures for `load_pdb_f64`, `extract_f64_backbone`, `residue_id`, `constrained_contacts`, and `precondition` module | §3.1 |
| **Challenge Obj 5** | Ambiguous Debt #1933 fallback contract | Conceded: eliminated either/or branch; defined `GeometryFallback` and `FallbackKind`; attached to `ResidueGeometryTable` | §4.7 |
| **Challenge Obj 6** | Unpurged upstream test tables | Conceded: Table 5.1 expanded to mandate complete removal of `test_drift_loadpb_small_pdb.rs`, `test_parity_1dc7.rs`, and purging `/home/marielle/repos/mosaist` paths from `tests/common/mod.rs` | §5.1 |
| **Challenge Obj 7** | Upstream tool names in metadata | Conceded: mandated scrubbing `Cargo.toml` description and `lib.rs` docs of `mstcondeg`/`Mosaist` | §1.1, §3.7 |
| **Challenge Obj 8** | Procedural Rust code in spec | Conceded: replaced imperative loop snippets with formal declarative mathematical invariants | §4.2, §4.7 |
| **Challenge Obj 9** | Chain boundaries without TER | Conceded: explicitly added `res_i.chain_id != res_{i-1}.chain_id` as boundary condition | §3.6 |
| **Challenge Obj 10** | Zero partition function division | Conceded: specified analytical short-circuit $\Omega_i = 0 \lor \Omega_j = 0 \implies c(i, j) = 0.0$ | §2.1, §3.3 |
| **Challenge Obj 11** | Rotlib test file relocation | Conceded: mandated moving `test_load.rs` and `test_parse_master.rs` to `proxide-rotlib-mosaist/tests/` | §4.1, §5.1 |
| **Challenge Obj 12** | Fixer repack portable paths | Conceded: specified `CARGO_MANIFEST_DIR` + relative traversal with workspace fallback | §4.6 |
| **Doc Fix** | `lib.rs:36-37` citation error | Replace helical bundle paper with Zheng et al. (2015) and Zheng & Grigoryan (2017) | §3.7 |
| **Doc Fix** | `confind.rs:184` endpoint doc | Correct docstring: both endpoints must belong to query residue set | §3.7 |
| **Doc Fix** | `parallel.rs:37-55` C++ comments | Purge all C++ line comments and upstream bug-replication logic | §3.7 |
| **Parked Gap** | Empirical freedom / cutoffs | Parked for owner decision (`awaiting_feedback`) with Options A, B (recommended), C | §3.8 |
