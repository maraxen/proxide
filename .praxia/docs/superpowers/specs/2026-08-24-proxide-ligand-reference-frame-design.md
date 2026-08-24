# Proxide Canonical Reference Frame for Ligands — Interface Spec (Draft v2)

**Status:** Revised post-adversarial-review
**Consumer:** demistify idea-002 (ligand extension)
**Provider:** proxide (`proxide-gaff2` worktree, `gaff2-parity/rust-port` @ `e007792`, plus new work proposed here)
**Scope:** Define what proxide must expose so a ligand can enter demistify's VMM→DBSCAN→MIST pipeline the way a protein residue's phi/psi does today.

---

## Changes from draft v1

This revision folds in every challenger finding the defender did not successfully rebut, plus the gaps the defender's own code-verification pass surfaced. Summary, most consequential first:

1. **Corrected a false factual premise and reversed the ring/aromaticity decision (Finding 1).** The claim "neither exists on `Molecule` today (confirmed zero grep hits)" was checked against the actual cited commit and is false: `Molecule` already has a `bond_aromatic` field and a working, edge-case-hardened `_to_rdkit()` sanitization path, and `rdkit` is already a real dependency. §4's "resolved, not left open" native-Rust SSSR/aromaticity commitment is walked back: v1 now reuses the existing RDKit-based path via the Python layer and passes rings/aromaticity into the Rust layer as plain inputs (mirroring how `assign_gaff_atom_types` already takes them). This is a real scope reduction, not a re-opened TBD — see §1 and §4.
2. **`RingPucker` is now actually defined** (traversal-order and phase-origin conventions pinned), and the §2a/§2b contradiction on ring-size coverage is resolved with an explicit, bounded rule plus a new `unrepresented_ring_dof` escape hatch for macrocycles (≥9-membered rings) instead of silent omission (Findings 2, 4).
3. **Canonical-ranking tie-breaking is now specified**, including the honest residual limitation for true graph-automorphic atoms (Finding 3).
4. **Torsion definition is now fully deterministic**: branch-point substituent selection is specified, and a bounded amide/ester/conjugated-bond exclusion list closes the "spurious near-flat torsion channel" failure mode (Findings 6, 7).
5. **Partial-charge caching is fixed**: the cache/fingerprint key now includes a reference-geometry hash and an Espaloma-weights-version token, not graph identity alone — closing a real silent-collision correctness bug (Findings 5, 15b).
6. **Errors are now typed** (`LigandFrameError` enum) instead of `String`, both `canonicalize_ligand_topology` and `extract_ligand_frame_coordinates` return `Result`, and a topology↔positions atom-identity contract check is added (Findings 8, 10, and the defender's own concession on the missing error path).
7. **New explicit non-goals**: multi-fragment/disconnected input (Finding 9) and bond-order-less input such as PDB `CONECT` records (Finding 14) are out of scope for v1, with the caller-side responsibility stated plainly instead of left unaddressed.
8. **`ligand_id` is now a defined, required input** (Finding 13), and per-array index-alignment is asserted explicitly (Finding 15a).
9. **A reference-geometry validity gate precedes Espaloma inference** (Finding 11).
10. **A new `frame_validity` output field addresses the defender's own verified gap** (pipeline hard-fails on any NaN, and the only masking granularity that exists is static per-entity, not per-frame) — documented as a genuine blocking dependency on a follow-up pipeline-side change, not swept under "zero pipeline-side changes."
11. **`positions` axis order is unified** (was self-contradictory between §2b's table and the Rust struct) — frame-major is kept (matches MD-reader convention and the actual struct/function signatures) and the table is fixed to match, with a note on why this one field is the exception to the otherwise entity-first convention.
12. **§2b's "zero pipeline-side changes" claim is scoped precisely**: the defender verified this is true for the D-genericity code path specifically; it is no longer allowed to imply the D=1-vs-grouped architectural choice itself is settled, or that frame-level validity handling is free (item 10 above). §5 item 1 is unchanged as an open question but the language conflict is resolved (Finding 12).
13. Sections defended successfully are left as-is, with the defender's reasoning now recorded in the spec text itself (§1's non-goals, §2's bond-length/angle extraction rationale, §3's single-reference-frame charge convention) — see inline "Why this stands" notes.

---

## 1. What "canonical reference frame" means here

A ligand's canonical reference frame is **a topology-derived atom ordering and a fixed set of static annotations keyed to that ordering, computed once per unique molecular graph and held constant across every conformer/frame of a trajectory.**

Grounded in the RDKit/OpenFF convention surfaced by research: canonicalization is a **graph-labeling problem**, never a geometric one. `Chem.CanonicalRankAtoms`-style algorithms (classic Morgan/EC refinement, or the more robust O'Boyle & Sayle "Get Your Atoms in Order" variant) derive atom rank purely from graph invariants — element, degree, charge, ring membership, aromaticity, neighbor-hash refinement — and **never reference 3D coordinates**. The practical payoff: the same molecule parsed from two MOL2 files with different atom-write orders (or the same MOL2 across different frames of a trajectory) collapses to one stable index space. This is the load-bearing property demistify needs: `MIST`, `dbscan`, and `fit_mixture` all index by a fixed axis-0 identity (`N_res` today), and a ligand's "residue-analog" axis has to be equally fixed or per-frame comparisons are meaningless.

**Canonical-ranking tie-breaking (resolved — previously a footnote, now a firm decision; closes adversarial Finding 3).** Refinement seeds from atomic number, degree, formal charge, aromaticity, and ring membership, then iterates neighbor-rank-multiset refinement to a fixed point (O'Boyle & Sayle). Two tie-break layers apply when the fixed point still leaves ties:
- **Layer 1 (structural, input-order-independent):** among tied atoms, prefer the one whose canonical-rank-sorted neighbor sequence is lexicographically smallest — the standard extension used to seed further refinement rounds, not a new invention.
- **Layer 2 (final fallback, input-order-dependent):** if ties still remain, this means the tied atoms are in a genuine graph automorphism (e.g., the three H's of a freely rotating terminal methyl, or symmetric para-substituent branches) — no invariant can distinguish them non-arbitrarily, by definition. Break by ascending original input atom index.

**Documented, accepted limitation:** Layer 2 means two different file-orderings of the identical molecule can assign different labels *among* a set of truly automorphic atoms specifically. This is called out explicitly rather than silently assumed away, but it does not break the property the rest of the document depends on: because such atoms are physically indistinguishable by definition of the symmetry, no downstream feature (torsion identity, distance-based state clustering) can produce a systematically different result depending on which physically-identical atom received which label. Non-automorphic atoms — the overwhelming majority in a typical ligand — are unaffected and remain fully input-order-independent via Layer 1 refinement alone.

Two things are explicitly **not** part of this definition, per the research's split:
- **Not a geometric/rotational frame.** No RMSD superposition, no fixed-axis alignment of the ligand in space. Positions stay whatever the trajectory gives; only atom *identity/order* is canonicalized. *Why this stands (adversarial review raised this as a candidate gap; defended successfully):* the pipeline discretizes states via DBSCAN/VMM operating on distances between fitted distributions, never on Cartesian RMSD — a rotational frame would be solving a problem this pipeline doesn't have.
- **Not per-frame.** Canonical order, atom typing, aromaticity, ring membership, and (per §3, with a caching fix) partial charges are single-conformer/topology-level facts, computed once and reused across all frames — mirroring how NequIP/MACE fix node (species) identity while only edges/geometry vary per frame, and how RDKit/OpenFF's canonical atom map is stable across every conformer attached to a `Mol`.

**Ring/aromaticity perception (revised — corrects a false premise in draft v1; closes adversarial Finding 1).** Draft v1 asserted aromaticity/ring data "does not exist on `Molecule` today" and used that to justify reimplementing SSSR + aromaticity natively in Rust. That premise is **false**: `Molecule` already carries a `bond_aromatic` field (populated by `from_mol2`) and a working `_to_rdkit()` method that performs ring/aromaticity perception via `Chem.SanitizeMol`, with hard-won handling of real edge cases (implicit-H valence inflation on carboxylates/phosphates, Kekulé-alternation requirements) as recently as 260821/260822. `rdkit` is already a real optional-extra dependency exercised by this exact path.

**Decision (v1 default):** reuse this existing path instead of reimplementing SSSR/aromaticity in Rust. The Python wrapper (§4) calls `Molecule._to_rdkit()`'s sanitization to obtain per-bond aromaticity and SSSR ring membership, converts them to plain arrays, and passes them **as inputs** into `canonicalize_ligand_topology` — the same "accept as inputs" pattern `assign_gaff_atom_types` already uses for exactly these two fields (`crates/proxide_py/src/py_chemistry.rs:99-126`). This is a real scope reduction versus draft v1: it reuses already-hardened edge-case handling instead of re-deriving it, and it removes the largest single engineering commitment in the original spec. Native Rust SSSR/aromaticity remains a plausible *later* optimization for Rust-side callers that don't go through Python, but is explicitly deferred, not committed to for v1 (see §5).

This does not change the ordering claim from draft v1 — canonicalization (now: RDKit-derived rings/aromaticity → canonical ranking → gaff2 typing) still must happen before typing, since `assign_gaff_atom_types` requires `is_aromatic`/`rings` as inputs and canonical ranking itself uses ring membership/aromaticity as invariants. What changed is *who* computes rings/aromaticity (the existing Python/RDKit layer, not a new Rust implementation), not *when* in the pipeline.

**New v1 non-goals (added this revision, closing Findings 9 and 14):**
- **Multi-fragment / disconnected input.** v1 requires a single connected molecular graph. Counter-ions, cofactors, and salts co-resident in one MOL2/SDF record must be split into separate per-fragment calls by the caller before invoking this API; `canonicalize_ligand_topology` validates connectivity and errors clearly rather than silently canonicalizing a disconnected graph (see §4).
- **Bond-order-less input (e.g., PDB `CONECT` records).** v1 requires bond order as an explicit input (from MOL2/SDF or an equivalent typed source). Geometry-based bond-order perception for `CONECT`-only ligands is out of scope; that would need a separate upstream step (e.g., RDKit's `DetermineBonds` or a geometry heuristic) feeding this API, left for future work.

---

## 2. Output shape

Two artifacts, split exactly along the frame-invariant / per-frame line from §1.

### 2a. `LigandTopology` — computed once per unique ligand (+ reference geometry; see §3 caching fix)

| Field | Shape | Notes |
|---|---|---|
| `ligand_id` | str | **new** — caller-supplied identity (see §4); closes adversarial Finding 13, which noted the naming convention `"{ligand_id}:torsion_{i}"` referenced an undefined field |
| `canonical_order` | `(n_atoms,)` int | permutation: input MOL2/SDF index → canonical index |
| `elements` | `(n_atoms,)` str | canonical order |
| `atom_names` | `(n_atoms,)` str | canonical order; preserved for round-trip to file formats |
| `gaff2_types` | `(n_atoms,)` str | from `assign_gaff_atom_types`, re-ordered into canonical order |
| `formal_charges` | `(n_atoms,)` int8 | input to gaff2 typing; carried through |
| `partial_charges` | `(n_atoms,)` float64 | Espaloma AM1-BCC-quality; see §3 for the revised caching contract |
| `aromaticity` | `(n_atoms,)` bool | derived via RDKit sanitization (§1), passed in as an input, not computed here |
| `ring_membership` | `list[list[int]]` (canonical indices) | SSSR rings, derived via RDKit sanitization (§1) |
| `bonds` | `(n_bonds, 2)` int + `bond_order (n_bonds,)` uint8 + `is_aromatic (n_bonds,)` bool + `restricted_rotation (n_bonds,)` bool | canonical-index pairs; `restricted_rotation` is **new**, see §2's torsion rule below |
| `torsion_definitions` | `(n_torsions, 4)` int | canonical atom-index quadruples, one per rotatable bond (definition below, now fully deterministic) |
| `pucker_definitions` | `list[RingPucker]` | one per ring sized 5–8; see §2c |
| `unrepresented_ring_dof` | `list[list[int]]` (canonical indices) | **new** — rings ≥9 atoms, explicitly flagged as having no extracted conformational descriptor in v1; see §2c |

**Rotatable-bond rule for `torsion_definitions` (v1, revised — closes Findings 6 and 7):**

A bond is a candidate if it is (a) acyclic, (b) neither endpoint has heavy-atom degree < 2 (i.e., each endpoint has at least one heavy-atom substituent besides the bond partner — this excludes both true terminal atoms *and* terminal groups like a freely-rotating methyl, whose only available 4th atom would be a symmetric H and whose rotation standard rotatable-bond definitions also exclude), and (c) it is not flagged `restricted_rotation`.

`restricted_rotation` is computed during the RDKit-derived aromaticity/ring pass (§1) using a bounded, explicit pattern list — not a full conjugation/resonance-detection engine: amide (C(=O)–N), ester/thioester (C(=O)–O/S), and the sp2–sp2 "twist bond" case where both endpoints are aromatic/conjugated-sp2 but the bond itself is not marked aromatic (biaryl, enamine, guanidine-type linkages). This directly targets the failure mode the adversarial review named — an amide bond formally encoded as bond-order 1 would otherwise emit a mechanically near-flat "torsion" channel into MIST/VMM fitting for the majority of drug-like ligands. The pattern list is intentionally bounded for v1 (catches the common, high-impact cases); broader conjugation detection is left open (§5).

**Branch-point substituent selection (new — closes Finding 7):** for a candidate bond `i–j`, the 4th atom on each side is the non-partner heavy-atom neighbor with the *highest canonical rank* (using the tie-break rule from §1) among that endpoint's heavy-atom neighbors. This is deterministic and reproducible for a fixed canonical order, closing the previously-unspecified ambiguity at branch points.

Ring-internal bonds are still handled via pucker (§2c), not as independent dihedrals, avoiding double-counting a ring's internal DOF two ways.

### 2b. `LigandFrameCoordinates` — per-frame, one instance per trajectory

| Field | Shape | Notes |
|---|---|---|
| `positions` | `(n_frames, n_atoms, 3)` float64 | canonical order; **axis order fixed this revision** — see note below |
| `torsions` | `(n_torsions, n_frames)` float64, radians | the direct ligand-analog of phi/psi |
| `feature_mask` | `(n_torsions,)` bool | valid-torsion mask, mirrors demistify's `(N_res, D)` mask convention — static, per-entity |
| `frame_validity` | `(n_frames,)` bool | **new** — see note below; per-frame, not per-entity |
| `pucker_phase` | `(n_rings, n_frames)` float64, radians | circular, VMM-fittable; phase for a near-planar ring (amplitude below a documented epsilon) is defined as `0` by convention, not `NaN` — see §2c |
| `bond_lengths` | `(n_bonds, n_frames)` float64 | extracted, **not yet VMM-fittable** (see below; unchanged from draft) |
| `bond_angles` | `(n_angles, n_frames)` float64 | extracted, **not yet VMM-fittable** (see below; unchanged from draft) |

**`positions` axis order — fixed (closes the defender's own verified self-contradiction).** Draft v1's §2b table said `(n_atoms, n_frames, 3)` while the Rust struct and function signature both said `(n_frames, n_atoms, 3)`. This revision keeps frame-major — it matches both the actual struct/signature and standard MD-trajectory-reader convention — and fixes the table to match. This makes `positions` the one field that is *not* entity-first, unlike `torsions`/`pucker_phase`/`bond_lengths`/`bond_angles`, which are entity-first to match demistify's `(N_res, N_frames, D)` convention; the intentional exception is now stated explicitly rather than left as an unflagged inconsistency. `extract_ligand_frame_coordinates`'s Python-side caller is responsible for any transpose needed before data reaches `run_demistify_pipeline`.

**`frame_validity` — new field, addresses a real gap the defender independently verified in `pipeline.py`.** `run_demistify_pipeline` hard-fails (`jnp.all(jnp.isfinite(angles))`) on any NaN/Inf anywhere in the array, and the only masking granularity that currently exists (`feature_mask: (N_res, D)`) is static per-entity, not per-frame. Ligand-derived quantities have real per-frame degeneracies that phi/psi mostly doesn't: a Cremer-Pople pucker phase is ill-conditioned as ring amplitude → 0 (transient planarity — handled above by defining phase as 0, not NaN), and a torsion is ill-conditioned when its four defining atoms go near-collinear (genuinely undefined, not just numerically noisy).

`extract_ligand_frame_coordinates` therefore never emits NaN for the *degenerate-but-physically-defined* case (pucker-phase-at-zero-amplitude uses the `0`-by-convention rule above); for the *genuinely undefined* case (e.g., missing/NaN input coordinates for a frame, or near-collinear torsion atoms below a documented angle epsilon), the affected frame's raw values may still be NaN, but `frame_validity[frame]` is set `False` unambiguously so a consumer can identify and exclude it.

**This is explicitly flagged as a blocking dependency, not a completed fix:** demistify's pipeline has no frame-level masking consumption path today. Shipping `frame_validity` from proxide is necessary but not sufficient — a ligand trajectory containing any invalid frame still cannot run end-to-end through `run_demistify_pipeline` until a corresponding pipeline-side change lands. This is listed as an open item in §5, not silently absorbed into the "zero pipeline-side changes" claim below (which is now scoped precisely to avoid that conflation).

**Packing decision for `torsions` into demistify's `(N_res, N_frames, D)` contract — language scoped this revision (closes Finding 12's confidence-inconsistency, without softening the actual decision):** v1 default is **each torsion as its own residue-analog entity with `D=1`**, rather than packing all of one ligand's torsions together as one high-D entity. This lets MIST discover torsion-torsion correlations at the graph level (its actual job) instead of baking an assumed grouping into the VMM fit. A ligand with `n_torsions` rotatable bonds contributes `n_torsions` rows to the pipeline's residue axis, each `(N_frames, 1)`, `names` set to `"{ligand_id}:torsion_{i}"`.

Two separate claims were previously conflated and are now split:
- **Verified fact, kept as-is:** `run_demistify_pipeline`'s current code (`pipeline.py:42-58,108-123`) accepts this packing shape — `angles.ndim==3`, `D=angles.shape[2]>=1`, `feature_mask.shape==(n_res,D)` — with no code changes, for the D-genericity path specifically.
- **Still-open architectural choice, not settled by the above:** whether `D=1`-per-torsion is the right grouping versus packing correlated torsions (e.g., sharing a ring or bond cluster) into shared-`D` entities closer to how phi/psi are jointly fit today. This remains §5 item 1, unchanged in substance, now stated without implying the code-compatibility fact settles the design question.

**Why bond lengths/angles are extracted but marked unconsumable (unchanged; defended successfully, reasoning kept in spec text):** denxity's `vmm_factorized.py` is hard-circular (atan2 mean, Mardia-resultant concentration) and there is no existing linear/Gaussian factorized-per-dimension family to fit bond lengths or bond angles (0–π, non-periodic) against — `gmm.py` fits a joint D-vector, not independent per-dimension marginals, and no mixed circular+linear family exists anywhere in denxity. Rather than block the whole spec on that gap, proxide exposes the raw geometry now — cheap, since the primitives `bond_angle`/`dihedral_angle` already exist in `crates/proxide-geometry/src/geometry/angles.rs:15,42,148` — so a future denxity linear-marginal family has a ready consumer with no new proxide work required. *Why this stands:* the adversarial review's implicit objection ("don't extract data nothing consumes yet") doesn't hold because the extraction cost is near-zero and the alternative (adding it later) would require a second proxide change; this is flagged as confirmable/reversible in §5, not a silent decision.

### 2c. `RingPucker` — defined this revision (closes Finding 4)

Draft v1 declared `pucker_definitions: Vec<RingPucker>` without ever defining the type — a real block on implementation, and inconsistent with §4's "resolved, not left open" framing for the ring/aromaticity section it lived under.

```rust
pub struct RingPucker {
    pub ring_atoms: Vec<usize>,   // canonical indices; fixed traversal order, see below
    pub ring_size: usize,
}
```

**Traversal-order convention (new — Cremer-Pople requires a fixed numbering around the ring; without one, phase values from two runs can be offset or mirrored):** `ring_atoms` is ordered by starting at the ring atom with the lowest canonical index, then walking the ring in the direction whose second atom has the lower canonical index of the two possible next-atoms. This is a fixed, deterministic rule derived entirely from canonical order — independent of input file order and independent of 3D geometry, matching the rest of the canonicalization contract.

**Ring-size coverage — the §2a/§2b contradiction is resolved with an explicit boundary (closes Finding 2):**
- **Rings of size 5–8:** the dominant (lowest-order, `m=2`) Cremer-Pople phase angle is extracted, via the standard generalized Cremer-Pople formalism. This discards higher-order pucker modes for 7–8-membered rings, consistent with — not a new gap versus — the existing draft-v1 choice to discard the `q3` chair/boat amplitude mode for 6-membered rings. Amplitude remains out of scope for all ring sizes in v1 (§5 item 2, unchanged, blocked on a denxity linear-marginal family).
- **Rings of size ≥9 (macrocycles):** v1 does **not** extract any conformational descriptor. Instead of vanishing silently — the adversarial review's core objection — these rings are enumerated in the new `unrepresented_ring_dof` field on `LigandTopology` (§2a), so a caller can detect the gap programmatically (e.g., warn or skip a macrocyclic ligand) rather than discover it only by noticing missing entropy contribution downstream. Whether this is acceptable for idea-002's actual ligand set is an open sign-off item (§5).

**Fused/bridged/spiro rings:** SSSR is not globally unique for such systems, but a fixed, deterministic algorithm (the RDKit-derived SSSR from §1, run deterministically on a fixed input) produces the same ring set every time on the same graph — sufficient for reproducibility even though it isn't "the" canonical minimum cycle basis in an abstract sense. Where a fusion bond is shared between two rings, `RingPucker` is defined independently per SSSR ring; shared atoms simply appear in more than one `RingPucker.ring_atoms` list. This is stated explicitly rather than left unaddressed.

---

## 3. Partial charges: decision — **wrap the existing native Espaloma pipeline, do not build or add a new dependency**

**Decision (unchanged; defended successfully — this section is the spec's strongest, per the adversarial review, and remains as-is):** `partial_charges` in `LigandTopology` is populated by calling proxide's **existing** native Rust Espaloma-Charge inference (`crates/proxide-core/src/chem/inference.rs:1-33`, exposed as `assign_espaloma_charges` / `assign_espaloma_charges_from_proxide_molecule`). No new charge-generation code, no `antechamber`/`sqm` subprocess wrapper, no RESP.

**Justification (kept as-is):**
- AM1-BCC is the documented sufficient choice for GAFF2-paired charging; RESP's added cost is only justified for new-FF-parameterization or unusual-functional-group cases, neither of which applies here.
- Recon confirms proxide has already done the "wrap, don't build" work the research recommends, better: a native Rust port of EspalomaCharge (ML surrogate validated as AM1-BCC-quality, Wang et al. 2024 JPCA) with embedded weights, exposed to Python at two layers (`py_chemistry.rs:223`, `src/proxide/chem/partial_charges.py`).
- **Known trap, avoided:** `parameterize_molecule()` (`crates/proxide-physics/src/physics/md_params.rs:613`) silently zero-charges by default (stale comment, predates the Espaloma work). The reference-frame builder calls `assign_espaloma_charges*` directly and **must not** route through `parameterize_molecule`.

**Frame variance of charges (kept as-is, one decision revised below):** charges are computed **once, from a single representative frame** (frame 0 by default, or a caller-supplied reference conformer/index), not per-frame — standard MD force-field convention, and it sidesteps a real caveat: ML charge models including Espaloma are conformer-sensitive, so re-running per frame would inject noise that looks like signal. *Why this stands:* the adversarial review's implicit alternative (OpenFF ELF10-style conformer-averaging) is a legitimate design the spec already anticipates and explicitly punts to sign-off (§5) with a stated reason, rather than deciding silently by omission.

**Reference-geometry validity gate before inference — new, closes Finding 11.** `canonicalize_ligand_topology` validates `ref_positions` before invoking Espaloma inference: (1) all coordinates finite; (2) no non-bonded atom pair closer than 0.7× the sum of covalent radii (standard structure-validation clash heuristic); (3) every declared bond length within 0.5–2.5× a per-element-pair reference covalent bond length. Failing any check returns `LigandFrameError::InvalidReferenceGeometry` instead of silently producing charges from an unequilibrated or corrupted structure that then propagate through an entire trajectory's downstream analysis.

**Caching/fingerprint key — revised, closes a real correctness bug (Finding 5).** Draft v1 placed `partial_charges` inside `LigandTopology` (described in §1 as "computed once per unique molecular graph") while §3 made charges depend on a caller-chosen reference frame, and §5 item 7 proposed caching `LigandTopology` keyed by a graph-only topology fingerprint. Combined, this meant two callers building "the same" ligand from different reference frames — or the same molecular graph across two campaigns with different starting structures — would collide on one cache key while legitimately producing different `partial_charges`; whichever call populated the cache first would silently win for every later caller.

**Decision:** the cache/fingerprint key is `(graph_fingerprint, ref_frame_geometry_hash, espaloma_weights_version)`, not graph identity alone:
- `graph_fingerprint`: a topology-fingerprint mechanism analogous to demistify's existing `TopologyFingerprint` (`replicas.py`), as originally proposed.
- `ref_frame_geometry_hash`: a hash of `ref_positions`, rounded to 3 decimal Å to tolerate float noise — new this revision.
- `espaloma_weights_version`: a version/hash token for the embedded `espaloma_v0_0_8.bin` weights — new this revision, closes Finding 15's cache-invalidation gap: if proxide updates the embedded weights, any previously-cached entry keyed on the old token simply misses rather than silently serving stale charges.

This is a deliberate correctness-over-cache-hit-rate tradeoff, stated explicitly: a graph-only cache would silently serve wrong charges across campaigns with different reference frames, which is strictly worse than an occasional cache miss recomputing a native, fast Espaloma inference call. §5 item 7 is now resolved by this decision rather than left open; the residual open item is only crate/feature-naming convention (§5).

---

## 4. API surface

**New crate:** `crates/proxide-ligand-frame`, sitting beside `proxide-gaff2` rather than inside it — `proxide-gaff2`'s own docstring (`lib.rs:6-13`) explicitly scopes it to atom typing only, so reference-frame assembly (canonicalization, torsion/pucker definition, charge wiring) does not belong there. (Ring/aromaticity perception is no longer part of this crate's scope — see §1.)

```rust
// crates/proxide-ligand-frame/src/lib.rs

pub struct LigandTopology {
    pub ligand_id: String,
    pub canonical_order: Vec<usize>,
    pub elements: Vec<String>,
    pub atom_names: Vec<String>,
    pub gaff2_types: Vec<String>,
    pub formal_charges: Vec<i8>,
    pub partial_charges: Vec<f64>,
    pub aromaticity: Vec<bool>,
    pub ring_membership: Vec<Vec<usize>>,
    pub bonds: Vec<(usize, usize, u8, bool, bool)>,  // (i, j, order, is_aromatic, restricted_rotation), canonical idx
    pub torsion_definitions: Vec<[usize; 4]>,
    pub pucker_definitions: Vec<RingPucker>,
    pub unrepresented_ring_dof: Vec<Vec<usize>>,      // rings >= 9 atoms, canonical idx
}

pub struct RingPucker {
    pub ring_atoms: Vec<usize>,
    pub ring_size: usize,
}

pub struct LigandFrameCoordinates {
    pub positions: Vec<[[f64; 3]; /* n_atoms */]>,    // (n_frames, n_atoms, 3)
    pub torsions: Vec<Vec<f64>>,                       // (n_torsions, n_frames)
    pub feature_mask: Vec<bool>,
    pub frame_validity: Vec<bool>,                     // (n_frames,)
    pub pucker_phase: Vec<Vec<f64>>,                   // (n_rings, n_frames)
    pub bond_lengths: Vec<Vec<f64>>,
    pub bond_angles: Vec<Vec<f64>>,
}

pub enum LigandFrameError {
    DisconnectedGraph { component_count: usize },
    UnsupportedElement { element: String },
    InvalidValence { atom_index: usize },
    SssrInputInvalid { reason: String },              // rings/aromaticity inputs inconsistent with bonds_in
    ChargeInferenceFailure { reason: String },
    InvalidReferenceGeometry { reason: String },
    TopologyPositionMismatch { expected_atoms: usize, got_atoms: usize },
}

/// Builds the frame-invariant topology: canonical ordering, gaff2 typing
/// (calls into proxide-gaff2), Espaloma charges from `ref_positions`.
/// Rings/aromaticity are supplied by the caller (derived from RDKit
/// sanitization in the Python layer — see §1) rather than computed here.
/// Validates single-connectedness (§1 non-goal) and reference-geometry
/// sanity (§3) before charge inference.
pub fn canonicalize_ligand_topology(
    ligand_id: &str,
    elements: &[String],
    bonds_in: &[(usize, usize, u8)],   // (i, j, bond_order), input-file atom indices
    bond_is_aromatic: &[bool],          // parallel to bonds_in
    rings: &[Vec<usize>],               // SSSR ring atom lists, input-file indices
    formal_charges: Option<&[i8]>,
    ref_positions: &[[f64; 3]],
) -> Result<LigandTopology, LigandFrameError>;

/// Per-frame extraction, indexed by an already-built LigandTopology.
/// Validates the positions/topology atom-identity contract via
/// `input_elements` before processing (§4 note below).
pub fn extract_ligand_frame_coordinates(
    topology: &LigandTopology,
    positions: &[[[f64; 3]; /* n_atoms */]],   // (n_frames, n_atoms, 3), input order
    input_elements: &[String],                  // input-file atom order, for the contract check
) -> Result<LigandFrameCoordinates, LigandFrameError>;
```

**Index-alignment contract (new — closes Finding 15a):** all parallel per-entity arrays are aligned strictly by position: `torsion_definitions[i]` ↔ `torsions[i]`, `pucker_definitions[i]` ↔ `pucker_phase[i]`, and the `bonds` list ↔ `bond_lengths`/`bond_angles`. This was previously assumed by construction order only; it is now asserted by internal tests, not left implicit.

**Topology↔positions contract check (new — closes Finding 8).** `extract_ligand_frame_coordinates` validates, before processing: (1) `positions`' atom axis length matches `topology.canonical_order.len()`; (2) `input_elements`, mapped through the inverse of `topology.canonical_order`, is element-wise consistent with `topology.elements`. A trajectory loaded via a different topology source (e.g., XTC/DCD + a separately-ordered topology file) than the one `LigandTopology` was built from will fail this check with `TopologyPositionMismatch` rather than silently misapplying the canonical permutation.

**Connected-component validation (new — closes Finding 9):** `canonicalize_ligand_topology` runs a union-find over `bonds_in` and returns `DisconnectedGraph` if more than one component is found. Splitting multi-fragment MOL2/SDF records (counter-ions, cofactors, salts) into single-fragment calls is the caller's responsibility (§1 non-goal).

**Feature gating:** new feature `ligand-frame`, depending on `gaff2-engine` (needs `assign_gaff2_atom_types`) and unconditionally on the Espaloma module (already core/unfeatured per recon). `canonicalize_ligand_topology` calls `proxide_gaff2::assign_gaff2_atom_types` internally after receiving aromaticity/rings as inputs — `proxide-gaff2`'s scope is unchanged, this crate is a caller of it, not an extension of it. *(Kept as-is from draft v1 — verified against existing cfg-gating precedent, not re-litigated by the review.)*

**PyO3 exposure:** `crates/proxide_py/src/py_chemistry.rs` gains `canonicalize_ligand_topology` / `extract_ligand_frame_coordinates` pyfunctions, gated the same way `assign_gaff_atom_types` is (`#[cfg(feature = "gaff2-engine")]`, plus `ligand-frame`).

**Python wrapper:** `src/proxide/chem/reference_frame.py` (new file, alongside `partial_charges.py`), with `LigandTopology`/`LigandFrameCoordinates` as dataclasses mirroring `Molecule` and `MolecularTopology`, plus one entry point:

```python
def build_ligand_reference_frame(
    molecule: Molecule,
    ligand_id: str,                      # new, required — see §2a
    trajectory_positions: np.ndarray,    # (n_frames, n_atoms, 3), input atom order
    ref_frame_index: int = 0,
) -> tuple[LigandTopology, LigandFrameCoordinates]:
    # 1. molecule._to_rdkit() -> SSSR rings + per-bond aromaticity (§1)
    # 2. canonicalize_ligand_topology(...) with those as inputs
    # 3. extract_ligand_frame_coordinates(...)
    ...
```

This is proxide's ligand-analog of what `ReplicaSet` loading does for proteins today (`replicas.py`, producing `(angles, feature_mask, names)`) — demistify's future ligand pipeline calls this, then flattens `LigandFrameCoordinates.torsions`/`pucker_phase` + `feature_mask` (and, once the pipeline-side follow-up in §5 lands, `frame_validity`) into the `(N_res, N_frames, D)` / `(N_res, D)` / `names` triple `run_demistify_pipeline` already accepts, per the §2b packing decision.

---

## 5. Open questions / explicit assumptions requiring human sign-off

Items resolved in this revision (ring/aromaticity source, `RingPucker` definition, ring-size boundary, tie-breaking, branch-point selection, amide exclusion, error typing, caching key, connectivity/bond-order non-goals, `ligand_id`, topology↔positions contract) are documented with their rationale in the relevant sections above and are **not** repeated here as open items — per review guidance, a resolved gap gets a firm decision in the body, not a TBD in this list. What remains genuinely open:

1. **Torsion packing (§2b):** `D=1`-per-torsion is the v1 default so MIST can find torsion-torsion correlation rather than it being baked into a joint VMM fit. Alternative: group correlated torsions (shared ring/bond cluster) into shared-`D` entities, closer to how phi/psi are jointly fit today. Needs demistify-side confirmation — this changes what MIST's ligand-side graph looks like. (Unchanged from draft v1; the earlier "zero pipeline-side changes" language no longer implies this choice is settled — see §2b.)
2. **Ring-pucker amplitude is punted entirely for v1** (§2c) — only phase is extracted, because no linear/Gaussian marginal family exists in denxity yet to fit amplitude against. Blocks on new denxity work (a `gmm.py`-based per-dimension factorized family), not on proxide.
3. **Bond lengths/angles are extracted but explicitly not VMM-fittable in v1**, for the same denxity-gap reason. Confirm idea-002 actually wants this stretch data emitted now versus dropping it from v1 output and adding later.
4. **Per-frame validity is a genuine blocking dependency, not a completed fix (new this revision, §2b).** `frame_validity` ships from proxide, but `run_demistify_pipeline` has no frame-level masking consumption path today (`feature_mask` is static per-entity). A ligand trajectory with any invalid frame cannot run end-to-end until a corresponding pipeline-side change lands. Needs prioritization/timeline sign-off, since it affects whether this spec's output is actually usable on real flexible-ligand trajectories at v1 ship time.
5. **Macrocycle coverage (new this revision, §2c):** rings ≥9 atoms get no conformational descriptor in v1 — only a flag (`unrepresented_ring_dof`) that the gap exists. Confirm this is acceptable for idea-002's actual ligand set; some macrocyclic drug-like ligands (macrolides, macrocyclic kinase inhibitors) may need their dominant DOF represented sooner than "later work."
6. **Restricted-rotation pattern list scope (new this revision, §2):** the amide/ester/thioamide/biaryl-twist list is a bounded v1 heuristic, not a full conjugation/resonance-detection engine. Confirm this coverage is sufficient, or whether a fuller approach is needed for idea-002's ligand set (e.g., guanidinium, extended push-pull conjugation).
7. **Canonical-ranking algorithm implementation** — recommended: O'Boyle & Sayle over classic Morgan, for documented robustness on symmetric graphs (tie-breaking now specified, §1); exact implementation remains a coding-time detail, not a design gap.
8. **Crate/feature naming** (`proxide-ligand-frame`, `ligand-frame` feature) is proposed, not checked against any existing proxide crate-naming convention document.
9. **Partial-charge conformer-averaging (§3):** static per-topology-and-reference-frame charges are the v1 default (standard FF convention); confirm v1 doesn't need OpenFF-ELF10-style multi-conformer averaging later.
