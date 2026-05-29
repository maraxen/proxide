---
name: 260529_confind_spec
description: Spec for proxide-confind crate — Rust/rayon reimplementation of ConFind contact-degree algorithm (rev 3 — rotlib extracted to proxide-rotlib, oracle-verified)
metadata:
  type: reference
---

# Spec: `proxide-confind` Crate (Revision 3)

**Task ID**: `260529_confind`
**Source reference**: `Grigoryanlab/Mosaist@450816a` — see `260529_confind_model.md` for full algorithmic derivation.
**Oracle verdict on rev 2**: PASS (rev 2 addressed all blocking gaps from rev 1).
**Rev 3 change**: `RotamerLibrary`, `Frame`/`Transform`, and `RotamerId` extracted to `proxide-rotlib` (separate crate). All other algorithmic content unchanged from rev 2.

---

## 0. Prerequisites

### `proxide-rotlib` (Task 260529_confind_rotlib)

`RotamerLibrary` is the single largest blocking dependency. It is now specified as its own crate — see `260529_rotlib.md`. Implement and publish `proxide-rotlib` before this crate can compile.

Types imported from `proxide-rotlib` in this crate:
- `RotamerId` — (aa, bin_index, rot_index), `Ord` by `(aa, bin_index, rot_index)`
- `RotamerLibrary` — load / backbone_bin / num_rotamers / rotamer_probability_by_id / place_rotamer
- `PlacedRotamer`, `PlacedAtom`
- `counts_as_sidechain(atom_name, aa) -> bool` — CB excluded for non-ALA; hydrogens excluded
- `RotlibError`

No rotamer or Frame/Transform code lives in `proxide-confind`.

---

## 1. Purpose

A Rust crate that reimplements the ConFind contact-degree algorithm with rayon-based parallelism. Produces output compatible with the Mosaist `testConFind` CLI (same tab-delimited format, same quantity semantics, relaxed numerical tolerance — see §9).

---

## 2. Crate Identity

```toml
# crates/proxide-confind/Cargo.toml
[package]
name = "proxide-confind"
version.workspace = true
edition.workspace = true
authors.workspace = true
license.workspace = true

[dependencies]
proxide-core     = { path = "../proxide-core" }
proxide-geometry = { path = "../proxide-geometry" }
proxide-io       = { path = "../proxide-io" }
proxide-rotlib   = { path = "../proxide-rotlib" }
rayon            = { workspace = true }
dashmap          = "6"
thiserror        = { workspace = true }
log              = { workspace = true }

[dependencies.serde]
workspace = true
optional  = true
features  = ["derive"]

[features]
default = []
serde   = ["dep:serde"]
```

Add to workspace `Cargo.toml` members list.

---

## 3. Algorithmic Constants

```rust
pub const DCUT:         f64 = 25.0;   // CA–CA neighbor cutoff (Å)
pub const CLASH_DIST:   f64 = 2.0;    // SC–BB clash distance (Å)
pub const CONT_DIST:    f64 = 3.0;    // SC–SC contact distance (Å)
pub const LO_COLL_PROB: f64 = 0.5;   // freedom type 2/3 low threshold
pub const HI_COLL_PROB: f64 = 2.0;   // freedom type 2/3 high threshold

/// 18 amino acids placed as rotamers (excludes GLY and PRO).
pub const AA_NAMES: [&str; 18] = [
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU",
    "HIS","ILE","LEU","LYS","MET","PHE","SER",
    "THR","TRP","TYR","VAL",
];

/// Background propensities in percent (must match Mosaist exactly).
/// GLY (7.11) and PRO (4.52) are in MSL's aaProp table but are not placed
/// as rotamers — they appear here for completeness and interference normalization.
pub fn aa_propensity(aa: &str) -> f64 {
    match aa {
        "ALA" => 7.73, "ARG" => 5.03, "ASN" => 4.50, "ASP" => 5.82,
        "CYS" => 1.84, "GLN" => 3.94, "GLU" => 6.61, "GLY" => 7.11,
        "HIS" | "HSD" => 2.35, "ILE" => 5.66, "LEU" => 8.83,
        "LYS" => 6.27, "MET" => 2.08, "PHE" => 4.05, "PRO" => 4.52,
        "SER" => 6.13, "THR" => 5.53, "TRP" => 1.51, "TYR" => 3.54,
        "VAL" => 6.91,
        _ => panic!("no propensity for {aa}"),
    }
}
```

---

## 4. Substrate Baseline (actual proxide types)

| Concept | Actual type | File |
|---|---|---|
| Structure (atoms) | `AtomicSystem` | `proxide-core/src/structure/systems.rs` |
| Processed structure (residues) | `ProcessedStructure` | `proxide-core/src/processing/residues.rs` |
| Per-residue metadata | `ResidueInfo { res_id, res_name, chain_id, insertion_code, start_atom, num_atoms }` | same |
| Residue identity | `ResidueId { chain_id: String, res_id: i32, insertion_code: char }` | same |
| PDB parser return | `(RawAtomData, Vec<usize>)` with `f32` coords | `proxide-io/src/formats/pdb.rs` |
| Spatial index | `CellList` (`f32`, single-cutoff, indices only) | `proxide-geometry/src/geometry/cell_list.rs` |
| Dihedrals | `dihedral_angle_f64`, `compute_backbone_dihedrals_f64` | `proxide-geometry/src/geometry/angles.rs` |
| Rotamer library | `proxide_rotlib::RotamerLibrary` | `crates/proxide-rotlib` |

All `AtomicSystem`/`ProcessedStructure` coordinates are `f32`. ConFind requires `f64` arithmetic throughout (see §5).

### 4a. f64 coordinate extraction (`src/coords.rs`)

ConFind reads coordinates from `ProcessedStructure` and immediately widens to `f64`. This is the **single f32→f64 boundary**; no `f32` appears inside `proxide-confind` past this point.

```rust
pub fn extract_f64_backbone(s: &ProcessedStructure) -> Result<ProteinBackbone, ConFindError>;

pub struct ProteinBackbone {
    pub bb:        Vec<ResidueBackbone>,   // one per protein residue
    pub ids:       Vec<ResidueId>,         // parallel, for output
    pub chain_map: Vec<usize>,            // residue_index → chain_index
}

pub struct ResidueBackbone {
    pub res_name: String,
    pub n:   Option<[f64; 3]>,
    pub ca:  Option<[f64; 3]>,
    pub c:   Option<[f64; 3]>,
    pub o:   Option<[f64; 3]>,
    pub phi: f64,   // degrees; 9999.0 if terminal or missing
    pub psi: f64,
}
```

**phi/psi post-processing** (two mandatory steps):
1. **Radians → degrees**: `compute_backbone_dihedrals_f64` returns `atan2` output in radians. Multiply by `180.0 / std::f64::consts::PI`.
2. **Per-chain segmentation**: call `compute_backbone_dihedrals_f64` per chain segment (using `chain_map`). First residue of each chain → `phi = 9999.0`. Last residue → `psi = 9999.0`. Matches Mosaist `Residue::getPhi(false)`/`getPsi(false)` at chain termini.
3. **None → sentinel**: `dihedrals.phi.map(|r| r.to_degrees()).unwrap_or(9999.0)`.

### 4b. Residue indexing

`ResidueIndex(u32)` — flat dense index into `ProteinBackbone::bb`, 0-based, protein residues only.

- **Output identity**: `ProteinBackbone::ids[idx]`.
- **Flanking check**: two residues are same-chain adjacent iff `chain_map[ri] == chain_map[rj]` **and** `|ri - rj| <= ignore_flanking`. `ProcessedStructure::from_raw` sorts within chain by `(res_id, insertion_code)`.

---

## 5. Coordinate Precision and Parity Contract

- All ConFind arithmetic uses **f64** throughout.
- **CLI path** (`load_pdb_f64` in `src/io.rs`): re-parses PDB ATOM columns 30–54 directly to `f64`, bypassing `RawAtomData.coords` (f32). This recovers full PDB text precision (3 decimal places).
- **Library path** (`extract_f64_backbone`): upcasts from `ProcessedStructure` f32 — acceptable for tests and Python bindings.
- **Parity tolerance**: agree with Mosaist within **1e-6** on same PDB + same rotamer library.
- **Accumulation order**: CD inner sum performed in sorted `RotamerId` order (lexicographic `(aa, bin_index, rot_index)` for both rotA and rotB) for reproducibility.
- **Known C++ bug replicated**: `contactDegree` line 209 checks `aaAllowedA.empty() && aaAllowedA.empty()` (checks A twice). Rust replicates: `no_aa_restriction = aa_allowed_a.is_none() && aa_allowed_a.is_none()`. Impacts only the deferred `getConstrainedContacts` path.

---

## 6. New Types in `proxide-confind`

### 6a. `ProximityGrid<T>` (`src/grid.rs`)

Generic 3D bucketed spatial index in f64. Replaces `CellList` (f32-only).

```rust
pub struct ProximityGrid<T: Clone> { /* cells, bbox, cell_size, dims */ }

impl<T: Clone + Send + Sync> ProximityGrid<T> {
    pub fn build(points: &[[f64; 3]], tags: Vec<T>, char_dist: f64) -> Self;
    pub fn points_within(&self, center: [f64; 3], dmin: f64, dmax: f64) -> Vec<T>;
    pub fn point_size(&self) -> usize;
    pub fn get_point(&self, i: usize) -> [f64; 3];
    pub fn get_tag(&self, i: usize) -> &T;
}
```

Backbone grid tags: `ResidueIndex`. Rotamer heavy-SC grid tags: `Arc<RotamerId>`.

### 6b. Normalization helpers (`src/cache.rs`)

```rust
/// Sum of aaProp[aa] * rotProb(rot) over surviving rotamers of `res` in `available_aa`.
pub fn weight_of_available_rotamers(
    cache: &ResidueCache,
    rotlib: &RotamerLibrary,
    available_aa: &HashSet<&str>,
) -> f64;

/// Sum of aaProp[aa] for each aa in `available_aa`, divided by 100.0.
pub fn weight_of_available_amino_acids(available_aa: &HashSet<&str>) -> f64;
```

Both return `0.0` gracefully when the input set is empty.

### 6c. `ResidueCache` (`src/cache.rs`)

Computed per-residue during Phase A; immutable thereafter.

```rust
pub struct ResidueCache {
    pub surviving_rotamers: Vec<Arc<RotamerId>>,
    /// per-aa: None if no surviving rotamers for that aa
    pub rotamer_grids:      HashMap<String, Option<ProximityGrid<Arc<RotamerId>>>>,
    pub fraction_pruned:    f64,
    pub n_library_rotamers: usize,
    /// interference[resB_idx][aa] = accumulated aaP * rotP / 100
    /// Populated during Phase A (see §9.1 for accumulation formula).
    pub interference:       HashMap<ResidueIndex, HashMap<String, f64>>,
    /// Indices into bb_atoms (global flat vec) of BB atoms that clash with
    /// any ALA rotamer of self. Used in the permanentContacts path.
    pub permanent_contacts: Vec<usize>,
}
```

**Interference accumulation** during Phase A cache (matching Mosaist `cache()` lines 158–160):
For each ALA rotamer atom that clashes (`bbNN.pointsWithin(atom, 0.0, clashDist)`):
- For the owning residue `resB` of the clashing backbone atom:
  - `interference[self][resB][aa] += aa_propensity(aa) * rotP / 100.0`
  - Accumulation uses the **actual AA** being placed (all 18), not just ALA — the ALA path also records permanent_contacts for `resB`.
  - Guard: accumulate only on **first encounter** of each `resB` per rotamer (avoid double-counting when multiple atoms of one rotamer clash the same residue's backbone — MSL `seen` set, model §3a).

### 6d. `ClashTuple` (`src/parallel.rs`)

```rust
pub struct ClashTuple {
    pub res_a:         ResidueIndex,
    pub rot_a:         Arc<RotamerId>,
    pub contrib_to_a:  f64,   // aaPropB * rotProbB
    pub res_b:         ResidueIndex,
    pub rot_b:         Arc<RotamerId>,
    pub contrib_to_b:  f64,   // aaPropA * rotProbA
}
```

`ClashTuple.res_a` is always the lower-index residue; `ClashTuple.res_b` is always the higher.

### 6e. `ContactList` (`src/contact_list.rs`)

```rust
pub struct ContactList {
    pub pairs:   Vec<(ResidueIndex, ResidueIndex)>,
    pub degrees: Vec<f64>,
}

impl ContactList {
    pub fn degree(&self, a: ResidueIndex, b: ResidueIndex) -> Option<f64>;
    pub fn ordered_pairs(&self) -> Vec<(ResidueIndex, ResidueIndex)>;
}
```

---

## 7. Core Type: `ConFind` (`src/confind.rs`)

```rust
pub struct ConFind {
    pub rotlib:   Arc<RotamerLibrary>,
    pub strict:   bool,
    pub backbone: Arc<ProteinBackbone>,

    bb_grid:  Arc<ProximityGrid<ResidueIndex>>,
    ca_grid:  Arc<ProximityGrid<ResidueIndex>>,
    bb_atoms: Arc<Vec<[f64; 3]>>,

    cache:     DashMap<ResidueIndex, Arc<ResidueCache>>,
    coll_prob: DashMap<ResidueIndex, HashMap<Arc<RotamerId>, f64>>,
    degrees:   DashMap<(ResidueIndex, ResidueIndex), f64>,
    freedom:   DashMap<ResidueIndex, f64>,
}

impl ConFind {
    pub fn new(rotlib: Arc<RotamerLibrary>, backbone: Arc<ProteinBackbone>, strict: bool) -> Self;
    pub fn n_residues(&self) -> usize { self.backbone.bb.len() }
}
```

---

## 8. Public API

```rust
impl ConFind {
    /// Phase A: cache one residue (idempotent, rayon-safe).
    pub fn cache_residue(&self, ri: ResidueIndex) -> Result<(), ConFindError>;

    /// Phase A: cache all residues in parallel.
    pub fn cache_all(&self) -> Result<(), ConFindError>;

    /// CA–CA neighbors of ri within DCUT.
    pub fn neighbors(&self, ri: ResidueIndex) -> Vec<ResidueIndex>;

    /// Compute CD for one pair; emit ClashTuples for Phase B2 collProb accumulation.
    pub fn contact_degree_with_clashes(
        &self,
        res_a: ResidueIndex,
        res_b: ResidueIndex,
        aa_allowed_a: Option<&[&str]>,
        aa_allowed_b: Option<&[&str]>,
    ) -> Result<(f64, Vec<ClashTuple>), ConFindError>;

    /// Full contact + freedom pipeline (Phases A–C).
    /// Populates self.coll_prob and self.freedom.
    /// v1 constraint: `residues` must be sorted ascending by index for correct
    /// collProb attribution (see §9.2).
    pub fn contacts(
        &self,
        residues: &[ResidueIndex],
        cdcut: f64,
    ) -> Result<ContactList, ConFindError>;

    /// Interference: pairs where EITHER endpoint is in `residues` and value >= incut.
    /// Directional (resA sidechain → resB backbone). Call after cache_all() or contacts().
    pub fn interference(
        &self,
        residues: &[ResidueIndex],
        incut: f64,
    ) -> Result<ContactList, ConFindError>;

    /// BB–BB minimum atom-pair distance, skipping same-chain adjacents.
    pub fn bb_interaction(
        &self,
        residues: &[ResidueIndex],
        dcut_bb: f64,
        ignore_flanking: usize,
    ) -> Result<ContactList, ConFindError>;

    pub fn crowdedness(&self, ri: ResidueIndex) -> Result<f64, ConFindError>;
    pub fn freedom(&self, ri: ResidueIndex) -> Result<f64, ConFindError>;
    pub fn residue_id(&self, ri: ResidueIndex) -> &ResidueId;
}
```

---

## 9. Parallelization Design

### 9.1 Phase A — Cache (`cache_all`, rayon `par_iter`)

```rust
pub fn cache_all(&self) -> Result<(), ConFindError> {
    (0..self.backbone.bb.len() as u32)
        .into_par_iter()
        .map(ResidueIndex)
        .try_for_each(|ri| self.cache_residue(ri))
}
```

`cache_residue` per residue:
1. Get `phi`/`psi` from `backbone.bb[ri]`.
2. For each aa in `AA_NAMES`: get `num_rotamers(aa, phi, psi)` from rotlib.
3. For each rotamer `ri` in `0..nr`:
   - Call `rotlib.place_rotamer(aa, phi, psi, ri, n, ca, c)` → `PlacedRotamer`.
   - For each `PlacedAtom` in `placed.atoms`: filter with `counts_as_sidechain(atom.name, aa)`.
   - For each surviving atom: query `bb_grid.points_within(atom.xyz, 0.0, CLASH_DIST)`.
     - For each hit `bb_atom_idx`: if owning residue ≠ `ri`:
       - Set `prune = true`.
       - If `aa == "ALA"`: add `bb_atom_idx` to `permanent_contacts`.
       - Accumulate interference (first-encounter guard per resB):  
         `interference[resB][aa] += aa_propensity(aa) * rotP / 100.0`
         where `rotP = rotlib.rotamer_probability_by_id(&placed.id)`.
       - For non-ALA: `break` after first clash atom (consistent with Mosaist).
   - If not pruned: add to `surviving_rotamers`; build per-aa `ProximityGrid`.
4. Compute `fraction_pruned = (tot - surviving) / tot`, `n_library_rotamers = tot`.
5. Write `Arc<ResidueCache>` into `self.cache`.

**Key invariant**: `coll_prob` keys (in Phase B) are a strict subset of `surviving_rotamers`.

### 9.2 Phase B — Contact Degree (parallel B1 + sequential B2)

**B1 — parallel pair enumeration:**

```rust
// Canonical pairs: ri < rj, avoiding double-counting.
let pairs: Vec<(ResidueIndex, ResidueIndex)> = residues
    .par_iter()
    .flat_map(|&ri| {
        self.neighbors(ri).into_iter()
            .filter(|&rj| rj > ri)
            .map(move |rj| (ri, rj))
    })
    .collect();

let results: Vec<(f64, Vec<ClashTuple>)> = pairs
    .par_iter()
    .map(|&(ri, rj)| self.contact_degree_with_clashes(ri, rj, None, None))
    .collect::<Result<_, _>>()?;

// Store symmetric degrees.
for (&(ri, rj), &(cd, _)) in pairs.iter().zip(&results) {
    self.degrees.insert((ri, rj), cd);
    self.degrees.insert((rj, ri), cd);
}
```

**CD normalization** (inside `contact_degree_with_clashes`):

```rust
let denom = weight_of_available_rotamers(&cache_a, &self.rotlib, &aa_set_a)
          * weight_of_available_rotamers(&cache_b, &self.rotlib, &aa_set_b);
let cd = if denom == 0.0 { 0.0 } else { cd_raw / denom };
```

**B2 — sequential collProb merge (asymmetric `ofInterest`):**

```rust
let of_interest: HashSet<ResidueIndex> = residues.iter().copied().collect();
let mut coll_prob: HashMap<ResidueIndex, HashMap<Arc<RotamerId>, f64>> = HashMap::new();

for (&(ri, rj), (_, tuples)) in pairs.iter().zip(&results) {
    for t in tuples {
        // ri always accumulates (ri is the lower-index, always in query set)
        *coll_prob.entry(t.res_a).or_default()
                  .entry(t.rot_a.clone()).or_insert(0.0) += t.contrib_to_a;
        // rj accumulates only if in query set
        if of_interest.contains(&t.res_b) {
            *coll_prob.entry(t.res_b).or_default()
                      .entry(t.rot_b.clone()).or_insert(0.0) += t.contrib_to_b;
        }
    }
}
for &ri in residues { coll_prob.entry(ri).or_default(); }
for (ri, map) in coll_prob { self.coll_prob.insert(ri, map); }
```

**v1 scope constraint**: `residues` must be sorted ascending by `ResidueIndex`. Document in `contacts()` docstring.

### 9.3 Phase C — Freedom (parallel per-residue, after B2)

```rust
residues.par_iter().try_for_each(|&ri| {
    let cp    = self.coll_prob.get(&ri).ok_or(ConFindError::NotCached(ri))?;
    let cache = self.cache.get(&ri).ok_or(ConFindError::NotCached(ri))?;
    let f = compute_freedom(&cp, &cache, LO_COLL_PROB, HI_COLL_PROB, 2);
    self.freedom.insert(ri, f);
    Ok(())
})?;
```

---

## 10. Freedom Formula

```rust
pub fn compute_freedom(
    coll_prob:    &HashMap<Arc<RotamerId>, f64>,
    cache:        &ResidueCache,
    lo_cut: f64, hi_cut: f64,
    freedom_type: u8,
) -> f64 {
    let n_surv = cache.surviving_rotamers.len();
    let n_lib  = cache.n_library_rotamers;
    let n_uncontested = n_surv.saturating_sub(coll_prob.len()) as f64;

    match freedom_type {
        1 => {
            let n = n_uncontested + coll_prob.values()
                .filter(|&&v| v / 100.0 < 0.5).count() as f64;
            n / n_lib as f64
        }
        2 => {
            let n1 = n_uncontested + coll_prob.values()
                .filter(|&&v| v / 100.0 < lo_cut).count() as f64;
            let n2 = n_uncontested + coll_prob.values()
                .filter(|&&v| v / 100.0 < hi_cut).count() as f64;
            ((n1*n1 + n2*n2) / 2.0).sqrt() / n_lib as f64
        }
        3 => {
            let n1 = (n_uncontested + coll_prob.values()
                .filter(|&&v| v / 100.0 < lo_cut).count() as f64) / n_lib as f64;
            let n2 = (n_uncontested + coll_prob.values()
                .filter(|&&v| v / 100.0 < hi_cut).count() as f64) / n_lib as f64;
            ((n2*n2 + n2*n1) / 2.0).sqrt()
        }
        _ => panic!("unknown freedom type {freedom_type}"),
    }
}
```

---

## 11. Parity Contract

- **Contact degree, freedom, crowdedness, interference**: within **1e-6** of Mosaist on same PDB + same rotamer library, using f64 coordinate path.
- **Boundary rotamers** (clash/contact at exactly 2.0 or 3.0 Å): acceptable divergence due to f64 accumulation-order differences.
- **Parity test**: run Mosaist `testConFind` binary and `confind` on ≥3 benchmark PDBs; compare with float-aware diff script.

---

## 12. CLI Binary: `confind`

```
crates/proxide-confind/src/bin/confind.rs
```

**v1 flags**:

```
confind --p <pdb> --rLib <path> [--o <out>] [--sel <chain:resrange>]
        [--cdcut <f64>] [--incut <f64>] [--threads <n>]
```

**Output format** (tab-delimited, `%.6f`):

```
contact       <chain>,<resnum>  <chain>,<resnum>  <cd>    <resA_name>  <resB_name>
crwdnes       <chain>,<resnum>  <crowdedness>     <res_name>
freedom       <chain>,<resnum>  <freedom>         <res_name>
interference  <chain>,<resnum>  <chain>,<resnum>  <value> <resA_name>  <resB_name>
SEQUENCE: ALA GLY ...
```

`<resnum>` = `<res_id><insertion_code>` where insertion_code is omitted when `' '`.

---

## 13. Error Type

```rust
#[derive(Debug, thiserror::Error)]
pub enum ConFindError {
    #[error("residue {0:?} not cached — call cache_residue first")]
    NotCached(ResidueIndex),
    #[error("residue {0:?} missing backbone atom {1}")]
    MissingBackbone(ResidueIndex, &'static str),
    #[error("freedom not computed for {0:?} — must be in contacts() query set")]
    FreedomNotComputed(ResidueIndex),
    #[error("rotamer library error: {0}")]
    RotlibError(#[from] proxide_rotlib::RotlibError),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}
```

---

## 14. Tests

```
crates/proxide-confind/tests/
  test_params.rs          — aa_propensity values, AA_NAMES set
  test_grid.rs            — ProximityGrid points_within vs brute-force (random, f64)
  test_f64_extract.rs     — phi/psi from extract_f64_backbone match known structures
  test_cache.rs           — fraction_pruned, n_library_rotamers on known PDB + rotlib
  test_contact_degree.rs  — CD values match reference on benchmark structures
  test_freedom.rs         — freedom / crowdedness match reference
  test_interference.rs    — interference values match reference
  test_parity.rs          — full output vs Mosaist binary (integration, requires rotlib)
  test_parallel.rs        — parallel == sequential output (determinism)
  test_flanking.rs        — bb_interaction ignores same-chain adjacents; cross-chain not ignored
```

---

## 15. File Layout

```
crates/proxide-confind/
  Cargo.toml
  src/
    lib.rs              — pub re-exports
    params.rs           — AA_NAMES, aa_propensity, algorithmic constants
    coords.rs           — extract_f64_backbone, ProteinBackbone, load_pdb_f64
    grid.rs             — ProximityGrid<T>
    cache.rs            — ResidueCache, ConFind::cache_residue,
                          weight_of_available_rotamers, weight_of_available_amino_acids
    confind.rs          — ConFind struct, contacts/freedom/interference
    contact_list.rs     — ContactList
    freedom.rs          — compute_freedom (types 1/2/3)
    parallel.rs         — ClashTuple, Phase B1/B2/C orchestration
    error.rs            — ConFindError
    bin/
      confind.rs        — CLI
  tests/
    (see §14)
  benches/
    bench_cache.rs
    bench_contacts.rs
```

No `rotlib.rs` or `transform.rs` — those live in `proxide-rotlib`.

---

## 16. Out of Scope (v1)

- Python bindings (v2 via `proxide_py`)
- WASM target
- Constrained contacts (`seq_const`, `--seq_const`)
- Rotamer output log (`--rout`), verbose (`--verb`), B-factor freedom (`--freeB`)
- `--pp` / `--omg` phi/psi/omega printing
- Non-standard residue support beyond `AA_NAMES`
- `getContactingResidues` single-residue convenience wrapper
