---
task_id: 260602_master-rmsd-search
created: 260602
status: ready-for-impl
---

# Spec: proxide-frag — Backbone RMSD Substructure Search

Pure-Rust MASTER-style backbone fragment search library. Implements exact Kabsch
RMSD over fixed-length backbone fragments, with phantom-typed centroid state and
Rayon-parallel database search.

---

## 1. Crate Layout

```
crates/proxide-frag/
  Cargo.toml
  src/
    lib.rs        — crate root; pub re-exports; no logic
    fragment.rs   — Fragment<N, State>, centering, BackboneAtom enum
    db.rs         — FragmentDb<N>, FragmentDbBuilder<N>, SourceLabel
    kabsch.rs     — kabsch_rmsd(), KabschResult; SVD via nalgebra
    search.rs     — FragmentDb::search(), search_serial(), SearchResult
```

No binary target. No `mod.rs` nesting — all modules are flat siblings under
`src/`.

---

## 2. Cargo.toml

```toml
[package]
name = "proxide-frag"
version.workspace = true
edition.workspace = true
authors.workspace = true
license.workspace = true

[dependencies]
proxide-confind = { path = "../proxide-confind" }
nalgebra = { workspace = true }  # SVD; already in workspace.dependencies at "0.33"
rayon = { workspace = true }
thiserror = { workspace = true }

[dev-dependencies]
approx = { workspace = true }
```

**Workspace Cargo.toml addition required:** add `"crates/proxide-frag"` to the
`[workspace] members` list.

---

## 3. Types

### 3.1 Phantom State Markers (`fragment.rs`)

```rust
use std::marker::PhantomData;

/// Marker: coordinates have NOT been centered.
pub struct Raw;

/// Marker: coordinates have been centered (centroid removed).
pub struct Centered;
```

Both are uninhabited (no fields), zero-size, and `Copy`.

### 3.2 `BackboneAtom` (`fragment.rs`)

Atom ordering within one residue slot. Canonical order: N=0, CA=1, C=2, O=3.
The `N×4×3` layout in `Fragment` follows this.

```rust
/// Index of a backbone atom within one residue slot (N=0, CA=1, C=2, O=3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum BackboneAtom {
    N  = 0,
    CA = 1,
    C  = 2,
    O  = 3,
}
```

### 3.3 `Fragment<const N: usize, State>` (`fragment.rs`)

```rust
/// A fixed-length backbone fragment of `N` residues.
///
/// `State` is either `Raw` (uncentered) or `Centered` (centroid removed).
/// The `coords` field is row-major: `coords[residue][atom][xyz]`.
///
/// `N` is the fragment length in residues; each residue contributes 4 atoms
/// (N, CA, C, O), so there are `N * 4` atom positions and `N * 4 * 3` floats.
#[derive(Debug, Clone)]
pub struct Fragment<const N: usize, State> {
    /// `coords[residue][atom][xyz]`, atom ordering per `BackboneAtom`.
    pub coords: [[[f32; 3]; 4]; N],
    _state: PhantomData<State>,
}

impl<const N: usize> Fragment<N, Raw> {
    /// Construct a raw fragment from a flat coords array.
    ///
    /// The caller is responsible for populating `coords` in the canonical atom
    /// order (N=0, CA=1, C=2, O=3).
    pub fn new(coords: [[[f32; 3]; 4]; N]) -> Self {
        Self { coords, _state: PhantomData }
    }

    /// Center the fragment by subtracting the centroid of all 4N atom positions.
    ///
    /// Returns `Err(AlreadyCenteredError)` if the centroid norm is < 1e-6 Å,
    /// which indicates the fragment is already centered (double-centering guard).
    ///
    /// On success, returns `(Fragment<N, Centered>, centroid: [f32; 3])`.
    /// The centroid is needed to reconstruct original coordinates.
    pub fn center(self) -> Result<(Fragment<N, Centered>, [f32; 3]), AlreadyCenteredError>;
}

impl<const N: usize> Fragment<N, Centered> {
    /// Squared Frobenius norm: sum of squared atom positions across all 4N atoms.
    ///
    /// Pre-computed and stored in `FragmentDbEntry` at build time.
    /// Used in the inner-product RMSD formula.
    pub fn norm_sq(&self) -> f32;
}
```

**Design note:** `Fragment<N, Centered>` cannot be constructed from outside the
crate except through `Fragment<N, Raw>::center()`. The `_state` field is private,
and `Fragment<N, Centered>` has no public constructor. This enforces the type-state
invariant at the API boundary.

### 3.4 `AlreadyCenteredError` (`fragment.rs`)

```rust
#[derive(Debug, thiserror::Error)]
#[error("fragment appears already centered (centroid norm {norm:.2e} < 1e-6 Å); \
         call center() only on Raw fragments")]
pub struct AlreadyCenteredError {
    pub norm: f32,
}
```

### 3.5 `SourceLabel` (`db.rs`)

```rust
/// Identifies the origin of a database fragment in the source PDB.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SourceLabel {
    /// Four-character PDB ID (e.g. "1DC7").
    pub pdb_id: [u8; 4],
    /// Chain identifier character (e.g. b'A').
    pub chain: u8,
    /// Sequence number of the first residue of the fragment (author numbering).
    pub start_res: i32,
    /// Insertion code of the first residue (b' ' = none).
    pub start_icode: u8,
    /// Sequence number of the last residue (inclusive, author numbering).
    pub end_res: i32,
    /// Insertion code of the last residue.
    pub end_icode: u8,
}

impl SourceLabel {
    pub fn new(
        pdb_id: &str,
        chain: char,
        start_res: i32,
        start_icode: char,
        end_res: i32,
        end_icode: char,
    ) -> Self;
}
```

`pdb_id` stored as fixed 4-byte array (no heap allocation per entry). Chain and
insertion codes are single bytes; `b' '` is the "no insertion code" sentinel
(PDB convention).

### 3.6 `FragmentDbEntry<N>` (crate-internal, `db.rs`)

Not exposed publicly. Stored contiguously in `FragmentDb`.

```rust
struct FragmentDbEntry<const N: usize> {
    /// Pre-centered coordinates. Layout: `coords[residue][atom][xyz]`.
    coords: [[[f32; 3]; 4]; N],
    /// Original centroid vector (allows coordinate reconstruction).
    centroid: [f32; 3],
    /// Pre-computed ||coords||² = sum of all squared atom coordinates.
    norm_sq: f32,
    /// Source provenance.
    label: SourceLabel,
}
```

### 3.7 `FragmentDb<N>` and `FragmentDbBuilder<N>` (`db.rs`)

```rust
/// Immutable database of centered backbone fragments of length `N`.
///
/// Built via `FragmentDbBuilder<N>` and frozen on `build()`.
/// Parallelism over entries during search is provided by Rayon.
pub struct FragmentDb<const N: usize> {
    entries: Vec<FragmentDbEntry<N>>,
}

impl<const N: usize> FragmentDb<N> {
    /// Number of fragments in the database.
    pub fn len(&self) -> usize;

    /// True if the database contains no fragments.
    pub fn is_empty(&self) -> bool;

    /// Parallel RMSD search using Rayon.
    ///
    /// Returns all database fragments with RMSD ≤ `epsilon` (Å), sorted by
    /// ascending RMSD.
    pub fn search(
        &self,
        query: &Fragment<N, Centered>,
        epsilon: f32,
    ) -> Vec<SearchResult>;

    /// Single-threaded RMSD search.
    ///
    /// Identical semantics to `search()` but runs on the calling thread.
    /// Useful in contexts where the caller manages its own thread pool or
    /// where Rayon's work-stealing causes unacceptable latency.
    pub fn search_serial(
        &self,
        query: &Fragment<N, Centered>,
        epsilon: f32,
    ) -> Vec<SearchResult>;
}

/// Builder for `FragmentDb<N>`.
///
/// Accepts `Fragment<N, Raw>` + `SourceLabel` pairs. On `build()`, produces
/// an immutable `FragmentDb<N>` with all fragments pre-centered.
pub struct FragmentDbBuilder<const N: usize> {
    entries: Vec<FragmentDbEntry<N>>,
}

impl<const N: usize> FragmentDbBuilder<N> {
    pub fn new() -> Self;

    /// Center `fragment`, compute its norm², and store with `label`.
    ///
    /// Returns `Err(AlreadyCenteredError)` if the fragment centroid norm is
    /// < 1e-6 Å (double-centering guard propagated from `Fragment::center()`).
    pub fn add_fragment(
        &mut self,
        fragment: Fragment<N, Raw>,
        label: SourceLabel,
    ) -> Result<(), AlreadyCenteredError>;

    /// Finalize the database. After this call the builder is consumed.
    pub fn build(self) -> FragmentDb<N>;
}

impl<const N: usize> Default for FragmentDbBuilder<N> {
    fn default() -> Self { Self::new() }
}
```

### 3.8 `SearchResult` (`search.rs`)

```rust
/// A database fragment that matched the query within the RMSD threshold.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// RMSD in Å between the query and this database fragment, after optimal
    /// superposition (Kabsch).
    pub rmsd: f32,
    /// Source provenance of the matched fragment.
    pub label: SourceLabel,
    /// Optimal rotation matrix R (3×3, row-major) that superposes the
    /// **database** fragment onto the **query** fragment.
    ///
    /// `R = V * Uᵀ` where `H = U Σ Vᵀ` (reflection-corrected).
    pub rotation: [[f32; 3]; 3],
}
```

---

## 4. Kabsch Algorithm (`kabsch.rs`)

### 4.1 Input / Output Contract

```rust
/// Result of Kabsch optimal superposition of two centered fragment coordinate
/// sets.
#[derive(Debug, Clone)]
pub struct KabschResult {
    /// RMSD in Å.
    pub rmsd: f32,
    /// Optimal rotation matrix (3×3, row-major). Rotates `b` coords onto `a`.
    pub rotation: [[f32; 3]; 3],
}

/// Compute optimal RMSD and rotation between two centered fragments.
///
/// Uses the inner-product (SVD) form:
///   RMSD² = (norm_sq_a + norm_sq_b) / (N * 4) − 2 * max_σ(SVD(AᵀB)) / (N * 4)
///
/// where `A` and `B` are (N*4) × 3 matrices of centered atom coordinates.
///
/// # Arguments
/// - `a`: query centered coords, shape `[N][4][3]`
/// - `norm_sq_a`: pre-computed ||A||² (sum of squared elements of A)
/// - `b`: database entry centered coords, shape `[N][4][3]`
/// - `norm_sq_b`: pre-computed ||B||²
pub fn kabsch_rmsd<const N: usize>(
    a: &[[[f32; 3]; 4]; N],
    norm_sq_a: f32,
    b: &[[[f32; 3]; 4]; N],
    norm_sq_b: f32,
) -> KabschResult;
```

### 4.2 Algorithm Steps

1. **Build cross-covariance matrix H.**
   Flatten `a` and `b` each to an `(N*4) × 3` view.
   Compute `H = Aᵀ B` as a `nalgebra::Matrix3<f32>`.

   ```
   H[i][j] = Σ_k  A[k][i] * B[k][j]   for i,j ∈ {0,1,2}
   ```

   Using nalgebra:
   ```rust
   let mut h = Matrix3::<f32>::zeros();
   for residue in 0..N {
       for atom in 0..4 {
           let ai = nalgebra::Vector3::from(a[residue][atom]);
           let bi = nalgebra::Vector3::from(b[residue][atom]);
           h += ai * bi.transpose();
       }
   }
   ```

2. **SVD.** Decompose `H = U Σ Vᵀ` using `nalgebra::SVD::new(h, true, true)`.

3. **Reflection guard.** Compute `d = det(V * Uᵀ)`.
   - If `d < 0`: negate the third column of `V` before forming `R = V * Uᵀ`.
   - Adjust the effective sum of singular values:
     `max_trace = σ₀ + σ₁ + d * σ₂`
     where `σ₀ ≥ σ₁ ≥ σ₂ ≥ 0` are the diagonal values from `Σ`.

4. **RMSD.**
   ```
   atoms_total = (N * 4) as f32
   rmsd² = max(0.0, (norm_sq_a + norm_sq_b) / atoms_total − 2.0 * max_trace / atoms_total)
   rmsd  = rmsd².sqrt()
   ```
   The `max(0.0, ...)` guard handles floating-point underflow for near-zero RMSD.

5. **Rotation matrix.** `R = V * Uᵀ` (with reflection-corrected V). Convert to
   `[[f32; 3]; 3]` row-major.

6. **SVD failure handling.** If `svd.u` or `svd.v_t` is `None` (degenerate
   input), return `KabschResult { rmsd: f32::INFINITY, rotation: identity }`.

---

## 5. Search Implementation (`search.rs`)

### 5.1 Parallel Search

```rust
use rayon::prelude::*;

impl<const N: usize> FragmentDb<N> {
    pub fn search(
        &self,
        query: &Fragment<N, Centered>,
        epsilon: f32,
    ) -> Vec<SearchResult> {
        let norm_sq_q = query.norm_sq();
        let mut results: Vec<SearchResult> = self
            .entries
            .par_iter()
            .filter_map(|entry| {
                let kr = kabsch_rmsd(
                    &query.coords, norm_sq_q,
                    &entry.coords, entry.norm_sq,
                );
                if kr.rmsd <= epsilon {
                    Some(SearchResult {
                        rmsd: kr.rmsd,
                        label: entry.label.clone(),
                        rotation: kr.rotation,
                    })
                } else {
                    None
                }
            })
            .collect();
        results.sort_unstable_by(|a, b| a.rmsd.partial_cmp(&b.rmsd).unwrap());
        results
    }
}
```

`search_serial` replaces `par_iter()` with `iter()`. Identical filtering and
sort logic.

**No pre-filter in Phase 1.** Brute-force Kabsch over all entries is the correct
baseline. A lower-bound pre-filter is deferred pending profiling.

---

## 6. `lib.rs` Public Surface

```rust
// crates/proxide-frag/src/lib.rs

#![deny(warnings)]

pub mod fragment;
pub mod db;
pub mod kabsch;
pub mod search;

pub use fragment::{BackboneAtom, Centered, Fragment, Raw, AlreadyCenteredError};
pub use db::{FragmentDb, FragmentDbBuilder, SourceLabel};
pub use kabsch::{kabsch_rmsd, KabschResult};
pub use search::SearchResult;
```

---

## 7. Integration with `proxide-confind`

`proxide-confind` provides:
- `ResidueBackbone` — per-residue N/CA/C/O positions as `Option<[f64; 3]>`
- `ProteinBackbone` — `Vec<ResidueBackbone>` with parallel `ids: Vec<ResidueId>`
- `ResidueId` — `{ chain_id: String, res_id: i32, insertion_code: char }`

The consumer is responsible for extracting a `Fragment<N, Raw>` from a
`ProteinBackbone` window. A helper function is not in scope for Phase 1 (it
belongs in a future integration module or the CLI layer). The spec defines the
types that helper must produce, not the helper itself.

**Precision:** `ResidueBackbone` stores coordinates as `f64`. The fragment stores
`f32`. The conversion (`f64 → f32`) happens in the consumer, not inside
`proxide-frag`. This is consistent with the existing `proxide-confind` pattern
where the single precision boundary is explicit.

**Missing atoms:** `ResidueBackbone` uses `Option<[f64; 3]>` for each atom. If
any of the 4N atom positions required for a window is `None`, the consumer should
skip that window or propagate an error. `Fragment<N, Raw>::new()` accepts only
fully-populated arrays.

---

## 8. Acceptance Criteria

### AC-1: Phantom type enforcement
`Fragment::<5, Raw>::new(coords).center()` compiles.
Passing a `Fragment<5, Raw>` to `FragmentDb::search()` does not compile
(type mismatch: `Raw` vs. `Centered`).
Verified by a `compile_fail` doctest or a `trybuild` test.

### AC-2: Double-centering guard
Centering a fragment whose centroid norm is < 1e-6 Å returns
`Err(AlreadyCenteredError)`. Test: construct a fragment that is already
zero-centered (all atoms arranged symmetrically around the origin) and verify the
error is returned.

### AC-3: RMSD golden value (Kabsch correctness)

**Fixture A (identity check):** 1UBQ chain A residues 1–5 (hardcoded constant
array; see section 9.1). Apply an arbitrary known rotation R₀ and translation t₀
to produce fragment B. After centering both fragments, `kabsch_rmsd` must return
`rmsd < 1e-4` Å.

**Fixture B (non-trivial RMSD):** Fragment A = 1UBQ residues 1–5; fragment B =
1UBQ residues 6–10. Reference RMSD computed by `scripts/analysis/compute_reference_rmsd.py`
(scipy Kabsch). Assert `|rmsd_rust − rmsd_ref| < 0.001` Å.

### AC-4: Search completeness (no false negatives)
Build a `FragmentDb<5>` with 100 randomly-generated centered fragments. Insert the
query itself as one of the 100 entries. Call `search(query, epsilon=0.001)`. Assert
the self-entry appears in the results with `rmsd ≤ 0.001`.

### AC-5: Search precision (no false positives above threshold)
All returned results from `search()` and `search_serial()` have `rmsd ≤ epsilon`.

### AC-6: Serial / parallel parity
`search()` and `search_serial()` return identical results (same RMSD values and
labels, same sort order) for the same query and database.

### AC-7: Empty database
`FragmentDbBuilder::new().build()` succeeds. `search()` on it returns an empty
`Vec`.

### AC-8: Cargo check clean
`cargo check -p proxide-frag` passes with `#![deny(warnings)]` in `lib.rs`.

### AC-9: norm_sq consistency
`Fragment::<N, Centered>::norm_sq()` equals the manual sum-of-squares computed
from the same `coords` array. Validates that the pre-computed value in
`FragmentDbEntry::norm_sq` is set correctly.

---

## 9. Test Fixtures

### 9.1 1UBQ 5-mer Hardcoded Array

Coordinates from PDB 1UBQ ATOM records (chain A, residues 1–5, atoms N/CA/C/O).
Used as a `const` array in Rust tests and in the Python reference script.

```
# 1UBQ chain A, residues 1-5 (Met-Gln-Ile-Phe-Val)
# PDB 1UBQ (1.80 Å, R=0.176), cols 30-54
# [residue, atom, [x, y, z]]
1  N   [ 1.885, 41.770, 73.681]
1  CA  [ 3.288, 41.413, 73.682]
1  C   [ 3.803, 41.074, 75.086]
1  O   [ 3.043, 41.142, 76.047]
2  N   [ 5.095, 40.754, 75.172]
2  CA  [ 5.748, 40.448, 76.494]
2  C   [ 6.009, 39.007, 76.500]
2  O   [ 6.924, 38.617, 77.226]
3  N   [ 5.254, 38.228, 75.739]
3  CA  [ 5.412, 36.822, 75.575]
3  C   [ 4.169, 36.281, 74.897]
3  O   [ 3.131, 36.917, 74.845]
4  N   [ 4.168, 35.027, 74.367]
4  CA  [ 3.022, 34.357, 73.747]
4  C   [ 2.736, 33.026, 74.393]
4  O   [ 1.600, 32.726, 74.561]
5  N   [ 3.679, 32.165, 74.773]
5  CA  [ 3.516, 30.869, 75.396]
5  C   [ 4.476, 29.832, 74.756]
5  O   [ 5.508, 30.119, 74.078]
```

### 9.2 Python Reference Script

`scripts/analysis/compute_reference_rmsd.py` — standalone analytical script.
Run once to produce the golden RMSD value for AC-3 fixture B. Committed result
is embedded as a constant in the Rust test.

The script uses:
- `numpy` SVD for manual Kabsch
- `scipy.spatial.transform.Rotation` for the known-rotation fixture A check
- Output: prints `rmsd_fixture_b = <value>` to stdout

---

## 10. Deferred / Phase 2

| Item | Deferral rationale |
|---|---|
| Disk persistence / mmap binary format | In-memory search must be validated first |
| Pre-filter (distance geometry lower bound) | Profile real databases before adding complexity |
| Variable-length search (multiple `FragmentDb<N>` instances) | Trivial caller-side wrapper; not in core API |
| Python bindings (pyo3) | Add after API stabilizes |
| Benchmarks against reference MASTER C++ | Requires building C MASTER; deferred to perf phase |

---

## 11. Design Decisions Made During Formalization

**D1: `norm_sq` stored on `FragmentDbEntry`, computed on-demand for query.**
Database entries pre-compute `norm_sq` once at `add_fragment` time (it is then
read N_db times during search). The query's `norm_sq` is computed on-demand via
`Fragment::norm_sq()` (called once per search invocation).

**D2: `search()` returns `Vec<SearchResult>` sorted ascending by RMSD.**
Sorted output is the natural contract for a threshold search. `sort_unstable_by`
is used (allocation-free, in-place).

**D3: `rotation` field always included in `SearchResult`.**
Including it unconditionally avoids forcing callers to recompute it. It is a flat
3×3 array (no heap allocation).

**D4: `SourceLabel` uses fixed `[u8; 4]` for `pdb_id`.**
Avoids heap allocation per database entry. PDB IDs are always exactly 4 chars.
`SourceLabel::new()` accepts `&str` and truncates/pads internally.

**D5: `Fragment<N, Centered>` has no public constructor.**
The `_state: PhantomData<State>` field is private. The only path to a
`Fragment<N, Centered>` is through `Fragment<N, Raw>::center()`, which always
applies the transform and runs the double-centering guard.

**D6: Centering uses f32 arithmetic throughout.**
Protein coordinates are < 200 Å in magnitude; f32 precision is sufficient and
consistent with the existing `proxide-geometry` kabsch implementation.

**D7: `kabsch_rmsd` is a free function, not a method on `Fragment`.**
`fragment.rs` handles type-state transitions and coordinate storage. `kabsch.rs`
owns the numerical algorithm. `search.rs` composes them. This separation makes
each module independently testable.
