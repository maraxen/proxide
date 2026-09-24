---
title: probe_residue_contacts — fused XTC decode + periodic nearest-residue assignment kernel
description: Spec for a Rust kernel plus PyO3 binding that reads only the requested XTC frames, decodes only protein and probe atoms, and assigns each probe molecule (and each probe atom) to its nearest protein residue within a cutoff under orthorhombic minimum image. Output is deterministic and memory-bounded. It implements sweetprots prereg 260923 §2.1 exactly.
task_id: 260923_proxide-probe-residue-contacts
status: draft
created: 260923
---

# Specification: `probe_residue_contacts` kernel

## Overview

Add a fused, parallel, deterministic kernel. It reads selected frames of an XTC straight from their
byte offsets and decodes only the protein and probe atoms. Using each frame's own orthorhombic box
with minimum image, it assigns every probe molecule, and every probe atom, to the residue of its
nearest protein atom within a cutoff. Contact channels come back as bounded arrays. The consumer is
the per-frame contact measurement of the sweetprots pre-registration
`260923_chirality-site-preference-and-mi-at-site.md` §2.1. This kernel replaces that section's
`mdtraj.iterload` + `cKDTree` implementation, and it must reproduce that implementation's semantics
exactly.

**Project rule this spec is written against.** `/home/marielle/projects/proxide/CLAUDE.md` says to
fail fast and loud, never fill a sentinel for an unknown value (ledger A1, A4), keep gates
machine-decidable (B1), treat silence as failure (B3) and never truncate silently (B9).

**The rule is not in this worktree's checkout.** This worktree has no `CLAUDE.md` or `AGENTS.md`
anywhere; Glob returned nothing. The only copy is in the main checkout, which is where it was read.

**Consequences for the kernel:**
- It never falls back to Euclidean distance.
- It never truncates output.
- It never reinterprets negative indices.
- It never emits NaN or ±inf.
- Anything it cannot determine raises an error.

---

## 1. Semantics (source of truth: prereg §2.1, pinned here)

### 1.1 Definitions

These definitions make prereg §2.1 unambiguous. Every rule below is pinned. The Rust kernel and the
Python reference (§5.2) implement it byte-for-byte.

- **Atom sets are supplied by the caller.**
  - `P` is the protein atoms (per §2.1: protein heavy atoms with the probe resname excluded).
  - `Q` is the probe atoms (per §2.1: probe heavy atoms), grouped into molecules.
  - The kernel does not infer elements, heaviness, residues, molecules or chemical groups. It
    cannot verify heaviness because it has no topology. The consumer's §2.1 gate (15 heavy atoms
    per TR molecule, 21 per MN) is what enforces that.
- **Frame coordinates.** These are the decoded XTC positions: molly `f32`, **nm**, exactly as stored.
  - No superposition, no unwrapping, no re-imaging of stored coordinates.
- **Probe shift (optional).** This is for the §2.1 positive control. `q' = f64(q) + shift_nm`, applied
  per probe atom in f64 before any distance is computed. The default is no shift. The shift is not
  applied to the stored coordinates.
- **Minimum-image difference, per axis k.** `L_k` is `f64` of the frame's box diagonal entry k.
  ```
  d_k = f64(p_k) - q'_k
  d_k = d_k - L_k * floor(d_k / L_k + 0.5)
  ```
  - The formula uses `floor(x + 0.5)`, **not** `round`.
  - Rust `f64::round` rounds half away from zero, and numpy `np.round` rounds half to even. The
    existing `pairwise_distances_mic` uses `.round()` on f32 (`distances.rs:140`).
  - Pinning `floor(x + 0.5)` makes both implementations produce the identical double.
- **Squared distance.** `d2 = d_x*d_x + d_y*d_y + d_z*d_z`, evaluated left to right in f64 with no
  FMA. Rust does not auto-contract. The numpy reference evaluates the same expression elementwise.
- **Contact predicate.** `d2 <= c2`, where `c2 = cutoff_nm * cutoff_nm` in f64.
  - This is **inclusive**, matching §2.1 "Distance ≤ 4.5 Å".
  - The consumer passes `cutoff_nm = 0.45`. 0.45 is not exactly representable, and that is fine,
    because both implementations compute the same `c2`.
- **Candidate key** for a (probe atom a, protein atom p) pair in contact:
  `key = (d2, residue_of[p], full_atom_index[p])`, compared lexicographically.
- **Atom label.** `atom_label[a]` is the residue of the minimum-key contact pair over all
  `p ∈ P` in contact with `a`. If `a` has no contact pair it is `NO_CONTACT = -1`.
- **Molecule label.** `molecule_label[m]` is the residue of the minimum-key contact pair over all
  `a ∈ m` and `p ∈ P`. If there is no contact pair it is `NO_CONTACT`.
  - This is §2.1's partition: "each probe molecule that contacts the protein is assigned to exactly
    one residue, the one at minimum heavy–heavy distance".
  - It equals the `atom_label` of the molecule's best atom.
- **Tie rule** (deterministic, independent of iteration order and thread count). When two protein
  atoms are at bit-identical `d2`:
  1. the lower residue index wins;
  2. then the lower full-topology atom index.

  The probe atom's identity never matters.
- **All-contacts channel.** This channel is not partitioned. The set `C_f` contains every distinct
  `(m, r)` such that some `a ∈ m` and some `p ∈ P` with `residue_of[p] = r` satisfy `d2 <= c2`.
  `all_contact_counts[f, r] = |{m : (m, r) ∈ C_f}|`.
- **Heavy-atom channel and group channels.** Each channel is an atom-level partition.
  - `channel_counts[f, c, r]` is the number of probe atoms `a` with `channel_of[a] = c` and
    `atom_label[a] = r`.
  - The heavy-atom channel is `channel_counts.sum(axis=1)`. It is not stored separately; the
    Python wrapper derives it on request, see §2.3.
- **`NO_CONTACT = -1` is a determined state.** It means "no protein atom within the cutoff", and it
  is disjoint from every valid residue index (≥ 0).
  - It is not an unknown and not a sentinel for a failed computation.
  - There is no "unknown" output state: anything the kernel cannot determine raises (CLAUDE.md
    hard rule).
  - It is exported as a named constant, `proxide.NO_CONTACT`.

### 1.2 Open semantic questions (flagged; the consumer must resolve them before Task 2 of the prereg)

- **A1 — atom set of the all-contacts channel.** §2.1 lists "all-contacts (not partitioned)". Prereg
  Task 3 says "mirror-image poses give identical **all-atom** counts". That could mean hydrogens are
  included in this channel.
  - This spec pins all-contacts as **heavy–heavy**, since §2.1's contact definition is heavy–heavy,
    and makes the kernel atom-set agnostic.
  - An H-inclusive variant is a second call with H-inclusive `P`/`Q`, which doubles decode I/O.
  - If A1 resolves to "all-atom", add a v1.1 same-pass channel: a second `(P_all, Q_all)` pair
    evaluated on the one decode. That is out of scope here.
- **A2 — unit of the heavy-atom and group channels.** This spec pins **probe atoms** as the counted
  unit (per-atom nearest residue). That is the grid analog: a voxel is assigned to its nearest heavy
  atom within 4.5 Å.
  - The alternative is **group instances**: one count per (molecule, group), placed at the group's
    min-key residue.
  - Either can be derived exactly in Python from `atom_labels` together with
    `return_atom_keys=True` (§2.3). No re-run is needed whichever way A2 is resolved.
- **A3 — tie rule.** §2.1 does not specify one. `cKDTree` nearest-neighbour ties depend on the
  implementation. The consumer should record §1.1's rule in a prereg disposition note, since this
  is an implementation detail and not a measurement change.

---

## 2. API

### 2.1 Rust — per-frame pure core (`proxide-geometry`, no I/O)

New file `crates/proxide-geometry/src/geometry/probe_contacts.rs`, registered in
`geometry/mod.rs`. It must not carry `#![allow(dead_code)]`, unlike `cell_list.rs:7`,
`neighbors.rs:7` and `distances.rs:5`.

```rust
pub const NO_CONTACT: i32 = -1;

pub struct ContactTopology<'a> {
    pub protein_full_idx: &'a [u32],   // len P, unique; used only for tie-break
    pub protein_residue: &'a [u32],    // len P, each < n_residues
    pub n_residues: u32,
    pub probe_mol_offsets: &'a [u32],  // CSR, len M+1, [0]=0, strictly increasing, last=Q
    pub probe_channel: &'a [u8],       // len Q, each < n_channels
    pub n_channels: u8,
    pub cutoff_nm: f64,
    pub probe_shift_nm: Option<[f64; 3]>,
}

pub struct FrameContacts {
    pub atom_labels: Vec<i32>,          // len Q
    pub atom_best: Vec<Option<(f64, u32, u32)>>, // len Q; key of the winning pair; None iff NO_CONTACT
    pub molecule_labels: Vec<i32>,      // len M
    pub molecule_best_d2: Vec<Option<f64>>,       // len M; None iff NO_CONTACT
    pub contact_pairs: Vec<(u32, u32)>, // (molecule, residue), sorted, deduplicated
}

pub fn assign_frame(
    protein_xyz_nm: &[[f32; 3]],        // len P, same order as protein_full_idx
    probe_xyz_nm: &[[f32; 3]],          // len Q
    box_lengths_nm: [f64; 3],           // already validated orthorhombic and periodic
    topo: &ContactTopology<'_>,
    scratch: &mut ContactScratch,       // reusable grid buffers; per thread
) -> FrameContacts;
```

`Option` stands for "no contact" inside Rust. The −1 exists only at the array boundary.

### 2.2 Rust — fused driver (`proxide-io`, features `xtc` + `parallel`)

New file `crates/proxide-io/src/formats/xtc_probe_contacts.rs`, declared in `formats/mod.rs`
behind `#[cfg(all(feature = "xtc", feature = "parallel"))]`.

```rust
pub enum OffsetCachePolicy { ReadWrite, ReadOnly, Refresh }

pub struct ProbeContactRequest<'a> {
    pub protein_full_idx: &'a [u32],
    pub protein_residue: &'a [u32],
    pub n_residues: u32,
    pub probe_full_idx: &'a [u32],      // len Q, unique, disjoint from protein_full_idx unless allow_index_overlap
    pub probe_mol_offsets: &'a [u32],
    pub probe_channel: &'a [u8],
    pub n_channels: u8,
    pub cutoff_nm: f64,
    pub ortho_tol_nm: f64,              // max |off-diagonal| accepted as orthorhombic; default 0.0
    pub probe_shift_nm: Option<[f64; 3]>,
    pub allow_index_overlap: bool,      // default false; see §2.4 E12
    pub emit_contact_pairs: bool,
    pub emit_atom_keys: bool,
    pub max_output_bytes: u64,          // default 4 GiB; exceeding it is an error before any I/O
    pub offset_cache: OffsetCachePolicy,
    pub n_threads: usize,               // >= 1; required, no implicit "all cores"
}

pub fn probe_residue_contacts<P: AsRef<Path>>(
    xtc_path: P,
    frame_indices: &[usize],            // strictly increasing, non-empty, each < frame_count
    req: &ProbeContactRequest<'_>,
) -> Result<ProbeContactResult, ProbeContactError>;
```

**`ProbeContactResult`** holds flat row-major buffers; shapes are in §2.3. It also carries
`offsets_source` ∈ {`ProxideSidecar`, `MdanalysisNpz`, `Scan`} and `total_frames_on_disk`.

**`ProbeContactError`** is a `thiserror` enum. It wraps `XtcError` (`xtc.rs:72-82`) and adds one
variant per validation failure in §2.4, each carrying the offending index or value.

### 2.3 Python

Thin wrapper `proxide.probe_residue_contacts` in the new module `src/proxide/probe_contacts.py`,
re-exported from `src/proxide/__init__.py` next to `read_xtc_ca_distogram` (`__init__.py:56-58`,
`__all__` at 82-86). It calls the raw binding `proxide._proxider._probe_residue_contacts`,
registered in `crates/proxide_py/src/lib.rs` under `#[cfg(all(feature = "xtc", feature =
"parallel"))]` like `lib.rs:124-125`.

```python
def probe_residue_contacts(
    xtc_path: str | os.PathLike,
    frame_indices: ArrayLike,              # int, 1-D
    protein_atom_indices: ArrayLike,       # int, 1-D, full-topology indices (heavy, probe-excluded)
    protein_residue_of_atom: ArrayLike,    # int, 1-D, same length; dense residue ids 0..n_residues-1
    n_residues: int,
    probe_atom_indices: ArrayLike,         # int, 1-D, all probe atoms (heavy), molecule-contiguous
    probe_molecule_offsets: ArrayLike,     # int, 1-D, CSR, len n_molecules+1
    probe_atom_channel: ArrayLike,         # int, 1-D, len(probe_atom_indices); caller-defined groups
    channel_names: Sequence[str],          # len n_channels, unique, e.g. ("carboxylate_O", ..., "other")
    *,
    cutoff_nm: float,                      # REQUIRED, no default (consumer: 0.45)
    n_threads: int,                        # REQUIRED, >= 1
    ortho_tol_nm: float = 0.0,
    probe_shift_nm: tuple[float, float, float] | None = None,
    allow_index_overlap: bool = False,
    return_contact_pairs: bool = True,
    return_atom_keys: bool = False,
    max_output_bytes: int = 4 * 2**30,
    offset_cache: Literal["read_write", "read_only", "refresh"] = "read_write",
) -> dict[str, Any]
```

**Input conversion.**
- The wrapper converts each index array with `np.asarray`.
- It requires an integer dtype. Bool and float arrays are errors, with no silent cast.
- It rejects any negative value **before** casting to unsigned, so there is no numpy-style negative
  indexing (ledger A1).

**The group definitions are caller-supplied.** `probe_atom_channel` assigns every probe atom to
exactly one channel, so the channels partition the probe atoms. The consumer supplies the
§2.1 table's five groups: carboxylate O, ammonium N, hydroxyl O, indole ring and other. The
kernel perceives nothing itself.

**Returned dict (fixed keys).** F = frames, M = molecules, Q = probe atoms, R = residues,
C = channels.

| key | dtype | shape | notes |
|---|---|---|---|
| `frame_indices` | int64 | [F] | echo of the validated input |
| `times_ps` | float32 | [F] | from the frame header |
| `box_lengths_nm` | float32 | [F, 3] | box diagonal as stored |
| `molecule_labels` | int32 | [F, M] | residue id or `NO_CONTACT` |
| `atom_labels` | int32 | [F, Q] | residue id or `NO_CONTACT` |
| `partition_counts` | int32 | [F, R] | `bincount(molecule_labels[f] >= 0)`; primary `n_r(t)` |
| `channel_counts` | int32 | [F, C, R] | per-atom partition by channel |
| `all_contact_counts` | int32 | [F, R] | all-contacts channel |
| `contact_pairs` | dict of int32 [K] arrays `frame_pos`, `molecule`, `residue` | K | only if `return_contact_pairs`; sorted lexicographically by (frame_pos, molecule, residue). `U(t)` for any site set is derived from these |
| `assigned` | dict of `frame_pos` int32, `molecule` int32, `residue` int32, `min_dist_nm` float64 | [K'] | rows **only** for molecules with a label ≥ 0, so there is no fill value for non-contacting molecules |
| `atom_keys` | dict of `frame_pos`, `atom`, `d2_nm2` (float64), `protein_atom` (int64) | [K''] | only if `return_atom_keys`; rows only for atoms with a label ≥ 0 (A2 re-derivation) |
| `meta` | dict | — | see below |

`meta` contains:
- `cutoff_nm`, `cutoff_rule="d2 <= cutoff_nm**2 (f64)"`, `mic_rule="d -= L*floor(d/L+0.5) (f64)"`
  and `tie_rule="min (d2, residue, protein_atom_index)"`;
- `n_threads`, `offsets_source`, `total_frames_on_disk`, `channel_names`, `n_residues`,
  `n_molecules`, `n_probe_atoms`, `proxide_version` and `probe_shift_nm`.

The heavy-atom channel is `channel_counts.sum(axis=1)`. The wrapper exposes it through the helper
`heavy_atom_counts(result)`, so it is not stored twice.

**Memory bound.** The bound is checked before any I/O, with K estimated at its worst case
`F·M·min(R, 32)` for the check:

```
bytes = 4·F·(M + Q + R·(C + 2)) + 12·K + 20·K'
```

For the consumer (F=2000, M≈300, Q≈6300, R≈1000, C=5, K≈1e6) that is about 110 MB. If the bound
exceeds `max_output_bytes`, the call raises `ValueError` naming the term that dominates. It never
truncates (ledger B9).

**Peak transient memory** is `n_threads × (decoded subset ≤ 12·(maxidx+1) bytes + grid O(P + n_cells))`.
That is ≤ 3 MB per thread for a 240k-atom frame. Decoded frames are never collected. Contrast
`read_xtc_distogram_parallel`, which collects every frame first (`xtc.rs:700`).

### 2.4 Validation (all loud; Python `ValueError`, prefix `probe_residue_contacts:`, naming the offending index or value)

Errors E1–E11 are raised before any frame decode. E13–E16 are raised per frame; the whole call
fails and the message names the frame index. No partial results are returned.

| # | Condition |
|---|---|
| E1 | `frame_indices` empty, not 1-D, non-integer, negative, **not strictly increasing** (this covers duplicates and unsorted input), or any value ≥ `frame_count` |
| E2 | `protein_atom_indices` empty, negative, duplicated, or ≥ `n_atoms` (header) |
| E3 | `protein_residue_of_atom` length ≠ P, negative, or ≥ `n_residues`; `n_residues` < 1 |
| E4 | `probe_atom_indices` empty, negative, duplicated, or ≥ `n_atoms` |
| E5 | `probe_molecule_offsets` not starting at 0, not strictly increasing (an empty molecule is an error), or last ≠ Q |
| E6 | `probe_atom_channel` length ≠ Q, negative, or ≥ `len(channel_names)`; `channel_names` empty, duplicated, or longer than 255 |
| E7 | `cutoff_nm` not finite or ≤ 0 |
| E8 | `n_threads` < 1 |
| E9 | `ortho_tol_nm` negative or not finite; `probe_shift_nm` not finite |
| E10 | output bound > `max_output_bytes` |
| E11 | the file cannot be opened, or the offset scan fails (`XtcError`, message passed through) |
| E12 | `protein_atom_indices ∩ probe_atom_indices ≠ ∅` and not `allow_index_overlap`. With the flag set, overlap is allowed, and a shared atom lies at `d2 = 0` from itself. This is the deliberate prereg §2.6 NC2 "defect channel", which recreates the 28cdbb9 failure. |
| E13 | any box off-diagonal with abs value > `ortho_tol_nm`. Triclinic boxes are unsupported; the kernel **never** reads the diagonal and drops the shear, which is what `BoxDims::from_diagonal_matrix` does silently at `distances.rs:86-90`. |
| E14 | box missing or degenerate: any diagonal entry not finite or ≤ 0. The kernel **never** falls back to Euclidean distance, unlike `pairwise_distances_mic` at `distances.rs:101-103, 132`. |
| E15 | `min(L_k) < 2·cutoff_nm`, where minimum image becomes ambiguous |
| E16 | a decoded frame's natoms differs from the header natoms, or any selected coordinate is non-finite |

---

## 3. Algorithm

1. **Validation.** Validate E1–E10 with the GIL held for the Python-side checks, then release the GIL
   (`py.allow_threads`, the same pattern as `py_xtc_reader.rs:208, 367`) for everything below.
2. **Offsets.** Open with `XtcReader` (`xtc.rs:305-373`) and get the offsets, honouring
   `offset_cache`:
   - `ReadWrite` is today's behaviour: load the sidecar or MDAnalysis npz, else scan and store
     (`xtc.rs:210-220, 385-399`).
   - `ReadOnly` loads if valid, else scans, but **never writes** a `.offsets` or converted sidecar.
     This needs a small `XtcReader` change: `store()` is currently called unconditionally at
     `xtc.rs:243, 394`.
   - `Refresh` means `refresh_offsets()` (`xtc.rs:339-344`).
   - Record the `offsets_source`.
3. **Decode plan.**
   - `U = sorted(unique(protein_full_idx ∪ probe_full_idx))`.
   - The selection is `AtomSelection::Mask` built from `U`
     (`molly-0.6.1/src/selection.rs:35-48`). molly compacts selected atoms in ascending index order
     and stops decoding at the last selected atom (`reading_limit`, `selection.rs:105-112`).
   - Precompute `pos_in_U` for every protein and probe atom, so the kernel maps back to caller order
     without relying on the caller's ordering.
   - This avoids the ordering and deduplication hazard noted at `py_xtc_reader.rs:36-42`, because the
     remap is explicit.
4. **Parallel map over positions `0..F`.**
   - `frame_indices.par().using(|_| WorkerState::new(path))` (orx-parallel 2.4.0 `using`,
     `par_iter.rs:469`) with an explicit `.num_threads(n_threads)` (`par_iter.rs:143`).
   - `n_threads` is applied on all targets. Today it is only applied on wasm (`xtc.rs:666-667`), and
     native runs use `NumThreads::Auto`.
   - Each worker opens **one** file handle and reuses it, plus one decode `Frame` and a
     `ContactScratch`. Today's `read_frames_parallel` opens a handle per frame (`xtc.rs:651`), which
     is an avoidable metadata operation per frame on network filesystems.
   - Per frame:
     1. Seek to `offsets[i]` and call `read_frame_at_offset::<BUFFERED>(frame, off, &mask)`
        (`molly lib.rs:477-488`). `BUFFERED = true` when `reading_limit < natoms`, otherwise
        `false`. Only this frame's bytes are read; there is no staging copy.
     2. Validate the box (E13–E15) and the coordinates (E16). The box is 3×3 column-major per molly
        (`lib.rs:133-136`). All six off-diagonals are checked, so the orientation question in §9 U10
        does not matter.
     3. Gather `protein_xyz` and `probe_xyz` via `pos_in_U`.
     4. Call `assign_frame` (§2.1):
        - **Build the periodic cell grid over protein atoms.** Per axis,
          `n_k = floor(L_k / (cutoff_nm·(1 + 1e-4)))`. If `n_k < 3`, set `n_k = 1`. A single cell
          spans the whole axis, so no neighbour offset is visited twice.
        - Cell width therefore exceeds the cutoff by a safety margin, so no in-cutoff pair is ever
          outside the 3×3×3 stencil under f64 rounding.
        - **Wrap for binning only.** `s = x/L − floor(x/L)`; if `s ≥ 1`, set `s = 0`;
          `cell = min(floor(s·n), n−1)`. Distances always use raw coordinates plus the §1.1 minimum
          image.
        - **Storage.** CSR built by counting sort: `cell_start[n_cells+1]`, and atoms sorted by
          (cell, protein ordinal). This is deterministic and allocation-reused, with no `HashMap`.
          The existing `CellList` (`cell_list.rs:15-63`) is non-periodic, bounding-box-relative and
          `HashMap`-based, so it is **not** reused.
        - **Per probe atom** (shifted if requested): visit the stencil cells, compute `d2` for each
          candidate, and keep the min key over candidates with `d2 <= c2`. Collect the
          `(molecule, residue)` contact pairs. Then fold the atom keys into the molecule min key.
          Finally sort and deduplicate the frame's pairs.
   - The map returns `(pos, FrameContacts, time, box)`.
5. **Assembly.**
   - Sort the collected vector by `pos`. The default `IterationOrder::Ordered` (`par_iter.rs:36`)
     already guarantees this, but the sort makes determinism independent of that default.
   - Write the dense arrays row by row, counting via bincount per frame.
   - Concatenate the COO tables in `pos` order.
   - Output is **byte-identical for any `n_threads`**.
6. **Return.** Re-acquire the GIL and build the numpy arrays (`PyArray1::from_vec_bound` + `reshape`,
   the pattern at `py_xtc_reader.rs:391-395`).

**Invariants.** These are asserted in the tests, and checked at runtime in debug builds.
- `partition_counts[f].sum() == (molecule_labels[f] >= 0).sum()`.
- `channel_counts[f].sum() == (atom_labels[f] >= 0).sum()`.
- For every m: `molecule_labels[f,m] >= 0` ⇔ any `atom_labels[f, m's atoms] >= 0`, and the molecule
  label is one of those atom labels.
- `all_contact_counts[f].sum() == #pairs in frame f`.
- Every assigned `(m, label)` is present in `contact_pairs`.

**Complexity.** Per frame, about Q × 27 × (P / n_cells) distance evaluations, roughly 1e6 for the
consumer. That is negligible next to decoding a ~240k-atom frame. **I/O plus decode dominates.**

---

## 4. Secondary API: general periodic neighbour search (scoped, optional, Task 8)

`proxide.neighbors_within_cutoff_pbc(query_xyz_nm, target_xyz_nm, box_lengths_nm, cutoff_nm,
n_threads)` operates on in-memory f32 arrays and returns COO `(query, target, d2)`, sorted. It
reuses Task 1's grid and follows §1.1's rules: inclusive `<=`, f64, `floor(x + 0.5)` minimum image,
and E13–E15-equivalent box checks.

**Why it is useful.** It serves the consumer's §2.1 "mean probe–probe cluster size" metric, which
this kernel does not compute.

**Deferral.** Task 8 can be deferred without affecting the core channels. If it is deferred, that
metric stays in consumer-side Python. The existing
`cell_list::find_neighbors_within_cutoff_fast` / `neighbors::find_neighbors_within_cutoff` are not
exposed. Their cutoff semantics disagree:
- `cell_list.rs` uses `<=` at lines 116 and 143, while its doc at line 131 says `<`;
- `neighbors.rs:66` uses `<`.

Exposing either would publish that inconsistency.

---

## 5. Correctness gates

All gates are machine-decidable against fixed artifacts (ledger B1). A skipped parity test is a
**failure**, not a pass (B3). Every Python gate runs with `PROXIDE_REQUIRE_MDTRAJ=1`, under which
the test module raises instead of skipping when mdtraj is missing. mdtraj is only in the
`trajectories` extra (`pyproject.toml:66-71`). The test files run individually, never as a whole
suite, per the local compute limits.

### 5.1 Rust unit tests (Tasks 1–2, 4)

Hand-constructed boxes with f32-exact coordinates:
- **G-R1 PBC wrap.** Protein atom at x=0.05, probe at x=L−0.05, L=3.0, cutoff 0.5. The pair is in
  contact at d=0.1 across the face, and also in the corner/edge variant across all three axes.
  Coordinates offset by +3L and −2L give identical labels.
- **G-R2 exact cutoff equality.** cutoff_nm=0.5 is exactly representable. Probe at 0.25, protein at
  0.75, so d2 = 0.25 = c2 exactly → contact. Protein at 0.75 plus one f32 ulp → no contact.
- **G-R3 ties.**
  - Two protein atoms in residues 7 and 3 at bit-identical d2 from one probe atom → label 3.
  - Same residue, different atom → the lower full index is recorded in the atom key.
  - Tied residues across two probe atoms of one molecule → deterministic.
  - The order of `protein_full_idx` is permuted and the result is unchanged.
- **G-R4 empty.**
  - No probe atom within the cutoff → all labels `NO_CONTACT`, counts all zero, no pairs.
  - A frame whose protein set is far from every probe → the same.
- **G-R5 partition vs all-contacts.** One molecule touches residues 2 and 5: `partition_counts` puts
  1 in exactly one of them, while `all_contact_counts` puts 1 in both.
- **G-R6 grid ≡ brute force.** 500 seeded random configurations comparing the Task 1 grid against an
  O(P·Q) brute force using the same §1.1 arithmetic.
  - Box shapes include `n_k ∈ {1, 2, 3, 7}` cells per axis, including the `L = 2·cutoff` boundary.
  - Atoms are placed exactly at 0, at L, at −L, and within ±1 ulp of cell boundaries.
  - All outputs must be identical.
- **G-R7 shift.** `probe_shift_nm` equals recomputing with coordinates pre-shifted in f64.
- **G-R8 validation.** One test per E-row that is reachable in Rust, asserting the error variant and
  its payload.
- **G-R9 subset decode.** On each checked-in XTC fixture under `tests/data/trajectories/`:
  - Mask decode (both `BUFFERED` modes) equals All-decode followed by index selection,
    **bit-for-bit**.
  - Cases: a prefix mask, a sparse non-prefix mask, and a mask whose last index is `natoms−1`.
- **G-R10 thread determinism** (driver). On `tests/data/trajectories/frame0.xtc` (22 atoms, 501
  frames, real box), with a synthetic protein/probe split, outputs are byte-identical for
  `n_threads ∈ {1, 2, 4, 8}` and for reversed `frame_indices` chunking.

**Gates:**
- `cargo test -p proxide-geometry probe_contacts periodic_grid`
- `cargo test -p proxide-io --features xtc,parallel xtc_probe_contacts xtc_subset_decode`

### 5.2 Python parity (Task 6): `tests/validation/test_probe_contacts_parity.py`

**Independent reference.** `tests/validation/_probe_contacts_reference.py` is pure numpy with no
proxide import. It is a brute force chunked over probe atoms: f64, `floor(x + 0.5)` minimum image,
`<=`, lexsort on `(protein_atom, residue, d2)` taking the first. It shares no code with the kernel.

**Synthetic XTC.** It is written in the test with `mdtraj.Trajectory.save_xtc`, the same construction
as `bench_xtc_decode.py:44-96`. The system has:
- about 400 protein atoms in 60 residues;
- 30 probe molecules of 15 atoms, in 5 channels;
- water filler;
- 60 frames with a varying orthorhombic box;
- probes deliberately planted across faces, edges and corners, and at exact ties.

**Coordinate source for the reference.** The reference must use coordinates bit-identical to the
kernel's. Existing proxide readers return `x*10` in f32 (`py_xtc_reader.rs:233`), which is lossy
for recovering nm. Task 5 therefore adds the test hook `proxide._proxider._read_xtc_frames_nm`,
which returns raw molly nm f32 and the box for the given frames and atoms.

**Assertions** (all exact, with `np.array_equal`):
- **G-P1.** `molecule_labels`, `atom_labels`, `partition_counts`, `channel_counts`,
  `all_contact_counts` and all `contact_pairs` columns are identical to the reference.
- **G-P2.** The reference is run on **mdtraj-decoded** coordinates (`mdtraj.load(...).xyz`,
  float32 nm) as well.
  - First assert `np.array_equal(mdtraj_xyz, proxide_raw_nm_xyz)`. This is the decode-identity
    check; see U4.
  - If they are identical, G-P1 must also hold for the mdtraj-fed reference.
  - If they are not identical, the test **fails**. It does not downgrade to a tolerance. The fixer
    must then report the max ULP delta, and G-P2 is re-specified by the reviewer. No silent
    tolerance is allowed (B1).
- **G-P3 determinism.** `n_threads ∈ {1, 2, 4, 8}` gives byte-identical dicts (all arrays).
- **G-P4 validation.** One `pytest.raises(ValueError, match=...)` per row E1–E16. The synthetic file
  gets a triclinic-box variant for E13 and a zero-box variant for E14.
- **G-P5 invariants** from §3, on every frame.
- **G-P6 offset cache.** With `offset_cache="read_only"`, no `.offsets` file appears in a fresh temp
  directory. `refresh` works after the XTC is rewritten with the same size, and a stale sidecar is
  not trusted; see risk R3.

**Gate:**
`PROXIDE_REQUIRE_MDTRAJ=1 uv run --extra trajectories pytest tests/validation/test_probe_contacts_parity.py -q -rs`.
It must exit 0 **and** report 0 skipped.

### 5.3 Real-data parity (consumer-side; spec only, executed in sweetprots)

The script is `sweetprots/scripts/analysis/sweet_probe_contacts_parity.py`, with a bathos sidecar.

**Input.**
- One replicate of `vft_apo-stripped_kd7p8A_MND_his-neutral` (prereg §2.6 build).
- 50 frames: `frame_indices = 0, 5, …, 245`, the prereg's `stride=5` grid.
- The probe resname is excluded from `P`.
- Groups come from the §2.1 perception.

**Where it runs.** On the prereg Task 0b excerpt if that excerpt holds ≥ 246 raw frames. Otherwise
it runs as a 1-task `sbatch`, never on the login node.

**Reference.** mdtraj decode with `atom_indices = U` feeds the §5.2 numpy reference.

**Pass conditions:**
- the decode-identity assert passes;
- labels, all three count arrays and the pairs are all identical;
- the box is orthorhombic (E13 not raised);
- one repeat run is byte-identical.

**Outcome.** Write `parity_pass ∈ {0,1}` plus `n_label_mismatch`, `n_decode_mismatch` and
`max_abs_decode_delta_nm`. Any value ≠ 0 means the kernel **must not** be used for prereg Task 2.

### 5.4 Pinned assumptions tested implicitly

- orx-parallel honours `num_threads` on native: G-R10 and G-P3 would not detect a violation of this,
  so the benchmark (§6) records the achieved speed-up per thread count.
- molly Mask decode equals All-decode: G-R9.
- molly decode equals mdtraj decode: G-P2.

---

## 6. Benchmark (Task 7): `benchmarks/bench_probe_contacts.py` + `bench_probe_contacts.py.bth.toml`

The script extends the `bench_xtc_decode.py` pattern:
- thread env caps (lines 31-33);
- a synthetic fixture in a tempdir that is never committed;
- a cold/warm sidecar split (lines 117-131);
- JSON output.

**Synthetic system.** About 15k atoms:
- 5k protein heavy atoms in ~600 residues;
- 200 probe molecules × 15 heavy atoms, 5 channels;
- water filler to 15k;
- orthorhombic box of 7 nm;
- probes placed so ~30–50% are in contact.
- Default 2000 frames; `--smoke` gives 50 frames and finishes in < 60 s.

**Arms** (all over the same `frame_indices`, stride 1):
1. **`reference_mdtraj`:** `mdtraj.iterload(chunk=200, atom_indices=U)` plus scipy
   `cKDTree(boxsize=L)` (`query_ball_point` + nearest with `distance_upper_bound`) per frame. This is
   the prereg §2.1 implementation. scipy is not a proxide dependency (`pyproject.toml`), so the
   script errors loudly without it: run `uv run --extra trajectories --with scipy python benchmarks/bench_probe_contacts.py`.
2. **`reference_numpy`:** the §5.2 reference, for information only.
3. **`proxide`:** at `n_threads ∈ {1, 4, 8}`, with a cold and a warm sidecar.

**Reported per arm.** `frames_per_s`, median/min/max over `--repeats 3`, and bytes read from
`/proc/self/io` `read_bytes` (Linux). The byte count detects read-ahead amplification during the
cold offset scan; see R6.

**Acceptance targets.** These are stated now and measured later, and none of them is claimed until
measured.
- **T-B1.** `proxide@8 ≥ 5 × reference_mdtraj` frames/s on the synthetic system (warm sidecar).
- **T-B2.** `proxide@1 ≥ 1 × reference_mdtraj`. This is a sanity floor; failing it means the kernel
  is slower than the thing it replaces.
- **T-B3.** Speed-up `proxide@8 / proxide@1 ≥ 3`. This detects `num_threads` being ignored (U2).
- **T-B4 (production, consumer-measured on Engaging).**
  - 2000 frames (stride 5) of one ~9.5 GB / 10k-frame `production.xtc` in **< 3 min** wall on 8
    cores, read directly from the pool path.
  - This holds conditional on pool throughput ≥ 20 MB/s, measured with `dd` in the same job
    immediately before the run (the throughput-collapse precedent).
  - Arithmetic: 2000 frames × ~0.95 MB = ~1.9 GB, which is ~95 s at 20 MB/s. At the 11 MB/s
    measured during the 260912 collapse, it is ~170 s, so T-B4 is I/O-bound and may miss for
    reasons outside the kernel. Record the measured MB/s alongside the result.

**Gate for Task 7.**
- `uv run --extra trajectories --with scipy python benchmarks/bench_probe_contacts.py --smoke --json-out /tmp/bpc.json`
  exits 0.
- The JSON has every arm and key above.
- `bth validate-sidecar benchmarks/bench_probe_contacts.py.bth.toml` exits 0.

The targets are recorded when measured; they do not gate merge.

---

## 7. Fixer tasks

The tasks are ordered, and each is independently testable. Dependencies: T1 → T2 → T4; T3 → T4;
T4 → T5 → {T6, T7}; T1 → T8.

### Task 1: periodic cell grid + pinned MIC arithmetic (`proxide-geometry`)
Add `geometry/periodic_grid.rs`, containing:
- `fn mic_d2(p: [f32;3], q: [f64;3], l: [f64;3]) -> f64`, implementing §1.1 exactly;
- `PeriodicCellGrid::build(points, box_lengths, min_cell_nm) -> Result<Self, GridError>`, with
  the §3 cell counts and CSR layout;
- `for_each_candidate(query, FnMut(u32))`.

`GridError` rejects a non-finite box, a box ≤ 0, and `L < 2·cutoff`. Register it in `mod.rs`, with
no `allow(dead_code)`. Tests: G-R1, G-R2 (at the `mic_d2` level) and G-R6.

**Files:** `crates/proxide-geometry/src/geometry/periodic_grid.rs` (create), `geometry/mod.rs` (modify)
**Gate:** `cargo test -p proxide-geometry periodic_grid`
**Scope:** ~250 LOC including tests

### Task 2: per-frame assignment core
Implement §2.1: `ContactTopology`, `FrameContacts`, `ContactScratch` and `assign_frame`, including
the key/tie rule and the pair deduplication. Tests: G-R1 through G-R5 and G-R7 at the frame level.

**Files:** `crates/proxide-geometry/src/geometry/probe_contacts.rs` (create), `geometry/mod.rs` (modify)
**Gate:** `cargo test -p proxide-geometry probe_contacts`
**Scope:** ~350 LOC

### Task 3: subset decode + offset-cache policy (`proxide-io/xtc.rs`)
- Add `pub struct SubsetPlan { mask: AtomSelection, union: Vec<u32>, reading_limit: usize, natoms: usize }`.
- Add `fn decode_subset_at(reader: &mut XTCReader<File>, offset, &SubsetPlan, &mut MollyFrame)`,
  choosing `BUFFERED` per §3.
- Add `OffsetCachePolicy` and thread it through `ensure_offsets` / `scan_and_cache_offsets` /
  `OffsetCache::load`. `ReadOnly` suppresses both `store()` calls, at `xtc.rs:243` and `xtc.rs:394`.
- Expose `offsets_source`.
- Existing callers keep `ReadWrite`, so their behaviour is unchanged.

Tests: G-R9, plus a `ReadOnly`-writes-nothing test.

**Files:** `crates/proxide-io/src/formats/xtc.rs` (modify), `formats/tests/xtc_tests.rs` (modify)
**Gate:** `cargo test -p proxide-io --features xtc,parallel xtc_subset_decode offset_cache_policy`
plus the existing `cargo test -p proxide-io --features xtc,parallel xtc`, which must stay green
**Scope:** ~150 LOC

### Task 4: fused parallel driver
Implement §2.2 and §3 steps 2–5:
- validation E1–E16 (E10 as a bound computation);
- `using` per-thread state, explicit `num_threads`, and ordered assembly;
- the invariant checks under `debug_assertions`.

Tests: G-R8 and G-R10.

**Files:** `crates/proxide-io/src/formats/xtc_probe_contacts.rs` (create), `formats/mod.rs` (modify),
`formats/tests/probe_contacts_tests.rs` (create)
**Gate:** `cargo test -p proxide-io --features xtc,parallel xtc_probe_contacts`
**Scope:** ~450 LOC

### Task 5: PyO3 binding + Python wrapper + raw-nm test hook
- Add `_probe_residue_contacts` in `py_xtc_reader.rs`, or a new `py_probe_contacts.rs`.
  - Inputs are extracted as `PyReadonlyArray1<i64>` and converted with a negativity check.
  - Compute runs inside `allow_threads`.
  - The dict is built after the GIL is re-acquired.
- Add `_read_xtc_frames_nm(path, frame_indices, atom_indices)`, returning raw f32 nm plus the 3×3
  box. It is test-only, with an underscore name and not in `__all__`.
- Register both in `lib.rs` under `cfg(all(feature="xtc", feature="parallel"))`.
- Add `src/proxide/probe_contacts.py`: a wrapper with dtype and shape checks, the `NO_CONTACT`
  constant, and `heavy_atom_counts()`. Re-export it in `__init__.py`.

**Files:** `crates/proxide_py/src/py_probe_contacts.rs` (create), `crates/proxide_py/src/lib.rs`,
`src/proxide/probe_contacts.py` (create), `src/proxide/__init__.py` (modify),
`tests/test_probe_contacts_api.py` (create: import, signature, required kwargs, E1–E9 dtype and
negativity errors)
**Gate:** rebuild the extension (`uv run maturin develop --release`, or the repo's documented
`uv pip install -e ".[dev]"`), then `uv run pytest tests/test_probe_contacts_api.py -q`
**Scope:** ~250 LOC Rust + ~150 LOC Python

### Task 6: Python parity, determinism and validation tests
Implement §5.2: the reference module plus the test module (G-P1 through G-P6).

**Files:** `tests/validation/_probe_contacts_reference.py` (create),
`tests/validation/test_probe_contacts_parity.py` (create)
**Gate:** the §5.2 gate command, with 0 skipped
**Scope:** ~400 LOC

### Task 7: benchmark + sidecar
Implement §6.

**Files:** `benchmarks/bench_probe_contacts.py` (create), `benchmarks/bench_probe_contacts.py.bth.toml` (create)
**Gate:** the §6 smoke gate
**Scope:** ~300 LOC

### Task 8 (optional; deferrable): Python neighbour search
Implement §4, reusing Task 1.

**Files:** `crates/proxide-geometry/src/geometry/periodic_grid.rs` (extend),
`crates/proxide_py/src/py_probe_contacts.rs` (extend), `src/proxide/probe_contacts.py` (extend),
`tests/validation/test_neighbors_pbc_parity.py` (create; numpy brute-force parity plus the tie and
`<=` cases)
**Gate:** `PROXIDE_REQUIRE_MDTRAJ=1 uv run pytest tests/validation/test_neighbors_pbc_parity.py -q -rs`,
with 0 skipped
**Scope:** ~200 LOC

### Task 9 (consumer; sweetprots repo, not this worktree)
Implement §5.3's real-data parity script and sidecar. Record the A1–A3 resolutions and the kernel
swap as a prereg disposition note **before** prereg Task 2 is submitted.

---

## 8. Risks

| # | Risk | Mitigation |
|---|---|---|
| R1 | molly decode differs from mdtraj/xdrfile in precision edge cases: large coordinates, precision ≠ 1000, small-int runs. Existing proxide parity tests only assert 0.01 Å (`test_xtc_distogram_parity.py:46`). | G-P2 and the §5.3 decode-identity asserts are bit-exact and fail loud. On any mismatch the kernel is not used for the prereg until the difference is characterised. |
| R2 | molly `Mask` or `BUFFERED` path is unexercised in proxide: every current call is `::<false>` with `AtomSelection::All` (`xtc.rs:473, 496, 653`). | G-R9 bit-equality against All-decode on every fixture. Falling back to `BUFFERED=false` with a full-range mask is allowed only as an explicit code path covered by G-R9. |
| R3 | A stale offset sidecar survives because the XTC was replaced with the same size and mtime (for example `rsync -t` of a re-run). `matches()` checks only magic, version, natoms, size and mtime (`xtc.rs:173-187`). | E16 per-frame natoms check. molly's `read_header` asserts `natoms == natoms_repeated` (`lib.rs:100`), so a misaligned offset errors instead of decoding garbage. `offset_cache="refresh"` is available. The consumer uses `refresh` on the first run per file. |
| R4 | Triclinic boxes are out of scope. If naurmalade production boxes are not orthorhombic (unverified, U5), every call raises E13. | Loud failure by design. The consumer keeps the prereg's `mdtraj.compute_neighbors` triclinic path. Triclinic minimum image (a reduced-cell 27-image search) is a separate spec. |
| R5 | GIL. Long compute holding the GIL would serialise Python threads; with `allow_threads`, Ctrl-C does not interrupt a running call. | All compute runs in `allow_threads`, as in the existing pattern. Interruptibility is accepted and documented. For long jobs the consumer should chunk `frame_indices` (for example 200 frames per call). |
| R6 | Cold offset scan on a network FS. 10k header reads can trigger read-ahead of most of a 9.5 GB file, which defeats "read only requested bytes". | Benchmark `read_bytes`. The consumer pre-generates the sidecar once per file (`offset_cache="read_write"` in a dedicated step) or imports the MDAnalysis npz, then runs `read_only`. |
| R7 | Writing `.offsets` into naurmalade campaign directories on the shared pool is a side effect in another project's output tree. | `offset_cache="read_only"` suppresses every write. The consumer decides whether sidecars may live there (U9). |
| R8 | Atom order. If probes follow water in the topology, `reading_limit ≈ natoms` and there is no early-stop saving (U6). | Correctness is unaffected, only speed. The benchmark reports `reading_limit / natoms`. |
| R9 | Semantics drift from prereg §2.1 (A1–A3). | Resolve A1–A3 in a prereg disposition note before prereg Task 2. `atom_keys` allows A2 re-derivation without a re-run. |
| R10 | Rollback | The change is additive: new files plus one opt-in `XtcReader` policy parameter whose default preserves current behaviour. Revert the branch. |

---

## 9. Assumptions not verified in code (flagged)

- **U1.** orx-parallel's ordered `collect` preserves input order. It is documented as the default
  (`par_iter.rs:36`), but the design does not rely on it: §3 step 5 sorts.
- **U2.** On native builds, orx-parallel honours `.num_threads(n)`. Today's native XTC path never
  sets it (`xtc.rs:666-667` is wasm-only), and `proxide-parallel-rt`'s global defaults to 1 but is
  only read on wasm (`crates/proxide-parallel-rt/src/lib.rs:3-10`). Checked by T-B3.
- **U3.** molly Mask decode is bit-identical to All-decode for the selected atoms. Checked by G-R9.
- **U4.** molly decode is bit-identical to mdtraj decode. It is unverified anywhere in the repo.
  Checked by G-P2 and §5.3.
- **U5.** naurmalade production boxes are orthorhombic. Prereg Task 0a item 7 (CRYST1 angles) is
  still pending in prereg §9.
- **U6.** Atom order (protein, probe, water) in `production.pdb` is unknown.
- **U7.** The meaning of prereg §2.1's channel units (A1, A2).
- **U8.** Engaging pool throughput at run time, and read-ahead behaviour on a cold scan.
- **U9.** Whether the campaign directories are writable, and whether sidecars are acceptable there.
- **U10.** Box matrix orientation. molly documents `boxvec` as column-major (`molly lib.rs:133-136`),
  while `xtc.rs:15-26` says "row i = box vector i". This is irrelevant here because all six
  off-diagonals are checked (E13), but it should be resolved in the module docs separately.
- **U11.** `frame_count` drops a truncated trailing frame (`xtc.rs:466-480`). mdtraj's frame count
  may differ by one for a file still being written. The files are finished, so this is not expected
  to matter; E1 rejects any out-of-range index.
- **U12.** The consumer's molecule and atom sizes (M ≈ 300, Q ≈ 6300, R ≈ 1000, about 240k atoms per
  frame) are estimated from 9.5 GB / 10k frames (prereg §9) and 0.13 M. They are not measured.
- **U13.** Build and install command for the extension. README says `uv pip install -e ".[dev]"`
  (maturin, `pyproject.toml:105-111`). A fixer should confirm the fastest dev rebuild in this repo.

## 10. Out of scope

These are explicit:
- triclinic boxes;
- the H-inclusive all-atom channel (pending A1);
- probe–probe clustering, beyond the optional Task 8 primitive;
- `U(t)` for site sets and relative SASA. `U(t)` is derivable from `contact_pairs`.
- numbering and UniProt mapping, and group perception. These stay consumer-side.
- frame equalisation, bootstrap and JSD;
- DCD and TRR inputs;
- GPU;
- changing `pairwise_distances_mic`'s silent Euclidean fallback (`distances.rs:132`) or
  `BoxDims::from_diagonal_matrix`'s silent shear drop. Both resemble ledger class A and should be
  filed as debt, not changed here.
- fixing the `cell_list.rs` doc/code `<` vs `<=` mismatch. File as debt.

## References

- Consumer: `/home/marielle/projects/sweetprots/.claude/worktrees/wt-20260831-162012/.praxia/docs/preregistration/260923_chirality-site-preference-and-mi-at-site.md` §2.1 (lines 128-197), §2.3–2.4 (209-250), §2.6 (263-300), §6 Tasks 1–3 (1160-1228), §9 (1299-1322)
- `crates/proxide-io/src/formats/xtc.rs` (offset cache 137-292, `XtcReader` 305-536, `read_frames_parallel` 634-669, distogram 695-724)
- `crates/proxide-geometry/src/geometry/{cell_list.rs, neighbors.rs, distances.rs}`
- `crates/proxide_py/src/py_xtc_reader.rs` (195-276, 287-298, 329-405); `crates/proxide_py/src/lib.rs:108-125`; `crates/proxide_py/Cargo.toml:43-50`
- molly 0.6.1 (`~/.cargo/registry/.../molly-0.6.1/src/{selection.rs,lib.rs,buffer.rs}`); orx-parallel 2.4.0 (`src/par_iter.rs`); both versions are from `Cargo.lock`
- `benchmarks/bench_xtc_decode.py`; `tests/validation/test_xtc_distogram_parity.py`
- `/home/marielle/projects/proxide/CLAUDE.md` (hard rules + failure-mode ledger)
