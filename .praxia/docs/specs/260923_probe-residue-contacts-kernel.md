---
title: probe_residue_contacts — fused XTC decode + periodic nearest-residue assignment kernel
description: Spec for a Rust kernel plus PyO3 binding that reads only the requested XTC frames straight from the pool path, using caller-supplied offsets, and decodes only protein and probe atoms. It assigns each probe molecule (and each probe atom) to its nearest protein residue within a cutoff under orthorhombic minimum image. Output is deterministic and memory-budgeted. It implements sweetprots prereg 260923 §2.1 exactly, and building it is gated on a Task 0 measurement.
task_id: 260923_proxide-probe-residue-contacts
status: draft
revision: r2 (responds to adversarial review r1, audit 260923_proxide-probe-residue-contacts_spec_challenge_1; see "Review r1 disposition")
created: 260923
---

# Specification: `probe_residue_contacts` kernel

## Overview

Add a fused, parallel, deterministic kernel. It reads selected frames of an XTC straight from their
byte offsets on the pool path, with no staging copy, and decodes only the protein and probe atoms.
Using each frame's own orthorhombic box with minimum image, it assigns every probe molecule, and
every probe atom, to the residue of its nearest protein atom within a cutoff. Contact channels come
back as memory-budgeted arrays.

The consumer is the per-frame contact measurement of the sweetprots pre-registration
`260923_chirality-site-preference-and-mi-at-site.md` §2.1. This kernel would replace that section's
`mdtraj.iterload` + `cKDTree` implementation, and it must reproduce that implementation's semantics
exactly.

**Build is conditional.** Building the kernel is **gated on Task 0**, a measurement of the
reference path on one real replicate, with a go/no-go rule stated before the numbers are in (§7,
Task 0). If the rule says no-go, the consumer keeps the prereg's reference path and this spec stays
`draft`.

**I/O mode, decided (review r1 M4 and minor 5).**
- The kernel reads **directly from the pool path, using offsets supplied by the caller**.
- It never writes a sidecar and never consults one: no proxide `.offsets`, no MDAnalysis npz.
- The benefit depends on the consumer switching from full-file staging to direct reads. A strided
  read of 2000 of 10k frames touches about 20% of the file's bytes. If the consumer keeps staging
  the whole file, only the compute-side gain remains, and Task 0's rule accounts for that.

**Project rule this spec is written against.** `/home/marielle/projects/proxide/CLAUDE.md` says to
fail fast and loud, never fill a sentinel for an unknown value (ledger A1, A4), keep gates
machine-decidable (B1), treat silence as failure (B3) and never truncate silently (B9).

**The rule is not in this worktree's checkout.** This worktree has no `CLAUDE.md` or `AGENTS.md`;
the only copy is in the main checkout, which is where it was read.

**Consequences for the kernel:**
- It never falls back to Euclidean distance.
- It never truncates output.
- It never reinterprets negative indices.
- It never emits NaN or ±inf.
- Anything it cannot determine raises an error.

**Box shape, verified by the orchestrator on Engaging (260923).** Production boxes are **cubic,
128.249 Å** (12.8249 nm). The orthorhombic-only design therefore applies. E13 still checks every
frame.

---

## 1. Semantics (source of truth: prereg §2.1, pinned here)

### 1.1 Definitions

Every rule below is pinned. The Rust kernel and the Python reference (§5.2) implement it
byte-for-byte.

- **Atom sets are supplied by the caller.**
  - `P` is the protein atoms (per §2.1: protein heavy atoms with the probe resname excluded).
  - `Q` is the probe atoms (per §2.1: probe heavy atoms), grouped into molecules.
  - The kernel does not infer elements, heaviness, residues, molecules or chemical groups. The
    consumer's §2.1 group gate (15 heavy atoms per TR molecule, 21 per MN) is what enforces those.
  - Because the atom sets come from the caller, the kernel shares that caller's parameters. §5.3's
    literal arm is the one reference that does not share them (review r1 M2).
- **Frame coordinates.** These are the decoded XTC positions: molly `f32`, **nm**, exactly as stored.
  - No superposition, no unwrapping, no re-imaging of stored coordinates.
- **Probe shift (optional).** This is for the §2.1 positive control. `q' = f64(q) + shift_nm`, applied
  per probe atom in f64 before any distance is computed.
- **Minimum-image difference, per axis k.** `L_k` is `f64` of the frame's box diagonal entry k.
  ```
  d_k = f64(p_k) - q'_k
  d_k = d_k - L_k * floor(d_k / L_k + 0.5)
  ```
  - The formula uses `floor(x + 0.5)`, **not** `round`. Rust `f64::round` rounds half away from zero,
    numpy's rounds half to even, and the existing `pairwise_distances_mic` uses `.round()` on f32
    (`distances.rs:140`).
- **Squared distance.** `d2 = d_x*d_x + d_y*d_y + d_z*d_z`, evaluated left to right in f64 with no
  FMA.
- **Contact predicate.** `d2 <= c2`, where `c2 = cutoff_nm * cutoff_nm` in f64.
  - This is **inclusive**, matching §2.1 "Distance ≤ 4.5 Å".
  - The consumer passes `cutoff_nm = 0.45`.
- **Candidate key** for a contacting pair (probe atom a, protein atom p):
  `key = (d2, residue_of[p], full_atom_index[p])`, compared lexicographically.
- **Atom label.** `atom_label[a]` is the residue of the minimum-key contact pair of `a`, or
  `NO_CONTACT = -1` if there is none.
- **Molecule label.** `molecule_label[m]` is the residue of the minimum-key contact pair over all
  `a ∈ m`, or `NO_CONTACT`. This is §2.1's partition: "the one at minimum heavy–heavy distance".
- **Tie rule.** At bit-identical `d2`:
  1. the lower residue index wins;
  2. then the lower full-topology atom index.

  The probe atom's identity never matters.
- **All-contacts channel.** This channel is not partitioned.
  - `C_f = {(m, r) : ∃ a ∈ m, p ∈ P, residue_of[p] = r, d2(a, p) <= c2}`.
  - `all_contact_counts[f, r] = #{m : (m, r) ∈ C_f}`, so the counted unit is **molecules**; see A1.
- **Partition channel.** `partition_counts[f, r] = #{m : molecule_labels[f, m] == r}`.
  - Equivalently, `np.bincount(molecule_labels[f][molecule_labels[f] >= 0], minlength=R)`.
- **Heavy-atom channel and group channels.** Each is an atom-level partition.
  - `channel_counts[f, c, r] = #{a : channel_of[a] == c and atom_labels[f, a] == r}`.
  - The heavy-atom channel is `channel_counts.sum(axis=1)`.
- **`NO_CONTACT = -1` is a determined state.** It means "no protein atom within the cutoff", and it
  is disjoint from every residue index (≥ 0).
  - There is no "unknown" output state.
  - It is exported as `proxide.NO_CONTACT`.

### 1.2 Open semantic questions (the consumer must resolve them before prereg Task 2)

- **A1 — the all-contacts channel.** It has two parts.
  - **(i) Atom set.** §2.1's contact definition is heavy–heavy, but prereg Task 3 says "identical
    **all-atom** counts", which suggests hydrogens might count.
  - **(ii) Counted unit.** It could be distinct molecules per residue (pinned), probe atoms per
    residue, or (probe atom, protein atom) pairs.
  - This spec pins **heavy–heavy, molecules**. Any other resolution is a v1.1 change: a different
    count over the same pass. It is out of scope here.
- **A2 — unit of the heavy-atom and group channels.** This spec pins **probe atoms** (the grid
  analog).
  - The alternative is group instances: one count per (molecule, group), at the group's min-key
    residue.
  - After r1 removed `atom_keys` (§10), the alternative **cannot** be derived from these outputs.
  - If A2 resolves to group instances, that is a v1.1 channel.
- **A3 — tie rule.** §2.1 does not specify one. The consumer records §1.1's rule in a prereg
  disposition note.

---

## 2. API

### 2.1 Rust — per-frame pure core (`proxide-geometry`, no I/O)

New file `crates/proxide-geometry/src/geometry/probe_contacts.rs`, registered in `geometry/mod.rs`,
with no `#![allow(dead_code)]` (contrast `cell_list.rs:7`, `neighbors.rs:7`, `distances.rs:5`).

```rust
pub const NO_CONTACT: i32 = -1;

pub struct ContactTopology<'a> {
    pub protein_full_idx: &'a [u32],   // len P, unique; tie-break only
    pub protein_residue: &'a [u32],    // len P, each < n_residues
    pub n_residues: u32,
    pub probe_mol_offsets: &'a [u32],  // CSR, len M+1, [0]=0, strictly increasing, last=Q
    pub probe_channel: &'a [u8],       // len Q, each < n_channels
    pub n_channels: u8,
    pub cutoff_nm: f64,
    pub probe_shift_nm: Option<[f64; 3]>,
}

/// Borrowed output rows for ONE frame; the driver hands out disjoint row slices of the
/// preallocated dense output (§3 step 5). Nothing per-atom survives the call except what is
/// written into these rows.
pub struct FrameRows<'o> {
    pub atom_labels: &'o mut [i32],        // len Q
    pub molecule_labels: &'o mut [i32],    // len M
    pub partition_counts: &'o mut [i32],   // len R
    pub channel_counts: &'o mut [i32],     // len C*R
    pub all_contact_counts: &'o mut [i32], // len R
}

pub struct FrameSparse {
    pub contact_pairs: Vec<(u32, u32)>,    // (molecule, residue), sorted, dedup; empty unless requested
    pub assigned: Vec<(u32, u32, f64)>,    // (molecule, residue, min_dist_nm) for labelled molecules
}

pub fn assign_frame(
    protein_xyz_nm: &[[f32; 3]],   // len P, order of protein_full_idx
    probe_xyz_nm: &[[f32; 3]],     // len Q
    box_lengths_nm: [f64; 3],      // validated orthorhombic + periodic
    topo: &ContactTopology<'_>,
    scratch: &mut ContactScratch,  // per-thread grid + per-atom key buffers, reused across frames
    rows: FrameRows<'_>,
    emit_pairs: bool,
) -> FrameSparse;
```

**Where the per-atom state lives.** The per-atom best keys (`Option<(f64, u32, u32)>`) live **only**
in `ContactScratch`. They are overwritten every frame and never collected. This fixes review r1 M3:
the retained `atom_best` would have cost about 300 MB.

### 2.2 Rust — offsets and fused driver (`proxide-io`, features `xtc` + `parallel`)

```rust
/// Read-only header scan; never reads or writes any sidecar.
pub struct XtcOffsets { pub offsets: Vec<u64>, pub file_size: u64, pub mtime_ns: i128, pub natoms: usize }
pub fn xtc_frame_offsets<P: AsRef<Path>>(path: P) -> Result<XtcOffsets, XtcError>;

pub struct ProbeContactRequest<'a> {
    pub protein_full_idx: &'a [u32],
    pub protein_residue: &'a [u32],
    pub n_residues: u32,
    pub probe_full_idx: &'a [u32],       // unique; disjoint from protein unless allow_index_overlap
    pub probe_mol_offsets: &'a [u32],
    pub probe_channel: &'a [u8],
    pub n_channels: u8,
    pub cutoff_nm: f64,
    pub allow_large_cutoff: bool,        // default false; see E7
    pub ortho_tol_nm: f64,               // default 0.0
    pub probe_shift_nm: Option<[f64; 3]>,
    pub allow_index_overlap: bool,       // default false; see E12
    pub emit_contact_pairs: bool,
    pub max_output_bytes: u64,           // default 4 GiB; see §2.3 memory
    pub offsets: Option<&'a XtcOffsets>, // None => in-memory scan (no file written); see §3 step 2
    pub n_threads: usize,                // >= 1, required
}

pub fn probe_residue_contacts<P: AsRef<Path>>(
    xtc_path: P,
    frame_indices: &[usize],             // strictly increasing, non-empty
    req: &ProbeContactRequest<'_>,
) -> Result<ProbeContactResult, ProbeContactError>;
```

**Files.**
- `xtc_frame_offsets` lives in `crates/proxide-io/src/formats/xtc.rs`. It reuses
  `determine_offsets_tolerant` and `drop_trailing_frame_if_truncated` (`xtc.rs:428-480`) **without**
  `OffsetCache::load` or `store` (`xtc.rs:210-291`).
- The driver goes in the new `crates/proxide-io/src/formats/xtc_probe_contacts.rs`.

**`ProbeContactError`** is a `thiserror` enum. It wraps `XtcError` (`xtc.rs:72-82`) and has one
variant per row of §2.4, each carrying the offending value and, for per-frame errors, `pos` and
`frame_index`.

### 2.3 Python

The wrapper is in `src/proxide/probe_contacts.py` and re-exported from `src/proxide/__init__.py`
(`__init__.py:56-58`, `__all__` at 82-86). It exports:
- `probe_residue_contacts`
- `xtc_frame_offsets`
- `NO_CONTACT`
- `heavy_atom_counts(result)`

The raw bindings are `_proxider._probe_residue_contacts` and `_proxider._xtc_frame_offsets`,
registered in `crates/proxide_py/src/lib.rs` under `cfg(all(feature="xtc", feature="parallel"))`,
like `lib.rs:124-125`.

```python
def xtc_frame_offsets(xtc_path) -> dict   # {"offsets": uint64[n], "file_size": int, "mtime_ns": int, "natoms": int}
# Consumer runs this once per file and stores the dict in ITS OWN output dir (np.savez); never next to the XTC.

def probe_residue_contacts(
    xtc_path, frame_indices, protein_atom_indices, protein_residue_of_atom, n_residues,
    probe_atom_indices, probe_molecule_offsets, probe_atom_channel, channel_names,
    *,
    cutoff_nm: float,                 # REQUIRED (consumer 0.45)
    n_threads: int,                   # REQUIRED, >= 1
    frame_offsets: dict | None = None,  # from xtc_frame_offsets; None => in-memory scan
    allow_large_cutoff: bool = False,
    ortho_tol_nm: float = 0.0,
    probe_shift_nm: tuple[float, float, float] | None = None,
    allow_index_overlap: bool = False,
    return_contact_pairs: bool = True,
    max_output_bytes: int = 4 * 2**30,
) -> dict
```

**Input conversion.** Index arrays must have an integer dtype. Bool and float arrays are rejected,
and negative values are rejected **before** any unsigned cast.

**Returned dict (fixed keys).** F = frames, M = molecules, Q = probe atoms, R = residues,
C = channels.

| key | dtype | shape | notes |
|---|---|---|---|
| `frame_indices` | int64 | [F] | echo |
| `times_ps` | float32 | [F] | strictly increasing (E17) |
| `box_lengths_nm` | float32 | [F, 3] | box diagonal as stored |
| `molecule_labels` | int32 | [F, M] | residue or `NO_CONTACT` |
| `atom_labels` | int32 | [F, Q] | residue or `NO_CONTACT` |
| `partition_counts` | int32 | [F, R] | §1.1 |
| `channel_counts` | int32 | [F, C, R] | §1.1 |
| `all_contact_counts` | int32 | [F, R] | §1.1 |
| `contact_pairs` | `frame_pos`, `molecule`, `residue`: int32 [K] each | K | only if requested; sorted by (frame_pos, molecule, residue). `U(t)` for any site set is derived from these. |
| `assigned` | `frame_pos` int32, `molecule` int32, `residue` int32, `min_dist_nm` float64 | [K'] | rows only for labelled molecules, so there is no fill value |
| `meta` | dict | — | see below |

`meta` contains:
- the rule strings: `cutoff_rule`, `mic_rule`, `tie_rule`;
- the run parameters: `cutoff_nm`, `n_threads`, `offsets_source ∈ {"caller", "scan"}`,
  `total_frames_on_disk` and `probe_shift_nm`;
- the sizes: `channel_names`, `n_residues`, `n_molecules`, `n_probe_atoms`;
- `accounted_peak_bytes` and `proxide_version`.

**Memory (review r1 M3).** Output bytes split into a dense part, which is exact, and a sparse part,
which is budgeted as it grows.

```
dense  = 4·F·(M + Q + R·(C + 2)) + 16·F          # labels, counts, times, boxes — exact
sparse = 12·K + 20·K'                            # contact_pairs + assigned — data-dependent
```

- **The dense part is exact.** It is computed before any I/O and must satisfy
  `dense ≤ max_output_bytes`, else E10.
- **The sparse part has no useful a-priori upper bound.** The true worst case, `K ≤ F·M·R`, is about
  7 GB at consumer size, so r1's `min(R, 32)` was an estimate, not a bound.
- **It is therefore enforced while it grows.** Each worker adds its frame's sparse bytes to a shared
  `AtomicU64`. When `dense + 2·sparse_so_far > max_output_bytes`, the call fails with **E18**. The
  factor 2 covers the final concatenation copy. The error reports the accumulated bytes and the
  frame position reached. It never truncates (B9).
- **The expected size** for the consumer (F=2000, M≈300, Q≈6300, R≈1000, C=5, K≈1e6, K'≈3e5) is
  about 90 MB dense plus about 18 MB sparse. This is an estimate that Task 0 refines.
- **Peak process memory is accounted.** It is
  `dense + 2·sparse + n_threads·transient`, where transient is the per-thread decoded subset
  (`12·|U|` bytes) plus the grid (`4·(P + n_cells + 1)`) plus the atom-key scratch (`16·Q`).
  - The driver computes this peak and returns it as `meta["accounted_peak_bytes"]`.
  - A `debug_assert!` checks that every buffer the driver allocates is counted in that total. The
    Rust tests run it.

### 2.4 Validation (Python `ValueError`, prefix `probe_residue_contacts:`, naming the value)

**When each check runs (minor 6).**
- **With the GIL held,** before release, on pure inputs: E1 (shape, dtype, sign, monotonicity),
  E2–E9 (except their upper bounds on `n_atoms`), E10 and E12.
- **After release, before any frame decode:** E11, then the `n_atoms` and `frame_count` bounds of
  E1, E2, E4 and E19.
- **After release, per frame:** E13–E17, with E18 checked continuously.

**Which per-frame error is reported (minor 7).** When per-frame errors occur, the error reported is
the one with the **lowest `pos`**, deterministically.
- A shared atomic `min_failed_pos` is kept. Frames at a higher `pos` may skip work.
- Frames at a lower `pos` still run, so a lower failure is never missed.
- No partial results are returned.

| # | Condition |
|---|---|
| E1 | `frame_indices` empty, not 1-D, non-integer, negative, not strictly increasing, or ≥ `frame_count` |
| E2 | `protein_atom_indices` empty, negative, duplicated, or ≥ `natoms` |
| E3 | `protein_residue_of_atom` length ≠ P, negative, or ≥ `n_residues`; `n_residues` < 1 |
| E4 | `probe_atom_indices` empty, negative, duplicated, or ≥ `natoms` |
| E5 | `probe_molecule_offsets` not starting at 0, not strictly increasing, or last ≠ Q |
| E6 | `probe_atom_channel` length ≠ Q or out of range; `channel_names` empty, duplicated, or longer than 255 |
| E7 | `cutoff_nm` not finite or ≤ 0; **or `cutoff_nm > 1.0` without `allow_large_cutoff=True`**. This catches Å passed as nm, e.g. 4.5 (review r1 M2). |
| E8 | `n_threads` < 1 |
| E9 | `ortho_tol_nm` < 0 or not finite; `probe_shift_nm` not finite |
| E10 | exact dense bytes > `max_output_bytes` |
| E11 | the file cannot be opened; the scan fails; or caller-supplied `frame_offsets` don't match the file. This means `file_size`, `mtime_ns` or `natoms` differ from `stat` and the first header; or offsets are not strictly increasing from 0; or the last offset ≥ `file_size`. |
| E12 | protein and probe index sets overlap without `allow_index_overlap`. With the flag set, overlap is allowed: a shared atom lies at d2 = 0 from itself. That flag is the deliberate prereg §2.6 NC2 defect channel. |
| E13 | any of the six box off-diagonals has abs value > `ortho_tol_nm`. The kernel never silently drops shear, contrast `distances.rs:86-90`. |
| E14 | any box diagonal entry is not finite or ≤ 0. The kernel never falls back to Euclidean distance, contrast `distances.rs:132`. |
| E15 | `min(L_k) < 2·cutoff_nm` |
| E16 | the frame header is inconsistent or the decode is short. See the note below the table. |
| E17 | `times_ps` is not strictly increasing across the returned frames (minor 4). This catches a stale or foreign offset table that decodes cleanly. |
| E18 | the sparse output budget was exceeded during the run |
| E19 | the file's `frame_count` < max(`frame_indices`) + 1 (the bound part of E1, reported with the count) |

**E16 detail (minor 2).**
- molly's `read_header` enforces `natoms == natoms_repeated` with `assert_eq!` (`molly lib.rs:100`),
  which is a panic. Each frame's header read and decode are wrapped in `std::panic::catch_unwind`,
  and a panic maps to E16.
- After decode, the kernel asserts `frame.positions.len() == 3·|U|` and that every selected
  coordinate is finite. A mismatch is E16.
- A header whose `natoms` differs from `XtcOffsets.natoms` is E16.

---

## 3. Algorithm

1. **Validation.** Run the GIL-held validation of §2.4, then `py.allow_threads` for everything below
   (the pattern at `py_xtc_reader.rs:208, 367`).
2. **Offsets.**
   - **With caller offsets.** If `offsets` is `Some`, validate them against `stat` and the first
     frame header (E11), then use them.
   - **Without.** Otherwise, run `xtc_frame_offsets` in memory. It **never** reads or writes any
     sidecar or npz.
   - Record `offsets_source`.
   - **What the consumer does.** The consumer computes offsets once per file and stores them in its
     own shard directory. That keeps the cold header scan (R6) off every later run and keeps writes
     out of the naurmalade campaign directories (R7).
3. **Decode plan.**
   - `U = sorted(unique(protein ∪ probe))`, selected with `AtomSelection::Mask`
     (`molly selection.rs:35-48`). molly compacts the selected atoms in ascending order and stops at
     the last one (`reading_limit`, `selection.rs:105-112`).
   - `pos_in_U` maps caller order explicitly, which avoids the hazard at `py_xtc_reader.rs:36-42`.
4. **Parallel map over positions `0..F`.**
   - Iterate `(pos, rows_f)` pairs with `.using(|_| WorkerState::new(path))` (orx-parallel 2.4.0
     `par_iter.rs:469`) and an explicit `.num_threads(n_threads)` (`par_iter.rs:143`). That is
     applied on all targets; today it is applied only on wasm (`xtc.rs:666-667`).
   - `rows_f` is frame f's disjoint `&mut` row slices of the preallocated dense buffers (step 5).
   - **`WorkerState`** holds:
     - one `XTCReader<File>`, opened once per worker (today's `read_frames_parallel` opens one per
       frame, `xtc.rs:651`);
     - one `MollyFrame`;
     - one `ContactScratch`;
     - in test builds only, the `thread::current().id()`, which is recorded (§5.1 G-R10).
   - **Per frame:**
     1. If `pos > min_failed_pos`, skip the frame.
     2. **Clear `frame.positions`** (minor 3: `positions.clear()`) so values from the previous frame
        can never survive a short decode.
     3. Seek to the frame's offset and call `read_frame_at_offset::<BUFFERED>(...)`
        (`molly lib.rs:477-488`) inside `catch_unwind`. `BUFFERED = reading_limit < natoms`.
     4. Check E16, then the box (E13–E15; molly's box is column-major, `lib.rs:133-136`, and all six
        off-diagonals are checked).
     5. Gather coordinates via `pos_in_U`.
     6. Call `assign_frame`:
        - **Grid.** A periodic cell grid over P. Per axis,
          `n_k = floor(L_k / (cutoff_nm·(1 + 1e-4)))`, and if `n_k < 3`, set `n_k = 1`.
        - **Binning.** Wrap for binning only: `s = x/L − floor(x/L)`; if `s ≥ 1`, set `s = 0`;
          `cell = min(floor(s·n), n − 1)`.
        - **Storage.** A CSR built by counting sort, ordered by (cell, protein ordinal). The existing
          `CellList` (`cell_list.rs:15-63`) is not periodic and is not reused.
        - **Search.** For each probe atom, take the min key over stencil candidates with
          `d2 <= c2`. Fold atom keys into molecule keys. Write the labels and counts into `rows_f`.
        - **Sparse output.** Return a `FrameSparse`, then add its bytes to the budget (E18).
   - The map returns `(pos, time, box, FrameSparse)` or `(pos, error)`.
5. **Assembly.**
   - **Dense buffers.** These are allocated **once, before step 4**, at their exact size, and split
     with `chunks_exact_mut` into per-frame rows. Nothing dense is ever collected or copied.
   - **Fallback.** If orx-parallel 2.4.0 cannot iterate a `Vec` of disjoint `&mut` row tuples, the
     allowed fallback is to return compact per-frame rows (labels and counts only, no per-atom keys)
     and copy them in. The accounted peak then includes one extra dense copy, and
     `accounted_peak_bytes` must reflect it.
   - **Sparse results.** Sort by `pos`, then concatenate.
   - **Errors.** If any error occurred, return the one with the lowest `pos`.
   - Output is **byte-identical for any `n_threads` and any split of `frame_indices`**.
6. **Return.** Build numpy arrays from the owned `Vec`s (`PyArray1::from_vec_bound` + `reshape`, as
   at `py_xtc_reader.rs:391-395`).

**Invariants.** These are asserted in the tests and checked with `debug_assert!`:
- `partition_counts[f].sum() == (molecule_labels[f] >= 0).sum()`;
- `channel_counts[f].sum() == (atom_labels[f] >= 0).sum()`;
- a molecule is labelled if and only if one of its atoms is, and its label is one of those atoms'
  labels;
- `all_contact_counts[f].sum() == #pairs(f)`;
- every `(m, label)` pair is in `contact_pairs`.

**Cost.** I/O plus decode dominates. The distance work is about 1e6 evaluations per frame.

---

## 4. Secondary API: general periodic neighbour search — OUT OF SCOPE (r1 M4)

It was the optional Task 8 in r1. It is removed, because the consumer's probe–probe cluster-size
metric stays in consumer-side Python. If it is revisited:
- it reuses Task 1's grid and the §1.1 rules;
- it must not expose `cell_list::find_neighbors_within_cutoff_fast` or
  `neighbors::find_neighbors_within_cutoff`, whose cutoff semantics disagree (`cell_list.rs:116,143`
  use `<=` while its doc at line 131 says `<`; `neighbors.rs:66` uses `<`).

---

## 5. Correctness gates

**Rules for every gate.**
- All gates are machine-decidable (B1).
- A skipped test is a failure (B3). Python gates run with `PROXIDE_REQUIRE_MDTRAJ=1`; mdtraj comes
  only from the `trajectories` extra (`pyproject.toml:66-71`).
- **Non-vacuity (review r1 M1).** Every parity gate first computes coverage **from decoded data** and
  asserts minimums before it compares any outputs. A comparison over a set that turned out empty
  fails.
- Tests run one file or filter at a time, never the whole suite.

### 5.1 Rust unit tests

These use hand-constructed boxes with f32-exact coordinates.

- **G-R1 PBC wrap.** L=3.0 nm, cutoff 0.5.
  - Face case: protein at x=0.05, probe at x=L−0.05, so d=0.1 → contact.
  - Edge (2 axes) and corner (3 axes) variants.
  - Offsetting coordinates by +3L and −2L gives identical labels.
- **G-R2 exact cutoff equality.** cutoff 0.5.
  - Probe at 0.25 and protein at 0.75 give d2 = c2 exactly → contact.
  - Moving the protein one f32 ulp further gives no contact.
- **G-R3 ties.** These use axis-symmetric construction, so the ties are exact:
  - probe at `(c, c, z)`, residue 7 atom at `(c+a, c, z)`, residue 3 atom at `(c, c+a, z)` → d2 is
    bit-identical → label 3;
  - same residue, different atom → the lower full index wins;
  - permuting `protein_full_idx` leaves the result unchanged.
- **G-R4 empty.** No probe atom within the cutoff → all labels `NO_CONTACT`, all counts 0, no pairs.
- **G-R5 partition vs all-contacts.** A molecule touching residues 2 and 5 counts once in
  `partition_counts` and in both residues of `all_contact_counts`.
- **G-R6 grid ≡ brute force.** 500 seeded configurations.
  - Every `n_k ∈ {1, 2→1, 3, 7}` must occur. `n_k = 1` is reached at `L ∈ [2c, 2c·1.0001)`.
  - Atoms are placed at 0, L, −L and within ±1 ulp of cell boundaries.
  - A counter asserts that each `n_k` class occurred at least 20 times.
- **G-R7 shift.** `probe_shift_nm` gives the same result as pre-shifting the coordinates in f64.
- **G-R8 validation.** One test per E-row reachable in Rust. This includes E16 via a crafted header
  mismatch (a byte-patched copy of a fixture with `natoms_repeated` changed) and E17 via an offset
  table with two entries swapped.
- **G-R9 subset decode.** On every fixture in `tests/data/trajectories/`, Mask decode in both
  `BUFFERED` modes equals All-decode followed by selection, bit-for-bit.
  - Masks tested: a prefix, a sparse mask, and a mask ending at `natoms−1`.
  - Includes a decode into a reused `MollyFrame` after a larger frame, to catch stale positions.
- **G-R10 determinism.** On `tests/data/trajectories/frame0.xtc` (22 atoms, 501 frames, box about
  2.57 nm), **cutoff_nm = 0.5**.
  - Split: atoms 0–11 are protein (residue ids from the fixture PDB); atoms 12–21 form two probe
    molecules of 5 atoms each.
  - **Non-vacuity:** assert `partition_counts.sum() > 0`, `all_contact_counts.sum() > 0`, and at
    least 1 frame where `molecule_labels` differ across the two molecules.
  - Outputs are byte-identical for `n_threads ∈ {1, 2, 4, 8}` and for **different splits** (minor 8):
    one call over all frames equals the concatenation of calls over `[0, 137)` + `[137, 501)` and
    over three uneven thirds.
  - **Thread count honoured (minor 9):** the recorded distinct worker thread ids number ≤
    `n_threads`, and exactly 1 when `n_threads = 1`.
  - Also asserts `meta.accounted_peak_bytes` ≥ the sum of returned buffer sizes (M3).

**Gates** (minor 1). Each is one command with one filter:
- `cargo test -p proxide-geometry periodic_grid`
- `cargo test -p proxide-geometry probe_contacts`
- `cargo test -p proxide-io --features xtc,parallel xtc_subset_decode`
- `cargo test -p proxide-io --features xtc,parallel xtc_frame_offsets`
- `cargo test -p proxide-io --features xtc,parallel xtc_probe_contacts`

### 5.2 Python parity (Task 6): `tests/validation/test_probe_contacts_parity.py`

**Reference.** `tests/validation/_probe_contacts_reference.py` is pure numpy with no proxide import.
It is a brute force chunked over probe atoms, using the §1.1 arithmetic.

**Synthetic XTC 1 ("main").** Written in the test with `mdtraj.Trajectory.save_xtc`
(construction as `bench_xtc_decode.py:44-96`). It has:
- about 400 protein atoms in 60 residues;
- 30 probe molecules × 15 atoms in 5 channels;
- water filler;
- 60 frames with a varying orthorhombic box;
- planted face, edge, corner and axis-symmetric tie geometries.

**Synthetic XTC 2 ("grid").** A few atoms, with per-frame boxes chosen so that
`n_k ∈ {1, 2→1, 3, ≥7}` each occur, with `L ≥ 2c` throughout.

**Coordinate source.** The reference uses `_proxider._read_xtc_frames_nm` (raw molly nm f32; Task 5).
Existing readers return `x*10` in f32 (`py_xtc_reader.rs:233`), which is lossy.

**Coverage minimums (M1).** The reference computes these from the **decoded** coordinates, and the
test asserts them before comparing anything:

| coverage item | minimum |
|---|---|
| contacting molecules per frame | ≥ 5 in ≥ 90% of frames |
| cross-face contacts (MIC shifted exactly 1 axis) | ≥ 20 pairs |
| edge/corner contacts (2 or 3 axes shifted) | ≥ 5 pairs |
| surviving **bit-exact** d2 ties that decide a label between different residues | ≥ 10 (atom- or molecule-level) |
| molecules with ≥ 2 residues in all-contacts | ≥ 10 (frame, molecule) instances |
| `n_k` classes in XTC 2 | each of {1, 2→1, 3, ≥7} in ≥ 1 frame |

**If the tie minimum cannot be met.** The axis-symmetric ties survive XTC quantisation because
equal stored integers decode to equal floats. If that nonetheless fails to reach the tie minimum on
decoded data, the test fails. The fix is **then** to drop the claim that ties are verified through
the full stack, with a written spec amendment. Ties would remain covered by G-R3 only. Lowering the
minimum is not allowed.

**Assertions** (all exact).
- **G-P1.** All label, count and `contact_pairs` arrays are `np.array_equal` to the reference.
- **G-P2.** Decode identity: `np.array_equal(mdtraj.load(...).xyz, raw_nm)`. If that holds, G-P1
  also holds on the mdtraj-fed reference. If it does not, the test **fails** (no tolerance). The
  fixer reports the max ULP delta for review.
- **G-P3.** Byte-identical results for `n_threads ∈ {1, 2, 4, 8}` and for two frame splits.
- **G-P4.** One `pytest.raises` per row E1–E19. XTC 1 variants provide the triclinic (E13) and zero
  box (E14) cases. A mismatched or swapped `frame_offsets` provides E11 and E17. `cutoff_nm=4.5`
  provides E7.
- **G-P5.** The §3 invariants hold on every frame.
- **G-P6. The kernel writes no files.** The directory listing around the XTC is identical before and
  after any call. `xtc_frame_offsets(path)["offsets"]` equals molly's `determine_offsets` (via the
  Rust test in §5.1).

**Gate:**
`PROXIDE_REQUIRE_MDTRAJ=1 uv run --extra trajectories pytest tests/validation/test_probe_contacts_parity.py -q -rs`.
It must exit 0 with 0 skipped.

### 5.3 Real-data parity (consumer-side; spec only, executed in sweetprots)

The script is `sweetprots/scripts/analysis/sweet_probe_contacts_parity.py`, with a bathos sidecar.

**Input.** One replicate of `vft_apo-stripped_kd7p8A_MND_his-neutral`, 50 frames at the prereg
stride 5 (raw frames 0, 5, …, 245). It runs as a 1-task `sbatch`, or locally on the Task 0b excerpt
if that excerpt holds ≥ 246 frames.

**Arm K (kernel).** The consumer's production call, with `cutoff_nm = 0.45`, the consumer's atom
sets and its offsets file.

**Arm R (shared-parameter reference).** mdtraj decode of U fed into the §5.2 numpy reference. It
uses the same inputs as K.

**Arm L (literal prereg path; review r1 M2).** This arm shares no caller parameters with K. It is
the prereg §2.1 implementation as written:
- its **own** topology selections from `production.pdb` by resname (protein heavy with the probe
  resname excluded, and probe heavy);
- its own residue map;
- `mdtraj.iterload(stride=5)` with **its own striding**;
- coordinates × 10 into Å;
- `scipy.spatial.cKDTree(boxsize=L_Å)`;
- the **literal cutoff 4.5 Å**.

L's labels are mapped to K's residue ids through topology atom indices.

**Checks.**
- **Decode identity between K and R.** Arrays must be identical.
- **K vs R.** Everything must be identical.
- **K vs L.** Compare `molecule_labels` and `atom_labels` and report `n_label_mismatch`, with every
  mismatch classified:
  - **boundary:** either implementation's minimum contact distance for that atom or molecule is
    within `|d − 4.5 Å| < 1e-5 Å`;
  - **real:** anything else. This includes near-ties resolved differently, which are not excused.

**Pass rule.** `n_decode_mismatch == 0`, K ≡ R, **`n_real == 0`**, and one repeat of K is
byte-identical. Boundary mismatches are allowed and reported as `n_boundary`.

**Outcome.** Write `parity_pass`, `n_label_mismatch`, `n_boundary`, `n_real`, `n_decode_mismatch`
and `max_abs_decode_delta_nm`. If `parity_pass = 0`, the kernel **must not** be used for prereg
Task 2.

### 5.4 Assumptions and where they are tested

| assumption | gate |
|---|---|
| orx-parallel honours `num_threads` on native | G-R10 thread-id count and T-B3 |
| molly Mask decode equals All-decode | G-R9 |
| molly decode equals mdtraj decode | G-P2 and §5.3 |
| no Å/nm mix-up | E7 and §5.3 arm L |

---

## 6. Benchmark (Task 7): `benchmarks/bench_probe_contacts.py` + `bench_probe_contacts.py.bth.toml`

The script follows the `bench_xtc_decode.py` pattern:
- thread env caps (lines 31-33);
- a tempdir fixture that is never committed;
- JSON output.

**Representative synthetic system (review r1 M4).**
- **Size and box.** About 240k atoms: 8k protein heavy atoms in ~1000 residues, 300 probe molecules
  × 21 heavy atoms, water to 240k. Cubic box **12.8249 nm**, the measured production box.
- **Two atom orders**, since the real order is unknown (U6): `probe_after_water`, the worst case for
  molly's early stop, and `probe_before_water`.
- **Writing the fixture.** It is written in chunks with `mdtraj.formats.XTCTrajectoryFile`, never
  as one in-memory array.
- **Frame counts.** 200 frames by default; `--smoke` uses 10 frames of the full-size system plus
  the existing 15k-atom pattern, and must finish in under 60 s.

**Arms** (all over the same frames):
1. **`reference_mdtraj`** — the prereg §2.1 implementation: `iterload` + scipy `cKDTree`. scipy is
   required and the script errors if it is missing. Run with
   `uv run --extra trajectories --with scipy python benchmarks/bench_probe_contacts.py`.
2. **`proxide@{1, 4, 8}`** — run with caller-supplied offsets (warm) and with an in-memory scan
   (cold).

**Reported per arm.**
- `frames_per_s` (median/min/max over 3 repeats);
- `/proc/self/io` `read_bytes`;
- `reading_limit / natoms`.

**Targets** (none is claimed until measured; they do not gate merge):

| target | condition |
|---|---|
| T-B1 | `proxide@8 (warm) ≥ 5 × reference_mdtraj` |
| T-B2 | `proxide@1 (warm) ≥ reference_mdtraj` |
| T-B3 | `proxide@8 / proxide@1 ≥ 3` |

**T-B4 (production, consumer-measured on Engaging, minor 11).**
- **Warm.** Caller-supplied offsets: 2000 frames (stride 5) of one ~9.5 GB / 10k-frame file in
  **< 3 min** on 8 cores, direct from pool. This is conditional on `dd`-measured pool throughput ≥
  20 MB/s in the same job; 1.9 GB at 20 MB/s is about 95 s.
- **Cold.** The one-time `xtc_frame_offsets` scan is reported separately, as wall time plus
  `read_bytes`, and is not counted against T-B4.

**Gate for Task 7.**
- `uv run --extra trajectories --with scipy python benchmarks/bench_probe_contacts.py --smoke --json-out /tmp/bpc.json`
  exits 0.
- The JSON has every arm and key above.
- `bth validate-sidecar benchmarks/bench_probe_contacts.py.bth.toml` exits 0.

---

## 7. Fixer tasks

**Dependencies:** T0 decides whether any of T1–T8 run. Then T1 → T2 → T4; T3 → T4; T4 → T5 → {T6,
T7}; T6 → T8.

### Task 0: measure the reference path on one real replicate (orchestrator, Engaging; IN PROGRESS)

**What is measured.** Run the prereg §2.1 reference implementation (staging + `iterload(stride=5)` +
`cKDTree`), as planned for prereg Task 2, on one real replicate. Use the full 2000 strided frames,
4 CPU and `mit_normal`. Record:

| quantity | value |
|---|---|
| file size (GB), raw frames, natoms, atom order (probe before/after water) | _pending_ |
| staging copy wall (min) | _pending_ |
| decode + count wall (min) | _pending_ |
| total wall per replicate (min) | _pending_ |
| peak RSS (GB) | _pending_ |
| pool throughput from `dd` in the same job (MB/s) | _pending_ |
| projected 68-replicate cost (core-h), against the prereg's 150–400 expected / ≤ 1100 worst | _pending_ |

**Go/no-go rule, stated before the numbers.** Build T1–T8 **only if at least one** of these holds:
- **G1.** The reference total wall is ≥ 90 min per replicate. Such runs risk the 4 h cap under a
  degraded pool, and 68 replicates cost ≥ 400 core-h.
- **G2.** Staging is ≥ 50% of the reference wall. Direct strided reads then remove most of the cost,
  **provided the consumer adopts direct pool reads**.
- **G3.** The reference exceeds 16 GB RSS or fails outright.

**Otherwise NO-GO.** The consumer runs the prereg reference path unchanged, this spec stays `draft`,
and the result is recorded in the disposition.

**Gate.** The table is filled in with the job id, and the verdict (`GO:G1|G2|G3` or `NO-GO`) is
written in this section before T1 starts.

### Task 1: periodic cell grid + pinned MIC (`proxide-geometry`)
- `mic_d2`, implementing §1.1.
- `PeriodicCellGrid::build(points, box, min_cell_nm) -> Result<_, GridError>`, with the §3 cell
  counts and a CSR layout.
- `for_each_candidate`.

Tests: G-R1, G-R2 and G-R6.

**Files:** `crates/proxide-geometry/src/geometry/periodic_grid.rs` (create), `geometry/mod.rs`
**Gate:** `cargo test -p proxide-geometry periodic_grid`
**Scope:** ~250 LOC

### Task 2: per-frame core
Implement §2.1 (`FrameRows`, scratch-only keys). Tests: G-R1 through G-R5 and G-R7 at the frame
level.

**Files:** `crates/proxide-geometry/src/geometry/probe_contacts.rs` (create), `geometry/mod.rs`
**Gate:** `cargo test -p proxide-geometry probe_contacts`
**Scope:** ~350 LOC

### Task 3: subset decode + uncached offsets (`proxide-io/xtc.rs`)
- `SubsetPlan` and `decode_subset_at`, which clears positions, wraps the decode in `catch_unwind`
  and checks the length (E16).
- `XtcOffsets` and `xtc_frame_offsets`, with no sidecar I/O.
- Existing `XtcReader` behaviour is unchanged.

Tests: G-R9 and the offsets-equal-molly test.

**Files:** `crates/proxide-io/src/formats/xtc.rs`, `formats/tests/xtc_tests.rs`
**Gates:**
- `cargo test -p proxide-io --features xtc,parallel xtc_subset_decode`
- `cargo test -p proxide-io --features xtc,parallel xtc_frame_offsets`
- the existing `cargo test -p proxide-io --features xtc,parallel xtc`, which must stay green

**Scope:** ~150 LOC

### Task 4: fused driver
Implement §2.2 and §3:
- validation, split per the §2.4 timing rules;
- preallocated rows, the E18 budget and the lowest-pos error;
- `using` + `num_threads`;
- thread-id recording under `cfg(test)`;
- the `debug_assert!` peak accounting.

Tests: G-R8 and G-R10.

**Files:** `crates/proxide-io/src/formats/xtc_probe_contacts.rs` (create), `formats/mod.rs`,
`formats/tests/probe_contacts_tests.rs` (create)
**Gate:** `cargo test -p proxide-io --features xtc,parallel xtc_probe_contacts`
**Scope:** ~500 LOC

### Task 5: bindings + wrapper + raw-nm hook
- `_probe_residue_contacts` and `_xtc_frame_offsets`.
- Test-only `_read_xtc_frames_nm`, not in `__all__`.
- `src/proxide/probe_contacts.py` and the re-export in `__init__.py`.

**Files:** `crates/proxide_py/src/py_probe_contacts.rs` (create), `crates/proxide_py/src/lib.rs`,
`src/proxide/probe_contacts.py` (create), `src/proxide/__init__.py`,
`tests/test_probe_contacts_api.py` (create)
**Gate:** rebuild the extension (`uv pip install -e ".[dev]"` per the README, or `maturin develop`;
U13), then `uv run pytest tests/test_probe_contacts_api.py -q`
**Scope:** ~250 LOC Rust + ~150 LOC Python

### Task 6: Python parity
Implement §5.2 with both synthetic XTCs, the coverage minimums and G-P1 through G-P6.

**Files:** `tests/validation/_probe_contacts_reference.py`,
`tests/validation/test_probe_contacts_parity.py`
**Gate:** the §5.2 command, with 0 skipped
**Scope:** ~450 LOC

### Task 7: benchmark + sidecar
Implement §6. **Gate:** as in §6. **Scope:** ~350 LOC

### Task 8 (consumer; sweetprots, not this worktree)
- Implement §5.3 with its three arms.
- Record A1–A3, the kernel swap, and the switch from staging to direct pool reads with stored
  offsets as a prereg disposition note, **before** prereg Task 2.

**Gate:** `parity_pass = 1`

---

## 8. Risks

| # | Risk | Mitigation |
|---|---|---|
| R1 | molly vs xdrfile decode differs in edge cases. Existing tests only check to 0.01 Å (`test_xtc_distogram_parity.py:46`). | G-P2 and §5.3 are bit-exact and fail loud. |
| R2 | molly's Mask and `BUFFERED` paths are unused in proxide today (`xtc.rs:473, 496, 653`). | G-R9 |
| R3 | Stale or foreign caller offsets | E11 (size, mtime and natoms against `stat` and the header), E16 (`catch_unwind` on header panics, length check), E17 (monotonic times) |
| R4 | Triclinic boxes | Out of scope; production boxes are measured cubic, 128.249 Å. E13 fails loud on every frame. |
| R5 | GIL release: Ctrl-C does not interrupt a running call | Accepted. The consumer chunks `frame_indices` at about 200 frames per call. |
| R6 | Cold header scan on the network FS triggers read-ahead amplification | The kernel accepts caller offsets; the scan is one-time and measured separately (T-B4 cold). |
| R7 | Writing into naurmalade campaign directories | The kernel writes no files (G-P6). Offsets live in the consumer's own directory. |
| R8 | Probe atoms after water mean no early stop | Benchmarked in both orders. Task 0 records the real order. |
| R9 | Semantics drift (A1–A3) | Resolved by the prereg disposition note before prereg Task 2. |
| R10 | Wasted build if the reference path is already adequate | Task 0 go/no-go rule |
| R11 | Unit error (Å vs nm) | E7 (`cutoff_nm > 1.0` refused) and §5.3 arm L |
| R12 | Rollback | The change is additive. Existing `XtcReader` behaviour is unchanged. Revert the branch. |

---

## 9. Assumptions not verified in code (flagged)

- **U1.** Ordered collect in orx-parallel. The design does not rely on it: assembly sorts by `pos`.
- **U2.** Native `num_threads` is honoured. Checked by G-R10 thread ids and T-B3.
- **U3.** molly Mask decode equals All-decode. Checked by G-R9.
- **U4.** molly decode equals mdtraj decode. Checked by G-P2 and §5.3.
- **U5.** Box shape. **Resolved:** cubic 128.249 Å, measured by the orchestrator 260923.
- **U6.** Atom order in `production.pdb`. Task 0 records it.
- **U7.** Channel semantics (A1 in both parts, A2).
- **U8.** Pool throughput at run time.
- **U9.** Resolved by design: no writes into campaign directories.
- **U10.** Box orientation. molly's comment says column-major (`lib.rs:133-136`); the proxide docs
  say row vectors (`xtc.rs:15-26`). Irrelevant here, since E13 checks all six off-diagonals.
- **U11.** A truncated trailing frame is dropped (`xtc.rs:466-480`). mdtraj may count it; E19 covers
  that.
- **U12.** Consumer sizes (M, Q, R, natoms) are estimates. Task 0 measures them.
- **U13.** The fastest extension rebuild command.
- **U14.** Whether orx-parallel 2.4.0 accepts a `Vec` of disjoint `&mut` row tuples as a parallel
  source. There is a pinned fallback in §3 step 5.
- **U15.** Whether `catch_unwind` is sound across molly's `read_header` panic. The reader state after
  a panic is discarded, since the worker's reader is re-opened.

## 10. Out of scope

- Triclinic boxes.
- H-inclusive or other-unit all-contacts (A1), and group-instance channels (A2).
- `atom_keys` output (dropped in r2).
- The general neighbour-search Python API (was r1 Task 8).
- MDAnalysis npz offset import, and any sidecar read or write by this kernel.
- Probe–probe clustering, relative SASA, numbering and UniProt mapping, and group perception.
- Frame equalisation, bootstrap and JSD.
- DCD and TRR inputs, and GPU.
- Changing `pairwise_distances_mic`'s silent Euclidean fallback (`distances.rs:132`),
  `BoxDims::from_diagonal_matrix`'s silent shear drop, or the `cell_list.rs` `<` / `<=` mismatch.
  File these as debt.

## References

- Consumer prereg: `/home/marielle/projects/sweetprots/.claude/worktrees/wt-20260831-162012/.praxia/docs/preregistration/260923_chirality-site-preference-and-mi-at-site.md` §2.1 (128-197), §2.3–2.4 (209-250), §2.6 (263-300), §6 (1118-1259), §9 (1299-1322)
- `crates/proxide-io/src/formats/xtc.rs` (offset cache 137-292, `XtcReader` 305-536, tolerant scan 428-480, `read_frames_parallel` 634-669, distogram 695-724)
- `crates/proxide-geometry/src/geometry/{cell_list.rs, neighbors.rs, distances.rs}`
- `crates/proxide_py/src/py_xtc_reader.rs` (195-276, 329-405); `crates/proxide_py/src/lib.rs:108-125`; `crates/proxide_py/Cargo.toml:43-50`
- molly 0.6.1 `src/{selection.rs, lib.rs}`; orx-parallel 2.4.0 `src/par_iter.rs` (versions from `Cargo.lock`)
- `benchmarks/bench_xtc_decode.py`; `tests/validation/test_xtc_distogram_parity.py`
- `/home/marielle/projects/proxide/CLAUDE.md`
- Review r1: `.praxia/audits.jsonl`, record `260923_proxide-probe-residue-contacts_spec_challenge_1`

---

## Review r1 disposition

**Verdict:** NEEDS_REVISION (0 BLOCKER, 4 MAJOR, 11 MINOR). All items are accepted.

### Majors

| # | Finding | Change |
|---|---|---|
| M1 | Parity gates had no non-vacuity coverage | §5.2 coverage-minimum table, computed on decoded data and asserted before comparison. This covers contacting molecules per frame, cross-face and edge/corner PBC contacts, surviving bit-exact ties (made reachable by axis-symmetric construction; if unmet, the full-stack tie claim is dropped by amendment and the minimum is not lowered), multi-residue all-contacts molecules, and `n_k ∈ {1, 2→1, 3, ≥7}` via a second synthetic XTC. G-R6 counts `n_k` classes. G-R10 states `cutoff_nm = 0.5` and asserts nonzero counts. |
| M2 | All references shared the caller's parameters | §5.3 arm L is the prereg's literal path: its own resname selections, stride, Å units and literal 4.5 Å cutoff. Mismatches are classified boundary (`|d − 4.5 Å| < 1e-5 Å`) or real, and only boundary is allowed. E7 refuses `cutoff_nm > 1.0` without `allow_large_cutoff`. |
| M3 | Memory claim was false | Per-atom keys are scratch-only. Dense rows are preallocated and written in place. The dense bound is exact (E10); the sparse size is enforced as it grows via an atomic budget (E18). `min(R, 32)` is withdrawn as a bound. The `atom_keys` term is moot because the output was removed. Peak is restated as output + 2·sparse + per-thread transient, reported as `accounted_peak_bytes` and `debug_assert!`ed. |
| M4 | Scope was not justified by measurement | New Task 0 has placeholders and a pre-stated go/no-go rule (G1/G2/G3). Direct pool reads with caller offsets are chosen explicitly over staging. The old Task 8 (neighbour API), `atom_keys` and the MDA-npz path are out of scope. The benchmark system is representative: ~240k atoms, the measured cubic box, both atom orders. |

### Minors

| # | Change |
|---|---|
| 1 | One cargo filter per command |
| 2 | E16 via `catch_unwind` around the header read and decode, plus `positions.len() == 3·|U|` |
| 3 | `positions.clear()` per frame; G-R9 covers a reused buffer |
| 4 | E17 requires strictly increasing times; no MDA npz is ever read |
| 5 | Caller-supplied `XtcOffsets`; no sidecar I/O |
| 6 | §2.4 states which checks run with the GIL held and which run after release |
| 7 | The lowest failing `pos` is reported via an atomic min |
| 8 | G-R10 uses different frame splits instead of reversed chunking |
| 9 | Worker thread ids must number ≤ `n_threads` (exactly 1 at `n_threads = 1`) |
| 10 | A1 covers the counted unit as well as the atom set; the bincount notation is fixed in §1.1 |
| 11 | T-B4 is split into warm (caller offsets) and cold (one-time scan) |

### Facts incorporated

Production boxes are cubic, 128.249 Å. U5 is resolved, and the benchmark box uses this size.
