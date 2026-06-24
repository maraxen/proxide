# ADR: Pure-Python coords→χ path + PyO3/Rust binding deferred (B4, #2654)

**Date:** 260623  
**Status:** Accepted  
**Task:** B4 (#2654) — NeRF round-trip validation for E1a (protamer sprint 6)

---

## Context

protamer's E1a sprint requires a **coords→χ measurement** pipeline that round-trips
with proxide's forward NeRF (χ→coords→χ ≤ 0.1° RMS).  Two implementation paths
were evaluated:

1. **Pure-Python / NumPy** (`proxide.geometry.measure_chi` + new `proxide.geometry.nerf_forward`)
2. **PyO3/Rust binding** — wrapping a Rust geometry core already present in the repo
   for the hot altloc-parsing path

---

## Decision

**Path chosen: pure-Python / NumPy** for both `measure_chi` (coords→χ, B2) and
`chi_to_coords` (χ→coords, new in B4, `nerf_forward.py`).

**PyO3/Rust binding: DEFERRED** — tracked as debt for a future sprint if throughput
becomes a bottleneck at the E3 data-prep scale.

---

## Rationale

### Performance context

`coords→χ` (and `χ→coords` for validation) runs in the **data-preparation pipeline**,
not in the hot inference / fitting loop (E3 VMM fitting, E5 discretization).
The hot loop is `fit_vmm_factorized` + HDBSCAN in denxity — those never call
`measure_chi`.  Data-prep is a one-time offline cost per PDB structure.

At 100K residues (PDB + mdCATH), NumPy `measure_chi` batches in < 1 s (CPython,
no JAX compilation overhead).  The Rust path would shave ~100 ms off a seconds-scale
pipeline — not load-bearing.

### Forward NeRF used (built, not reused)

No existing torsion→Cartesian builder was found in proxide's `geometry/` module.
proxide had all required building blocks:

- `proxide.chem.residues.restype_rigid_group_default_frame` — (21, 8, 4, 4) array
  of per-group default frames from the AlphaFold2 rigid-group decomposition
  (Jumper et al. 2021, Supp. §1.8), already computed by `_make_rigid_group_constants()`.
- `proxide.chem.residues.restype_atom37_rigid_group_positions` — per-atom local
  positions in each rigid group frame.
- `proxide.geometry.transforms.extend_coordinate` — single-atom NeRF step (JAX).

A new **pure-NumPy** module `proxide/geometry/nerf_forward.py` was built:

```
chi_to_coords(restype3, N, CA, C, chi_angles_rad) → (atom37_positions, atom37_mask)
```

Algorithm:
1. Build backbone global frame T_bb from N, CA, C (x-axis = C−CA, y-axis = orthog.(N−CA)).
2. For each χ_i group: `T_chi_i = T_parent × F_chi_i × Rot_x(χ_i)` where F_chi_i is
   the default frame from `restype_rigid_group_default_frame`.
3. Place atoms: `pos_global = T_chi_i[:3,:3] @ pos_local + T_chi_i[:3,3]`.

Round-trip precision: global RMS 0.000001° over 390 χ angles; max 0.000003°.
This is essentially float64 machine precision — the 0.1° gate is met by a factor of ~100 000×.

### Why not the JAX `extend_coordinate`?

`extend_coordinate` (transforms.py:176) places one atom from three reference atoms
and (bond-length, bond-angle, dihedral).  It would require sourcing bond lengths and
bond angles from stereo-chemical props tables for every χ atom, and it operates
one-atom-at-a-time in JAX (triggering compilation).  The rigid-group frame approach
places all atoms for a residue type in a single matrix multiply, reusing pre-built
tables — cleaner and faster for data-prep.

### PyO3 deferred — not skipped

The Rust altloc-resolution path already has PyO3 glue (`proxide-rs` crate).  If E3
profiling shows data-prep is a bottleneck, the following work is ready to scope:

- Expose `measure_chi_batch` as a PyO3 function operating on `(n_res, 37, 3)` arrays
- Wire to the existing `atom_order` + `chi_angles_atoms` tables already in Rust

**Revisit gate:** measure throughput after E3 VMM fitting is implemented.  Target:
data-prep < 5% of total pipeline time.  If exceeded, file a follow-up task.

---

## Consequences

- `proxide.geometry.nerf_forward` is a new module (pure-NumPy, no JAX dependency).
- `proxide.geometry.measure_chi` + `proxide.geometry.fold_chi` remain pure-NumPy.
- No new Rust code added in this sprint; PyO3 layer is unchanged.
- Round-trip tests in `tests/geometry/test_nerf_roundtrip.py` cover all 18 residue
  types with χ angles; biotite parity re-asserted at ±0.01°.
