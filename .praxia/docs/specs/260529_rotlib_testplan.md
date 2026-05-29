---
name: 260529_rotlib_testplan
description: Test plan for proxide-rotlib — 55 unit + 4 integration tests across 6 modules
metadata:
  type: reference
---

# Test Plan: `proxide-rotlib`

**task_id**: `260529_confind_rotlib`
**Spec oracle PASS**: `260529_rotlib.md @ 268e002`
**Real fixture**: `/home/marielle/repos/mosaist/testfiles/rotlib.bin` (check path below)
**AA_NAMES** (18): ALA ARG ASN ASP CYS GLN GLU HIS ILE LEU LYS MET PHE SER THR TRP TYR VAL

---

## Shared Fixture Infrastructure

**Location**: `tests/helpers.rs` (included via `#[path = "helpers.rs"] mod helpers;`)

**`fn write_minimal_lib(aa: &str, bins: &[BinSpec], rotamers_per_bin: &[RotSpec]) -> tempfile::NamedTempFile`**
Writes a well-formed single-AA binary blob. `BinSpec { phi: f32, psi: f32, freq: f32 }`, `RotSpec { prob: f32, coords: Vec<[f32;3]> }`. Atom names default to `["CB"]` (na=1, nc=0).

**`fn real_rotlib() -> RotamerLibrary`** (only for `#[ignore]` tests)
Reads `ROTLIB_PATH` env var, else `/home/marielle/repos/mosaist/testfiles/rotlib.bin`. Panics if absent.

**Dev-dependencies to add**:
```toml
[dev-dependencies]
tempfile = "3"
```

---

## Module 1: `tests/test_load.rs`

| # | Test | Kind | Input | Expected |
|---|------|------|-------|----------|
| 1 | `test_load_all_aa_names_present` | integration `#[ignore]` | real rotlib.bin | `contains_aa(aa)` true for all 18 AAs |
| 2 | `test_num_rotamers_sentinel_positive` | integration `#[ignore]` | real rotlib.bin, phi/psi=9999.0 | `num_rotamers > 0` for each of 18 AAs |
| 3 | `test_load_truncated_file` | unit | `write_minimal_lib(...)` truncated mid-rotamer | `Err(RotlibError::Io(_))` |
| 4 | `test_load_non_rectangular_grid` | unit | 3 bins: (-60,-60), (-60,60), (60,-60) → unique_phi=2, unique_psi=2, product=4≠3 | `Err(RotlibError::InvalidFormat(_))` |
| 5 | `test_load_duplicate_phi_psi` | unit | 2 bins both with phi=-60, psi=-40 | `Err(RotlibError::InvalidFormat(_))` |

**Test 4 detail**: bins = (-60,-60,1.0), (-60,60,1.0), (60,-60,1.0). unique_phi={-60,60} len=2; unique_psi={-60,60} len=2; 2×2=4 ≠ nb=3 → `InvalidFormat`.

---

## Module 2: `tests/test_binning.rs`

**Fixture**: 1-AA library "TST", 3×3 grid: phi_centers={-120,0,120}, psi_centers={-120,0,120}, 9 bins. `default_bin=4` (phi=0,psi=0) with freq=999.9, all others 1.0.

`bin_index = phi_ind * 3 + psi_ind`

| # | Test | Input | Expected |
|---|------|-------|----------|
| 1 | `test_binning_sentinel_phi` | phi=9999.0, psi=-120.0 | `Ok(4)` |
| 2 | `test_binning_sentinel_psi` | phi=-120.0, psi=9999.0 | `Ok(4)` |
| 3 | `test_binning_sentinel_both` | phi=9999.0, psi=9999.0 | `Ok(4)` |
| 4 | `test_binning_exact_center` | phi=0.0, psi=0.0 | `Ok(4)` = 1×3+1 |
| 5 | `test_binning_known_corner` | phi=-120.0, psi=-120.0 | `Ok(0)` = 0×3+0 |
| 6 | `test_binning_known_edge` | phi=120.0, psi=0.0 | `Ok(7)` = 2×3+1 |
| 7 | `test_binning_near_center_offset` | phi=40.0, psi=-40.0 | `Ok(4)` (nearest 0°) |
| 8 | `test_binning_wrap_around_pos` | phi=179.9, psi=0.0 | `Ok(7)` (nearest 120°) |
| 9 | `test_binning_wrap_around_neg` | phi=-180.0, psi=0.0 → equidist from -120° and 120°; first=-120° | `Ok(1)` = 0×3+1 |
| 10 | `test_binning_tie_lower_bin_wins` | phi=60.0, psi=0.0 → equidist 0° and 120°; first=0° (ind=1) | `Ok(4)` = 1×3+1 |
| 11 | `test_binning_unknown_aa` | aa="ZZZ" | `Err(RotlibError::UnknownAa(_))` |

---

## Module 3: `tests/test_probability.rs`

**Fixture**: Same 9-bin "TST" library. Each bin has 2 rotamers. Bin 4: probs=[0.6, 0.3]. Bin 0: probs=[0.5, 0.4].

| # | Test | Input | Expected |
|---|------|-------|----------|
| 1 | `test_prob_sum_le_one_per_bin` | all 9 bins | sum ≤ 1.0 each |
| 2 | `test_prob_by_id_matches_direct` | phi=0,psi=0 vs RotamerId{bin=4,rot=0} | both return 0.6 |
| 3 | `test_prob_sentinel_bin` | phi=9999,psi=9999, rot=1 | `Ok(0.3)` (default_bin=4, second) |
| 4 | `test_prob_oob_rot_index` | aa="TST", rot=99, phi=0, psi=0 | `Err(RotIndexOob("TST", 99, 2))` |
| 5 | `test_prob_unknown_aa` | aa="ZZZ" | `Err(UnknownAa("ZZZ"))` |
| 6 | `test_prob_by_id_unknown_aa` | RotamerId{aa:"ZZZ"} | `Err(UnknownAa("ZZZ"))` |
| 7 | `test_prob_by_id_oob_rot_index` | RotamerId{aa:"TST",bin=4,rot=99} | `Err(RotIndexOob("TST",99,2))` |
| 8 | `test_prob_sum_real_rotlib` `#[ignore]` | real rotlib.bin, all 18 AAs | sum ≤ 1.0 per bin |

---

## Module 4: `tests/test_placement.rs`

**Unit fixture**: hand-crafted lib with ALA (na=1, atom=["CB"]) and ARG (na=6, atoms=["CB","CG","CD","NE","CZ","NH1"]), 1-bin grid, nr=1. CB coord stored as [1.0,0.0,0.0].

**Backbone for tests**: N=[0,0,0], CA=[1.458,0,0], C=[1.980,1.418,0].

| # | Test | Input | Expected |
|---|------|-------|----------|
| 1 | `test_place_ala_one_atom` | ALA, rot=0 | `atoms.len()==1`, name=="CB" |
| 2 | `test_place_arg_atom_count` | ARG, rot=0 | `atoms.len()>=5` (6 in fixture) |
| 3 | `test_place_no_backbone_atoms` | ARG, rot=0 | none of atoms has name in {N,CA,C,O} |
| 4 | `test_place_correct_rotamer_id` | ALA, rot=0 | `placed.id == RotamerId{aa:"ALA",bin=0,rot=0}` |
| 5 | `test_place_unknown_aa` | aa="ZZZ" | `Err(UnknownAa("ZZZ"))` |
| 6 | `test_place_oob_rot_index` | ALA, rot=99 | `Err(RotIndexOob("ALA",99,1))` |
| 7 | `test_place_sentinel_uses_default_bin` | ALA, phi=9999, psi=9999, rot=0 | `Ok(_)`, `placed.id.bin_index==0` |
| 8 | `test_place_parity_mosaist` `#[ignore]` | real rotlib, backbone N/CA/C above, ALA rot=0 | each xyz within 1e-5 Å of MSL reference |

**Test 8**: Pre-compute MSL reference values offline; store as `const` arrays.

---

## Module 5: `tests/test_frame.rs`

**Backbone**: N=[0,0,0], CA=[1.458,0,0], C=[1.980,1.418,0].  
Derived: x=[1,0,0], y=[0,1,0], z=[0,0,1], origin=CA=[1.458,0,0].  
`switch_frames(backbone, identity)` applied to [1,0,0] → [2.458, 0.0, 0.0].

| # | Test | Input | Expected (tolerance) |
|---|------|-------|----------------------|
| 1 | `test_frame_identity_switch_is_identity` | switch_frames(I,I) on [1,2,3] | [1,2,3] (1e-12) |
| 2 | `test_frame_new_normalizes_x` | Frame::new([0,0,0],[3,0,0],[0,1,0],[0,0,1]) | x=[1,0,0] (1e-12) |
| 3 | `test_frame_new_normalizes_all_axes` | Frame::new(0,[5,0,0],[0,3,0],[0,0,7]) | x=[1,0,0],y=[0,1,0],z=[0,0,1] (1e-12) |
| 4 | `test_transform_identity_apply` | identity.apply([3.14,-2.71,0]) | [3.14,-2.71,0] (1e-15) |
| 5 | `test_switch_frames_known_backbone` | switch_frames(backbone_frame(N,CA,C), I) on [1,0,0] | [2.458,0,0] (1e-10) |
| 6 | `test_switch_frames_roundtrip` | compose F→I then I→F on [3,5,7] | [3,5,7] (1e-10) — best regression guard |
| 7 | `test_switch_frames_translated_origin` | from=Frame at [5,0,0] identity axes, to=I; apply [0,0,0] | [5,0,0] (1e-10) |

---

## Module 6: `tests/test_sidechain.rs`

All pure function, no fixture.

| # | Test | Input | Expected |
|---|------|-------|----------|
| 1 | `test_counts_cb_ala_true` | CB/ALA | true |
| 2 | `test_counts_cb_arg_false` | CB/ARG | false |
| 3 | `test_counts_cb_all_non_ala_false` | CB + all 17 non-ALA | all false |
| 4–7 | `test_counts_backbone_N/CA/C/O_false` | N,CA,C,O | false |
| 8 | `test_counts_hydrogen_prefix_false` | HB2/LEU | false |
| 9 | `test_counts_hydrogen_bare_h_false` | H/ALA | false |
| 10 | `test_counts_sidechain_heavy_true` | CD/LEU | true |
| 11 | `test_counts_sidechain_heavy_arg` | NE/ARG | true |
| 12–16 | `test_is_backbone_N/CA/C/O/H` | N,CA,C,O,H | true |
| 17 | `test_is_backbone_h_prefix` | HB2 | true |
| 18 | `test_is_not_backbone_cd` | CD | false |
| 19 | `test_is_not_backbone_cb` | CB | false |
| 20 | `test_is_not_backbone_empty` | "" | false |

---

## Summary

| Module | Unit | Integration (`#[ignore]`) |
|--------|------|---------------------------|
| test_load | 3 | 2 |
| test_binning | 11 | 0 |
| test_probability | 7 | 1 |
| test_placement | 7 | 1 |
| test_frame | 7 | 0 |
| test_sidechain | 20 | 0 |
| **Total** | **55** | **4** |

Gate: `cargo test -p proxide-rotlib` (unit only)
Integration gate: `ROTLIB_PATH=/path/to/rotlib.bin cargo test -p proxide-rotlib -- --ignored`
