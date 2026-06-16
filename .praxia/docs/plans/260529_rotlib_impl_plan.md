# Implementation Plan: `proxide-rotlib`

**task_id**: `260529_confind_rotlib`
**Spec oracle PASS @ 268e002 (worktree-rotlib-spec)**

## Phase order (each = one fixer dispatch)

### P1 — Scaffold
1.1 Create `crates/proxide-rotlib/Cargo.toml`
1.2 Modify root `Cargo.toml` (members + workspace.dependencies)
1.3 Create `src/error.rs` — RotlibError (Io, InvalidFormat, UnknownAa, RotIndexOob)
1.4 Create `src/lib.rs` — module decls + pub re-exports (stub)
Gate: `cargo check -p proxide-rotlib`

### P2 — Type definitions (no logic)
2.1 `src/rotamer_id.rs` — RotamerId, PlacedRotamer, PlacedAtom
2.2 `src/frame.rs` — Frame, Transform struct defs + identity ctors
2.3 `src/rotlib.rs` skeleton — RotamerLibrary, AaEntry, BinData
2.4 `src/binning.rs` — fn stubs (unimplemented!())
2.5 `src/sidechain.rs` — fn stubs
Gate: `cargo check -p proxide-rotlib`

### P3 — Angle helpers
3.1 `src/binning.rs` — implement angle_to_standard, angle_diff, find_closest_angle
Gate: `cargo check -p proxide-rotlib --lib`

### P4 — Frame/Transform methods
4.1 Frame::new (normalize axes), Frame::identity
4.2 Transform::identity, Transform::apply (row-major)
4.3 Transform::switch_frames (T2ᵀ·T1, T2ᵀ·ori)
4.4 backbone_frame(n, ca, c) free fn
Gate: `cargo check -p proxide-rotlib --lib`

### P5 — Binary parser
5.1 Grid validation helper in rotlib.rs
5.2 RotamerLibrary::load() — full binary parser (~200 LOC)
5.3 Public query method stubs (backbone_bin = unimplemented!())
Gate: `cargo check -p proxide-rotlib --lib`

### P6 — Binning + placement integration
6.1 Implement backbone_bin in binning.rs
6.2 RotamerLibrary::backbone_bin method in rotlib.rs
6.3 RotamerLibrary::place_rotamer (~30 LOC)
6.4 Implement sidechain.rs bodies
Gate: `cargo check -p proxide-rotlib --lib` (no unimplemented! left)

### P7 — Tests (6 files, one fixer each)
7.1 tests/test_load.rs (~80 LOC) — load, 18 AAs, 3 negative format cases
7.2 tests/test_binning.rs (~60 LOC) — sentinel, wrap-around, tie-break
7.3 tests/test_probability.rs (~50 LOC) — sum≤1, by_id parity, 2 negatives
7.4 tests/test_placement.rs (~100 LOC) — ALA/ARG counts, 2 negatives, parity
7.5 tests/test_frame.rs (~70 LOC) — identity, normalize, switch_frames
7.6 tests/test_sidechain.rs (~40 LOC) — CB/ALA, CB/ARG, backbone, H atoms
Gate: `cargo test -p proxide-rotlib` green

### P8 — Polish
8.1 Doc comments on all public items
8.2 `cargo clippy -p proxide-rotlib -- -D warnings` clean
Gate: all pass

## File manifest (~1,135 LOC, 14 files + 1 modified)
crates/proxide-rotlib/Cargo.toml      create  ~35
src/lib.rs                             create  ~15
src/error.rs                           create  ~20
src/rotamer_id.rs                      create  ~30
src/frame.rs                           create  ~150
src/rotlib.rs                          create  ~350
src/binning.rs                         create  ~60
src/sidechain.rs                       create  ~20
tests/test_load.rs                     create  ~80
tests/test_binning.rs                  create  ~60
tests/test_probability.rs              create  ~50
tests/test_placement.rs                create  ~100
tests/test_frame.rs                    create  ~70
tests/test_sidechain.rs                create  ~40
root Cargo.toml                        modify  +5 lines
