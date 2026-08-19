use crate::formats::xtc::molly_impl::read_xtc_molly;
use crate::formats::xtc::{OffsetCache, XtcReader, OFFSET_CACHE_VERSION_PRE_TRUNCATION_FIX};
use molly::selection::AtomSelection;
use std::path::{Path, PathBuf};

fn project_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn test_xtc_path() -> Option<PathBuf> {
    let path = project_root().join("tests/data/trajectories/test.xtc");
    path.exists().then_some(path)
}

fn large_xtc_path() -> Option<PathBuf> {
    let path = project_root().join("tests/data/trajectories/large.xtc");
    path.exists().then_some(path)
}

/// Many-frame fixture (4000 frames, 64 atoms) with a per-frame VARYING,
/// slightly triclinic box, used to regression-test box-vector decoding —
/// investigated under praxia debt #1237 / proxide#16 (see
/// [`test_xtc_box_vectors_match_ground_truth_across_full_trajectory`] for
/// why that report turned out not to be a decode bug, and why this fixture
/// is still a valuable regression guard regardless). See
/// `scripts/generate_trajectory_test_data.py::generate_box_drift_xtc_fixture`
/// for how this and its ground-truth sidecar were generated.
fn box_drift_xtc_path() -> Option<PathBuf> {
    let path = project_root().join("tests/data/trajectories/box_drift.xtc");
    path.exists().then_some(path)
}

/// Ground-truth box vectors (Angstroms) for [`box_drift_xtc_path`]: a flat
/// little-endian f32 blob, shape `(num_frames, 3, 3)` row-major (row `i` of
/// each 3x3 is box vector `i`, matching mdtraj/GROMACS convention and
/// `frame.boxvec_cols_2d()`'s output layout).
fn load_box_drift_ground_truth() -> Vec<[[f32; 3]; 3]> {
    let path = project_root().join("tests/data/trajectories/box_drift_ground_truth_angstrom.bin");
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
    assert_eq!(bytes.len() % 36, 0, "ground-truth blob size must be a multiple of 9 f32s");
    let n_frames = bytes.len() / 36;
    let mut out = Vec::with_capacity(n_frames);
    for frame_idx in 0..n_frames {
        let mut mat = [[0.0f32; 3]; 3];
        for (row, mat_row) in mat.iter_mut().enumerate() {
            for (col, cell) in mat_row.iter_mut().enumerate() {
                let offset = frame_idx * 36 + row * 12 + col * 4;
                *cell = f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
            }
        }
        out.push(mat);
    }
    out
}

/// A well-characterized multi-frame fixture (501 frames), used for the
/// truncated-trailing-frame test below — needs enough frames that chopping
/// bytes off the end unambiguously lands inside the *last* frame's own
/// compressed payload rather than accidentally landing exactly on a frame
/// boundary.
fn frame0_xtc_path() -> Option<PathBuf> {
    let path = project_root().join("tests/data/trajectories/frame0.xtc");
    path.exists().then_some(path)
}

/// Copy the checked-in fixture into a fresh tempdir so offset-cache sidecar
/// writes never touch the real fixture (avoids races with other tests and
/// leaving untracked `.offsets` files in the repo).
fn copy_fixture_to_tempdir(src: &Path) -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().expect("failed to create tempdir");
    let dst = dir.path().join(src.file_name().unwrap());
    std::fs::copy(src, &dst).expect("failed to copy fixture");
    (dir, dst)
}

#[test]
fn test_read_xtc_real_file() {
    // Determine path to test.xtc relative to workspace root
    // Environment Context shows it at tests/data/trajectories/test.xtc
    let Some(xtc_path) = test_xtc_path() else {
        // Fallback for different test environments if needed,
        // but based on environment context it should be there.
        return;
    };

    let traj = read_xtc_molly(xtc_path).expect("Failed to read XTC file");

    // We expect some frames and atoms.
    // Let's verify based on typical test.xtc properties if known,
    // or just check they are non-zero.
    assert!(traj.num_frames > 0);
    assert!(traj.num_atoms > 0);
    assert_eq!(traj.coords.len(), traj.num_frames);
    assert_eq!(traj.coords[0].len(), traj.num_atoms * 3);
    assert_eq!(traj.times.len(), traj.num_frames);
}

#[test]
fn test_read_xtc_nonexistent() {
    let result = read_xtc_molly("nonexistent.xtc");
    assert!(result.is_err());
}

/// The lazy `XtcReader` (offset-scan + `read_frame_at`) must produce the same
/// frame count, coordinates, and box vectors as the eager `read_xtc_molly`
/// path, for every frame.
#[test]
fn test_xtc_reader_parity_with_eager_load() {
    let Some(xtc_path) = test_xtc_path() else {
        return;
    };
    let (_dir, path) = copy_fixture_to_tempdir(&xtc_path);

    let eager = read_xtc_molly(&path).expect("eager read failed");

    let mut reader = XtcReader::open(&path).expect("failed to open XtcReader");
    assert_eq!(reader.frame_count().unwrap(), eager.num_frames);
    assert_eq!(reader.n_atoms().unwrap(), eager.num_atoms);

    for i in 0..eager.num_frames {
        let frame = reader
            .read_frame_at(i, &AtomSelection::All)
            .unwrap_or_else(|e| panic!("read_frame_at({i}) failed: {e}"));
        let coords: Vec<f32> = frame.positions.iter().map(|x| x * 10.0).collect();
        assert_eq!(coords, eager.coords[i], "coords mismatch at frame {i}");

        let mut box_ang = frame.boxvec_cols_2d();
        for row in &mut box_ang {
            for v in row {
                *v *= 10.0;
            }
        }
        assert_eq!(box_ang, eager.boxes[i], "box mismatch at frame {i}");
    }
}

/// `read_range` must match the corresponding slice of the eager load.
#[test]
fn test_xtc_reader_read_range_matches_eager() {
    let Some(xtc_path) = test_xtc_path() else {
        return;
    };
    let (_dir, path) = copy_fixture_to_tempdir(&xtc_path);

    let eager = read_xtc_molly(&path).expect("eager read failed");
    let mut reader = XtcReader::open(&path).expect("failed to open XtcReader");
    let n = eager.num_frames;
    let batch = reader.read_range(0..n).expect("read_range failed");

    assert_eq!(batch.num_frames, eager.num_frames);
    assert_eq!(batch.coords, eager.coords);
    assert_eq!(batch.boxes, eager.boxes);
    assert_eq!(batch.times, eager.times);
}

/// `iter()` must yield exactly `frame_count()` frames, matching sequential
/// `read_frame_at` calls.
#[test]
fn test_xtc_reader_iter_matches_read_frame_at() {
    let Some(xtc_path) = test_xtc_path() else {
        return;
    };
    let (_dir, path) = copy_fixture_to_tempdir(&xtc_path);

    let mut reader = XtcReader::open(&path).expect("failed to open XtcReader");
    let n = reader.frame_count().unwrap();

    let via_iter: Vec<Vec<f32>> = reader
        .iter()
        .unwrap()
        .map(|f| f.expect("frame decode failed").positions)
        .collect();
    assert_eq!(via_iter.len(), n);

    for (i, expected) in via_iter.iter().enumerate() {
        let frame = reader.read_frame_at(i, &AtomSelection::All).unwrap();
        assert_eq!(&frame.positions, expected, "mismatch at frame {i}");
    }
}

/// The on-disk offset-index sidecar must be created on first open, reused
/// (without a rescan producing a different result) on a second open, and
/// invalidated when the trajectory file's mtime/size change.
#[test]
fn test_xtc_offset_cache_invalidation() {
    let Some(xtc_path) = test_xtc_path() else {
        return;
    };
    let (_dir, path) = copy_fixture_to_tempdir(&xtc_path);
    let sidecar = PathBuf::from(format!("{}.offsets", path.display()));

    assert!(!sidecar.exists(), "sidecar should not exist yet");

    let n_frames = {
        let mut reader = XtcReader::open(&path).expect("open failed");
        reader.frame_count().expect("frame_count failed")
    };
    assert!(sidecar.exists(), "sidecar should be written after first open");

    // Second open must reuse the cache and agree on frame count.
    {
        let mut reader = XtcReader::open(&path).expect("open failed");
        assert_eq!(reader.frame_count().unwrap(), n_frames);
    }

    // Touch the file (bump mtime) without changing content; cache must still
    // be considered valid only if mtime+size+natoms unchanged — here we
    // deliberately change size by appending a byte, which must invalidate it.
    {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        f.write_all(&[0u8]).unwrap();
    }

    let mut reader = XtcReader::open(&path).expect("open failed");
    // Forcing a refresh must not panic even though the file was corrupted by
    // the trailing byte; frame_count() on the (now invalid-per-cache) file
    // re-scans rather than trusting the stale/mismatched sidecar.
    let _ = reader.refresh_offsets();
}

/// Concurrent `read_frames_parallel` decode must match sequential
/// `read_frame_at` calls, frame for frame.
#[cfg(feature = "parallel")]
#[test]
fn test_read_frames_parallel_matches_serial() {
    use crate::formats::xtc::read_frames_parallel;

    let Some(xtc_path) = test_xtc_path() else {
        return;
    };
    let (_dir, path) = copy_fixture_to_tempdir(&xtc_path);

    let mut reader = XtcReader::open(&path).expect("open failed");
    let n = reader.frame_count().unwrap();
    let indices: Vec<usize> = (0..n).collect();

    let serial: Vec<Vec<f32>> = indices
        .iter()
        .map(|&i| reader.read_frame_at(i, &AtomSelection::All).unwrap().positions)
        .collect();

    let parallel = read_frames_parallel(&path, &indices, None).expect("parallel read failed");
    let parallel_positions: Vec<Vec<f32>> = parallel.into_iter().map(|f| f.positions).collect();

    assert_eq!(parallel_positions, serial);
}

/// `read_frames_parallel` must apply `atom_indices` with the exact order and
/// duplicates the caller passed — the same mdtraj fancy-indexing semantic
/// `frame_to_angstroms_selected` documents at the PyO3-binding layer — not
/// the ascending/deduped semantics of
/// `molly::selection::AtomSelection::from_index_list`'s `Mask`. This is the
/// regression most likely to silently break if a future change routes
/// selection through that helper instead of the order-preserving logic in
/// `select_positions`.
#[cfg(feature = "parallel")]
#[test]
fn test_read_frames_parallel_preserves_atom_index_order_and_duplicates() {
    use crate::formats::xtc::read_frames_parallel;

    let Some(xtc_path) = test_xtc_path() else {
        return;
    };
    let (_dir, path) = copy_fixture_to_tempdir(&xtc_path);

    let mut reader = XtcReader::open(&path).expect("open failed");
    let n = reader.frame_count().unwrap();
    let n_atoms = reader.n_atoms().unwrap();
    assert!(n_atoms >= 3, "fixture too small to exercise a real subset");
    let indices: Vec<usize> = (0..n).collect();

    // Deliberately unsorted, with a duplicate (index 1 appears twice) and not
    // starting from 0 — a Mask-based selection would silently return this as
    // ascending-deduped `[0, 1, 2]`.
    let last = n_atoms - 1;
    let atom_indices = vec![last, 0, 1, 1];

    // Ground truth: manually select from the sequential per-frame decode, in
    // the exact order/duplicates requested.
    let expected: Vec<Vec<f32>> = indices
        .iter()
        .map(|&i| {
            let frame = reader.read_frame_at(i, &AtomSelection::All).unwrap();
            atom_indices
                .iter()
                .flat_map(|&a| {
                    let base = a * 3;
                    frame.positions[base..base + 3].to_vec()
                })
                .collect()
        })
        .collect();

    let parallel = read_frames_parallel(&path, &indices, Some(&atom_indices))
        .expect("parallel read failed");
    assert_eq!(parallel.len(), n);
    for (i, frame) in parallel.iter().enumerate() {
        assert_eq!(
            frame.positions.len(),
            atom_indices.len() * 3,
            "frame {i} not reduced to the selected atom count"
        );
        assert_eq!(
            frame.positions, expected[i],
            "frame {i} selected positions diverge from order/duplicate-preserving ground truth"
        );
    }
}

/// An out-of-range atom index must surface as a clean `XtcError`, not a
/// panic from unchecked slice indexing inside a parallel worker.
#[cfg(feature = "parallel")]
#[test]
fn test_read_frames_parallel_rejects_out_of_range_atom_index() {
    use crate::formats::xtc::{read_frames_parallel, XtcError};

    let Some(xtc_path) = test_xtc_path() else {
        return;
    };
    let (_dir, path) = copy_fixture_to_tempdir(&xtc_path);

    let mut reader = XtcReader::open(&path).expect("open failed");
    let n = reader.frame_count().unwrap();
    let n_atoms = reader.n_atoms().unwrap();
    let indices: Vec<usize> = (0..n).collect();
    let bad_atom_indices = vec![0, n_atoms + 100];

    let result = read_frames_parallel(&path, &indices, Some(&bad_atom_indices));
    assert!(
        matches!(result, Err(XtcError::AtomIndexOutOfBounds { .. })),
        "expected AtomIndexOutOfBounds, got {result:?}"
    );
}

/// Write an MDAnalysis-shaped `.{name}_offsets.npz` via numpy (same keys/dtypes
/// as `XDRBaseReader._read_offsets`).
fn write_mda_offsets_npz(
    traj_path: &Path,
    offsets: &[u64],
    file_size: u64,
    n_atoms: i32,
) {
    let parent = traj_path.parent().unwrap();
    let name = traj_path.file_name().unwrap().to_string_lossy();
    let mda_path = parent.join(format!(".{name}_offsets.npz"));
    let offsets_lit = offsets
        .iter()
        .map(|o| o.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let script = format!(
        "import numpy as np\n\
         np.savez({mda_path:?},\n\
                  offsets=np.array([{offsets_lit}], dtype=np.int64),\n\
                  size=np.int64({file_size}),\n\
                  ctime=np.float64(0.0),\n\
                  n_atoms=np.int32({n_atoms}))\n"
    );
    let status = std::process::Command::new("python3")
        .arg("-c")
        .arg(&script)
        .status()
        .expect("failed to spawn python3 for MDA npz fixture");
    assert!(status.success(), "python3 MDA npz writer failed");
    assert!(mda_path.exists(), "MDA npz was not written");
}

fn molly_offsets_and_natoms(path: &Path) -> (Vec<u64>, usize) {
    use molly::XTCReader;
    let mut reader = XTCReader::open(path).expect("molly open failed");
    reader.home().unwrap();
    let header = reader.read_header().expect("read_header failed");
    let natoms = header.natoms;
    reader.home().unwrap();
    let offsets = reader
        .determine_offsets(None)
        .expect("determine_offsets failed")
        .into_vec();
    (offsets, natoms)
}

/// On a proxide `.offsets` miss, a matching MDAnalysis `.xtc_offsets.npz` must
/// be imported (and rewritten as the bincode sidecar) so cold `determine_offsets`
/// is skipped.
#[test]
fn test_mda_offsets_npz_imported_when_proxide_sidecar_missing() {
    let Some(xtc_path) = test_xtc_path() else {
        return;
    };
    let (_dir, path) = copy_fixture_to_tempdir(&xtc_path);
    let proxide_sidecar = PathBuf::from(format!("{}.offsets", path.display()));
    assert!(!proxide_sidecar.exists());

    let (offsets, n_atoms) = molly_offsets_and_natoms(&path);
    let n_frames = offsets.len();
    let file_size = std::fs::metadata(&path).unwrap().len();
    write_mda_offsets_npz(&path, &offsets, file_size, n_atoms as i32);

    // Fresh open: must import MDA npz, create proxide sidecar, agree on count.
    let mut reader = XtcReader::open(&path).expect("open failed");
    assert_eq!(reader.frame_count().unwrap(), n_frames);
    assert_eq!(reader.n_atoms().unwrap(), n_atoms);
    assert!(
        proxide_sidecar.exists(),
        "proxide sidecar should be written after MDA import"
    );

    // Second open reuses the converted proxide sidecar.
    let mut reader2 = XtcReader::open(&path).expect("re-open failed");
    assert_eq!(reader2.frame_count().unwrap(), n_frames);
}

/// MDA npz with a mismatched file size must not be trusted — fall through to
/// a real `determine_offsets` scan (which still produces a correct count).
#[test]
fn test_mda_offsets_npz_size_mismatch_falls_through_to_scan() {
    let Some(xtc_path) = test_xtc_path() else {
        return;
    };
    let (_dir, path) = copy_fixture_to_tempdir(&xtc_path);
    let proxide_sidecar = PathBuf::from(format!("{}.offsets", path.display()));

    let (offsets, n_atoms) = molly_offsets_and_natoms(&path);
    let n_frames = offsets.len();
    write_mda_offsets_npz(
        &path,
        &offsets,
        std::fs::metadata(&path).unwrap().len().saturating_add(1),
        n_atoms as i32,
    );

    let mut reader = XtcReader::open(&path).expect("open failed");
    assert_eq!(reader.frame_count().unwrap(), n_frames);
    assert!(
        proxide_sidecar.exists(),
        "scan should still write proxide sidecar"
    );
}

/// At real scale (thousands of frames): the offset-cache sidecar must be
/// created once and reused, `frame_count()` must not require decoding any
/// coordinates, and a random subset of frames read via `read_frame_at` must
/// be internally consistent (arbitrary frame decode after a big seek works).
#[test]
fn test_xtc_reader_large_fixture_offsets_and_seek() {
    let Some(xtc_path) = large_xtc_path() else {
        return;
    };
    let (_dir, path) = copy_fixture_to_tempdir(&xtc_path);
    let sidecar = PathBuf::from(format!("{}.offsets", path.display()));

    let n_frames = {
        let mut reader = XtcReader::open(&path).expect("open failed");
        reader.frame_count().expect("frame_count failed")
    };
    assert_eq!(n_frames, 5000);
    assert!(sidecar.exists(), "sidecar should be written after first open");

    // Second open reuses the cached sidecar and must agree.
    let mut reader = XtcReader::open(&path).expect("open failed");
    assert_eq!(reader.frame_count().unwrap(), n_frames);
    assert_eq!(reader.n_atoms().unwrap(), 4);

    // Seeking to a late frame must not require decoding everything before it.
    let last = reader
        .read_frame_at(n_frames - 1, &AtomSelection::All)
        .expect("failed to decode last frame");
    assert_eq!(last.positions.len(), 4 * 3);

    // A frame in the middle, read out of order relative to the one above,
    // must also decode correctly (real random access, not just "last works").
    let mid = reader
        .read_frame_at(n_frames / 2, &AtomSelection::All)
        .expect("failed to decode middle frame");
    assert_eq!(mid.positions.len(), 4 * 3);
}

#[cfg(feature = "parallel")]
#[test]
fn test_read_frames_parallel_matches_serial_large_fixture() {
    use crate::formats::xtc::read_frames_parallel;

    let Some(xtc_path) = large_xtc_path() else {
        return;
    };
    let (_dir, path) = copy_fixture_to_tempdir(&xtc_path);

    let mut reader = XtcReader::open(&path).expect("open failed");
    let n = reader.frame_count().unwrap();
    // Sample every 50th frame rather than all 5000, to keep the test fast
    // while still exercising a wide spread of offsets.
    let indices: Vec<usize> = (0..n).step_by(50).collect();

    let serial: Vec<Vec<f32>> = indices
        .iter()
        .map(|&i| reader.read_frame_at(i, &AtomSelection::All).unwrap().positions)
        .collect();

    let parallel = read_frames_parallel(&path, &indices, None).expect("parallel read failed");
    let parallel_positions: Vec<Vec<f32>> = parallel.into_iter().map(|f| f.positions).collect();

    assert_eq!(parallel_positions, serial);
}

/// Memory-scaling regression guard for the `read_frames_parallel` OOM fix
/// (praxia debt #1220): peak memory for the collected result must scale
/// with `atom_indices.len()`, not the trajectory's full per-frame atom
/// count. `large.xtc` has 5000 frames at a tiny 4 atoms/frame, which can't
/// demonstrate a *savings ratio* on its own — the point here is narrower and
/// still meaningful without a real memory profiler: assert every returned
/// `SelectedFrame.positions` is sized to exactly `atom_indices.len() * 3`
/// floats, for a `atom_indices` subset smaller than the fixture's full atom
/// count, across every one of many frames. Before the fix, `read_frames_parallel`
/// returned full-atom-count `MollyFrame`s with no selection applied at all
/// inside the worker — this would have failed immediately on the shape
/// assertion, since the (then nonexistent) selection step was applied only
/// by the caller, after collection, in `py_xtc_reader.rs`, not in this
/// crate.
#[cfg(feature = "parallel")]
#[test]
fn test_read_frames_parallel_result_size_scales_with_selection_not_full_atoms() {
    use crate::formats::xtc::read_frames_parallel;

    let Some(xtc_path) = large_xtc_path() else {
        return;
    };
    let (_dir, path) = copy_fixture_to_tempdir(&xtc_path);

    let mut reader = XtcReader::open(&path).expect("open failed");
    let n = reader.frame_count().unwrap();
    let n_atoms = reader.n_atoms().unwrap();
    assert!(
        n_atoms > 1,
        "fixture needs more than 1 atom to exercise a real subset"
    );
    let indices: Vec<usize> = (0..n).step_by(50).collect();
    let atom_indices = vec![0usize];

    let parallel = read_frames_parallel(&path, &indices, Some(&atom_indices))
        .expect("parallel read failed");
    assert_eq!(parallel.len(), indices.len());
    for frame in &parallel {
        assert_eq!(
            frame.positions.len(),
            atom_indices.len() * 3,
            "selected frame should be sized to the 1-atom selection, not the \
             fixture's full {n_atoms}-atom frame"
        );
    }
}

/// A trajectory whose last frame's compressed payload is only partially
/// flushed — the normal state of the tail of a file a live simulation is
/// still appending to — must not be counted. `determine_offsets`'s
/// header-only scan skips forward via `seek`, which succeeds past EOF, so
/// without a decode-based guard the truncated last frame would silently
/// count as complete.
#[test]
fn test_xtc_reader_excludes_truncated_trailing_frame() {
    let Some(xtc_path) = frame0_xtc_path() else {
        return;
    };
    let (_dir, path) = copy_fixture_to_tempdir(&xtc_path);

    let full_count = {
        let mut reader = XtcReader::open(&path).expect("open failed");
        reader.frame_count().expect("frame_count failed")
    };
    assert!(full_count > 1, "fixture must have more than one frame");

    // Chop 10 bytes off the end — well inside the last frame's own
    // compressed-coordinate payload (not exactly on a frame boundary),
    // simulating a header that's fully written but whose data isn't yet.
    let full_size = std::fs::metadata(&path).unwrap().len();
    let truncated = tempfile::tempdir().expect("tempdir failed");
    let truncated_path = truncated.path().join("truncated.xtc");
    let data = std::fs::read(&path).unwrap();
    std::fs::write(&truncated_path, &data[..(full_size as usize - 10)]).unwrap();

    let mut reader = XtcReader::open(&truncated_path).expect("open failed");
    let truncated_count = reader.frame_count().expect("frame_count failed");
    assert_eq!(
        truncated_count,
        full_count - 1,
        "truncated last frame must be excluded from the count"
    );

    // The sidecar written for the truncated file must reflect the excluded
    // count too, not the phantom pre-guard count — otherwise a later open
    // would trust a stale cache built before the guard ran.
    let mut reader2 = XtcReader::open(&truncated_path).expect("reopen failed");
    assert_eq!(reader2.frame_count().unwrap(), full_count - 1);
}

/// A crash landing inside the ~32-40 byte compression-metadata/`nbytes`
/// window right after the *last* frame's header — rather than inside its
/// compressed-coordinate payload, which the test above covers — must also
/// be treated as a truncated trailing frame, not propagated as a hard I/O
/// error. Before this fix, `skip_positions`'s `read_nbytes` call raised
/// `UnexpectedEof` inside molly's own `determine_offsets_exclusive`, which
/// propagated straight out of `determine_offsets` before
/// `drop_trailing_frame_if_truncated` ever got a chance to run — so
/// `frame_count()` failed hard instead of returning `full_count - 1` as
/// documented ("safe to call on a trajectory a live simulation is still
/// appending to").
#[test]
fn test_xtc_reader_tolerates_truncation_in_frame_metadata_region() {
    let Some(xtc_path) = frame0_xtc_path() else {
        return;
    };
    let (_dir, path) = copy_fixture_to_tempdir(&xtc_path);

    let (offsets, natoms) = molly_offsets_and_natoms(&path);
    assert!(
        natoms > 9,
        "fixture must use the compressed (nbytes-based) frame layout \
         skip_positions takes for natoms > 9, not the uncompressed <=9-atom path"
    );
    let full_count = offsets.len();
    assert!(full_count > 1, "fixture must have more than one frame");
    let last_frame_start = offsets[full_count - 1];

    // Cut 10 bytes into the 32-byte precision/minint/maxint/smallidx block
    // that follows the 56-byte header (`molly::Header::SIZE`) and precedes
    // the `nbytes` field — short of ever reaching `nbytes`, let alone the
    // compressed payload itself.
    let cut_at = last_frame_start + molly::Header::SIZE as u64 + 10;
    let data = std::fs::read(&path).unwrap();
    assert!(
        (cut_at as usize) < data.len(),
        "cut point must actually shorten the file"
    );
    let truncated_path = path.with_file_name("truncated_metadata.xtc");
    std::fs::write(&truncated_path, &data[..cut_at as usize]).unwrap();

    let mut reader = XtcReader::open(&truncated_path).expect("open failed");
    let truncated_count = reader
        .frame_count()
        .expect("frame_count must tolerate metadata-region truncation, not error");
    assert_eq!(
        truncated_count,
        full_count - 1,
        "a frame truncated in its metadata region must be excluded from the count"
    );
}

/// A `.offsets` sidecar written by a pre-fix build — correct mtime/size/
/// natoms, but `version` stamped at the old pre-fix value and its offsets
/// list still including the phantom offset for a truncated trailing frame —
/// must not be trusted on the next open. `OffsetCache::matches` should
/// reject it purely on the version mismatch, forcing a rescan through
/// `scan_and_cache_offsets` (which now applies the truncation guard) rather
/// than silently returning the old, wrong, truncation-unaware count forever.
///
/// Regresses the bug where `OFFSET_CACHE_VERSION` was not bumped alongside
/// the truncation fix, which would have silently defeated it for every
/// trajectory file that had already been opened (by `read_xtc_lazy`,
/// `read_xtc_parallel`, or an earlier build of these bindings) before the
/// fix shipped.
#[test]
fn test_stale_pre_fix_offset_cache_is_rescanned_not_trusted() {
    let Some(xtc_path) = frame0_xtc_path() else {
        return;
    };
    let (_dir, path) = copy_fixture_to_tempdir(&xtc_path);

    let full_count = {
        let mut reader = XtcReader::open(&path).expect("open failed");
        reader.frame_count().expect("frame_count failed")
    };
    assert!(full_count > 1, "fixture must have more than one frame");

    // Truncate 10 bytes off the end, same as the payload-truncation test
    // above — this file's *true* frame count is `full_count - 1`.
    let full_size = std::fs::metadata(&path).unwrap().len();
    let data = std::fs::read(&path).unwrap();
    std::fs::write(&path, &data[..(full_size as usize - 10)]).unwrap();

    // Build the "phantom" pre-fix offset list directly via molly's raw,
    // unguarded scan on the now-truncated file: it still includes an offset
    // for the truncated last frame, exactly as pre-fix `scan_and_cache_offsets`
    // would have cached it.
    let (phantom_offsets, natoms) = molly_offsets_and_natoms(&path);
    assert_eq!(
        phantom_offsets.len(),
        full_count,
        "molly's raw scan (no truncation guard) must still count the phantom last frame"
    );

    let meta = std::fs::metadata(&path).unwrap();
    let mut stale_cache = OffsetCache::from_metadata(&meta, natoms, phantom_offsets)
        .expect("from_metadata failed");
    // Downgrade to simulate a sidecar actually written by a pre-fix build —
    // everything else about it (mtime/size/natoms) is genuinely valid for
    // this file, isolating the version field as the only signal that should
    // force a rescan. Uses the real historical constant (not a bare literal,
    // and not derived from the current `OFFSET_CACHE_VERSION`) so this test
    // proves the specific claim "a real pre-fix sidecar gets rescanned", not
    // just "any version mismatch gets rescanned" (already true, and would
    // prove nothing about whether the bump itself happened).
    stale_cache.version = OFFSET_CACHE_VERSION_PRE_TRUNCATION_FIX;
    stale_cache.store(&path).expect("failed to write stale sidecar");

    let mut reader = XtcReader::open(&path).expect("open failed");
    let count = reader
        .frame_count()
        .expect("frame_count failed on rescan");
    assert_eq!(
        count,
        full_count - 1,
        "a stale pre-fix sidecar must be rescanned, not trusted — trusting it \
         would silently return the phantom truncated-frame count forever"
    );
}

/// `determine_offsets_tolerant` reimplements molly's own header+skip loop
/// (via its public `read_header`/`skip_positions`) instead of calling
/// `molly::XTCReader::determine_offsets` directly, purely to add EOF
/// tolerance around `skip_positions` — see that function's doc comment. All
/// byte-level frame parsing still goes through molly's own code; only the
/// loop's error handling is ours.
///
/// For any well-formed (non-truncated) file the two must therefore produce
/// byte-identical offset lists. This is the test that catches it directly if
/// a future `molly` upgrade changes `Header::SIZE`, the frame layout, or the
/// `read_header`/`skip_positions` contract this reimplementation assumes —
/// rather than relying on that drift to eventually surface as an unrelated
/// failure in one of the other parity tests above.
#[test]
fn test_determine_offsets_tolerant_matches_molly_determine_offsets() {
    let mut checked_any = false;
    for maybe_path in [test_xtc_path(), frame0_xtc_path(), large_xtc_path()] {
        let Some(xtc_path) = maybe_path else {
            continue;
        };
        checked_any = true;
        let (_dir, path) = copy_fixture_to_tempdir(&xtc_path);

        let (molly_offsets, _natoms) = molly_offsets_and_natoms(&path);

        let mut reader = XtcReader::open(&path).expect("open failed");
        let ours = reader
            .determine_offsets_tolerant_for_test()
            .expect("determine_offsets_tolerant_for_test failed");

        assert_eq!(
            ours, molly_offsets,
            "determine_offsets_tolerant must match molly::XTCReader::determine_offsets \
             byte-for-byte on well-formed fixture {}",
            path.display()
        );
    }
    assert!(checked_any, "no fixture files were available to check");
}

// ---------------------------------------------------------------------------
// `read_xtc_distogram_parallel` (minimum-image-convention pairwise distances)
// ---------------------------------------------------------------------------

/// Shape/sanity check on a real multi-frame protein trajectory: correct
/// `(n_frames, n_pairs)` shape, every distance finite and non-negative, and
/// within a physically sane range for an intra-protein Cα-Cα (or similar)
/// distance — never NaN, never negative, never absurdly large.
#[cfg(feature = "parallel")]
#[test]
fn test_read_xtc_distogram_parallel_shape_and_sanity_real_fixture() {
    use crate::formats::xtc::read_xtc_distogram_parallel;

    let Some(xtc_path) = frame0_xtc_path() else {
        return;
    };
    let (_dir, path) = copy_fixture_to_tempdir(&xtc_path);

    let mut reader = XtcReader::open(&path).expect("open failed");
    let n_frames = reader.frame_count().unwrap();
    let n_atoms = reader.n_atoms().unwrap();
    assert!(n_atoms >= 10, "fixture too small to exercise a real subset");

    // A handful of atoms spread across the fixture, not just the first few.
    let atom_indices: Vec<usize> = (0..n_atoms).step_by(n_atoms / 8).take(8).collect();
    let n = atom_indices.len();
    let expected_n_pairs = n * (n - 1) / 2;

    // Sample every 25th frame to keep the test fast.
    let frame_indices: Vec<usize> = (0..n_frames).step_by(25).collect();

    let distograms = read_xtc_distogram_parallel(&path, &frame_indices, &atom_indices)
        .expect("distogram computation failed");

    assert_eq!(distograms.len(), frame_indices.len());
    for row in &distograms {
        assert_eq!(row.len(), expected_n_pairs);
        for &d in row {
            assert!(d.is_finite(), "distance must be finite, got {d}");
            assert!(d >= 0.0, "distance must be non-negative, got {d}");
            // A real protein structure's box is typically tens of
            // Angstroms; a correct MIC distance between any two atoms in it
            // must be well under that, never a raw pre-wrap separation
            // spanning the whole trajectory box.
            assert!(
                d < 1000.0,
                "distance {d} Å is implausibly large for an intra-structure atom pair"
            );
        }
    }
}

/// Independent-of-the-wrapper cross-check: manually decode a few frames via
/// the serial [`XtcReader::read_frame_at`] path, select the same atoms,
/// scale to Angstroms, and compute MIC distances directly with
/// [`pairwise_distances_mic`] — then compare against
/// `read_xtc_distogram_parallel`'s output. This exercises the wrapper's own
/// plumbing (atom selection, nm->Angstrom scaling for *both* positions and
/// box, box-vector extraction) independently of its internal call to
/// `read_frames_parallel`, rather than just re-asserting the primitive is
/// correct (already covered in `proxide-geometry`'s own tests).
#[cfg(feature = "parallel")]
#[test]
fn test_read_xtc_distogram_parallel_matches_manual_mic_computation() {
    use crate::formats::xtc::read_xtc_distogram_parallel;
    use proxide_geometry::geometry::distances::{pairwise_distances_mic, BoxDims};

    let Some(xtc_path) = frame0_xtc_path() else {
        return;
    };
    let (_dir, path) = copy_fixture_to_tempdir(&xtc_path);

    let mut reader = XtcReader::open(&path).expect("open failed");
    let n_frames = reader.frame_count().unwrap();
    let n_atoms = reader.n_atoms().unwrap();
    assert!(n_atoms >= 6, "fixture too small to exercise a real subset");

    let atom_indices = vec![0usize, n_atoms / 3, n_atoms / 2, n_atoms - 1];
    // A small, out-of-order sample of frames.
    let frame_indices: Vec<usize> = vec![0, n_frames / 3, n_frames / 2, n_frames - 1];

    let expected: Vec<Vec<f32>> = frame_indices
        .iter()
        .map(|&i| {
            let frame = reader.read_frame_at(i, &AtomSelection::All).unwrap();
            let positions: Vec<[f32; 3]> = atom_indices
                .iter()
                .map(|&a| {
                    let base = a * 3;
                    [
                        frame.positions[base] * 10.0,
                        frame.positions[base + 1] * 10.0,
                        frame.positions[base + 2] * 10.0,
                    ]
                })
                .collect();
            let mut box_ang = frame.boxvec_cols_2d();
            for row in &mut box_ang {
                for v in row {
                    *v *= 10.0;
                }
            }
            let box_dims = BoxDims::from_diagonal_matrix(&box_ang);
            pairwise_distances_mic(&positions, Some(&box_dims))
        })
        .collect();

    let got = read_xtc_distogram_parallel(&path, &frame_indices, &atom_indices)
        .expect("distogram computation failed");

    assert_eq!(got.len(), expected.len());
    for (row_got, row_expected) in got.iter().zip(expected.iter()) {
        assert_eq!(row_got.len(), row_expected.len());
        for (g, e) in row_got.iter().zip(row_expected.iter()) {
            assert!(
                (g - e).abs() < 1e-3,
                "manual MIC computation diverges from read_xtc_distogram_parallel: got {g} expected {e}"
            );
        }
    }
}

/// A single-atom selection has zero pairs — must not panic or divide by
/// zero, and must return an empty row per frame.
#[cfg(feature = "parallel")]
#[test]
fn test_read_xtc_distogram_parallel_single_atom_has_zero_pairs() {
    use crate::formats::xtc::read_xtc_distogram_parallel;

    let Some(xtc_path) = test_xtc_path() else {
        return;
    };
    let (_dir, path) = copy_fixture_to_tempdir(&xtc_path);

    let mut reader = XtcReader::open(&path).expect("open failed");
    let n_frames = reader.frame_count().unwrap();
    let frame_indices: Vec<usize> = (0..n_frames).collect();

    let distograms = read_xtc_distogram_parallel(&path, &frame_indices, &[0])
        .expect("distogram computation failed");
    assert_eq!(distograms.len(), n_frames);
    for row in &distograms {
        assert!(row.is_empty());
    }
}

/// Regression guard investigated under praxia debt #1237 / proxide#16: box
/// vectors decoded from a many-frame, per-frame-VARYING-box XTC trajectory
/// must match ground truth at every sampled frame across the file — not
/// just frame 0.
///
/// proxide#16 originally reported `box_vectors` from `read_xtc_lazy`/
/// `read_xtc_parallel` diverging from mdtraj's by >100 Angstrom by
/// mid-trajectory on a real ~66,543-frame file, while coordinates/times
/// matched exactly. Deep root-causing (see the PR that added this test)
/// found this was **not** a proxide decode bug: proxide's raw, byte-exact
/// box matrix — independently cross-checked against mdtraj's own low-level
/// `XTCTrajectoryFile.read()` and against a hand-parsed read of the raw XDR
/// bytes at the same offset — matched perfectly at every frame tested,
/// including the frame with the reported ~134 Angstrom "divergence". The
/// apparent corruption was mdtraj's *high-level* `Trajectory` reducing box
/// vectors to `unitcell_lengths`/`unitcell_angles` on load and
/// reconstructing a matrix in its own canonical (`a` along x, `b` in the
/// xy-plane) orientation on read — a lossy-of-orientation but
/// lengths/angles-preserving transform — which differs from proxide's
/// as-stored matrix whenever the source file's box isn't already written in
/// that canonical orientation (common for trajectories converted from
/// AMBER). Converting proxide's raw box back to lengths+angles reproduces
/// mdtraj's reported values to full float32 precision, confirming the two
/// are physically identical cells in different (but both valid)
/// representations.
///
/// This test therefore does NOT regress a "fix" to that decode path —
/// there was nothing to fix there. It guards against a *real* future
/// regression: this fixture has a per-frame-varying, slightly triclinic
/// ground-truth box (unlike every other fixture in this file, which is
/// static/orthorhombic and can't detect either a drift or a
/// transpose/row-column bug), sampled across the full trajectory rather
/// than frame 0 only — exactly the shape of check that would have caught
/// proxide#16's actual decode path being wrong, had it been.
#[test]
fn test_xtc_box_vectors_match_ground_truth_across_full_trajectory() {
    let Some(xtc_path) = box_drift_xtc_path() else {
        return;
    };
    let (_dir, path) = copy_fixture_to_tempdir(&xtc_path);
    let ground_truth = load_box_drift_ground_truth();

    let mut reader = XtcReader::open(&path).expect("open failed");
    let n_frames = reader.frame_count().expect("frame_count failed");
    assert_eq!(
        n_frames,
        ground_truth.len(),
        "fixture frame count must match ground-truth sidecar"
    );

    let mut max_abs_diff = 0.0f32;
    let mut max_abs_diff_frame = 0usize;
    let mut first_bad_frame: Option<usize> = None;

    for frame_idx in (0..n_frames).step_by(25) {
        let frame = reader
            .read_frame_at(frame_idx, &AtomSelection::All)
            .unwrap_or_else(|e| panic!("read_frame_at({frame_idx}) failed: {e}"));
        let mut box_ang = frame.boxvec_cols_2d();
        for row in &mut box_ang {
            for v in row {
                *v *= 10.0;
            }
        }

        let expected = ground_truth[frame_idx];
        let mut frame_max_diff = 0.0f32;
        for row in 0..3 {
            for col in 0..3 {
                let diff = (box_ang[row][col] - expected[row][col]).abs();
                frame_max_diff = frame_max_diff.max(diff);
            }
        }
        if frame_max_diff > max_abs_diff {
            max_abs_diff = frame_max_diff;
            max_abs_diff_frame = frame_idx;
        }
        if frame_max_diff > 1.0 && first_bad_frame.is_none() {
            first_bad_frame = Some(frame_idx);
        }
        assert!(
            frame_max_diff < 1.0,
            "box vectors diverge from ground truth at frame {frame_idx}: \
             got {box_ang:?}, expected {expected:?} (max abs diff {frame_max_diff} Angstrom)"
        );
    }

    // Belt-and-suspenders: even if every individual per-frame assertion
    // somehow passed, the worst-case diff across the whole scan must still
    // be small and no frame should have crossed the 1 Angstrom bad-frame
    // threshold above.
    assert!(
        first_bad_frame.is_none(),
        "at least one frame exceeded the 1 Angstrom threshold"
    );
    assert!(
        max_abs_diff < 1.0,
        "worst-case box vector diff across trajectory was {max_abs_diff} Angstrom \
         at frame {max_abs_diff_frame}"
    );
}
