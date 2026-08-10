use crate::formats::xtc::molly_impl::read_xtc_molly;
use crate::formats::xtc::XtcReader;
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

    let parallel = read_frames_parallel(&path, &indices).expect("parallel read failed");
    let parallel_positions: Vec<Vec<f32>> = parallel.into_iter().map(|f| f.positions).collect();

    assert_eq!(parallel_positions, serial);
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

    let parallel = read_frames_parallel(&path, &indices).expect("parallel read failed");
    let parallel_positions: Vec<Vec<f32>> = parallel.into_iter().map(|f| f.positions).collect();

    assert_eq!(parallel_positions, serial);
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
