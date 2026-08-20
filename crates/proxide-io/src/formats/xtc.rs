//! XTC trajectory file format parser using the molly crate.
//!
//! XTC uses XDR encoding with lossy compression of coordinates.
//! Format reference: https://manual.gromacs.org/current/reference-manual/file-formats.html#xtc
//!
//! molly (pure Rust, no C/FFI) already implements the header-only offset scan and
//! offset-based random access that mdtraj/MDAnalysis get from their vendored C
//! `xdrfile` library (`XTCReader::determine_offsets`/`read_frame_at_offset`). This
//! module wires that up: [`XtcReader`] is a lazy, seekable cursor over an XTC
//! trajectory backed by those offsets, with an on-disk sidecar cache (mtime/size/
//! natoms-validated, mirroring MDAnalysis's `.npz` offset cache) so repeat opens
//! of the same file skip the rescan. [`XtcTrajectory`]/[`read_xtc_molly`] remain
//! the eager, whole-array (mdtraj-style) product for existing call sites.
//!
//! ## Box vector convention (praxia debt #1237 / proxide#16)
//!
//! Box vectors returned by this module (`XtcTrajectory::boxes`,
//! [`XtcReader::read_frame_at`]'s `MollyFrame::boxvec`/`boxvec_cols_2d()`, and
//! the Python-facing `box_vectors` from `read_xtc_lazy`/`read_xtc_parallel`)
//! are the raw 3x3 matrix **exactly as stored in the file**, row `i` = box
//! vector `i`, in whatever orientation the file's writer chose — the same
//! reference frame as the coordinates returned alongside them, so using a
//! frame's box together with that same frame's coordinates (e.g. for PBC
//! wrapping/unwrapping or unit-cell-relative distances) is always
//! self-consistent and correct.
//!
//! This is *not* necessarily the same matrix mdtraj's high-level
//! `Trajectory.unitcell_vectors` reports for the same file. mdtraj reduces
//! box vectors to `unitcell_lengths`/`unitcell_angles` (3+3 scalars) on
//! load and reconstructs a matrix in its own canonical orientation
//! (`a` along x, `b` in the xy-plane) every time `.unitcell_vectors` is
//! read — see `mdtraj/core/trajectory.py`'s `unitcell_vectors` property.
//! That reconstruction is lossless for the cell's *shape* (lengths + angles
//! between vectors) but discards the original matrix's absolute
//! orientation/rotation. For trajectories whose box wasn't already written
//! in that canonical orientation — common for files converted from AMBER,
//! as opposed to ones GROMACS itself wrote — mdtraj's reported matrix and
//! this module's raw matrix will differ even though they describe the
//! identical periodic cell: reducing either one back to lengths+angles
//! (e.g. via `mdtraj.utils.box_vectors_to_lengths_and_angles`) reproduces
//! the same 6 numbers to float32 precision. This was root-caused after
//! proxide#16 reported the raw matrix as "corrupted" purely from a naive
//! element-wise comparison against mdtraj's reoriented one; see
//! `test_xtc_box_vectors_match_ground_truth_across_full_trajectory` in
//! `tests/xtc_tests.rs` for the full writeup and the byte-level/
//! low-level-mdtraj cross-checks that ruled out an actual decode bug.
//! Comparing this module's box vectors against mdtraj element-wise is only
//! valid once both are reduced to (or reconstructed from) lengths+angles,
//! or the file's box is already GROMACS-canonical.

use molly::selection::AtomSelection;
use molly::{Frame as MollyFrame, XTCReader};
use std::fs::File;
use std::io;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;
use thiserror::Error;

#[cfg(feature = "parallel")]
use orx_parallel::{IntoParIter, ParIter};

#[cfg(feature = "parallel")]
use proxide_geometry::geometry::distances::{pairwise_distances_mic, BoxDims};

#[cfg(test)]
mod tests {
    include!("tests/xtc_tests.rs");
}

/// Errors from the lazy [`XtcReader`] cursor and offset-cache logic.
#[derive(Error, Debug)]
pub enum XtcError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("bincode error: {0}")]
    Bincode(#[from] bincode::Error),
    #[error("frame index {index} out of bounds ({n_frames} frames)")]
    FrameOutOfBounds { index: usize, n_frames: usize },
    #[error("atom index {index} out of bounds ({n_atoms} atoms)")]
    AtomIndexOutOfBounds { index: usize, n_atoms: usize },
}

/// XTC trajectory data structure
#[derive(Debug, Clone)]
pub struct XtcTrajectory {
    /// Number of frames
    pub num_frames: usize,
    /// Number of atoms per frame
    pub num_atoms: usize,
    /// Time for each frame in ps
    pub times: Vec<f32>,
    /// Coordinates for each frame (n_frames x n_atoms*3) in Angstroms
    pub coords: Vec<Vec<f32>>,
    /// Box vectors for each frame (n_frames x 3x3), if present
    pub boxes: Vec<[[f32; 3]; 3]>,
}

/// Coordinates (flat, Angstrom-scale) + 3x3 box vectors for one frame.
type AngstromFrame = (Vec<f32>, [[f32; 3]; 3]);

/// Convert a decoded molly [`MollyFrame`] into Angstrom-scale coordinates and box vectors.
fn frame_to_angstroms(frame: &MollyFrame) -> AngstromFrame {
    let coords: Vec<f32> = frame.positions.iter().map(|x| x * 10.0).collect();
    let mut box_ang = frame.boxvec_cols_2d();
    for row in &mut box_ang {
        for v in row {
            *v *= 10.0;
        }
    }
    (coords, box_ang)
}

// ---------------------------------------------------------------------------
// On-disk offset-index cache (MDAnalysis-style: mtime + size + natoms validated)
// ---------------------------------------------------------------------------

const OFFSET_CACHE_MAGIC: u32 = 0x5058_5443; // "PXTC"

/// v2: `scan_and_cache_offsets` gained `drop_trailing_frame_if_truncated` and
/// `determine_offsets_tolerant`'s metadata-region EOF tolerance. Bumped so a
/// sidecar written by a pre-v2 build — which may have a phantom offset for a
/// truncated trailing frame baked in — is treated as a version mismatch by
/// `OffsetCache::matches` and rescanned, instead of being trusted forever.
const OFFSET_CACHE_VERSION: u16 = 2;

/// The historical `OFFSET_CACHE_VERSION` value, before the bump to 2 above.
/// Exists only so `test_stale_pre_fix_offset_cache_is_rescanned_not_trusted`
/// can construct a byte-for-byte accurate "sidecar written by a pre-fix
/// build". Never update this alongside a future `OFFSET_CACHE_VERSION` bump
/// — it is a historical fact, not a second copy of the current value — and
/// never delete it just because nothing but that test reads it; doing either
/// would silently defang the regression test it exists for.
#[cfg(test)]
const OFFSET_CACHE_VERSION_PRE_TRUNCATION_FIX: u16 = 1;

#[derive(serde::Serialize, serde::Deserialize)]
struct OffsetCache {
    magic: u32,
    version: u16,
    mtime_secs: u64,
    mtime_nanos: u32,
    file_size: u64,
    natoms: usize,
    offsets: Vec<u64>,
}

impl OffsetCache {
    fn sidecar_path(traj_path: &Path) -> PathBuf {
        let mut s = traj_path.as_os_str().to_owned();
        s.push(".offsets");
        PathBuf::from(s)
    }

    fn from_metadata(
        meta: &std::fs::Metadata,
        natoms: usize,
        offsets: Vec<u64>,
    ) -> io::Result<Self> {
        let modified = meta.modified()?;
        let dur = modified.duration_since(UNIX_EPOCH).unwrap_or_default();
        Ok(Self {
            magic: OFFSET_CACHE_MAGIC,
            version: OFFSET_CACHE_VERSION,
            mtime_secs: dur.as_secs(),
            mtime_nanos: dur.subsec_nanos(),
            file_size: meta.len(),
            natoms,
            offsets,
        })
    }

    fn matches(&self, meta: &std::fs::Metadata, natoms: usize) -> bool {
        if self.magic != OFFSET_CACHE_MAGIC || self.version != OFFSET_CACHE_VERSION {
            return false;
        }
        if self.natoms != natoms || self.file_size != meta.len() {
            return false;
        }
        match meta.modified() {
            Ok(modified) => {
                let dur = modified.duration_since(UNIX_EPOCH).unwrap_or_default();
                dur.as_secs() == self.mtime_secs && dur.subsec_nanos() == self.mtime_nanos
            }
            Err(_) => false,
        }
    }

    /// MDAnalysis `XDRBaseReader` sidecar: `.{basename}_offsets.npz`
    /// (e.g. `production.xtc` → `.production.xtc_offsets.npz`).
    fn mda_sidecar_path(traj_path: &Path) -> PathBuf {
        let parent = traj_path.parent().unwrap_or_else(|| Path::new("."));
        let name = traj_path
            .file_name()
            .map(|n| n.to_owned())
            .unwrap_or_default();
        let mut hidden = std::ffi::OsString::from(".");
        hidden.push(&name);
        hidden.push("_offsets.npz");
        parent.join(hidden)
    }

    /// Load and validate the sidecar for `traj_path`. Returns `None` on any
    /// miss, mismatch, or corruption — callers fall back to a rescan.
    ///
    /// On a proxide `.offsets` miss, also tries an MDAnalysis
    /// `.xtc_offsets.npz` sidecar. Matching size + natoms converts it into a
    /// proxide bincode sidecar (mtime taken from `stat()` at convert time —
    /// MDA's `ctime` is intentionally ignored).
    fn load(traj_path: &Path, meta: &std::fs::Metadata, natoms: usize) -> Option<Self> {
        let sidecar = Self::sidecar_path(traj_path);
        if let Ok(file) = File::open(sidecar) {
            if let Ok(cache) = bincode::deserialize_from::<_, Self>(io::BufReader::new(file)) {
                if cache.matches(meta, natoms) {
                    return Some(cache);
                }
            }
        }
        Self::try_load_mda_npz(traj_path, meta, natoms)
    }

    /// Best-effort import of an MDAnalysis offset npz. Any failure returns
    /// `None` so the caller falls through to `determine_offsets`.
    fn try_load_mda_npz(traj_path: &Path, meta: &std::fs::Metadata, natoms: usize) -> Option<Self> {
        let mda_path = Self::mda_sidecar_path(traj_path);
        let file = File::open(&mda_path).ok()?;
        let mut npz = ndarray_npy::NpzReader::new(file).ok()?;

        let offsets_arr: ndarray::Array1<i64> = npz.by_name("offsets").ok()?;
        if offsets_arr.is_empty() {
            return None;
        }

        let file_size = Self::npz_scalar_as_u64(&mut npz, "size")?;
        let n_atoms = Self::npz_scalar_as_u64(&mut npz, "n_atoms")? as usize;
        if file_size != meta.len() || n_atoms != natoms {
            return None;
        }

        let offsets: Vec<u64> = offsets_arr.iter().map(|&x| x as u64).collect();
        let cache = Self::from_metadata(meta, natoms, offsets).ok()?;
        // Persist proxide sidecar so the next open skips both MDA and rescan.
        if cache.store(traj_path).is_ok() {
            log::info!(
                "Imported MDAnalysis offset cache {} → {}",
                mda_path.display(),
                Self::sidecar_path(traj_path).display()
            );
        }
        Some(cache)
    }

    /// Read a 0-d (or length-1) integer array from an NPZ under `name`.
    /// Tries i64 then i32 — MDAnalysis writes `size` as `<i8` and `n_atoms` as `<i4`.
    fn npz_scalar_as_u64<R: io::Read + io::Seek>(
        npz: &mut ndarray_npy::NpzReader<R>,
        name: &str,
    ) -> Option<u64> {
        if let Ok(arr) = npz.by_name::<ndarray::OwnedRepr<i64>, ndarray::Ix0>(name) {
            return Some(arr.into_scalar() as u64);
        }
        if let Ok(arr) = npz.by_name::<ndarray::OwnedRepr<i64>, ndarray::Ix1>(name) {
            if arr.len() == 1 {
                return Some(arr[0] as u64);
            }
        }
        if let Ok(arr) = npz.by_name::<ndarray::OwnedRepr<i32>, ndarray::Ix0>(name) {
            return Some(arr.into_scalar() as u64);
        }
        if let Ok(arr) = npz.by_name::<ndarray::OwnedRepr<i32>, ndarray::Ix1>(name) {
            if arr.len() == 1 {
                return Some(arr[0] as u64);
            }
        }
        None
    }

    /// Persist via temp-file-then-rename, so a concurrent reader never observes
    /// a partially-written sidecar. Best-effort: failures (e.g. read-only data
    /// directory) are not fatal — the caller already has the offsets in memory.
    fn store(&self, traj_path: &Path) -> io::Result<()> {
        let sidecar = Self::sidecar_path(traj_path);
        let tmp = sidecar.with_extension("offsets.tmp");
        {
            let file = File::create(&tmp)?;
            bincode::serialize_into(io::BufWriter::new(file), self)
                .map_err(|e| io::Error::other(e.to_string()))?;
        }
        std::fs::rename(tmp, sidecar)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Lazy, seekable cursor (MDAnalysis-style) over an XTC trajectory
// ---------------------------------------------------------------------------

/// Lazy, seekable cursor over an XTC trajectory.
///
/// Unlike [`XtcTrajectory`], opening a reader does not decode any frame data.
/// Frame byte offsets come from a cheap header-only scan
/// (`molly::XTCReader::determine_offsets`, no coordinate decompression) and are
/// cached in memory plus persisted to a `<path>.offsets` sidecar so repeated
/// opens of the same trajectory (a common workflow) skip the rescan.
pub struct XtcReader {
    path: PathBuf,
    reader: XTCReader<File>,
    offsets: Option<Vec<u64>>,
    natoms: Option<usize>,
}

impl XtcReader {
    /// Open a trajectory. Does not scan for offsets or decode any frame yet.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, XtcError> {
        let path = path.as_ref().to_path_buf();
        let reader = XTCReader::open(&path)?;
        Ok(Self {
            path,
            reader,
            offsets: None,
            natoms: None,
        })
    }

    /// Number of frames. Triggers (and caches) the offset scan on first call;
    /// free afterwards. Never decodes coordinate data.
    pub fn frame_count(&mut self) -> Result<usize, XtcError> {
        Ok(self.ensure_offsets()?.len())
    }

    /// Number of atoms per frame, from the first frame's header.
    pub fn n_atoms(&mut self) -> Result<usize, XtcError> {
        self.ensure_offsets()?;
        Ok(self.natoms.unwrap_or(0))
    }

    /// Force a rescan of frame offsets, bypassing (and overwriting) any cached
    /// sidecar. Equivalent to MDAnalysis's `Universe(..., refresh_offsets=True)`.
    pub fn refresh_offsets(&mut self) -> Result<usize, XtcError> {
        self.offsets = None;
        self.natoms = None;
        self.scan_and_cache_offsets()?;
        Ok(self.offsets.as_ref().map(Vec::len).unwrap_or(0))
    }

    /// Test-only accessor for `determine_offsets_tolerant`'s raw output,
    /// deliberately bypassing `drop_trailing_frame_if_truncated` (unlike
    /// `ensure_offsets`/`frame_count`) so tests can assert byte-exact parity
    /// against `molly::XTCReader::determine_offsets` on `determine_offsets_tolerant`
    /// in isolation. Going through the full pipeline would let
    /// `drop_trailing_frame_if_truncated`'s own probe-decode-and-pop mask an
    /// off-by-one in `determine_offsets_tolerant` that happens to add
    /// exactly one bogus trailing offset — indistinguishable, from that
    /// guard's point of view, from a genuinely truncated last frame.
    #[cfg(test)]
    fn determine_offsets_tolerant_for_test(&mut self) -> Result<Vec<u64>, XtcError> {
        self.reader.home()?;
        self.determine_offsets_tolerant()
    }

    fn ensure_offsets(&mut self) -> Result<&Vec<u64>, XtcError> {
        if self.offsets.is_none() {
            let natoms = self.peek_natoms()?;
            let meta = std::fs::metadata(&self.path)?;
            if let Some(cache) = OffsetCache::load(&self.path, &meta, natoms) {
                self.natoms = Some(cache.natoms);
                self.offsets = Some(cache.offsets);
            } else {
                self.scan_and_cache_offsets()?;
            }
        }
        Ok(self.offsets.as_ref().expect("offsets populated above"))
    }

    /// Read just the first frame's header to learn `natoms`, then rewind.
    fn peek_natoms(&mut self) -> io::Result<usize> {
        self.reader.home()?;
        let header = self.reader.read_header()?;
        self.reader.home()?;
        Ok(header.natoms)
    }

    /// Header-only scan (no coordinate decompression) via molly's
    /// `determine_offsets`, then persist the result to the sidecar cache.
    fn scan_and_cache_offsets(&mut self) -> Result<(), XtcError> {
        let natoms = self.peek_natoms()?;
        self.reader.home()?;
        let mut offsets = self.determine_offsets_tolerant()?;
        self.drop_trailing_frame_if_truncated(&mut offsets)?;
        let meta = std::fs::metadata(&self.path)?;
        if let Ok(cache) = OffsetCache::from_metadata(&meta, natoms, offsets.clone()) {
            // Best-effort: a failure to persist (e.g. read-only directory)
            // must not fail the read itself, since offsets are already in hand.
            let _ = cache.store(&self.path);
        }
        self.natoms = Some(natoms);
        self.offsets = Some(offsets);
        Ok(())
    }

    /// Header-only frame-offset scan, mirroring `molly::XTCReader::determine_offsets`
    /// but tolerant of a truncated *last* frame's compression-metadata/`nbytes`
    /// field, not just its header.
    ///
    /// molly's own `determine_offsets_exclusive` already tolerates
    /// `UnexpectedEof` from `read_header` (a frame whose header itself was
    /// never fully written stops the scan cleanly). But the very next call in
    /// its loop, `skip_positions`, propagates any I/O error — including
    /// `UnexpectedEof` from the `nbytes` read inside the ~32-40 byte
    /// compression-metadata block that follows a fully-written header — via a
    /// bare `?`. A crash landing in that narrow window (rather than inside
    /// the bulk compressed-coordinate payload, which `read_frame_at_offset`
    /// in [`Self::drop_trailing_frame_if_truncated`] already catches) made
    /// the whole scan fail hard instead of simply stopping one frame short.
    ///
    /// This reimplements the same header+skip loop using molly's own public
    /// `read_header`/`skip_positions` methods, treating `UnexpectedEof` from
    /// either as "stop, that frame was never fully written" — the frame is
    /// left out of the returned offsets rather than raising.
    ///
    /// Only the loop's error-tolerance is reimplemented; all byte-level
    /// frame parsing still goes through molly's own `read_header`/
    /// `skip_positions`. `test_determine_offsets_tolerant_matches_molly_determine_offsets`
    /// asserts this produces byte-identical output to
    /// `molly::XTCReader::determine_offsets` on every well-formed fixture —
    /// if a future `molly` upgrade changes the header/skip_positions
    /// contract this assumes, that is the test that catches it.
    fn determine_offsets_tolerant(&mut self) -> Result<Vec<u64>, XtcError> {
        let mut exclusive_offsets = Vec::new();
        loop {
            let header = match self.reader.read_header() {
                Ok(header) => header,
                Err(err) if err.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(err) => return Err(err.into()),
            };
            match self.reader.skip_positions(&header) {
                Ok(offset) => exclusive_offsets.push(offset),
                Err(err) if err.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(err) => return Err(err.into()),
            }
        }
        let mut offsets = vec![0u64];
        let complete = exclusive_offsets.len().saturating_sub(1);
        offsets.extend_from_slice(&exclusive_offsets[..complete]);
        Ok(offsets)
    }

    /// Drop the last entry of `offsets` if it points to a frame that doesn't
    /// actually decode.
    ///
    /// `determine_offsets`'s header-only scan finds each frame's start by
    /// `seek`-ing forward past its *declared* compressed-coordinate length
    /// (from the header's `nbytes` field) without ever reading that data.
    /// POSIX `seek` past EOF succeeds silently, so a trailing frame whose
    /// header is fully written but whose coordinate payload is only
    /// partially flushed — the normal state of the last frame of a
    /// trajectory a live simulation is still appending to — gets an offset
    /// pushed for it exactly as if it were complete. The *next* iteration's
    /// header read then hits real EOF and the scan stops, but by then the
    /// phantom offset is already in the list.
    ///
    /// This does one real decode of just the last frame to confirm its data
    /// is actually present on disk, and removes it if not. Cheap (one frame,
    /// not the whole file) and reuses molly's own tested decode path rather
    /// than re-deriving the frame-length arithmetic ourselves.
    fn drop_trailing_frame_if_truncated(&mut self, offsets: &mut Vec<u64>) -> Result<(), XtcError> {
        let Some(&last_offset) = offsets.last() else {
            return Ok(());
        };
        let mut probe = MollyFrame::default();
        let decodes = self
            .reader
            .read_frame_at_offset::<false>(&mut probe, last_offset, &AtomSelection::All)
            .is_ok();
        self.reader.home()?;
        if !decodes {
            offsets.pop();
        }
        Ok(())
    }

    /// Decode the frame at `index`, seeking directly via the cached offset —
    /// O(1) once the offset index exists.
    pub fn read_frame_at(
        &mut self,
        index: usize,
        selection: &AtomSelection,
    ) -> Result<MollyFrame, XtcError> {
        let offsets = self.ensure_offsets()?;
        let n_frames = offsets.len();
        let offset = *offsets
            .get(index)
            .ok_or(XtcError::FrameOutOfBounds { index, n_frames })?;
        let mut frame = MollyFrame::default();
        self.reader
            .read_frame_at_offset::<false>(&mut frame, offset, selection)?;
        Ok(frame)
    }

    /// Materialize a bounded eager batch (mdtraj `iterload`-equivalent),
    /// without ever loading frames outside `range`.
    pub fn read_range(&mut self, range: Range<usize>) -> Result<XtcTrajectory, XtcError> {
        let n_frames = self.frame_count()?;
        let end = range.end.min(n_frames);
        let mut times = Vec::new();
        let mut coords = Vec::new();
        let mut boxes = Vec::new();
        let mut num_atoms = 0;
        for i in range.start..end {
            let frame = self.read_frame_at(i, &AtomSelection::All)?;
            let (c, b) = frame_to_angstroms(&frame);
            num_atoms = c.len() / 3;
            times.push(frame.time);
            coords.push(c);
            boxes.push(b);
        }
        Ok(XtcTrajectory {
            num_frames: coords.len(),
            num_atoms,
            times,
            coords,
            boxes,
        })
    }

    /// Sequential iteration over every frame, without materializing the whole
    /// trajectory in memory.
    pub fn iter(&mut self) -> Result<XtcFrameIter<'_>, XtcError> {
        let n_frames = self.frame_count()?;
        Ok(XtcFrameIter {
            reader: self,
            next_index: 0,
            n_frames,
        })
    }
}

/// Sequential frame iterator produced by [`XtcReader::iter`].
pub struct XtcFrameIter<'a> {
    reader: &'a mut XtcReader,
    next_index: usize,
    n_frames: usize,
}

impl Iterator for XtcFrameIter<'_> {
    type Item = Result<MollyFrame, XtcError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_index >= self.n_frames {
            return None;
        }
        let result = self
            .reader
            .read_frame_at(self.next_index, &AtomSelection::All);
        self.next_index += 1;
        Some(result)
    }
}

/// A single decoded XTC frame, already reduced to the caller's requested
/// atoms (or the full atom set, if no selection was requested). Coordinates
/// remain nanometer-scale (matching [`MollyFrame::positions`]) — Angstrom
/// conversion is a binding-layer concern, same as [`AngstromFrame`] elsewhere
/// in this module.
///
/// Deliberately not just a re-exported [`MollyFrame`]: the whole point of
/// this type is that it's sized to `atom_indices.len()` (or the full atom
/// count when unselected), not the trajectory's full per-frame atom count —
/// see [`read_frames_parallel`]'s doc comment for why that distinction is the
/// entire fix for the OOM this type exists to prevent.
#[derive(Debug, Clone)]
pub struct SelectedFrame {
    /// Flat (x, y, z)-per-atom positions in nanometers, in exactly the order
    /// `atom_indices` specified (duplicates preserved) — or the full frame's
    /// positions, unselected, when no `atom_indices` were given.
    pub positions: Vec<f32>,
    /// Column-major box vectors in nanometers (always the full 3x3 — never
    /// atom-selected, and cheap regardless of atom count).
    pub box_vectors: [[f32; 3]; 3],
    /// Frame time in picoseconds.
    pub time: f32,
}

/// Select atoms from a decoded frame's positions, preserving `atom_indices`'
/// exact order and duplicates (mdtraj fancy-indexing semantics) rather than
/// the ascending-order/deduped semantics of
/// `molly::selection::AtomSelection::from_index_list`'s `Mask`. Mirrors
/// `proxide_py::py_xtc_reader::frame_to_angstroms_selected`'s selection logic
/// (that copy operates on already-Angstrom-scaled coordinates behind PyO3
/// types, so it can't be reused directly from this crate).
fn select_positions(
    frame: &MollyFrame,
    atom_indices: Option<&[usize]>,
) -> Result<Vec<f32>, XtcError> {
    match atom_indices {
        Some(indices) => {
            let n_atoms = frame.positions.len() / 3;
            let mut out = Vec::with_capacity(indices.len() * 3);
            for &i in indices {
                if i >= n_atoms {
                    return Err(XtcError::AtomIndexOutOfBounds { index: i, n_atoms });
                }
                let base = i * 3;
                out.extend_from_slice(&frame.positions[base..base + 3]);
            }
            Ok(out)
        }
        None => Ok(frame.positions.clone()),
    }
}

/// Decode a set of frames concurrently via `orx-parallel`. Each worker opens
/// its own file handle and seeks directly to its assigned offset (the same
/// per-worker-handle pattern MDAnalysis's multiprocessing/Dask analysis
/// backends use); only already-computed offsets are shared across workers.
///
/// `atom_indices` is applied *inside* each worker, immediately after that
/// worker's single [`MollyFrame`] is decoded and before it returns — the
/// full-atom-count `MollyFrame` is dropped at the end of the worker closure,
/// so the `Vec` this function ultimately collects and returns holds only
/// [`SelectedFrame`]s sized to `atom_indices.len()` (or the full atom count,
/// when `atom_indices` is `None`), never `n_requested_frames` full-atom-count
/// frames all alive simultaneously. That "collect everything at full atom
/// count, filter after the fact" pattern was the actual OOM bug this
/// function existed to fix (praxia debt #1220): for a system with
/// `n_full_atoms` in the hundreds of thousands, requesting a modest
/// `atom_indices` subset of e.g. a few hundred atoms across many frames
/// still allocated `n_requested_frames * n_full_atoms * 3 * 4` bytes at the
/// old collection step, before any filtering ran at all.
///
/// Order and duplicates in `atom_indices` are preserved exactly as passed —
/// see [`select_positions`].
#[cfg(feature = "parallel")]
pub fn read_frames_parallel<P: AsRef<Path>>(
    path: P,
    frame_indices: &[usize],
    atom_indices: Option<&[usize]>,
) -> Result<Vec<SelectedFrame>, XtcError> {
    let path = path.as_ref();
    let mut reader = XtcReader::open(path)?;
    let offsets = reader.ensure_offsets()?.clone();
    let n_frames = offsets.len();

    let indices = frame_indices.to_vec();
    let par = indices
        .into_par()
        .map(|index| -> Result<SelectedFrame, XtcError> {
            let offset = *offsets
                .get(index)
                .ok_or(XtcError::FrameOutOfBounds { index, n_frames })?;
            let mut worker_reader = XTCReader::open(path)?;
            let mut frame = MollyFrame::default();
            worker_reader.read_frame_at_offset::<false>(&mut frame, offset, &AtomSelection::All)?;
            // `frame` (full atom count) is filtered here, inside the worker,
            // and dropped at the end of this closure — it never travels back
            // to the caller's collected Vec at full size.
            let positions = select_positions(&frame, atom_indices)?;
            let box_vectors = frame.boxvec_cols_2d();
            let time = frame.time;
            Ok(SelectedFrame {
                positions,
                box_vectors,
                time,
            })
        });
    #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
    let par = par.num_threads(proxide_parallel_rt::num_threads());
    par.collect::<Vec<_>>().into_iter().collect()
}

/// Compute per-frame pairwise-distance "distograms" for a caller-selected
/// set of atoms (typically one representative atom per residue, e.g. Cα),
/// using the minimum-image convention against each frame's own box.
///
/// A thin XTC-specific wrapper: it reuses [`read_frames_parallel`] for the
/// actual decode + atom-filter-inside-worker step (the same OOM fix from
/// praxia debt #1220 — atoms are filtered before frames are collected, so
/// memory scales with `atom_indices.len()`, not the trajectory's full
/// per-frame atom count), then hands each frame's selected positions and
/// box to [`pairwise_distances_mic`] — a system-agnostic primitive that
/// knows nothing about chains, residues, or XTC itself. A frame with no
/// real periodic box (zero/degenerate box vectors — e.g. an implicit-
/// solvent or in-vacuo trajectory) automatically falls back to plain
/// Euclidean distance there; this function does not need its own special
/// case for that.
///
/// Returns one `Vec<f32>` of length `n * (n - 1) / 2` per requested frame
/// (`n = atom_indices.len()`), in Angstroms, with pair order matching
/// `numpy.triu_indices(n, k=1)` (row-major over the upper triangle) — same
/// convention as [`pairwise_distances_mic`]. Order and duplicates in
/// `atom_indices` are preserved exactly as passed (mdtraj fancy-indexing
/// semantics, same as [`read_frames_parallel`]) — duplicate indices would
/// simply produce zero-distance pairs, not an error.
#[cfg(feature = "parallel")]
pub fn read_xtc_distogram_parallel<P: AsRef<Path>>(
    path: P,
    frame_indices: &[usize],
    atom_indices: &[usize],
) -> Result<Vec<Vec<f32>>, XtcError> {
    let frames = read_frames_parallel(path, frame_indices, Some(atom_indices))?;

    Ok(frames
        .iter()
        .map(|frame| {
            let positions: Vec<[f32; 3]> = frame
                .positions
                .as_chunks::<3>()
                .0
                .iter()
                .map(|c| [c[0] * 10.0, c[1] * 10.0, c[2] * 10.0]) // nm -> Angstrom
                .collect();

            let mut box_ang = frame.box_vectors;
            for row in &mut box_ang {
                for v in row {
                    *v *= 10.0; // nm -> Angstrom
                }
            }
            let box_dims = BoxDims::from_diagonal_matrix(&box_ang);

            pairwise_distances_mic(&positions, Some(&box_dims))
        })
        .collect())
}

pub mod molly_impl {
    use super::*;

    /// Build a frame offset index for an XTC file via molly's header-only scan
    /// (`XTCReader::determine_offsets`) — no coordinate decompression. Kept as
    /// a standalone function for backward compatibility; prefer [`XtcReader`]
    /// for new code, which additionally caches this to a disk sidecar.
    pub fn determine_offsets<P: AsRef<Path>>(
        path: P,
    ) -> Result<Box<[u64]>, Box<dyn std::error::Error>> {
        let mut reader = XTCReader::open(path.as_ref())?;
        Ok(reader.determine_offsets(None)?)
    }

    /// Read an XTC file using pure-Rust molly crate
    /// Returns coordinates in Angstroms.
    pub fn read_xtc_molly<P: AsRef<Path>>(
        path: P,
    ) -> Result<XtcTrajectory, Box<dyn std::error::Error>> {
        let mut xtc_reader = XTCReader::open(path.as_ref())?;
        let molly_frames = xtc_reader.read_all_frames()?;

        let num_frames = molly_frames.len();
        if num_frames == 0 {
            return Ok(XtcTrajectory {
                num_frames: 0,
                num_atoms: 0,
                times: Vec::new(),
                coords: Vec::new(),
                boxes: Vec::new(),
            });
        }

        let num_atoms = molly_frames[0].positions.len() / 3;
        let mut times = Vec::with_capacity(num_frames);
        let mut all_coords = Vec::with_capacity(num_frames);
        let mut all_boxes = Vec::with_capacity(num_frames);

        for frame in molly_frames.iter() {
            let (coords, box_ang) = frame_to_angstroms(frame);
            times.push(frame.time);
            all_coords.push(coords);
            all_boxes.push(box_ang);
        }

        Ok(XtcTrajectory {
            num_frames,
            num_atoms,
            times,
            coords: all_coords,
            boxes: all_boxes,
        })
    }

    /// Read a specific frame from XTC file via O(1) offset-based seeking.
    /// Returns (coordinates, box_vectors).
    pub fn read_frame_at<P: AsRef<Path>>(
        path: P,
        frame_index: usize,
    ) -> Result<Option<AngstromFrame>, Box<dyn std::error::Error>> {
        let mut reader = XtcReader::open(path)?;
        let n_frames = reader.frame_count()?;
        if frame_index >= n_frames {
            return Ok(None);
        }
        let frame = reader.read_frame_at(frame_index, &AtomSelection::All)?;
        Ok(Some(frame_to_angstroms(&frame)))
    }
}

/// XTC trajectory file writer stub
pub struct XtcWriter {
    // TODO: Implement XTC writing with XDR encoding and lossy compression.
    // This requires a pure-Rust XDR implementation and the XTC compression algorithm.
}

impl XtcWriter {
    /// Create a new XTC writer
    pub fn create<P: AsRef<std::path::Path>>(_path: P) -> Result<Self, Box<dyn std::error::Error>> {
        // TODO: Implement file creation and header writing
        Err("XTC writing is not yet implemented. Use DCD or NPZ for trajectory output.".into())
    }

    /// Write a single frame to the XTC file
    pub fn write_frame(
        &mut self,
        _time: f32,
        _coords: &[f32],
    ) -> Result<(), Box<dyn std::error::Error>> {
        // TODO: Implement frame compression and XDR writing
        Err("XTC writing is not yet implemented.".into())
    }
}
