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

/// Convert a decoded molly [`MollyFrame`] into Angstrom-scale coordinates and box vectors.
fn frame_to_angstroms(frame: &MollyFrame) -> (Vec<f32>, [[f32; 3]; 3]) {
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
    fn try_load_mda_npz(
        traj_path: &Path,
        meta: &std::fs::Metadata,
        natoms: usize,
    ) -> Option<Self> {
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

/// Decode a set of frames concurrently via `orx-parallel`. Each worker opens
/// its own file handle and seeks directly to its assigned offset (the same
/// per-worker-handle pattern MDAnalysis's multiprocessing/Dask analysis
/// backends use); only already-computed offsets are shared across workers.
#[cfg(feature = "parallel")]
pub fn read_frames_parallel<P: AsRef<Path>>(
    path: P,
    frame_indices: &[usize],
) -> Result<Vec<MollyFrame>, XtcError> {
    let path = path.as_ref();
    let mut reader = XtcReader::open(path)?;
    let offsets = reader.ensure_offsets()?.clone();
    let n_frames = offsets.len();

    let indices = frame_indices.to_vec();
    let par = indices
        .into_par()
        .map(|index| -> Result<MollyFrame, XtcError> {
            let offset = *offsets
                .get(index)
                .ok_or(XtcError::FrameOutOfBounds { index, n_frames })?;
            let mut worker_reader = XTCReader::open(path)?;
            let mut frame = MollyFrame::default();
            worker_reader.read_frame_at_offset::<false>(&mut frame, offset, &AtomSelection::All)?;
            Ok(frame)
        });
    #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
    let par = par.num_threads(proxide_parallel_rt::num_threads());
    par.collect::<Vec<_>>().into_iter().collect()
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
    ) -> Result<Option<(Vec<f32>, [[f32; 3]; 3])>, Box<dyn std::error::Error>> {
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
