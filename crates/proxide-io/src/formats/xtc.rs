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
const OFFSET_CACHE_VERSION: u16 = 1;

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

    /// Load and validate the sidecar for `traj_path`. Returns `None` on any
    /// miss, mismatch, or corruption — callers fall back to a rescan.
    fn load(traj_path: &Path, meta: &std::fs::Metadata, natoms: usize) -> Option<Self> {
        let sidecar = Self::sidecar_path(traj_path);
        let file = File::open(sidecar).ok()?;
        let cache: Self = bincode::deserialize_from(io::BufReader::new(file)).ok()?;
        if cache.matches(meta, natoms) {
            Some(cache)
        } else {
            None
        }
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
        let offsets = self.reader.determine_offsets(None)?.into_vec();
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
