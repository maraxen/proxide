//! XTC trajectory file format parser using the molly crate.
//!
//! XTC uses XDR encoding with lossy compression of coordinates.
//! Format reference: https://manual.gromacs.org/current/reference-manual/file-formats.html#xtc

use molly::XTCReader;
use std::path::Path;

#[cfg(test)]
mod tests {
    include!("tests/xtc_tests.rs");
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

pub mod molly_impl {
    use super::*;

    /// Build a frame offset index for XTC file.
    /// XTC has no random-access header, so we must scan the file to find frame boundaries.
    ///
    /// LIMITATION: The molly crate's XTCReader does not expose frame byte positions
    /// during parsing, making it impossible to build a true byte-offset index without
    /// reimplementing the XDR/XTC parsing layer with position tracking.
    ///
    /// WORKAROUND: Callers must use the eager-load path via read_xtc_molly() and cache
    /// the entire trajectory in memory (TrajectoryIndexCache in src-tauri). This is
    /// acceptable for typical use (frames ≤ 500), as documented in MAX_EAGER_FRAMES.
    ///
    /// For future random-access support, we would need to either:
    /// 1. Extend molly to expose XTCReader.stream_position() or frame boundaries, or
    /// 2. Implement a custom XTCReader wrapper that tracks positions during read_all_frames()
    pub fn determine_offsets<P: AsRef<Path>>(
        path: P,
    ) -> Result<Box<[u64]>, Box<dyn std::error::Error>> {
        let mut reader = XTCReader::open(path.as_ref())?;
        let frames = reader.read_all_frames()?;

        // Return placeholder offsets (all zeros) since true byte positions are unavailable.
        // Callers must not rely on these for seeking; use eager loading instead.
        Ok(vec![0u64; frames.len()].into_boxed_slice())
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
            times.push(frame.time);
            // Convert to Angstroms (multiply by 10.0)
            let coords: Vec<f32> = frame.positions.iter().map(|x| x * 10.0).collect();
            all_coords.push(coords);

            // Extract box vectors from molly's Mat3 (which is [[f32; 3]; 3])
            // Convert from nm to Angstroms (multiply by 10.0)
            let box_mat = frame.boxvec;
            let box_ang: [[f32; 3]; 3] = [
                [
                    box_mat.x_axis[0] * 10.0,
                    box_mat.x_axis[1] * 10.0,
                    box_mat.x_axis[2] * 10.0,
                ],
                [
                    box_mat.y_axis[0] * 10.0,
                    box_mat.y_axis[1] * 10.0,
                    box_mat.y_axis[2] * 10.0,
                ],
                [
                    box_mat.z_axis[0] * 10.0,
                    box_mat.z_axis[1] * 10.0,
                    box_mat.z_axis[2] * 10.0,
                ],
            ];
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

    /// Read a specific frame from XTC file.
    /// Since XTC has no built-in random access, we read all frames and extract the one we want.
    /// Returns (coordinates, box_vectors).
    pub fn read_frame_at<P: AsRef<Path>>(
        path: P,
        frame_index: usize,
    ) -> Result<Option<(Vec<f32>, [[f32; 3]; 3])>, Box<dyn std::error::Error>> {
        let traj = read_xtc_molly(path)?;
        if frame_index >= traj.num_frames {
            return Ok(None);
        }
        Ok(Some((
            traj.coords[frame_index].clone(),
            traj.boxes[frame_index],
        )))
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
