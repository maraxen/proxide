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
}

pub mod molly_impl {
    use super::*;

    /// Build a frame offset index for XTC file.
    /// XTC has no random-access header, so we scan the file to find frame boundaries.
    /// Returns Vec<(offset, time)> for each frame.
    pub fn determine_offsets<P: AsRef<Path>>(
        path: P,
    ) -> Result<Box<[u64]>, Box<dyn std::error::Error>> {
        let mut reader = XTCReader::open(path.as_ref())?;
        let frames = reader.read_all_frames()?;

        // For now, we return empty offsets since molly doesn't expose frame byte positions.
        // The real implementation will read all frames and build the index,
        // or we'll extend molly to expose this information.
        // For the Tauri layer, we'll do eager reading and cache frame data instead.
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
            });
        }

        let num_atoms = molly_frames[0].positions.len() / 3;
        let mut times = Vec::with_capacity(num_frames);
        let mut all_coords = Vec::with_capacity(num_frames);

        for frame in molly_frames.iter() {
            times.push(frame.time);
            // Convert to Angstroms (multiply by 10.0)
            let coords: Vec<f32> = frame.positions.iter().map(|x| x * 10.0).collect();
            all_coords.push(coords);
        }

        Ok(XtcTrajectory {
            num_frames,
            num_atoms,
            times,
            coords: all_coords,
        })
    }

    /// Read a specific frame from XTC file.
    /// Since XTC has no built-in random access, we read all frames and extract the one we want.
    pub fn read_frame_at<P: AsRef<Path>>(
        path: P,
        frame_index: usize,
    ) -> Result<Option<(Vec<f32>, f32)>, Box<dyn std::error::Error>> {
        let traj = read_xtc_molly(path)?;
        if frame_index >= traj.num_frames {
            return Ok(None);
        }
        Ok(Some((traj.coords[frame_index].clone(), traj.times[frame_index])))
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
