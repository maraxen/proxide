// TODO: Review allow attributes at a later point
#![allow(clippy::needless_range_loop)]

//! Pure-Rust TRR trajectory format parser
//!
//! TRR is a binary format used by GROMACS.
//! This implementation is pure-Rust and does not depend on chemfiles.

use super::xdr::XdrReader;

use std::fs::File;
use std::io::{BufReader, Read, Seek};
use thiserror::Error;

#[cfg(test)]
mod tests {
    include!("tests/trr_tests.rs");
}

#[derive(Error, Debug)]
pub enum TrrError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid TRR file: {0}")]
    InvalidFormat(String),
    #[error("Unexpected end of file")]
    _UnexpectedEof,
}

#[derive(Debug, Clone)]
pub struct TrrHeader {
    pub box_size: i32,
    pub vir_size: i32,
    pub pres_size: i32,
    pub top_size: i32,
    pub sym_size: i32,
    pub x_size: i32,
    pub v_size: i32,
    pub f_size: i32,
    pub natoms: usize,
    pub step: i32,
    pub t: f64,
    pub _lambda: f64,
    pub _version: String,
}

#[derive(Debug, Clone)]
pub struct TrrFrame {
    pub coordinates: Option<Vec<f32>>,      // Angstroms
    pub velocities: Option<Vec<f32>>,       // nm / ps
    pub forces: Option<Vec<f32>>,           // kJ / mol / nm
    pub box_vectors: Option<[[f32; 3]; 3]>, // Angstroms
    pub _step: i32,
    pub time: f32,
}

pub struct TrrReader<R: std::io::Read> {
    reader: XdrReader<R>,
}

impl<R: std::io::Read> TrrReader<R> {
    pub fn new(reader: R) -> Self {
        Self {
            reader: XdrReader::new(reader),
        }
    }
}

impl TrrReader<BufReader<File>> {
    pub fn open(path: &str) -> Result<Self, TrrError> {
        let file = File::open(path)?;
        Ok(Self::new(BufReader::new(file)))
    }
}

impl<R: std::io::Read> TrrReader<R> {
    pub fn read_next_header(&mut self) -> Result<Option<TrrHeader>, TrrError> {
        let magic = match self.reader.read_i32() {
            Ok(m) => m,
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(e.into()),
        };

        if magic != 1993 {
            return Err(TrrError::InvalidFormat(format!(
                "Invalid TRR magic: expected 1993, got {}",
                magic
            )));
        }

        let version = self.reader.read_string()?;
        let _ir_size = self.reader.read_i32()?;
        let _e_size = self.reader.read_i32()?;
        let box_size = self.reader.read_i32()?;
        let vir_size = self.reader.read_i32()?;
        let pres_size = self.reader.read_i32()?;
        let top_size = self.reader.read_i32()?;
        let sym_size = self.reader.read_i32()?;
        let x_size = self.reader.read_i32()?;
        let v_size = self.reader.read_i32()?;
        let f_size = self.reader.read_i32()?;
        let natoms = self.reader.read_i32()? as usize;
        let step = self.reader.read_i32()?;
        let _nre = self.reader.read_i32()?;

        // Detect precision for time/lambda
        // Header sizes for t and lambda:
        // Gromacs source: if (t_size == sizeof(float)) ...
        // We can check the total header size if we had it, but usually, we just check if
        // (ir_size + e_size + box_size + ... + f_size) matches what's expected.
        // Actually, internal TRR headers have a fixed size depending on precision.
        // For now, let's assume f32 and if it feels wrong we can look at data.
        // Standard TRR uses f32 for time unless explicitly compiled for double.

        // Wait, the header itself doesn't tell us the size of 't' and 'lambda'.
        // But we can infer it if we know the total size of the header... which we don't.
        // Gromacs rule: if x_size / (natoms * 3) == 8, then it's double precision.
        // Let's assume the same for t and lambda.
        let is_double = if natoms > 0 && x_size > 0 {
            (x_size as usize / (natoms * 3)) == 8
        } else if box_size > 0 {
            (box_size / 9) == 8
        } else {
            false
        };

        let (t, lambda) = if is_double {
            (self.reader.read_f64()?, self.reader.read_f64()?)
        } else {
            (
                self.reader.read_f32()? as f64,
                self.reader.read_f32()? as f64,
            )
        };

        Ok(Some(TrrHeader {
            box_size,
            vir_size,
            pres_size,
            top_size,
            sym_size,
            x_size,
            v_size,
            f_size,
            natoms,
            step,
            t,
            _lambda: lambda,
            _version: version,
        }))
    }

    pub fn read_frame(&mut self) -> Result<Option<TrrFrame>, TrrError> {
        let header = match self.read_next_header()? {
            Some(h) => h,
            None => return Ok(None),
        };

        let is_double = if header.natoms > 0 && header.x_size > 0 {
            (header.x_size as usize / (header.natoms * 3)) == 8
        } else if header.box_size > 0 {
            (header.box_size / 9) == 8
        } else {
            false
        };

        let mut box_vectors = None;
        if header.box_size > 0 {
            let mut mat = [[0.0f32; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    let val = if is_double {
                        self.reader.read_f64()? as f32
                    } else {
                        self.reader.read_f32()?
                    };
                    mat[i][j] = val * 10.0; // nm -> Angstroms
                }
            }
            box_vectors = Some(mat);
        }

        // Skip vir, pres, top, sym if present
        // (These are usually 0 in trajectory files but can be non-zero in energy files)
        // Wait, TRR can have vir_size and pres_size.
        // We MUST skip them correctly.
        if header.vir_size > 0 {
            self.reader.skip(header.vir_size as usize)?;
        }
        if header.pres_size > 0 {
            self.reader.skip(header.pres_size as usize)?;
        }
        if header.top_size > 0 {
            self.reader.skip(header.top_size as usize)?;
        }
        if header.sym_size > 0 {
            self.reader.skip(header.sym_size as usize)?;
        }

        let mut coordinates = None;
        if header.x_size > 0 {
            let mut coords = vec![0.0f32; header.natoms * 3];
            for i in 0..(header.natoms * 3) {
                let val = if is_double {
                    self.reader.read_f64()? as f32
                } else {
                    self.reader.read_f32()?
                };
                coords[i] = val * 10.0; // nm -> Angstroms
            }
            coordinates = Some(coords);
        }

        let mut velocities = None;
        if header.v_size > 0 {
            let mut vels = vec![0.0f32; header.natoms * 3];
            for i in 0..(header.natoms * 3) {
                vels[i] = if is_double {
                    self.reader.read_f64()? as f32
                } else {
                    self.reader.read_f32()?
                };
            }
            velocities = Some(vels);
        }

        let mut forces = None;
        if header.f_size > 0 {
            let mut frcs = vec![0.0f32; header.natoms * 3];
            for i in 0..(header.natoms * 3) {
                frcs[i] = if is_double {
                    self.reader.read_f64()? as f32
                } else {
                    self.reader.read_f32()?
                };
            }
            forces = Some(frcs);
        }

        Ok(Some(TrrFrame {
            coordinates,
            velocities,
            forces,
            box_vectors,
            _step: header.step,
            time: header.t as f32,
        }))
    }

    pub fn read_all_frames(&mut self) -> Result<Vec<TrrFrame>, TrrError> {
        let mut frames = Vec::new();
        while let Some(frame) = self.read_frame()? {
            frames.push(frame);
        }
        Ok(frames)
    }
}

#[derive(Debug, Clone)]
pub struct TrrTrajectory {
    pub num_frames: usize,
    pub num_atoms: usize,
    pub times: Vec<f32>,
    pub coords: Vec<Vec<f32>>,
    pub velocities: Option<Vec<Vec<f32>>>,
    pub _forces: Option<Vec<Vec<f32>>>,
    pub box_vectors: Option<Vec<[[f32; 3]; 3]>>,
}

pub struct FrameWithBox {
    pub coordinates: Vec<f32>,
    pub box_vectors: Option<[[f32; 3]; 3]>,
}

/// Build a frame offset index for TRR by reading all frames and recording their file positions.
/// Returns a Vec of byte offsets for each frame start.
///
/// Strategy: Do one sequential forward scan, recording stream_position() before each frame's
/// magic number read. Since TRR frames are variable-sized and may contain optional data,
/// we must parse through each frame to determine its end. The offset vector enables
/// future O(1) random seeks via read_trr_frame_at_offset (future enhancement).
pub fn build_trr_frame_offsets(path: &str) -> Result<Vec<u64>, TrrError> {
    let file = File::open(path)?;
    let mut buf_reader = BufReader::new(file);
    let mut offsets = Vec::new();

    loop {
        // Record offset before attempting to read magic
        let offset_before_magic = buf_reader.stream_position()?;

        // Try to read magic to detect EOF
        let mut magic_buf = [0u8; 4];
        match buf_reader.read_exact(&mut magic_buf) {
            Ok(()) => {
                // Successfully read magic
            }
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                // Normal EOF
                break;
            }
            Err(e) => return Err(e.into()),
        }

        let magic = i32::from_be_bytes(magic_buf);
        if magic != 1993 {
            return Err(TrrError::InvalidFormat(format!(
                "Invalid TRR magic at offset {}: expected 1993, got {}",
                offset_before_magic, magic
            )));
        }

        offsets.push(offset_before_magic);

        // Create XdrReader to parse the rest of the header and skip frame data
        let mut xdr = XdrReader::new(&mut buf_reader);

        // Read header fields
        let _version = xdr.read_string()?;
        let _ir_size = xdr.read_i32()?;
        let _e_size = xdr.read_i32()?;
        let box_size = xdr.read_i32()?;
        let vir_size = xdr.read_i32()?;
        let pres_size = xdr.read_i32()?;
        let top_size = xdr.read_i32()?;
        let sym_size = xdr.read_i32()?;
        let x_size = xdr.read_i32()?;
        let v_size = xdr.read_i32()?;
        let f_size = xdr.read_i32()?;
        let natoms = xdr.read_i32()?;
        let _step = xdr.read_i32()?;
        let _nre = xdr.read_i32()?;

        // Detect precision for time/lambda
        let is_double = if natoms > 0 && x_size > 0 {
            (x_size as usize / (natoms as usize * 3)) == 8
        } else if box_size > 0 {
            (box_size / 9) == 8
        } else {
            false
        };

        // Skip time and lambda
        if is_double {
            xdr.read_f64()?;
            xdr.read_f64()?;
        } else {
            xdr.read_f32()?;
            xdr.read_f32()?;
        }

        // Skip all data blocks in order: box, vir, pres, top, sym, x, v, f
        if box_size > 0 {
            xdr.skip(box_size as usize)?;
        }
        if vir_size > 0 {
            xdr.skip(vir_size as usize)?;
        }
        if pres_size > 0 {
            xdr.skip(pres_size as usize)?;
        }
        if top_size > 0 {
            xdr.skip(top_size as usize)?;
        }
        if sym_size > 0 {
            xdr.skip(sym_size as usize)?;
        }
        if x_size > 0 {
            xdr.skip(x_size as usize)?;
        }
        if v_size > 0 {
            xdr.skip(v_size as usize)?;
        }
        if f_size > 0 {
            xdr.skip(f_size as usize)?;
        }

        // XdrReader is now positioned at the end of this frame.
        // Reconstruct buf_reader by dropping the xdr reference so buf_reader is usable again.
        drop(xdr);
    }

    Ok(offsets)
}

pub fn read_trr_frame_at(path: &str, frame_index: usize) -> Result<FrameWithBox, TrrError> {
    let mut reader = TrrReader::open(path)?;
    let mut current_index = 0;

    while let Some(frame) = reader.read_frame()? {
        if current_index == frame_index {
            return Ok(FrameWithBox {
                coordinates: frame.coordinates.unwrap_or_default(),
                box_vectors: frame.box_vectors,
            });
        }
        current_index += 1;
    }

    Err(TrrError::InvalidFormat(format!(
        "Frame {} not found in TRR file",
        frame_index
    )))
}

pub fn parse_trr(path: &str) -> Result<TrrTrajectory, TrrError> {
    let mut reader = TrrReader::open(path)?;
    let frames = reader.read_all_frames()?;

    if frames.is_empty() {
        return Err(TrrError::InvalidFormat(
            "No frames found in TRR file".into(),
        ));
    }

    let num_frames = frames.len();
    let num_atoms = frames[0]
        .coordinates
        .as_ref()
        .map(|c| c.len() / 3)
        .unwrap_or(0);

    let mut times = Vec::with_capacity(num_frames);
    let mut coords = Vec::with_capacity(num_frames);
    let mut vels = Vec::with_capacity(num_frames);
    let mut frcs = Vec::with_capacity(num_frames);
    let mut boxes = Vec::with_capacity(num_frames);

    let has_vel = frames.iter().any(|f| f.velocities.is_some());
    let has_frc = frames.iter().any(|f| f.forces.is_some());
    let has_box = frames.iter().any(|f| f.box_vectors.is_some());

    for frame in frames {
        times.push(frame.time);
        coords.push(
            frame
                .coordinates
                .unwrap_or_else(|| vec![0.0; num_atoms * 3]),
        );

        if has_vel {
            vels.push(frame.velocities.unwrap_or_else(|| vec![0.0; num_atoms * 3]));
        }
        if has_frc {
            frcs.push(frame.forces.unwrap_or_else(|| vec![0.0; num_atoms * 3]));
        }
        if has_box {
            boxes.push(frame.box_vectors.unwrap_or([[0.0; 3]; 3]));
        }
    }

    Ok(TrrTrajectory {
        num_frames,
        num_atoms,
        times,
        coords,
        velocities: if has_vel { Some(vels) } else { None },
        _forces: if has_frc { Some(frcs) } else { None },
        box_vectors: if has_box { Some(boxes) } else { None },
    })
}
