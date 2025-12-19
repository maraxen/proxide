//! Pure-Rust DCD trajectory format parser
//!
//! DCD is a binary format used by CHARMM, NAMD, and OpenMM.
//! This implementation is pure-Rust and does not depend on chemfiles.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum DcdError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid DCD file: {0}")]
    InvalidFormat(String),
    #[error("Unexpected end of file")]
    UnexpectedEof,
}

#[derive(Debug, Clone)]
pub struct DcdHeader {
    pub n_frames: usize,
    pub n_atoms: usize,
    pub _start_step: i32,
    pub _save_freq: i32,
    pub delta: f32,
    pub has_unit_cell: bool,
    pub _charmm_version: i32,
    pub is_little_endian: bool,
}

#[derive(Debug, Clone)]
pub struct DcdFrame {
    /// Coordinates as flattened [x1, y1, z1, x2, y2, z2, ...] in Angstroms
    pub coordinates: Vec<f32>,
    /// Unit cell dimensions (a, b, c, alpha, beta, gamma)
    pub unit_cell: Option<[f64; 6]>,
}

pub struct DcdReader {
    file: File,
    pub header: DcdHeader,
}

impl DcdReader {
    pub fn open(path: &str) -> Result<Self, DcdError> {
        let mut file = File::open(path)?;

        // Detect endianness by reading first 4 bytes (record length)
        let mut buf4 = [0u8; 4];
        file.read_exact(&mut buf4)?;
        let len_le = i32::from_le_bytes(buf4);
        let len_be = i32::from_be_bytes(buf4);

        let is_little_endian = if len_le == 84 {
            true
        } else if len_be == 84 {
            false
        } else {
            return Err(DcdError::InvalidFormat(format!(
                "First record length should be 84, got {} (LE) or {} (BE)",
                len_le, len_be
            )));
        };

        // Read CORD magic
        file.read_exact(&mut buf4)?;
        if &buf4 != b"CORD" {
            return Err(DcdError::InvalidFormat(
                "DCD magic signature 'CORD' not found".into(),
            ));
        }

        // Read the rest of the 84-byte header block (80 more bytes)
        let mut header_data = [0u8; 80];
        file.read_exact(&mut header_data)?;

        let get_i32 = |offset: usize| {
            let slice = &header_data[offset..offset + 4];
            let b: [u8; 4] = slice.try_into().unwrap();
            if is_little_endian {
                i32::from_le_bytes(b)
            } else {
                i32::from_be_bytes(b)
            }
        };

        let get_f32 = |offset: usize| {
            let slice = &header_data[offset..offset + 4];
            let b: [u8; 4] = slice.try_into().unwrap();
            if is_little_endian {
                f32::from_le_bytes(b)
            } else {
                f32::from_be_bytes(b)
            }
        };

        let n_frames = get_i32(0) as usize;
        let start_step = get_i32(4);
        let save_freq = get_i32(8);
        // Skip NSTEP (12), NPRIV (16), NSAVC (20), etc.
        let delta = get_f32(36);
        let has_unit_cell = get_i32(40) != 0;
        let charmm_version = get_i32(76);

        // Closing record length
        file.read_exact(&mut buf4)?;

        // Block 2: Title block
        file.read_exact(&mut buf4)?;
        let title_block_len = if is_little_endian {
            i32::from_le_bytes(buf4)
        } else {
            i32::from_be_bytes(buf4)
        };
        // Skip title block content + closing record length
        file.seek(SeekFrom::Current(title_block_len as i64 + 4))?;

        // Block 3: NATOM block
        file.read_exact(&mut buf4)?; // Start 4
        file.read_exact(&mut buf4)?;
        let n_atoms = if is_little_endian {
            i32::from_le_bytes(buf4)
        } else {
            i32::from_be_bytes(buf4)
        } as usize;
        file.read_exact(&mut buf4)?; // End 4

        Ok(DcdReader {
            file,
            header: DcdHeader {
                n_frames,
                n_atoms,
                _start_step: start_step,
                _save_freq: save_freq,
                delta,
                has_unit_cell,
                _charmm_version: charmm_version,
                is_little_endian,
            },
        })
    }

    pub fn read_frame(&mut self) -> Result<DcdFrame, DcdError> {
        let n_atoms = self.header.n_atoms;
        let is_le = self.header.is_little_endian;

        let mut unit_cell = None;
        let mut buf4 = [0u8; 4];

        // Process unit cell block if present
        if self.header.has_unit_cell {
            if self.file.read_exact(&mut buf4).is_err() {
                return Err(DcdError::UnexpectedEof);
            }
            let len = if is_le {
                i32::from_le_bytes(buf4)
            } else {
                i32::from_be_bytes(buf4)
            };

            // Should be 48 bytes (6 f64)
            // Order is [A, cos(gamma), B, cos(beta), cos(alpha), C]
            let mut box_buf = vec![0u8; len as usize];
            self.file.read_exact(&mut box_buf)?;

            let mut vals = [0.0f64; 6];
            for i in 0..6 {
                let slice = &box_buf[i * 8..(i + 1) * 8];
                let b: [u8; 8] = slice.try_into().unwrap();
                vals[i] = if is_le {
                    f64::from_le_bytes(b)
                } else {
                    f64::from_be_bytes(b)
                };
            }

            let a = vals[0];
            let gamma = vals[1].acos().to_degrees();
            let b = vals[2];
            let beta = vals[3].acos().to_degrees();
            let alpha = vals[4].acos().to_degrees();
            let c = vals[5];

            unit_cell = Some([a, b, c, alpha, beta, gamma]);

            self.file.read_exact(&mut buf4)?; // Closing record length
        }

        let mut coords = vec![0.0f32; n_atoms * 3];

        // X, Y, Z axes are stored in separate blocks
        for axis in 0..3 {
            if self.file.read_exact(&mut buf4).is_err() {
                return Err(DcdError::UnexpectedEof);
            }
            let len = if is_le {
                i32::from_le_bytes(buf4)
            } else {
                i32::from_be_bytes(buf4)
            };
            if len as usize != n_atoms * 4 {
                return Err(DcdError::InvalidFormat(format!(
                    "Expected {} bytes for axis {}, got {}",
                    n_atoms * 4,
                    axis,
                    len
                )));
            }

            let mut axis_buf = vec![0u8; len as usize];
            self.file.read_exact(&mut axis_buf)?;

            for i in 0..n_atoms {
                let slice = &axis_buf[i * 4..(i + 1) * 4];
                let b: [u8; 4] = slice.try_into().unwrap();
                let val = if is_le {
                    f32::from_le_bytes(b)
                } else {
                    f32::from_be_bytes(b)
                };
                coords[i * 3 + axis] = val;
            }

            self.file.read_exact(&mut buf4)?; // Closing record length
        }

        Ok(DcdFrame {
            coordinates: coords,
            unit_cell,
        })
    }

    pub fn read_all_frames(&mut self) -> Result<Vec<DcdFrame>, DcdError> {
        let mut frames = Vec::new();
        // If n_frames is 0 in header, read until EOF
        if self.header.n_frames > 0 {
            frames.reserve(self.header.n_frames);
            for _ in 0..self.header.n_frames {
                match self.read_frame() {
                    Ok(frame) => frames.push(frame),
                    Err(DcdError::UnexpectedEof) => break,
                    Err(e) => return Err(e),
                }
            }
        } else {
            loop {
                match self.read_frame() {
                    Ok(frame) => frames.push(frame),
                    Err(DcdError::UnexpectedEof) => break,
                    Err(e) => return Err(e),
                }
            }
        }
        Ok(frames)
    }
}

pub fn parse_dcd(path: &str) -> Result<DcdTrajectory, DcdError> {
    let mut reader = DcdReader::open(path)?;
    let frames = reader.read_all_frames()?;

    let num_frames = frames.len();
    let num_atoms = reader.header.n_atoms;
    let mut times = Vec::with_capacity(num_frames);
    let mut coords = Vec::with_capacity(num_frames);
    let mut unit_cells = Vec::with_capacity(num_frames);

    let has_unit_cells = frames.iter().any(|f| f.unit_cell.is_some());

    for (i, frame) in frames.into_iter().enumerate() {
        times.push(i as f32 * reader.header.delta); // Rough estimate
        coords.push(frame.coordinates);
        if let Some(cell) = frame.unit_cell {
            unit_cells.push(cell);
        } else if has_unit_cells {
            unit_cells.push([0.0; 6]);
        }
    }

    Ok(DcdTrajectory {
        num_frames,
        num_atoms,
        times,
        coords,
        unit_cells: if has_unit_cells {
            Some(unit_cells)
        } else {
            None
        },
    })
}

#[derive(Debug, Clone)]
pub struct DcdTrajectory {
    pub num_frames: usize,
    pub num_atoms: usize,
    pub times: Vec<f32>,
    pub coords: Vec<Vec<f32>>,
    pub unit_cells: Option<Vec<[f64; 6]>>,
}
