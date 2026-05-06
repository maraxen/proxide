//! Pure-Rust DCD trajectory format parser
//!
//! DCD is a binary format used by CHARMM, NAMD, and OpenMM.
//! This implementation is pure-Rust and does not depend on chemfiles.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
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
    pub coordinates: Vec<f32>,
    pub unit_cell: Option<[f64; 6]>,
}

pub struct DcdReader<R: Read + Seek> {
    reader: R,
    pub header: DcdHeader,
}

impl DcdReader<File> {
    pub fn open(path: &str) -> Result<Self, DcdError> {
        let file = File::open(path)?;
        Self::new(file)
    }
}

impl<R: Read + Seek> DcdReader<R> {
    pub fn new(mut reader: R) -> Result<Self, DcdError> {
        let mut buf4 = [0u8; 4];
        reader.read_exact(&mut buf4)?;
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

        reader.read_exact(&mut buf4)?;
        if &buf4 != b"CORD" {
            return Err(DcdError::InvalidFormat(
                "DCD magic signature 'CORD' not found".into(),
            ));
        }

        let mut header_data = [0u8; 80];
        reader.read_exact(&mut header_data)?;

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
        let delta = get_f32(36);
        let has_unit_cell = get_i32(40) != 0;
        let charmm_version = get_i32(76);

        reader.read_exact(&mut buf4)?;

        reader.read_exact(&mut buf4)?;
        let title_block_len = if is_little_endian {
            i32::from_le_bytes(buf4)
        } else {
            i32::from_be_bytes(buf4)
        };
        reader.seek(SeekFrom::Current(title_block_len as i64 + 4))?;

        reader.read_exact(&mut buf4)?;
        reader.read_exact(&mut buf4)?;
        let n_atoms = if is_little_endian {
            i32::from_le_bytes(buf4)
        } else {
            i32::from_be_bytes(buf4)
        } as usize;
        reader.read_exact(&mut buf4)?;

        Ok(DcdReader {
            reader,
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_dcd_invalid_magic() {
        let data = b"NOTACORD";
        let file = Cursor::new(data);
        let result = DcdReader::new(file);
        assert!(result.is_err());
    }
}
