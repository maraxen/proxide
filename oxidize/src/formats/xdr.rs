//! XDR (eXternal Data Representation) decoding utilities
//!
//! Used by the TRR trajectory parser. XDR is big-endian and pads to 4 bytes.

use std::io::{Read, Result};

pub struct XdrReader<R: Read> {
    inner: R,
}

impl<R: Read> XdrReader<R> {
    pub fn new(inner: R) -> Self {
        Self { inner }
    }

    pub fn read_i32(&mut self) -> Result<i32> {
        let mut buf = [0u8; 4];
        self.inner.read_exact(&mut buf)?;
        Ok(i32::from_be_bytes(buf))
    }

    pub fn read_f32(&mut self) -> Result<f32> {
        let mut buf = [0u8; 4];
        self.inner.read_exact(&mut buf)?;
        Ok(f32::from_be_bytes(buf))
    }

    pub fn read_f64(&mut self) -> Result<f64> {
        let mut buf = [0u8; 8];
        self.inner.read_exact(&mut buf)?;
        Ok(f64::from_be_bytes(buf))
    }

    pub fn read_string(&mut self) -> Result<String> {
        let len = self.read_i32()? as usize;
        let mut buf = vec![0u8; len];
        self.inner.read_exact(&mut buf)?;

        // Discard padding to 4-byte boundary
        let padding = (4 - (len % 4)) % 4;
        if padding > 0 {
            let mut pad_buf = [0u8; 4];
            self.inner.read_exact(&mut pad_buf[..padding])?;
        }

        Ok(String::from_utf8_lossy(&buf).into_owned())
    }

    pub fn _read_opaque(&mut self, len: usize) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; len];
        self.inner.read_exact(&mut buf)?;

        // Discard padding
        let padding = (4 - (len % 4)) % 4;
        if padding > 0 {
            let mut pad_buf = [0u8; 4];
            self.inner.read_exact(&mut pad_buf[..padding])?;
        }

        Ok(buf)
    }

    pub fn skip(&mut self, len: usize) -> Result<()> {
        let mut buf = vec![0u8; 64];
        let mut remaining = len;
        while remaining > 0 {
            let to_read = std::cmp::min(remaining, buf.len());
            self.inner.read_exact(&mut buf[..to_read])?;
            remaining -= to_read;
        }

        // Discard padding
        let padding = (4 - (len % 4)) % 4;
        if padding > 0 {
            let mut pad_buf = [0u8; 4];
            self.inner.read_exact(&mut pad_buf[..padding])?;
        }

        Ok(())
    }
}
