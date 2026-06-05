//! IO and parsing modules for protein structures
//!
//! Note: WASM is supported via orx-parallel's `num_threads(1)` sequential gate.
//! For full WASM parallelism, consider:
//! 1. Implement a zero-copy parsing path that accepts `&[u8]` from JS `ArrayBuffer`.
//! 2. Use Origin Private File System (OPFS) for high-performance synchronous file access.

#[cfg(feature = "pdb")]
pub mod formats;
#[cfg(feature = "pdb")]
pub mod formatters;
#[cfg(feature = "fetching")]
pub mod io;

#[cfg(feature = "ssbond")]
#[derive(thiserror::Error, Debug)]
pub enum IOParseError {
    #[error("Recursion depth exceeded")]
    RecursionExceeded,
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

#[cfg(feature = "ssbond")]
pub struct CIFRegistry {
    pub max_recursion_depth: usize,
}

#[cfg(feature = "ssbond")]
impl Default for CIFRegistry {
    fn default() -> Self {
        Self {
            max_recursion_depth: 10,
        }
    }
}
