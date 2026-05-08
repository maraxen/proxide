//! IO and parsing modules for protein structures
//!
//! TODO: For full WASM compatibility, consider the following:
//! 1. Use `wasm-bindgen-rayon` to enable multi-threading in the browser.
//! 2. Implement a zero-copy parsing path that accepts `&[u8]` from JS `ArrayBuffer`.
//! 3. Use Origin Private File System (OPFS) for high-performance synchronous file access.

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
