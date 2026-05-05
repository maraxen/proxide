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
