//! File format parsers
//!
//! TODO: Implement a zero-copy parsing path that accepts `&[u8]` (from JS `ArrayBuffer` in WASM)
//! to avoid unnecessary copies between the host environment and Rust.

pub mod dcd;
#[cfg(feature = "foldcomp")]
pub mod foldcomp;
pub mod mmcif;
pub mod pdb;
pub mod pqr;
pub mod trr;
pub mod xdr;
#[cfg(feature = "xtc")]
pub mod xtc;

// HDF5 formats (feature-gated)
#[cfg(feature = "hdf5")]
pub mod mdcath_h5;
#[cfg(feature = "hdf5")]
pub mod mdtraj_h5;

// Re-export for convenience
#[cfg(feature = "hdf5")]
#[allow(unused_imports)]
pub use mdcath_h5::{MdcathDomain, MdcathFrame};
#[cfg(feature = "hdf5")]
#[allow(unused_imports)]
pub use mdtraj_h5::{MdtrajFrame, MdtrajH5Result};
