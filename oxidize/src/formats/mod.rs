//! File format parsers

pub mod dcd;
pub mod foldcomp;
pub mod mmcif;
pub mod pdb;
pub mod pqr;
pub mod trr;
pub mod xdr;
pub mod xtc;

// HDF5 formats (feature-gated)
#[cfg(feature = "mdcath")]
pub mod mdcath_h5;
#[cfg(feature = "mdcath")]
pub mod mdtraj_h5;

// MDCompress formats (feature-gated)
#[cfg(feature = "mdc")]
pub mod mdc;

// Re-export for convenience
#[cfg(feature = "mdcath")]
#[allow(unused_imports)]
pub use mdcath_h5::{MdcathDomain, MdcathFrame};
#[cfg(feature = "mdcath")]
#[allow(unused_imports)]
pub use mdtraj_h5::{MdtrajFrame, MdtrajH5Result};
