//! File format parsers

pub mod dcd;
pub mod mmcif;
pub mod pdb;
pub mod pqr;
pub mod trr;
pub mod xtc;

// HDF5 formats (feature-gated)
#[cfg(feature = "mdcath")]
pub mod mdcath_h5;
#[cfg(feature = "mdcath")]
pub mod mdtraj_h5;

// Re-export for convenience
#[cfg(feature = "mdcath")]
pub use mdcath_h5::{MdcathDomain, MdcathFrame};
#[cfg(feature = "mdcath")]
pub use mdtraj_h5::{MdtrajFrame, MdtrajH5Result};
