//! Proxide Core Library
//!
//! Foundational data structures and chemical constants for protein parsing.

pub mod chem;
pub mod forcefield;
pub mod processing;
pub mod spec;
pub mod structure;
#[cfg(any(test, feature = "testing"))]
pub mod testing;

// Re-exports for convenience
pub use forcefield::*;
pub use processing::ProcessedStructure;
pub use spec::*;
pub use structure::systems::*;
