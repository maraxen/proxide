//! Proxide Core Library
//!
//! Foundational data structures and chemical constants for protein parsing.

pub mod chem;
pub mod forcefield;
pub mod processing;
pub mod spec;
pub mod structure;

// Re-exports for convenience
pub use spec::*;
pub use structure::systems::*;
pub use processing::ProcessedStructure;
pub use forcefield::*;
