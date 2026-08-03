//! Proxider Core Library
//!
//! High-performance protein structure parsing library.
//! Provides zero-copy parsing for PDB, mmCIF, and trajectory formats.

// Re-export core modules
pub use proxide_core::chem;
pub use proxide_core::spec;
pub use proxide_core::structure;

// Re-export unit system
pub use proxide_units::UnitSystem;

// Re-export algorithm and physics modules
pub use proxide_geometry::geometry as algo_geometry;
pub use proxide_io::formats;
pub use proxide_io::formatters;
#[cfg(feature = "fetching")]
pub use proxide_io::io;
pub use proxide_physics::physics;

/// Force field and topology modules
pub mod forcefield {
    pub use proxide_core::forcefield::*;
    pub use proxide_gaff as gaff;
}

// Re-export gaff at top level for convenience
pub use proxide_gaff as gaff;

// Internal modules (still being migrated or extended)
pub mod geometry;
pub mod processing;
