//! Χ → Cartesian geometry engine for sidechain building.
//!
//! Builds sidechain coordinates from dihedral angles using internal coordinates (nerf).
//! Supports proline ring closure via cyclic coordinate descent (CCD).

pub mod template;
pub mod proline;

pub use template::ResidueTemplate;
pub use proline::ProlineBuilder;
