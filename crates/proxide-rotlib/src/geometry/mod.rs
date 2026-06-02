//! Χ → Cartesian geometry engine for sidechain building.
//!
//! Builds sidechain coordinates from dihedral angles using internal coordinates (nerf).
//! Proline ring closure keeps χ exact and relaxes the CB-CG-CD angle (P3 v2).

pub mod template;
pub mod proline;

pub use template::{ResidueTemplate, proline_template};
pub use proline::{ProlineBuilder, ProlineCoords};
