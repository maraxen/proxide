//! Geometry operations for protein structures
//!
//! Provides distance calculations, dihedral angle computations, radial basis
//! functions, neighbor search algorithms, coordinate transforms, alignment,
//! solvent handling, and hydrogen addition via fragment library.

pub mod alignment;
pub mod angles;
pub mod cell_list;
pub mod distances;
pub mod fragment_library;
pub mod hydrogens;
pub mod neighbors;
pub mod radial_basis;
pub mod relax;
pub mod solvent;
pub mod topology;
pub mod transforms;

// Re-export public items for external use
#[allow(unused_imports)]
pub use alignment::*;
#[allow(unused_imports)]
pub use angles::*;
#[allow(unused_imports)]
pub use cell_list::*;
#[allow(unused_imports)]
pub use distances::*;
#[allow(unused_imports)]
pub use hydrogens::*;
#[allow(unused_imports)]
pub use neighbors::*;
#[allow(unused_imports)]
pub use radial_basis::*;
#[allow(unused_imports)]
pub use solvent::*;
#[allow(unused_imports)]
pub use topology::*;
#[allow(unused_imports)]
pub use transforms::*;
