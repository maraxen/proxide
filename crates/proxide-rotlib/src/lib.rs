//! Backbone-dependent rotamer library for protein sidechain placement (proxide-rotlib).
//!
//! Loads and queries the MSL binary rotamer library format used by the
//! Mosaist protein design suite.  For each amino acid type the library
//! stores a rectangular φ/ψ grid of rotamer populations and canonical
//! backbone-relative sidechain coordinates.
//!
//! # Typical usage
//!
//! ```ignore
//! let lib = RotamerLibrary::load(Path::new("rotlib.bin"))?;
//! let placed = lib.place_rotamer("LEU", phi, psi, rot_index, n, ca, c)?;
//! ```
//!
//! # References
//!
//! - Mosaist / MSL: <https://grigoryanlab.org/mosaist/>
//! - Dunbrack RL Jr. "Rotamer libraries in the 21st century."
//!   *Curr Opin Struct Biol.* 2002;12(4):431-440.

pub mod error;
pub mod rotamer_id;
pub mod frame;
pub mod rotlib;
pub mod binning;
pub mod sidechain;

pub use error::RotlibError;
pub use rotamer_id::{RotamerId, PlacedRotamer, PlacedAtom};
pub use rotlib::RotamerLibrary;
pub use frame::{Frame, Transform, backbone_frame};
pub use sidechain::{counts_as_sidechain, is_backbone_or_hydrogen};
