//! Backbone-dependent rotamer library for protein sidechain placement (proxide-rotlib).
//!
//! Loads and queries the MSL binary rotamer library format used by the
//! Mosaist protein design suite.  For each amino acid type the library
//! stores a rectangular φ/ψ grid of rotamer populations and canonical
//! backbone-relative sidechain coordinates.
//!
//! # Typical usage
//!
//! ```no_run
//! use proxide_rotlib::RotamerLibrary;
//! use std::path::Path;
//!
//! let lib = RotamerLibrary::load(Path::new("rotlib.bin")).unwrap();
//! let n = [0.0f64, 0.0, 0.0];
//! let ca = [1.458, 0.0, 0.0];
//! let c = [2.009, 1.420, 0.0];
//! let placed = lib.place_rotamer("LEU", -60.0, -40.0, 0, false, n, ca, c).unwrap();
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
pub mod geometry;

pub use error::RotlibError;
pub use rotamer_id::{RotamerId, PlacedRotamer, PlacedAtom};
pub use rotlib::RotamerLibrary;
pub use frame::{Frame, Transform, backbone_frame};
pub use sidechain::{counts_as_sidechain, is_backbone_or_hydrogen};
pub use geometry::{ResidueTemplate, ProlineBuilder};
