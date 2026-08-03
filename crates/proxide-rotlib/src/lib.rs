#![deny(warnings)]

//! Backbone-dependent rotamer library for protein sidechain placement (proxide-rotlib).
//!
//! Supports two loader paths:
//!
//! - **MSL binary format** (legacy, via `load()`): reads the binary rotamer library used by the
//!   Mosaist protein design suite. For each amino acid type the library stores a rectangular
//!   φ/ψ grid of rotamer populations and canonical backbone-relative sidechain coordinates.
//! - **Dunbrack BBDEP2010 protobuf** (preferred, via `load_pb()`): loads precomputed rotamer
//!   coordinates from the Dunbrack backbone-dependent rotamer library in compressed protobuf format.
//!
//! # Licensing
//!
//! Code is MIT-licensed. Rotamer data (coordinates and backbone-dependent statistics) derived
//! from the Dunbrack 2010 backbone-dependent rotamer library is ODC-BY-1.0 (Open Data Commons
//! Attribution License 1.0). See the crate README for full attribution and citation details.
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
//! - Shapovalov MV, Dunbrack RL Jr. "A smoothed backbone-dependent rotamer library for proteins
//!   derived from adaptive kernel density estimates and regressions."
//!   *Structure* 19(6):844–858 (2011).

pub mod binning;
pub mod error;
pub mod frame;
pub mod geometry;
pub mod pb;
pub mod rotamer_id;
pub mod rotlib;
pub mod rotlib_source;
pub mod sidechain;

pub use error::RotlibError;
pub use frame::{backbone_frame, Frame, Transform};
pub use geometry::{ProlineBuilder, ResidueTemplate};
pub use pb::rotlib_v1;
pub use rotamer_id::{PlacedAtom, PlacedRotamer, RotamerId};
pub use rotlib::RotamerLibrary;
pub use sidechain::{counts_as_sidechain, is_backbone_or_hydrogen};
