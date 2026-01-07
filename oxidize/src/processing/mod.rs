//! Structure processing module
//!
//! Handles residue grouping, chain mapping, structure organization,
//! coordinate noising, and projection to ML-ready formats.

pub mod models;
pub mod noising;
pub mod projection;
pub mod residues;

pub use models::*;
pub use noising::*;
pub use projection::*;
pub use residues::*;
