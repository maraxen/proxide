//! Force field module for parsing and representing molecular mechanics parameters.
//!
//! This module provides:
//! - Data types for force field parameters (types.rs)
//! - XML parser for OpenMM-style force field files (xml_parser.rs)
//! - Molecular topology generation (topology.rs)
//! - Nonbonded exclusion lists (exclusions.rs)
//! - GAFF atom typing for ligands (gaff.rs)
//!
//! # Supported Force Field Types
//!
//! - **Protein force fields**: ff14SB, ff19SB
//! - **GAFF**: General Amber Force Field for small molecules
//! - **Water models**: TIP3P, OPC, TIP4P-Ew
//! - **Implicit solvent**: GBSA-OBC parameters

pub mod exclusions;
pub mod gaff;
pub mod gaff_generator;
pub mod topology;
mod types;
mod xml_parser;

// Re-export public items for external use
#[allow(unused_imports)]
pub use exclusions::Exclusions;
#[allow(unused_imports)]
pub use gaff::{GaffAtomType, GaffParameters};
#[allow(unused_imports)]
pub use topology::{Angle, Bond, Dihedral, Topology};
#[allow(unused_imports)]
pub use types::*;
pub use xml_parser::parse_forcefield_xml;
#[allow(unused_imports)]
pub use xml_parser::ParseError;
