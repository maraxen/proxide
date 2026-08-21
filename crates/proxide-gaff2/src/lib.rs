//! Proxide GAFF2 Library
//!
//! GAFF2 atom typing engine — atom type descriptors (ATD) from ATOMTYPE_GFF2.DEF.
//! Ports the ciMIST reference implementation's atom-typing logic.

pub mod alternation;
pub mod atom_bond_facts;
pub mod atomic_prop;
pub mod charges;
pub mod chem_env;
pub mod def_parser;
pub mod ffxml_builder;
pub mod mol;
pub mod orchestrate;
pub mod param_loader;
pub mod param_lookup;
pub mod parameterize;
pub mod pdb_names;
#[cfg(feature = "python-validation")]
pub mod py_validation;
pub mod rings;
pub mod rules_loader;

pub use orchestrate::assign_gaff2_atom_types;
pub use param_loader::{load_default_gaff2_parameters, load_gaff2_parameters, Gaff2Parameters};
pub use param_lookup::{lookup_angle_params, lookup_bond_params};
pub use parameterize::{parameterize_gaff2, Gaff2Parameterization};
pub use pdb_names::assign_pdb_atom_names;
pub use ffxml_builder::{
    build_gaff2_ffxml, build_gaff2_ffxml_from_types, load_gaff2_parameters_default as load_default_gaff2_ffxml_params,
    Gaff2FfxmlError, Gaff2Params as Gaff2FfxmlParams,
};
