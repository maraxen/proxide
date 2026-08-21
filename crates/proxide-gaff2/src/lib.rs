//! Proxide GAFF2 Library
//!
//! GAFF2 atom typing engine — atom type descriptors (ATD) from ATOMTYPE_GFF2.DEF.
//! Ports the ciMIST reference implementation's atom-typing logic.
//!
//! Scope is atom typing only. Force-field parameter generation (bond/angle/
//! torsion parameter lookup, PDB atom naming, partial charges, OpenMM ffxml
//! generation) is `crates/proxide-gaff`'s job, not this crate's — six modules
//! in that space (`charges`, `parameterize`, `ffxml_builder`, `param_loader`,
//! `param_lookup`, `pdb_names`) were ported here during initial scaffolding
//! against the architecture decision's own explicit exclusion list, then
//! removed once that scope creep was caught (see
//! `.praxia/docs/reference/260821_gaff2-rust-port-lessons.md`, Open Item #4)
//! — three of the four unresolved adversarial-verify defects from that port
//! lived in exactly this excluded scope.

pub mod alternation;
pub mod atom_bond_facts;
pub mod atomic_prop;
pub mod chem_env;
pub mod def_parser;
pub mod mol;
pub mod orchestrate;
#[cfg(feature = "python-validation")]
pub mod py_validation;
pub mod rings;
pub mod rules_loader;

pub use orchestrate::assign_gaff2_atom_types;
