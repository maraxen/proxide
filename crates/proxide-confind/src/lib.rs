//! Sequence-independent contact degree, crowdedness, and structural freedom algorithm for protein backbones.
//!
//! Given a protein backbone and a backbone-dependent rotamer library, ConFind estimates
//! how much conformational freedom each residue loses due to sidechain clashes with the
//! surrounding backbone and neighboring residue ensembles.
//!
//! # Algorithm phases
//!
//! - **Phase A — Cache:** for each residue, enumerate its rotamers, prune those
//!   clashing with backbone atoms (distance <= 2.0 Å), and compute crowdedness
//!   C(i) = 1.0 - (Omega_i / Omega_lib).
//! - **Phase B — Contacts:** for each candidate pair within the 25.0 Å Cα–Cα cutoff,
//!   enumerate cross-rotamer contacts (distance <= 3.0 Å) in parallel (orx-parallel)
//!   weighted by P(a) P(b) P(r_a) P(r_b) and normalized by Omega_i * Omega_j.
//! - **Phase C — Freedom:** aggregate collision probabilities into a scalar residual
//!   conformational freedom value F(i) per residue.
//!
//! # References
//!
//! - Zheng, F., Zhang, J., & Grigoryan, G. (2015). "Tertiary Structural Propensities Reveal
//!   Fundamental Sequence/Structure Relationships in Proteins." *Structure*, 23(5), 961–971.
//! - Zheng, W., & Grigoryan, G. (2017). "Sequence statistics of tertiary structural motifs
//!   reflect protein stability." *PLoS ONE*, 12(5), e0178272.

pub mod cache;
pub mod confind;
pub mod contact_list;
pub mod coords;
pub mod error;
pub mod freedom;
pub mod grid;
pub mod parallel;
pub mod params;
pub mod precondition;

pub use confind::ConFind;
pub use contact_list::{ContactList, CONTACT_THRESHOLD};
pub use coords::{
    extract_f64_backbone, extract_f64_backbone_with_options, load_pdb_f64, BackboneOptions,
    ChainBreakPolicy, MissingAtomPolicy, ProteinBackbone, ResidueBackbone, ResidueIndex,
};
pub use error::ConFindError;
pub use freedom::compute_freedom;
pub use params::{
    aa_propensity, AA_NAMES, CLASH_DIST, CONT_DIST, DCUT, HI_COLL_PROB_CUT, LO_COLL_PROB_CUT,
};
pub use precondition::{
    chain_breaks, check_preconditions, require_preconditions, ChainBreakAt, PreconditionReport,
    PreconditionViolation, Severity, ViolationKind, CHAIN_BREAK_CA_ANGSTROM,
};
