//! Sequence-independent contact degree, crowdedness, and structural freedom algorithm for protein backbones.
//!
//! Given a protein backbone and a backbone-dependent rotamer library, ConFind estimates
//! how much conformational freedom each residue loses due to sidechain clashes with the
//! surrounding backbone and neighboring residue ensembles.
//!
//! # Algorithm phases
//!
//! - **Phase A — Cache:** for each residue, enumerate its rotamers, prune those
//!   clashing with backbone atoms (distance <= `d_clash`, default 2.0 Å), and derive
//!   crowdedness C(i) (prior-mass or rotamer-count fraction, see [`CrowdednessMode`]) and
//!   the directed interference I(i, j) summed over the unpruned library.
//! - **Phase B — Contacts:** for each candidate pair within the `d_cut` (default 25.0 Å)
//!   Cα–Cα cutoff, enumerate cross-rotamer contacts (distance <= `d_cont`, default 3.0 Å)
//!   in parallel (orx-parallel) weighted by P(a) P(b) P(r_a) P(r_b) and normalized by
//!   Omega_i * Omega_j.
//! - **Phase C — Freedom:** aggregate collision probabilities of the residues in the
//!   configured [`ResidueScope`] into a scalar residual conformational freedom value
//!   F(i) per residue (see [`FreedomMode`]).
//!
//! All options live in one typed [`ConFindConfig`] accepted by [`ConFind::with_config`];
//! see the [`config`] module for defaults, validation rules and the option/output matrix.
//!
//! # References
//!
//! - Zheng, F., Zhang, J., & Grigoryan, G. (2015). "Tertiary Structural Propensities Reveal
//!   Fundamental Sequence/Structure Relationships in Proteins." *Structure*, 23(5), 961–971.
//! - Zheng, W., & Grigoryan, G. (2017). "Sequence statistics of tertiary structural motifs
//!   reflect protein stability." *PLoS ONE*, 12(5), e0178272.

pub mod cache;
pub mod config;
pub mod confind;
pub mod contact_list;
pub mod coords;
pub mod error;
pub mod freedom;
pub mod grid;
pub mod parallel;
pub mod params;
pub mod precondition;

pub use config::{
    CbClashPolicy, ConFindConfig, ConfigError, CrowdednessMode, Cutoffs, FreedomMode, ResidueScope,
};
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
