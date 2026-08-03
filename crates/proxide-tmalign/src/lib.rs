//! `proxide-tmalign` — TM-align structural alignment algorithm.
//!
//! A Rust/orx-parallel reimplementation of the pairwise (protein, Cα-based)
//! algorithm at the core of Zhang Lab's TM-align / USalign
//! (<https://github.com/pylelab/USalign>), following the same design
//! conventions as this workspace's other single-algorithm crates
//! (`proxide-frag`, `proxide-confind`, `proxide-jaccard`).
//!
//! # Status
//!
//! Under active, phased construction. Landed so far: Cα/sequence
//! extraction ([`structure`]), amino-acid code lookup ([`seq`]), TM-score
//! length-normalization formulas ([`d0`]), a generalized Kabsch
//! superposition ([`kabsch`]), and a generic affine-gap Needleman-Wunsch DP
//! core ([`nw`]). Not yet landed: the 5 TM-align initial-seeding
//! strategies, the iterative `DP_iter` refinement loop tying the pieces
//! above together into a full pairwise alignment, and the
//! parallel/pairwise-batch APIs.
//!
//! # References
//!
//! - Zhang Y, Skolnick J. "TM-align: a protein structure alignment
//!   algorithm based on the TM-score." *Nucleic Acids Res.* 2005;33(7):2302-9.
//! - Zhang Y, Skolnick J. "Scoring function for automated assessment of
//!   protein structure template quality." *Proteins.* 2004;57(4):702-10.

pub mod d0;
pub mod error;
pub mod kabsch;
pub mod nw;
pub mod nwdp_tm;
pub mod parallel;
pub mod pipeline;
pub mod refine;
pub mod score;
pub mod seed;
pub mod seq;
pub mod ss;
pub mod structure;

pub use error::TmAlignError;
pub use structure::{extract_ca_trace, load_pdb_ca_trace, CaTrace};
