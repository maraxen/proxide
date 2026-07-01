//! Pairwise Jaccard distance matrices over scaled-MinHash genome sketches.
//!
//! Input is a 2-column parquet (`accession: Utf8`, `hashes_list: List<Int64>`)
//! of sourmash-style scaled MinHash signatures. Given a set of requested
//! accessions, this crate loads their sketches, fills the upper-triangle of
//! an n×n Jaccard distance matrix in parallel (`orx-parallel`), mirrors it
//! to a dense symmetric matrix, and writes it as `.npy` for downstream
//! `numpy`/`jax.numpy` consumption.

pub mod accessions;
mod distance;
mod error;
pub mod ipc_index;
mod matrix;
mod output;
mod sketch;

pub use distance::{
    containment, jaccard_distance, jaccard_similarity, overlap, overlap_coefficient, Overlap,
};
pub use error::{JaccardError, Result};
pub use matrix::{pairwise_containment, pairwise_jaccard_distance};
pub use output::write_distance_matrix;
pub use sketch::SketchStore;
