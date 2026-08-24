/// Error type for `proxide-tmalign` operations.
#[derive(Debug, thiserror::Error)]
pub enum TmAlignError {
    #[error("structure has no residues with a resolved Cα atom")]
    EmptyStructure,
    /// A fixed-correspondence batch was given candidates of differing lengths.
    /// Distinct from a parse failure: the inputs are individually valid, but the
    /// caller asserted they share a topology and they do not.
    #[error("expected {expected} residues to match the query's topology, found {found}")]
    LengthMismatch { expected: usize, found: usize },
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("PDB parse error: {0}")]
    Parse(String),
}
