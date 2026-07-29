/// Error type for `proxide-tmalign` operations.
#[derive(Debug, thiserror::Error)]
pub enum TmAlignError {
    #[error("structure has no residues with a resolved Cα atom")]
    EmptyStructure,
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("PDB parse error: {0}")]
    Parse(String),
}
