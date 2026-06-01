use crate::coords::ResidueIndex;

#[derive(Debug, thiserror::Error)]
pub enum ConFindError {
    #[error("residue {0:?} not cached — call cache_residue first")]
    NotCached(ResidueIndex),
    #[error("residue {0:?} missing backbone atom {1}")]
    MissingBackbone(ResidueIndex, &'static str),
    #[error("freedom not computed for {0:?} — must be in contacts() query set")]
    FreedomNotComputed(ResidueIndex),
    #[error("rotamer library error: {0}")]
    RotlibError(#[from] proxide_rotlib::RotlibError),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}
