use crate::config::ConfigError;
use crate::coords::ResidueIndex;

/// Error type for ConFind operations.
#[derive(Debug, thiserror::Error)]
pub enum ConFindError {
    #[error("residue {0:?} not cached — call cache_residue first")]
    NotCached(ResidueIndex),
    #[error("residue {0:?} missing backbone atom {1}")]
    MissingBackbone(ResidueIndex, &'static str),
    #[error("residue {residue:?} atom '{atom_name}' has a non-finite coordinate")]
    NonFiniteCoordinate {
        residue: ResidueIndex,
        atom_name: String,
    },
    #[error("residue {0:?} is not in the configured analysis scope")]
    OutOfScope(ResidueIndex),
    #[error("invalid configuration: {0}")]
    InvalidConfig(#[from] ConfigError),
    #[error("structure failed {0} precondition check(s); see log for per-residue diagnostics")]
    PreconditionsFailed(usize),
    #[error("invalid BackboneOptions: {0}")]
    InvalidOptions(String),
    #[error("rotamer library error: {0}")]
    RotlibError(#[from] proxide_rotlib::RotlibError),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}
