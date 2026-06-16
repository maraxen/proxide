use thiserror::Error;

#[derive(Debug, Error)]
pub enum RotlibError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("invalid library format: {0}")]
    InvalidFormat(String),

    #[error("unknown amino acid '{0}'")]
    UnknownAa(String),

    #[error("rotamer index {1} out of range for '{0}' (bin has {2})")]
    RotIndexOob(String, usize, usize),

    #[error("protobuf decode error: {0}")]
    Protobuf(#[from] prost::DecodeError),

    #[error("missing required attribution field in rotamer library")]
    MissingAttribution,

    #[error("unsupported geometry mode: {0}")]
    UnsupportedGeometryMode(i32),
}
