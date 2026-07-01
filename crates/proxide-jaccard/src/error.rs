use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum JaccardError {
    #[error("failed to open parquet file {path}: {source}")]
    Open {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("parquet error: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),

    #[error("arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),

    #[error("column '{column}' has unexpected arrow type {found} (expected {expected})")]
    Schema {
        column: &'static str,
        expected: &'static str,
        found: String,
    },

    #[error("column '{column}' not found in parquet schema")]
    MissingColumn { column: &'static str },

    #[error("requested accessions not found in the parquet input: {missing:?}")]
    MissingAccessions { missing: Vec<String> },

    #[error("failed to write npy output to {path}: {message}")]
    NpyWrite { path: PathBuf, message: String },

    #[error("failed to (de)serialize accession index at {path}: {message}")]
    Serialize { path: PathBuf, message: String },

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, JaccardError>;
