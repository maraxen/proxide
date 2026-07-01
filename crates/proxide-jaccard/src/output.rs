//! Writes the dense distance matrix to a `.npy` file readable via
//! `numpy.load` / `jax.numpy.load`, plus a sidecar manifest listing
//! accessions in row/column order (`<stem>.accessions.txt`, one per line).
//!
//! Dense symmetric `.npy` is an MVP choice, not the long-term format — see
//! the tech-debt note filed alongside this crate for the planned
//! condensed/sparse output path once Rust owns the full read→compute→write
//! pipeline and the matrix sizes that need it are known.

use crate::error::{JaccardError, Result};
use ndarray::Array2;
use std::path::{Path, PathBuf};

pub fn write_distance_matrix(
    npy_path: &Path,
    mat: &Array2<f32>,
    accessions: &[String],
) -> Result<()> {
    ndarray_npy::write_npy(npy_path, mat).map_err(|e| JaccardError::NpyWrite {
        path: npy_path.to_path_buf(),
        message: e.to_string(),
    })?;

    let manifest_path = sidecar_path(npy_path);
    std::fs::write(&manifest_path, accessions.join("\n")).map_err(|source| JaccardError::Open {
        path: manifest_path,
        source,
    })?;

    Ok(())
}

fn sidecar_path(npy_path: &Path) -> PathBuf {
    let stem = npy_path
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned();
    let mut p = npy_path.to_path_buf();
    p.set_file_name(format!("{stem}.accessions.txt"));
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn writes_npy_and_manifest_sidecar() {
        let dir = tempfile::tempdir().unwrap();
        let npy_path = dir.path().join("dist.npy");
        let mat = array![[0.0f32, 0.5], [0.5, 0.0]];
        let accessions = vec!["GCA_000002425.3".to_string(), "GCA_000002426.1".to_string()];

        write_distance_matrix(&npy_path, &mat, &accessions).unwrap();

        assert!(npy_path.exists());
        let manifest_path = dir.path().join("dist.accessions.txt");
        let contents = std::fs::read_to_string(manifest_path).unwrap();
        assert_eq!(contents, "GCA_000002425.3\nGCA_000002426.1");
    }
}
