//! Python bindings for `proxide-jaccard`: pairwise MinHash Jaccard/containment
//! distance matrices over a sourmash-style scaled-MinHash sketch parquet.

use numpy::PyArray1;
use numpy::PyArrayMethods;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::Path;

use proxide_jaccard::accessions::read_accession_list;
use proxide_jaccard::{pairwise_containment, pairwise_jaccard_distance, SketchStore};

/// Compute a pairwise MinHash distance/similarity matrix from a sketch parquet.
///
/// `parquet_path` is a 2-column parquet (`accession: Utf8`, `hashes_list:
/// List<Int64>`). `accessions_path`, if given, is a text file with one
/// accession per line; only those rows are loaded (order preserved as found).
/// Omit it to use every row in the parquet.
///
/// `metric` is `"jaccard"` (default, symmetric Jaccard *distance*,
/// `1 - |A∩B|/|A∪B|`, zero diagonal) or `"containment"` (asymmetric:
/// `matrix[i, j]` = fraction of accession `i` contained in `j`, unit diagonal).
///
/// Returns a dict: `{"matrix": float32 (n, n), "accessions": list[str],
/// "metric": str}`. Row/column `k` corresponds to `accessions[k]`.
#[pyfunction]
#[pyo3(signature = (parquet_path, accessions_path=None, metric="jaccard"))]
pub fn jaccard_distance_matrix(
    py: Python<'_>,
    parquet_path: String,
    accessions_path: Option<String>,
    metric: &str,
) -> PyResult<PyObject> {
    let metric_owned = metric.to_string();
    let (flat, n, accessions) = py
        .allow_threads(|| -> Result<(Vec<f32>, usize, Vec<String>), String> {
            let wanted = match accessions_path.as_deref() {
                Some(p) => Some(read_accession_list(Path::new(p)).map_err(|e| e.to_string())?),
                None => None,
            };
            let store = SketchStore::load_parquet(Path::new(&parquet_path), wanted.as_deref())
                .map_err(|e| e.to_string())?;
            let mat = match metric_owned.as_str() {
                "jaccard" => pairwise_jaccard_distance(&store),
                "containment" => pairwise_containment(&store),
                other => {
                    return Err(format!(
                        "unknown metric: {other} (expected 'jaccard' or 'containment')"
                    ))
                }
            };
            let n = store.len();
            // Matrices are built via ndarray `Array2::zeros`, so they are
            // C-contiguous and `as_slice()` yields the row-major buffer.
            let flat = mat
                .as_slice()
                .ok_or_else(|| "distance matrix is not contiguous".to_string())?
                .to_vec();
            let accessions = store.accessions().to_vec();
            Ok((flat, n, accessions))
        })
        .map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "jaccard_distance_matrix failed: {}",
                e
            ))
        })?;

    let dict = PyDict::new_bound(py);
    let arr = PyArray1::from_slice_bound(py, &flat).reshape((n, n))?;
    dict.set_item("matrix", arr)?;
    dict.set_item("accessions", accessions)?;
    dict.set_item("metric", metric)?;
    Ok(dict.into_py(py))
}
