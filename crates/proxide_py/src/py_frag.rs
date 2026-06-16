use pyo3::prelude::*;
use std::collections::HashMap;

/// Search a fragment database (proxide-frag). Stub — returns empty results.
#[pyfunction]
#[pyo3(signature = (query, db_path, n_results = 100, n_threads = None))]
pub fn search_fragments(
    query: &crate::bindings::atomic_system::PyAtomicSystem,
    db_path: &str,
    n_results: usize,
    n_threads: Option<usize>,
) -> PyResult<Vec<HashMap<String, PyObject>>> {
    let _ = (query, db_path, n_results, n_threads);
    Ok(vec![])
}
