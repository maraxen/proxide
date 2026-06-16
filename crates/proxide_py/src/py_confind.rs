use pyo3::prelude::*;
use std::collections::HashMap;

/// Run ConFind on a structure and return contact pairs as a list of dicts.
/// Each dict: {"res_i": str, "res_j": str, "contact_degree": f64}
#[pyfunction]
#[pyo3(signature = (structure, cd_threshold = 0.0))]
pub fn run_confind(
    structure: &crate::bindings::atomic_system::PyAtomicSystem,
    cd_threshold: f64,
) -> PyResult<Vec<HashMap<String, PyObject>>> {
    // TODO: wire to proxide_confind::run_confind()
    let _ = (structure, cd_threshold);
    Ok(vec![])
}
