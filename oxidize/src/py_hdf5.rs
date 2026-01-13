// TODO: Review allow attributes at a later point
#![allow(clippy::useless_conversion)]

use crate::formats;
use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;

// =============================================================================
// HDF5 Parsing Functions (feature-gated)
// =============================================================================

/// Parse MDTraj HDF5 file metadata
#[cfg(feature = "mdcath")]
#[pyfunction]
pub fn parse_mdtraj_h5_metadata(path: String) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let result = formats::mdtraj_h5::parse_mdtraj_h5_metadata(&path)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let dict = PyDict::new_bound(py);
        dict.set_item("num_frames", result.num_frames)?;
        dict.set_item("num_atoms", result.num_atoms)?;
        dict.set_item("atom_names", &result.atom_names)?;
        dict.set_item("res_names", &result.res_names)?;
        dict.set_item("res_ids", &result.res_ids)?;
        dict.set_item("chain_ids", &result.chain_ids)?;
        dict.set_item("elements", &result.elements)?;

        Ok(dict.into_py(py))
    })
}

/// Parse a single frame from MDTraj HDF5 file
#[cfg(feature = "mdcath")]
#[pyfunction]
pub fn parse_mdtraj_h5_frame(path: String, frame_idx: usize) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let frame = formats::mdtraj_h5::parse_mdtraj_h5_frame(&path, frame_idx)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let dict = PyDict::new_bound(py);
        dict.set_item("index", frame.index)?;
        dict.set_item("time", frame.time)?;

        // Convert coords to NumPy array
        let coords_array = PyArray1::from_slice_bound(py, &frame.coords);
        let num_atoms = frame.coords.len() / 3;
        let coords_reshaped = coords_array.reshape((num_atoms, 3)).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Failed to reshape coords: {}", e))
        })?;
        dict.set_item("coords", coords_reshaped)?;

        Ok(dict.into_py(py))
    })
}

/// Parse MDCATH HDF5 file metadata
#[cfg(feature = "mdcath")]
#[pyfunction]
pub fn parse_mdcath_metadata(path: String) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let result = formats::mdcath_h5::parse_mdcath_metadata(&path)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let dict = PyDict::new_bound(py);
        dict.set_item("domain_id", &result.domain_id)?;
        dict.set_item("num_residues", result.num_residues)?;
        dict.set_item("resnames", &result.resnames)?;
        dict.set_item("chain_ids", &result.chain_ids)?;
        dict.set_item("temperatures", &result.temperatures)?;

        Ok(dict.into_py(py))
    })
}

/// Get list of replicas for a temperature in MDCATH file
#[cfg(feature = "mdcath")]
#[pyfunction]
pub fn get_mdcath_replicas(
    path: String,
    domain_id: String,
    temperature: String,
) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let replicas =
            formats::mdcath_h5::get_replicas(&path, &domain_id, &temperature).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Failed to get MDCATH replicas: {}",
                    e
                ))
            })?;

        Ok(replicas.into_py(py))
    })
}

/// Parse a single frame from MDCATH HDF5 file
#[cfg(feature = "mdcath")]
#[pyfunction]
#[pyo3(signature = (path, domain_id, temperature, replica, frame_idx))]
pub fn parse_mdcath_frame(
    path: String,
    domain_id: String,
    temperature: String,
    replica: String,
    frame_idx: usize,
) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let frame = formats::mdcath_h5::parse_mdcath_frame(
            &path,
            &domain_id,
            &temperature,
            &replica,
            frame_idx,
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let dict = PyDict::new_bound(py);
        dict.set_item("temperature", &frame.temperature)?;
        dict.set_item("replica", &frame.replica)?;
        dict.set_item("frame_idx", frame.frame_idx)?;

        // Convert coords to NumPy array
        let coords_array = PyArray1::from_slice_bound(py, &frame.coords);
        let num_atoms = frame.coords.len() / 3;
        let coords_reshaped = coords_array.reshape((num_atoms, 3)).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Failed to reshape coords: {}", e))
        })?;
        dict.set_item("coords", coords_reshaped)?;

        Ok(dict.into_py(py))
    })
}

// Stub functions when mdcath feature is not enabled
#[cfg(not(feature = "mdcath"))]
#[pyfunction]
pub fn parse_mdtraj_h5_metadata(_path: String) -> PyResult<PyObject> {
    Err(pyo3::exceptions::PyImportError::new_err(
        "HDF5 support requires compiling with 'mdcath' feature. Rebuild with: maturin develop --features mdcath",
    ))
}

#[cfg(not(feature = "mdcath"))]
#[pyfunction]
pub fn parse_mdtraj_h5_frame(_path: String, _frame_idx: usize) -> PyResult<PyObject> {
    Err(pyo3::exceptions::PyImportError::new_err(
        "HDF5 support requires compiling with 'mdcath' feature.",
    ))
}

#[cfg(not(feature = "mdcath"))]
#[pyfunction]
pub fn parse_mdcath_metadata(_path: String) -> PyResult<PyObject> {
    Err(pyo3::exceptions::PyImportError::new_err(
        "HDF5 support requires compiling with 'mdcath' feature.",
    ))
}

#[cfg(not(feature = "mdcath"))]
#[pyfunction]
pub fn get_mdcath_replicas(
    _path: String,
    _domain_id: String,
    _temperature: String,
) -> PyResult<PyObject> {
    Err(pyo3::exceptions::PyImportError::new_err(
        "HDF5 support requires compiling with 'mdcath' feature.",
    ))
}

#[cfg(not(feature = "mdcath"))]
#[pyfunction]
#[pyo3(signature = (_path, _domain_id, _temperature, _replica, _frame_idx))]
pub fn parse_mdcath_frame(
    _path: String,
    _domain_id: String,
    _temperature: String,
    _replica: String,
    _frame_idx: usize,
) -> PyResult<PyObject> {
    Err(pyo3::exceptions::PyImportError::new_err(
        "HDF5 support requires compiling with 'mdcath' feature.",
    ))
}
