// General-purpose geometry utilities exposed to Python (not tied to a
// specific chemistry/force-field or trajectory-format concern).
#![allow(clippy::useless_conversion)]

use crate::py_chemistry::extract_coords;
use pyo3::prelude::*;

/// Compute the (unweighted) radius of gyration of a set of coordinates.
///
/// `coordinates` may be a list of [x, y, z] points or an (N, 3) numpy array.
/// Used as an MD trajectory frame-quality signal: elevated Rg relative to
/// the native/crystal reference indicates an unfolded/extended conformation.
#[pyfunction]
pub fn radius_of_gyration(py: Python<'_>, coordinates: PyObject) -> PyResult<f32> {
    let coords = extract_coords(py, &coordinates)?;
    Ok(proxide_geometry::geometry::radius_of_gyration::radius_of_gyration(&coords))
}

/// Compute the mass-weighted radius of gyration of a set of coordinates.
///
/// `weights` must be the same length as `coordinates` (e.g. atomic masses).
#[pyfunction]
pub fn weighted_radius_of_gyration(
    py: Python<'_>,
    coordinates: PyObject,
    weights: Vec<f32>,
) -> PyResult<f32> {
    let coords = extract_coords(py, &coordinates)?;
    if coords.len() != weights.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "coordinates and weights must be the same length",
        ));
    }
    Ok(proxide_geometry::geometry::radius_of_gyration::weighted_radius_of_gyration(&coords, &weights))
}
