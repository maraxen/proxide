// General-purpose geometry utilities exposed to Python (not tied to a
// specific chemistry/force-field or trajectory-format concern).
#![allow(clippy::useless_conversion)]

use crate::py_chemistry::extract_coords;
use pyo3::prelude::*;
use pyo3::types::PyDict;

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
    Ok(
        proxide_geometry::geometry::radius_of_gyration::weighted_radius_of_gyration(
            &coords, &weights,
        ),
    )
}

/// Compute the optimal RMSD (and superposition rotation) between two
/// equal-length coordinate sets, via the Kabsch algorithm.
///
/// Both `coords_a` and `coords_b` may be a list of [x, y, z] points or an
/// (N, 3) numpy array, and are centered (centroid subtracted) internally --
/// callers do not need to pre-center. Returns a dict with `rmsd` (float,
/// `inf` if the inputs are degenerate: mismatched length, empty, or SVD
/// failure) and `rotation` (3x3 row-major list of lists, R such that
/// R @ a ~= b).
///
/// This is a general, arbitrary-length RMSD -- distinct from proxide-frag's
/// fixed-N backbone-fragment `kabsch_rmsd` used for MASTER-style structural
/// search. Used as an MD trajectory frame-quality signal alongside
/// `radius_of_gyration`: elevated RMSD to the native/crystal reference
/// indicates a conformation that has drifted from the reference state.
#[pyfunction]
pub fn kabsch_rmsd(py: Python<'_>, coords_a: PyObject, coords_b: PyObject) -> PyResult<PyObject> {
    let a = extract_coords(py, &coords_a)?;
    let b = extract_coords(py, &coords_b)?;
    if a.len() != b.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "coords_a and coords_b must be the same length",
        ));
    }

    let result = proxide_geometry::geometry::rmsd::rmsd_with_centering(&a, &b);

    let dict = PyDict::new_bound(py);
    dict.set_item("rmsd", result.rmsd)?;
    dict.set_item(
        "rotation",
        result
            .rotation
            .iter()
            .map(|row| row.to_vec())
            .collect::<Vec<_>>(),
    )?;
    Ok(dict.into_py(py))
}
