// TODO: Review allow attributes at a later point
#![allow(clippy::useless_conversion)]

use crate::{chem, forcefield, physics};
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// Assign atomic masses based on atom names
#[pyfunction]
pub fn assign_masses(atom_names: Vec<String>) -> PyResult<Vec<f32>> {
    Ok(chem::masses::assign_masses(&atom_names))
}

/// Assign GAFF atom types to a structure (exposed to Python)
#[pyfunction]
pub fn assign_gaff_atom_types(
    py: Python<'_>,
    coordinates: PyObject,
    elements: Vec<String>,
) -> PyResult<Vec<Option<String>>> {
    let coords = extract_coords(py, &coordinates)?;

    // Default tolerance for bond inference
    let topology = forcefield::topology::Topology::from_coords(&coords, &elements, 1.3);

    let gaff = forcefield::gaff::GaffParameters::new();
    let types = forcefield::gaff::assign_gaff_types(&elements, &topology, &gaff);

    Ok(types)
}

/// Assign intrinsic radii using the MBondi2 scheme
#[pyfunction]
pub fn assign_mbondi2_radii(atom_names: Vec<String>, bonds: Vec<[usize; 2]>) -> PyResult<Vec<f32>> {
    let radii = physics::gbsa::assign_mbondi2_radii(&atom_names, &bonds);
    Ok(radii)
}

/// Assign scaling factors for OBC2 GBSA calculation
#[pyfunction]
pub fn assign_obc2_scaling_factors(atom_names: Vec<String>) -> Result<Vec<f32>, PyErr> {
    let factors = physics::gbsa::assign_obc2_scaling_factors(&atom_names);
    Ok(factors)
}

/// Get water model parameters
#[pyfunction]
pub fn get_water_model(name: String, rigid: bool) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let model = physics::water::get_water_model(&name, rigid)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;

        let dict = PyDict::new_bound(py);
        dict.set_item("name", &model.name)?;
        dict.set_item("atoms", &model.atoms)?;
        dict.set_item("has_virtual_sites", model.has_virtual_sites)?;

        // Charges dict
        let charges_dict = PyDict::new_bound(py);
        for (k, v) in &model.charges {
            charges_dict.set_item(k, *v)?;
        }
        dict.set_item("charges", charges_dict)?;

        // Sigmas dict
        let sigmas_dict = PyDict::new_bound(py);
        for (k, v) in &model.sigmas {
            sigmas_dict.set_item(k, *v)?;
        }
        dict.set_item("sigmas", sigmas_dict)?;

        // Epsilons dict
        let epsilons_dict = PyDict::new_bound(py);
        for (k, v) in &model.epsilons {
            epsilons_dict.set_item(k, *v)?;
        }
        dict.set_item("epsilons", epsilons_dict)?;

        // Bonds: list of (atom1, atom2, length, k)
        let bonds: Vec<(&str, &str, f32, f32)> = model
            .bonds
            .iter()
            .map(|(a, b, l, k)| (a.as_str(), b.as_str(), *l, *k))
            .collect();
        dict.set_item("bonds", bonds)?;

        // Angles: list of (a1, a2, a3, theta, k)
        let angles: Vec<(&str, &str, &str, f32, f32)> = model
            .angles
            .iter()
            .map(|(a, b, c, t, k)| (a.as_str(), b.as_str(), c.as_str(), *t, *k))
            .collect();
        dict.set_item("angles", angles)?;

        // Constraints
        let constraints: Vec<(&str, &str, f32)> = model
            .constraints
            .iter()
            .map(|(a, b, d)| (a.as_str(), b.as_str(), *d))
            .collect();
        dict.set_item("constraints", constraints)?;

        Ok(dict.into_py(py))
    })
}

/// Compute bicubic interpolation parameters for CMAP
#[pyfunction]
pub fn compute_bicubic_params(grid: Vec<Vec<f64>>) -> PyResult<Vec<Vec<[f64; 4]>>> {
    let params = physics::cmap::compute_bicubic_params(&grid);
    Ok(params)
}

/// Parameterize a molecule using GAFF for ligands and small molecules
#[pyfunction]
#[pyo3(signature = (coordinates, elements, bond_tolerance=1.3))]
pub fn parameterize_molecule(
    py: Python<'_>,
    coordinates: PyObject,
    elements: Vec<String>,
    bond_tolerance: f32,
) -> PyResult<PyObject> {
    let coords = extract_coords(py, &coordinates)?;

    let params = physics::md_params::parameterize_molecule(&coords, &elements, bond_tolerance)
        .map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Parameterization failed: {}", e))
        })?;

    let dict = PyDict::new_bound(py);

    // Basic info
    dict.set_item("num_parameterized", params.num_parameterized)?;
    dict.set_item("num_skipped", params.num_skipped)?;

    // Atom types
    let atom_types: Vec<&str> = params.atom_types.iter().map(|s| s.as_str()).collect();
    dict.set_item("atom_types", atom_types)?;

    // LJ parameters
    let charges = PyArray1::from_slice_bound(py, &params.charges);
    let sigmas = PyArray1::from_slice_bound(py, &params.sigmas);
    let epsilons = PyArray1::from_slice_bound(py, &params.epsilons);
    dict.set_item("charges", charges)?;
    dict.set_item("sigmas", sigmas)?;
    dict.set_item("epsilons", epsilons)?;

    // Bonds (N, 2)
    if !params.bonds.is_empty() {
        let mut flat = Vec::with_capacity(params.bonds.len() * 2);
        for b in &params.bonds {
            flat.extend_from_slice(b);
        }
        let arr = PyArray1::from_slice_bound(py, &flat);
        dict.set_item("bonds", arr.reshape((params.bonds.len(), 2)).unwrap())?;
    }

    // Bond params (N, 2)
    if !params.bond_params.is_empty() {
        let mut flat = Vec::with_capacity(params.bond_params.len() * 2);
        for p in &params.bond_params {
            flat.extend_from_slice(p);
        }
        let arr = PyArray1::from_slice_bound(py, &flat);
        dict.set_item(
            "bond_params",
            arr.reshape((params.bond_params.len(), 2)).unwrap(),
        )?;
    }

    // Angles (N, 3)
    if !params.angles.is_empty() {
        let mut flat = Vec::with_capacity(params.angles.len() * 3);
        for a in &params.angles {
            flat.extend_from_slice(a);
        }
        let arr = PyArray1::from_slice_bound(py, &flat);
        dict.set_item("angles", arr.reshape((params.angles.len(), 3)).unwrap())?;
    }

    // Dihedrals (N, 4)
    if !params.dihedrals.is_empty() {
        let mut flat = Vec::with_capacity(params.dihedrals.len() * 4);
        for d in &params.dihedrals {
            flat.extend_from_slice(d);
        }
        let arr = PyArray1::from_slice_bound(py, &flat);
        dict.set_item(
            "dihedrals",
            arr.reshape((params.dihedrals.len(), 4)).unwrap(),
        )?;
    }

    Ok(dict.into_py(py))
}

fn extract_coords(py: Python<'_>, obj: &PyObject) -> PyResult<Vec<[f32; 3]>> {
    let bound = obj.bind(py);

    if let Ok(l) = bound.downcast::<PyList>() {
        let mut coords = Vec::with_capacity(l.len());
        for item in l {
            let point: Vec<f32> = item.extract()?;
            if point.len() != 3 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Coordinates must be 3D points",
                ));
            }
            coords.push([point[0], point[1], point[2]]);
        }
        return Ok(coords);
    }

    if let Ok(array) = bound.downcast::<PyArray2<f32>>() {
        let binding = array.readonly();
        let data = binding.as_array();
        let shape = data.shape();
        if shape[1] != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Numpy array must be Nx3",
            ));
        }

        let mut coords = Vec::with_capacity(shape[0]);
        for i in 0..shape[0] {
            coords.push([data[[i, 0]], data[[i, 1]], data[[i, 2]]]);
        }
        return Ok(coords);
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "Expected list of lists or numpy array for coordinates",
    ))
}
