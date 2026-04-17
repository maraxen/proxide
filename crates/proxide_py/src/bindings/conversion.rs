use numpy::{PyArray1, PyArrayMethods};
use proxide_rs::formatters::atom14::FormattedAtom14;
use proxide_rs::formatters::atom37::FormattedAtom37;
use proxide_rs::formatters::backbone::FormattedBackbone;
use proxide_rs::formatters::cache::CachedStructure;
use proxide_rs::formatters::full::FormattedFull;
use proxide_rs::structure::RawAtomData;
use pyo3::prelude::*;
use pyo3::types::PyDict;

pub trait ToPyDict {
    fn to_py_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>>;
}

impl ToPyDict for RawAtomData {
    fn to_py_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new_bound(py);

        let coords_arr = PyArray1::from_slice_bound(py, &self.coords);
        let reshaped = coords_arr.reshape((self.num_atoms, 3)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Failed to reshape coords: {}",
                e
            ))
        })?;

        dict.set_item("num_atoms", self.num_atoms)?;
        dict.set_item("coords", reshaped.clone())?;
        dict.set_item("coordinates", reshaped)?;

        dict.set_item("atom_names", self.atom_names.clone())?;
        dict.set_item("elements", self.elements.clone())?;
        dict.set_item("res_names", self.res_names.clone())?;
        dict.set_item("res_ids", self.res_ids.clone())?;
        dict.set_item("residue_index", self.res_ids.clone())?;
        dict.set_item("chain_ids", self.chain_ids.clone())?;
        dict.set_item("b_factors", self.b_factors.clone())?;
        dict.set_item("occupancy", self.occupancy.clone())?;
        dict.set_item("occupancies", self.occupancy.clone())?;
        dict.set_item("is_hetatm", self.is_hetatm.clone())?;
        dict.set_item("hetero", self.is_hetatm.clone())?;

        if let Some(ref charges) = self.charges {
            dict.set_item("charges", charges.clone())?;
        }
        if let Some(ref radii) = self.radii {
            dict.set_item("radii", radii.clone())?;
        }

        Ok(dict)
    }
}

impl ToPyDict for CachedStructure {
    fn to_py_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new_bound(py);
        dict.set_item(
            "coordinates",
            PyArray1::from_slice_bound(py, &self.coordinates),
        )?;
        dict.set_item("atom_mask", PyArray1::from_slice_bound(py, &self.atom_mask))?;
        dict.set_item("aatype", PyArray1::from_slice_bound(py, &self.aatype))?;
        dict.set_item(
            "residue_index",
            PyArray1::from_slice_bound(py, &self.residue_index),
        )?;
        dict.set_item(
            "chain_index",
            PyArray1::from_slice_bound(py, &self.chain_index),
        )?;

        if let Some(ref names) = self.atom_names {
            dict.set_item("atom_names", names.clone())?;
        }
        if let Some(shape) = self.coord_shape {
            dict.set_item("coord_shape", shape)?;
        }

        Ok(dict)
    }
}

impl ToPyDict for FormattedAtom37 {
    fn to_py_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new_bound(py);
        dict.set_item(
            "coordinates",
            PyArray1::from_slice_bound(py, &self.coordinates),
        )?;
        dict.set_item("atom_mask", PyArray1::from_slice_bound(py, &self.atom_mask))?;
        dict.set_item("aatype", PyArray1::from_slice_bound(py, &self.aatype))?;
        dict.set_item(
            "residue_index",
            PyArray1::from_slice_bound(py, &self.residue_index),
        )?;
        dict.set_item(
            "chain_index",
            PyArray1::from_slice_bound(py, &self.chain_index),
        )?;
        Ok(dict)
    }
}

impl ToPyDict for FormattedAtom14 {
    fn to_py_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new_bound(py);
        dict.set_item(
            "coordinates",
            PyArray1::from_slice_bound(py, &self.coordinates),
        )?;
        dict.set_item("atom_mask", PyArray1::from_slice_bound(py, &self.atom_mask))?;
        dict.set_item("aatype", PyArray1::from_slice_bound(py, &self.aatype))?;
        dict.set_item(
            "residue_index",
            PyArray1::from_slice_bound(py, &self.residue_index),
        )?;
        dict.set_item(
            "chain_index",
            PyArray1::from_slice_bound(py, &self.chain_index),
        )?;
        Ok(dict)
    }
}

impl ToPyDict for FormattedBackbone {
    fn to_py_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new_bound(py);
        dict.set_item(
            "coordinates",
            PyArray1::from_slice_bound(py, &self.coordinates),
        )?;
        dict.set_item("atom_mask", PyArray1::from_slice_bound(py, &self.atom_mask))?;
        dict.set_item("aatype", PyArray1::from_slice_bound(py, &self.aatype))?;
        dict.set_item(
            "residue_index",
            PyArray1::from_slice_bound(py, &self.residue_index),
        )?;
        dict.set_item(
            "chain_index",
            PyArray1::from_slice_bound(py, &self.chain_index),
        )?;
        Ok(dict)
    }
}

impl ToPyDict for FormattedFull {
    fn to_py_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new_bound(py);
        dict.set_item(
            "coordinates",
            PyArray1::from_slice_bound(py, &self.coordinates),
        )?;
        dict.set_item("atom_mask", PyArray1::from_slice_bound(py, &self.atom_mask))?;
        dict.set_item("aatype", PyArray1::from_slice_bound(py, &self.aatype))?;
        dict.set_item(
            "residue_index",
            PyArray1::from_slice_bound(py, &self.residue_index),
        )?;
        dict.set_item(
            "chain_index",
            PyArray1::from_slice_bound(py, &self.chain_index),
        )?;
        dict.set_item("atom_names", self.atom_names.clone())?;
        dict.set_item("atom_residue_ids", self.atom_residue_ids.clone())?;
        dict.set_item("coord_shape", self.coord_shape)?;
        Ok(dict)
    }
}
