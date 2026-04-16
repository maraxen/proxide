use pyo3::prelude::*;
use proxide_rs::structure::systems::AtomicSystem;

#[pyclass(name = "AtomicSystem")]
pub struct PyAtomicSystem {
    pub inner: AtomicSystem,
}

impl PyAtomicSystem {
    pub fn from_core(inner: AtomicSystem) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyAtomicSystem {
    #[new]
    #[pyo3(signature = (coordinates, atom_mask, atom_names = None, elements = None))]
    fn new(coordinates: Vec<f32>, atom_mask: Vec<f32>, atom_names: Option<Vec<String>>, elements: Option<Vec<String>>) -> Self {
        Self {
            inner: AtomicSystem::new(coordinates, atom_mask, atom_names, elements),
        }
    }

    #[getter]
    fn get_coordinates(&self, py: Python) -> PyObject {
        numpy::PyArray1::from_vec_bound(py, self.inner.coordinates.clone()).into_any().unbind()
    }

    #[getter]
    fn get_atom_mask(&self, py: Python) -> PyObject {
        numpy::PyArray1::from_vec_bound(py, self.inner.atom_mask.clone()).into_any().unbind()
    }

    #[getter]
    fn get_atom_names(&self) -> Vec<String> {
        self.inner.atom_names.clone()
    }

    #[getter]
    fn get_elements(&self) -> Vec<String> {
        self.inner.elements.clone()
    }

    #[getter]
    fn get_bonds(&self, py: Python) -> PyObject {
        use numpy::PyArrayMethods;
        if let Some(ref bonds) = self.inner.bonds {
            let flat_bonds: Vec<i32> = bonds.iter().flatten().map(|&x| x as i32).collect();
            let arr = numpy::PyArray1::from_vec_bound(py, flat_bonds).reshape([bonds.len(), 2]).unwrap();
            arr.into_any().unbind()
        } else {
            py.None()
        }
    }
}
