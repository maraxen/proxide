use proxide_rs::structure::systems::AtomicSystem;
use pyo3::prelude::*;

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
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (coordinates, atom_mask, atom_names = None, elements = None, bonds = None, charges = None, sigmas = None, epsilons = None, radii = None, residue_index = None, chain_index = None))]
    fn new(
        coordinates: Vec<f32>,
        atom_mask: Vec<f32>,
        atom_names: Option<Vec<String>>,
        elements: Option<Vec<String>>,
        bonds: Option<Vec<[usize; 2]>>,
        charges: Option<Vec<f32>>,
        sigmas: Option<Vec<f32>>,
        epsilons: Option<Vec<f32>>,
        radii: Option<Vec<f32>>,
        residue_index: Option<Vec<i32>>,
        chain_index: Option<Vec<i32>>,
    ) -> Self {
        use proxide_rs::structure::systems::AtomicSystemArgs;
        Self {
            inner: AtomicSystem::new(AtomicSystemArgs {
                coordinates,
                atom_mask,
                atom_names,
                elements,
                bonds,
                charges,
                sigmas,
                epsilons,
                radii,
                residue_index,
                chain_index,
            }),
        }
    }

    #[getter]
    fn get_coordinates(&self, py: Python) -> PyObject {
        numpy::PyArray1::from_vec_bound(py, self.inner.coordinates.clone())
            .into_any()
            .unbind()
    }

    #[getter]
    fn get_atom_mask(&self, py: Python) -> PyObject {
        numpy::PyArray1::from_vec_bound(py, self.inner.atom_mask.clone())
            .into_any()
            .unbind()
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
            let arr = numpy::PyArray1::from_vec_bound(py, flat_bonds)
                .reshape([bonds.len(), 2])
                .unwrap();
            arr.into_any().unbind()
        } else {
            py.None()
        }
    }
}
