use pyo3::prelude::*;
use proxide_rs::structure::systems::AtomicSystem;

#[pyclass]
pub struct PyAtomicSystem {
    pub inner: AtomicSystem,
}

#[pymethods]
impl PyAtomicSystem {
    #[new]
    fn new(coordinates: Vec<f32>, atom_mask: Vec<f32>, atom_names: Option<Vec<String>>, elements: Option<Vec<String>>) -> Self {
        Self {
            inner: AtomicSystem::new(coordinates, atom_mask, atom_names, elements),
        }
    }
}
