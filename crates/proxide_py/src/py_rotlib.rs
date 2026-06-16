use pyo3::prelude::*;

#[pyclass]
pub struct PyRotamerLibrary {}

#[pymethods]
impl PyRotamerLibrary {
    #[staticmethod]
    pub fn load_pb(path: &str) -> PyResult<Self> {
        // TODO: wire to proxide_rotlib::RotamerLibrary::load_pb()
        let _ = path;
        Ok(Self {})
    }

    pub fn residue_codes(&self) -> PyResult<Vec<String>> {
        // TODO: return real codes from inner library
        Ok(vec![])
    }
}
