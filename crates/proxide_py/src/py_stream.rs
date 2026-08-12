#![allow(clippy::useless_conversion)]
use crate::formats;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::fs::File;
use std::path::Path;

#[pyclass]
pub struct PyTrajectoryIterator {
    dcd_reader: Option<formats::dcd::DcdReader<File>>,
    num_atoms: usize,
    chunk_size: usize,
}

#[pymethods]
impl PyTrajectoryIterator {
    #[new]
    pub fn new(path: &str, chunk_size: usize) -> PyResult<Self> {
        let path_obj = Path::new(path);
        let ext = path_obj
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        if ext == "dcd" {
            let reader = formats::dcd::DcdReader::open(path)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            let num_atoms = reader.header.n_atoms;
            Ok(PyTrajectoryIterator {
                dcd_reader: Some(reader),
                num_atoms,
                chunk_size,
            })
        } else {
            Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unsupported or pending trajectory extension: {}",
                ext
            )))
        }
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>, py: Python<'_>) -> PyResult<Option<PyObject>> {
        let chunk_size = slf.chunk_size;
        let num_atoms = slf.num_atoms;
        let mut frames_coords = Vec::with_capacity(chunk_size * num_atoms * 3);
        let mut actual_frames = 0;

        if let Some(ref mut reader) = slf.dcd_reader {
            for _ in 0..chunk_size {
                match reader.read_frame() {
                    Ok(Some(frame)) => {
                        frames_coords.extend(frame.coordinates);
                        actual_frames += 1;
                    }
                    Ok(None) => break,
                    Err(e) => return Err(pyo3::exceptions::PyRuntimeError::new_err(e.to_string())),
                }
            }
        }

        if actual_frames == 0 {
            return Ok(None);
        }

        let dict = PyDict::new_bound(py);
        let shape = (actual_frames, num_atoms, 3);
        let array = PyArray1::from_slice_bound(py, &frames_coords)
            .reshape(shape)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        dict.set_item("coordinates", array)?;

        Ok(Some(dict.into_py(py)))
    }
}

#[pyclass]
pub struct PyDcdWriter {
    writer: formats::dcd::DcdWriter,
}

#[pymethods]
impl PyDcdWriter {
    #[new]
    pub fn new(path: &str, n_atoms: usize, delta: f32, has_unit_cell: bool) -> PyResult<Self> {
        let writer = formats::dcd::DcdWriter::create(path, n_atoms, delta, has_unit_cell)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(PyDcdWriter { writer })
    }

    #[pyo3(signature = (coords, unit_cell=None))]
    pub fn write_frame(
        &mut self,
        coords: PyReadonlyArray2<'_, f32>,
        unit_cell: Option<[f64; 6]>,
    ) -> PyResult<()> {
        let view = coords.as_array();
        let flat_coords: Vec<f32> = view.iter().copied().collect();
        self.writer
            .write_frame(&flat_coords, unit_cell.as_ref())
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    pub fn finalize(&mut self) -> PyResult<()> {
        self.writer
            .finalize()
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[pyo3(signature = (_exc_type=None, _exc_value=None, _traceback=None))]
    fn __exit__(
        &mut self,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc_value: Option<&Bound<'_, PyAny>>,
        _traceback: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        self.finalize()
    }
}
