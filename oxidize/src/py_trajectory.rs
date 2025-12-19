use crate::formats;
use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Parse an XTC trajectory file
#[pyfunction]
pub fn parse_xtc(path: String) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        // Prefer pure-Rust molly implementation (xtc-pure feature)
        #[cfg(feature = "xtc-pure")]
        {
            use formats::xtc::molly_impl::read_xtc_molly;
            let traj = read_xtc_molly(&path).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("XTC parsing failed: {}", e))
            })?;

            let dict = PyDict::new_bound(py);
            dict.set_item("num_frames", traj.num_frames)?;
            dict.set_item("num_atoms", traj.num_atoms)?;

            // Convert to NumPy arrays
            let times = PyArray1::from_slice_bound(py, &traj.times);
            dict.set_item("times", times)?;

            // Combine all coords into (N_frames, N_atoms, 3)
            let mut flat_coords = Vec::with_capacity(traj.num_frames * traj.num_atoms * 3);
            for frame_coords in &traj.coords {
                flat_coords.extend_from_slice(frame_coords);
            }
            let coords_array = PyArray1::from_slice_bound(py, &flat_coords);
            let shape = (traj.num_frames, traj.num_atoms, 3);
            let coords_reshaped = coords_array.reshape(shape).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("Failed to reshape coords: {}", e))
            })?;

            dict.set_item("coordinates", coords_reshaped)?;

            return Ok(dict.into_py(py));
        }

        // Fallback to chemfiles (trajectories feature) - may crash with SIGFPE
        #[cfg(all(feature = "trajectories", not(feature = "xtc-pure")))]
        {
            use formats::xtc::chemfiles_impl::read_xtc_chemfiles;
            let traj = read_xtc_chemfiles(&path).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("XTC parsing failed: {}", e))
            })?;

            let dict = PyDict::new_bound(py);
            dict.set_item("num_frames", traj.num_frames)?;
            dict.set_item("num_atoms", traj.num_atoms)?;

            // Convert to NumPy arrays
            let times = PyArray1::from_slice_bound(py, &traj.times);
            dict.set_item("times", times)?;

            // Combine all coords into (N_frames, N_atoms, 3)
            let mut flat_coords = Vec::with_capacity(traj.num_frames * traj.num_atoms * 3);
            for frame_coords in &traj.coords {
                flat_coords.extend_from_slice(frame_coords);
            }
            let coords_array = PyArray1::from_slice_bound(py, &flat_coords);
            let shape = (traj.num_frames, traj.num_atoms, 3);
            let coords_reshaped = coords_array.reshape(shape).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("Failed to reshape coords: {}", e))
            })?;

            dict.set_item("coordinates", coords_reshaped)?;

            return Ok(dict.into_py(py));
        }

        #[cfg(not(any(feature = "trajectories", feature = "xtc-pure")))]
        {
            Err(pyo3::exceptions::PyImportError::new_err(
                "XTC support requires compiling with 'xtc-pure' (recommended) or 'trajectories' feature.",
            ))
        }
    })
}

/// Parse a DCD trajectory file
#[pyfunction]
pub fn parse_dcd(path: String) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        #[cfg(feature = "trajectories")]
        {
            use formats::dcd::chemfiles_impl::read_dcd_chemfiles;
            let traj = read_dcd_chemfiles(&path).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("DCD parsing failed: {}", e))
            })?;

            let dict = PyDict::new_bound(py);
            dict.set_item("num_frames", traj.num_frames)?;
            dict.set_item("num_atoms", traj.num_atoms)?;

            // Convert times to NumPy
            let times = PyArray1::from_slice_bound(py, &traj.times);
            dict.set_item("times", times)?;

            // Combine all coords into (N_frames, N_atoms, 3)
            let mut flat_coords = Vec::with_capacity(traj.num_frames * traj.num_atoms * 3);
            for frame_coords in &traj.coords {
                flat_coords.extend_from_slice(frame_coords);
            }
            let coords_array = PyArray1::from_slice_bound(py, &flat_coords);
            let shape = (traj.num_frames, traj.num_atoms, 3);
            let coords_reshaped = coords_array.reshape(shape).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("Failed to reshape coords: {}", e))
            })?;

            dict.set_item("coordinates", coords_reshaped)?;

            // Unit cells if available
            if let Some(ref unit_cells) = traj.unit_cells {
                let mut flat_cells = Vec::with_capacity(traj.num_frames * 6);
                for cell in unit_cells {
                    flat_cells.extend_from_slice(cell);
                }
                let cells_array = PyArray1::from_slice_bound(py, &flat_cells);
                let cells_reshaped = cells_array.reshape((traj.num_frames, 6)).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "Failed to reshape unit_cells: {}",
                        e
                    ))
                })?;
                dict.set_item("unit_cells", cells_reshaped)?;
            }

            Ok(dict.into_py(py))
        }

        #[cfg(not(feature = "trajectories"))]
        {
            let _ = path;
            Err(pyo3::exceptions::PyImportError::new_err(
                "DCD support requires compiling with 'trajectories' feature (chemfiles).",
            ))
        }
    })
}

/// Parse a TRR trajectory file
#[pyfunction]
pub fn parse_trr(path: String) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        #[cfg(feature = "trajectories")]
        {
            use formats::trr::chemfiles_impl::read_trr_chemfiles;
            let traj = read_trr_chemfiles(&path).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("TRR parsing failed: {}", e))
            })?;

            let dict = PyDict::new_bound(py);
            dict.set_item("num_frames", traj.num_frames)?;
            dict.set_item("num_atoms", traj.num_atoms)?;

            // Convert times to NumPy
            let times = PyArray1::from_slice_bound(py, &traj.times);
            dict.set_item("times", times)?;

            // Combine all coords into (N_frames, N_atoms, 3)
            let mut flat_coords = Vec::with_capacity(traj.num_frames * traj.num_atoms * 3);
            for frame_coords in &traj.coords {
                flat_coords.extend_from_slice(frame_coords);
            }
            let coords_array = PyArray1::from_slice_bound(py, &flat_coords);
            let shape = (traj.num_frames, traj.num_atoms, 3);
            let coords_reshaped = coords_array.reshape(shape).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("Failed to reshape coords: {}", e))
            })?;

            dict.set_item("coordinates", coords_reshaped)?;

            // Velocities if available
            if let Some(ref velocities) = traj.velocities {
                let mut flat_vel = Vec::with_capacity(traj.num_frames * traj.num_atoms * 3);
                for frame_vel in velocities {
                    flat_vel.extend_from_slice(frame_vel);
                }
                let vel_array = PyArray1::from_slice_bound(py, &flat_vel);
                let vel_reshaped = vel_array.reshape(shape).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "Failed to reshape velocities: {}",
                        e
                    ))
                })?;
                dict.set_item("velocities", vel_reshaped)?;
            }

            // Box vectors if available
            if let Some(ref box_vectors) = traj.box_vectors {
                let mut flat_box = Vec::with_capacity(traj.num_frames * 9);
                for frame_box in box_vectors {
                    flat_box.extend_from_slice(&frame_box[0]);
                    flat_box.extend_from_slice(&frame_box[1]);
                    flat_box.extend_from_slice(&frame_box[2]);
                }
                let box_array = PyArray1::from_slice_bound(py, &flat_box);
                let box_reshaped = box_array.reshape((traj.num_frames, 3, 3)).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "Failed to reshape box_vectors: {}",
                        e
                    ))
                })?;
                dict.set_item("box_vectors", box_reshaped)?;
            }

            Ok(dict.into_py(py))
        }

        #[cfg(not(feature = "trajectories"))]
        {
            let _ = path;
            Err(pyo3::exceptions::PyImportError::new_err(
                "TRR support requires compiling with 'trajectories' feature (chemfiles).",
            ))
        }
    })
}
