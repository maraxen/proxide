// TODO: Review allow attributes at a later point
#![allow(clippy::useless_conversion)]

#[cfg(feature = "xtc")]
use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
#[cfg(feature = "xtc")]
use pyo3::types::PyDict;

#[cfg(feature = "xtc")]
fn build_frame_indices(n_frames: usize, stride: usize) -> Vec<usize> {
    (0..n_frames).step_by(stride.max(1)).collect()
}

#[cfg(feature = "xtc")]
fn build_selection(atom_indices: &Option<Vec<usize>>) -> molly::selection::AtomSelection {
    match atom_indices {
        Some(indices) => {
            let indices: Vec<u32> = indices.iter().map(|&i| i as u32).collect();
            molly::selection::AtomSelection::from_index_list(&indices)
        }
        None => molly::selection::AtomSelection::All,
    }
}

/// Convert a decoded molly frame (nm) into Angstrom-scale flat coords + box vectors.
///
/// Mirrors `proxide_io::formats::xtc`'s private `frame_to_angstroms` helper —
/// duplicated here since that helper isn't exported, and `read_frames_parallel`
/// hands back raw (nm-scale) `molly::Frame`s with no atom-selection support.
#[cfg(feature = "xtc")]
fn frame_to_angstroms_selected(
    frame: &molly::Frame,
    atom_indices: &Option<Vec<usize>>,
) -> (Vec<f32>, [[f32; 3]; 3]) {
    let coords: Vec<f32> = match atom_indices {
        Some(indices) => indices
            .iter()
            .flat_map(|&i| {
                let base = i * 3;
                [
                    frame.positions[base] * 10.0,
                    frame.positions[base + 1] * 10.0,
                    frame.positions[base + 2] * 10.0,
                ]
            })
            .collect(),
        None => frame.positions.iter().map(|x| x * 10.0).collect(),
    };
    let mut box_ang = frame.boxvec_cols_2d();
    for row in &mut box_ang {
        for v in row {
            *v *= 10.0;
        }
    }
    (coords, box_ang)
}

#[cfg(feature = "xtc")]
fn frames_to_pydict(
    py: Python<'_>,
    num_frames: usize,
    num_atoms: usize,
    flat_coords: Vec<f32>,
    flat_boxes: Vec<f32>,
) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("num_frames", num_frames)?;
    dict.set_item("num_atoms", num_atoms)?;

    let coords_array = PyArray1::from_slice_bound(py, &flat_coords);
    let coords_reshaped = coords_array
        .reshape((num_frames, num_atoms, 3))
        .map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Failed to reshape coords: {}", e))
        })?;
    dict.set_item("coordinates", coords_reshaped)?;

    let box_array = PyArray1::from_slice_bound(py, &flat_boxes);
    let box_reshaped = box_array.reshape((num_frames, 3, 3)).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("Failed to reshape box_vectors: {}", e))
    })?;
    dict.set_item("box_vectors", box_reshaped)?;

    Ok(dict.into_py(py))
}

/// Lazily read an XTC trajectory via the seekable, offset-cached `XtcReader`
/// cursor — mdtraj `md.load(path, stride=..., atom_indices=...)`-equivalent,
/// but backed by proxide-io's on-disk `.offsets` sidecar so repeat opens of
/// the same file skip the header scan.
#[pyfunction]
#[pyo3(signature = (path, stride=1, atom_indices=None))]
pub fn read_xtc_lazy(
    py: Python<'_>,
    path: String,
    stride: usize,
    atom_indices: Option<Vec<usize>>,
) -> PyResult<PyObject> {
    #[cfg(feature = "xtc")]
    {
        use crate::formats::xtc::XtcReader;

        let (num_frames, num_atoms, flat_coords, flat_boxes) = py
            .allow_threads(|| -> Result<_, String> {
                let mut reader = XtcReader::open(&path).map_err(|e| e.to_string())?;
                let total_frames = reader.frame_count().map_err(|e| e.to_string())?;
                let indices = build_frame_indices(total_frames, stride);
                let selection = build_selection(&atom_indices);

                let mut flat_coords = Vec::new();
                let mut flat_boxes = Vec::new();
                let mut num_atoms = 0usize;
                for &index in &indices {
                    let frame = reader
                        .read_frame_at(index, &selection)
                        .map_err(|e| e.to_string())?;
                    let (c, b) = frame_to_angstroms_selected(&frame, &None);
                    num_atoms = c.len() / 3;
                    flat_coords.extend_from_slice(&c);
                    flat_boxes.extend_from_slice(&b[0]);
                    flat_boxes.extend_from_slice(&b[1]);
                    flat_boxes.extend_from_slice(&b[2]);
                }
                Ok((indices.len(), num_atoms, flat_coords, flat_boxes))
            })
            .map_err(pyo3::exceptions::PyValueError::new_err)?;

        frames_to_pydict(py, num_frames, num_atoms, flat_coords, flat_boxes)
    }

    #[cfg(not(feature = "xtc"))]
    {
        let _ = (py, path, stride, atom_indices);
        Err(pyo3::exceptions::PyImportError::new_err(
            "XTC support requires compiling with 'xtc' feature.",
        ))
    }
}

/// Read an XTC trajectory's frames concurrently via `read_frames_parallel`
/// (each worker opens its own file handle and seeks directly to its assigned
/// offset). Same stride/atom_indices calling convention as [`read_xtc_lazy`].
#[pyfunction]
#[pyo3(signature = (path, stride=1, atom_indices=None))]
pub fn read_xtc_parallel(
    py: Python<'_>,
    path: String,
    stride: usize,
    atom_indices: Option<Vec<usize>>,
) -> PyResult<PyObject> {
    #[cfg(all(feature = "xtc", feature = "parallel"))]
    {
        use crate::formats::xtc::{read_frames_parallel, XtcReader};

        let (num_frames, num_atoms, flat_coords, flat_boxes) = py
            .allow_threads(|| -> Result<_, String> {
                let total_frames = {
                    let mut reader = XtcReader::open(&path).map_err(|e| e.to_string())?;
                    reader.frame_count().map_err(|e| e.to_string())?
                };
                let indices = build_frame_indices(total_frames, stride);
                let frames = read_frames_parallel(&path, &indices).map_err(|e| e.to_string())?;

                let mut flat_coords = Vec::new();
                let mut flat_boxes = Vec::new();
                let mut num_atoms = 0usize;
                for frame in &frames {
                    let (c, b) = frame_to_angstroms_selected(frame, &atom_indices);
                    num_atoms = c.len() / 3;
                    flat_coords.extend_from_slice(&c);
                    flat_boxes.extend_from_slice(&b[0]);
                    flat_boxes.extend_from_slice(&b[1]);
                    flat_boxes.extend_from_slice(&b[2]);
                }
                Ok((frames.len(), num_atoms, flat_coords, flat_boxes))
            })
            .map_err(pyo3::exceptions::PyValueError::new_err)?;

        frames_to_pydict(py, num_frames, num_atoms, flat_coords, flat_boxes)
    }

    #[cfg(not(all(feature = "xtc", feature = "parallel")))]
    {
        let _ = (py, path, stride, atom_indices);
        Err(pyo3::exceptions::PyImportError::new_err(
            "Parallel XTC reading requires compiling with the 'xtc' and 'parallel' features.",
        ))
    }
}
