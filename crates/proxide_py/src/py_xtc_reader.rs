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

/// Validate that every requested atom index is in bounds for a trajectory with
/// `n_atoms` atoms. Called once up front so an out-of-range index surfaces as
/// a clean `PyValueError`, not a panic from unchecked indexing deep in the
/// per-frame decode loop.
#[cfg(feature = "xtc")]
fn validate_atom_indices(atom_indices: &Option<Vec<usize>>, n_atoms: usize) -> Result<(), String> {
    if let Some(indices) = atom_indices {
        if let Some(&bad) = indices.iter().find(|&&i| i >= n_atoms) {
            return Err(format!(
                "atom index {bad} out of range for trajectory with {n_atoms} atoms"
            ));
        }
    }
    Ok(())
}

/// Convert a decoded molly frame (nm) into Angstrom-scale flat coords + box
/// vectors, selecting/reordering atoms exactly as `atom_indices` specifies —
/// order and duplicates preserved, matching mdtraj's `atom_indices` fancy-
/// indexing semantics.
///
/// Deliberately does *not* use `molly::selection::AtomSelection::from_index_list`
/// for this: that builds a boolean `Mask`, which silently returns atoms in
/// ascending index order with duplicates collapsed — diverging from mdtraj
/// (and from a caller's expectations) whenever `atom_indices` isn't already
/// sorted and unique. Mirrors `proxide_io::formats::xtc`'s private
/// `frame_to_angstroms` helper for the no-selection case (that helper isn't
/// exported).
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
#[allow(clippy::too_many_arguments)]
fn frames_to_pydict(
    py: Python<'_>,
    num_frames: usize,
    num_atoms: usize,
    flat_coords: Vec<f32>,
    flat_boxes: Vec<f32>,
    flat_times: Vec<f32>,
    total_frames_on_disk: usize,
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

    let times_array = PyArray1::from_slice_bound(py, &flat_times);
    dict.set_item("times", times_array)?;

    dict.set_item("total_frames_on_disk", total_frames_on_disk)?;

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
        use molly::selection::AtomSelection;

        let (num_frames, num_atoms, flat_coords, flat_boxes, flat_times, total_frames) = py
            .allow_threads(|| -> Result<_, String> {
                let mut reader = XtcReader::open(&path).map_err(|e| e.to_string())?;
                let total_frames = reader.frame_count().map_err(|e| e.to_string())?;
                let n_atoms = reader.n_atoms().map_err(|e| e.to_string())?;
                validate_atom_indices(&atom_indices, n_atoms)?;

                let indices = build_frame_indices(total_frames, stride);
                let selected_natoms = atom_indices.as_ref().map_or(n_atoms, Vec::len);

                let mut flat_coords = Vec::with_capacity(indices.len() * selected_natoms * 3);
                let mut flat_boxes = Vec::with_capacity(indices.len() * 9);
                let mut flat_times = Vec::with_capacity(indices.len());
                let mut num_atoms = 0usize;
                for &index in &indices {
                    let frame = reader
                        .read_frame_at(index, &AtomSelection::All)
                        .map_err(|e| e.to_string())?;
                    let (c, b) = frame_to_angstroms_selected(&frame, &atom_indices);
                    num_atoms = c.len() / 3;
                    flat_coords.extend_from_slice(&c);
                    flat_boxes.extend_from_slice(&b[0]);
                    flat_boxes.extend_from_slice(&b[1]);
                    flat_boxes.extend_from_slice(&b[2]);
                    flat_times.push(frame.time);
                }
                Ok((
                    indices.len(),
                    num_atoms,
                    flat_coords,
                    flat_boxes,
                    flat_times,
                    total_frames,
                ))
            })
            .map_err(pyo3::exceptions::PyValueError::new_err)?;

        frames_to_pydict(
            py,
            num_frames,
            num_atoms,
            flat_coords,
            flat_boxes,
            flat_times,
            total_frames,
        )
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
/// offset). Same stride/atom_indices calling convention as [`read_xtc_lazy`]
/// — order and duplicates in `atom_indices` are preserved exactly, matching
/// mdtraj's `atom_indices` semantics.
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

        let (num_frames, num_atoms, flat_coords, flat_boxes, flat_times, total_frames) = py
            .allow_threads(|| -> Result<_, String> {
                let (total_frames, n_atoms) = {
                    let mut reader = XtcReader::open(&path).map_err(|e| e.to_string())?;
                    let total_frames = reader.frame_count().map_err(|e| e.to_string())?;
                    let n_atoms = reader.n_atoms().map_err(|e| e.to_string())?;
                    (total_frames, n_atoms)
                };
                validate_atom_indices(&atom_indices, n_atoms)?;

                let indices = build_frame_indices(total_frames, stride);
                let selected_natoms = atom_indices.as_ref().map_or(n_atoms, Vec::len);
                let frames = read_frames_parallel(&path, &indices).map_err(|e| e.to_string())?;

                let mut flat_coords = Vec::with_capacity(frames.len() * selected_natoms * 3);
                let mut flat_boxes = Vec::with_capacity(frames.len() * 9);
                let mut flat_times = Vec::with_capacity(frames.len());
                let mut num_atoms = 0usize;
                for frame in &frames {
                    let (c, b) = frame_to_angstroms_selected(frame, &atom_indices);
                    num_atoms = c.len() / 3;
                    flat_coords.extend_from_slice(&c);
                    flat_boxes.extend_from_slice(&b[0]);
                    flat_boxes.extend_from_slice(&b[1]);
                    flat_boxes.extend_from_slice(&b[2]);
                    flat_times.push(frame.time);
                }
                Ok((
                    frames.len(),
                    num_atoms,
                    flat_coords,
                    flat_boxes,
                    flat_times,
                    total_frames,
                ))
            })
            .map_err(pyo3::exceptions::PyValueError::new_err)?;

        frames_to_pydict(
            py,
            num_frames,
            num_atoms,
            flat_coords,
            flat_boxes,
            flat_times,
            total_frames,
        )
    }

    #[cfg(not(all(feature = "xtc", feature = "parallel")))]
    {
        let _ = (py, path, stride, atom_indices);
        Err(pyo3::exceptions::PyImportError::new_err(
            "Parallel XTC reading requires compiling with the 'xtc' and 'parallel' features.",
        ))
    }
}

/// Number of frames in an XTC trajectory, from a header-only scan — never
/// decodes coordinate data.
///
/// Backed by `XtcReader`'s on-disk `.offsets` sidecar cache, so repeat calls
/// on an unchanged file are effectively free after the first scan. Safe to
/// call on a trajectory a live simulation is still appending to: the last
/// frame is excluded from the count unless it fully decodes (see
/// `XtcReader::drop_trailing_frame_if_truncated`), rather than counting a
/// partially-flushed in-flight frame as complete.
#[pyfunction]
pub fn frame_count(py: Python<'_>, path: String) -> PyResult<usize> {
    #[cfg(feature = "xtc")]
    {
        use crate::formats::xtc::XtcReader;

        py.allow_threads(|| -> Result<usize, String> {
            let mut reader = XtcReader::open(&path).map_err(|e| e.to_string())?;
            reader.frame_count().map_err(|e| e.to_string())
        })
        .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    #[cfg(not(feature = "xtc"))]
    {
        let _ = (py, path);
        Err(pyo3::exceptions::PyImportError::new_err(
            "XTC support requires compiling with 'xtc' feature.",
        ))
    }
}

/// Number of atoms per frame in an XTC trajectory, from the first frame's
/// header — never decodes coordinate data.
#[pyfunction]
pub fn n_atoms(py: Python<'_>, path: String) -> PyResult<usize> {
    #[cfg(feature = "xtc")]
    {
        use crate::formats::xtc::XtcReader;

        py.allow_threads(|| -> Result<usize, String> {
            let mut reader = XtcReader::open(&path).map_err(|e| e.to_string())?;
            reader.n_atoms().map_err(|e| e.to_string())
        })
        .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    #[cfg(not(feature = "xtc"))]
    {
        let _ = (py, path);
        Err(pyo3::exceptions::PyImportError::new_err(
            "XTC support requires compiling with 'xtc' feature.",
        ))
    }
}
