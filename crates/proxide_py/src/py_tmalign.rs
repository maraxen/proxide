//! Python bindings for `proxide-tmalign`: pairwise TM-align structural
//! alignment between two Cα coordinate sets.

use nalgebra::Vector3;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use proxide_tmalign::parallel::tm_scores_fixed_correspondence as tm_scores_fixed_rs;
use proxide_tmalign::pipeline::tmalign_pair_serial;

fn to_vector3_coords(arr: &PyReadonlyArray2<'_, f32>) -> PyResult<Vec<Vector3<f32>>> {
    let view = arr.as_array();
    if view.shape()[1] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "expected an (N, 3) coordinate array, got shape {:?}",
            view.shape()
        )));
    }
    if let Some(bad) = view.iter().find(|v| !v.is_finite()) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "coordinate array contains a non-finite value ({bad}); NaN/Inf coordinates would \
             silently corrupt the alignment instead of erroring"
        )));
    }
    Ok(view
        .rows()
        .into_iter()
        .map(|r| Vector3::new(r[0], r[1], r[2]))
        .collect())
}

/// Compute the TM-align structural alignment between two Cα traces.
///
/// `coords1`/`coords2` are `(N, 3)` float32 arrays of Cα positions (Å).
///
/// Only structure is used — this binding does not take sequence arguments,
/// unlike the original phase-plan sketch in
/// `.praxia/docs/specs/260729_proxide-tmalign-phases-2-5.md`'s Phase 4
/// section: the underlying `tmalign_pair_serial` never consumes sequence
/// data (TM-align's core alignment is purely Cα-geometric), so `seq1`/`seq2`
/// parameters would be silently ignored dead weight. Likewise, the returned
/// dict omits `rmsd`/`seq_id1`/`seq_id2`/`seq_id_ali` from that same
/// sketch — `proxide_tmalign::pipeline::TmAlignResult` (as landed in
/// Phase 2) never computed those fields; only what the Rust struct actually
/// exposes is surfaced here (`rotation`, `translation`, `tm_score_norm1`,
/// `tm_score_norm2`, `n_aligned`).
///
/// Returns a dict: `{"rotation": (3, 3) f32, "translation": (3,) f32,
/// "tm_score_norm1": float, "tm_score_norm2": float, "n_aligned": int}`.
#[pyfunction]
pub fn tm_align(
    py: Python<'_>,
    coords1: PyReadonlyArray2<'_, f32>,
    coords2: PyReadonlyArray2<'_, f32>,
) -> PyResult<PyObject> {
    let c1 = to_vector3_coords(&coords1)?;
    let c2 = to_vector3_coords(&coords2)?;

    let result = py
        .allow_threads(|| tmalign_pair_serial(&c1, &c2))
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("tm_align failed: {e}")))?;

    let dict = PyDict::new_bound(py);
    let rot_flat: Vec<f32> = result.rotation.iter().flatten().copied().collect();
    let rotation = PyArray1::from_slice_bound(py, &rot_flat).reshape((3, 3))?;
    let translation = PyArray1::from_slice_bound(py, &result.translation);
    dict.set_item("rotation", rotation)?;
    dict.set_item("translation", translation)?;
    dict.set_item("tm_score_norm1", result.tm_score_norm1)?;
    dict.set_item("tm_score_norm2", result.tm_score_norm2)?;
    dict.set_item("n_aligned", result.n_aligned)?;
    Ok(dict.into_py(py))
}

/// TM-scores of one query conformation against many same-topology candidates.
///
/// `query` is `(n_res, 3)` float32 Cα positions (Å); `candidates` is
/// `(n_frames, n_res, 3)`. Returns `(n_frames,)` float32 TM-scores, each
/// normalized by `n_res`, so a candidate identical to the query scores `1.0`.
///
/// The residue correspondence is taken to be the **identity** — residue `i` of
/// the query corresponds to residue `i` of every candidate. That is what makes
/// this O(n_frames) rather than O(n_frames) full alignment searches: for frames
/// of one MD trajectory the correspondence is already known, and rediscovering
/// it costs far more than the scoring it feeds. Use [`tm_align`] instead when
/// the correspondence is genuinely unknown (different constructs, different
/// lengths, unknown residue ordering).
///
/// Every candidate must have exactly `n_res` residues; a mismatch raises
/// `ValueError` rather than being padded or truncated around, since silently
/// comparing residue `i` to a different residue would still return a plausible
/// number.
///
/// Non-finite coordinates are rejected, as in [`tm_align`]: callers holding a
/// partially-observed structure must select the shared residues explicitly
/// rather than relying on NaN to propagate.
#[pyfunction]
pub fn tm_scores_fixed_correspondence(
    py: Python<'_>,
    query: PyReadonlyArray2<'_, f32>,
    candidates: PyReadonlyArray3<'_, f32>,
) -> PyResult<Py<PyArray1<f32>>> {
    let q = to_vector3_coords(&query)?;

    let cview = candidates.as_array();
    let shape = cview.shape();
    if shape[2] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "expected an (n_frames, n_res, 3) candidate array, got shape {shape:?}"
        )));
    }
    if let Some(bad) = cview.iter().find(|v| !v.is_finite()) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "candidate array contains a non-finite value ({bad}); NaN/Inf coordinates \
             would silently corrupt the score instead of erroring"
        )));
    }

    let cands: Vec<Vec<Vector3<f32>>> = (0..shape[0])
        .map(|f| {
            (0..shape[1])
                .map(|r| Vector3::new(cview[[f, r, 0]], cview[[f, r, 1]], cview[[f, r, 2]]))
                .collect()
        })
        .collect();

    let scores = py
        .allow_threads(|| tm_scores_fixed_rs(&q, &cands))
        .map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "tm_scores_fixed_correspondence failed: {e}"
            ))
        })?;

    Ok(PyArray1::from_vec_bound(py, scores).unbind())
}
