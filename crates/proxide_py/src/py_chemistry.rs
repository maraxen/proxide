// TODO: Review allow attributes at a later point
#![allow(clippy::useless_conversion)]

use crate::{chem, physics};
use nalgebra::DMatrix;
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// Assign atomic masses based on atom names
#[pyfunction]
pub fn assign_masses(atom_names: Vec<String>) -> PyResult<Vec<f32>> {
    Ok(chem::masses::assign_masses(&atom_names))
}

/// Assign GAFF atom types to a structure (exposed to Python).
///
/// **Two structurally different implementations, selected at build time by
/// the `gaff2-engine` Cargo feature** (off by default -- see this crate's
/// Cargo.toml). This is not a stopgap; it is the deliberate outcome of the
/// `gaff2-rust-port` Cutover decision
/// (`.praxia/docs/reference/260821_gaff2-rust-port-lessons.md`): the real,
/// geostd-corpus-validated GAFF2 engine (`proxide-gaff2`, 100% Rust-vs-Python
/// match on 36,297 real ligands) is GPL-2.0-or-later (see
/// `crates/proxide-gaff2/NOTICE`), and linking it into this crate's cdylib
/// output makes the distributed `_proxider.so` wheel a GPL-encumbered
/// combined work -- not acceptable as the *default* published build. Building
/// with `--features gaff2-engine` opts into that tradeoff deliberately; the
/// default build keeps `proxide-gaff`'s coordinate-only heuristic typer
/// (MIT, unchanged behavior, but NOT a validated port of any reference
/// implementation -- see that crate's own module docs).
///
/// The two variants also have genuinely different *signatures*, not just
/// different bodies behind the same one: the heuristic path only ever had
/// coordinates + elements to work with (bond order/aromaticity/rings were
/// never available to it), while the real engine requires them -- retrofitting
/// bond-order perception onto the coordinate-only path is a separate, unsolved
/// problem (see the Cutover section's "input type cannot stay identical"
/// finding), not something this cutover attempts.
#[cfg(not(feature = "gaff2-engine"))]
#[pyfunction]
pub fn assign_gaff_atom_types(
    py: Python<'_>,
    coordinates: PyObject,
    elements: Vec<String>,
) -> PyResult<Vec<Option<String>>> {
    let coords = extract_coords(py, &coordinates)?;

    // Default tolerance for bond inference
    let topology = proxide_geometry::geometry::topology::generate_topology(&coords, &elements, 1.3);

    let gaff = proxide_gaff::gaff::GaffParameters::new();
    let types = proxide_gaff::gaff::assign_gaff_types(&elements, &topology, &gaff);

    Ok(types)
}

/// `1`/`2`/`3` -> Kekule bond order. Mirrors
/// `proxide-gaff2::py_validation::bond_order_from_u8` exactly (this function
/// supersedes that module's throwaway validation entrypoint as the real
/// production binding; that module is left in place since
/// `scripts/validation/gaff2_rust_parity.py` still builds against it
/// independently of the full `proxide_py` dependency graph).
#[cfg(feature = "gaff2-engine")]
fn bond_order_from_u8(order: u8) -> Result<proxide_gaff2::mol::BondOrder, String> {
    use proxide_gaff2::mol::BondOrder;
    match order {
        1 => Ok(BondOrder::Single),
        2 => Ok(BondOrder::Double),
        3 => Ok(BondOrder::Triple),
        other => Err(format!(
            "invalid bond order {other}: expected 1 (single), 2 (double), or 3 (triple) \
             -- pass the true Kekule bond identity, not an aromatic/1.5 order"
        )),
    }
}

/// Real, validated GAFF2 atom typer (see the module-level doc comment above
/// for why this is feature-gated). Caller contract mirrors
/// `proxide-gaff2::py_validation::assign_gaff2_atom_types_rs` exactly:
///
/// - `elements: list[str]` -- element symbols, index-aligned atom order. H
///   atoms must be explicit (`Chem.AddHs(mol)` on the caller's RDKit side
///   first).
/// - `bonds: list[tuple[int, int, int, bool]]` -- `(atom_i, atom_j,
///   bond_order, is_aromatic)`, already Kekulized (`Chem.Kekulize(mol,
///   clearAromaticFlags=False)` on the caller's side -- see
///   `proxide-gaff2::orchestrate`'s module doc for why Kekulization must
///   happen before this call, not inside it).
/// - `formal_charges` / `rings`: optional; see
///   `proxide-gaff2::mol::MolGraph::new`'s doc for the omission contract.
///
/// Returns one GAFF2 atom type per atom (including H), `"x"` (not `None`)
/// for any atom no DEF rule matched -- this is a deliberate signature change
/// from the pre-cutover `Vec<Option<String>>` contract (see the Cutover
/// section's `"x"` sentinel note: a naive `if t:`-style Python check would
/// silently treat `"x"` as truthy/matched).
#[cfg(feature = "gaff2-engine")]
#[pyfunction]
#[pyo3(signature = (elements, bonds, formal_charges=None, rings=None))]
pub fn assign_gaff_atom_types(
    elements: Vec<String>,
    bonds: Vec<(usize, usize, u8, bool)>,
    formal_charges: Option<Vec<i8>>,
    rings: Option<Vec<Vec<usize>>>,
) -> PyResult<Vec<String>> {
    use proxide_gaff2::mol::{Bond, MolGraph};

    let bonds: Vec<Bond> = bonds
        .into_iter()
        .map(|(i, j, order, aromatic)| {
            bond_order_from_u8(order).map(|order| Bond {
                i,
                j,
                order,
                aromatic,
            })
        })
        .collect::<Result<_, String>>()
        .map_err(pyo3::exceptions::PyValueError::new_err)?;

    let mol = MolGraph::new(elements, bonds, formal_charges, None, rings)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;

    proxide_gaff2::assign_gaff2_atom_types(&mol).map_err(pyo3::exceptions::PyValueError::new_err)
}

/// Assign intrinsic radii using the MBondi2 scheme
#[pyfunction]
pub fn assign_mbondi2_radii(atom_names: Vec<String>, bonds: Vec<[usize; 2]>) -> PyResult<Vec<f32>> {
    let radii = physics::gbsa::assign_mbondi2_radii(&atom_names, &bonds);
    Ok(radii)
}

/// Assign scaling factors for OBC2 GBSA calculation
#[pyfunction]
pub fn assign_obc2_scaling_factors(atom_names: Vec<String>) -> Result<Vec<f32>, PyErr> {
    let factors = physics::gbsa::assign_obc2_scaling_factors(&atom_names);
    Ok(factors)
}

/// Get water model parameters
#[pyfunction]
pub fn get_water_model(name: String, rigid: bool) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let model = physics::water::get_water_model(&name, rigid)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;

        let dict = PyDict::new_bound(py);
        dict.set_item("name", &model.name)?;
        dict.set_item("atoms", &model.atoms)?;
        dict.set_item("has_virtual_sites", model.has_virtual_sites)?;

        // Charges dict
        let charges_dict = PyDict::new_bound(py);
        for (k, v) in &model.charges {
            charges_dict.set_item(k, *v)?;
        }
        dict.set_item("charges", charges_dict)?;

        // Sigmas dict
        let sigmas_dict = PyDict::new_bound(py);
        for (k, v) in &model.sigmas {
            sigmas_dict.set_item(k, *v)?;
        }
        dict.set_item("sigmas", sigmas_dict)?;

        // Epsilons dict
        let epsilons_dict = PyDict::new_bound(py);
        for (k, v) in &model.epsilons {
            epsilons_dict.set_item(k, *v)?;
        }
        dict.set_item("epsilons", epsilons_dict)?;

        // Bonds: list of (atom1, atom2, length, k)
        let bonds: Vec<(&str, &str, f32, f32)> = model
            .bonds
            .iter()
            .map(|(a, b, l, k)| (a.as_str(), b.as_str(), *l, *k))
            .collect();
        dict.set_item("bonds", bonds)?;

        // Angles: list of (a1, a2, a3, theta, k)
        let angles: Vec<(&str, &str, &str, f32, f32)> = model
            .angles
            .iter()
            .map(|(a, b, c, t, k)| (a.as_str(), b.as_str(), c.as_str(), *t, *k))
            .collect();
        dict.set_item("angles", angles)?;

        // Constraints
        let constraints: Vec<(&str, &str, f32)> = model
            .constraints
            .iter()
            .map(|(a, b, d)| (a.as_str(), b.as_str(), *d))
            .collect();
        dict.set_item("constraints", constraints)?;

        Ok(dict.into_py(py))
    })
}

/// Compute bicubic interpolation parameters for CMAP
#[pyfunction]
pub fn compute_bicubic_params(grid: Vec<Vec<f64>>) -> Vec<Vec<[f64; 4]>> {
    physics::cmap::compute_bicubic_params(&grid)
}

/// Parameterize a molecule using GAFF for ligands and small molecules
#[pyfunction]
#[pyo3(signature = (coordinates, elements, bond_tolerance=1.3))]
pub fn parameterize_molecule(
    py: Python<'_>,
    coordinates: PyObject,
    elements: Vec<String>,
    bond_tolerance: f32,
) -> PyResult<PyObject> {
    let coords = extract_coords(py, &coordinates)?;

    let params = physics::md_params::parameterize_molecule(&coords, &elements, bond_tolerance)
        .map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Parameterization failed: {}", e))
        })?;

    let dict = PyDict::new_bound(py);

    // Basic info
    dict.set_item("num_parameterized", params.num_parameterized)?;
    dict.set_item("num_skipped", params.num_skipped)?;

    // Atom types
    let atom_types: Vec<&str> = params.atom_types.iter().map(|s| s.as_str()).collect();
    dict.set_item("atom_types", atom_types)?;

    // LJ parameters
    let charges = PyArray1::from_slice_bound(py, &params.charges);
    let sigmas = PyArray1::from_slice_bound(py, &params.sigmas);
    let epsilons = PyArray1::from_slice_bound(py, &params.epsilons);
    dict.set_item("charges", charges)?;
    dict.set_item("sigmas", sigmas)?;
    dict.set_item("epsilons", epsilons)?;

    // Bonds (N, 2)
    if !params.bonds.is_empty() {
        let mut flat = Vec::with_capacity(params.bonds.len() * 2);
        for b in &params.bonds {
            flat.extend_from_slice(b);
        }
        let arr = PyArray1::from_slice_bound(py, &flat);
        dict.set_item("bonds", arr.reshape((params.bonds.len(), 2)).unwrap())?;
    }

    // Bond params (N, 2)
    if !params.bond_params.is_empty() {
        let mut flat = Vec::with_capacity(params.bond_params.len() * 2);
        for p in &params.bond_params {
            flat.extend_from_slice(p);
        }
        let arr = PyArray1::from_slice_bound(py, &flat);
        dict.set_item(
            "bond_params",
            arr.reshape((params.bond_params.len(), 2)).unwrap(),
        )?;
    }

    // Angles (N, 3)
    if !params.angles.is_empty() {
        let mut flat = Vec::with_capacity(params.angles.len() * 3);
        for a in &params.angles {
            flat.extend_from_slice(a);
        }
        let arr = PyArray1::from_slice_bound(py, &flat);
        dict.set_item("angles", arr.reshape((params.angles.len(), 3)).unwrap())?;
    }

    // Dihedrals (N, 4)
    if !params.dihedrals.is_empty() {
        let mut flat = Vec::with_capacity(params.dihedrals.len() * 4);
        for d in &params.dihedrals {
            flat.extend_from_slice(d);
        }
        let arr = PyArray1::from_slice_bound(py, &flat);
        dict.set_item(
            "dihedrals",
            arr.reshape((params.dihedrals.len(), 4)).unwrap(),
        )?;
    }

    // Nonbonded exceptions (N, 5)
    let exceptions: Vec<(String, String, f32, f32, f32)> = params.nonbonded_exceptions;
    dict.set_item("nonbonded_exceptions", exceptions)?;

    Ok(dict.into_py(py))
}

/// Assign partial charges using the native Rust Espaloma engine.
///
/// This implementation is GIL-free and uses nalgebra for inference.
///
/// ## Citation
/// Wang Y, Pulido I, Takaba K, Kaminow B, Scheen J, Wang L, Chodera JD.
/// "EspalomaCharge: Machine Learning-Enabled Ultrafast Partial Charge Assignment"
/// J. Phys. Chem. A 2024, 128, 20, 4160-4167. DOI: 10.1021/acs.jpca.4c01287
///
/// ## Example
/// ```python
/// from proxide._proxider import assign_espaloma_charges
/// import numpy as np
///
/// features = np.random.randn(100, 116).astype(np.float32)
/// senders = np.array([0, 1, 2], dtype=np.uint32)
/// receivers = np.array([1, 2, 3], dtype=np.uint32)
/// charges = assign_espaloma_charges(features, senders, receivers, [0, 0, 0], 1, [0.0])
/// ```
#[pyfunction]
#[pyo3(signature = (features, senders, receivers, segment_ids, num_graphs, total_charges))]
pub fn assign_espaloma_charges(
    py: Python<'_>,
    features: PyObject, // (n_atoms, 116)
    senders: Vec<u32>,
    receivers: Vec<u32>,
    segment_ids: Vec<u32>,
    num_graphs: usize,
    total_charges: Vec<f32>,
) -> PyResult<PyObject> {
    let feat_array = features.bind(py).downcast::<PyArray2<f32>>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err("features must be a 2D float32 numpy array")
    })?;

    let binding = feat_array.readonly();
    let feat_view = binding.as_array();

    // Check dimensions
    if feat_view.shape()[1] != chem::inference::FEATURE_UNITS {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "features must have {} columns, got {}",
            chem::inference::FEATURE_UNITS,
            feat_view.shape()[1]
        )));
    }

    // Convert ndarray view to nalgebra DMatrix
    let n_atoms = feat_view.shape()[0];
    let mut x = DMatrix::zeros(n_atoms, chem::inference::FEATURE_UNITS);
    for i in 0..n_atoms {
        for j in 0..chem::inference::FEATURE_UNITS {
            x[(i, j)] = feat_view[[i, j]];
        }
    }

    // Load weights (lazy static or similar would be better, but for now we parse from embedded bytes)
    // In a production scenario, we'd cache this.
    let weights = chem::inference::EspalomaWeights::from_bytes(chem::inference::EMBEDDED_WEIGHTS)
        .map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to load weights: {}", e))
    })?;

    // Release GIL and run inference
    let charges = py.allow_threads(|| {
        chem::inference::infer_charges(
            &weights,
            &x,
            &senders,
            &receivers,
            &segment_ids,
            num_graphs,
            &total_charges,
        )
    });

    Ok(PyArray1::from_vec_bound(py, charges).into_py(py))
}

/// Opaque handle around a built `LigandTopology`, so
/// `extract_ligand_frame_coordinates` can consume it without round-tripping
/// through a dict (avoids the exact field-order bugs the spec's
/// index-alignment contract warns against).
#[cfg(feature = "ligand-frame")]
#[pyclass]
pub struct PyLigandTopology {
    pub(crate) inner: proxide_ligand_frame::LigandTopology,
}

#[cfg(feature = "ligand-frame")]
#[pymethods]
impl PyLigandTopology {
    #[getter]
    fn ligand_id(&self) -> String {
        self.inner.ligand_id.clone()
    }
    #[getter]
    fn canonical_order(&self) -> Vec<usize> {
        self.inner.canonical_order.clone()
    }
    #[getter]
    fn elements(&self) -> Vec<String> {
        self.inner.elements.clone()
    }
    #[getter]
    fn atom_names(&self) -> Vec<String> {
        self.inner.atom_names.clone()
    }
    #[getter]
    fn gaff2_types(&self) -> Vec<String> {
        self.inner.gaff2_types.clone()
    }
    #[getter]
    fn formal_charges(&self) -> Vec<i8> {
        self.inner.formal_charges.clone()
    }
    #[getter]
    fn partial_charges(&self, py: Python<'_>) -> PyObject {
        PyArray1::from_slice_bound(py, &self.inner.partial_charges).into_py(py)
    }
    #[getter]
    fn aromaticity(&self) -> Vec<bool> {
        self.inner.aromaticity.clone()
    }
    #[getter]
    fn ring_membership(&self) -> Vec<Vec<usize>> {
        self.inner.ring_membership.clone()
    }
    #[getter]
    fn bonds(&self) -> Vec<(usize, usize, u8, bool, bool)> {
        self.inner.bonds.clone()
    }
    #[getter]
    fn torsion_definitions(&self) -> Vec<[usize; 4]> {
        self.inner.torsion_definitions.clone()
    }
    #[getter]
    fn pucker_definitions(&self) -> Vec<(Vec<usize>, usize)> {
        self.inner
            .pucker_definitions
            .iter()
            .map(|p| (p.ring_atoms.clone(), p.ring_size))
            .collect()
    }
    #[getter]
    fn unrepresented_ring_dof(&self) -> Vec<Vec<usize>> {
        self.inner.unrepresented_ring_dof.clone()
    }
}

/// PyO3 binding for `proxide_ligand_frame::canonicalize_ligand_topology`,
/// gated the same way `assign_gaff_atom_types`'s real engine is
/// (`#[cfg(feature = "gaff2-engine")]`), plus `ligand-frame`.
#[cfg(feature = "ligand-frame")]
#[pyfunction]
#[pyo3(signature = (
    ligand_id, elements, atom_names, bonds, bond_is_aromatic, rings,
    formal_charges, ref_positions, espaloma_features, espaloma_senders,
    espaloma_receivers, espaloma_total_charge
))]
#[allow(clippy::too_many_arguments)]
pub fn canonicalize_ligand_topology(
    py: Python<'_>,
    ligand_id: String,
    elements: Vec<String>,
    atom_names: Vec<String>,
    bonds: Vec<(usize, usize, u8, bool)>,
    bond_is_aromatic: Vec<bool>,
    rings: Vec<Vec<usize>>,
    formal_charges: Option<Vec<i8>>,
    ref_positions: PyObject,
    espaloma_features: PyObject,
    espaloma_senders: Vec<u32>,
    espaloma_receivers: Vec<u32>,
    espaloma_total_charge: f32,
) -> PyResult<PyLigandTopology> {
    let bonds_in: Vec<(usize, usize, u8)> = bonds
        .iter()
        .map(|&(i, j, order, _)| (i, j, order))
        .collect();
    let ref_positions = extract_coords(py, &ref_positions)?
        .into_iter()
        .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
        .collect::<Vec<_>>();

    let feat_array = espaloma_features
        .bind(py)
        .downcast::<PyArray2<f32>>()
        .map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(
                "espaloma_features must be a 2D float32 numpy array",
            )
        })?;
    let binding = feat_array.readonly();
    let feat_view = binding.as_array();
    if feat_view.shape()[1] != chem::inference::FEATURE_UNITS {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "espaloma_features must have {} columns, got {}",
            chem::inference::FEATURE_UNITS,
            feat_view.shape()[1]
        )));
    }
    let n_atoms = feat_view.shape()[0];
    let mut features = vec![[0.0f32; chem::inference::FEATURE_UNITS]; n_atoms];
    for i in 0..n_atoms {
        for j in 0..chem::inference::FEATURE_UNITS {
            features[i][j] = feat_view[[i, j]];
        }
    }

    let inner = proxide_ligand_frame::canonicalize_ligand_topology(
        &ligand_id,
        &elements,
        &atom_names,
        &bonds_in,
        &bond_is_aromatic,
        &rings,
        formal_charges.as_deref(),
        &ref_positions,
        &features,
        &espaloma_senders,
        &espaloma_receivers,
        espaloma_total_charge,
    )
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    Ok(PyLigandTopology { inner })
}

/// PyO3 binding for `proxide_ligand_frame::extract_ligand_frame_coordinates`.
#[cfg(feature = "ligand-frame")]
#[pyfunction]
pub fn extract_ligand_frame_coordinates(
    py: Python<'_>,
    topology: &PyLigandTopology,
    positions: PyObject,
    input_elements: Vec<String>,
) -> PyResult<PyObject> {
    let pos_array = positions
        .bind(py)
        .downcast::<numpy::PyArray3<f64>>()
        .map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(
                "positions must be a (n_frames, n_atoms, 3) float64 numpy array",
            )
        })?;
    let binding = pos_array.readonly();
    let view = binding.as_array();
    let (n_frames, n_atoms) = (view.shape()[0], view.shape()[1]);
    let mut frames = Vec::with_capacity(n_frames);
    for f in 0..n_frames {
        let mut frame = vec![[0.0f64; 3]; n_atoms];
        for a in 0..n_atoms {
            frame[a] = [view[[f, a, 0]], view[[f, a, 1]], view[[f, a, 2]]];
        }
        frames.push(frame);
    }

    let coords = proxide_ligand_frame::extract_ligand_frame_coordinates(
        &topology.inner,
        &frames,
        &input_elements,
    )
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new_bound(py);
    let mut flat = Vec::with_capacity(n_frames * n_atoms * 3);
    for frame in &coords.positions {
        for p in frame {
            flat.extend_from_slice(p);
        }
    }
    let pos_out = PyArray1::from_vec_bound(py, flat)
        .reshape((n_frames, n_atoms, 3))
        .unwrap();
    dict.set_item("positions", pos_out)?;
    dict.set_item("torsions", &coords.torsions)?;
    dict.set_item("feature_mask", &coords.feature_mask)?;
    dict.set_item("frame_validity", &coords.frame_validity)?;
    dict.set_item("pucker_phase", &coords.pucker_phase)?;
    dict.set_item("bond_lengths", &coords.bond_lengths)?;
    dict.set_item("bond_angles", &coords.bond_angles)?;
    Ok(dict.into_py(py))
}

pub(crate) fn extract_coords(py: Python<'_>, obj: &PyObject) -> PyResult<Vec<[f32; 3]>> {
    let bound = obj.bind(py);

    if let Ok(l) = bound.downcast::<PyList>() {
        let mut coords = Vec::with_capacity(l.len());
        for item in l {
            let point: Vec<f32> = item.extract()?;
            if point.len() != 3 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Coordinates must be 3D points",
                ));
            }
            coords.push([point[0], point[1], point[2]]);
        }
        return Ok(coords);
    }

    if let Ok(array) = bound.downcast::<PyArray2<f32>>() {
        let binding = array.readonly();
        let data = binding.as_array();
        let shape = data.shape();
        if shape[1] != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Numpy array must be Nx3",
            ));
        }

        let mut coords = Vec::with_capacity(shape[0]);
        for i in 0..shape[0] {
            coords.push([data[[i, 0]], data[[i, 1]], data[[i, 2]]]);
        }
        return Ok(coords);
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "Expected list of lists or numpy array for coordinates",
    ))
}
