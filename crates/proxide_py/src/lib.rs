#![allow(clippy::useless_conversion)]
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

// Import the wrapper modules
mod bindings {
    pub mod atomic_system;
    pub mod conversion;
    pub mod spec;
}

mod py_chemistry;
mod py_confind;
mod py_frag;
mod py_forcefield;
mod py_hdf5;
mod py_parsers;
mod py_rotlib;
mod py_stream;
mod py_trajectory;

// Internal re-exports of core modules
pub(crate) use proxide_rs::{
    chem, forcefield, formats, formatters, geometry, physics, processing, spec,
};

// Import wrapper types
use bindings::atomic_system::PyAtomicSystem;
use bindings::spec::{
    PyCoordFormat, PyErrorMode, PyHydrogenSource, PyMissingResidueMode, PyOutputSpec, PyUnitSystem,
};

/// Wrapper for proxide_rs::io::fetching::fetch_rcsb
#[cfg(feature = "fetching")]
#[pyfunction]
#[pyo3(signature = (id, output_dir = ".cache", format_type = "cif"))]
fn fetch_rcsb(id: &str, output_dir: &str, format_type: &str) -> PyResult<String> {
    proxide_rs::io::fetching::fetch_rcsb(id, output_dir, format_type)
        .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)
}

/// Wrapper for proxide_rs::io::fetching::fetch_md_cath
#[cfg(feature = "fetching")]
#[pyfunction]
#[pyo3(signature = (id, output_dir = ".cache"))]
fn fetch_md_cath(id: &str, output_dir: &str) -> PyResult<String> {
    proxide_rs::io::fetching::fetch_md_cath(id, output_dir)
        .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)
}

/// Wrapper for proxide_rs::io::fetching::fetch_afdb
#[cfg(feature = "fetching")]
#[pyfunction]
#[pyo3(signature = (id, output_dir = ".cache", version = 4))]
fn fetch_afdb(id: &str, output_dir: &str, version: u32) -> PyResult<String> {
    proxide_rs::io::fetching::fetch_afdb(id, output_dir, version)
        .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)
}

/// Wrapper for proxide_rs::io::fetching::fetch_foldcomp_database
#[cfg(feature = "fetching")]
#[pyfunction]
#[pyo3(signature = (db_name, output_dir = ".cache", download_chunks = 1))]
fn fetch_foldcomp_database(
    db_name: &str,
    output_dir: &str,
    download_chunks: usize,
) -> PyResult<String> {
    proxide_rs::io::fetching::fetch_foldcomp_database(db_name, output_dir, download_chunks)
        .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)
}

/// Python module
#[pymodule]
fn _proxider(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();

    // Initialize fragment library eagerly to avoid GIL deadlock
    proxide_rs::geometry::hydrogens::init_fragment_library();

    // Structure parsing functions (from py_parsers)
    m.add_function(wrap_pyfunction!(py_parsers::parse_pdb, m)?)?;
    m.add_function(wrap_pyfunction!(py_parsers::parse_mmcif, m)?)?;
    m.add_function(wrap_pyfunction!(py_parsers::parse_pqr, m)?)?;
    #[cfg(feature = "foldcomp")]
    m.add_function(wrap_pyfunction!(py_parsers::parse_foldcomp, m)?)?;
    m.add_function(wrap_pyfunction!(py_parsers::parse_structure, m)?)?;
    m.add_function(wrap_pyfunction!(py_parsers::project_to_mpnn_batch, m)?)?;

    // Force field functions (from py_forcefield)
    m.add_function(wrap_pyfunction!(py_forcefield::load_forcefield, m)?)?;

    // Trajectory parsing functions (from py_trajectory)
    #[cfg(feature = "xtc")]
    m.add_function(wrap_pyfunction!(py_trajectory::parse_xtc, m)?)?;
    m.add_function(wrap_pyfunction!(py_trajectory::parse_dcd, m)?)?;
    m.add_function(wrap_pyfunction!(py_trajectory::parse_trr, m)?)?;
    m.add_function(wrap_pyfunction!(py_trajectory::parse_mdc, m)?)?;

    // HDF5 parsing functions (from py_hdf5)
    #[cfg(feature = "hdf5")]
    {
        m.add_function(wrap_pyfunction!(py_hdf5::parse_mdtraj_h5_metadata, m)?)?;
        m.add_function(wrap_pyfunction!(py_hdf5::parse_mdtraj_h5_frame, m)?)?;
        m.add_function(wrap_pyfunction!(py_hdf5::parse_mdcath_metadata, m)?)?;
        m.add_function(wrap_pyfunction!(py_hdf5::get_mdcath_replicas, m)?)?;
        m.add_function(wrap_pyfunction!(py_hdf5::parse_mdcath_frame, m)?)?;
    }

    // Chemistry utilities (from py_chemistry)
    m.add_function(wrap_pyfunction!(py_chemistry::assign_masses, m)?)?;
    m.add_function(wrap_pyfunction!(py_chemistry::assign_espaloma_charges, m)?)?;
    m.add_function(wrap_pyfunction!(py_chemistry::assign_gaff_atom_types, m)?)?;
    m.add_function(wrap_pyfunction!(py_chemistry::assign_mbondi2_radii, m)?)?;
    m.add_function(wrap_pyfunction!(
        py_chemistry::assign_obc2_scaling_factors,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(py_chemistry::get_water_model, m)?)?;
    m.add_function(wrap_pyfunction!(py_chemistry::compute_bicubic_params, m)?)?;
    m.add_function(wrap_pyfunction!(py_chemistry::parameterize_molecule, m)?)?;

    // Register Python wrappers
    m.add_class::<PyAtomicSystem>()?;
    m.add_class::<PyOutputSpec>()?;
    m.add_class::<PyCoordFormat>()?;
    m.add_class::<PyErrorMode>()?;
    m.add_class::<PyMissingResidueMode>()?;
    m.add_class::<PyHydrogenSource>()?;
    m.add_class::<PyUnitSystem>()?;

    #[cfg(feature = "foldcomp")]
    m.add_class::<py_parsers::FoldCompDatabase>()?;

    // Fetching functions (using wrappers)
    #[cfg(feature = "fetching")]
    {
        m.add_function(wrap_pyfunction!(fetch_rcsb, m)?)?;
        m.add_function(wrap_pyfunction!(fetch_md_cath, m)?)?;
        m.add_function(wrap_pyfunction!(fetch_afdb, m)?)?;
        m.add_function(wrap_pyfunction!(fetch_foldcomp_database, m)?)?;
    }

    // Streaming support
    m.add_class::<py_stream::PyTrajectoryIterator>()?;
    m.add_class::<py_stream::PyDcdWriter>()?;

    // ConFind, RotLib, and fragment search stubs
    m.add_function(wrap_pyfunction!(py_confind::run_confind, m)?)?;
    m.add_class::<py_rotlib::PyRotamerLibrary>()?;
    m.add_function(wrap_pyfunction!(py_frag::search_fragments, m)?)?;

    Ok(())
}
