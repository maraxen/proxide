//! PrioX Rust Extension
//!
//! High-performance protein structure parsing library for Python.
//! Provides zero-copy parsing for PDB, mmCIF, and trajectory formats.

use pyo3::prelude::*;

// Internal modules
mod chem;
mod forcefield;
mod formats;
mod formatters;
mod geometry;
mod io;
mod physics;
mod processing;
mod spec;
mod structure;

// Python binding modules - re-exported for the pymodule
mod py_chemistry;
mod py_forcefield;
mod py_hdf5;
mod py_parsers;
mod py_trajectory;

// Re-export spec types for Python bindings
use spec::{CoordFormat, OutputSpec};
use structure::systems::AtomicSystem;

/// Python module
#[pymodule]
fn _oxidize(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();

    // Initialize fragment library eagerly to avoid GIL deadlock
    geometry::hydrogens::init_fragment_library();

    // Structure parsing functions (from py_parsers)
    m.add_function(wrap_pyfunction!(py_parsers::parse_pdb, m)?)?;
    m.add_function(wrap_pyfunction!(py_parsers::parse_mmcif, m)?)?;
    m.add_function(wrap_pyfunction!(py_parsers::parse_pqr, m)?)?;
    m.add_function(wrap_pyfunction!(py_parsers::parse_foldcomp, m)?)?;
    m.add_function(wrap_pyfunction!(py_parsers::parse_structure, m)?)?;
    m.add_function(wrap_pyfunction!(py_parsers::project_to_mpnn_batch, m)?)?;

    // Force field functions (from py_forcefield)
    m.add_function(wrap_pyfunction!(py_forcefield::load_forcefield, m)?)?;

    // Trajectory parsing functions (from py_trajectory)
    m.add_function(wrap_pyfunction!(py_trajectory::parse_xtc, m)?)?;
    m.add_function(wrap_pyfunction!(py_trajectory::parse_dcd, m)?)?;
    m.add_function(wrap_pyfunction!(py_trajectory::parse_trr, m)?)?;

    // HDF5 parsing functions (from py_hdf5)
    m.add_function(wrap_pyfunction!(py_hdf5::parse_mdtraj_h5_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(py_hdf5::parse_mdtraj_h5_frame, m)?)?;
    m.add_function(wrap_pyfunction!(py_hdf5::parse_mdcath_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(py_hdf5::get_mdcath_replicas, m)?)?;
    m.add_function(wrap_pyfunction!(py_hdf5::parse_mdcath_frame, m)?)?;

    // Chemistry utilities (from py_chemistry)
    m.add_function(wrap_pyfunction!(py_chemistry::assign_masses, m)?)?;
    m.add_function(wrap_pyfunction!(py_chemistry::assign_gaff_atom_types, m)?)?;
    m.add_function(wrap_pyfunction!(py_chemistry::assign_mbondi2_radii, m)?)?;
    m.add_function(wrap_pyfunction!(
        py_chemistry::assign_obc2_scaling_factors,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(py_chemistry::get_water_model, m)?)?;
    m.add_function(wrap_pyfunction!(py_chemistry::compute_bicubic_params, m)?)?;
    m.add_function(wrap_pyfunction!(py_chemistry::parameterize_molecule, m)?)?;

    // Python classes
    m.add_class::<OutputSpec>()?;
    m.add_class::<CoordFormat>()?;
    m.add_class::<spec::ErrorMode>()?;
    m.add_class::<spec::MissingResidueMode>()?;
    m.add_class::<spec::HydrogenSource>()?;

    // Atomic System Architecture
    m.add_class::<AtomicSystem>()?;
    m.add_class::<py_parsers::FoldCompDatabase>()?;

    // Fetching functions (from io::fetching)
    m.add_function(wrap_pyfunction!(io::fetching::fetch_rcsb, m)?)?;
    m.add_function(wrap_pyfunction!(io::fetching::fetch_md_cath, m)?)?;
    m.add_function(wrap_pyfunction!(io::fetching::fetch_afdb, m)?)?;
    m.add_function(wrap_pyfunction!(io::fetching::fetch_foldcomp_database, m)?)?;

    Ok(())
}
