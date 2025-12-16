//! Proxide Rust Extension
//!
//! High-performance protein structure parsing library for Python.
//! Provides zero-copy parsing for PDB, mmCIF, and trajectory formats.

use pyo3::prelude::*;

mod chem;
mod forcefield;
mod formats;
mod formatters;
mod geometry;
mod physics;
mod processing;
mod spec;
mod structure;

mod py_chemistry;
mod py_forcefield;
mod py_hdf5;
mod py_parsers;
mod py_trajectory;

use spec::{CoordFormat, OutputSpec};
use structure::systems::AtomicSystem;

/// Python module
#[pymodule]
fn oxidize(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();
    
    // Parsers
    m.add_function(wrap_pyfunction!(py_parsers::parse_pdb, m)?)?;
    m.add_function(wrap_pyfunction!(py_parsers::parse_mmcif, m)?)?;
    m.add_function(wrap_pyfunction!(py_parsers::parse_pqr, m)?)?;
    m.add_function(wrap_pyfunction!(py_parsers::parse_structure, m)?)?;
    
    // Forcefield
    m.add_function(wrap_pyfunction!(py_forcefield::load_forcefield, m)?)?;
    
    // Chemistry
    m.add_function(wrap_pyfunction!(py_chemistry::assign_gaff_atom_types, m)?)?;
    m.add_function(wrap_pyfunction!(py_chemistry::assign_masses, m)?)?;
    m.add_function(wrap_pyfunction!(py_chemistry::assign_mbondi2_radii, m)?)?;
    m.add_function(wrap_pyfunction!(py_chemistry::assign_obc2_scaling_factors, m)?)?;
    m.add_function(wrap_pyfunction!(py_chemistry::get_water_model, m)?)?;
    m.add_function(wrap_pyfunction!(py_chemistry::compute_bicubic_params, m)?)?;
    m.add_function(wrap_pyfunction!(py_chemistry::parameterize_molecule, m)?)?;

    // Trajectory
    m.add_function(wrap_pyfunction!(py_trajectory::parse_xtc, m)?)?;
    m.add_function(wrap_pyfunction!(py_trajectory::parse_dcd, m)?)?;
    m.add_function(wrap_pyfunction!(py_trajectory::parse_trr, m)?)?;

    // HDF5 parsing functions
    m.add_function(wrap_pyfunction!(py_hdf5::parse_mdtraj_h5_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(py_hdf5::parse_mdtraj_h5_frame, m)?)?;
    m.add_function(wrap_pyfunction!(py_hdf5::parse_mdcath_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(py_hdf5::get_mdcath_replicas, m)?)?;
    m.add_function(wrap_pyfunction!(py_hdf5::parse_mdcath_frame, m)?)?;

    m.add_class::<OutputSpec>()?;
    m.add_class::<CoordFormat>()?;
    m.add_class::<spec::ErrorMode>()?;
    m.add_class::<spec::MissingResidueMode>()?;

    // Atomic System Architecture
    m.add_class::<AtomicSystem>()?;

    Ok(())
}