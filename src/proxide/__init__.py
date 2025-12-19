"""Proxide: Protein I/O and processing utilities for JAX.

This package provides high-performance protein structure parsing and processing,
backed by a Rust extension for optimal performance.
"""

# Re-export Rust extension functions for unified API
from proxide._oxidize import (
  # Structure parsing
  parse_pdb,
  parse_mmcif,
  parse_pqr,
  parse_structure,
  # Force fields
  load_forcefield,
  # Trajectory parsing
  parse_xtc,
  parse_dcd,
  parse_trr,
  # HDF5/MDCATH
  parse_mdtraj_h5_metadata,
  parse_mdtraj_h5_frame,
  parse_mdcath_metadata,
  get_mdcath_replicas,
  parse_mdcath_frame,
  # Chemistry
  assign_masses,
  assign_gaff_atom_types,
  assign_mbondi2_radii,
  assign_obc2_scaling_factors,
  get_water_model,
  compute_bicubic_params,
  parameterize_molecule,
  # Types/Classes
  OutputSpec,
  CoordFormat,
  ErrorMode,
  MissingResidueMode,
  HydrogenSource,
  AtomicSystem,
)

__all__ = [
  # Structure parsing
  "parse_pdb",
  "parse_mmcif",
  "parse_pqr",
  "parse_structure",
  # Force fields
  "load_forcefield",
  # Trajectory
  "parse_xtc",
  "parse_dcd",
  "parse_trr",
  # HDF5
  "parse_mdtraj_h5_metadata",
  "parse_mdtraj_h5_frame",
  "parse_mdcath_metadata",
  "get_mdcath_replicas",
  "parse_mdcath_frame",
  # Chemistry
  "assign_masses",
  "assign_gaff_atom_types",
  "assign_mbondi2_radii",
  "assign_obc2_scaling_factors",
  "get_water_model",
  "compute_bicubic_params",
  "parameterize_molecule",
  # Types
  "OutputSpec",
  "CoordFormat",
  "ErrorMode",
  "MissingResidueMode",
  "HydrogenSource",
  "AtomicSystem",
]
