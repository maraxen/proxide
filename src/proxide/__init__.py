"""Proxide: Protein I/O and processing utilities for JAX.

Proxide is a high-performance library that combines a Python/JAX frontend with a
Rust backend (`_proxider`) for fast protein I/O, force field parameterization,
and seamless integration with JAX MD.
"""

__version__ = "0.1.0"

# Re-export Rust extension functions for unified API
from proxide._proxider import (  # type: ignore[unresolved-import]
  AtomicSystem,
  CoordFormat,
  ErrorMode,
  HydrogenSource,
  MissingResidueMode,
  # Types/Classes
  OutputSpec,
  assign_gaff_atom_types,
  # Chemistry
  assign_masses,
  assign_mbondi2_radii,
  assign_obc2_scaling_factors,
  compute_bicubic_params,
  get_mdcath_replicas,
  get_water_model,
  # Force fields
  load_forcefield,
  parameterize_molecule,
  parse_dcd,
  parse_mdcath_frame,
  parse_mdcath_metadata,
  parse_mdtraj_h5_frame,
  # HDF5/MDCATH
  parse_mdtraj_h5_metadata,
  parse_mmcif,
  # Structure parsing
  parse_pdb,
  parse_pqr,
  parse_trr,
  # Trajectory parsing
  parse_xtc,
)
from proxide.io.fetching import (
  fetch_afdb,
  fetch_foldcomp_database,
  fetch_rcsb,
)
from proxide.io.parsing.backend import (
  TrajectoryStream,
  iterload,
  parse_structure,
  write_dcd,
)

# Expose subpackages so `import proxide; proxide.io.*` / `proxide.jaccard.*` work.
from proxide import io, jaccard  # noqa: E402

__all__ = [
  "io",
  "jaccard",
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
  "TrajectoryStream",
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
  "iterload",
  "write_dcd",
  # Fetching
  "fetch_rcsb",
  "fetch_afdb",
  "fetch_foldcomp_database",
]
