"""Pins proxide's top-level public surface.

A name added to or removed from ``proxide.__all__`` must be a deliberate change here too.
"""

from __future__ import annotations

import proxide

EXPECTED_ALL = {
  "io",
  "jaccard",
  "tmalign",
  "parse_pdb",
  "parse_mmcif",
  "parse_pqr",
  "parse_structure",
  "parse_parm7",
  "parse_amber_trajectory",
  "load_forcefield",
  "parse_xtc",
  "parse_dcd",
  "parse_trr",
  "read_xtc_lazy",
  "read_xtc_parallel",
  "read_xtc_ca_distogram",
  "frame_count",
  "n_atoms",
  "TrajectoryStream",
  "radius_of_gyration",
  "weighted_radius_of_gyration",
  "kabsch_rmsd",
  "kabsch_rmsd_batch",
  "parse_mdtraj_h5_metadata",
  "parse_mdtraj_h5_frame",
  "parse_mdcath_metadata",
  "get_mdcath_replicas",
  "parse_mdcath_frame",
  "assign_masses",
  "assign_gaff_atom_types",
  "assign_mbondi2_radii",
  "assign_obc2_scaling_factors",
  "assign_mbondi2_radii_with_provenance",
  "assign_obc2_scaling_factors_with_provenance",
  "mbondi2_table_info",
  "get_water_model",
  "compute_bicubic_params",
  "parameterize_molecule",
  "OutputSpec",
  "CoordFormat",
  "ErrorMode",
  "MissingResidueMode",
  "HydrogenSource",
  "AtomicSystem",
  "iterload",
  "write_dcd",
  "fetch_rcsb",
  "fetch_afdb",
  "fetch_foldcomp_database",
}

EXPECTED_EXPERIMENTAL = {"assign_gaff_atom_types"}


def test_all_is_pinned() -> None:
  assert len(proxide.__all__) == len(set(proxide.__all__)), "duplicate names in __all__"
  assert set(proxide.__all__) == EXPECTED_ALL


def test_experimental_is_pinned_and_within_all() -> None:
  experimental = set(proxide.__experimental__)
  assert len(proxide.__experimental__) == len(experimental), "duplicate names in __experimental__"
  assert experimental <= set(proxide.__all__)
  assert experimental == EXPECTED_EXPERIMENTAL


def test_every_all_name_resolves() -> None:
  missing = [name for name in proxide.__all__ if not hasattr(proxide, name)]
  assert missing == []
  for name in proxide.__all__:
    getattr(proxide, name)
