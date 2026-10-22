"""Pins proxide's top-level public surface.

Only core IO is stable at 0.1.0. Every other name in ``proxide.__all__`` is experimental.
A name moving between the two sets, or added to or removed from ``__all__``, must be a
deliberate change here too.
"""

from __future__ import annotations

import proxide

STABLE = {
  "io",
  "parse_pdb",
  "parse_mmcif",
  "parse_pqr",
  "parse_structure",
  "parse_parm7",
  "parse_amber_trajectory",
  "parse_xtc",
  "parse_dcd",
  "parse_trr",
  "read_xtc_lazy",
  "read_xtc_parallel",
  "read_xtc_ca_distogram",
  "frame_count",
  "n_atoms",
  "TrajectoryStream",
  "parse_mdtraj_h5_metadata",
  "parse_mdtraj_h5_frame",
  "parse_mdcath_metadata",
  "parse_mdcath_frame",
  "get_mdcath_replicas",
  "write_dcd",
  "fetch_rcsb",
  "fetch_afdb",
  "fetch_foldcomp_database",
  "OutputSpec",
  "CoordFormat",
  "ErrorMode",
  "MissingResidueMode",
  "HydrogenSource",
}

EXPERIMENTAL = {
  "load_forcefield",
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
  "radius_of_gyration",
  "weighted_radius_of_gyration",
  "kabsch_rmsd",
  "kabsch_rmsd_batch",
  "jaccard",
  "tmalign",
  "AtomicSystem",
  "iterload",
}


def test_all_partitions_into_stable_and_experimental() -> None:
  assert len(proxide.__all__) == len(set(proxide.__all__)), "duplicate names in __all__"
  assert STABLE.isdisjoint(EXPERIMENTAL)
  assert STABLE | EXPERIMENTAL == set(proxide.__all__)


def test_experimental_tuple_is_all_minus_stable() -> None:
  assert len(proxide.__experimental__) == len(set(proxide.__experimental__))
  assert set(proxide.__experimental__) == EXPERIMENTAL
  assert set(proxide.__experimental__) == set(proxide.__all__) - STABLE


def test_experimental_names_are_documented_as_experimental() -> None:
  undocumented = [
    name for name in proxide.__experimental__ if "Experimental" not in (getattr(proxide, name).__doc__ or "")
  ]
  assert undocumented == []


def test_every_all_name_resolves() -> None:
  missing = [name for name in proxide.__all__ if not hasattr(proxide, name)]
  assert missing == []
