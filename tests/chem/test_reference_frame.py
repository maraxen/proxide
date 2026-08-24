"""PyO3 binding + Python wrapper tests for the ligand reference frame."""

from pathlib import Path

import numpy as np
import pytest


def _require_ligand_frame():
  try:
    from proxide._proxider import canonicalize_ligand_topology  # noqa: F401
  except ImportError:
    pytest.skip("proxide built without --features ligand-frame")


@pytest.mark.ligand_frame
def test_canonicalize_ligand_topology_rejects_disconnected_graph():
  _require_ligand_frame()
  from proxide._proxider import canonicalize_ligand_topology

  # Two disjoint 2-atom fragments.
  elements = ["C", "H", "C", "H"]
  atom_names = ["C1", "H1", "C2", "H2"]
  bonds = [(0, 1, 1, False), (2, 3, 1, False)]
  bond_is_aromatic = [False, False]
  rings = []
  formal_charges = None
  # float32: matches the `extract_coords` helper's strict dtype contract
  # (shared with `assign_gaff_atom_types`/`parameterize_molecule`), which
  # `canonicalize_ligand_topology`'s ref_positions argument reuses.
  ref_positions = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [5.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
    dtype=np.float32,
  )
  features = np.zeros((4, 116), dtype=np.float32)

  with pytest.raises(ValueError, match="disconnected"):
    canonicalize_ligand_topology(
      "two-fragments",
      elements,
      atom_names,
      bonds,
      bond_is_aromatic,
      rings,
      formal_charges,
      ref_positions,
      features,
      [],
      [],
      0.0,
    )


_BENZENE_MOL2 = Path(__file__).resolve().parent.parent / "io" / "parsing" / "benzene.mol2"


@pytest.mark.ligand_frame
def test_build_ligand_reference_frame_on_benzene():
  _require_ligand_frame()
  pytest.importorskip("rdkit")
  pytest.importorskip("expaloma")
  from proxide.chem.reference_frame import build_ligand_reference_frame
  from proxide.io.parsing.molecule import Molecule

  molecule = Molecule.from_mol2(_BENZENE_MOL2)
  trajectory = np.asarray(molecule.positions, dtype=np.float64)[np.newaxis, :, :]

  topology, coordinates = build_ligand_reference_frame(molecule, "benzene", trajectory)

  n_atoms = molecule.n_atoms
  assert topology.canonical_order.shape == (n_atoms,)
  assert sorted(topology.canonical_order.tolist()) == list(range(n_atoms))
  assert len(topology.pucker_definitions) == 1
  assert topology.pucker_definitions[0][1] == 6  # ring_size
  assert len(topology.torsion_definitions) == 0  # aromatic ring, no rotatable bonds

  assert coordinates.positions.shape == (1, n_atoms, 3)
  assert coordinates.frame_validity.shape == (1,)
  assert coordinates.frame_validity[0]
  assert coordinates.pucker_phase.shape == (1, 1)  # (n_rings, n_frames)
