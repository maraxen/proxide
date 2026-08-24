"""PyO3 binding + Python wrapper tests for the ligand reference frame."""

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
