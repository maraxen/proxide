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


def _explicit_h_benzene() -> "Molecule":
  """Build an explicit-hydrogen benzene `Molecule` via RDKit (SMILES ->
  `AddHs` -> 3D embed), rather than relying on a hand-authored mol2
  fixture -- avoids hand-typing 12 atoms' worth of 3D coordinates that
  still have to pass the Rust geometry gate (bond-length/clash checks).

  `benzene.mol2` (used by the disconnected-graph-adjacent tests above) is
  deliberately H-free and left as-is: see
  `test_explicit_h_benzene_gets_ca_not_c1_gaff2_types` below and Finding 1
  of the 260824 final-review fix round -- `build_ligand_reference_frame`
  now rejects H-free input outright, so an H-free fixture can no longer
  reach `build_ligand_reference_frame` and produce a silently-wrong
  `gaff2_types` result the way it used to.
  """
  from rdkit import Chem
  from rdkit.Chem import AllChem

  from proxide.io.parsing.molecule import Molecule

  mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
  AllChem.EmbedMolecule(mol, randomSeed=0xC0FFEE)
  AllChem.MMFFOptimizeMolecule(mol)
  conf = mol.GetConformer()

  elements = [atom.GetSymbol() for atom in mol.GetAtoms()]
  positions = np.array(
    [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())],
    dtype=np.float32,
  )
  bond_type_to_order = {
    Chem.BondType.SINGLE: 1,
    Chem.BondType.DOUBLE: 2,
    Chem.BondType.TRIPLE: 3,
    Chem.BondType.AROMATIC: 1,
  }
  bonds: list[tuple[int, int]] = []
  bond_orders: list[int] = []
  bond_aromatic: list[bool] = []
  for bond in mol.GetBonds():
    bonds.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
    bond_orders.append(bond_type_to_order.get(bond.GetBondType(), 1))
    bond_aromatic.append(bond.GetIsAromatic())

  return Molecule(
    name="benzene_explicit_h",
    atom_names=[f"{e}{i + 1}" for i, e in enumerate(elements)],
    atom_types=[""] * len(elements),
    elements=elements,
    positions=positions,
    charges=np.zeros(len(elements), dtype=np.float32),
    bonds=bonds,
    bond_orders=bond_orders,
    bond_aromatic=bond_aromatic,
  )


@pytest.mark.ligand_frame
def test_build_ligand_reference_frame_on_benzene():
  _require_ligand_frame()
  pytest.importorskip("rdkit")
  pytest.importorskip("expaloma")
  from proxide.chem.reference_frame import build_ligand_reference_frame

  molecule = _explicit_h_benzene()
  trajectory = np.asarray(molecule.positions, dtype=np.float64)[np.newaxis, :, :]

  topology, coordinates = build_ligand_reference_frame(molecule, "benzene", trajectory)

  n_atoms = molecule.n_atoms
  assert topology.canonical_order.shape == (n_atoms,)
  assert sorted(topology.canonical_order.tolist()) == list(range(n_atoms))
  assert len(topology.pucker_definitions) == 1
  assert topology.pucker_definitions[0][1] == 6  # ring_size
  assert len(topology.torsion_definitions) == 0  # aromatic ring, no rotatable bonds

  # Finding 1 (260824 final-review fix round): with real explicit H, the
  # 6 ring carbons must type as `ca` (aromatic carbon) and the 6 H's as
  # `ha` -- not `c1` (sp1 carbon), the wrong type an H-free input silently
  # produced. Filtered/order-independent since `gaff2_types` is index-
  # aligned to `topology.canonical_order`, not input order.
  carbon_types = [t for e, t in zip(topology.elements, topology.gaff2_types, strict=True) if e == "C"]
  hydrogen_types = [t for e, t in zip(topology.elements, topology.gaff2_types, strict=True) if e == "H"]
  assert carbon_types == ["ca"] * 6
  assert hydrogen_types == ["ha"] * 6

  assert coordinates.positions.shape == (1, n_atoms, 3)
  assert coordinates.frame_validity.shape == (1,)
  assert coordinates.frame_validity[0]
  assert coordinates.pucker_phase.shape == (1, 1)  # (n_rings, n_frames)


@pytest.mark.ligand_frame
def test_build_ligand_reference_frame_rejects_h_free_molecule():
  """Finding 1 (260824 final-review fix round): a molecule missing
  explicit hydrogens must be rejected loudly, not silently mistyped (the
  original `benzene.mol2`-based bug: `gaff2_types` came out `c1` instead
  of `ca` because gaff2 typing was never told the ring carbons' real H
  count)."""
  _require_ligand_frame()
  pytest.importorskip("rdkit")
  pytest.importorskip("expaloma")
  from proxide.chem.reference_frame import build_ligand_reference_frame
  from proxide.io.parsing.molecule import Molecule

  molecule = Molecule.from_mol2(_BENZENE_MOL2)
  trajectory = np.asarray(molecule.positions, dtype=np.float64)[np.newaxis, :, :]

  with pytest.raises(ValueError, match="explicit hydrogens"):
    build_ligand_reference_frame(molecule, "benzene", trajectory)
