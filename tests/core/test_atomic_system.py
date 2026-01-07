"""Tests for atomic system definitions."""

import jax.numpy as jnp
import pytest

from proxide.core.atomic_system import (
  AtomicConstants,
  AtomicState,
  AtomicSystem,
  Molecule,
  MolecularTopology,
)
from proxide.core.containers import Protein


def test_atomic_system_initialization():
  """Test initializing AtomicSystem with hierarchical structure."""
  topo = MolecularTopology(
    elements=["C"] * 5,
    atom_names=["CA"] * 5,
  )
  state = AtomicState(coordinates=jnp.zeros((5, 3)))

  sys = AtomicSystem(
    topology=topo,
    state=state,
    atom_mask=jnp.ones((5,)),
  )

  assert sys.state.coordinates.shape == (5, 3)
  assert sys.num_atoms == 5
  assert len(sys.topology.elements) == 5


def test_atomic_system_getattr_delegation():
  """Test backward-compatible attribute delegation via __getattr__."""
  topo = MolecularTopology(elements=["C", "N", "O"])
  state = AtomicState(coordinates=jnp.ones((3, 3)))
  constants = AtomicConstants(charges=jnp.array([0.1, -0.2, 0.1]))

  sys = AtomicSystem(topology=topo, state=state, constants=constants)

  # Delegation to state
  assert sys.coordinates.shape == (3, 3)

  # Delegation to topology
  assert sys.elements == ["C", "N", "O"]

  # Delegation to constants
  assert sys.charges is not None
  assert len(sys.charges) == 3


def test_atomic_system_from_arrays():
  """Test factory method for constructing from flat arrays."""
  sys = AtomicSystem.from_arrays(
    coordinates=jnp.zeros((10, 3)),
    atom_mask=jnp.ones((10,)),
    elements=["C"] * 10,
    charges=jnp.zeros((10,)),
  )

  assert sys.num_atoms == 10
  assert sys.elements is not None
  assert sys.constants is not None
  assert sys.constants.charges is not None


def test_molecule_inheritance():
  """Test Molecule initialization and inheritance."""
  topo = MolecularTopology(elements=["C", "C", "O"])
  state = AtomicState(coordinates=jnp.zeros((3, 3)))

  mol = Molecule(
    topology=topo,
    state=state,
    atom_mask=jnp.ones((3,)),
  )

  assert isinstance(mol, AtomicSystem)
  assert mol.num_atoms == 3


def test_atomic_system_merge():
  """Test merging two AtomicSystems."""
  sys1 = AtomicSystem.from_arrays(
    coordinates=jnp.zeros((3, 3)),
    elements=["C", "N", "O"],
  )
  sys2 = AtomicSystem.from_arrays(
    coordinates=jnp.ones((2, 3)),
    elements=["H", "H"],
  )

  merged = sys1.merge_with(sys2)
  assert merged.num_atoms == 5
  assert len(merged.topology.elements) == 5


def test_protein_is_standalone():
  """Test that Protein is now standalone (not AtomicSystem subclass)."""
  prot = Protein(
    coordinates=jnp.zeros((10, 37, 3)),
    aatype=jnp.zeros((10,), dtype=jnp.int32),
    one_hot_sequence=jnp.zeros((10, 21)),
    mask=jnp.zeros((10,)),
    residue_index=jnp.arange(10),
    chain_index=jnp.zeros((10,)),
    atom_mask=jnp.zeros((10, 37)),
  )

  # Protein is no longer an AtomicSystem subclass
  assert not isinstance(prot, AtomicSystem)
  assert prot.coordinates.ndim == 3


def test_atomic_system_properties():
  """Test computed properties."""
  topo = MolecularTopology(
    molecule_type=jnp.array([0, 0, 1, 2]),  # 2 protein, 1 ligand, 1 solvent
  )
  state = AtomicState(coordinates=jnp.zeros((4, 3)))

  sys = AtomicSystem(topology=topo, state=state)

  assert sys.num_atoms == 4
  assert sys.num_protein_atoms == 2
  assert sys.num_ligand_atoms == 1
  assert sys.has_ligands is True
  assert sys.has_solvent is True
