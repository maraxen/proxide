"""Tests for atomic system definitions."""

import jax.numpy as jnp
from proxide.core.atomic_system import AtomicSystem, Molecule
from proxide.core.containers import Protein
import pytest

def test_atomic_system_initialization():
    """Test initializing AtomicSystem."""
    coords = jnp.zeros((5, 3))
    mask = jnp.ones((5,))
    
    sys = AtomicSystem(
        coordinates=coords,
        atom_mask=mask,
        elements=["C"] * 5,
        atom_names=["CA"] * 5
    )
    
    assert sys.coordinates.shape == (5, 3)
    assert sys.atom_mask.shape == (5,)
    assert len(sys.elements) == 5

def test_molecule_inheritance():
    """Test Molecule initialization and inheritance."""
    coords = jnp.zeros((3, 3))
    mask = jnp.ones((3,))
    
    mol = Molecule(
        coordinates=coords,
        atom_mask=mask,
        elements=["C", "C", "O"],
        atom_names=["C1", "C2", "O1"]
    )
    
    assert isinstance(mol, AtomicSystem)
    assert mol.coordinates.shape == (3, 3)
    assert mol.atom_mask.shape == (3,)

def test_protein_inheritance():
    """Test that Protein inherits from AtomicSystem."""
    # Note: Protein requires explicit fields due to kw_only=True
    prot = Protein(
        coordinates=jnp.zeros((10, 37, 3)),
        aatype=jnp.zeros((10,), dtype=jnp.int32),
        one_hot_sequence=jnp.zeros((10, 21)),
        mask=jnp.zeros((10,)),
        residue_index=jnp.arange(10),
        chain_index=jnp.zeros((10,)),
        atom_mask=jnp.zeros((10, 37)), # Explicitly provided or None default?
    ) 
    
    assert isinstance(prot, AtomicSystem)
    assert prot.coordinates.ndim == 3 # Protein specific shape

