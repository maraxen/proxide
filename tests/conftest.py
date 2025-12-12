"""Shared test fixtures."""

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from jax import random

from priox.io.parsing.dispatch import load_structure as parse_input
from priox.core.containers import Protein


@pytest.fixture(scope="session")
def protein_structure() -> Protein:
    """Load a sample protein structure from a PDB file."""
    pdb_path = Path(__file__).parent / "data" / "1ubq.pdb"
    return next(parse_input(str(pdb_path)))


@pytest.fixture(scope="session")
def pqr_protein() -> Protein:
    """Load a sample protein structure from a PQR file."""
    pqr_path = Path(__file__).parent / "data" / "1a00.pqr"
    return next(parse_input(str(pqr_path)))


@pytest.fixture(scope="session")
def model_inputs(protein_structure: Protein) -> dict:
    """Create model inputs from a protein structure."""
    return {
        "structure_coordinates": protein_structure.coordinates,
        "mask": protein_structure.mask,
        "residue_index": protein_structure.residue_index,
        "chain_index": protein_structure.chain_index,
        "sequence": protein_structure.aatype,
    }


@pytest.fixture
def rng_key() -> random.PRNGKey:
    """Create a new random key for testing."""
    return random.PRNGKey(0)

@pytest.fixture(params=[False, True], ids=["eager", "jit"])
def apply_jit(request):
    """Returns a function that conditionally JITs the input function."""
    should_jit = request.param

    def _wrapper(fn, **kwargs):
        if should_jit:
            return jax.jit(fn, **kwargs)
        return fn

    return _wrapper
