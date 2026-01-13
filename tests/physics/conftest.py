import jax.numpy as jnp
import pytest


@pytest.fixture
def backbone_positions_single_residue():
    """Returns backbone positions for a single residue (N, CA, C, O, CB)."""
    # N at origin, CA on x-axis, C in xy-plane
    # O and CB added for shape consistency (5 atoms)
    return jnp.array([[
        [0.0, 0.0, 0.0],    # N
        [1.458, 0.0, 0.0],  # CA
        [2.0, 1.0, 0.0],    # C
        [2.0, 2.0, 0.0],    # O (dummy)
        [1.5, -1.0, 0.0],   # CB (dummy)
    ]])

@pytest.fixture
def backbone_positions_multi_residue():
    """Returns backbone positions for two residues."""
    return jnp.array([
        [
            [0.0, 0.0, 0.0],    # N
            [1.458, 0.0, 0.0],  # CA
            [2.0, 1.0, 0.0],    # C
            [2.0, 2.0, 0.0],    # O
            [1.5, -1.0, 0.0],   # CB
        ],
        [
            [3.0, 0.0, 0.0],    # N
            [4.458, 0.0, 0.0],  # CA
            [5.0, 1.0, 0.0],    # C
            [5.0, 2.0, 0.0],    # O
            [4.5, -1.0, 0.0],   # CB
        ]
    ])

@pytest.fixture
def simple_charges():
    """Returns simple charges for electrostatics tests."""
    return jnp.array([1.0, -1.0])

@pytest.fixture
def simple_positions():
    """Returns simple positions for VDW tests."""
    return jnp.array([
        [0.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
    ])

@pytest.fixture
def lj_parameters():
    """Returns LJ parameters for VDW tests."""
    return {
        "sigma": jnp.array([1.0, 1.0]),
        "epsilon": jnp.array([0.1, 0.1]),
    }
