"""Tests for physics-based node features."""
import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from dataclasses import fields

from proxide.physics.features import (
    compute_electrostatic_features_batch,
    compute_electrostatic_node_features,
)
from proxide.core.containers import Protein
from proxide.core.atomic_system import AtomicSystem


def protein_to_dict(protein: Protein | AtomicSystem) -> dict:
    """Convert Protein or AtomicSystem dataclass to dictionary."""
    # Handle AtomicSystem's hierarchical structure
    if isinstance(protein, AtomicSystem) and not isinstance(protein, Protein):
        result = {}
        # Get coordinates from state
        if protein.state is not None:
            result["coordinates"] = protein.state.coordinates
        # Get topology fields
        if protein.topology is not None:
            result["residue_index"] = protein.topology.residue_index
            result["chain_index"] = protein.topology.chain_index
            result["elements"] = protein.topology.elements
            result["atom_names"] = protein.topology.atom_names
        # Get constants fields
        if protein.constants is not None:
            result["charges"] = protein.constants.charges
            result["radii"] = protein.constants.radii
        result["atom_mask"] = protein.atom_mask
        return result
    # Regular Protein dataclass
    return {f.name: getattr(protein, f.name) for f in fields(protein)}


def deep_tuple(x):
    """Recursively convert numpy arrays and lists to nested tuples."""
    if isinstance(x, np.ndarray):
        return tuple(deep_tuple(y) for y in x)
    if isinstance(x, list):
        return tuple(deep_tuple(y) for y in x)
    return x

def _create_dummy_protein(pqr_protein):
    data_dict = protein_to_dict(pqr_protein)
    
    # Infer n_res
    residue_index = data_dict.get("residue_index")
    # If residue_index is a tuple, convert to array
    if isinstance(residue_index, tuple):
        residue_index = np.array(residue_index)
        
    if residue_index is not None:
        unique_res = np.unique(residue_index)
        n_res = len(unique_res)
    else:
        # Check coordinates shape. If tuple, len(). If array, shape[0].
        coords = data_dict["coordinates"]
        if hasattr(coords, "shape"):
            l = coords.shape[0]
        else:
            l = len(coords)
            
        if l % 37 == 0:
             n_res = l // 37
        else:
             n_res = 10 # arbitrary fallback if structure is Full and we can't guess
             
    new_dict = {}
    
    # Ensure coordinates is array
    raw_coords = np.array(data_dict["coordinates"])
    
    if raw_coords.ndim == 2 and raw_coords.shape[0] != n_res * 37:
        # Full format input
        new_dict["full_coordinates"] = raw_coords
        # Dummy Atom37 coordinates (N, 37, 3)
        # Use random to avoid singularities
        new_dict["coordinates"] = np.random.randn(n_res, 37, 3).astype(np.float32)
        
        if "charges" in data_dict: new_dict["charges"] = np.array(data_dict["charges"])
        
        new_dict["atom_mask"] = np.ones((n_res, 37), dtype=np.float32)
        new_dict["mask"] = np.ones((n_res,), dtype=np.float32)
    else:
        new_dict["coordinates"] = raw_coords
        if "atom_mask" in data_dict: new_dict["atom_mask"] = np.array(data_dict["atom_mask"])
    
    # Fill required
    if "aatype" not in data_dict: new_dict["aatype"] = np.zeros((n_res,), dtype=np.int32)
    else: new_dict["aatype"] = np.array(data_dict["aatype"])
        
    if "one_hot_sequence" not in data_dict: new_dict["one_hot_sequence"] = np.eye(21)[new_dict["aatype"]]
    else: new_dict["one_hot_sequence"] = np.array(data_dict["one_hot_sequence"])
        
    if "mask" not in new_dict: new_dict["mask"] = np.ones((n_res,), dtype=np.float32)
    
    if "residue_index" not in new_dict: new_dict["residue_index"] = np.arange(n_res, dtype=np.int32)
    else: 
        ri = np.array(data_dict["residue_index"])
        if ri.shape[0] != n_res:
             new_dict["residue_index"] = np.arange(n_res, dtype=np.int32)
        else:
             new_dict["residue_index"] = ri
    
    if "chain_index" not in new_dict: new_dict["chain_index"] = np.zeros((n_res,), dtype=np.int32)
    else:
        ci = np.array(data_dict["chain_index"])
        if ci.shape[0] != n_res:
             new_dict["chain_index"] = np.zeros((n_res,), dtype=np.int32)
        else:
             new_dict["chain_index"] = ci

    # Handle full_coordinates
    if "full_coordinates" not in new_dict or new_dict.get("full_coordinates") is None:
         if raw_coords.shape[0] != n_res * 37:
             new_dict["full_coordinates"] = raw_coords
         else:
             new_dict["full_coordinates"] = raw_coords.reshape(-1, 3)
             
    # Ensure full coords are not all zero to prevent 1/0 infs
    fc = new_dict["full_coordinates"]
    if np.all(fc == 0):
        new_dict["full_coordinates"] = np.random.randn(*fc.shape).astype(np.float32)

    if "charges" not in new_dict or new_dict.get("charges") is None:
         n_full = new_dict["full_coordinates"].shape[0]
         new_dict["charges"] = np.zeros((n_full,), dtype=np.float32)

    return Protein(**new_dict)

@pytest.mark.parametrize("jit_compile", [True, False], ids=["jit", "eager"])
def test_compute_electrostatic_node_features_shape(
    pqr_protein: AtomicSystem, jit_compile,
):
    """Test that the computed features have the correct shape."""
    hashable_protein_tuple = _create_dummy_protein(pqr_protein)

    fn = compute_electrostatic_node_features
    if jit_compile:
        fn = jax.jit(fn)

    features = fn(hashable_protein_tuple)
    # Use n_res from the dummy protein, not the original pqr_protein
    n_residues = hashable_protein_tuple.coordinates.shape[0]
    chex.assert_shape(features, (n_residues, 5))
    chex.assert_tree_all_finite(features)


def test_compute_electrostatic_node_features_no_charges(
    pqr_protein: AtomicSystem,
):
    """Test that a ValueError is raised if protein has no charges."""
    p = _create_dummy_protein(pqr_protein)
    protein_no_charges = p.replace(charges=None)
    with pytest.raises(ValueError, match="must have charges"):
        compute_electrostatic_node_features(protein_no_charges)


def test_compute_electrostatic_node_features_no_full_coordinates(
    pqr_protein: AtomicSystem,
):
    """Test that a ValueError is raised if protein has no full_coordinates."""
    p = _create_dummy_protein(pqr_protein)
    protein_no_full_coords = p.replace(full_coordinates=None)
    with pytest.raises(ValueError, match="must have full_coordinates"):
        compute_electrostatic_node_features(protein_no_full_coords)


@pytest.mark.parametrize("jit_compile", [True, False], ids=["jit", "eager"])
def test_compute_electrostatic_node_features_jittable(
    pqr_protein: AtomicSystem, jit_compile,
):
    """Test that the feature computation can be JIT compiled."""
    hashable_protein_tuple = _create_dummy_protein(pqr_protein)

    fn = compute_electrostatic_node_features
    if jit_compile:
        fn = jax.jit(fn)
    features = fn(hashable_protein_tuple)
    chex.assert_tree_all_finite(features)


@pytest.mark.parametrize("jit_compile", [True, False], ids=["jit", "eager"])
def test_compute_electrostatic_features_batch_shape(
    pqr_protein: AtomicSystem, jit_compile,
):
    """Test that the batched features have the correct shape."""
    hashable_protein_tuple = _create_dummy_protein(pqr_protein)
    proteins = (hashable_protein_tuple, hashable_protein_tuple)
    
    fn = compute_electrostatic_features_batch
    if jit_compile:
        fn = jax.jit(fn)

    features, mask = fn(proteins)
    n_residues = hashable_protein_tuple.coordinates.shape[0]
    chex.assert_shape(features, (2, n_residues, 5))
    chex.assert_shape(mask, (2, n_residues))
    chex.assert_trees_all_close(mask, jnp.ones_like(mask))


@pytest.mark.parametrize("jit_compile", [True, False], ids=["jit", "eager"])
def test_compute_electrostatic_features_batch_padding(
    pqr_protein: AtomicSystem, jit_compile,
):
    """Test that padding is applied correctly."""
    hashable_protein_tuple = _create_dummy_protein(pqr_protein)
    proteins = (hashable_protein_tuple,)

    n_residues = hashable_protein_tuple.coordinates.shape[0]
    max_length = n_residues + 10

    fn = compute_electrostatic_features_batch
    if jit_compile:
        fn = jax.jit(fn, static_argnames=["max_length"])

    features, mask = fn(proteins, max_length=max_length)
    chex.assert_shape(features, (1, max_length, 5))
    chex.assert_shape(mask, (1, max_length))
    chex.assert_trees_all_close(jnp.sum(mask), n_residues)


def test_compute_electrostatic_features_batch_empty_list():
    """Test that an empty list of proteins raises a ValueError."""
    with pytest.raises(ValueError, match="Must provide at least one protein"):
        compute_electrostatic_features_batch([])


def test_compute_electrostatic_features_batch_max_length_too_small(
    pqr_protein: AtomicSystem,
):
    """Test that a small max_length raises a ValueError."""
    p = _create_dummy_protein(pqr_protein)
    proteins = [p]
    max_length = p.coordinates.shape[0] - 1
    with pytest.raises(ValueError, match="is less than longest sequence"):
        compute_electrostatic_features_batch(proteins, max_length=max_length)


def test_compute_electrostatic_node_features_thermal_mode(
    pqr_protein: AtomicSystem,
):
    """Test that thermal mode works correctly."""
    p = _create_dummy_protein(pqr_protein)
    # Use a dummy key for noise
    key = jax.random.key(0)

    # Mode: direct
    sigma_direct = 1.0
    features_direct = compute_electrostatic_node_features(
        p,
        noise_scale=sigma_direct,
        noise_mode="direct",
        key=key,
    )

    # Mode: thermal
    # Calculate T such that sigma is roughly 1.0
    # sigma = sqrt(0.5 * R * T) => sigma^2 = 0.5 * R * T => T = sigma^2 / (0.5 * R)
    from proxide.physics.constants import BOLTZMANN_KCAL

    t = 1.0 / (0.5 * BOLTZMANN_KCAL)

    features_temp = compute_electrostatic_node_features(
        p,
        noise_scale=t,
        noise_mode="thermal",
        key=key,
    )

    # The values should be close (floating point differences)
    chex.assert_trees_all_close(features_direct, features_temp, atol=1e-5)
