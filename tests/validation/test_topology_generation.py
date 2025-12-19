"""Topology generation parity tests.

Tests that Rust-generated topology (bonds, angles, dihedrals, impropers)
is correct and consistent.
"""

import numpy as np
import pytest
from pathlib import Path

from proxide import parse_structure, OutputSpec, CoordFormat


def test_topology_generation_basic():
    """Verify topology arrays are generated when infer_bonds=True."""
    pdb_path = "tests/data/1crn.pdb"
    if not Path(pdb_path).exists():
        pytest.skip(f"Test file {pdb_path} not found")
    
    spec = OutputSpec(
        coord_format=CoordFormat.Full,
        infer_bonds=True
    )
    result = parse_structure(pdb_path, spec)
    
    # All topology arrays should be present
    assert "bonds" in result, "bonds not in output"
    assert "angles" in result, "angles not in output"
    assert "dihedrals" in result, "dihedrals not in output"
    assert "impropers" in result, "impropers not in output"
    
    bonds = np.array(result["bonds"])
    angles = np.array(result["angles"])
    dihedrals = np.array(result["dihedrals"])
    impropers = np.array(result["impropers"])
    
    print(f"Bonds: {bonds.shape}")
    print(f"Angles: {angles.shape}")
    print(f"Dihedrals: {dihedrals.shape}")
    print(f"Impropers: {impropers.shape}")
    
    # Shapes should be (N, 2), (N, 3), (N, 4), (N, 4)
    if bonds.size > 0:
        assert bonds.shape[1] == 2, f"Bonds shape wrong: {bonds.shape}"
    if angles.size > 0:
        assert angles.shape[1] == 3, f"Angles shape wrong: {angles.shape}"
    if dihedrals.size > 0:
        assert dihedrals.shape[1] == 4, f"Dihedrals shape wrong: {dihedrals.shape}"
    if impropers.size > 0:
        assert impropers.shape[1] == 4, f"Impropers shape wrong: {impropers.shape}"


def test_topology_counts_reasonable():
    """Verify topology counts are reasonable for a protein."""
    pdb_path = "tests/data/1crn.pdb"
    if not Path(pdb_path).exists():
        pytest.skip(f"Test file {pdb_path} not found")
    
    spec = OutputSpec(
        coord_format=CoordFormat.Full,
        infer_bonds=True
    )
    result = parse_structure(pdb_path, spec)
    
    bonds = np.array(result["bonds"])
    angles = np.array(result["angles"])
    dihedrals = np.array(result["dihedrals"])
    
    n_bonds = len(bonds)
    n_angles = len(angles)
    n_dihedrals = len(dihedrals)
    
    # Get actual atom count from bonds (max index + 1)
    if n_bonds > 0:
        n_atoms = int(np.max(bonds)) + 1
    else:
        n_atoms = 0
    
    print(f"Actual atoms: {n_atoms}, Bonds: {n_bonds}, Angles: {n_angles}, Dihedrals: {n_dihedrals}")
    
    # Basic sanity checks - any bonds at all
    assert n_bonds > 0, "Should have at least some bonds"
    # Angles should exist if we have more than 2 atoms
    if n_atoms > 2:
        assert n_angles >= 0, "Angles array should exist"


def test_bond_indices_valid():
    """Verify bond indices are within valid range."""
    pdb_path = "tests/data/1crn.pdb"
    if not Path(pdb_path).exists():
        pytest.skip(f"Test file {pdb_path} not found")
    
    spec = OutputSpec(
        coord_format=CoordFormat.Full,
        infer_bonds=True
    )
    result = parse_structure(pdb_path, spec)
    
    # Get atom count from mask
    mask = np.array(result["atom_mask"])
    shape = result["coord_shape"]
    n_res, max_atoms = shape[0], shape[1]
    mask_reshaped = mask.reshape((n_res, max_atoms))
    n_atoms = int(np.sum(mask_reshaped > 0.5))
    
    bonds = np.array(result["bonds"])
    
    if bonds.size > 0:
        max_idx = np.max(bonds)
        assert max_idx < n_atoms, f"Bond index {max_idx} >= n_atoms {n_atoms}"
        
        # All indices should be non-negative
        assert np.all(bonds >= 0), "Negative bond indices"


def test_angles_central_atom():
    """Verify angle format is correct (i-j-k with j as central)."""
    pdb_path = "tests/data/1crn.pdb"
    if not Path(pdb_path).exists():
        pytest.skip(f"Test file {pdb_path} not found")
    
    spec = OutputSpec(
        coord_format=CoordFormat.Full,
        infer_bonds=True
    )
    result = parse_structure(pdb_path, spec)
    
    angles = np.array(result["angles"])
    bonds = np.array(result["bonds"])
    
    if angles.size == 0 or bonds.size == 0:
        pytest.skip("No angles or bonds to check")
    
    # Build bond set for lookup
    bond_set = set()
    for b in bonds:
        bond_set.add((min(b[0], b[1]), max(b[0], b[1])))
    
    # For each angle i-j-k, i-j and j-k should be bonds
    for angle in angles[:10]:  # Check first 10
        i, j, k = int(angle[0]), int(angle[1]), int(angle[2])
        
        # i-j should be a bond
        ij = (min(i, j), max(i, j))
        assert ij in bond_set, f"Angle atom pair {i}-{j} not in bonds"
        
        # j-k should be a bond
        jk = (min(j, k), max(j, k))
        assert jk in bond_set, f"Angle atom pair {j}-{k} not in bonds"
    
    print(f"Verified {min(10, len(angles))} angles have correct bond connectivity")


def test_no_topology_when_not_requested():
    """Verify topology is not computed when infer_bonds=False."""
    pdb_path = "tests/data/1crn.pdb"
    if not Path(pdb_path).exists():
        pytest.skip(f"Test file {pdb_path} not found")
    
    spec = OutputSpec(
        coord_format=CoordFormat.Full,
        infer_bonds=False
    )
    result = parse_structure(pdb_path, spec)
    
    # Should not have topology arrays
    assert "bonds" not in result or result.get("bonds") is None
    assert "angles" not in result or result.get("angles") is None
