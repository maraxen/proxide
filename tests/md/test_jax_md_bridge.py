"""Tests for JAX MD bridge."""

import jax.numpy as jnp
import numpy as np
import pytest

from priox.md import jax_md_bridge
from priox.chem import residues as residue_constants

# Mock FullForceField
class MockFullForceField:
    def __init__(self):
        self.atom_class_map = {}
        self.atom_type_map = {}
        self.atom_key_to_id = {}
        self.bonds = []
        self.angles = []
        self.propers = []
        self.impropers = []
        self.cmap_torsions = []
        self.cmap_energy_grids = []
        self.residue_templates = {}

    def get_charge(self, res, atom):
        return 0.0

    def get_lj_params(self, res, atom):
        return 1.0, 0.1

@pytest.fixture
def mock_ff():
    ff = MockFullForceField()
    # Add some dummy data to avoid key errors if accessed
    return ff

def test_solve_periodic_spline_derivatives():
    # y = sin(x) on [0, 2pi]. derivatives should be cos(x).
    # periodic cubic spline on uniform grid.
    N = 20
    x = np.linspace(0, 2*np.pi, N, endpoint=False)
    y = np.sin(x)
    dx = x[1] - x[0]

    # The function calculates derivatives w.r.t index.
    # So dy/di = dy/dx * dx/di = cos(x) * dx
    expected_k = np.cos(x) * dx

    # This solves A * k = 3 * (y_{i+1} - y_{i-1})
    k = jax_md_bridge.solve_periodic_spline_derivatives(y)

    # Check if close
    np.testing.assert_allclose(k, expected_k, atol=1e-2)

def test_compute_bicubic_params():
    # Flat surface
    grid = np.zeros((10, 10))
    params = jax_md_bridge.compute_bicubic_params(grid)
    assert params.shape == (10, 10, 4)
    np.testing.assert_allclose(params, 0)

    # Periodic 2D function: f = sin(x) + cos(y)
    N = 20
    x = np.linspace(0, 2*np.pi, N, endpoint=False)
    y = np.linspace(0, 2*np.pi, N, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')
    grid = np.sin(X) + np.cos(Y)
    
    # derivatives w.r.t index (assuming dx=1 for simplicity in check logic,
    # but actual code computes derivative w.r.t index)
    # The function solve_periodic_spline_derivatives assumes uniform grid spacing of 1 unit?
    # Let's check source:
    # rhs = 3.0 * (y_next - y_prev)
    # A = [4, 1, 1]
    # This solves for k_i approx 3 * (y_{i+1} - y_{i-1}) / (4+1+1) approx (y_{i+1}-y_{i-1})/2
    # Yes, centered difference.
    # So it computes derivative w.r.t index.
    # dy/di = dy/dx * dx/di
    # dx/di = 2pi / N
    
    dx = 2*np.pi / N

    params = jax_md_bridge.compute_bicubic_params(grid)

    np.testing.assert_allclose(params[..., 0], grid)

    # fx should be cos(X) * dx
    np.testing.assert_allclose(params[..., 1], np.cos(X) * dx, atol=1e-2)

    # fy should be -sin(Y) * dy (dy=dx)
    np.testing.assert_allclose(params[..., 2], -np.sin(Y) * dx, atol=1e-2)

    # fxy should be 0
    np.testing.assert_allclose(params[..., 3], 0.0, atol=1e-2)

def test_assign_masses():
    names = ["CA", "H", "N", "O", "S", "X"]
    masses = jax_md_bridge.assign_masses(names)
    assert masses[0] == 12.011 # C
    assert masses[1] == 1.008 # H
    assert masses[2] == 14.007 # N
    assert masses[3] == 15.999 # O
    assert masses[4] == 32.06 # S
    assert masses[5] == 12.0 # Default

def test_assign_obc2_scaling_factors():
    names = ["H1", "CA", "N", "O", "F", "P", "S", "X"]
    factors = jax_md_bridge.assign_obc2_scaling_factors(names)
    assert factors[0] == 0.85 # H
    assert factors[1] == 0.72 # C
    assert factors[2] == 0.79 # N
    assert factors[3] == 0.85 # O
    assert factors[4] == 0.88 # F
    assert factors[5] == 0.86 # P
    assert factors[6] == 0.96 # S
    assert factors[7] == 0.80 # Other

def test_assign_mbondi2_radii():
    names = ["C", "N", "O", "S", "H", "H_bound_to_N"]
    # We need bonds to determine H context
    # C-N, N-H_bound_to_N, C-H
    # Indices: 0-C, 1-N, 2-O, 3-S, 4-H, 5-H_bound_to_N
    bonds = [[0, 1], [1, 5], [0, 4]]
    residues = ["UNK"] * 6

    radii = jax_md_bridge.assign_mbondi2_radii(names, residues, bonds)

    assert radii[0] == 1.70 # C
    assert radii[1] == 1.55 # N
    assert radii[2] == 1.50 # O
    assert radii[3] == 1.80 # S
    assert radii[4] == 1.20 # H (generic)
    assert radii[5] == 1.30 # H bound to N

def test_parameterize_system_topology(mock_ff):
    # Setup mock FF with specific topology rules

    # Bonds: Class A-B has length 1.0, k 100
    mock_ff.bonds.append(("A", "B", 0.1, 209.2)) # 0.1 nm = 1.0 A, 209.2 kJ -> k=1.0 kcal

    # Angles: A-B-C theta 1.57, k 100
    mock_ff.angles.append(("A", "B", "C", 1.57, 418.4))

    # Propers: A-B-C-D
    mock_ff.propers.append({
        "classes": ["A", "B", "C", "D"],
        "terms": [(1, 0, 1.0), (2, 3.14, 0.5)] # periodicity, phase, k
    })

    # Maps
    mock_ff.atom_class_map = {
        "RES_A": "A", "RES_B": "B", "RES_C": "C", "RES_D": "D"
    }

    # Residue template for internal bonds
    mock_ff.residue_templates = {
        "RES": [("A", "B"), ("B", "C"), ("C", "D")]
    }

    residues = ["RES"]
    atom_names = ["A", "B", "C", "D"] # Flattened
    atom_counts = [4]

    params = jax_md_bridge.parameterize_system(
        mock_ff, residues, atom_names, atom_counts
    )

    # Check Bonds
    # Should have A-B, B-C, C-D from templates
    assert len(params["bonds"]) == 3
    # Bond 0: A-B. Class A-B.
    # Bond 0 params: length 1.0, k = 209.2 / 418.4 = 0.5
    np.testing.assert_allclose(params["bond_params"][0], [1.0, 0.5])

    # Check Angles
    # Should have A-B-C, B-C-D generated from bonds
    assert len(params["angles"]) == 2

    # Check Dihedrals
    # A-B-C-D
    # Should have 2 dihedrals (one for each term)
    assert len(params["dihedrals"]) == 2
    # Check params (should have 2 terms)
    assert len(params["dihedral_params"]) == 2

def test_parameterize_system_improper_topology(mock_ff):
    # Test improper detection: Center K, neighbors I, J, L
    mock_ff.residue_templates = {
        "RES": [("K", "I"), ("K", "J"), ("K", "L")]
    }
    mock_ff.atom_class_map = {
        "RES_K": "K", "RES_I": "I", "RES_J": "J", "RES_L": "L"
    }
    mock_ff.impropers.append({
        "classes": ["K", "I", "J", "L"], # Center K
        "terms": [(2, 3.14, 10.0)]
    })

    residues = ["RES"]
    atom_names = ["K", "I", "J", "L"]
    atom_counts = [4]

    params = jax_md_bridge.parameterize_system(
        mock_ff, residues, atom_names, atom_counts
    )

    assert len(params["impropers"]) == 1
    # Params
    assert params["improper_params"][0][2] == 10.0
