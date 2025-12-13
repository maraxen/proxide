
"""Extended tests for priox.md.jax_md_bridge to increase coverage."""

import unittest
import numpy as np
import jax.numpy as jnp
from proxide.md import jax_md_bridge

class TestJaxMDBridgeExtended(unittest.TestCase):
    
    def test_solve_periodic_spline_derivatives(self):
        """Test spline derivative solver."""
        # Simple case: constant function y=1 -> derivatives should be 0
        y = np.ones(10)
        k = jax_md_bridge.solve_periodic_spline_derivatives(y)
        np.testing.assert_allclose(k, 0.0, atol=1e-6)
        
        # Linear function (periodic saw-tooth approximation)
        # Not perfect derivative, but check shape and finite values
        y = np.linspace(0, 1, 10)
        k = jax_md_bridge.solve_periodic_spline_derivatives(y)
        self.assertEqual(k.shape, (10,))
        self.assertTrue(np.all(np.isfinite(k)))

    def test_compute_bicubic_params(self):
        """Test bicubic parameter computation."""
        # Flat grid -> all derivatives 0
        grid = np.ones((5, 5))
        params = jax_md_bridge.compute_bicubic_params(grid)
        
        # params: (5, 5, 4) [f, fx, fy, fxy]
        self.assertEqual(params.shape, (5, 5, 4))
        np.testing.assert_allclose(params[..., 0], 1.0) # f
        np.testing.assert_allclose(params[..., 1:], 0.0, atol=1e-6) # derivatives

    def test_assign_mbondi2_radii(self):
        """Test MBondi2 radii assignment."""
        atom_names = ["CA", "N", "O", "S", "H", "H"]
        res_names = ["ALA"] * 6
        # Bond H to N (indices 1 and 5)
        bonds = [[1, 5]] 
        
        radii = jax_md_bridge.assign_mbondi2_radii(atom_names, res_names, bonds)
        
        self.assertEqual(len(radii), 6)
        self.assertAlmostEqual(radii[0], 1.70) # C
        self.assertAlmostEqual(radii[1], 1.55) # N
        self.assertAlmostEqual(radii[2], 1.50) # O
        self.assertAlmostEqual(radii[3], 1.80) # S
        self.assertAlmostEqual(radii[4], 1.20) # H (generic)
        self.assertAlmostEqual(radii[5], 1.30) # H (bound to N)

    def test_assign_obc2_scaling_factors(self):
        """Test OBC2 scaling factors."""
        atom_names = ["H", "C", "N", "O", "F", "P", "S", "X"]
        factors = jax_md_bridge.assign_obc2_scaling_factors(atom_names)
        
        expected = [0.85, 0.72, 0.79, 0.85, 0.88, 0.86, 0.96, 0.80]
        np.testing.assert_allclose(factors, expected)

if __name__ == "__main__":
    unittest.main()
