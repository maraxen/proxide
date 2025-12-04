
"""Extended tests for priox.physics.features to increase coverage."""

import unittest
from unittest import mock
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from priox.physics import features
from priox.core.containers import ProteinTuple

class TestFeaturesExtended(unittest.TestCase):
    
    def setUp(self):
        # Create dummy protein tuple
        n_res = 2
        self.protein = ProteinTuple(
            coordinates=np.zeros((n_res, 5, 3)),
            aatype=np.zeros((n_res,), dtype=np.int32),
            atom_mask=np.ones((n_res, 5)),
            residue_index=np.arange(n_res),
            chain_index=np.zeros((n_res,), dtype=np.int32),
            full_coordinates=np.zeros((n_res, 37, 3)),
            charges=np.zeros((n_res, 37)),
            sigmas=np.ones((n_res, 37)),
            epsilons=np.ones((n_res, 37)),
        )

    def test_resolve_sigma(self):
        """Test sigma resolution."""
        # Direct
        self.assertEqual(features._resolve_sigma(1.0, "direct"), 1.0)
        self.assertEqual(features._resolve_sigma(None, "direct"), 0.0)
        
        # Thermal
        # sigma = sqrt(0.5 * R * T)
        # T=0 -> sigma=0
        self.assertEqual(features._resolve_sigma(0.0, "thermal"), 0.0)
        
        # Invalid mode
        with self.assertRaises(ValueError):
            features._resolve_sigma(1.0, "invalid")

    def test_compute_vdw_node_features(self):
        """Test vdW feature computation."""
        # Mock compute_lj_forces_at_backbone to avoid complex physics logic
        with mock.patch("priox.physics.features.compute_lj_forces_at_backbone") as mock_forces:
            # Return dummy forces (n_res, 5, 3)
            mock_forces.return_value = jnp.zeros((2, 5, 3))
            
            feats = features.compute_vdw_node_features(self.protein)
            
            self.assertEqual(feats.shape, (2, 5))
            mock_forces.assert_called_once()

    def test_compute_vdw_node_features_missing_data(self):
        """Test error handling for missing data."""
        # Missing sigmas
        p_no_sigma = self.protein._replace(sigmas=None)
        with self.assertRaisesRegex(ValueError, "must have sigmas"):
            features.compute_vdw_node_features(p_no_sigma)
            
        # Missing full_coordinates
        p_no_coords = self.protein._replace(full_coordinates=None)
        with self.assertRaisesRegex(ValueError, "must have full_coordinates"):
            features.compute_vdw_node_features(p_no_coords)

    def test_compute_vdw_node_features_noise(self):
        """Test vdW features with noise."""
        with mock.patch("priox.physics.features.compute_noised_lj_forces_at_backbone") as mock_forces:
            mock_forces.return_value = jnp.zeros((2, 5, 3))
            
            key = jax.random.key(0)
            features.compute_vdw_node_features(self.protein, noise_scale=1.0, key=key)
            
            # Check if noise_scale was passed to forces
            _, kwargs = mock_forces.call_args
            self.assertEqual(kwargs["noise_scale"], 1.0)

if __name__ == "__main__":
    unittest.main()
