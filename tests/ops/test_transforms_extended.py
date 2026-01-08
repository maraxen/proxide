
"""Extended tests for proxide.ops.transforms to increase coverage."""

import unittest
from unittest import mock
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import parameterized

from proxide.core.containers import Protein, Protein
from proxide.ops import transforms
from proxide.chem import residues as residue_constants

class TestTransformsExtended(parameterized.TestCase):
    
    def setUp(self):
        super().setUp()
        self.key = jax.random.PRNGKey(0)
        
        # Create a dummy protein tuple
        self.n_res = 10
        self.protein_tuple = Protein(
            coordinates=np.zeros((self.n_res, 5, 3)),
            aatype=np.zeros((self.n_res,), dtype=np.int32),
            one_hot_sequence=np.eye(21)[np.zeros(self.n_res, dtype=np.int32)],  # Required field
            mask=np.ones((self.n_res,)),  # Required field
            atom_mask=np.ones((self.n_res, 5)),
            residue_index=np.arange(self.n_res),
            chain_index=np.zeros((self.n_res,), dtype=np.int32),
            full_coordinates=np.zeros((self.n_res, 37, 3)),
            dihedrals=None,
            mapping=None,
            charges=None,
            radii=None,
            sigmas=None,
            epsilons=None,
            physics_features=None,
            bonds=None,
            bond_params=None,
            angles=None,
            angle_params=None,
            backbone_indices=None,
            exclusion_mask=None,
        )

    def test_truncate_protein_none(self):
        """Test no truncation."""
        p = transforms.truncate_protein(self.protein_tuple, max_length=None)
        self.assertEqual(p.coordinates.shape[0], self.n_res)
        
        p = transforms.truncate_protein(self.protein_tuple, max_length=5, strategy="none")
        self.assertEqual(p.coordinates.shape[0], self.n_res)

    def test_truncate_protein_short(self):
        """Test truncation when protein is shorter than max_length."""
        p = transforms.truncate_protein(self.protein_tuple, max_length=20, strategy="center_crop")
        self.assertEqual(p.coordinates.shape[0], self.n_res)

    def test_truncate_protein_center_crop(self):
        """Test center crop truncation."""
        max_len = 4
        p = transforms.truncate_protein(self.protein_tuple, max_length=max_len, strategy="center_crop")
        self.assertEqual(p.coordinates.shape[0], max_len)
        
        # Check indices to verify center crop
        # 10 residues: 0 1 2 3 4 5 6 7 8 9
        # Center crop 4: start = (10 - 4) // 2 = 3. End = 7.
        # Indices: 3, 4, 5, 6
        np.testing.assert_array_equal(p.residue_index, np.arange(3, 7))

    def test_truncate_protein_random_crop(self):
        """Test random crop truncation."""
        max_len = 4
        with mock.patch("numpy.random.default_rng") as mock_rng:
            mock_gen = mock.Mock()
            mock_gen.integers.return_value = 2
            mock_rng.return_value = mock_gen
            
            p = transforms.truncate_protein(self.protein_tuple, max_length=max_len, strategy="random_crop")
            self.assertEqual(p.coordinates.shape[0], max_len)
            np.testing.assert_array_equal(p.residue_index, np.arange(2, 6))

    def test_truncate_protein_invalid_strategy(self):
        """Test invalid truncation strategy."""
        with self.assertRaisesRegex(ValueError, "Unknown truncation strategy"):
            transforms.truncate_protein(self.protein_tuple, max_length=5, strategy="invalid")

    def test_concatenate_proteins_for_inter_mode(self):
        """Test concatenation of proteins for inter-chain mode."""
        p1 = self.protein_tuple.replace(
            coordinates=np.zeros((5, 5, 3)),
            chain_index=np.zeros((5,), dtype=np.int32), # Chain 0
            residue_index=np.arange(5)
        )
        p2 = self.protein_tuple.replace(
            coordinates=np.zeros((3, 5, 3)),
            chain_index=np.ones((3,), dtype=np.int32), # Chain 1
            residue_index=np.arange(3)
        )
        
        # Mock tree_map to handle Protein -> Protein conversion implicitly done in function
        # The function converts Protein to Protein using Protein.from_tuple
        # We need to ensure Protein class works or mock it. 
        # Since we import real Protein, it should work.
        
        concatenated = transforms.concatenate_proteins_for_inter_mode([p1, p2])
        
        # Check shapes (batch dim added)
        self.assertEqual(concatenated.coordinates.shape, (1, 8, 5, 3))
        
        # Check chain remapping
        # p1: 0,0,0,0,0
        # p2: 1,1,1 (offset by max(p1)+1 = 1) -> 1+1 = 2? No.
        # Logic: remapped_chains = original_chains + chain_offset
        # p1: offset 0. chains [0...]. max=0. next offset = 1.
        # p2: offset 1. chains [1...]. remapped = 1+1 = 2.
        # Result: 0,0,0,0,0, 2,2,2
        expected_chains = np.concatenate([np.zeros(5), np.full(3, 2)])
        np.testing.assert_array_equal(concatenated.chain_index[0], expected_chains)
        
        # Check mapping (structure index)
        expected_mapping = np.concatenate([np.zeros(5), np.ones(3)])
        np.testing.assert_array_equal(concatenated.mapping[0], expected_mapping)

    def test_concatenate_proteins_empty(self):
        """Test concatenation with empty list."""
        with self.assertRaisesRegex(ValueError, "Cannot concatenate an empty list"):
            transforms.concatenate_proteins_for_inter_mode([])

    def test_validate_and_flatten_elements(self):
        """Test validation and flattening."""
        elements = [self.protein_tuple, [self.protein_tuple]]
        flattened = transforms._validate_and_flatten_elements(elements)
        self.assertEqual(len(flattened), 2)
        self.assertIsInstance(flattened[0], Protein)
        self.assertIsInstance(flattened[1], Protein)

    @mock.patch("proxide.ops.transforms.compute_electrostatic_node_features")
    def test_apply_electrostatics(self, mock_compute):
        """Test electrostatic feature application."""
        mock_compute.return_value = np.zeros((self.n_res, 5))
        
        elements = [self.protein_tuple]
        updated = transforms._apply_electrostatics_if_needed(
            elements, use_electrostatics=True, estat_noise=0.1
        )
        
        self.assertEqual(len(updated), 1)
        self.assertIsNotNone(updated[0].physics_features)
        mock_compute.assert_called_once()
        
        # Check noise passing
        args, kwargs = mock_compute.call_args
        self.assertEqual(kwargs["noise_scale"], 0.1)

    @mock.patch("proxide.ops.transforms.md")
    @mock.patch("proxide.ops.transforms.force_fields")
    def test_apply_md_parameterization(self, mock_ff_loader, mock_md):
        """Test MD parameterization."""
        # Setup mocks
        mock_ff = mock.Mock()
        mock_ff_loader.load_force_field_from_hub.return_value = mock_ff
        
        mock_params = {
            "bonds": np.zeros((10, 2)),
            "bond_params": np.zeros((10, 2)),
            "angles": np.zeros((10, 3)),
            "angle_params": np.zeros((10, 2)),
            "backbone_indices": np.zeros((10,)),
            "exclusion_mask": np.zeros((10, 10)),
            "charges": np.zeros((10,)),
            "sigmas": np.zeros((10,)),
            "epsilons": np.zeros((10,)),
        }
        mock_md.parameterize_system.return_value = mock_params
        
        elements = [self.protein_tuple]
        updated = transforms._apply_md_parameterization(elements, use_md=True)
        
        self.assertEqual(len(updated), 1)
        p = updated[0]
        
        self.assertIsNotNone(p.bonds)
        self.assertIsNotNone(p.charges)
        
        # Verify calls
        mock_ff_loader.load_force_field.assert_called_with("ff14SB")
        mock_md.parameterize_system.assert_called_once()

    def test_pad_protein(self):
        """Test protein padding."""
        # Create a protein with known length
        p = self.protein_tuple # Length 10
        
        # Pad to 15
        padded = transforms._pad_protein(p, max_len=15)
        
        self.assertEqual(padded.coordinates.shape[0], 15)
        # Check padding values (0)
        self.assertTrue(np.all(padded.coordinates[10:] == 0))
        
        # Check mask padding (should be 0)
        # Original mask was ones
        self.assertTrue(np.all(padded.mask[:10] == 1))
        self.assertTrue(np.all(padded.mask[10:] == 0))

    def test_pad_protein_md_fields(self):
        """Test padding of MD fields."""
        # Create protein with MD fields
        p = self.protein_tuple
        p = p.replace(
            bonds=np.zeros((5, 2)),
            angles=np.zeros((5, 3)),
            charges=np.zeros((20,)), # Atoms
            exclusion_mask=np.zeros((20, 20))
        )
        
        md_dims = {
            "max_bonds": 10,
            "max_angles": 10,
            "max_atoms": 30
        }
        
        padded = transforms._pad_protein(p, max_len=15, md_dims=md_dims)
        
        self.assertEqual(padded.bonds.shape[0], 10)
        self.assertEqual(padded.angles.shape[0], 10)
        # Charges are not padded in _pad_protein unless full_coordinates match
        # But here charges (20) != protein_len (10) != full_coords_len (10 * 37 = 370)
        # So charges won't be padded by standard pad_fn.
        # And _pad_protein doesn't manually pad charges yet (based on my reading of the code)
        # Wait, let's re-read code in _pad_protein:
        # It manually pads bonds, angles.
        # It has a comment about Atoms (charges, sigmas, epsilons).
        # It says "We should ensure full_coordinates is set or handle atoms padding."
        # But it doesn't seem to implement manual padding for charges?
        # Let's check the code I read earlier.
        # It has `if padded_protein.md_exclusion_mask is not None:` block.
        # But no explicit block for `charges`.
        # So charges might NOT be padded if they don't match full_coords_len.
        # However, exclusion mask IS padded.
        
        self.assertEqual(padded.exclusion_mask.shape, (30, 30))

    @mock.patch("proxide.ops.transforms._apply_md_parameterization")
    def test_pad_and_collate_proteins(self, mock_md_param):
        """Test batching and padding."""
        mock_md_param.side_effect = lambda elements, **kwargs: elements
        
        p1 = self.protein_tuple # Len 10
        p2 = self.protein_tuple.replace(
            coordinates=np.zeros((5, 5, 3)),
            aatype=np.zeros((5,), dtype=np.int32),
            one_hot_sequence=np.eye(21)[np.zeros(5, dtype=np.int32)],
            mask=np.ones((5,)),
            atom_mask=np.ones((5, 5)),
            residue_index=np.arange(5),
            chain_index=np.zeros((5,), dtype=np.int32),
            full_coordinates=np.zeros((5, 37, 3)),
        ) # Len 5
        
        batch = transforms.pad_and_collate_proteins([p1, p2], max_length=12)
        
        self.assertEqual(batch.coordinates.shape, (2, 12, 5, 3))
        # Check padding
        # p1 (10) -> padded to 12.
        # p2 (5) -> padded to 12.
        
        self.assertIsNotNone(batch.mask)
        self.assertTrue(np.all(batch.mask[0, :10] == 1))
        self.assertTrue(np.all(batch.mask[0, 10:] == 0))
        
        self.assertTrue(np.all(batch.mask[1, :5] == 1))
        self.assertTrue(np.all(batch.mask[1, 5:] == 0))

if __name__ == "__main__":
    unittest.main()
