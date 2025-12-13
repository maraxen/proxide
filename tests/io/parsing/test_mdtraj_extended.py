
"""Extended tests for priox.io.parsing.mdtraj to increase coverage."""

import unittest
from unittest import mock
import numpy as np
import pytest
import mdtraj as md
from proxide.io.parsing import mdtraj

class TestMDTrajExtended(unittest.TestCase):
    
    def setUp(self):
        # Create a dummy topology
        self.top = md.Topology()
        c = self.top.add_chain()
        r = self.top.add_residue("ALA", c)
        self.top.add_atom("N", md.element.nitrogen, r)
        self.top.add_atom("CA", md.element.carbon, r)
        self.top.add_atom("C", md.element.carbon, r)
        
        # Create dummy trajectory
        self.xyz = np.zeros((1, 3, 3))
        self.traj = md.Trajectory(self.xyz, self.top)

    def test_mdtraj_dihedrals(self):
        """Test dihedral calculation."""
        # Mock md.compute_*
        with mock.patch("mdtraj.compute_phi") as mock_phi, \
             mock.patch("mdtraj.compute_psi") as mock_psi, \
             mock.patch("mdtraj.compute_omega") as mock_omega:
            
            # Return indices for residue 0
            mock_phi.return_value = (np.array([[0, 1]]), np.array([[1.0]]))
            mock_psi.return_value = (np.array([[0, 1]]), np.array([[2.0]]))
            mock_omega.return_value = (np.array([[0, 1]]), np.array([[3.0]]))
            
            # Nitrogen mask for 1 residue (index 0)
            nitrogen_mask = np.array([True]) 
            
            # We need to ensure the indices returned by compute_* match the residue count
            # My setup has 1 residue.
            # compute_phi returns (indices, angles). indices is (N, 2).
            # If I have 1 residue, I expect 1 angle?
            # But compute_phi returns angles for valid phi.
            # Let's assume it returns 1 angle for residue 0.
            
            dihedrals = mdtraj.mdtraj_dihedrals(self.traj, num_residues=1, nitrogen_mask=nitrogen_mask)
            
            self.assertEqual(dihedrals.shape, (1, 3))
            self.assertEqual(dihedrals[0, 0], 1.0)
            self.assertEqual(dihedrals[0, 1], 2.0)
            self.assertEqual(dihedrals[0, 2], 3.0)

    def test_select_chain_mdtraj(self):
        """Test chain selection."""
        # Use real topology logic instead of patching property
        # MDTraj chains usually have index. chain_id might be empty string by default.
        # We can try to set it?
        # c = self.top.chain(0) -> c.chain_id is read-only?
        # In newer MDTraj it might be.
        # Let's try to select by index if chain_id fails.
        # But `_select_chain_mdtraj` uses `c.chain_id`.
        
        # Alternative: Create a mock trajectory object that behaves like md.Trajectory
        # but has a mock topology with controllable chains.
        
        mock_traj = mock.Mock(spec=md.Trajectory)
        mock_top = mock.Mock()
        mock_traj.top = mock_top
        
        # Mock chains
        c1 = mock.Mock()
        c1.chain_id = "A"
        c1.index = 0
        
        c2 = mock.Mock()
        c2.chain_id = "B"
        c2.index = 1
        
        mock_top.chains = [c1, c2]
        mock_top.select.return_value = np.array([0, 1, 2]) # Atoms for chain A
        mock_traj.atom_slice.return_value = mock_traj # Return self for chaining
        
        result = mdtraj._select_chain_mdtraj(mock_traj, chain_id="A")
        
        mock_top.select.assert_called()
        # Check selection string
        args, _ = mock_top.select.call_args
        self.assertIn("chainid 0", args[0])

    def test_extract_mdtraj_static_features(self):
        """Test static feature extraction."""
        features = mdtraj._extract_mdtraj_static_features(self.traj)
        
        self.assertEqual(features.num_residues, 1)
        self.assertTrue(features.nitrogen_mask[0])
        self.assertEqual(features.aatype[0], 0) # ALA -> 0

    def test_mdtraj_to_atom_array(self):
        """Test conversion to AtomArray."""
        atom_array = mdtraj._mdtraj_to_atom_array(self.traj)
        
        self.assertEqual(atom_array.array_length(), 3)
        self.assertEqual(atom_array.res_name[0], "ALA")
        self.assertEqual(atom_array.atom_name[0], "N")
        
        # Test stack conversion
        xyz_stack = np.zeros((2, 3, 3))
        traj_stack = md.Trajectory(xyz_stack, self.top)
        stack = mdtraj._mdtraj_to_atom_array(traj_stack)
        
        self.assertEqual(stack.stack_depth(), 2)

    @mock.patch("priox.io.parsing.mdtraj.hydride.add_hydrogen")
    def test_add_hydrogens_if_needed(self, mock_add):
        """Test hydrogen addition."""
        atom_array = mdtraj._mdtraj_to_atom_array(self.traj)
        mock_add.return_value = (atom_array, None)
        
        result = mdtraj._add_hydrogens_if_needed(atom_array)
        
        mock_add.assert_called_once()
        self.assertTrue("charge" in result.get_annotation_categories())

    @mock.patch("mdtraj.load_frame")
    @mock.patch("mdtraj.iterload")
    def test_parse_mdtraj_to_processed_structure(self, mock_iterload, mock_load_frame):
        """Test parsing pipeline."""
        mock_load_frame.return_value = self.traj
        mock_iterload.return_value = [self.traj]
        
        # Disable add_hydrogens to keep atom count predictable (3)
        gen = mdtraj.parse_mdtraj_to_processed_structure("dummy.h5", chain_id=None, add_hydrogens=False)
        results = list(gen)
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].atom_array.array_length(), 3)

    def test_parse_mdtraj_mdcath_warning(self):
        """Test warning for mdCATH files."""
        with mock.patch("h5py.File") as mock_h5:
            mock_file = mock.Mock()
            mock_file.attrs = {"layout": "mdcath"}
            mock_h5.return_value.__enter__.return_value = mock_file
            
            # We expect a warning if chain_id is provided
            with pytest.warns(UserWarning, match="Chain selection is not supported"):
                # Mock load_frame and _select_chain_mdtraj to avoid errors
                with mock.patch("mdtraj.load_frame") as mock_load:
                    mock_load.return_value = self.traj
                    with mock.patch("priox.io.parsing.mdtraj._select_chain_mdtraj") as mock_select:
                        mock_select.return_value = self.traj
                        with mock.patch("mdtraj.iterload") as mock_iter:
                            mock_iter.return_value = []
                            list(mdtraj.parse_mdtraj_to_processed_structure("test.h5", chain_id="A"))

if __name__ == "__main__":
    unittest.main()
