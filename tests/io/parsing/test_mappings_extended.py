
"""Extended tests for priox.io.parsing.mappings to increase coverage."""

import unittest
from unittest import mock
import numpy as np
import pytest
from proxide.io.parsing import mappings

class TestMappingsExtended(unittest.TestCase):
    
    def test_alphabet_conversion(self):
        """Test AF <-> MPNN alphabet conversion."""
        # AF: A=0, R=1, ...
        # MPNN: A=0, C=1, ...
        # Just check round trip
        seq_af = np.array([0, 1, 2, 20])
        seq_mpnn = mappings.af_to_mpnn(seq_af)
        seq_af_back = mappings.mpnn_to_af(seq_mpnn)
        
        np.testing.assert_array_equal(seq_af, seq_af_back)

    def test_check_if_file_empty(self):
        """Test empty file check."""
        with mock.patch("pathlib.Path.open") as mock_open:
            # Empty text file
            mock_f = mock.Mock()
            mock_f.readable.return_value = True
            mock_f.read.return_value = ""
            mock_open.return_value.__enter__.return_value = mock_f
            
            self.assertTrue(mappings._check_if_file_empty("empty.txt"))
            
            # Non-empty text file
            mock_f.read.return_value = "content"
            self.assertFalse(mappings._check_if_file_empty("full.txt"))
            
            # File not found
            mock_open.side_effect = FileNotFoundError
            self.assertTrue(mappings._check_if_file_empty("missing.txt"))

    def test_string_key_to_index(self):
        """Test string key to index mapping."""
        keys = np.array(["A", "B", "C"])
        key_map = {"A": 0, "B": 1, "C": 2}
        
        indices = mappings.string_key_to_index(keys, key_map)
        np.testing.assert_array_equal(indices, [0, 1, 2])
        
        # Unknown key
        keys = np.array(["A", "X"])
        indices = mappings.string_key_to_index(keys, key_map, unk_index=99)
        np.testing.assert_array_equal(indices, [0, 99])

    def test_string_to_protein_sequence(self):
        """Test string to protein sequence conversion."""
        # Using default map (AF alphabet?)
        # mappings.string_to_protein_sequence uses restype_order (MPNN?)
        # And calls af_to_mpnn?
        # Let's check implementation
        # if aa_map is None: aa_map = restype_order; return af_to_mpnn(...)
        
        seq_str = "ACD"
        indices = mappings.string_to_protein_sequence(seq_str)
        self.assertEqual(len(indices), 3)
        
        # Round trip
        # protein_sequence_to_string expects MPNN indices?
        # It calls mpnn_to_af.
        
        back_str = mappings.protein_sequence_to_string(indices)
        self.assertEqual(back_str, seq_str)

    def test_residue_names_to_aatype(self):
        """Test residue name to aatype conversion."""
        res_names = np.array(["ALA", "CYS", "ASP"])
        aatypes = mappings.residue_names_to_aatype(res_names)
        
        self.assertEqual(len(aatypes), 3)
        # Check specific values if known, or just type
        self.assertTrue(np.issubdtype(aatypes.dtype, np.integer))

    def test_atom_names_to_index(self):
        """Test atom name to index conversion."""
        atom_names = np.array(["N", "CA", "C", "O"])
        indices = mappings.atom_names_to_index(atom_names)
        
        self.assertEqual(len(indices), 4)
        # Unknown atom
        atom_names = np.array(["X"])
        indices = mappings.atom_names_to_index(atom_names)
        self.assertEqual(indices[0], -1)

if __name__ == "__main__":
    unittest.main()
