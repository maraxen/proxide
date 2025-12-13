
"""Extended tests for priox.io.parsing.foldcomp to increase coverage."""

import unittest
from unittest import mock
import numpy as np
import pytest
from priox.io.parsing import foldcomp

class TestFoldCompExtended(unittest.TestCase):
    
    def setUp(self):
        # Mock the foldcomp module if not installed, or patch it if it is
        self.mock_foldcomp = mock.Mock()
        self.patcher = mock.patch.dict("sys.modules", {"foldcomp": self.mock_foldcomp})
        self.patcher.start()
        
        # Force reload or re-import might be needed if it was already imported?
        # foldcomp.py handles import at top level with try/except.
        # If I patch sys.modules, I might need to reload priox.io.parsing.foldcomp?
        # But foldcomp.py checks FOLDCOMP_INSTALLED at runtime in functions too.
        
        # Let's patch the module-level variable FOLDCOMP_INSTALLED in priox.io.parsing.foldcomp
        self.foldcomp_installed_patch = mock.patch("priox.io.parsing.foldcomp.FOLDCOMP_INSTALLED", True)
        self.foldcomp_installed_patch.start()
        
        # Also patch the 'foldcomp' symbol in the module
        self.foldcomp_symbol_patch = mock.patch("priox.io.parsing.foldcomp.foldcomp", self.mock_foldcomp)
        self.foldcomp_symbol_patch.start()
        
        # Clear cache
        foldcomp._setup_foldcomp_database.cache_clear()

    def tearDown(self):
        self.patcher.stop()
        self.foldcomp_installed_patch.stop()
        self.foldcomp_symbol_patch.stop()

    def test_setup_foldcomp_database(self):
        """Test database setup."""
        foldcomp._setup_foldcomp_database("afdb_rep_v4")
        self.mock_foldcomp.setup.assert_called_with("afdb_rep_v4")

    def test_get_protein_structures(self):
        """Test retrieving protein structures."""
        # Mock context manager for foldcomp.open
        mock_ctx = mock.MagicMock()
        self.mock_foldcomp.open.return_value = mock_ctx
        
        # Mock iterator yielding (name, fcz)
        mock_ctx.__enter__.return_value = [("test_prot", "fcz_handle")]
        
        # Mock get_data
        self.mock_foldcomp.get_data.return_value = {
            "phi": [1.0, 1.0],
            "psi": [2.0, 2.0],
            "omega": [3.0, 3.0],
            "coordinates": np.zeros((2, 37, 3)),
            "residues": "AA"
        }
        
        structures = list(foldcomp.get_protein_structures(["test_prot"]))
        
        self.assertEqual(len(structures), 1)
        self.assertEqual(structures[0].aatype.shape, (2,))


    def test_get_protein_structures_error_handling(self):
        """Test error handling during processing."""
        mock_ctx = mock.MagicMock()
        self.mock_foldcomp.open.return_value = mock_ctx
        mock_ctx.__enter__.return_value = [("test_prot", "fcz_handle")]
        
        # Raise exception in get_data
        self.mock_foldcomp.get_data.side_effect = Exception("Corrupt data")
        
        structures = list(foldcomp.get_protein_structures(["test_prot"]))
        
        # Should skip the error and return empty list
        self.assertEqual(len(structures), 0)

    def test_not_installed(self):
        """Test behavior when foldcomp is not installed."""
        with mock.patch("priox.io.parsing.foldcomp.FOLDCOMP_INSTALLED", False):
            with self.assertRaises(ImportError):
                foldcomp._setup_foldcomp_database("afdb_rep_v4")
                
            with self.assertRaises(ImportError):
                list(foldcomp.get_protein_structures(["test"]))

if __name__ == "__main__":
    unittest.main()
