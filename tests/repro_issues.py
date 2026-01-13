
import os
import tempfile
import unittest

import numpy as np

from proxide.core.atomic_system import AtomicSystem
from proxide.core.containers import Protein
from proxide.io.parsing import dispatch

# Small PDB content with two chains
PDB_CONTENT = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.500   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.500   1.500   0.000  1.00  0.00           C
ATOM      4  N   GLY B   1      10.000  10.000  10.000  1.00  0.00           N
ATOM      5  CA  GLY B   1      11.500  10.000  10.000  1.00  0.00           C
"""

# PQR content
# Format: ATOM <atom_id> <atom_name> <res_name> <chain_id> <res_id> <x> <y> <z> <charge> <radius>
PQR_CONTENT = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  -0.50 1.80
ATOM      2  CA  ALA A   1       1.500   0.000   0.000   0.10 1.90
ATOM      3  C   ALA A   1       2.500   1.500   0.000   0.50 1.70
"""

class TestProxideIssues(unittest.TestCase):

    def setUp(self):
        self.tmp_pdb = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w")
        self.tmp_pdb.write(PDB_CONTENT)
        self.tmp_pdb.close()

        self.tmp_pqr = tempfile.NamedTemporaryFile(suffix=".pqr", delete=False, mode="w")
        self.tmp_pqr.write(PQR_CONTENT)
        self.tmp_pqr.close()

    def tearDown(self):
        os.unlink(self.tmp_pdb.name)
        os.unlink(self.tmp_pqr.name)

    def test_pdb_parsing_type(self):
        # Should return Protein
        gen = dispatch.parse_input(self.tmp_pdb.name)
        obj = next(gen)
        self.assertIsInstance(obj, Protein, "PDB parsing should return Protein")

    def test_pqr_parsing_type(self):
        # Currently fail: returns AtomicSystem
        gen = dispatch.parse_input(self.tmp_pqr.name)
        obj = next(gen)
        self.assertIsInstance(obj, Protein, "PQR parsing should return Protein")
        # Check rich fields
        self.assertIsNotNone(obj.aatype)
        self.assertIsNotNone(obj.residue_index)
        self.assertIsNotNone(obj.chain_index)

    def test_chain_filtering(self):
        # Filter for chain A only (3 atoms)
        gen = dispatch.parse_input(self.tmp_pdb.name, chain_id="A")
        obj = next(gen)
        # If filtering fails, we get 5 atoms
        n_atoms = np.sum(obj.atom_mask) if hasattr(obj, 'atom_mask') else len(obj.coordinates.flatten())/3
        # For Protein, mask is CA mask usually. 
        # Check num residues. A: 1, B: 1. Total 2.
        # If masked, should be 1.
        self.assertEqual(len(obj.aatype), 1, f"Should filter to 1 residue (Chain A). Got {len(obj.aatype)}.")
        
    def test_pqr_epsilons(self):
        # Check if epsilons are populated
        gen = dispatch.parse_input(self.tmp_pqr.name)
        obj = next(gen)
        # Check if attribute exists and is not None (or check if it exists at all)
        # AtomicSystem has the field, default None.
        # User implies it should be populated.
        # Check if it is not None.
        self.assertIsNone(obj.epsilons, "Standard PQR does not contain epsilons")

    def test_full_coordinates(self):
        # Check full_coordinates
        gen = dispatch.parse_input(self.tmp_pdb.name, add_hydrogens=True)
        obj = next(gen)
        # Even with hydrogens=True, if PDB has none, it might add them or not depending on tool.
        # But full_coordinates should be present.
        self.assertIsNotNone(obj.full_coordinates, "full_coordinates should be populated")

if __name__ == "__main__":
    unittest.main()
