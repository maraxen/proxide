
"""Extended tests for priox.io.parsing.biotite to increase coverage."""

import unittest
from unittest import mock
import numpy as np
import pytest
from biotite.structure import Atom, AtomArray, AtomArrayStack, array

from priox.io.parsing import biotite

class TestBiotiteExtended(unittest.TestCase):
    
    def setUp(self):
        # Create a simple AtomArray
        # 3 atoms: N, CA, C
        atom1 = Atom([0, 0, 0], chain_id="A", res_id=1, res_name="ALA", atom_name="N", element="N")
        atom2 = Atom([1, 0, 0], chain_id="A", res_id=1, res_name="ALA", atom_name="CA", element="C")
        atom3 = Atom([2, 0, 0], chain_id="A", res_id=1, res_name="ALA", atom_name="C", element="C")
        self.atom_array = array([atom1, atom2, atom3])
        
        # Solvent atom
        self.solvent = Atom([5, 5, 5], chain_id="A", res_id=2, res_name="HOH", atom_name="O", element="O")

    def test_remove_solvent(self):
        """Test solvent removal."""
        # Create array with solvent
        structure_with_solvent = self.atom_array + array([self.solvent])
        
        # Mock filter_solvent to return mask
        with mock.patch("biotite.structure.filter_solvent") as mock_filter:
            # Mask: True for solvent, False for protein
            mock_filter.return_value = np.array([False, False, False, True])
            
            cleaned = biotite._remove_solvent_from_structure(structure_with_solvent)
            
            self.assertEqual(cleaned.array_length(), 3)
            self.assertEqual(cleaned[0].res_name, "ALA")

    def test_remove_solvent_stack(self):
        """Test solvent removal from stack."""
        # Create stack
        structure_with_solvent = self.atom_array + array([self.solvent])
        from biotite.structure import stack as stack_arrays
        stack = stack_arrays([structure_with_solvent, structure_with_solvent])
        # Need to set annotations on stack? AtomArrayStack shares annotations with AtomArray?
        # No, AtomArrayStack constructor takes coords. Annotations are separate?
        # Actually Biotite AtomArrayStack is usually created by stacking AtomArrays or from file.
        # Let's mock isinstance check or just use mock object.
        
        # Easier: Mock the function logic or use real Biotite objects if simple.
        # Real Biotite objects are fine.
        # But setting annotations on stack created from coords is tricky.
        # Let's use `biotite.structure.stack([structure_with_solvent, structure_with_solvent])`
        from biotite.structure import stack as stack_arrays
        stack = stack_arrays([structure_with_solvent, structure_with_solvent])
        
        with mock.patch("biotite.structure.filter_solvent") as mock_filter:
            mock_filter.return_value = np.array([False, False, False, True])
            
            cleaned = biotite._remove_solvent_from_structure(stack)
            
            self.assertEqual(cleaned.stack_depth(), 2)
            self.assertEqual(cleaned.array_length(), 3)

    @mock.patch("priox.io.parsing.biotite.hydride.add_hydrogen")
    def test_add_hydrogens_existing(self, mock_add):
        """Test skipping hydrogen addition if present."""
        # Add H to array
        h_atom = Atom([0, 1, 0], chain_id="A", res_id=1, res_name="ALA", atom_name="H", element="H")
        structure_with_h = self.atom_array + array([h_atom])
        
        result = biotite._add_hydrogens_to_structure(structure_with_h)
        
        self.assertEqual(result.array_length(), 4)
        mock_add.assert_not_called()

    @mock.patch("priox.io.parsing.biotite.hydride.add_hydrogen")
    def test_add_hydrogens_new(self, mock_add):
        """Test adding hydrogens when missing."""
        # Mock hydride return
        mock_add.return_value = (self.atom_array, None)
        
        # Mock connect_via_residue_names
        with mock.patch("biotite.structure.connect_via_residue_names") as mock_connect:
            mock_connect.return_value = None # Just to avoid error
            
            result = biotite._add_hydrogens_to_structure(self.atom_array)
            
            mock_add.assert_called_once()
            # Check if charges were added (annotation)
            self.assertTrue("charge" in result.get_annotation_categories())

    def test_fix_arg_protonation_no_arg(self):
        """Test fix_arg_protonation with no ARG."""
        result = biotite._fix_arg_protonation(self.atom_array)
        self.assertEqual(result.array_length(), 3)

    def test_fix_arg_protonation_logic(self):
        """Test detailed logic for ARG protonation."""
        # Create an ARG residue with missing hydrogens
        # N, CA, C, O, CB, CG, CD, NE, HE, CZ, NH1, NH2
        # Missing HH11, HH12, HH21, HH22
        
        atoms = []
        res_name = "ARG"
        rid = 1
        chain = "A"
        
        # Backbone + Sidechain heavy atoms (simplified positions)
        atoms.append(Atom([0,0,0], chain_id=chain, res_id=rid, res_name=res_name, atom_name="N", element="N"))
        atoms.append(Atom([1,0,0], chain_id=chain, res_id=rid, res_name=res_name, atom_name="CA", element="C"))
        atoms.append(Atom([2,0,0], chain_id=chain, res_id=rid, res_name=res_name, atom_name="C", element="C"))
        
        # Sidechain (approximate linear for simplicity, test logic handles geometry)
        atoms.append(Atom([1,1,0], chain_id=chain, res_id=rid, res_name=res_name, atom_name="CB", element="C"))
        atoms.append(Atom([1,2,0], chain_id=chain, res_id=rid, res_name=res_name, atom_name="CG", element="C"))
        atoms.append(Atom([1,3,0], chain_id=chain, res_id=rid, res_name=res_name, atom_name="CD", element="C"))
        atoms.append(Atom([1,4,0], chain_id=chain, res_id=rid, res_name=res_name, atom_name="NE", element="N"))
        atoms.append(Atom([1,4,1], chain_id=chain, res_id=rid, res_name=res_name, atom_name="HE", element="H"))
        atoms.append(Atom([1,5,0], chain_id=chain, res_id=rid, res_name=res_name, atom_name="CZ", element="C"))
        atoms.append(Atom([0,6,0], chain_id=chain, res_id=rid, res_name=res_name, atom_name="NH1", element="N"))
        atoms.append(Atom([2,6,0], chain_id=chain, res_id=rid, res_name=res_name, atom_name="NH2", element="N"))
        
        arg_array = array(atoms)
        
        fixed = biotite._fix_arg_protonation(arg_array)
        
        # Should have added 4 hydrogens: HH11, HH12, HH21, HH22
        # Original 11 atoms. New 15 atoms.
        self.assertEqual(fixed.array_length(), 15)
        
        atom_names = fixed.atom_name
        self.assertIn("HH11", atom_names)
        self.assertIn("HH12", atom_names)
        self.assertIn("HH21", atom_names)
        self.assertIn("HH22", atom_names)

    @mock.patch("priox.io.parsing.biotite.structure_io.load_structure")
    @mock.patch("priox.io.parsing.biotite._add_hydrogens_to_structure")
    def test_load_structure_with_hydride(self, mock_add_h, mock_load):
        """Test loading structure with options."""
        mock_load.return_value = self.atom_array
        mock_add_h.return_value = self.atom_array
        
        # Test basic load
        res = biotite.load_structure_with_hydride("dummy.pdb", add_hydrogens=True)
        mock_load.assert_called()
        mock_add_h.assert_called()
        
        # Test with topology
        mock_load.reset_mock()
        res = biotite.load_structure_with_hydride("dummy.xtc", topology="top.pdb")
        # Should load topology first, then trajectory
        self.assertEqual(mock_load.call_count, 2)

    def test_biotite_to_jax_md_system(self):
        """Test conversion to JAX MD system."""
        # Mock jax_md_bridge
        with mock.patch("priox.md.jax_md_bridge.parameterize_system") as mock_param:
            mock_param.return_value = {"test": "params"}
            
            params, coords = biotite.biotite_to_jax_md_system(self.atom_array, force_field="ff")
            
            self.assertEqual(params["test"], "params")
            self.assertEqual(coords.shape, (3, 3))
            mock_param.assert_called_once()

    def test_parse_biotite_generator(self):
        """Test generator output."""
        with mock.patch("priox.io.parsing.biotite.load_structure_with_hydride") as mock_load:
            mock_load.return_value = self.atom_array
            
            # Mock processed_structure_to_protein_tuples
            with mock.patch("priox.io.parsing.biotite.processed_structure_to_protein_tuples") as mock_convert:
                mock_convert.return_value = iter(["protein_tuple"])
                
                gen = biotite._parse_biotite("dummy.pdb", model=1, altloc="A", chain_id=None)
                result = list(gen)
                
                self.assertEqual(result, ["protein_tuple"])

if __name__ == "__main__":
    unittest.main()
