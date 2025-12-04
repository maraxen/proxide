"""Tests for Biotite parsing utilities."""

import numpy as np
import pytest
from unittest import mock
from biotite.structure import Atom, AtomArray, AtomArrayStack
from priox.io.parsing import biotite
from priox.io.parsing.structures import ProcessedStructure
from priox.chem import residues as rc
from priox.io.parsing.utils import (
    atom_array_dihedrals,
)

# 1UBQ PDB content (first model, residues 1-3 to ensure dihedrals can be calculated)
# MET 1, GLN 2, ILE 3
PDB_1UBQ_FRAG = """
ATOM      1  N   MET A   1      27.340  24.430   2.614  1.00  9.67           N
ATOM      2  CA  MET A   1      26.266  25.413   2.842  1.00 10.38           C
ATOM      3  C   MET A   1      26.913  26.639   3.531  1.00  9.62           C
ATOM      4  O   MET A   1      27.886  26.463   4.263  1.00  9.62           O
ATOM      5  CB  MET A   1      25.112  24.880   3.649  1.00 13.77           C
ATOM      6  CG  MET A   1      25.353  24.860   5.134  1.00 16.29           C
ATOM      7  SD  MET A   1      23.930  23.959   5.904  1.00 17.17           S
ATOM      8  CE  MET A   1      24.447  23.984   7.620  1.00 16.11           C
ATOM      9  N   GLN A   2      26.335  27.770   3.258  1.00  9.27           N
ATOM     10  CA  GLN A   2      26.850  29.021   3.898  1.00  9.07           C
ATOM     11  C   GLN A   2      26.100  29.253   5.202  1.00  8.72           C
ATOM     12  O   GLN A   2      24.865  29.024   5.330  1.00  9.13           O
ATOM     13  CB  GLN A   2      28.317  28.703   4.172  1.00 12.96           C
ATOM     14  CG  GLN A   2      29.537  28.318   3.270  1.00 16.92           C
ATOM     15  CD  GLN A   2      30.826  28.974   3.784  1.00 18.25           C
ATOM     16  OE1 GLN A   2      31.332  28.625   4.857  1.00 17.52           O
ATOM     17  NE2 GLN A   2      31.339  29.932   3.023  1.00 17.70           N
ATOM     18  N   ILE A   3      26.832  29.774   6.179  1.00  9.54           N
ATOM     19  CA  ILE A   3      26.230  30.158   7.451  1.00 10.37           C
ATOM     20  C   ILE A   3      26.963  31.428   7.842  1.00 10.34           C
ATOM     21  O   ILE A   3      27.817  31.896   7.078  1.00 10.70           O
ATOM     22  CB  ILE A   3      26.299  29.043   8.527  1.00 11.23           C
ATOM     23  CG1 ILE A   3      25.127  28.093   8.314  1.00 12.59           C
ATOM     24  CG2 ILE A   3      27.632  28.329   8.487  1.00 12.00           C
ATOM     25  CD1 ILE A   3      23.939  28.691   7.533  1.00 14.99           C
"""

def test_atom_array_dihedrals():
    """Test the atom_array_dihedrals function."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as tmp:
        tmp.write(PDB_1UBQ_FRAG)
        filepath = tmp.name
    # Use biotite.load_structure_with_hydride to get AtomArray directly
    atom_array = biotite.load_structure_with_hydride(filepath)
    dihedrals = atom_array_dihedrals(atom_array)

    # Biotite calculates dihedrals for internal residues.
    # For a 3-residue chain, only the middle residue has both Phi and Psi.
    # The first residue lacks Phi (no previous C).
    # The last residue lacks Psi (no next N).
    # omega is usually calculated for the bond between Res(i) and Res(i+1).

    # atom_array_dihedrals implementation in utils.py filters out NaN values:
    # clean_dihedrals = dihedrals[~np.any(np.isnan(dihedrals), axis=-1)]

    # So we expect only 1 valid set of dihedrals (for residue 2).

    assert dihedrals is not None
    assert len(dihedrals) == 1

def create_atom_array(atoms_list):
    # atoms_list: list of dicts with props
    length = len(atoms_list)
    array = AtomArray(length)

    # Ensure standard annotations
    for cat, dtype in [("chain_id", str), ("atom_name", str), ("res_name", str),
                       ("res_id", int), ("element", str), ("hetero", bool)]:
        if cat not in array.get_annotation_categories():
            array.add_annotation(cat, dtype)

    for i, props in enumerate(atoms_list):
        for k, v in props.items():
            if k == 'coord':
                array.coord[i] = v
            else:
                # To set a single element, we must index into the array returned by getattr
                getattr(array, k)[i] = v
    return array

def test_remove_solvent_from_structure():
    # Water
    water = Atom([0,0,0], res_name="HOH", element="O", hetero=True)
    # Protein
    prot = Atom([1,0,0], res_name="ALA", element="C", hetero=False)

    array = AtomArray(2)
    array[0] = prot
    array[1] = water

    # Verify pre-condition
    assert array.array_length() == 2

    # Remove solvent
    cleaned = biotite._remove_solvent_from_structure(array)

    assert cleaned.array_length() == 1
    assert cleaned[0].res_name == "ALA"

def test_add_hydrogens_to_structure_existing():
    # Has H already
    atom = Atom([0,0,0], element="H")
    array = AtomArray(1)
    array[0] = atom

    with mock.patch("hydride.add_hydrogen") as mock_add:
        result = biotite._add_hydrogens_to_structure(array)
        assert result is array
        mock_add.assert_not_called()

def test_add_hydrogens_to_structure_stack():
    # Stack
    array = AtomArrayStack(2, 1) # 2 frames, 1 atom
    array.element = ["C"]

    # Should warn and return as is
    result = biotite._add_hydrogens_to_structure(array)
    assert result is array

def test_add_hydrogens_to_structure_hydride():
    # No H
    atom = Atom([0,0,0], element="C", res_name="ALA", atom_name="CA", res_id=1)
    array = AtomArray(1)
    array[0] = atom

    # Mock hydride
    with mock.patch("hydride.add_hydrogen") as mock_add:
        # returns (array, mask)
        mock_add.return_value = (array, np.array([False]))

        # Also mock connect_via_residue_names to avoid errors if bonds missing
        with mock.patch("biotite.structure.connect_via_residue_names"):
             result = biotite._add_hydrogens_to_structure(array)

        mock_add.assert_called()
        assert "charge" in result.get_annotation_categories()

def test_fix_arg_protonation():
    # ARG with missing HH
    # Construct minimal ARG sidechain
    # NE at (0,0,0), CZ at (0,1,0), NH1 at (1,1,0), NH2 at (-1,1,0)

    atoms = [
        {"res_name": "ARG", "res_id": 1, "atom_name": "NE", "coord": [0,0,0], "element": "N", "chain_id": "A"},
        {"res_name": "ARG", "res_id": 1, "atom_name": "CZ", "coord": [0,1,0], "element": "C", "chain_id": "A"},
        {"res_name": "ARG", "res_id": 1, "atom_name": "NH1", "coord": [1,2,0], "element": "N", "chain_id": "A"},
        {"res_name": "ARG", "res_id": 1, "atom_name": "NH2", "coord": [-1,2,0], "element": "N", "chain_id": "A"},
        # Add HE to avoid checking it
        {"res_name": "ARG", "res_id": 1, "atom_name": "HE", "coord": [0,-1,0], "element": "H", "chain_id": "A"},
    ]
    array = create_atom_array(atoms)

    # Missing HH11, HH12, HH21, HH22

    fixed = biotite._fix_arg_protonation(array)

    names = fixed.atom_name
    assert "HH11" in names
    assert "HH12" in names
    assert "HH21" in names
    assert "HH22" in names
    assert fixed.array_length() == 5 + 4

def test_load_structure_with_hydride():
    # Mock structure_io.load_structure
    with mock.patch("biotite.structure.io.load_structure") as mock_load:
        # Need a valid AtomArray with coordinates
        array = AtomArray(1)
        array.coord[0] = [0,0,0]
        mock_load.return_value = array

        with mock.patch("priox.io.parsing.biotite._add_hydrogens_to_structure") as mock_add_h:
            mock_add_h.return_value = array

            # Test simple load
            biotite.load_structure_with_hydride("dummy.pdb", add_hydrogens=True)
            mock_load.assert_called()
            mock_add_h.assert_called()

            # Test remove solvent
            with mock.patch("priox.io.parsing.biotite._remove_solvent_from_structure") as mock_rem:
                 biotite.load_structure_with_hydride("dummy.pdb", remove_solvent=True)
                 mock_rem.assert_called()

def test_biotite_to_jax_md_system():
    # Mock AtomArray
    atoms = [
        {"res_name": "ALA", "atom_name": "N", "coord": [0,0,0]},
        {"res_name": "ALA", "atom_name": "CA", "coord": [1,0,0]},
    ]
    array = create_atom_array(atoms)

    mock_ff = mock.Mock()

    with mock.patch("priox.md.jax_md_bridge.parameterize_system") as mock_param:
        mock_param.return_value = {"bonds": []}

        params, coords = biotite.biotite_to_jax_md_system(array, mock_ff)

        assert coords.shape == (2, 3)
        mock_param.assert_called()
        # Verify call args
        args = mock_param.call_args
        # args[0] is force_field
        # args[1] is residues list: ["ALA"]
        # args[2] is atom_names: ["N", "CA"]
        # args[3] is atom_counts: [2]
        assert args[0][1] == ["ALA"]
        assert args[0][2] == ["N", "CA"]
        assert args[0][3] == [2]
