"""Tests for Biotite parsing utilities."""

import numpy as np
import pytest
from unittest import mock
from biotite.structure import Atom, AtomArray, AtomArrayStack
from priox.io.parsing import biotite

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
