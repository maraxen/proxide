"""Unit tests for proxide.geometry.contacts (positive, negative, and batching equivalence)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from proxide.geometry.contacts import (
    InterfaceContactResult,
    compute_interface_contacts,
    compute_interface_contacts_batch,
)


def _mk_atom(chain: str, res: str, name: str, num: int, xyz: tuple[float, float, float]) -> dict[str, Any]:
    return {"chain": chain, "res": res, "name": name, "num": num, "xyz": xyz}


def test_positive_selftest_ion_pair() -> None:
    """Positive selftest: LYS:NZ at (0,0,0) and GLU:OE1 at (3,0,0) registers ion_pairs == 1."""
    atoms = [
        _mk_atom("A", "LYS", "NZ", 1, (0.0, 0.0, 0.0)),
        _mk_atom("B", "GLU", "OE1", 10, (3.0, 0.0, 0.0)),
    ]
    res = compute_interface_contacts(atoms, chain_a="A", chain_b="B")
    assert res.ion_pairs == 1
    assert res.like_close == 0
    assert res.polar_pairs == 1  # Cross-chain N...O <= 3.5 A is also polar
    assert res.nonpolar_pairs == 0
    assert res.n_binder_interface == 1
    assert res.n_target_interface == 1
    assert res.interface_residues_a == [1]
    assert res.interface_residues_b == [10]
    assert res.binder_if_charge == 1
    assert res.target_if_charge == -1


def test_positive_selftest_polar_pair() -> None:
    """Positive selftest: SER:OG at (0,0,0) and LYS:NZ at (3,0,0) registers polar_pairs == 1."""
    atoms = [
        _mk_atom("A", "SER", "OG", 2, (0.0, 0.0, 0.0)),
        _mk_atom("B", "LYS", "NZ", 20, (3.0, 0.0, 0.0)),
    ]
    res = compute_interface_contacts(atoms, chain_a="A", chain_b="B")
    assert res.polar_pairs == 1
    assert res.ion_pairs == 0
    assert res.like_close == 0
    assert res.nonpolar_pairs == 0


def test_positive_selftest_nonpolar_pair() -> None:
    """Positive selftest: LEU:CD1 at (0,0,0) and ILE:CG2 at (4,0,0) registers nonpolar_pairs == 1."""
    atoms = [
        _mk_atom("A", "LEU", "CD1", 3, (0.0, 0.0, 0.0)),
        _mk_atom("B", "ILE", "CG2", 30, (4.0, 0.0, 0.0)),
    ]
    res = compute_interface_contacts(atoms, chain_a="A", chain_b="B")
    assert res.nonpolar_pairs == 1
    assert res.ion_pairs == 0
    assert res.polar_pairs == 0
    assert res.like_close == 0
    assert res.binder_if_frac_hydrophobic == 1.0


def test_input_representations_pdb_text_and_file(tmp_path: Path) -> None:
    """Test compute_interface_contacts accepts atom dicts, PDB text string, and Path object."""
    pdb_lines = [
        "ATOM      1  NZ  LYS A   1       0.000   0.000   0.000  1.00 20.00           N",
        "ATOM      2  OE1 GLU B  10       3.000   0.000   0.000  1.00 20.00           O",
    ]
    pdb_text = "\n".join(pdb_lines) + "\n"

    # From string
    res_str = compute_interface_contacts(pdb_text, chain_a="A", chain_b="B")
    assert res_str.ion_pairs == 1

    # From file Path
    pdb_file = tmp_path / "test.pdb"
    pdb_file.write_text(pdb_text)
    res_path = compute_interface_contacts(pdb_file, chain_a="A", chain_b="B")
    assert res_path == res_str

    # Test to_dict()
    d = res_path.to_dict()
    assert isinstance(d, dict)
    assert d["ion_pairs"] == 1


def test_negative_selftest_ion_pair_distance_gate() -> None:
    """Negative selftest: LYS:NZ at (0,0,0) and GLU:OE1 at (4.5,0,0) registers ion_pairs == 0."""
    atoms = [
        _mk_atom("A", "LYS", "NZ", 1, (0.0, 0.0, 0.0)),
        _mk_atom("B", "GLU", "OE1", 10, (4.5, 0.0, 0.0)),
    ]
    res = compute_interface_contacts(atoms, chain_a="A", chain_b="B")
    assert res.ion_pairs == 0


def test_negative_selftest_like_charge_repulsive_gate() -> None:
    """Negative selftest: GLU:OE1 at (0,0,0) and ASP:OD1 at (4.0,0,0) registers like_close == 1 and ion_pairs == 0."""
    atoms = [
        _mk_atom("A", "GLU", "OE1", 1, (0.0, 0.0, 0.0)),
        _mk_atom("B", "ASP", "OD1", 10, (4.0, 0.0, 0.0)),
    ]
    res = compute_interface_contacts(atoms, chain_a="A", chain_b="B")
    assert res.like_close == 1
    assert res.ion_pairs == 0


def test_negative_selftest_backbone_carbon_exclusion() -> None:
    """Negative selftest: LEU:CA at (0,0,0) and ILE:CB at (4.0,0,0) registers nonpolar_pairs == 0."""
    atoms = [
        _mk_atom("A", "LEU", "CA", 1, (0.0, 0.0, 0.0)),
        _mk_atom("B", "ILE", "CB", 10, (4.0, 0.0, 0.0)),
    ]
    res = compute_interface_contacts(atoms, chain_a="A", chain_b="B")
    assert res.nonpolar_pairs == 0


def test_negative_selftest_distant_chains() -> None:
    """Negative selftest: chains separated by 500 A register 0 interface residues and contacts."""
    atoms = [
        _mk_atom("A", "LYS", "NZ", 1, (0.0, 0.0, 0.0)),
        _mk_atom("B", "GLU", "OE1", 10, (500.0, 0.0, 0.0)),
    ]
    res = compute_interface_contacts(atoms, chain_a="A", chain_b="B")
    assert res.n_binder_interface == 0
    assert res.n_target_interface == 0
    assert res.ion_pairs == 0
    assert res.like_close == 0
    assert res.polar_pairs == 0
    assert res.nonpolar_pairs == 0
    assert res.binder_if_frac_ek == 0.0
    assert res.binder_if_frac_hydrophobic == 0.0


def test_negative_selftest_error_handling() -> None:
    """Negative selftest: empty structure, missing chains, and missing file raise expected errors."""
    with pytest.raises(ValueError, match="contains no atoms"):
        compute_interface_contacts([])

    atoms_a_only = [_mk_atom("A", "ALA", "CA", 1, (0.0, 0.0, 0.0))]
    with pytest.raises(KeyError, match="Chain 'B' not found"):
        compute_interface_contacts(atoms_a_only, chain_a="A", chain_b="B")

    with pytest.raises(KeyError, match="Chain 'Z' not found"):
        compute_interface_contacts(atoms_a_only, chain_a="Z", chain_b="A")

    with pytest.raises(FileNotFoundError):
        compute_interface_contacts(Path("/nonexistent/structure.pdb"))


def test_batched_equals_looped_equivalence() -> None:
    """Batch equivalence test: compute_interface_contacts_batch matches looped calls."""
    struct1 = [
        _mk_atom("A", "LYS", "NZ", 1, (0.0, 0.0, 0.0)),
        _mk_atom("B", "GLU", "OE1", 10, (3.0, 0.0, 0.0)),
    ]
    struct2 = [
        _mk_atom("A", "LEU", "CD1", 2, (0.0, 0.0, 0.0)),
        _mk_atom("B", "ILE", "CG2", 20, (4.0, 0.0, 0.0)),
    ]
    structs = [struct1, struct2]

    batched = compute_interface_contacts_batch(structs)
    looped = [compute_interface_contacts(s) for s in structs]

    assert len(batched) == len(looped)
    for b, l in zip(batched, looped):
        assert b == l


def test_distinct_chains_validation() -> None:
    """Distinct chain validation: chain_a == chain_b raises ValueError."""
    atoms = [
        _mk_atom("A", "LYS", "NZ", 1, (0.0, 0.0, 0.0)),
        _mk_atom("A", "GLU", "OE1", 10, (3.0, 0.0, 0.0)),
    ]
    with pytest.raises(ValueError, match="Chains must be distinct"):
        compute_interface_contacts(atoms, chain_a="A", chain_b="A")


def test_endmdl_record_stops_parsing() -> None:
    """Multi-model PDB files stop at ENDMDL to prevent atom duplication."""
    pdb_text = """MODEL        1
ATOM      1  NZ  LYS A   1       0.000   0.000   0.000  1.00 20.00           N
ATOM      2  OE1 GLU B  10       3.000   0.000   0.000  1.00 20.00           O
ENDMDL
MODEL        2
ATOM      1  NZ  LYS A   1       0.000   0.000   0.000  1.00 20.00           N
ATOM      2  OE1 GLU B  10       3.000   0.000   0.000  1.00 20.00           O
ENDMDL
"""
    res = compute_interface_contacts(pdb_text, chain_a="A", chain_b="B")
    assert res.ion_pairs == 1
    assert res.polar_pairs == 1
    assert res.n_binder_interface == 1
    assert res.n_target_interface == 1


def test_solvent_water_filtering() -> None:
    """Solvent water residues (HOH, WAT, TIP3, H2O) are excluded from interface residues."""
    pdb_text = """ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C
ATOM      2  O   HOH A 100       2.000   0.000   0.000  1.00 20.00           O
ATOM      3  CA  ALA B   2      10.000   0.000   0.000  1.00 20.00           C
ATOM      4  O   WAT B 200       8.000   0.000   0.000  1.00 20.00           O
"""
    res = compute_interface_contacts(pdb_text, chain_a="A", chain_b="B")
    assert res.n_binder_interface == 0
    assert res.n_target_interface == 0


def test_hydrogen_atom_filtering() -> None:
    """Hydrogens are excluded so heavy-atom distance cutoffs are strictly respected."""
    pdb_text = """ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C
ATOM      2  H   ALA A   1       4.000   0.000   0.000  1.00 20.00           H
ATOM      3  CA  ALA B   2       5.500   0.000   0.000  1.00 20.00           C
"""
    res = compute_interface_contacts(pdb_text, chain_a="A", chain_b="B")
    assert res.n_binder_interface == 0
    assert res.n_target_interface == 0


def test_insertion_code_preservation() -> None:
    """Insertion codes are preserved in residue identifiers and prevent key collisions."""
    pdb_text = """ATOM      1  CA  ALA A 100       0.000   0.000   0.000  1.00 20.00           C
ATOM      2  CA  GLU A 100A      1.500   0.000   0.000  1.00 20.00           C
ATOM      3  CA  LYS B 200       3.000   0.000   0.000  1.00 20.00           C
"""
    res = compute_interface_contacts(pdb_text, chain_a="A", chain_b="B")
    assert res.details["binder_len"] == 2
    assert res.n_binder_interface == 2
    assert res.interface_residues_a == [100, (100, "A")]
    assert res.details["interface_residues_a_types"] == {100: "ALA", (100, "A"): "GLU"}

