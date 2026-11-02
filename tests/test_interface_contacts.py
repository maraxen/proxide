"""Unit tests for proxide.geometry.contacts (positive, negative, and batching equivalence)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from proxide.geometry.contacts import (
    InterfaceContactResult,
    aligned_contact_overlap,
    associates,
    compute_buried_sasa,
    compute_interface_contacts,
    compute_interface_contacts_batch,
    compute_shrake_rupley,
    contact_jaccard,
    contacts_and_dock,
    contacts_and_dock_batch,
    interface_counts,
    load_aligned_pairs,
    screen_self_association,
    selftest_contacts,
    selftest_slice2,
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


# ==============================================================================
# Slice 2 Tests: Jaccard, Alignment Overlap, and Docking
# ==============================================================================


def test_contact_jaccard_positive_and_negative() -> None:
    """Test contact_jaccard across identical, partial, disjoint, and empty sets."""
    # Positive: partial overlap
    assert contact_jaccard({1, 2, 3}, {2, 3, 4}) == 2 / 4 == 0.5
    # Positive: identical sets
    assert contact_jaccard({1, 2, 3}, {1, 2, 3}) == 1.0
    # Positive: insertion code tuples
    assert contact_jaccard({(10, "A"), (11, "")}, {(10, "A"), (12, "")}) == 1 / 3
    # Negative: disjoint sets
    assert contact_jaccard({1, 2}, {3, 4}) == 0.0
    # Negative: both empty
    assert contact_jaccard(set(), set()) == 0.0
    # Negative: one empty
    assert contact_jaccard({1, 2}, set()) == 0.0


def test_aligned_contact_overlap_drops_unaligned() -> None:
    """Test aligned_contact_overlap projects onto ortholog A and drops unaligned residues."""
    aligned_pairs = [(200, 200), (205, 205)]
    # Human contacts include 212 which has no mouse partner in aligned_pairs
    human_contacts = {200, 212}
    mouse_contacts = {200}
    j, nh, nm = aligned_contact_overlap(human_contacts, mouse_contacts, aligned_pairs)
    assert j == 1.0
    assert nh == 1
    assert nm == 1

    # Disjoint aligned contacts
    j_disjoint, nh2, nm2 = aligned_contact_overlap({200}, {205}, aligned_pairs)
    assert j_disjoint == 0.0
    assert nh2 == 1
    assert nm2 == 1

    # Empty contacts
    j_empty, nh3, nm3 = aligned_contact_overlap(set(), set(), aligned_pairs)
    assert j_empty == 0.0
    assert nh3 == 0
    assert nm3 == 0


def test_load_aligned_pairs_text_and_file(tmp_path: Path) -> None:
    """Test load_aligned_pairs handles CSV text and files, validating header and integers."""
    csv_text = "human_resnum,mouse_resnum,in_crop\n189,191,1\n212,,1\n190,192,0\n191,193,True\n"
    pairs = load_aligned_pairs(csv_text)
    assert pairs == [(189, 191), (191, 193)]

    # From file path
    p = tmp_path / "numbering.csv"
    p.write_text(csv_text)
    assert load_aligned_pairs(p) == pairs

    # Empty and header-only return []
    assert load_aligned_pairs("") == []
    assert load_aligned_pairs("   \n  ") == []
    assert load_aligned_pairs("human_resnum,mouse_resnum,in_crop\n") == []

    # Missing required header column raises KeyError
    bad_header = "human_resnum,other_col\n1,2\n"
    with pytest.raises(KeyError, match="mouse_resnum"):
        load_aligned_pairs(bad_header)

    # Non-integer in passing row raises ValueError
    bad_int = "human_resnum,mouse_resnum,in_crop\nfoo,191,1\n"
    with pytest.raises(ValueError, match="Could not parse integer"):
        load_aligned_pairs(bad_int)

    # Unsupported type raises TypeError
    with pytest.raises(TypeError):
        load_aligned_pairs(123)  # type: ignore[arg-type]


def test_contacts_and_dock_synthetic_collinear() -> None:
    """Test contacts_and_dock with collinear synthetic atoms per spec §4.9.1."""
    # Target: 15 collinear atoms at 4.0 A spacing along x-axis
    target_coords = np.array([[i * 4.0, 0.0, 0.0] for i in range(15)])
    target_resids = np.array(list(range(1, 16)))
    epitope_residues = {1, 2, 3, 4, 5}

    # Positive binder: 12 atoms placed at y = 3.0 A near target atoms 0..11
    binder_pos = np.array([[i * 4.0, 3.0, 0.0] for i in range(12)])
    res_pos = contacts_and_dock(
        target_coords,
        target_resids,
        binder_pos,
        epitope_residues=epitope_residues,
        min_epitope_contacts=4,
        min_interface_residues=10,
    )
    assert res_pos["n_contacted"] == 13
    assert res_pos["epitope_contacts"] == 5
    assert res_pos["docked"] is True

    # Negative gate: far binder at y = 50.0 A
    binder_far = np.array([[i * 4.0, 50.0, 0.0] for i in range(12)])
    res_far = contacts_and_dock(target_coords, target_resids, binder_far, epitope_residues=epitope_residues)
    assert res_far["n_contacted"] == 0
    assert res_far["epitope_contacts"] == 0
    assert res_far["docked"] is False

    # Negative gate: epitope shortfall (atoms 8..14 near target, 0 epitope contacts)
    binder_short_epi = np.array([[i * 4.0, 3.0, 0.0] for i in range(8, 15)])
    res_short_epi = contacts_and_dock(
        target_coords, target_resids, binder_short_epi, epitope_residues=epitope_residues
    )
    assert res_short_epi["n_contacted"] == 8
    assert res_short_epi["epitope_contacts"] == 0
    assert res_short_epi["docked"] is False

    # Negative gate: interface count shortfall (3 atoms, 3 contacts < 10)
    binder_short_if = np.array([[i * 4.0, 3.0, 0.0] for i in range(3)])
    res_short_if = contacts_and_dock(
        target_coords, target_resids, binder_short_if, epitope_residues=epitope_residues
    )
    assert res_short_if["n_contacted"] == 4
    assert res_short_if["docked"] is False


def test_contacts_and_dock_invariants_and_batch_equivalence() -> None:
    """Test contacts_and_dock error boundaries and contacts_and_dock_batch equivalence."""
    target_coords = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    target_resids = [1, 2]

    # Empty target raises ValueError
    with pytest.raises(ValueError, match="target_coords cannot be empty"):
        contacts_and_dock(np.zeros((0, 3)), [], np.zeros((1, 3)))

    # Length mismatch raises ValueError
    with pytest.raises(ValueError, match="Length mismatch"):
        contacts_and_dock(target_coords, [1], np.zeros((1, 3)))

    # Empty binder returns zeroed dict
    res_empty = contacts_and_dock(target_coords, target_resids, np.zeros((0, 3)))
    assert res_empty == {"contacted": [], "n_contacted": 0, "epitope_contacts": 0, "docked": False}

    # Batching equivalence: batched == looped
    b1 = np.array([[0.0, 3.0, 0.0]])
    b2 = np.array([[4.0, 3.0, 0.0]])
    b3 = np.zeros((0, 3))
    binders = [b1, b2, b3]

    batched = contacts_and_dock_batch(target_coords, target_resids, binders)
    looped = [contacts_and_dock(target_coords, target_resids, b) for b in binders]
    assert batched == looped


# ==============================================================================
# Slice 2 Tests: Self-Association (SA) Homodimer Screen
# ==============================================================================


def test_interface_counts_and_associates() -> None:
    """Test interface_counts on synthetic geometry and associates decision rule."""
    chain_a = {"xyz": np.array([[i * 3.8, 0.0, 0.0] for i in range(20)]), "resid": np.arange(1, 21)}
    chain_b_near = {"xyz": np.array([[i * 3.8, 1.0, 0.0] for i in range(20)]), "resid": np.arange(1, 21)}
    chain_b_far = {"xyz": np.array([[i * 3.8, 500.0, 0.0] for i in range(20)]), "resid": np.arange(1, 21)}

    na_near, nb_near = interface_counts(chain_a, chain_b_near)
    assert na_near >= 12 and nb_near >= 12
    na_far, nb_far = interface_counts(chain_a, chain_b_far)
    assert na_far == 0 and nb_far == 0

    # Test associates rule
    assert associates(20, 20, 0.65) is True
    assert associates(20, 5, 0.65) is False  # chain B shortfall
    assert associates(5, 20, 0.65) is False  # chain A shortfall
    assert associates(20, 20, 0.20) is False  # ipsae shortfall
    assert associates(20, 20, None) is False  # None ipsae
    assert associates(20, 20, float("nan")) is False  # NaN ipsae


def test_screen_self_association_3tier_outcomes() -> None:
    """Test screen_self_association 3-tier outcome partitions and control validation."""
    base_pos = [{"design_id": "pos", "group": "pos_ctl", "valid": 1, "n_if_a": 18, "n_if_b": 18, "ipsae": 0.5}]
    base_neg = [{"design_id": "neg", "group": "neg_ctl", "valid": 1, "n_if_a": 16, "n_if_b": 16, "ipsae": 0.1}]

    # 1. Monomeric outcome: design_rate <= 0.20
    design_monomeric = [
        {"design_id": f"d_{i}", "group": "design", "valid": 1, "n_if_a": 5, "n_if_b": 5, "ipsae": 0.1}
        for i in range(10)
    ]
    res_m = screen_self_association(base_pos + base_neg + design_monomeric)
    assert res_m["controls_ok"] is True
    assert res_m["design_rate"] == 0.0
    assert res_m["outcome"] == "monomeric"

    # 2. Self_associates outcome: design_rate >= 0.50
    design_associating = [
        {"design_id": f"d_{i}", "group": "design", "valid": 1, "n_if_a": 20, "n_if_b": 20, "ipsae": 0.6}
        for i in range(10)
    ]
    res_sa = screen_self_association(base_pos + base_neg + design_associating)
    assert res_sa["controls_ok"] is True
    assert res_sa["design_rate"] == 1.0
    assert res_sa["outcome"] == "self_associates"

    # 3. Inconclusive outcome: intermediate rate (e.g. 0.35)
    design_inter = (
        [{"design_id": f"d_{i}", "group": "design", "valid": 1, "n_if_a": 20, "n_if_b": 20, "ipsae": 0.6} for i in range(3)]
        + [{"design_id": f"d_{i}", "group": "design", "valid": 1, "n_if_a": 5, "n_if_b": 5, "ipsae": 0.1} for i in range(7)]
    )
    res_inconcl = screen_self_association(base_pos + base_neg + design_inter)
    assert res_inconcl["controls_ok"] is True
    assert res_inconcl["design_rate"] == 0.3
    assert res_inconcl["outcome"] == "inconclusive"

    # 4. Control failure: positive control fails to associate
    base_pos_fail = [{"design_id": "pos", "group": "pos_ctl", "valid": 1, "n_if_a": 5, "n_if_b": 5, "ipsae": 0.1}]
    res_ctl_fail = screen_self_association(base_pos_fail + base_neg + design_monomeric)
    assert res_ctl_fail["controls_ok"] is False
    assert res_ctl_fail["outcome"] == "inconclusive"

    # 5. Invalid run handling: invalid run recorded in problems and excluded from denominator
    runs_with_invalid = [
        {"design_id": "pos", "group": "pos_ctl", "valid": 1, "n_if_a": 18, "n_if_b": 18, "ipsae": 0.5},
        {"design_id": "neg", "group": "neg_ctl", "valid": 1, "n_if_a": 16, "n_if_b": 16, "ipsae": 0.1},
        {"design_id": "bad", "group": "design", "valid": 0, "problem": "cif parse failure"},
        {"design_id": "d1", "group": "design", "valid": 1, "n_if_a": 5, "n_if_b": 5, "ipsae": 0.1},
    ]
    res_inv = screen_self_association(runs_with_invalid)
    assert res_inv["n_problems"] == 1
    assert "bad: cif parse failure" in res_inv["problems"][0]
    assert res_inv["rows_complete"] is False


# ==============================================================================
# Slice 2 Tests: Shrake-Rupley SASA and Buried Surface Area
# ==============================================================================


def test_compute_shrake_rupley_isolated_carbon() -> None:
    """Test compute_shrake_rupley on single isolated carbon atom yields ~120.76 A^2."""
    coords = np.array([[0.0, 0.0, 0.0]])
    elements = ["C"]
    total_sasa, per_atom = compute_shrake_rupley(coords, elements, probe_radius=1.40, n_points=100)
    expected_area = 4.0 * np.pi * (1.70 + 1.40) ** 2  # 120.7628 A^2
    assert np.isclose(total_sasa, expected_area, atol=1e-2)
    assert np.isclose(per_atom[0], expected_area, atol=1e-2)

    # Empty coords
    tot_empty, arr_empty = compute_shrake_rupley(np.zeros((0, 3)), [])
    assert tot_empty == 0.0
    assert len(arr_empty) == 0

    # Alpha carbon element resolution: "CA" resolves to Carbon (1.70 A), not Calcium (2.31 A)
    tot_ca, _ = compute_shrake_rupley(coords, ["CA"], probe_radius=1.40, n_points=100)
    assert np.isclose(tot_ca, expected_area, atol=1e-2)


def test_compute_buried_sasa_and_s1_invariant() -> None:
    """Test compute_buried_sasa and enforcement of physical invariant S1."""
    # Free binder: 2 carbons at 3.0 A spacing
    free_coords = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    free_elements = ["C", "C"]
    free_resids = [1, 2]

    # Target placed close (distance 2.0 A) in complex
    target_coords = np.array([[0.0, 2.0, 0.0], [3.0, 2.0, 0.0]])
    target_elements = ["C", "C"]
    target_resids = [10, 11]

    complex_coords = np.vstack([free_coords, target_coords])
    complex_elements = free_elements + target_elements
    complex_resids = free_resids + target_resids

    delta_float, delta_int, details = compute_buried_sasa(
        free_coords, free_elements, free_resids, complex_coords, complex_elements, complex_resids
    )
    assert delta_float > 0.0
    assert delta_int == round(delta_float)
    assert details["free_sasa"] > details["complex_sasa"]

    # Invariant S1 check: if complex SASA exceeds free SASA, raises ValueError
    # Create artificial structure where target is absent but complex is evaluated with negative buried area
    with pytest.raises(ValueError, match="complex SASA exceeds free SASA"):
        # artifically pass smaller free coords and larger complex coords to trigger S1
        far_free = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # tightly packed free
        sparse_complex = np.array([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0]])  # sparse complex has more SASA
        compute_buried_sasa(
            far_free, ["C", "C"], [1, 2], sparse_complex, ["C", "C"], [1, 2]
        )


def test_compute_interface_contacts_with_sasa(tmp_path: Path) -> None:
    """Test compute_interface_contacts populates buried_binder_a2 when compute_sasa=True."""
    pdb_lines = [
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C",
        "ATOM      2  CA  ALA B   2       3.000   0.000   0.000  1.00 20.00           C",
    ]
    pdb_text = "\n".join(pdb_lines) + "\n"

    # With compute_sasa=False (default)
    res_no_sasa = compute_interface_contacts(pdb_text, chain_a="A", chain_b="B", compute_sasa=False)
    assert res_no_sasa.buried_binder_a2 is None

    # With compute_sasa=True
    res_sasa = compute_interface_contacts(pdb_text, chain_a="A", chain_b="B", compute_sasa=True)
    assert res_sasa.buried_binder_a2 is not None
    assert isinstance(res_sasa.buried_binder_a2, int)
    assert "buried_binder_a2_float" in res_sasa.details
    assert "sasa_details" in res_sasa.details


def test_contacts_and_dock_default_epitope_and_target_offset() -> None:
    """Test contacts_and_dock with epitope_residues=None yields docked=True when n_contacted >= 10,
    and verify target_offset is applied correctly.
    """
    target_coords = np.array([[float(i) * 3.0, 0.0, 0.0] for i in range(20)])
    target_resids = list(range(1, 21))
    binder_coords = np.array([[float(i) * 3.0, 2.0, 0.0] for i in range(20)])

    # With epitope_residues=None (default) and n_contacted >= 10 -> docked must be True
    res = contacts_and_dock(target_coords, target_resids, binder_coords, min_interface_residues=10)
    assert res["n_contacted"] == 20
    assert res["epitope_contacts"] == 0
    assert res["docked"] is True

    # With target_offset non-zero (e.g. +188)
    res_offset = contacts_and_dock(
        target_coords, target_resids, binder_coords, target_offset=188, min_interface_residues=10
    )
    assert res_offset["docked"] is True
    assert res_offset["contacted"] == [i + 188 for i in range(1, 21)]

    # Batch equivalence with target_offset != 0 and epitope_residues
    binders = [binder_coords, np.zeros((0, 3))]
    batched = contacts_and_dock_batch(
        target_coords, target_resids, binders, target_offset=188, epitope_residues={189, 190, 191, 192}
    )
    looped = [
        contacts_and_dock(
            target_coords, target_resids, b, target_offset=188, epitope_residues={189, 190, 191, 192}
        )
        for b in binders
    ]
    assert batched == looped
    assert batched[0]["docked"] is True
    assert batched[0]["epitope_contacts"] == 4


def test_aligned_contact_overlap_empty_pairs() -> None:
    """Test aligned_contact_overlap returns (0.0, 0, 0) when aligned_pairs is empty."""
    res = aligned_contact_overlap({1, 2, 3}, {101, 102, 103}, [])
    assert res == (0.0, 0, 0)


def test_load_aligned_pairs_ragged_csv_and_errors(tmp_path: Path) -> None:
    """Test load_aligned_pairs safely handles ragged lines and raises FileNotFoundError."""
    # Ragged CSV with missing columns on some rows
    ragged_csv = (
        "human_resnum,mouse_resnum,in_crop,aligned\n"
        "1,101,1,1\n"
        "2\n"  # ragged row where mouse_resnum, in_crop, aligned are None
        "3,103,1,1\n"
    )
    pairs = load_aligned_pairs(ragged_csv, in_crop_col="in_crop", aligned_col="aligned")
    assert pairs == [(1, 101), (3, 103)]

    # Non-existent file path
    with pytest.raises(FileNotFoundError):
        load_aligned_pairs(tmp_path / "does_not_exist.csv")


def test_interface_counts_empty_inputs() -> None:
    """Test interface_counts returns (0, 0) for empty lists, dicts, or arrays."""
    assert interface_counts([], []) == (0, 0)
    assert interface_counts({}, {}) == (0, 0)
    assert interface_counts(np.zeros((0, 3)), np.zeros((0, 3))) == (0, 0)
    assert interface_counts([], {"xyz": np.zeros((1, 3)), "resid": [1]}) == (0, 0)


def test_screen_self_association_negative_control_failure_and_options() -> None:
    """Test negative control failure forces outcome='inconclusive', expected_runs_per_design, and NaN ipsae."""
    # Negative control failure (n_neg_ctl_assoc > 0)
    runs_neg_fail = [
        {"design_id": "pos", "group": "pos_ctl", "valid": 1, "n_if_a": 18, "n_if_b": 18, "ipsae": 0.5},
        {"design_id": "neg", "group": "neg_ctl", "valid": 1, "n_if_a": 16, "n_if_b": 16, "ipsae": 0.5},
        {"design_id": "d1", "group": "design", "valid": 1, "n_if_a": 5, "n_if_b": 5, "ipsae": 0.1},
    ]
    res_neg_fail = screen_self_association(runs_neg_fail)
    assert res_neg_fail["controls_ok"] is False
    assert res_neg_fail["n_neg_ctl_assoc"] == 1
    assert res_neg_fail["outcome"] == "inconclusive"

    # NaN / None / "" ipsae handling in runs
    runs_nan_ipsae = [
        {"design_id": "pos", "group": "pos_ctl", "valid": 1, "n_if_a": 18, "n_if_b": 18, "ipsae": 0.5},
        {"design_id": "neg", "group": "neg_ctl", "valid": 1, "n_if_a": 16, "n_if_b": 16, "ipsae": 0.1},
        {"design_id": "d1", "group": "design", "valid": 1, "n_if_a": 20, "n_if_b": 20, "ipsae": float("nan")},
        {"design_id": "d2", "group": "design", "valid": 1, "n_if_a": 20, "n_if_b": 20, "ipsae": None},
        {"design_id": "d3", "group": "design", "valid": 1, "n_if_a": 20, "n_if_b": 20, "ipsae": ""},
    ]
    res_nan = screen_self_association(runs_nan_ipsae)
    assert res_nan["n_design_assoc"] == 0
    assert res_nan["design_rate"] == 0.0

    # expected_runs_per_design parameter
    designs = {"d1": {}, "d2": {}}
    runs_1per = [
        {"design_id": "d1", "group": "design", "valid": 1, "n_if_a": 5, "n_if_b": 5, "ipsae": 0.1},
        {"design_id": "d2", "group": "design", "valid": 1, "n_if_a": 5, "n_if_b": 5, "ipsae": 0.1},
    ]
    # With default expected_runs_per_design=2: n_runs (2) != 2 * 2 (4), so rows_complete is False
    res_2per = screen_self_association(runs_1per, designs=designs)
    assert res_2per["rows_complete"] is False

    # With expected_runs_per_design=1: n_runs (2) == 1 * 2 (2), so rows_complete is True
    res_1per = screen_self_association(runs_1per, designs=designs, expected_runs_per_design=1)
    assert res_1per["rows_complete"] is True

    # With expected_runs_per_design=None: rows_complete evaluates solely on n_runs > 0 and no problems
    res_none = screen_self_association(runs_1per, designs=designs, expected_runs_per_design=None)
    assert res_none["rows_complete"] is True


def test_compute_shrake_rupley_and_buried_sasa_validations_and_infinite_separation() -> None:
    """Test length validations in SASA functions, infinite separation yielding delta_sasa == 0.0,
    and batching equivalence with compute_sasa=True.
    """
    # compute_shrake_rupley length mismatch
    with pytest.raises(ValueError, match="Length mismatch"):
        compute_shrake_rupley(np.zeros((5, 3)), ["C", "C"])

    # compute_buried_sasa coordinate array length mismatches
    c3 = np.zeros((3, 3))
    c2 = np.zeros((2, 3))
    with pytest.raises(ValueError, match="Length mismatch: free_coords"):
        compute_buried_sasa(c3, ["C", "C"], [1, 2], c3, ["C"] * 3, [1, 2, 3])
    with pytest.raises(ValueError, match="Length mismatch: complex_coords"):
        compute_buried_sasa(c2, ["C", "C"], [1, 2], c3, ["C", "C"], [1, 2, 3])
    with pytest.raises(ValueError, match="cannot have fewer atoms than free_coords"):
        compute_buried_sasa(c3, ["C"] * 3, [1, 2, 3], c2, ["C"] * 2, [1, 2])

    # Infinitely separated chains (offset +1000 A) yield delta_sasa == 0.0
    free_coords = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    tgt_far = np.array([[0.0, 1000.0, 0.0], [3.0, 1000.0, 0.0]])
    comp_far = np.vstack([free_coords, tgt_far])
    delta_float, delta_int, details = compute_buried_sasa(
        free_coords, ["C", "C"], [1, 2], comp_far, ["C", "C", "C", "C"], [1, 2, 10, 11]
    )
    assert abs(delta_float) < 1e-6
    assert delta_int == 0
    assert abs(details["free_sasa"] - details["complex_sasa"]) < 1e-6

    # compute_interface_contacts_batch equivalence with compute_sasa=True
    atoms_near = [
        {"chain": "A", "name": "CA", "res": "ALA", "num": 1, "xyz": (0.0, 0.0, 0.0), "element": "C"},
        {"chain": "B", "name": "CA", "res": "ALA", "num": 2, "xyz": (3.0, 0.0, 0.0), "element": "C"},
    ]
    atoms_far = [
        {"chain": "A", "name": "CA", "res": "ALA", "num": 1, "xyz": (0.0, 0.0, 0.0), "element": "C"},
        {"chain": "B", "name": "CA", "res": "ALA", "num": 2, "xyz": (1000.0, 0.0, 0.0), "element": "C"},
    ]
    structures = [atoms_near, atoms_far]
    batched = compute_interface_contacts_batch(structures, compute_sasa=True)
    looped = [compute_interface_contacts(s, compute_sasa=True) for s in structures]
    assert batched == looped
    assert batched[0].buried_binder_a2 is not None and batched[0].buried_binder_a2 > 0
    assert batched[1].buried_binder_a2 == 0


def test_selftest_slice2_executes_and_passes() -> None:
    """Test that selftest_slice2 runs synthetic tests in-memory and returns all_pass=True."""
    res = selftest_slice2()
    assert res["all_pass"] is True
    assert res["jaccard_pass"] is True
    assert res["overlap_pass"] is True
    assert res["dock_pass"] is True
    assert res["sa_pass"] is True
    assert res["sasa_pass"] is True
    assert selftest_contacts() == res

