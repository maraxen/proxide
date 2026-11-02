"""Parity verification tests for registered bathos runs dda7de75 and bba33e3a."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from proxide.geometry.contacts import (
    RESIDUE_CHARGE,
    compute_interface_contacts,
)
from proxide.seqqc import AA, screen

# Table 5.1 Golden SHA256 hashes
FIXTURE_HASHES: dict[str, str] = {
    "tfr1_final_slate.csv": "346c70e21e626c33fa803f18346a1b5b7caebb3615766e36b58670ff45671b5f",
    "tfr1_f2_candidates.tsv": "fc4c470cf19b6ca7ea692c99dc6d2a0f758fce317c176d1557f135b464a5fa88",
    "tfr1_final_order.tsv": "bc5b1570564b2847c457b9cb056d70192ad065755a8d25d7dd5758f41d4b2003",
    "composition_result.json": "66377549e3564c7287eb599bee4464523b0cdf796b9a48b00c6fbf6b608f91a6",
    "tfr1_composition.tsv": "681506b5bdafe439fe24705726bd270671bd2d289c3c30203deca14657805692",
    "interface_result.json": "109e0a6a65897612be337180ba9027891357c10eb81745625eaaa53e7a893b79",
    "tfr1_interface.tsv": "f90040b53322d346a4a7847a1ee025618877288a924e5b779526e7585199ba6c",
}

STRUCTURE_HASHES: dict[str, str] = {
    "f2c006r13.pdb": "35b33d416edc6b6991765a6b74082c9d1593a2cce68b42c4412e9b89d0a10ab9",
    "f2c008r18.pdb": "cc6c71cbd338809d5def07d78e9a609b418e422852cafad714c0b76dfea0e001",
    "f2c016r5_H32.pdb": "816fd52830793b5b600dacc831ba0ab6fc455911032fa0dd4f9e9820680cc073",
    "f2c020r4_H62.pdb": "0802ec32057fa83af2302a9b03fbaf087a45b34e8edd375524a64bfbb4a0e1b2",
    "f2c025r15_H60.pdb": "c6004dd2bf964c98f30880e4513c44c1ec38d8acd6875b13c1afa2c1de78d5fd",
}

# Decoupling rule per spec §5.3: TfR1 motif definition resides in test suite
MOTIF: set[int] = {n + 188 for n in list(range(14, 28)) + [160, 183]}


def _resolve_fixture_dir() -> Path:
    """Resolve fixture directory with multiple fallback locations."""
    candidates = [
        Path(__file__).parent / "fixtures" / "tfr1",
        Path(__file__).parent.parent / "fixtures" / "tfr1",
        Path("/home/marielle/projects/proxide/.agents/teamwork/ref/outputs/tfr1"),
        Path("/home/marielle/projects/adaptyv/.claude/worktrees/261023-tfr1-design-dag/outputs/tfr1"),
    ]
    for c in candidates:
        if c.exists() and (c / "tfr1_final_slate.csv").exists():
            return c
    raise FileNotFoundError("Could not locate TfR1 test fixtures directory")


@pytest.fixture(autouse=True)
def check_fixture_hashes() -> None:
    """Pre-verification: assert all fixture files match golden SHA256 hashes before assertions."""
    fix_dir = _resolve_fixture_dir()
    for fname, expected_hash in FIXTURE_HASHES.items():
        file_path = fix_dir / fname
        if not file_path.exists():
            # Check nested subdirectories (e.g. final/, composition/, interface/)
            matching = list(fix_dir.glob(f"**/{fname}"))
            assert len(matching) > 0, f"Fixture file not found: {fname}"
            file_path = matching[0]
        actual_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
        assert actual_hash == expected_hash, f"Hash mismatch for {fname}: got {actual_hash}, expected {expected_hash}"

    # Also verify structure hashes
    struct_dir = fix_dir / "structures"
    if not struct_dir.exists():
        struct_dir = fix_dir / "f3" / "structures"
    assert struct_dir.exists(), "Structure directory not found"
    for sname, expected_hash in STRUCTURE_HASHES.items():
        spath = struct_dir / sname
        assert spath.exists(), f"Structure file not found: {sname}"
        actual_hash = hashlib.sha256(spath.read_bytes()).hexdigest()
        assert actual_hash == expected_hash, f"Hash mismatch for {sname}: got {actual_hash}, expected {expected_hash}"


def test_fixture_hashes_match_spec_table_5_1() -> None:
    """Explicit test function ensuring fixture hashes pass under direct invocation."""
    pass


def test_bathos_parity_dda7de75_composition() -> None:
    """Parity verification for dda7de75 (composition analysis of full pool and final slate).

    Reproduces exact registered outputs from composition_result.json and all rows of tfr1_composition.tsv.
    """
    fix_dir = _resolve_fixture_dir()
    slate_file = fix_dir / "tfr1_final_slate.csv"
    if not slate_file.exists():
        slate_file = next(fix_dir.glob("**/tfr1_final_slate.csv"))

    rows = list(csv.DictReader(slate_file.open()))
    assert len(rows) == 5, f"Expected 5 final designs, got {len(rows)}"
    final_seqs = [r["sequence"] for r in rows]

    # Load candidate pool
    candidates_file = fix_dir / "tfr1_f2_candidates.tsv"
    if not candidates_file.exists():
        candidates_file = next(fix_dir.glob("**/tfr1_f2_candidates.tsv"))
    cand_rows = list(csv.DictReader(candidates_file.open(), delimiter="\t"))
    assert len(cand_rows) == 594, f"Expected 594 candidates, got {len(cand_rows)}"

    pool_all = [r["best_seq"] for r in cand_rows]
    pool_cond = [r["best_seq"] for r in cand_rows if r["conditioned"] == "1"]
    pool_uncond = [r["best_seq"] for r in cand_rows if r["conditioned"] == "0"]
    pool_high = [r["best_seq"] for r in cand_rows if r["high"] == "1"]
    pool_not_high = [r["best_seq"] for r in cand_rows if r["high"] == "0"]

    groups_input: dict[str, list[str]] = {
        "pool_all": pool_all,
        "pool_conditioned": pool_cond,
        "pool_high": pool_high,
        "pool_not_high": pool_not_high,
        "final_five": final_seqs,
    }
    if pool_uncond:
        groups_input["pool_unconditioned"] = pool_uncond

    # Run screen across all candidate groups and final slate
    res = screen(groups_input)
    assert res["controls_pass"] is True

    # Golden comparison from composition_result.json
    golden_json_file = fix_dir / "composition_result.json"
    if not golden_json_file.exists():
        golden_json_file = next(fix_dir.glob("**/composition_result.json"))
    golden_json = json.loads(golden_json_file.read_text())

    # Top-level parity assertions
    assert res["pool_n"] == golden_json["pool_n"] == 594
    assert np.isclose(res["pool_mean_ek"], golden_json["pool_mean_ek"], atol=1e-4)
    assert np.isclose(res["pool_ek_enrichment"], golden_json["pool_ek_enrichment"], atol=1e-2)
    assert np.isclose(res["final_mean_ek"], golden_json["final_mean_ek"], atol=1e-4)
    assert np.isclose(res["background_ek"], golden_json["background_ek"], atol=1e-4)

    # All group-level assertions matching composition_result.json
    for grp_name, golden_group in golden_json["groups"].items():
        assert grp_name in res["groups"], f"Group '{grp_name}' missing from screen result"
        group_res = res["groups"][grp_name]
        assert group_res["n"] == golden_group["n"]
        assert np.isclose(group_res["mean_len"], golden_group["mean_len"], atol=1e-3)
        assert np.isclose(group_res["mean_ek"], golden_group["mean_ek"], atol=1e-4)
        assert np.isclose(group_res["frac_ek_gt_30pct"], golden_group["frac_ek_gt_30pct"], atol=1e-4)
        assert np.isclose(group_res["mean_net_charge"], golden_group["mean_net_charge"], atol=0.05)
        assert np.isclose(group_res["frac_net_neg"], golden_group["frac_net_neg"], atol=1e-4)

        # Per-amino acid frequency parity
        for a in AA:
            expected_freq = golden_group["mean_freq"][a]
            actual_freq = group_res["mean_freq"][a]
            assert np.isclose(actual_freq, expected_freq, atol=1e-4), (
                f"Frequency mismatch for group '{grp_name}', AA '{a}': got {actual_freq}, expected {expected_freq}"
            )

    # Fold over background assertions matching composition_result.json
    for grp_name, golden_fold in golden_json["fold_over_background"].items():
        assert grp_name in res["fold_over_background"], f"Fold over background missing group '{grp_name}'"
        for a in AA:
            assert np.isclose(res["fold_over_background"][grp_name][a], golden_fold[a], atol=0.02), (
                f"Fold enrichment mismatch for group '{grp_name}', AA '{a}'"
            )

    # Parity verification for all rows of tfr1_composition.tsv
    tsv_file = fix_dir / "tfr1_composition.tsv"
    if not tsv_file.exists():
        tsv_file = next(fix_dir.glob("**/tfr1_composition.tsv"))
    tsv_rows = list(csv.DictReader(tsv_file.open(), delimiter="\t"))
    assert len(tsv_rows) == 5, f"Expected 5 rows in tfr1_composition.tsv, got {len(tsv_rows)}"

    for row in tsv_rows:
        grp = row["group"]
        assert grp in res["groups"], f"TSV group '{grp}' not found in screen groups"
        g_res = res["groups"][grp]
        assert g_res["n"] == int(row["n"])
        assert np.isclose(g_res["mean_len"], float(row["mean_len"]), atol=1e-3)
        assert np.isclose(g_res["mean_ek"], float(row["mean_ek"]), atol=1e-3)
        assert np.isclose(g_res["frac_ek_gt_30pct"], float(row["frac_ek_gt_30pct"]), atol=1e-3)
        assert np.isclose(g_res["mean_net_charge"], float(row["mean_net_charge"]), atol=0.05)
        assert np.isclose(g_res["frac_net_neg"], float(row["frac_net_neg"]), atol=1e-3)
        for a in AA:
            col = f"fold_{a}"
            assert np.isclose(res["fold_over_background"][grp][a], float(row[col]), atol=0.02), (
                f"TSV fold mismatch for group '{grp}', AA '{a}'"
            )


def test_bathos_parity_bba33e3a_interface_audit() -> None:
    """Parity verification for bba33e3a (interface charge and contact analysis).

    Reproduces exact registered contract summary from interface_result.json and per-complex
    rows in tfr1_interface.tsv (strictly excluding buried_binder_A2).
    """
    fix_dir = _resolve_fixture_dir()
    order_file = fix_dir / "tfr1_final_order.tsv"
    if not order_file.exists():
        order_file = next(fix_dir.glob("**/tfr1_final_order.tsv"))
    order_rows = list(csv.DictReader(order_file.open(), delimiter="\t"))

    golden_tsv_file = fix_dir / "tfr1_interface.tsv"
    if not golden_tsv_file.exists():
        golden_tsv_file = next(fix_dir.glob("**/tfr1_interface.tsv"))
    golden_tsv = list(csv.DictReader(golden_tsv_file.open(), delimiter="\t"))
    golden_by_id = {r["design_id"]: r for r in golden_tsv}

    struct_dir = fix_dir / "structures"
    if not struct_dir.exists():
        struct_dir = fix_dir / "f3" / "structures"

    total_ion_pairs = 0
    total_like_close = 0

    for r in order_rows:
        did = r["design_id"]
        golden = golden_by_id[did]

        pdb = struct_dir / f"{did}.pdb"
        if not pdb.exists():
            # Rank-1 variant has no F3 medoid complex; parent is used
            pdb = struct_dir / f"{did.split('_i')[0]}.pdb"
        assert pdb.exists(), f"PDB structure {pdb} not found"

        result = compute_interface_contacts(pdb, chain_a="A", chain_b="B")

        # Contact counts and residue counts
        assert result.n_binder_interface == int(golden["n_binder_interface"])
        assert result.n_target_interface == int(golden["n_target_interface"])
        assert result.ion_pairs == int(golden["ion_pairs"])
        assert result.like_close == int(golden["like_close"])
        assert result.polar_pairs == int(golden["polar_pairs"])
        assert result.nonpolar_pairs == int(golden["nonpolar_pairs"])

        # Interface charges and fractions
        assert result.binder_if_charge == int(golden["binder_if_charge"])
        assert result.target_if_charge == int(golden["target_if_charge"])
        assert np.isclose(result.binder_if_frac_ek, float(golden["binder_if_frac_EK"]), atol=1e-4)
        assert np.isclose(result.binder_if_frac_hydrophobic, float(golden["binder_if_frac_hydrophobic"]), atol=1e-4)

        # Target interface residue string
        assert result.details["target_if_residues"] == golden["target_if_residues"]

        # Decoupling rule: verify targeted motif and 6A target charge
        from proxide.geometry.contacts import _parse_pdb_text
        atoms = _parse_pdb_text(pdb.read_text())
        atoms_a = [x for x in atoms if x["chain"] == "A"]
        atoms_b = [x for x in atoms if x["chain"] == "B"]
        coords_a = np.array([x["xyz"] for x in atoms_a], dtype=float)
        coords_b = np.array([x["xyz"] for x in atoms_b], dtype=float)
        d_mat = np.linalg.norm(coords_a[:, None, :] - coords_b[None, :, :], axis=2)

        # Target residues within 6.0 A of binder
        t_near6 = sorted({atoms_b[j]["num"] for j in range(len(atoms_b)) if d_mat[:, j].min() <= 6.0})
        b_res_map = {x["num"]: x["res"] for x in atoms_b}
        target_charge_6a = sum(RESIDUE_CHARGE.get(b_res_map[n], 0) for n in t_near6)
        assert target_charge_6a == int(golden["target_charge_within_6A"])

        # Motif residues contacted (within 4.5 A)
        motif_contacted = sorted(
            n for n in MOTIF if any(d_mat[:, j].min() <= 4.5 for j in range(len(atoms_b)) if atoms_b[j]["num"] == n)
        )
        assert len(motif_contacted) == int(golden["motif_contacted"])

        total_ion_pairs += result.ion_pairs
        total_like_close += result.like_close

    # Verify overall contract summary matching interface_result.json
    golden_result_file = fix_dir / "interface_result.json"
    if not golden_result_file.exists():
        golden_result_file = next(fix_dir.glob("**/interface_result.json"))
    golden_summary = json.loads(golden_result_file.read_text())

    assert golden_summary["selftest_all_pass"] is True
    assert total_ion_pairs == golden_summary["total_ion_pairs"] == 21
    assert total_like_close == golden_summary["total_like_charge_close"] == 2
