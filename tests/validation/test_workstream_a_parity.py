"""Parity and verification tests for Workstream A:
- Batch mmCIF parser (#2642): batched == looped parity, ligand failure, per-file error isolation
- Trajectory throughput: read_xtc_parallel == read_xtc_lazy bit-exact parity, distogram parity
- Sidecar cache resolver: 5-tier resolution and zero adjacent sidecars when off
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import proxide

CRN_CIF = REPO_ROOT / "crates" / "proxide-io" / "tests" / "data" / "1CRN.cif"
UBQ_CIF = REPO_ROOT / "crates" / "proxide-io" / "tests" / "data" / "1UBQ.cif"
TEST_XTC = REPO_ROOT / "tests" / "data" / "trajectories" / "test.xtc"


def test_mmcif_batch_parity_with_serial_loop():
    """Verify parse_mmcif_batch matches parse_mmcif looped serially bit-for-bit."""
    paths = [str(CRN_CIF), str(UBQ_CIF), str(CRN_CIF)]
    batch_results = proxide.parse_mmcif_batch(paths, fail_on_ligands=False, numbering="auth")
    assert len(batch_results) == 3

    for path_str, res in zip(paths, batch_results):
        assert res["success"] is True
        assert res["error"] is None

        serial_data = proxide.parse_mmcif(path_str, numbering="auth")
        assert res["element"] == serial_data["elements"]
        assert res["atom_name"] == serial_data["atom_names"]
        assert res["chain"] == serial_data["chain_ids"]
        assert res["residue"] == serial_data["res_names"]

        # Reshaped (N, 3) coords
        np.testing.assert_array_equal(res["xyz"], serial_data["coords"])
        np.testing.assert_array_equal(res["b_factor"], serial_data["b_factors"])
        np.testing.assert_array_equal(res["label_seq_id"], serial_data["res_ids"])
        assert res["ligand_flag"] == serial_data["is_hetatm"]


def test_mmcif_batch_fail_on_ligands():
    """Verify fail_on_ligands=True rejects structures with HETATM/ligands."""
    paths = [str(CRN_CIF), str(UBQ_CIF)]
    batch_results = proxide.parse_mmcif_batch(paths, fail_on_ligands=True, numbering="auth")
    assert len(batch_results) == 2

    # 1CRN has no ligands -> succeeds
    assert batch_results[0]["success"] is True
    assert batch_results[0]["error"] is None

    # 1UBQ has HOH waters -> fails with structured error message
    assert batch_results[1]["success"] is False
    assert batch_results[1]["error"] is not None
    assert "contains ligand/HETATM atoms" in batch_results[1]["error"]


def test_mmcif_batch_error_isolation_swallows_none():
    """Verify per-file error isolation: failing files do not abort valid files."""
    paths = [
        str(CRN_CIF),
        str(REPO_ROOT / "nonexistent" / "missing.cif"),
        str(UBQ_CIF),
    ]
    batch_results = proxide.parse_mmcif_batch(paths, fail_on_ligands=False, numbering="auth")
    assert len(batch_results) == 3

    assert batch_results[0]["success"] is True
    assert batch_results[1]["success"] is False
    assert "missing.cif" in batch_results[1]["error"]
    assert batch_results[2]["success"] is True


def test_trajectory_parallel_vs_lazy_parity():
    """Verify read_xtc_parallel matches read_xtc_lazy bit-for-bit across strides and atom subsets."""
    path = str(TEST_XTC)

    for stride in [1, 2]:
        for atom_indices in [None, [0, 1], [1, 2, 3], [0, 2]]:
            lazy = proxide.read_xtc_lazy(path, stride=stride, atom_indices=atom_indices)
            parallel = proxide.read_xtc_parallel(path, stride=stride, atom_indices=atom_indices)

            assert lazy["num_frames"] == parallel["num_frames"]
            assert lazy["num_atoms"] == parallel["num_atoms"]
            np.testing.assert_array_equal(lazy["coordinates"], parallel["coordinates"])
            np.testing.assert_array_equal(lazy["box_vectors"], parallel["box_vectors"])
            np.testing.assert_array_equal(lazy["times"], parallel["times"])


def test_trajectory_ca_distogram_parity():
    """Verify read_xtc_ca_distogram matches distances computed directly from coordinates."""
    path = str(TEST_XTC)
    atom_indices = [0, 1, 2]
    dist = proxide.read_xtc_ca_distogram(path, atom_indices=atom_indices, stride=1)

    lazy = proxide.read_xtc_lazy(path, stride=1, atom_indices=atom_indices)
    coords = lazy["coordinates"]  # (n_frames, 3, 3)
    n_frames = coords.shape[0]

    expected_dists = []
    for f in range(n_frames):
        c = coords[f]
        d01 = np.linalg.norm(c[0] - c[1])
        d02 = np.linalg.norm(c[0] - c[2])
        d12 = np.linalg.norm(c[1] - c[2])
        expected_dists.append([d01, d02, d12])

    np.testing.assert_allclose(dist, np.array(expected_dists, dtype=np.float32), rtol=1e-5, atol=1e-5)


def test_sidecar_resolver_tier_5_off_writes_nothing_adjacent(tmp_path):
    """Verify that by default (caching off / tier 5), no sidecar is created adjacent to trajectory."""
    # Copy test.xtc to tmp_path
    traj_copy = tmp_path / "copy.xtc"
    traj_copy.write_bytes(TEST_XTC.read_bytes())

    # Ensure PROXIDE_OFFSETS_DIR is unset or empty
    old_env = os.environ.pop("PROXIDE_OFFSETS_DIR", None)
    try:
        frames = proxide.read_xtc_parallel(str(traj_copy))
        assert frames["coordinates"].shape[0] == 5

        # Confirm NO .offsets file was created adjacent to copy.xtc
        sidecar = tmp_path / "copy.xtc.offsets"
        assert not sidecar.exists(), f"Sidecar {sidecar} should NOT exist when caching is off"
    finally:
        if old_env is not None:
            os.environ["PROXIDE_OFFSETS_DIR"] = old_env


def test_sidecar_resolver_tier_1_explicit_dir(tmp_path):
    """Verify explicit offsets_dir argument stores sidecar in the designated directory."""
    traj_copy = tmp_path / "copy.xtc"
    traj_copy.write_bytes(TEST_XTC.read_bytes())
    cache_dir = tmp_path / "custom_cache"
    cache_dir.mkdir()

    frames = proxide.read_xtc_parallel(str(traj_copy), offsets_dir=str(cache_dir))
    assert frames["coordinates"].shape[0] == 5

    # Sidecar should exist in cache_dir
    sidecars = list(cache_dir.glob("*.offsets"))
    assert len(sidecars) == 1, f"Expected 1 sidecar in {cache_dir}, found {sidecars}"
