#!/usr/bin/env python3
"""Verification and Parity Runner for Workstream A (#2231, #2226, #2232, #2227, #2229, #2642).

Outputs structured JSON matching benchmarks/verify_workstream_a.bth.toml.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

# Ensure src is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import proxide  # noqa: E402


def run_verification() -> dict:
    crn_cif = REPO_ROOT / "crates" / "proxide-io" / "tests" / "data" / "1CRN.cif"
    ubq_cif = REPO_ROOT / "crates" / "proxide-io" / "tests" / "data" / "1UBQ.cif"
    test_xtc = REPO_ROOT / "tests" / "data" / "trajectories" / "test.xtc"

    # --- 1. mmCIF Parity ---
    paths = [str(crn_cif), str(ubq_cif), str(crn_cif)]
    batch_results = proxide.parse_mmcif_batch(paths, fail_on_ligands=False, numbering="auth")
    assert len(batch_results) == 3

    for path_str, res in zip(paths, batch_results, strict=True):
        assert res["success"] is True
        assert res["error"] is None
        serial_data = proxide.parse_mmcif(path_str, numbering="auth")
        assert res["element"] == serial_data["elements"]
        assert res["atom_name"] == serial_data["atom_names"]
        assert res["chain"] == serial_data["chain_ids"]
        assert res["residue"] == serial_data["res_names"]
        np.testing.assert_array_equal(res["xyz"], serial_data["coords"])
        np.testing.assert_array_equal(res["b_factor"], serial_data["b_factors"])
        np.testing.assert_array_equal(res["label_seq_id"], serial_data["res_ids"])
        assert res["ligand_flag"] == serial_data["is_hetatm"]

    # Test ligand filtering
    lig_batch = proxide.parse_mmcif_batch(
        [str(crn_cif), str(ubq_cif)],
        fail_on_ligands=True,
        numbering="auth",
    )
    assert lig_batch[0]["success"] is True
    assert lig_batch[1]["success"] is False
    assert "contains ligand/HETATM atoms" in lig_batch[1]["error"]

    # Test error isolation
    err_batch = proxide.parse_mmcif_batch(
        [str(crn_cif), str(REPO_ROOT / "nonexistent.cif"), str(ubq_cif)],
        fail_on_ligands=False,
    )
    assert err_batch[0]["success"] is True
    assert err_batch[1]["success"] is False
    assert "nonexistent.cif" in err_batch[1]["error"]
    assert err_batch[2]["success"] is True

    mmcif_parity_passed = True
    mmcif_files_verified = len(paths) + 2 + 3

    # --- 2. Trajectory Parity ---
    traj_path = str(test_xtc)
    traj_configs = 0
    for stride in [1, 2]:
        for atom_indices in [None, [0, 1], [1, 2, 3], [0, 2]]:
            lazy = proxide.read_xtc_lazy(traj_path, stride=stride, atom_indices=atom_indices)
            parallel = proxide.read_xtc_parallel(
                traj_path,
                stride=stride,
                atom_indices=atom_indices,
            )

            assert lazy["num_frames"] == parallel["num_frames"]
            assert lazy["num_atoms"] == parallel["num_atoms"]
            np.testing.assert_array_equal(lazy["coordinates"], parallel["coordinates"])
            np.testing.assert_array_equal(lazy["box_vectors"], parallel["box_vectors"])
            np.testing.assert_array_equal(lazy["times"], parallel["times"])
            traj_configs += 1

    # CA distogram parity
    dist = proxide.read_xtc_ca_distogram(traj_path, atom_indices=[0, 1, 2], stride=1)
    lazy_coords = proxide.read_xtc_lazy(traj_path, stride=1, atom_indices=[0, 1, 2])["coordinates"]
    expected_dists = []
    for f in range(lazy_coords.shape[0]):
        c = lazy_coords[f]
        expected_dists.append([
            np.linalg.norm(c[0] - c[1]),
            np.linalg.norm(c[0] - c[2]),
            np.linalg.norm(c[1] - c[2]),
        ])
    np.testing.assert_allclose(
        dist,
        np.array(expected_dists, dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )
    traj_configs += 1
    trajectory_parity_passed = True

    # --- 3. Sidecar Resolver Hierarchy ---
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        copy_xtc = tmp_path / "test_copy.xtc"
        copy_xtc.write_bytes(test_xtc.read_bytes())

        # Tier 5 (off): no sidecar written adjacent to file
        old_env = os.environ.pop("PROXIDE_OFFSETS_DIR", None)
        try:
            res = proxide.read_xtc_parallel(str(copy_xtc))
            assert res["num_frames"] == 5
            adjacent_sidecar = tmp_path / "test_copy.xtc.offsets"
            assert not adjacent_sidecar.exists(), (
                "Sidecar must not be created adjacent when caching is off"
            )

            # Tier 1 (explicit argument): writes to explicit dir
            explicit_dir = tmp_path / "explicit_cache"
            explicit_dir.mkdir()
            res2 = proxide.read_xtc_parallel(str(copy_xtc), offsets_dir=str(explicit_dir))
            assert res2["num_frames"] == 5
            explicit_sidecars = list(explicit_dir.glob("*.offsets"))
            assert len(explicit_sidecars) == 1, f"Expected 1 sidecar in {explicit_dir}"

            # Tier 2 (environment variable): writes to env var dir
            env_dir = tmp_path / "env_cache"
            env_dir.mkdir()
            os.environ["PROXIDE_OFFSETS_DIR"] = str(env_dir)
            res3 = proxide.read_xtc_parallel(str(copy_xtc))
            assert res3["num_frames"] == 5
            env_sidecars = list(env_dir.glob("*.offsets"))
            assert len(env_sidecars) == 1, f"Expected 1 sidecar in {env_dir}"

            # Empty / "none" env var disables caching
            os.environ["PROXIDE_OFFSETS_DIR"] = "none"
            none_dir = tmp_path / "none_check"
            none_dir.mkdir()
            none_copy = none_dir / "none_copy.xtc"
            none_copy.write_bytes(test_xtc.read_bytes())
            res4 = proxide.read_xtc_parallel(str(none_copy))
            assert res4["num_frames"] == 5
            assert not (none_dir / "none_copy.xtc.offsets").exists()
        finally:
            if old_env is not None:
                os.environ["PROXIDE_OFFSETS_DIR"] = old_env

    sidecar_resolver_passed = True
    parity_passed = mmcif_parity_passed and trajectory_parity_passed and sidecar_resolver_passed

    return {
        "parity_passed": parity_passed,
        "mmcif_parity_passed": mmcif_parity_passed,
        "trajectory_parity_passed": trajectory_parity_passed,
        "sidecar_resolver_passed": sidecar_resolver_passed,
        "mmcif_files_verified": mmcif_files_verified,
        "trajectory_configs_verified": traj_configs,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Verify Workstream A parity")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/verify_workstream_a.bth-results.json"),
        help="Path to write result JSON",
    )
    cli_args = parser.parse_args()

    results = run_verification()
    print(json.dumps(results, indent=2))

    for path in filter(None, [os.environ.get("BTH_RESULTS_PATH"), cli_args.out]):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(results, indent=2))

    if not results["parity_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
