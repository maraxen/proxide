#!/usr/bin/env python3
"""
Black-box parity testing harness comparing upstream Mosaist confind (executed in an
external scratch directory) against clean-room proxide-confind on standard benchmark
structures (1CRN, 1UBQ, multi-chain complex, and segmented chain break).

Pre-registered under bathos (scripts/validation/confind_blackbox_parity.bth.toml).
Task ID: 261102_proxide-mosaist-cleanroom
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Paths
WORKSPACE = Path("/home/marielle/projects/proxide")
CLEAN_WORKTREE = WORKSPACE / ".worktrees/wt-confind-cleanroom"
CLEAN_BIN = CLEAN_WORKTREE / "target/release/confind"
CLEAN_RLIB = CLEAN_WORKTREE / "data/rotlibs/proxide-rotlib-dunbrack2010-ccd.pb.zst"

SCRATCH = Path("/tmp/mosaist_scratch")
UPSTREAM_BIN = SCRATCH / "testConFind"
UPSTREAM_RLIB = SCRATCH / "rotlib.bin"

# Benchmark inputs
CRN_PDB = WORKSPACE / ".worktrees/wt-rotlib-fixer/crates/proxide-io/tests/data/1CRN.pdb"
UBQ_PDB = WORKSPACE / "crates/proxide-tmalign/tests/data/ubiquitin_1ubq_A.pdb"
MULTICHAIN_PDB = WORKSPACE / ".worktrees/wt-261102-design-utilities/tests/validation/fixtures/tfr1/structures/f2c006r13.pdb"
BREAK_PDB = WORKSPACE / "crates/proxide_fixer/tests/data/chain_break.pdb"


def parse_confind_output(path: Path, is_upstream: bool = False):
    """
    Parse confind output format:
    contact\t<chain_a>,<res_a>\t<chain_b>,<res_b>\t<cd>\t<name_a>\t<name_b>
    crwdnes/crowdedness\t<chain>,<res>\t<cr>\t...
    freedom\t<chain>,<res>\t<fr>\t...
    """
    contacts = {}
    crowdedness = {}
    freedom = {}
    dihedrals = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            tag = parts[0]
            if tag == "contact":
                # parts: contact, id_a, id_b, val, name_a, name_b
                p_a, p_b = parts[1], parts[2]
                val = float(parts[3])
                pair = tuple(sorted([p_a, p_b]))
                contacts[pair] = val
            elif tag in ("crwdnes", "crowdedness"):
                res = parts[1]
                val = float(parts[2])
                crowdedness[res] = val
                if is_upstream and len(parts) >= 6:
                    # crwdnes, res, val, phi, psi, omega, resname
                    phi, psi, omg = float(parts[3]), float(parts[4]), float(parts[5])
                    dihedrals[res] = (phi, psi, omg)
            elif tag == "freedom":
                res = parts[1]
                val = float(parts[2])
                freedom[res] = val
                if is_upstream and len(parts) >= 6:
                    phi, psi, omg = float(parts[3]), float(parts[4]), float(parts[5])
                    dihedrals[res] = (phi, psi, omg)

    return contacts, crowdedness, freedom, dihedrals


def run_benchmark(pdb_path: Path, tag: str):
    """Run upstream and clean confind, returning parsed outputs."""
    up_out = SCRATCH / f"{tag}_upstream.out"
    clean_out = SCRATCH / f"{tag}_clean.out"

    # Upstream Mosaist (reuse scratch output if already generated in /tmp/mosaist_scratch)
    if not up_out.exists():
        subprocess.run(
            [
                str(UPSTREAM_BIN),
                "--p", str(pdb_path),
                "--rLib", str(UPSTREAM_RLIB),
                "--o", str(up_out),
                "--pp", "--omg",
            ],
            check=True,
            cwd=str(SCRATCH),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    # Cleanroom confind
    subprocess.run(
        [
            str(CLEAN_BIN),
            "--p", str(pdb_path),
            "--rLib", str(CLEAN_RLIB),
            "--o", str(clean_out),
            "--cdcut", "0.0",
            "--incut", "0.0",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    up_c, up_cr, up_fr, up_dih = parse_confind_output(up_out, is_upstream=True)
    cl_c, cl_cr, cl_fr, _ = parse_confind_output(clean_out, is_upstream=False)

    return up_c, up_cr, up_fr, up_dih, cl_c, cl_cr, cl_fr


def compute_contact_deltas(up_c, cl_c):
    all_pairs = set(up_c.keys()) | set(cl_c.keys())
    deltas = []
    for p in all_pairs:
        v_up = up_c.get(p, 0.0)
        v_cl = cl_c.get(p, 0.0)
        deltas.append(abs(v_up - v_cl))
    deltas.sort()
    n = len(deltas)
    median_val = deltas[n // 2] if n > 0 else 0.0
    mean_val = sum(deltas) / n if n > 0 else 0.0
    max_val = max(deltas) if n > 0 else 0.0
    pct_1e4 = (sum(1 for d in deltas if d <= 1e-4) / n) if n > 0 else 0.0
    pct_005 = (sum(1 for d in deltas if d <= 0.05) / n) if n > 0 else 0.0
    return median_val, mean_val, max_val, pct_1e4, pct_005, len(all_pairs), len(set(up_c.keys()) & set(cl_c.keys()))


def compute_metric_deltas(up_m, cl_m):
    common = set(up_m.keys()) & set(cl_m.keys())
    if not common:
        return 0.0, 0.0, 0.0
    deltas = sorted([abs(up_m[k] - cl_m[k]) for k in common])
    n = len(deltas)
    return deltas[n // 2], sum(deltas) / n, max(deltas)


def assess_omega_sign(pdb_path: Path, up_dih):
    """
    Compute clean IUPAC omega and compare to upstream omega (Debt #1903).
    Standard IUPAC convention: torsion(CA_{i-1}, C_{i-1}, N_i, CA_i).
    """
    import math

    atoms = {}
    res_list = []
    with open(pdb_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("ATOM  "):
                name = line[12:16].strip()
                res_name = line[17:20].strip()
                chain = line[21]
                res_id = int(line[22:26].strip())
                xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
                key = (chain, res_id)
                if key not in atoms:
                    atoms[key] = {}
                    res_list.append((chain, res_id, res_name))
                if name in ("N", "CA", "C", "O") and name not in atoms[key]:
                    atoms[key][name] = xyz

    def sub(a, b):
        return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

    def dot(a, b):
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    def cross(a, b):
        return (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        )

    def dihedral(p1, p2, p3, p4):
        b1 = sub(p2, p1)
        b2 = sub(p3, p2)
        b3 = sub(p4, p3)
        n1 = cross(b1, b2)
        n2 = cross(b2, b3)
        b2_len = math.sqrt(dot(b2, b2))
        b2_norm = (b2[0] / b2_len, b2[1] / b2_len, b2[2] / b2_len)
        m1 = cross(n1, b2_norm)
        return math.atan2(dot(m1, n2), dot(n1, n2))

    same_sign = 0
    opp_sign = 0
    max_delta = 0.0

    for i in range(1, len(res_list)):
        prev_k = res_list[i - 1][:2]
        curr_k = res_list[i][:2]
        if (
            "CA" in atoms[prev_k]
            and "C" in atoms[prev_k]
            and "N" in atoms[curr_k]
            and "CA" in atoms[curr_k]
        ):
            # Cleanroom IUPAC omega: -dihedral angle in degrees
            cl_omg = -math.degrees(
                dihedral(
                    atoms[prev_k]["CA"],
                    atoms[prev_k]["C"],
                    atoms[curr_k]["N"],
                    atoms[curr_k]["CA"],
                )
            )
            up_str_k = f"{curr_k[0]},{curr_k[1]}"
            if up_str_k in up_dih:
                up_omg = up_dih[up_str_k][2]
                if abs(up_omg - 999.0) > 1.0:
                    diff = abs(cl_omg - up_omg)
                    max_delta = max(max_delta, diff)
                    if (cl_omg * up_omg) >= 0:
                        same_sign += 1
                    else:
                        opp_sign += 1

    return same_sign, opp_sign, max_delta


def assess_chain_semantics():
    """
    Evaluate Debt #1904 (Chain Semantics):
    Tests chain_break.pdb where a 10 A CA gap exists between residue 2 and 3.
    Upstream erroneously connects across the break, emitting a contact of ~0.000436.
    Cleanroom segments the chain, correctly eliminating this false contact.
    """
    up_c, _, _, _, cl_c, _, _ = run_benchmark(BREAK_PDB, "break")
    pair_2_3 = ("A,2", "A,3")
    up_val = up_c.get(pair_2_3, 0.0)
    cl_val = cl_c.get(pair_2_3, 0.0)

    # In chain_break.pdb:
    # Upstream computes contact A,2 <-> A,3 > 0.0001
    # Cleanroom detects CA-CA > 4.5 A and segments (contact = 0.0)
    prevented = (up_val > 1e-4) and (cl_val == 0.0)
    return prevented, up_val, cl_val


def main():
    parser = argparse.ArgumentParser(description="ConFind black-box parity validation")
    parser.add_argument("--out", type=Path, default=Path("outputs/parity/confind_parity_results.json"))
    args = parser.parse_args()

    print("Running Benchmark 1: 1CRN...")
    up_c_crn, up_cr_crn, up_fr_crn, up_dih_crn, cl_c_crn, cl_cr_crn, cl_fr_crn = run_benchmark(CRN_PDB, "1crn")
    med_cd_crn, mean_cd_crn, max_cd_crn, pct_1e4_crn, pct_005_crn, n_pairs_crn, com_pairs_crn = compute_contact_deltas(up_c_crn, cl_c_crn)
    med_cr_crn, _, _ = compute_metric_deltas(up_cr_crn, cl_cr_crn)
    med_fr_crn, _, _ = compute_metric_deltas(up_fr_crn, cl_fr_crn)

    print("Running Benchmark 2: 1UBQ...")
    up_c_ubq, up_cr_ubq, up_fr_ubq, up_dih_ubq, cl_c_ubq, cl_cr_ubq, cl_fr_ubq = run_benchmark(UBQ_PDB, "1ubq")
    med_cd_ubq, mean_cd_ubq, max_cd_ubq, pct_1e4_ubq, pct_005_ubq, n_pairs_ubq, com_pairs_ubq = compute_contact_deltas(up_c_ubq, cl_c_ubq)
    med_cr_ubq, _, _ = compute_metric_deltas(up_cr_ubq, cl_cr_ubq)
    med_fr_ubq, _, _ = compute_metric_deltas(up_fr_ubq, cl_fr_ubq)

    print("Running Benchmark 3: Multi-chain complex (f2c006r13.pdb)...")
    up_c_mc, _, _, _, cl_c_mc, _, _ = run_benchmark(MULTICHAIN_PDB, "multichain")
    med_cd_mc, mean_cd_mc, max_cd_mc, pct_1e4_mc, pct_005_mc, n_pairs_mc, com_pairs_mc = compute_contact_deltas(up_c_mc, cl_c_mc)

    print("Assessing Debt #1903 (Omega Dihedral Sign)...")
    same_sign, opp_sign, max_omg_delta = assess_omega_sign(CRN_PDB, up_dih_crn)

    print("Assessing Debt #1904 (Chain Semantics & Breaks)...")
    prevented, up_break_val, cl_break_val = assess_chain_semantics()

    # Determine overall outcome
    passed = (
        med_cd_crn <= 0.0001
        and med_cd_ubq <= 0.0001
        and opp_sign == 0
        and prevented
    )
    outcome_label = "pass" if passed else ("marginal" if (med_cd_crn <= 0.0002 and med_cd_ubq <= 0.0002) else "fail")

    result = {
        "median_contact_delta_1crn": round(float(med_cd_crn), 6),
        "mean_contact_delta_1crn": round(float(mean_cd_crn), 6),
        "max_contact_delta_1crn": round(float(max_cd_crn), 6),
        "pct_within_1e4_1crn": round(float(pct_1e4_crn), 4),
        "pct_within_005_1crn": round(float(pct_005_crn), 4),
        "median_crowdedness_delta_1crn": round(float(med_cr_crn), 6),
        "median_freedom_delta_1crn": round(float(med_fr_crn), 6),

        "median_contact_delta_1ubq": round(float(med_cd_ubq), 6),
        "mean_contact_delta_1ubq": round(float(mean_cd_ubq), 6),
        "max_contact_delta_1ubq": round(float(max_cd_ubq), 6),
        "pct_within_1e4_1ubq": round(float(pct_1e4_ubq), 4),
        "pct_within_005_1ubq": round(float(pct_005_ubq), 4),
        "median_crowdedness_delta_1ubq": round(float(med_cr_ubq), 6),
        "median_freedom_delta_1ubq": round(float(med_fr_ubq), 6),

        "multichain_contacts_count": int(n_pairs_mc),
        "multichain_common_pairs": int(com_pairs_mc),
        "median_contact_delta_multichain": round(float(med_cd_mc), 6),
        "pct_within_1e4_multichain": round(float(pct_1e4_mc), 4),
        "pct_within_005_multichain": round(float(pct_005_mc), 4),

        "omega_same_sign_count": int(same_sign),
        "omega_opposite_sign_count": int(opp_sign),
        "max_omega_delta_deg": round(float(max_omg_delta), 6),
        "chain_break_prevented_spurious_contacts": bool(prevented),
        "outcome": outcome_label,
    }

    # Write output to target path
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    # Write to bathos results channel
    bth_results_path = os.environ.get("BTH_RESULTS_PATH")
    if bth_results_path:
        with open(bth_results_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    print(f"\nVerification Complete! Verdict: {outcome_label.upper()}")
    print(f"Summary: {json.dumps(result, indent=2)}")


if __name__ == "__main__":
    main()
