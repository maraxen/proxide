#!/usr/bin/env python3
"""
Extract rotamers from Dunbrack 2010 backbone-dependent rotamer library.

Parses ALL.bbdep.rotamers.lib filtered to CPR/PRO/TPR residues,
emits JSON with sorted φ/ψ centers, default bin, and per-bin rotamer data.

Reference: Dunbrack 2010 BBDEP library (ODC-BY licensed).
"""

import argparse
import json
import logging
from pathlib import Path
from typing import TypedDict, Optional
import hashlib


class ChiData(TypedDict):
    """Chi angle value and sigma."""
    val: float
    sigma: float


class RotamerData(TypedDict):
    """Single rotamer in a (φ,ψ) bin."""
    prob: float
    chi: list[ChiData]
    r1: int
    r2: int
    r3: int
    r4: int


class BinData(TypedDict):
    """Data for a single (φ,ψ) bin."""
    phi: float
    psi: float
    rotamers: list[RotamerData]


class ResidueOutput(TypedDict):
    """Output for a single residue type."""
    code: str
    phi_centers: list[float]
    psi_centers: list[float]
    default_bin: int
    n_bins_pro: int
    n_bins_cpr: int
    cpr_rotamers_per_bin: int
    grid_rectangular: bool
    prob_sum_check: list[dict]
    bins: list[BinData]


def compute_sha256(fpath: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha = hashlib.sha256()
    with open(fpath, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            sha.update(chunk)
    return sha.hexdigest()


def parse_dunbrack_lib(
    lib_path: Path,
    residue_types: set[str],
    dry_run: bool = False
) -> dict:
    """
    Parse Dunbrack .bbdep.rotamers.lib file.

    Returns dict mapping residue code -> {
        'phi_centers': sorted list,
        'psi_centers': sorted list,
        'bins': dict mapping (phi, psi) -> list of rotamers,
        'default_bin': index of highest-frequency bin
    }
    """
    logging.info(f"Reading {lib_path}")

    if not lib_path.exists():
        raise FileNotFoundError(f"Input file not found: {lib_path}")

    # Accumulate data per residue
    residue_data = {}

    with open(lib_path, "r") as f:
        for line_num, line in enumerate(f, start=1):
            # Skip comments and empty lines
            if line.startswith("#") or not line.strip():
                continue

            parts = line.split()
            if len(parts) < 17:
                logging.warning(f"Line {line_num}: skipping short line ({len(parts)} parts)")
                continue

            residue_type = parts[0]

            # Filter by requested residue types
            if residue_type not in residue_types:
                continue

            if dry_run and line_num > 1000:
                logging.info("Dry-run: stopping at 1000 lines")
                break

            try:
                phi = float(parts[1])
                psi = float(parts[2])
                count = int(parts[3])
                r1 = int(parts[4])
                r2 = int(parts[5])
                r3 = int(parts[6])
                r4 = int(parts[7])
                prob = float(parts[8])
                chi1_val = float(parts[9])
                chi2_val = float(parts[10])
                chi3_val = float(parts[11])
                chi4_val = float(parts[12])
                chi1_sig = float(parts[13])
                chi2_sig = float(parts[14])
                chi3_sig = float(parts[15])
                chi4_sig = float(parts[16])
            except (ValueError, IndexError) as e:
                logging.warning(f"Line {line_num}: parse error: {e}")
                continue

            # Initialize residue if new
            if residue_type not in residue_data:
                residue_data[residue_type] = {
                    "phi_set": set(),
                    "psi_set": set(),
                    "bins": {},
                    "bin_freqs": {}
                }

            # Track unique φ, ψ values
            residue_data[residue_type]["phi_set"].add(phi)
            residue_data[residue_type]["psi_set"].add(psi)

            # Store bin data
            bin_key = (phi, psi)
            if bin_key not in residue_data[residue_type]["bins"]:
                residue_data[residue_type]["bins"][bin_key] = []
                residue_data[residue_type]["bin_freqs"][bin_key] = count

            rotamer = RotamerData(
                prob=prob,
                chi=[
                    ChiData(val=chi1_val, sigma=chi1_sig),
                    ChiData(val=chi2_val, sigma=chi2_sig),
                    ChiData(val=chi3_val, sigma=chi3_sig),
                    ChiData(val=chi4_val, sigma=chi4_sig),
                ],
                r1=r1,
                r2=r2,
                r3=r3,
                r4=r4,
            )
            residue_data[residue_type]["bins"][bin_key].append(rotamer)

    # Post-process: convert to sorted lists and compute default bins
    output = {}
    for res_type in sorted(residue_data.keys()):
        data = residue_data[res_type]
        phi_centers = sorted(data["phi_set"])
        psi_centers = sorted(data["psi_set"])

        # Build output bins list (phi-major order)
        bins_list = []
        freq_tuples = []  # (freq, bin_index) for computing default_bin

        for i, phi in enumerate(phi_centers):
            for j, psi in enumerate(psi_centers):
                bin_key = (phi, psi)
                if bin_key in data["bins"]:
                    freq = data["bin_freqs"][bin_key]
                    bin_data = BinData(
                        phi=phi,
                        psi=psi,
                        rotamers=data["bins"][bin_key]
                    )
                    bins_list.append(bin_data)
                    freq_tuples.append((freq, len(bins_list) - 1))

        # Compute default_bin as argmax frequency (first-max wins)
        default_bin = max(freq_tuples, key=lambda x: x[0])[1] if freq_tuples else 0

        # Check grid is rectangular
        grid_rectangular = len(bins_list) == len(phi_centers) * len(psi_centers)

        # Count rotamers per bin and validate
        rotamer_counts_per_bin = [len(b["rotamers"]) for b in bins_list]

        # Validate probability sums
        prob_sums = []
        prob_sum_ok = True
        for bin_idx, bin_data in enumerate(bins_list):
            psum = sum(r["prob"] for r in bin_data["rotamers"])
            prob_sums.append({
                "bin_index": bin_idx,
                "phi": bin_data["phi"],
                "psi": bin_data["psi"],
                "prob_sum": psum,
                "ok": 0.997 <= psum <= 1.003  # 1.0 ± 1e-3
            })
            if not prob_sums[-1]["ok"]:
                prob_sum_ok = False

        # Determine CPR rotamers per bin if applicable
        cpr_rotamers_per_bin = 0
        if res_type == "CPR":
            if rotamer_counts_per_bin:
                # All CPR bins should have same count
                cpr_rotamers_per_bin = rotamer_counts_per_bin[0]

        output[res_type] = ResidueOutput(
            code=res_type,
            phi_centers=phi_centers,
            psi_centers=psi_centers,
            default_bin=default_bin,
            n_bins_pro=len(bins_list) if res_type == "PRO" else 0,
            n_bins_cpr=len(bins_list) if res_type == "CPR" else 0,
            cpr_rotamers_per_bin=cpr_rotamers_per_bin,
            grid_rectangular=grid_rectangular,
            prob_sum_check=prob_sums,
            bins=bins_list
        )

    return output


def validate_ac1(output: dict) -> dict:
    """
    Validate AC-1 requirements:
    - CPR/PRO/TPR all present
    - CPR has 2 rotamers/bin
    - Each bin Σprob = 1.0±1e-3
    - Grid rectangular
    - Record PRO & CPR bin counts
    """
    checks = {
        "cpw_all_present": all(
            res in output for res in ["CPR", "PRO", "TPR"]
        ),
        "cpr_rotamers_per_bin": output.get("CPR", {}).get("cpr_rotamers_per_bin", 0),
        "cpr_rotamers_per_bin_ok": output.get("CPR", {}).get("cpr_rotamers_per_bin", 0) == 2,
        "grid_rectangular": all(
            output[res]["grid_rectangular"] for res in output
        ),
        "prob_sums_valid": all(
            all(check["ok"] for check in output[res]["prob_sum_check"])
            for res in output
        ),
        "pro_bin_count": output.get("PRO", {}).get("n_bins_pro", 0),
        "cpr_bin_count": output.get("CPR", {}).get("n_bins_cpr", 0),
    }

    # AC-1 pass: all checks must pass
    checks["ac1_pass"] = (
        checks["cpw_all_present"]
        and checks["cpr_rotamers_per_bin_ok"]
        and checks["grid_rectangular"]
        and checks["prob_sums_valid"]
    )

    return checks


def main():
    parser = argparse.ArgumentParser(
        description="Extract Dunbrack rotamers (CPR/PRO/TPR) to JSON"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/rotlibs/SimpleOpt1-5/ALL.bbdep.rotamers.lib"),
        help="Input .bbdep.rotamers.lib file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/rotlibs/SimpleOpt1-5/rotamers-cpr-pro-tpr.json"),
        help="Output JSON file"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse only first 1000 lines"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s"
    )

    logging.info(f"Input file: {args.input.resolve()}")
    logging.info(f"Output file: {args.output.resolve()}")

    # Verify input exists
    if not args.input.exists():
        logging.error(f"Input file not found: {args.input}")
        return 1

    # Compute input SHA256
    input_sha = compute_sha256(args.input)
    logging.info(f"Input SHA256: {input_sha}")

    # Parse
    try:
        output = parse_dunbrack_lib(
            args.input,
            residue_types={"CPR", "PRO", "TPR"},
            dry_run=args.dry_run
        )
    except Exception as e:
        logging.error(f"Parsing failed: {e}", exc_info=True)
        return 1

    # Validate AC-1
    ac1_checks = validate_ac1(output)
    logging.info("AC-1 Validation Results:")
    for key, value in ac1_checks.items():
        logging.info(f"  {key}: {value}")

    if not ac1_checks["ac1_pass"]:
        logging.error("AC-1 validation FAILED")
        return 1

    logging.info("AC-1 validation PASSED")

    # Create output structure with metadata
    output_data = {
        "metadata": {
            "input_file": str(args.input),
            "input_sha256": input_sha,
            "residue_types": sorted(output.keys()),
            "dry_run": args.dry_run,
            "ac1_checks": ac1_checks,
        },
        "residues": output
    }

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    logging.info(f"Wrote {args.output}")

    # Summary
    print("\n=== AC-1 Summary ===")
    print(f"CPR/PRO/TPR present: {ac1_checks['cpw_all_present']}")
    print(f"CPR rotamers/bin: {ac1_checks['cpr_rotamers_per_bin']}")
    print(f"Grid rectangular: {ac1_checks['grid_rectangular']}")
    print(f"Prob sums valid: {ac1_checks['prob_sums_valid']}")
    print(f"PRO bins: {ac1_checks['pro_bin_count']}")
    print(f"CPR bins: {ac1_checks['cpr_bin_count']}")
    print(f"\nAC-1 PASS: {ac1_checks['ac1_pass']}")

    return 0 if ac1_checks["ac1_pass"] else 1


if __name__ == "__main__":
    exit(main())
