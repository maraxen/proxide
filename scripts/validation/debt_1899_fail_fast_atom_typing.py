"""Verification script for Debt #1899: Fail-Fast GAFF2 Atom Typing.

Verifies that:
1. Python assign_gaff2_atom_types raises Gaff2AtomTypeMissingError for unmatched atoms
   (carbon, oxygen, hydrogen, etc.) instead of silently defaulting to 'c3', 'oh', 'ha'.
2. Python assign_gaff2_atom_types continues to successfully type molecules under full rules.
3. Rust assign_gaff2_atom_types and assign_types_with_rules return Err for unmatched atoms.
4. Rust test suite passes with fail-fast atom typing.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

from proxide.chem.gaff2 import (
    Gaff2AtomTypeMissingError,
    assign_gaff2_atom_types,
    load_gaff2_rules,
)


def prepare_mol(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.SanitizeMol(mol)
    return mol


def main() -> int:
    print("[debt-1899] Starting GAFF2 fail-fast atom typing verification...")

    # Load real bundled rules
    all_rules, wildatoms = load_gaff2_rules()
    mol = prepare_mol("CC(=O)C")  # acetone

    # Positive control: standard rules pass
    try:
        types = assign_gaff2_atom_types(mol, rules=all_rules, wildatom_map=wildatoms)
        assert len(types) == mol.GetNumAtoms()
        assert types[0] in ("c3", "c")
        python_standard_rules_pass = True
        print("  [PASS] Python positive control: standard rules type acetone successfully")
    except Exception as exc:
        print(f"  [FAIL] Python positive control failed: {exc}")
        python_standard_rules_pass = False

    # Negative control 1: Carbon unmatched
    rules_no_c = [r for r in all_rules if r.atomic_num != 6]
    python_unmatched_carbon_raises = False
    try:
        assign_gaff2_atom_types(mol, rules=rules_no_c, wildatom_map=wildatoms)
        print("  [FAIL] Python: unmatched carbon silently defaulted instead of raising!")
    except Gaff2AtomTypeMissingError as exc:
        if "carbon" in str(exc).lower():
            python_unmatched_carbon_raises = True
            print(f"  [PASS] Python: unmatched carbon raised Gaff2AtomTypeMissingError: {exc}")
        else:
            print(f"  [FAIL] Python: unexpected error message: {exc}")
    except Exception as exc:
        print(f"  [FAIL] Python: wrong exception type raised: {type(exc)}: {exc}")

    # Negative control 2: Oxygen unmatched
    rules_no_o = [r for r in all_rules if r.atomic_num != 8]
    python_unmatched_oxygen_raises = False
    try:
        assign_gaff2_atom_types(mol, rules=rules_no_o, wildatom_map=wildatoms)
        print("  [FAIL] Python: unmatched oxygen silently defaulted instead of raising!")
    except Gaff2AtomTypeMissingError as exc:
        if "oxygen" in str(exc).lower():
            python_unmatched_oxygen_raises = True
            print(f"  [PASS] Python: unmatched oxygen raised Gaff2AtomTypeMissingError: {exc}")
        else:
            print(f"  [FAIL] Python: unexpected error message: {exc}")
    except Exception as exc:
        print(f"  [FAIL] Python: wrong exception type raised: {type(exc)}: {exc}")

    # Negative control 3: Hydrogen unmatched
    rules_no_h = [r for r in all_rules if r.atomic_num != 1]
    python_unmatched_hydrogen_raises = False
    try:
        assign_gaff2_atom_types(mol, rules=rules_no_h, wildatom_map=wildatoms)
        print("  [FAIL] Python: unmatched hydrogen silently defaulted instead of raising!")
    except Gaff2AtomTypeMissingError as exc:
        if "hydrogen" in str(exc).lower():
            python_unmatched_hydrogen_raises = True
            print(f"  [PASS] Python: unmatched hydrogen raised Gaff2AtomTypeMissingError: {exc}")
        else:
            print(f"  [FAIL] Python: unexpected error message: {exc}")
    except Exception as exc:
        print(f"  [FAIL] Python: wrong exception type raised: {type(exc)}: {exc}")

    # Run Rust tests
    rust_tests_pass = False
    worktree_root = Path(__file__).resolve().parent.parent.parent
    cmd = ["cargo", "test", "-p", "proxide-gaff2"]
    env = os.environ.copy()
    env["RUSTC_WRAPPER"] = ""
    print("  [INFO] Running cargo test -p proxide-gaff2...")
    proc = subprocess.run(cmd, cwd=worktree_root, env=env, capture_output=True, text=True)
    if proc.returncode == 0:
        rust_tests_pass = True
        print("  [PASS] Rust: cargo test -p proxide-gaff2 passed")
    else:
        print(f"  [FAIL] Rust: cargo test failed with code {proc.returncode}:")
        print(f"{proc.stderr}\n{proc.stdout}")

    all_pass = (
        python_standard_rules_pass
        and python_unmatched_carbon_raises
        and python_unmatched_oxygen_raises
        and python_unmatched_hydrogen_raises
        and rust_tests_pass
    )

    results = {
        "all_pass": all_pass,
        "python_unmatched_carbon_raises": python_unmatched_carbon_raises,
        "python_unmatched_oxygen_raises": python_unmatched_oxygen_raises,
        "python_unmatched_hydrogen_raises": python_unmatched_hydrogen_raises,
        "python_standard_rules_pass": python_standard_rules_pass,
        "rust_tests_pass": rust_tests_pass,
    }

    # Write results payload
    results_path = os.environ.get("BTH_RESULTS_PATH")
    if not results_path:
        out_dir = worktree_root / "outputs" / "validation"
        out_dir.mkdir(parents=True, exist_ok=True)
        results_path = str(out_dir / "debt_1899_results.json")

    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  [INFO] Results written to {results_path}")

    # Also write adjacent fallback results file
    adjacent_results = Path(__file__).with_name("debt_1899_fail_fast_atom_typing.bth-results.json")
    with open(adjacent_results, "w") as f:
        json.dump(results, f, indent=2)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
