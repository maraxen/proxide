#!/usr/bin/env python3
"""Verification script for Debt #2375: _BOND_TYPE_SUB visibility & impropers emission."""

import io
import json
import os
import sys
from pathlib import Path

# Add src to sys.path
repo_root = Path(__file__).resolve().parent.parent.parent
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from rdkit import Chem
import openmm.app as app

from proxide.chem.gaff2 import (
    build_gaff2_ffxml,
    parameterize_gaff_with_rdkit,
    load_gaff2_parameters,
    _lookup_bond_params,
    _lookup_angle_params,
)


def main():
    print("Running verification for Debt #2375...")

    # 1. build_gaff2_ffxml on acetone emits <Improper> and OpenMM parses it
    mol_acetone = Chem.AddHs(Chem.MolFromSmiles("CC(=O)C"))
    charges_acetone = [0.0] * mol_acetone.GetNumAtoms()
    xml_acetone = build_gaff2_ffxml(mol_acetone, "ACE", charges_acetone)

    impropers_emitted = (
        ("<Improper" in xml_acetone)
        and ('type1="ACE_1"' in xml_acetone)
        and ('periodicity1="2"' in xml_acetone)
    )
    ff = app.ForceField(io.StringIO(xml_acetone))
    assert ff is not None, "OpenMM failed to load generated FFXML"
    print(f"1. Impropers emitted in FFXML: {impropers_emitted}")

    # 2. parameterize_gaff_with_rdkit returns impropers
    res_acetone = parameterize_gaff_with_rdkit(mol_acetone, charge_method="gasteiger")
    has_impropers_key = "impropers" in res_acetone
    num_acetone_impropers = len(res_acetone.get("impropers", []))
    impropers_returned = has_impropers_key and (num_acetone_impropers == 1)
    if impropers_returned:
        impr = res_acetone["impropers"][0]
        assert impr["center"] == 1
        assert abs(impr["pk"] - 10.5) < 1e-4
        assert abs(impr["phase"] - 180.0) < 1e-4
    print(f"2. Impropers returned for acetone: {impropers_returned} (count={num_acetone_impropers})")

    # Benzene check
    mol_benzene = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
    res_benzene = parameterize_gaff_with_rdkit(mol_benzene, charge_method="gasteiger")
    num_benzene_impropers = len(res_benzene.get("impropers", []))
    print(f"   Impropers returned for benzene: count={num_benzene_impropers}")

    # 3. _BOND_TYPE_SUB substitutions recorded in lookups
    params = load_gaff2_parameters()
    bond_subs = []
    kb, r0 = _lookup_bond_params("cx", "o", params, substitutions=bond_subs)
    bond_sub_recorded = (
        len(bond_subs) == 1
        and bond_subs[0] == {"term": "bond", "from": ["cx", "o"], "to": ["c3", "o"]}
        and kb > 0.0
        and r0 > 0.0
    )
    print(f"3. Bond substitutions recorded: {bond_sub_recorded} ({bond_subs})")

    angle_subs = []
    kt, t0 = _lookup_angle_params("cx", "c2", "hc", params, substitutions=angle_subs)
    angle_sub_recorded = (
        len(angle_subs) == 1
        and angle_subs[0] == {
            "term": "angle",
            "from": ["cx", "c2", "hc"],
            "to": ["c3", "c2", "hc"],
        }
        and kt > 0.0
        and t0 > 0.0
    )
    print(f"4. Angle substitutions recorded: {angle_sub_recorded} ({angle_subs})")

    result = {
        "impropers_emitted": bool(impropers_emitted),
        "impropers_returned": bool(impropers_returned),
        "bond_sub_recorded": bool(bond_sub_recorded),
        "angle_sub_recorded": bool(angle_sub_recorded),
        "num_acetone_impropers": int(num_acetone_impropers),
        "num_benzene_impropers": int(num_benzene_impropers),
    }

    # Write output payload for bathos
    bth_results_path = os.environ.get("BTH_RESULTS_PATH")
    if bth_results_path:
        with open(bth_results_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Wrote bathos results to BTH_RESULTS_PATH: {bth_results_path}")
    else:
        fallback_path = Path(__file__).with_name("verify_debt_2375.bth-results.json")
        with open(fallback_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Wrote bathos results to fallback path: {fallback_path}")

    all_ok = (
        impropers_emitted
        and impropers_returned
        and bond_sub_recorded
        and angle_sub_recorded
        and (num_acetone_impropers == 1)
        and (num_benzene_impropers == 6)
    )
    if not all_ok:
        sys.exit(1)
    print("All Debt #2375 verifications PASSED successfully.")


if __name__ == "__main__":
    main()
