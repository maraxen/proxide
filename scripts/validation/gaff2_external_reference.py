"""Cross-check proxide's GAFF2 atom typing against a real external reference.

Shells out to AmberTools' `antechamber` (via a pre-existing conda-forge
micromamba environment, `espaloma-smoke` -- confirmed present and working on
this machine 2026-08-20; not a proxide/naurmalade-managed environment, so its
path is resolved at runtime rather than assumed to exist everywhere this repo
is checked out) to get ground-truth GAFF2 (gaff-2.11) atom types for a small
set of molecules where proxide's implementation had open questions:

- AR1/AR2/AR3 aromaticity sub-classification (furan/pyrrole/thiophene): the
  ATOMTYPE_GFF2.DEF footer gives no algorithm to compute which 5-membered
  heteroaromatic ring carbons are AR1 ("pure aromatic", -> ca) vs AR2/AR3
  ("planar conjugated ring", -> cc/cd).
- The naphthalene bridgehead `1RG6` ring-count reading (exact-count vs.
  "a 6-ring is present" as one qualifying condition).

`-c dc` skips antechamber's charge computation (AM1-BCC/sqm) entirely -- this
script only needs atom TYPES, and charge calculation is both slow and
irrelevant here.

Usage:
    uv run python scripts/validation/gaff2_external_reference.py
    ANTECHAMBER=/path/to/antechamber uv run python scripts/validation/gaff2_external_reference.py
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

from proxide.chem.gaff2 import assign_gaff2_atom_types

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("gaff2_external_reference")

_DEFAULT_ANTECHAMBER = str(
    Path.home() / ".local/share/mamba/envs/espaloma-smoke/bin/antechamber"
)

MOLECULES = {
    "furan": "c1ccoc1",
    "pyrrole": "c1cc[nH]c1",
    "thiophene": "c1ccsc1",
    "naphthalene": "c1ccc2ccccc2c1",
}


def antechamber_path() -> str:
    path = os.environ.get("ANTECHAMBER", _DEFAULT_ANTECHAMBER)
    if not Path(path).exists():
        raise FileNotFoundError(
            f"antechamber not found at {path!r}. Set ANTECHAMBER=/path/to/antechamber "
            f"(a conda-forge/micromamba AmberTools environment has it; see this "
            f"script's module docstring for the environment confirmed working "
            f"2026-08-20 on this machine)."
        )
    return path


def run_antechamber_gaff2(smiles: str) -> list[tuple[str, str]]:
    """Return [(element, gaff2_type), ...] in RDKit atom order, via antechamber."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Failed to parse SMILES: {smiles!r}")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)

    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = Path(tmpdir) / "mol.mol"
        out_path = Path(tmpdir) / "mol_out.mol2"
        Chem.MolToMolFile(mol, str(in_path))

        result = subprocess.run(
            [
                antechamber_path(),
                "-i", str(in_path), "-fi", "mdl",
                "-o", str(out_path), "-fo", "mol2",
                "-at", "gaff2",
                "-c", "dc",  # skip charge computation -- types only
                "-pf", "y",
                "-s", "0",
            ],
            cwd=tmpdir,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0 or not out_path.exists():
            raise RuntimeError(
                f"antechamber failed for {smiles!r} (exit {result.returncode}):\n"
                f"{result.stdout}\n{result.stderr}"
            )
        return _parse_mol2_atom_types(out_path, mol)


def _parse_mol2_atom_types(mol2_path: Path, mol: Chem.Mol) -> list[tuple[str, str]]:
    lines = mol2_path.read_text().splitlines()
    start = lines.index("@<TRIPOS>ATOM") + 1
    end = lines.index("@<TRIPOS>BOND")
    atom_lines = lines[start:end]
    if len(atom_lines) != mol.GetNumAtoms():
        raise RuntimeError(
            f"antechamber output has {len(atom_lines)} atoms, expected "
            f"{mol.GetNumAtoms()} -- atom order/count mismatch, cannot align "
            f"safely."
        )
    parsed = []
    for atom, line in zip(mol.GetAtoms(), atom_lines, strict=True):
        gaff_type = line.split()[5]
        parsed.append((atom.GetSymbol(), gaff_type))
    return parsed


def proxide_gaff2(smiles: str) -> list[tuple[str, str]]:
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.SanitizeMol(mol)
    types = assign_gaff2_atom_types(mol)
    # zip(strict=False): assign_gaff2_atom_types currently returns a
    # heavy-atom-only compacted list (pre Phase B of the H-atom typing
    # rework), shorter than mol.GetAtoms() when H atoms are present -- matches
    # tests/test_gaff2_golden.py's existing convention. Once Phase B lands
    # (full-index-aligned return contract), this can drop the `if
    # atom.GetAtomicNum() != 1` filter and use strict=True.
    return [
        (atom.GetSymbol(), t)
        for atom, t in zip(mol.GetAtoms(), types, strict=False)
        if atom.GetAtomicNum() != 1
    ]


def main() -> None:
    for name, smiles in MOLECULES.items():
        reference = [t for elem, t in run_antechamber_gaff2(smiles) if elem != "H"]
        actual = [t for elem, t in proxide_gaff2(smiles)]
        match = "MATCH" if reference == actual else "DIFFERS"
        logger.info("%-12s antechamber=%-40s proxide=%-40s %s", name, reference, actual, match)


if __name__ == "__main__":
    main()
