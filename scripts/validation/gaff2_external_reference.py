"""Cross-check proxide's GAFF2 atom typing against a real external reference.

Shells out to AmberTools' `antechamber` (via a pre-existing conda-forge
micromamba environment, `espaloma-smoke` -- confirmed present and working on
this machine 2026-08-20; not a proxide/naurmalade-managed environment, so its
path is resolved at runtime rather than assumed to exist everywhere this repo
is checked out) to get ground-truth GAFF2 (gaff-2.11) atom types.

EXPANDED 260820 (post-merge PARITY audit): a 3-agent audit of the merged
"PARITY" verdict found that most of the molecules the session claimed to
verify against antechamber were only ever checked via ephemeral, untracked
bash/python one-liners -- this script previously covered only 4 of them. Now
covers every molecule actually used to justify a claim in the verdict report
or a code comment, compares BOTH heavy and H atoms (assign_gaff2_atom_types's
return contract is full-index-aligned as of the 260820 H-typing rework, so
there's no reason to only check heavy atoms anymore), and exits non-zero on
any mismatch so this can gate CI instead of only printing to a log a human
has to read.

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
import sys
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

# (name, smiles, net_charge) -- what each molecule verifies, and why it's here:
MOLECULES: list[tuple[str, str, int]] = [
    # AR1/AR2/AR3 five-membered heteroaromatics (cc/cd family vs ca).
    ("furan", "c1ccoc1", 0),
    ("pyrrole", "c1cc[nH]c1", 0),
    ("thiophene", "c1ccsc1", 0),
    ("imidazole", "C1=CN=CN1", 0),
    # cc/cd-family chains that correctly do NOT split (no double bond
    # connects two same-family atoms).
    ("1,3-butadiene", "C=CC=C", 0),
    ("divinyl-ketone", "C=CC(=O)C=C", 0),
    ("acrolein", "C=CC=O", 0),
    # cp/cq: real biphenyl connector, correctly stays cp/cp (single
    # inter-ring bond, no flip).
    ("biphenyl", "c1ccc(-c2ccccc2)cc1", 0),
    # AR1 ring-count/ring-size sanity: naphthalene's bridgeheads (2 six-rings
    # each) must stay ca, not be wrongly demoted by the exocyclic-multibond
    # check (which must key on ANY-ring membership, not just "this ring").
    ("naphthalene", "c1ccc2ccccc2c1", 0),
    ("anthracene", "c1ccc2cc3ccccc3cc2c1", 0),
    # AR1 exocyclic-double-bond exception (2-pyranone) and its boundary
    # (benzaldehyde/styrene must NOT be demoted -- the double bond is one
    # atom removed from the ring, not directly on a ring atom).
    ("2-pyranone", "O=C1C=CC=CO1", 0),
    ("benzaldehyde", "O=Cc1ccccc1", 0),
    ("styrene", "C=Cc1ccccc1", 0),
    # H-atom typing (h1-h5/hx/hn/ho/hs/hp electron-withdrawing-neighbor
    # family + _EW_ATOMS element set, including sulfur).
    ("methane", "C", 0),
    ("ethanol", "CCO", 0),
    ("formamide", "NC=O", 0),
    ("chloroform", "ClC(Cl)(Cl)[H]", 0),
    ("chloromethane", "CCl", 0),
    ("dichloromethane", "ClCCl", 0),
    ("tetramethylammonium", "C[N+](C)(C)C", 1),
    ("water", "O", 0),
]


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


def run_antechamber_gaff2(smiles: str, net_charge: int = 0) -> list[tuple[str, str]]:
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
                "-nc", str(net_charge),
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
    # assign_gaff2_atom_types returns one type per atom index (heavy + H)
    # since the 260820 H-typing rework -- see its own docstring for the
    # contract. strict=True: a length mismatch here is a real bug, not
    # something to silently paper over.
    types = assign_gaff2_atom_types(mol)
    return list(zip((a.GetSymbol() for a in mol.GetAtoms()), types, strict=True))


def main() -> int:
    any_mismatch = False
    for name, smiles, net_charge in MOLECULES:
        reference = run_antechamber_gaff2(smiles, net_charge)
        actual = proxide_gaff2(smiles)
        match = reference == actual
        any_mismatch = any_mismatch or not match
        status = "MATCH" if match else "DIFFERS"
        logger.info("%-22s %s", name, status)
        if not match:
            logger.info("  antechamber: %s", reference)
            logger.info("  proxide:     %s", actual)
    if any_mismatch:
        logger.info("\nFAIL: one or more molecules differ from the real antechamber reference.")
        return 1
    logger.info(
        "\nAll %d molecules MATCH the real antechamber (gaff-2.11) reference.",
        len(MOLECULES),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
