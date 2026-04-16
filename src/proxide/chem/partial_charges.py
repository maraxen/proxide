"""Optional partial charge assignment via Espaloma Charge (AM1-BCC surrogate).

Requires the ``espaloma`` optional extra (``expaloma`` + RDKit): native JAX/Equinox inference
with bundled weights—no PyTorch or DGL at runtime. Charges are NumPy arrays in **atom index
order** matching the RDKit molecule or :class:`proxide.io.parsing.molecule.Molecule`.

**Conformer protocol:** Golden regression tests use ``Chem.MolFromSmiles`` **without**
embedding (same as upstream README for ``N#N``). Molecules built without RDKit sanitization
(e.g. from :meth:`~proxide.io.parsing.molecule.Molecule._to_rdkit`) are sanitized before
featurization.

**OpenFF path:** Use optional extra ``espaloma-openff`` and :func:`assign_espaloma_charges_openff`
(still uses upstream ``espaloma_charge`` toolkit wrappers).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from proxide.io.parsing.molecule import Molecule

CHARGE_SOURCE_ESPALOMA_AM1BCC = "espaloma-am1bcc"


def assign_espaloma_charges_rdkit(mol: Any) -> np.ndarray:
  """Assign partial charges using the JAX ``expaloma`` port (``charges_for_rdkit_mol``).

  Args:
      mol: RDKit molecule (implicit or explicit hydrogens per upstream protocol).

  Returns:
      Partial charges (n_atoms,) in elementary charge units, same atom order as ``mol``.

  Raises:
      ImportError: If ``expaloma`` is not installed.
  """
  try:
    from expaloma.infer import charges_for_rdkit_mol
  except ImportError as e:
    raise ImportError(
      "expaloma is not installed. Install proxide with optional dependency "
      "group [espaloma] (see proxide README for private GitHub install)."
    ) from e

  from rdkit import Chem

  mol = Chem.Mol(mol)
  Chem.SanitizeMol(mol)

  out = charges_for_rdkit_mol(mol)
  return np.asarray(out, dtype=np.float64)


def assign_espaloma_charges_from_proxide_molecule(molecule: Molecule) -> np.ndarray:
  """Build an RDKit mol from :class:`~proxide.io.parsing.molecule.Molecule` and assign charges.

  Atom order is guaranteed via ``AtomMapNum`` index preservation.
  """
  mol = molecule._to_rdkit()
  q_raw = assign_espaloma_charges_rdkit(mol)

  # Defensive sorting: ensure charges align with Molecule's original indices
  # Molecule atom i is mapped to RDKit atom j via atom.SetAtomMapNum(i + 1)
  q_final = np.zeros_like(q_raw)
  for j in range(mol.GetNumAtoms()):
    map_num = mol.GetAtomWithIdx(j).GetAtomMapNum()
    if map_num > 0:
      q_final[map_num - 1] = q_raw[j]
    else:
      # Fallback if map num is lost for some reason
      q_final[j] = q_raw[j]

  return q_final


def assign_espaloma_charges_openff(molecule: Any) -> np.ndarray:
  """Assign charges using OpenFF Toolkit + EspalomaCharge toolkit wrapper.

  Use when you already have an ``openff.toolkit.topology.Molecule``.

  Requires optional extra ``espaloma-openff`` (``espaloma_charge`` + OpenFF).

  Args:
      molecule: An OpenFF ``Molecule`` instance.

  Raises:
      ImportError: If ``openff-toolkit`` or ``espaloma_charge`` is missing.
  """
  try:
    from espaloma_charge.openff_wrapper import EspalomaChargeToolkitWrapper
  except ImportError as e:
    raise ImportError(
      "openff-toolkit and espaloma_charge are required for this entry point. "
      "Install proxide with optional extra [espaloma-openff]."
    ) from e

  registry = EspalomaChargeToolkitWrapper()
  molecule.assign_partial_charges("espaloma-am1bcc", toolkit_registry=registry)
  q = molecule.partial_charges.m_as("elementary_charge")
  return np.asarray(q, dtype=np.float64)
