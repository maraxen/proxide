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


def assign_espaloma_charges_rdkit(
    mol: Any,
    *,
    backend: str = "rust",
) -> np.ndarray:
    """Assign partial charges using the Expaloma engine (JAX or Native Rust).

    Args:
        mol: RDKit molecule (implicit or explicit hydrogens per upstream protocol).
        backend: Backend to use, either "rust" (default, GIL-free) or "jax" (original).

    Returns:
        Partial charges (n_atoms,) in elementary charge units, same atom order as ``mol``.

    Raises:
        ImportError: If the chosen backend's dependencies (expaloma for JAX) are not installed.
        ValueError: If an invalid backend is specified.
    """
    from rdkit import Chem

    # Deep copy and sanitize to ensure consistent featurization
    mol = Chem.Mol(mol)
    Chem.SanitizeMol(mol)

    if backend == "jax":
        try:
            from expaloma.infer import charges_for_rdkit_mol
        except ImportError as e:
            raise ImportError(
                "expaloma is not installed. Install proxide with [espaloma] extra."
            ) from e
        out = charges_for_rdkit_mol(mol)
        return np.asarray(out, dtype=np.float64)

    elif backend == "rust":
        from proxide._proxider import assign_espaloma_charges

        try:
            from expaloma.featurize import from_rdkit_mol
        except ImportError as e:
            raise ImportError(
                "expaloma (for featurization) is not installed. Install proxide with [espaloma] extra."
            ) from e

        # 1. Featurize (still using Python/RDKit part of expaloma)
        g = from_rdkit_mol(mol)

        # 2. Extract into contiguous arrays for Rust FFI
        h0 = np.ascontiguousarray(g.h0, dtype=np.float32)
        senders = np.ascontiguousarray(g.senders, dtype=np.uint32)
        receivers = np.ascontiguousarray(g.receivers, dtype=np.uint32)
        q_ref = np.ascontiguousarray(g.q_ref, dtype=np.float32)

        # Total charge per molecule (sum of formal charges)
        # For a single molecule, num_graphs=1 and total_charges is a list of 1 scalar
        total_charge = float(q_ref.sum())

        # 3. Call native Rust inference (GIL is released inside)
        q_rust = assign_espaloma_charges(
            h0,
            senders,
            receivers,
            np.zeros(h0.shape[0], dtype=np.uint32),  # segment_ids
            1,  # num_graphs
            [total_charge],
        )

        return np.asarray(q_rust, dtype=np.float64)

    else:
        raise ValueError(f"Invalid backend: {backend}. Use 'rust' or 'jax'.")


def assign_espaloma_charges_from_proxide_molecule(
    molecule: Molecule,
    *,
    backend: str = "rust",
) -> np.ndarray:
    """Build an RDKit mol from :class:`~proxide.io.parsing.molecule.Molecule` and assign charges.

    Atom order is guaranteed via ``AtomMapNum`` index preservation.
    """
    mol = molecule._to_rdkit()
    q_raw = assign_espaloma_charges_rdkit(mol, backend=backend)

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
