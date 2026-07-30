"""Writing utilities for protein structures."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
  from proxide.core.containers import Protein


def write_pdb(protein: Protein, path: str | Path) -> Path:
  """Write a Protein structure to a PDB file.

  Args:
      protein: The Protein instance to write.
      path: Target file path.

  Returns:
      Path to the written file.
  """
  path = Path(path)

  # Ensure we have coordinates in Angstroms (Protein is already in Angstroms)
  coords = protein.coordinates
  if coords.ndim == 3:
    # Use CA for a simplified PDB if only 1 residue per row
    # Or properly format Atom37.
    # For now, let's use full_coordinates if available
    coords = (
      protein.full_coordinates if protein.full_coordinates is not None else coords.reshape(-1, 3)
    )

  atom_names = getattr(protein, "atom_names", None)
  res_names = getattr(protein, "res_names", None)
  res_ids = getattr(protein, "residue_index", None)
  chain_ids = getattr(protein, "chain_ids", None)
  elements = getattr(protein, "elements", None)

  # Fallbacks
  if atom_names is None:
    atom_names = ["CA"] * len(coords)
  if res_names is None:
    res_names = ["UNK"] * len(coords)
  if res_ids is None:
    res_ids = np.arange(1, len(coords) + 1)
  if chain_ids is None:
    chain_ids = ["A"] * len(coords)
  if elements is None:
    elements = [name[0] for name in atom_names]

  with open(path, "w") as f:
    for i in range(len(coords)):
      # PDB Format
      # 1-6   ATOM
      # 7-11  Atom serial number
      # 13-16 Atom name
      # 17    Alternate location indicator
      # 18-20 Residue name
      # 22    Chain identifier
      # 23-26 Residue sequence number
      # 27    Code for insertion of residues
      # 31-38 X
      # 39-46 Y
      # 47-54 Z
      # 55-60 Occupancy
      # 61-66 Temperature factor
      # 77-78 Element symbol

      # Handle potential mismatched lengths for flat arrays
      atom_name = atom_names[i] if i < len(atom_names) else "CA"
      res_name = res_names[i] if i < len(res_names) else "UNK"
      res_id = res_ids[i] if i < len(res_ids) else (i + 1)
      chain_id = (
        chain_ids[0]
        if chain_ids and len(chain_ids) == 1
        else (chain_ids[i] if i < len(chain_ids) else "A")
      )
      element = elements[i] if i < len(elements) else atom_name[0]

      x, y, z = coords[i]

      f.write(
        f"ATOM  {i+1:>5} {atom_name:<4} {res_name:>3} {chain_id}{res_id:>4}    "
        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}"
        f"  1.00  0.00          {element:>2}\n"
      )
    f.write("END\n")

  return path


def write_mmcif(protein: Protein, path: str | Path) -> Path:
  """Write a Protein structure to an mmCIF file.

  Args:
      protein: The Protein instance to write.
      path: Target file path.

  Returns:
      Path to the written file.
  """
  path = Path(path)

  coords = protein.coordinates
  if coords.ndim == 3:
    coords = (
      protein.full_coordinates if protein.full_coordinates is not None else coords.reshape(-1, 3)
    )

  atom_names = getattr(protein, "atom_names", None)
  res_names = getattr(protein, "res_names", None)
  res_ids = getattr(protein, "residue_index", None)
  chain_ids = getattr(protein, "chain_ids", None)
  elements = getattr(protein, "elements", None)

  if atom_names is None:
    atom_names = ["CA"] * len(coords)
  if res_names is None:
    res_names = ["UNK"] * len(coords)
  if res_ids is None:
    res_ids = np.arange(1, len(coords) + 1)
  if chain_ids is None:
    chain_ids = ["A"] * len(coords)
  if elements is None:
    elements = [name[0] for name in atom_names]

  with open(path, "w") as f:
    f.write(f"data_{path.stem}\n")
    f.write("#\n")
    f.write("loop_\n")
    f.write("_atom_site.group_PDB\n")
    f.write("_atom_site.id\n")
    f.write("_atom_site.type_symbol\n")
    f.write("_atom_site.label_atom_id\n")
    f.write("_atom_site.label_comp_id\n")
    f.write("_atom_site.label_asym_id\n")
    f.write("_atom_site.label_seq_id\n")
    f.write("_atom_site.Cartn_x\n")
    f.write("_atom_site.Cartn_y\n")
    f.write("_atom_site.Cartn_z\n")
    f.write("_atom_site.occupancy\n")
    f.write("_atom_site.B_iso_or_equiv\n")

    for i in range(len(coords)):
      atom_name = atom_names[i] if i < len(atom_names) else "CA"
      res_name = res_names[i] if i < len(res_names) else "UNK"
      res_id = res_ids[i] if i < len(res_ids) else (i + 1)
      chain_id = (
        chain_ids[0]
        if chain_ids and len(chain_ids) == 1
        else (chain_ids[i] if i < len(chain_ids) else "A")
      )
      element = elements[i] if i < len(elements) else atom_name[0]

      x, y, z = coords[i]

      f.write(
        f"ATOM {i+1} {element} {atom_name} {res_name} {chain_id} {res_id} "
        f"{x:.3f} {y:.3f} {z:.3f} 1.00 0.00\n"
      )
  return path


def write_npz(protein: Protein, path: str | Path) -> Path:
  """Write Protein structural data to a JAX-ready NPZ file.

  Args:
      protein: The Protein instance to export.
      path: Target file path.

  Returns:
      Path to the written file.
  """
  path = Path(path)

  # Basic fields for structural reconstruction
  data = {
    "coordinates": np.asarray(protein.coordinates),
    "aatype": np.asarray(protein.aatype),
    "residue_index": np.asarray(protein.residue_index),
    "chain_index": np.asarray(protein.chain_index),
    "atom_mask": np.asarray(protein.atom_mask) if protein.atom_mask is not None else None,
  }

  # Optional physics fields
  if protein.charges is not None:
    data["charges"] = np.asarray(protein.charges)
  if protein.sigmas is not None:
    data["sigmas"] = np.asarray(protein.sigmas)
  if protein.epsilons is not None:
    data["epsilons"] = np.asarray(protein.epsilons)

  # Filter out None values
  data = {k: v for k, v in data.items() if v is not None}

  np.savez_compressed(path, **data)
  return path
