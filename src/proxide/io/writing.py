"""Writing utilities for protein structures."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
  from proxide.core.containers import Protein


def _reject_if_batched(protein: Protein, fn_name: str) -> None:
  """Raise loudly if `protein` looks like a batched (multi-row) Protein.

  write_pdb/write_mmcif write a single structure to a single file. Since the
  `_stack_padded_proteins` fix, a batched Protein's `chain_ids` is
  `list[list[str] | None]` (one entry per batch row) rather than the single
  structure's `list[str]`. Indexing that per-row list as if it were a flat
  per-atom chain-letter list would silently stamp a Python list repr (e.g.
  "['A', 'B']") into the fixed-width PDB/mmCIF chain-ID column, corrupting the
  file rather than merely mislabeling it. Batched coordinates (an extra
  leading batch axis) are an independent, equally-unsupported case -- reject
  both rather than attempt to guess which row was meant.
  """
  chain_ids = getattr(protein, "chain_ids", None)
  if chain_ids and isinstance(chain_ids[0], (list, tuple)):
    msg = (
      f"{fn_name}() writes a single structure, but `protein.chain_ids` is "
      f"list[list[str]] -- this looks like a batched Protein (e.g. the output "
      f"of pad_and_collate_proteins). Index into the batch first, e.g. "
      f"`protein.replace(chain_ids=protein.chain_ids[row], "
      f"coordinates=protein.coordinates[row], ...)` for the fields you need, "
      f"or write each row separately."
    )
    raise ValueError(msg)
  coords = getattr(protein, "coordinates", None)
  if coords is not None and coords.ndim == 4:
    msg = (
      f"{fn_name}() writes a single structure, but `protein.coordinates` has "
      f"{coords.ndim} dimensions (expected 3, (N_res, atoms_per_res, 3)) -- "
      f"this looks like a batched Protein. Index into the batch axis first."
    )
    raise ValueError(msg)


def _resolve_chain_letters(protein: Protein, n_rows: int) -> list[str]:
  """Resolve a chain letter for each of the `n_rows` flattened coordinate rows.

  `chain_ids` is a per-CHAIN vocabulary (Shape (N_chains,), e.g. ["A", "B"]) --
  see `Protein.from_rust_dict`, which populates it from `unique_chain_ids` -- and
  `chain_index` is per-RESIDUE (Shape (N_res,)) for Atom37/Atom14-format Proteins.
  Neither is safe to index directly by a flattened per-atom-slot row index: the
  previous code did `chain_ids[i]` for `i` up to `len(coords) - 1`, which for an
  Atom37 Protein is `len(full_coordinates) == N_res * atoms_per_residue`, far
  larger than `len(chain_ids) == N_chains` -- so every residue past the first
  `N_chains` atom-slots silently fell back to "A".

  Resolve through `chain_index`, expanded to match whatever per-residue-slot
  flattening produced `n_rows` rows of coordinates, then look up each row's
  chain letter in `chain_ids`.
  """
  chain_ids = getattr(protein, "chain_ids", None)
  chain_index = getattr(protein, "chain_index", None)
  if not chain_ids or chain_index is None:
    return ["A"] * n_rows

  chain_index = np.asarray(chain_index)
  n_res = chain_index.shape[0]
  if n_res == 0:
    return ["A"] * n_rows
  if n_rows == n_res:
    # Already per-residue-aligned (e.g. the flat "Full" format).
    row_chain_index = chain_index
  elif n_rows % n_res == 0:
    # Atom37/Atom14: each residue expands to n_rows // n_res atom slots.
    row_chain_index = np.repeat(chain_index, n_rows // n_res)
  else:
    # Cannot align chain_index to the coordinate rows -- degrade to the
    # single-chain default rather than guess at a misaligned mapping.
    return ["A"] * n_rows

  return [chain_ids[idx] if 0 <= idx < len(chain_ids) else "A" for idx in row_chain_index]


def _reject_nonfinite(coords: np.ndarray, fn_name: str) -> None:
  """Raise if any coordinate is NaN/inf -- "nan" fits an 8-char PDB column."""
  bad = ~np.isfinite(np.asarray(coords, dtype=np.float64)).all(axis=-1)
  if bad.any():
    msg = (
      f"{fn_name}: {int(bad.sum())} atom(s) have non-finite coordinates "
      f"(first at row {int(np.argmax(bad))}); refusing to write them"
    )
    raise ValueError(msg)


def _atom_rows(
  protein: Protein,
  fn_name: str,
) -> tuple[np.ndarray, list[str], list[str], np.ndarray, list[str], list[str]]:
  """Resolve the per-atom rows `write_pdb`/`write_mmcif` emit, without inventing any.

  Returns (coords, atom_names, res_names, res_seqs, chain_letters, elements),
  all of one length. Two sources are supported:

  * per-atom fields (``atom_names`` present, e.g. the flat "Full" format):
    ``res_names`` and per-atom residue ids are required, and every per-atom
    field must match the coordinate count exactly;
  * Atom37 (no ``atom_names``): names come from the Atom37 vocabulary,
    residue names from ``aatype``, and only slots flagged in ``atom_mask``
    are written -- an unresolved slot is zero-filled, so without the mask a
    real atom cannot be told from a hole.

  Anything else raises. The previous writer filled gaps with "CA", "UNK",
  ``i + 1`` and ``atom_name[0]`` (debt #1928, ledger A1/A5); unknown elements
  are returned as "" -- written blank in PDB (which proxide's reader resolves
  through the canonical ``infer_element``) and as the CIF unknown marker "?"
  in mmCIF.
  """
  atom_names = getattr(protein, "atom_names", None)
  elements = getattr(protein, "elements", None)

  if atom_names is not None:
    coords = np.asarray(
      protein.full_coordinates if protein.full_coordinates is not None else protein.coordinates
    )
    if coords.ndim != 2:
      msg = (
        f"{fn_name}: protein.atom_names is per-atom ({len(atom_names)}) but no flat "
        f"(N_atoms, 3) coordinates are available (got shape {coords.shape})"
      )
      raise ValueError(msg)
    n = coords.shape[0]
    res_names = getattr(protein, "res_names", None)
    res_seqs = protein.atom_residue_ids
    fields = {
      "atom_names": atom_names,
      "res_names": res_names,
      "residue ids": res_seqs,
      "elements": elements if elements is not None else [""] * n,
    }
    for field_name, values in fields.items():
      if values is None:
        msg = f"{fn_name}: per-atom {field_name} are missing; refusing to invent them"
        raise ValueError(msg)
      if len(values) != n:
        msg = (
          f"{fn_name}: {len(values)} {field_name} for {n} atoms; refusing to pad or "
          "truncate a per-atom field"
        )
        raise ValueError(msg)
    assert res_names is not None  # narrowed by the loop above
    return (
      coords,
      [str(a) for a in atom_names],
      [str(r) for r in res_names],
      np.asarray(res_seqs),
      _resolve_chain_letters(protein, n),
      [str(e) for e in fields["elements"]],
    )

  coords = np.asarray(protein.coordinates)
  from proxide.chem.residues import atom_types, resnames

  if coords.ndim != 3 or coords.shape[1] != len(atom_types):
    msg = (
      f"{fn_name}: protein has no per-atom atom_names and its coordinates are not "
      f"Atom37 (got shape {coords.shape}); cannot name the atoms without inventing them"
    )
    raise ValueError(msg)
  if protein.atom_mask is None:
    msg = (
      f"{fn_name}: Atom37 protein has no atom_mask, so resolved atoms cannot be told "
      "from zero-filled empty slots; refusing to write every slot"
    )
    raise ValueError(msg)
  atom_mask = np.asarray(protein.atom_mask).astype(bool)
  residue_index = np.asarray(protein.residue_index)
  aatype = np.asarray(protein.aatype)
  res_chain = _resolve_chain_letters(protein, coords.shape[0])

  res_idx, slot_idx = np.nonzero(atom_mask)
  return (
    coords[res_idx, slot_idx],
    [atom_types[k] for k in slot_idx],
    [resnames[int(aatype[r])] for r in res_idx],
    residue_index[res_idx],
    [res_chain[r] for r in res_idx],
    [""] * len(res_idx),
  )


def write_pdb(protein: Protein, path: str | Path) -> Path:
  """Write a Protein structure to a PDB file.

  Args:
      protein: The Protein instance to write.
      path: Target file path.

  Returns:
      Path to the written file.

  Raises:
      ValueError: If a field would have to be invented (see `_atom_rows`),
          or if a value does not fit its fixed-width PDB column (more than
          99999 atoms, a residue number outside -999..9999, an over-long name,
          or a coordinate outside -999.999..9999.999) -- the old writer
          silently shifted every later column (debt #1927). Use
          `write_mmcif` for such structures.
  """
  path = Path(path)
  _reject_if_batched(protein, "write_pdb")

  coords, atom_names, res_names, res_seqs, chain_letters, elements = _atom_rows(
    protein, "write_pdb"
  )
  _reject_nonfinite(coords, "write_pdb")

  def _overflow(what: str) -> ValueError:
    return ValueError(f"write_pdb: {what} does not fit the fixed-width PDB format; use write_mmcif")

  if len(coords) > 99999:
    raise _overflow(f"{len(coords)} atoms (serial > 99999)")

  lines = []
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
    atom_name, res_name, element = atom_names[i], res_names[i], elements[i]
    res_seq = int(res_seqs[i])
    chain_id = chain_letters[i]
    if len(atom_name) > 4:
      raise _overflow(f"atom name {atom_name!r} (atom {i})")
    if len(res_name) > 3:
      raise _overflow(f"residue name {res_name!r} (atom {i})")
    if len(element) > 2:
      raise _overflow(f"element {element!r} (atom {i})")
    if len(chain_id) != 1:
      raise _overflow(f"chain id {chain_id!r} (atom {i})")
    if not -999 <= res_seq <= 9999:
      raise _overflow(f"residue number {res_seq} (atom {i})")
    # The criterion is the formatted width itself (8 chars, %8.3f), not a
    # raw-float range: float32 -999.999 is -999.99902..., which still prints
    # as "-999.999" and fits.
    xyz = [f"{float(c):>8.3f}" for c in coords[i]]
    if any(len(c) > 8 for c in xyz):
      raise _overflow(f"coordinate ({', '.join(c.strip() for c in xyz)}) (atom {i})")

    lines.append(
      f"ATOM  {i + 1:>5} {atom_name:<4} {res_name:>3} {chain_id}{res_seq:>4}    "
      f"{''.join(xyz)}"
      f"  1.00  0.00          {element:>2}\n"
    )

  # Validate everything before touching the file, so a rejected structure
  # never leaves a truncated PDB behind.
  with open(path, "w") as f:
    f.writelines(lines)
    f.write("END\n")

  return path


def write_mmcif(protein: Protein, path: str | Path) -> Path:
  """Write a Protein structure to an mmCIF file.

  Args:
      protein: The Protein instance to write.
      path: Target file path.

  Returns:
      Path to the written file.

  Raises:
      ValueError: If a field would have to be invented (see `_atom_rows`).
  """
  path = Path(path)
  _reject_if_batched(protein, "write_mmcif")

  coords, atom_names, res_names, res_seqs, chain_letters, elements = _atom_rows(
    protein, "write_mmcif"
  )
  _reject_nonfinite(coords, "write_mmcif")

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
      element = elements[i] or "?"  # CIF's own "unknown" marker
      x, y, z = coords[i]

      f.write(
        f"ATOM {i + 1} {element} {atom_names[i]} {res_names[i]} {chain_letters[i]} "
        f"{int(res_seqs[i])} {x:.3f} {y:.3f} {z:.3f} 1.00 0.00\n"
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

  np.savez_compressed(path, **data)  # ty: ignore[invalid-argument-type]
  return path
