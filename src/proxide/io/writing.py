"""Writing utilities for protein structures."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from proxide import _proxider

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


def _resolve_chain_letters(protein: Protein, n_rows: int, fn_name: str = "write") -> list[str]:
  """Resolve a chain id for each of `n_rows` rows, or "" where it is unknown.

  `chain_ids` is a per-CHAIN vocabulary (Shape (N_chains,), e.g. ["A", "B"]) --
  see `Protein.from_rust_dict`, which populates it from `unique_chain_ids` -- and
  `chain_index` maps each row to an entry in it. Callers pass rows that
  `chain_index` is aligned with: residues for Atom37/Atom14, atoms for per-atom
  Proteins.

  * No chain information at all -> "" for every row: unknown, written as the
    PDB's blank chain column / mmCIF "?".
  * `chain_index` not aligned with the rows, or pointing outside `chain_ids`
    -> ValueError.

  This used to return "A" in all of those cases and to `np.repeat` a
  per-residue `chain_index` over atom rows whenever the counts happened to
  divide, which silently merged or mislabeled chains (debt #2354, ledger A1).
  """
  chain_ids = getattr(protein, "chain_ids", None)
  chain_index = getattr(protein, "chain_index", None)
  if not chain_ids or chain_index is None:
    return [""] * n_rows

  chain_index = np.asarray(chain_index)
  if chain_index.shape != (n_rows,):
    msg = (
      f"{fn_name}: chain_index has shape {chain_index.shape} but there are {n_rows} "
      "rows to label; refusing to guess an alignment"
    )
    raise ValueError(msg)
  bad = sorted({int(i) for i in chain_index if not 0 <= int(i) < len(chain_ids)})
  if bad:
    msg = (
      f"{fn_name}: chain_index values {bad} are outside chain_ids "
      f"(len {len(chain_ids)}); refusing to relabel them"
    )
    raise ValueError(msg)
  return [str(chain_ids[int(i)]) for i in chain_index]


# Element of each Atom37 slot. Definitional data for a closed protein
# heavy-atom vocabulary, written out per name -- not derived from the name
# string (ledger A2/A5). Kept in lockstep with residues.atom_types by test.
_ATOM37_ELEMENT = {
  "N": "N", "CA": "C", "C": "C", "CB": "C", "O": "O", "CG": "C", "CG1": "C",
  "CG2": "C", "OG": "O", "OG1": "O", "SG": "S", "CD": "C", "CD1": "C",
  "CD2": "C", "ND1": "N", "ND2": "N", "OD1": "O", "OD2": "O", "SD": "S",
  "CE": "C", "CE1": "C", "CE2": "C", "CE3": "C", "NE": "N", "NE1": "N",
  "NE2": "N", "OE1": "O", "OE2": "O", "CH2": "C", "NH1": "N", "NH2": "N",
  "OH": "O", "CZ": "C", "CZ2": "C", "CZ3": "C", "NZ": "N", "OXT": "O",
}  # fmt: skip


def _reject_nonfinite(coords: np.ndarray, fn_name: str) -> None:
  """Raise if any written coordinate is NaN/inf -- "nan" fits an 8-char PDB column.

  Callers pass only the rows to be written: for Atom37/Atom14 that is the
  atom_mask-selected rows (see `_atom_rows`), so an unplaced zero-filled slot is
  never checked here. Use atom_mask, not NaN, to find missing atoms.
  """
  bad = ~np.isfinite(np.asarray(coords, dtype=np.float64)).all(axis=-1)
  if bad.any():
    msg = (
      f"{fn_name}: {int(bad.sum())} atom(s) have non-finite coordinates "
      f"(first at row {int(np.argmax(bad))}); refusing to write them"
    )
    raise ValueError(msg)


# The Rust formatters' residue alias policies and HETATM residues, read from the extension
# (crates/proxide-io/src/formatters/residue_policy.rs is the single source). Per array format:
# as-read residue name -> (layout residue, {structure atom: layout atom}).
_LAYOUT_ALIASES = {
  fmt: {res: (layout_res, dict(renames)) for res, (layout_res, renames) in table.items()}
  for fmt, table in _proxider._layout_aliases().items()
}
_HETATM_RESIDUES = frozenset(_proxider._hetatm_residues())


def _layout_alias(layout: str, res_name: str):
  return _LAYOUT_ALIASES.get(layout, {}).get(res_name)


def _written_atom_name(layout: str, res_name: str, layout_atom: str) -> str:
  """The structure atom name for a layout atom, undoing the format's alias renames."""
  alias = _layout_alias(layout, res_name)
  if alias is None:
    return layout_atom
  back = {layout_name: name for name, layout_name in alias[1].items()}
  return back.get(layout_atom, layout_atom)


def _per_residue_names(protein: Protein, n_res: int, fn_name: str) -> list[str] | None:
  residue_names = getattr(protein, "residue_names", None)
  if residue_names is None:
    return None
  names = [str(r) for r in residue_names]
  if len(names) != n_res:
    msg = (
      f"{fn_name}: {len(names)} residue_names for {n_res} residues; refusing to pad or "
      "truncate a per-residue field"
    )
    raise ValueError(msg)
  return names


def _atom_rows(
  protein: Protein,
  fn_name: str,
) -> tuple[
  np.ndarray,
  list[str],
  list[str],
  np.ndarray,
  list[str],
  list[str],
  np.ndarray | None,
  np.ndarray | None,
]:
  """Resolve the per-atom rows `write_pdb`/`write_mmcif` emit, without inventing any.

  Returns (coords, atom_names, res_names, res_seqs, chain_letters, elements,
  b_factors, occupancies), the first six of one length. b_factors and occupancies
  are the Protein's own values for those rows, or None when it carries none.
  Two sources are supported:

  * per-atom fields (``atom_names`` present, e.g. the flat "Full" format):
    ``res_names`` and per-atom residue ids are required, and every per-atom
    field must match the coordinate count exactly;
  * Atom37 / Atom14 (no ``atom_names``): names come from the Atom37
    vocabulary or the per-residue-type Atom14 table, residue names from
    ``aatype``, and only slots flagged in ``atom_mask`` are written -- an
    unresolved slot is zero-filled, so without the mask a real atom cannot
    be told from a hole.

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
    # Per-atom chain ids, when the Protein carries them (Full format,
    # backlog #5684), are the ground truth; otherwise resolve through the
    # per-chain vocabulary, which needs a per-atom chain_index.
    atom_chain_ids = getattr(protein, "atom_chain_ids", None)
    if atom_chain_ids is not None:
      if len(atom_chain_ids) != n:
        msg = (
          f"{fn_name}: {len(atom_chain_ids)} atom_chain_ids for {n} atoms; refusing to "
          "pad or truncate a per-atom field"
        )
        raise ValueError(msg)
      chain_letters = [str(c) for c in atom_chain_ids]
    else:
      chain_letters = _resolve_chain_letters(protein, n, fn_name)
    return (
      coords,
      [str(a) for a in atom_names],
      [str(r) for r in res_names],
      np.asarray(res_seqs),
      chain_letters,
      [str(e) for e in fields["elements"]],
      _per_row_values(getattr(protein, "b_factors", None), (n,), "b_factors", fn_name),
      _per_row_values(getattr(protein, "occupancies", None), (n,), "occupancies", fn_name),
    )

  coords = np.asarray(protein.coordinates)
  from proxide.chem.residues import atom_types, resnames, restype_name_to_atom14_names

  # Dispatch on shape, not `protein.format`: from_rust_dict labels Atom14
  # output "Atom37".
  n_slots = coords.shape[1] if coords.ndim == 3 else None
  if n_slots not in (len(atom_types), 14):
    msg = (
      f"{fn_name}: protein has no per-atom atom_names and its coordinates are neither "
      f"Atom37 nor Atom14 (got shape {coords.shape}); cannot name the atoms without "
      "inventing them"
    )
    raise ValueError(msg)
  layout = "Atom37" if n_slots == len(atom_types) else "Atom14"
  if protein.atom_mask is None:
    msg = (
      f"{fn_name}: {layout} protein has no atom_mask, so resolved atoms cannot be told "
      "from zero-filled empty slots; refusing to write every slot"
    )
    raise ValueError(msg)
  atom_mask = np.asarray(protein.atom_mask).astype(bool)
  residue_index = np.asarray(protein.residue_index)
  aatype = np.asarray(protein.aatype)
  res_chain = _resolve_chain_letters(protein, coords.shape[0], fn_name)
  residue_names = _per_residue_names(protein, coords.shape[0], fn_name)

  res_idx, slot_idx = np.nonzero(atom_mask)
  # aatype indexes `resnames` (20 = the explicit "UNK" class). Anything outside
  # that range is a sentinel (e.g. -1 padding), and Python's negative indexing
  # would silently turn it into a real residue name.
  bad = sorted({int(aatype[r]) for r in res_idx} - set(range(len(resnames))))
  if bad:
    msg = (
      f"{fn_name}: aatype values {bad} on residues with resolved atoms are outside "
      f"0..{len(resnames) - 1}; refusing to name those residues"
    )
    raise ValueError(msg)
  if layout == "Atom37":
    layout_names = [atom_types[k] for k in slot_idx]
  else:
    # Atom14 slots are per-residue-type: slot k of an ALA is not slot k of a
    # TRP. The table gives "" for slots that residue type does not have, so a
    # masked "" slot is inconsistent data, not something to name.
    layout_names = [
      restype_name_to_atom14_names[resnames[int(aatype[r])]][k]
      for r, k in zip(res_idx, slot_idx, strict=True)
    ]
    empty = [
      (int(r), int(k)) for r, k, n in zip(res_idx, slot_idx, layout_names, strict=True) if not n
    ]
    if empty:
      msg = (
        f"{fn_name}: {len(empty)} Atom14 slot(s) are flagged in atom_mask but have no "
        f"atom for their residue type (first (residue, slot): {empty[0]}); "
        "refusing to name them"
      )
      raise ValueError(msg)
  # The residue name is the as-read one for an aliased residue (MSE), so it is
  # written as MSE with SE, not as its MET model type.
  written_res = [
    residue_names[r]
    if residue_names is not None and _layout_alias(layout, residue_names[r]) is not None
    else resnames[int(aatype[r])]
    for r in res_idx
  ]
  names = [
    _written_atom_name(layout, rn, ln) for rn, ln in zip(written_res, layout_names, strict=True)
  ]
  b_slots = _per_row_values(
    getattr(protein, "b_factors", None), atom_mask.shape, "b_factors", fn_name
  )
  occ_slots = _per_row_values(
    getattr(protein, "occupancies", None), atom_mask.shape, "occupancies", fn_name
  )

  return (
    coords[res_idx, slot_idx],
    names,
    written_res,
    residue_index[res_idx],
    [res_chain[r] for r in res_idx],
    ["SE" if n == "SE" else _ATOM37_ELEMENT[n] for n in names],
    None if b_slots is None else b_slots[res_idx, slot_idx],
    None if occ_slots is None else occ_slots[res_idx, slot_idx],
  )


def _per_row_values(values, shape: tuple, what: str, fn_name: str) -> np.ndarray | None:
  """A per-row B-factor or occupancy array, or None when the Protein carries none.

  `shape` is the layout the rows are read from: (N_atoms,) for a per-atom
  Protein, the atom_mask shape (N_res, slots) for Atom37/Atom14. A mismatch
  raises: a per-atom field is never padded or truncated (debt #2367).
  """
  if values is None:
    return None
  arr = np.asarray(values, dtype=np.float64)
  if arr.shape != tuple(shape):
    msg = (
      f"{fn_name}: protein.{what} has shape {arr.shape}, expected {tuple(shape)}; "
      "refusing to guess an alignment"
    )
    raise ValueError(msg)
  return arr


def _pdb_occupancy_and_b_factor(occupancy, b_factor, i: int) -> tuple[str, str]:
  """The PDB occupancy (columns 55-60) and B-factor (61-66) fields of atom `i`.

  A value the Protein carries is written as its source value in the 6-character
  field. A value it does not carry is written blank, never as a 1.00 / 0.00
  that no one measured (debt #2367). proxide's PDB reader takes blank columns
  as the documented defaults (occupancy 1.0, B-factor 0.0).
  """

  def field(value, what: str) -> str:
    if value is None:
      return " " * 6
    if not np.isfinite(value):
      msg = f"write_pdb: {what} of atom {i} is not finite ({value}); refusing to write it"
      raise ValueError(msg)
    text = f"{float(value):6.2f}"
    if len(text) > 6:
      msg = (
        f"write_pdb: {what} {value} (atom {i}) does not fit the fixed-width PDB column; "
        "use write_mmcif"
      )
      raise ValueError(msg)
    return text

  return field(occupancy, "occupancy"), field(b_factor, "B-factor")


def _cif_occupancy_or_b_factor(value, what: str, i: int) -> str:
  """One mmCIF occupancy or B_iso_or_equiv value; '?' (CIF's unknown) when absent."""
  if value is None:
    return "?"
  if not np.isfinite(value):
    msg = f"write_mmcif: {what} of atom {i} is not finite ({value}); refusing to write it"
    raise ValueError(msg)
  return f"{float(value):.2f}"


def write_pdb(protein: Protein, path: str | Path) -> Path:
  """Write a Protein structure to a PDB file.

  MSE is restored as HETATM MSE/SE only when the Protein carries residue_names
  (set by the parser); proteins without them are written with model residue
  names (MET/SD).

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

  coords, atom_names, res_names, res_seqs, chain_letters, elements, b_factors, occupancies = (
    _atom_rows(protein, "write_pdb")
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
    record = "HETATM" if res_name in _HETATM_RESIDUES else "ATOM"
    res_seq = int(res_seqs[i])
    chain_id = chain_letters[i]
    if len(atom_name) > 4:
      raise _overflow(f"atom name {atom_name!r} (atom {i})")
    if len(res_name) > 3:
      raise _overflow(f"residue name {res_name!r} (atom {i})")
    if len(element) > 2:
      raise _overflow(f"element {element!r} (atom {i})")
    if len(chain_id) > 1:
      raise _overflow(f"chain id {chain_id!r} (atom {i})")
    chain_id = chain_id or " "  # unknown chain: the PDB's own blank column
    if not -999 <= res_seq <= 9999:
      raise _overflow(f"residue number {res_seq} (atom {i})")
    # The criterion is the formatted width itself (8 chars, %8.3f), not a
    # raw-float range: float32 -999.999 is -999.99902..., which still prints
    # as "-999.999" and fits.
    xyz = [f"{float(c):>8.3f}" for c in coords[i]]
    if any(len(c) > 8 for c in xyz):
      raise _overflow(f"coordinate ({', '.join(c.strip() for c in xyz)}) (atom {i})")

    occ_field, b_field = _pdb_occupancy_and_b_factor(
      None if occupancies is None else occupancies[i],
      None if b_factors is None else b_factors[i],
      i,
    )
    # PDB convention: a name whose element is one letter starts in column 14
    # (" CA "), so readers that infer elements from columns 13-14 don't read
    # CA as calcium or NE as neon. Four-letter names and two-letter or
    # unknown elements start in column 13.
    name_field = (
      f" {atom_name:<3}" if len(element) == 1 and len(atom_name) < 4 else f"{atom_name:<4}"
    )
    lines.append(
      f"{record:<6}{i + 1:>5} {name_field} {res_name:>3} {chain_id}{res_seq:>4}    "
      f"{''.join(xyz)}"
      f"{occ_field}{b_field}          {element.upper():>2}\n"
    )

  # Validate everything before touching the file, so a rejected structure
  # never leaves a truncated PDB behind.
  with open(path, "w") as f:
    f.writelines(lines)
    f.write("END\n")

  return path


def write_mmcif(protein: Protein, path: str | Path) -> Path:
  """Write a Protein structure to an mmCIF file.

  MSE is restored as HETATM MSE/SE only when the Protein carries residue_names
  (set by the parser); proteins without them are written with model residue
  names (MET/SD).

  Args:
      protein: The Protein instance to write.
      path: Target file path.

  Returns:
      Path to the written file.

  Raises:
      ValueError: If a field would have to be invented (see `_atom_rows`),
          a coordinate is non-finite, chain ids are unknown (mmCIF requires
          label_asym_id), or a name/chain value is empty or
          contains whitespace (see `_cif_token`).
  """
  path = Path(path)
  _reject_if_batched(protein, "write_mmcif")

  coords, atom_names, res_names, res_seqs, chain_letters, elements, b_factors, occupancies = (
    _atom_rows(protein, "write_mmcif")
  )
  _reject_nonfinite(coords, "write_mmcif")
  if not all(chain_letters):
    # label_asym_id is mandatory in mmCIF, and proxide's own reader rejects
    # "?" there -- so an unknown chain cannot be written honestly. (PDB has a
    # legitimate blank chain column; use write_pdb.)
    msg = "write_mmcif: chain ids are unknown (no chain_ids); mmCIF requires them -- use write_pdb"
    raise ValueError(msg)

  rows = []
  for i in range(len(coords)):
    # Unknown element -> "?", CIF's own unknown marker (left unquoted on purpose).
    element = _cif_token(elements[i].upper(), "element", i) if elements[i] else "?"
    record = "HETATM" if res_names[i] in _HETATM_RESIDUES else "ATOM"

    x, y, z = coords[i]
    occ_text = _cif_occupancy_or_b_factor(
      None if occupancies is None else occupancies[i], "occupancy", i
    )
    b_text = _cif_occupancy_or_b_factor(None if b_factors is None else b_factors[i], "B-factor", i)
    rows.append(
      f"{record} {i + 1} {element} {_cif_token(atom_names[i], 'atom name', i)} "
      f"{_cif_token(res_names[i], 'residue name', i)} "
      f"{_cif_token(chain_letters[i], 'chain id', i)} "
      f"{int(res_seqs[i])} {x:.3f} {y:.3f} {z:.3f} {occ_text} {b_text}\n"
    )

  # Validate every row before touching the file.
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

    f.writelines(rows)
  return path


def _cif_token(value: str, what: str, i: int) -> str:
  """One whitespace-free CIF loop value, quoted when its first char demands it.

  An empty or whitespace-containing value would shift every later column of
  the whitespace-tokenised `_atom_site` loop (the mmCIF analogue of debt
  #1927), so it raises instead.
  """
  if not value or any(ch.isspace() for ch in value):
    raise ValueError(f"write_mmcif: {what} {value!r} (atom {i}) is empty or contains whitespace")
  if value[0] in "'\"_#$;[]" or value in ("?", "."):
    if "'" in value:
      raise ValueError(f"write_mmcif: {what} {value!r} (atom {i}) cannot be CIF-quoted")
    return f"'{value}'"
  return value


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
  # Source B-factor and occupancy (debt #2367), present only when the Protein has them.
  if protein.b_factors is not None:
    data["b_factors"] = np.asarray(protein.b_factors)
  if protein.occupancies is not None:
    data["occupancies"] = np.asarray(protein.occupancies)

  if protein.residue_names is not None:
    data["residue_names"] = np.asarray([str(r) for r in protein.residue_names], dtype=str)

  # Filter out None values
  data = {k: v for k, v in data.items() if v is not None}

  np.savez_compressed(path, **data)  # ty: ignore[invalid-argument-type]
  return path
