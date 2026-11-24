"""Writing utilities for protein structures."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import json
import logging
import re
import urllib.parse
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from proxide import _proxider
from proxide.io import write_codes as _codes

if TYPE_CHECKING:
  from proxide.core.containers import Protein

logger = logging.getLogger("proxide.io.writing")


@dataclass(frozen=True)
class WriteFinding:
  """One thing the writer noticed about a write (spec 261123 D4). Experimental."""

  code: str
  severity: Literal["info", "warning"]
  count: int
  message: str


@dataclass(frozen=True)
class WriteReport:
  """What one `write_pdb`/`write_mmcif` call wrote, and how (spec 261123 D4). Experimental.

  ``sha256`` is of the exact bytes written. ``options`` are the effective content options,
  sorted; ``provenance_path`` is set only when a sidecar was written.
  """

  path: Path
  format: Literal["pdb", "mmcif"]
  sha256: str
  n_atoms: int
  n_hetatm: int
  n_hetatm_from_hetero: int
  n_insertion_codes: int
  options: tuple[tuple[str, str], ...]
  source: str | None
  proxide_version: str
  findings: tuple[WriteFinding, ...]
  provenance_path: Path | None


def _finding(code: str, count: int, message: str) -> WriteFinding:
  """A finding whose severity comes from the literal code table, never from the caller."""
  return WriteFinding(code=code, severity=_codes.SEVERITY_OF[code], count=count, message=message)


def _emit_warnings(findings: list[WriteFinding] | tuple[WriteFinding, ...]) -> None:
  """Warn once per warning finding. Info findings never warn.

  stacklevel=4 is fixed: this frame (1), `_write_pdb`/`_write_mmcif` (2), the public
  `write_pdb`/`write_pdb_reported` (or mmCIF pair) (3), so the warning is attributed to the
  caller of the public function. Do not add a frame between the public function and here.
  """
  for finding in findings:
    if finding.severity == "warning":
      warnings.warn(finding.message, UserWarning, stacklevel=4)


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
  codes = getattr(protein, "insertion_codes", None)
  if codes is not None and len(codes) > 0 and isinstance(codes[0], (list, tuple, np.ndarray)):
    msg = (
      f"{fn_name}() writes a single structure, but `protein.insertion_codes` is a list of "
      f"lists -- this looks like a batched Protein. Index into the batch first."
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
  list[WriteFinding],
]:
  """Resolve the per-atom rows `write_pdb`/`write_mmcif` emit, without inventing any.

  Returns (coords, atom_names, res_names, res_seqs, chain_letters, elements,
  b_factors, occupancies, findings), the first six of one length. b_factors and occupancies
  are the Protein's own values for those rows, or None when it carries none. ``findings`` are
  the findings about the rows (spec 261123 D4); this function warns nothing itself.
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
      [],
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
  alias_findings: list[WriteFinding] = []
  # A residue whose as-read name has a layout alias (MSE) but whose type is not that alias's
  # layout residue was parsed with modified_residue_typing="unknown": the format placed no atom
  # for the alias target (SE), so the record is written without it. Say so rather than emit a
  # silently incomplete MSE.
  if residue_names is not None:
    short = sorted(
      {
        residue_names[int(r)]
        for r in set(res_idx.tolist())
        if (alias := _layout_alias(layout, residue_names[int(r)])) is not None
        and resnames[int(aatype[int(r)])] != alias[0]
      }
    )
    if short:
      msg = (
        f"{fn_name}: residue(s) {short} were typed as something other than their layout residue "
        "(modified_residue_typing='unknown'), so their alias atoms (for MSE, SE) were not placed "
        "and are missing from the written records"
      )
      alias_findings.append(_finding(_codes.ALIAS_ATOMS_MISSING, len(short), msg))
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
    alias_findings,
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


_RECORD_TYPES = ("source", "residue_table")
_INSERTION_CODE_OPTIONS = ("write", "blank")
_PROVENANCE_OPTIONS = ("none", "sidecar")


def _check_writer_options(
  fn_name: str, record_type: str, insertion_codes: str, provenance: str = "none"
) -> None:
  """Reject an unknown option value before anything is read or written (spec D3, D5)."""
  if record_type not in _RECORD_TYPES:
    msg = (
      f"{fn_name}: record_type={record_type!r} is not accepted; "
      f"accepted values are {', '.join(map(repr, _RECORD_TYPES))}"
    )
    raise ValueError(msg)
  if insertion_codes not in _INSERTION_CODE_OPTIONS:
    msg = (
      f"{fn_name}: insertion_codes={insertion_codes!r} is not accepted; "
      f"accepted values are {', '.join(map(repr, _INSERTION_CODE_OPTIONS))}"
    )
    raise ValueError(msg)
  if provenance not in _PROVENANCE_OPTIONS:
    msg = (
      f"{fn_name}: provenance={provenance!r} is not accepted; "
      f"accepted values are {', '.join(map(repr, _PROVENANCE_OPTIONS))}"
    )
    raise ValueError(msg)


def _row_annotations(
  protein: Protein, fn_name: str, n_rows: int, insertion_codes: str
) -> tuple[np.ndarray | None, list[str], list[str]]:
  """The per-row hetero flags and the insertion codes of each row, before and after the option.

  Returns (hetero_rows, codes_rows, source_codes_rows). ``hetero_rows`` is a bool array per atom
  for the Full layout and None otherwise. ``codes_rows`` has one entry per row: the Protein's code
  for that row (Atom37/14 codes are per residue and broadcast to the residue's atom rows), or " "
  (blank) when the Protein has none or ``insertion_codes="blank"``. ``source_codes_rows`` is the
  same without the option applied, so a finding can count the codes the option blanked. The layout
  is decided by the same dispatch as `_atom_rows` (``atom_names`` present means Full).
  """
  from proxide.core.containers import validate_hetero, validate_insertion_codes

  is_full = getattr(protein, "atom_names", None) is not None
  hetero = getattr(protein, "hetero", None)
  if is_full:
    hetero_rows = None if hetero is None else validate_hetero(hetero, n_rows)
    row_entity = None  # rows are atoms; codes are per atom
    n_entities = n_rows
  else:
    if hetero is not None:
      msg = (
        f"{fn_name}: protein.hetero is set on an Atom37/Atom14 Protein; the per-atom hetero "
        "flag exists only for the Full layout, and these layouts hold protein residues only"
      )
      raise ValueError(msg)
    hetero_rows = None
    row_entity = np.nonzero(np.asarray(protein.atom_mask).astype(bool))[0]
    n_entities = int(np.asarray(protein.coordinates).shape[0])

  codes = getattr(protein, "insertion_codes", None)
  if codes is None:
    return hetero_rows, [" "] * n_rows, [" "] * n_rows
  entity_codes = validate_insertion_codes(codes, n_entities, f"{fn_name}: insertion_codes")
  if row_entity is None:
    source_rows = list(entity_codes)
  else:
    source_rows = [entity_codes[int(r)] for r in row_entity]
  if insertion_codes == "blank":
    return hetero_rows, [" "] * n_rows, source_rows
  return hetero_rows, source_rows, source_rows


def _is_hetatm(res_name: str, hetero_flag: bool, record_type: str) -> bool:
  """The record rule of spec D3 for one row: a source hetero flag wins, never downgrades."""
  if record_type == "source" and hetero_flag:
    return True
  return res_name in _HETATM_RESIDUES


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


def _record_plan(
  res_names: list[str], hetero_rows: np.ndarray | None, record_type: str
) -> tuple[list[str], int, int]:
  """The record type of each row, the HETATM count, and the rows HETATM only because of hetero."""
  records: list[str] = []
  n_hetatm = 0
  n_from_hetero = 0
  for i, res_name in enumerate(res_names):
    hetero_flag = bool(hetero_rows[i]) if hetero_rows is not None else False
    is_hetatm = _is_hetatm(res_name, hetero_flag, record_type)
    records.append("HETATM" if is_hetatm else "ATOM")
    if is_hetatm:
      n_hetatm += 1
      if record_type == "source" and hetero_flag and res_name not in _HETATM_RESIDUES:
        n_from_hetero += 1
  return records, n_hetatm, n_from_hetero


def _annotation_findings(
  protein: Protein,
  res_names: list[str],
  hetero_rows: np.ndarray | None,
  source_codes: list[str],
  record_type: str,
  insertion_codes: str,
) -> list[WriteFinding]:
  """Findings about hetero flags and insertion codes (spec 261123 D4). Warns nothing itself."""
  findings: list[WriteFinding] = []
  n_rows = len(res_names)
  is_full = getattr(protein, "atom_names", None) is not None
  if is_full and hetero_rows is None:
    findings.append(
      _finding(
        _codes.HETERO_UNKNOWN,
        n_rows,
        "Full Protein has no per-atom hetero flag (hetero is None): HETATM follows the residue "
        "table only",
      )
    )
  if record_type == "residue_table" and hetero_rows is not None:
    ignored = sum(
      1 for i, flag in enumerate(hetero_rows) if flag and res_names[i] not in _HETATM_RESIDUES
    )
    if ignored:
      findings.append(
        _finding(
          _codes.HETERO_IGNORED,
          ignored,
          f"record_type='residue_table' ignores hetero=True on {ignored} row(s) whose residue is "
          "not in the residue table; they are written ATOM",
        )
      )
  if getattr(protein, "insertion_codes", None) is None:
    findings.append(
      _finding(
        _codes.ICODE_UNKNOWN,
        n_rows,
        "Protein.insertion_codes is None: no row carries an insertion code",
      )
    )
  if insertion_codes == "blank":
    blanked = sum(1 for code in source_codes if code != " ")
    if blanked:
      findings.append(
        _finding(
          _codes.ICODE_BLANKED,
          blanked,
          f"insertion_codes='blank' writes the insertion-code column blank on {blanked} row(s) "
          "that carry a non-blank insertion code",
        )
      )
  return findings


def _proxide_version() -> str:
  try:
    return importlib.metadata.version("proxide")
  except importlib.metadata.PackageNotFoundError:
    return "unknown"


def _finish(
  path: Path,
  data: bytes,
  *,
  fmt: Literal["pdb", "mmcif"],
  protein: Protein,
  n_atoms: int,
  n_hetatm: int,
  n_hetatm_from_hetero: int,
  n_insertion_codes: int,
  record_type: str,
  insertion_codes: str,
  provenance: str,
  findings: list[WriteFinding],
) -> WriteReport:
  """Write the already-built bytes, then build the report, log it, and write the sidecar."""
  with open(path, "wb") as f:
    f.write(data)
  report = WriteReport(
    path=path,
    format=fmt,
    sha256=hashlib.sha256(data).hexdigest(),
    n_atoms=n_atoms,
    n_hetatm=n_hetatm,
    n_hetatm_from_hetero=n_hetatm_from_hetero,
    n_insertion_codes=n_insertion_codes,
    options=tuple(sorted({"record_type": record_type, "insertion_codes": insertion_codes}.items())),
    source=protein.source,
    proxide_version=_proxide_version(),
    findings=tuple(findings),
    provenance_path=None,
  )
  logger.info(
    "wrote %s %s: atoms=%d hetatm=%d hetatm_from_hetero=%d insertion_codes=%d options=%s codes=%s",
    fmt,
    path,
    n_atoms,
    n_hetatm,
    n_hetatm_from_hetero,
    n_insertion_codes,
    report.options,
    [f.code for f in findings],
  )
  if provenance == "sidecar":
    report = dataclasses.replace(report, provenance_path=_write_sidecar(report))
  return report


def _write_pdb(
  protein: Protein,
  path: str | Path,
  *,
  record_type: str,
  insertion_codes: str,
  provenance: str,
) -> WriteReport:
  """The one PDB emitter. `write_pdb` and `write_pdb_reported` are one-line wrappers over it."""
  path = Path(path)
  _reject_if_batched(protein, "write_pdb")
  _check_writer_options("write_pdb", record_type, insertion_codes, provenance)

  (
    coords,
    atom_names,
    res_names,
    res_seqs,
    chain_letters,
    elements,
    b_factors,
    occupancies,
    findings,
  ) = _atom_rows(protein, "write_pdb")
  _reject_nonfinite(coords, "write_pdb")
  hetero_rows, codes_rows, source_codes = _row_annotations(
    protein, "write_pdb", len(coords), insertion_codes
  )
  records, n_hetatm, n_from_hetero = _record_plan(res_names, hetero_rows, record_type)
  findings = [
    *findings,
    *_annotation_findings(
      protein, res_names, hetero_rows, source_codes, record_type, insertion_codes
    ),
  ]

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
    record = records[i]
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
      f"{record:<6}{i + 1:>5} {name_field} {res_name:>3} {chain_id}{res_seq:>4}{codes_rows[i]}   "
      f"{''.join(xyz)}"
      f"{occ_field}{b_field}          {element.upper():>2}\n"
    )
  data = ("".join(lines) + "END\n").encode("utf-8")

  # Everything is validated and built before the file is opened, so a rejected structure
  # never leaves a truncated PDB behind.
  _emit_warnings(findings)
  return _finish(
    path,
    data,
    fmt="pdb",
    protein=protein,
    n_atoms=len(coords),
    n_hetatm=n_hetatm,
    n_hetatm_from_hetero=n_from_hetero,
    n_insertion_codes=sum(1 for code in codes_rows if code != " "),
    record_type=record_type,
    insertion_codes=insertion_codes,
    provenance=provenance,
    findings=findings,
  )


def write_pdb(
  protein: Protein,
  path: str | Path,
  *,
  record_type: str = "source",
  insertion_codes: str = "write",
  provenance: str = "none",
) -> Path:
  """Write a Protein structure to a PDB file.

  MSE is restored as HETATM MSE/SE only when the Protein carries residue_names
  (set by the parser); proteins without them are written with model residue
  names (MET/SD).

  Args:
      protein: The Protein instance to write.
      path: Target file path.
      record_type: ``"source"`` (default) takes a Full Protein's per-atom ``hetero``
          flag as HETATM where it is True; ``"residue_table"`` uses only the residue
          table (MSE). Atom37/14 output is the same under both.
      insertion_codes: ``"write"`` (default) writes the Protein's insertion codes in
          column 27; ``"blank"`` leaves column 27 blank.
      provenance: ``"none"`` (default) writes only the output. ``"sidecar"`` also writes
          ``<path>.prov.toml`` after it (experimental; see `write_pdb_reported`).

  Returns:
      Path to the written file.

  Raises:
      ValueError: If a field would have to be invented (see `_atom_rows`), an option
          value is not accepted, ``hetero`` is set on an Atom37/Atom14 Protein, an
          insertion code is not one printable ASCII character, or a value does not fit its
          fixed-width PDB column (more than
          99999 atoms, a residue number outside -999..9999, an over-long name,
          or a coordinate outside -999.999..9999.999) -- the old writer
          silently shifted every later column (debt #1927). Use
          `write_mmcif` for such structures.
  """
  return _write_pdb(
    protein,
    path,
    record_type=record_type,
    insertion_codes=insertion_codes,
    provenance=provenance,
  ).path


def write_pdb_reported(
  protein: Protein,
  path: str | Path,
  *,
  record_type: str = "source",
  insertion_codes: str = "write",
  provenance: str = "none",
) -> WriteReport:
  """Experimental: `write_pdb`, returning a `WriteReport` (path, sha256, counts, findings).

  Same arguments, validation and bytes as `write_pdb`. Warnings are attributed to the caller
  of this function, as for `write_pdb`. The report and the sidecar format are experimental.
  """
  return _write_pdb(
    protein,
    path,
    record_type=record_type,
    insertion_codes=insertion_codes,
    provenance=provenance,
  )


def _write_mmcif(
  protein: Protein,
  path: str | Path,
  *,
  record_type: str,
  insertion_codes: str,
  provenance: str,
) -> WriteReport:
  """The one mmCIF emitter. `write_mmcif` and `write_mmcif_reported` wrap it."""
  path = Path(path)
  _reject_if_batched(protein, "write_mmcif")
  _check_writer_options("write_mmcif", record_type, insertion_codes, provenance)

  (
    coords,
    atom_names,
    res_names,
    res_seqs,
    chain_letters,
    elements,
    b_factors,
    occupancies,
    findings,
  ) = _atom_rows(protein, "write_mmcif")
  _reject_nonfinite(coords, "write_mmcif")
  hetero_rows, codes_rows, source_codes = _row_annotations(
    protein, "write_mmcif", len(coords), insertion_codes
  )
  records, n_hetatm, n_from_hetero = _record_plan(res_names, hetero_rows, record_type)
  findings = [
    *findings,
    *_annotation_findings(
      protein, res_names, hetero_rows, source_codes, record_type, insertion_codes
    ),
  ]
  # The column is emitted only when a written code is non-blank (owner decision 2026-11-23).
  write_codes = any(code != " " for code in codes_rows)
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
    code_field = f" {_cif_insertion_code(codes_rows[i], i)}" if write_codes else ""

    x, y, z = coords[i]
    occ_text = _cif_occupancy_or_b_factor(
      None if occupancies is None else occupancies[i], "occupancy", i
    )
    b_text = _cif_occupancy_or_b_factor(None if b_factors is None else b_factors[i], "B-factor", i)
    rows.append(
      f"{records[i]} {i + 1} {element} {_cif_token(atom_names[i], 'atom name', i)} "
      f"{_cif_token(res_names[i], 'residue name', i)} "
      f"{_cif_token(chain_letters[i], 'chain id', i)} "
      f"{int(res_seqs[i])}{code_field} {x:.3f} {y:.3f} {z:.3f} {occ_text} {b_text}\n"
    )

  head = [
    f"data_{path.stem}\n",
    "#\n",
    "loop_\n",
    "_atom_site.group_PDB\n",
    "_atom_site.id\n",
    "_atom_site.type_symbol\n",
    "_atom_site.label_atom_id\n",
    "_atom_site.label_comp_id\n",
    "_atom_site.label_asym_id\n",
    "_atom_site.label_seq_id\n",
  ]
  if write_codes:
    head.append("_atom_site.pdbx_PDB_ins_code\n")
  head += [
    "_atom_site.Cartn_x\n",
    "_atom_site.Cartn_y\n",
    "_atom_site.Cartn_z\n",
    "_atom_site.occupancy\n",
    "_atom_site.B_iso_or_equiv\n",
  ]
  data = ("".join(head) + "".join(rows)).encode("utf-8")

  # Every row is validated before the file is opened.
  _emit_warnings(findings)
  return _finish(
    path,
    data,
    fmt="mmcif",
    protein=protein,
    n_atoms=len(coords),
    n_hetatm=n_hetatm,
    n_hetatm_from_hetero=n_from_hetero,
    n_insertion_codes=sum(1 for code in codes_rows if code != " "),
    record_type=record_type,
    insertion_codes=insertion_codes,
    provenance=provenance,
    findings=findings,
  )


def write_mmcif(
  protein: Protein,
  path: str | Path,
  *,
  record_type: str = "source",
  insertion_codes: str = "write",
  provenance: str = "none",
) -> Path:
  """Write a Protein structure to an mmCIF file.

  MSE is restored as HETATM MSE/SE only when the Protein carries residue_names
  (set by the parser); proteins without them are written with model residue
  names (MET/SD).

  Args:
      protein: The Protein instance to write.
      path: Target file path.
      record_type: As in `write_pdb`; sets ``group_PDB``.
      insertion_codes: ``"write"`` (default) emits ``_atom_site.pdbx_PDB_ins_code`` when at
          least one written code is non-blank (blank rows get ``?``); ``"blank"`` never emits
          it. A Protein without insertion codes is written exactly as before.
      provenance: As in `write_pdb`.

  Returns:
      Path to the written file.

  Raises:
      ValueError: If a field would have to be invented (see `_atom_rows`), an option
          value is not accepted, ``hetero`` is set on an Atom37/Atom14 Protein, an
          insertion code is invalid, a coordinate is non-finite, chain ids are unknown
          (mmCIF requires label_asym_id), or a name/chain value is empty or
          contains whitespace (see `_cif_token`).
  """
  return _write_mmcif(
    protein,
    path,
    record_type=record_type,
    insertion_codes=insertion_codes,
    provenance=provenance,
  ).path


def write_mmcif_reported(
  protein: Protein,
  path: str | Path,
  *,
  record_type: str = "source",
  insertion_codes: str = "write",
  provenance: str = "none",
) -> WriteReport:
  """Experimental: `write_mmcif`, returning a `WriteReport`. See `write_pdb_reported`."""
  return _write_mmcif(
    protein,
    path,
    record_type=record_type,
    insertion_codes=insertion_codes,
    provenance=provenance,
  )


# --- Provenance sidecar (spec 261123 D5, experimental) -----------------------------------------

PROVENANCE_SCHEMA = "proxide-write-prov/0"
_TOML_BARE_KEY = re.compile(r"[A-Za-z0-9_-]+")


def _toml_str(text: str) -> str:
  """A TOML basic string: quote and backslash escaped, newline as \\n, other controls as \\uXXXX."""
  out = ['"']
  for ch in text:
    code = ord(ch)
    if ch == '"':
      out.append('\\"')
    elif ch == "\\":
      out.append("\\\\")
    elif ch == "\n":
      out.append("\\n")
    elif code < 0x20 or code == 0x7F:
      out.append(f"\\u{code:04x}")
    else:
      out.append(ch)
  out.append('"')
  return "".join(out)


def _toml_value(value: object) -> str:
  """The sidecar's value shapes only: str, int, a list of those, and a flat inline table."""
  if isinstance(value, bool):
    raise TypeError("the sidecar has no boolean values")
  if isinstance(value, int):
    return str(value)
  if isinstance(value, str):
    return _toml_str(value)
  if isinstance(value, (list, tuple)):
    return "[" + ", ".join(_toml_value(v) for v in value) + "]"
  if isinstance(value, dict):
    if not value:
      return "{}"
    return (
      "{ " + ", ".join(f"{_toml_key(str(k))} = {_toml_value(v)}" for k, v in value.items()) + " }"
    )
  raise TypeError(f"the sidecar cannot encode {type(value).__name__}")


def _toml_key(key: str) -> str:
  if not _TOML_BARE_KEY.fullmatch(key):
    raise ValueError(f"sidecar key {key!r} is not a bare TOML key")
  return key


def _proxide_install() -> tuple[str, str, str | None]:
  """How this proxide was installed: (install kind, source commit, editable path or None).

  From PEP 610 `direct_url.json`. A distribution with no record is a wheel install. The commit is
  the record's vcs_info commit, else "unknown" (editable and wheel installs carry none).
  """
  try:
    dist = importlib.metadata.distribution("proxide")
    raw = dist.read_text("direct_url.json")
  except (importlib.metadata.PackageNotFoundError, OSError):
    return "unknown", "unknown", None
  if raw is None:
    return "wheel", "unknown", None
  try:
    record = json.loads(raw)
  except json.JSONDecodeError:
    return "unknown", "unknown", None
  if not isinstance(record, dict):
    return "unknown", "unknown", None
  vcs = record.get("vcs_info")
  commit = vcs.get("commit_id") if isinstance(vcs, dict) else None
  commit = commit if isinstance(commit, str) and commit else "unknown"
  dir_info = record.get("dir_info")
  if isinstance(dir_info, dict) and dir_info.get("editable") is True:
    parsed = urllib.parse.urlparse(str(record.get("url", "")))
    editable_path = urllib.parse.unquote(parsed.path) if parsed.scheme == "file" else None
    return "editable", commit, editable_path
  if isinstance(vcs, dict):
    return "vcs", commit, None
  return "wheel", commit, None


def _provenance_toml(report: WriteReport) -> str:
  install, commit, editable_path = _proxide_install()
  provenance: list[tuple[str, object]] = [
    ("proxide_version", report.proxide_version),
    ("proxide_install", install),
    ("proxide_source_commit", commit),
  ]
  if install == "editable":
    provenance.append(("proxide_editable_path", editable_path or "unknown"))
  provenance += [
    ("output_path", str(report.path.resolve())),
    ("output_sha256", report.sha256),
    ("input_source", report.source or "unknown"),
    ("input_sha256", "unavailable"),
    ("options", dict(report.options)),
    (
      "counts",
      {
        "n_atoms": report.n_atoms,
        "n_hetatm": report.n_hetatm,
        "n_hetatm_from_hetero": report.n_hetatm_from_hetero,
        "n_insertion_codes": report.n_insertion_codes,
      },
    ),
    ("finding_codes", [f.code for f in report.findings]),
  ]
  lines = [f"schema = {_toml_str(PROVENANCE_SCHEMA)}", "", "[provenance]"]
  lines += [f"{_toml_key(key)} = {_toml_value(value)}" for key, value in provenance]
  return "\n".join(lines) + "\n"


def _write_sidecar(report: WriteReport) -> Path:
  """Write `<path>.prov.toml` after the output. An OSError propagates; the output stays."""
  prov_path = report.path.with_name(report.path.name + ".prov.toml")
  with open(prov_path, "wb") as f:
    f.write(_provenance_toml(report).encode("utf-8"))
  return prov_path


def _cif_insertion_code(code: str, i: int) -> str:
  """One mmCIF pdbx_PDB_ins_code value: "?" for a blank code, else the code as a CIF token."""
  if code == " ":
    return "?"
  return _cif_token(code, "insertion code", i)


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
      # CIF 1.1 allows a double-quoted token holding a single quote (e.g. the PDB icode "'").
      if '"' in value:
        raise ValueError(f"write_mmcif: {what} {value!r} (atom {i}) cannot be CIF-quoted")
      return f'"{value}"'
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
  # Insertion codes (spec 261123) and the per-atom HETATM flag, present only when set.
  if protein.insertion_codes is not None:
    data["insertion_codes"] = np.asarray([str(c) for c in protein.insertion_codes], dtype=str)
  if protein.hetero is not None:
    data["hetero"] = np.asarray(protein.hetero, dtype=np.bool_)

  # Filter out None values
  data = {k: v for k, v in data.items() if v is not None}

  np.savez_compressed(path, **data)  # ty: ignore[invalid-argument-type]
  return path
