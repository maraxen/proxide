"""Interface contact analysis, charge classification, Shrake-Rupley SASA, and docking screens.

Experimental: not covered by proxide's stability guarantee; may change in a minor release.
"""

from __future__ import annotations

import csv
import io
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

SOLVENT_RESIDUES: frozenset[str] = frozenset({"HOH", "WAT", "TIP3", "H2O"})

# Cation and anion atom identification
CHARGE_SIGN: dict[tuple[str, str], int] = {
  ("LYS", "NZ"): 1,
  ("ARG", "NE"): 1,
  ("ARG", "NH1"): 1,
  ("ARG", "NH2"): 1,
  ("ASP", "OD1"): -1,
  ("ASP", "OD2"): -1,
  ("GLU", "OE1"): -1,
  ("GLU", "OE2"): -1,
}

BACKBONE_ATOMS: frozenset[str] = frozenset({"N", "CA", "C", "O", "OXT"})
RESIDUE_CHARGE: dict[str, int] = {"LYS": 1, "ARG": 1, "ASP": -1, "GLU": -1}
HYDROPHOBIC_RESIDUES: frozenset[str] = frozenset(
  {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "TYR"}
)

THREE_TO_ONE: dict[str, str] = {
  "ALA": "A",
  "ARG": "R",
  "ASN": "N",
  "ASP": "D",
  "CYS": "C",
  "GLU": "E",
  "GLN": "Q",
  "GLY": "G",
  "HIS": "H",
  "ILE": "I",
  "LEU": "L",
  "LYS": "K",
  "MET": "M",
  "PHE": "F",
  "PRO": "P",
  "SER": "S",
  "THR": "T",
  "TRP": "W",
  "TYR": "Y",
  "VAL": "V",
}

# Standard Bondi (1964) atomic van der Waals radii in Angstroms
BONDI_RADII: dict[str, float] = {
  "H": 1.20,
  "C": 1.70,
  "N": 1.55,
  "O": 1.52,
  "S": 1.80,
  "P": 1.80,
  "F": 1.47,
  "CL": 1.75,
  "BR": 1.85,
  "I": 1.98,
}
DEFAULT_RADIUS: float = 2.00
PROBE_RADIUS: float = 1.40


@dataclass(frozen=True)
class InterfaceContactResult:
  """Interface contact metrics between two chains in a structure.

  Experimental: not covered by proxide's stability guarantee; may change in a minor release.
  """

  n_binder_interface: int
  n_target_interface: int
  interface_residues_a: list[Any]
  interface_residues_b: list[Any]
  ion_pairs: int
  like_close: int
  polar_pairs: int
  nonpolar_pairs: int
  binder_if_charge: int
  target_if_charge: int
  binder_if_frac_ek: float
  binder_if_frac_hydrophobic: float
  details: dict[str, Any]
  buried_binder_a2: int | None = None

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)


def _infer_element_symbol(atom_name: str) -> str:
  """Infer elemental symbol from atom name conforming to proxide masses canon."""
  cleaned = atom_name.strip().lstrip("0123456789")
  if not cleaned:
    return ""
  if len(cleaned) >= 2:
    prefix2 = cleaned[:2].upper()
    if prefix2 in ("CL", "BR", "NA", "MG", "ZN", "FE", "CU", "MN", "SE", "SI"):
      return prefix2.capitalize()
  ch = cleaned[0].upper()
  if ch in ("H", "C", "N", "O", "S", "P", "F", "I", "K"):
    return ch
  return ""


def _parse_pdb_text(text: str) -> list[dict[str, Any]]:
  """Parse standard PDB ATOM/HETATM lines into atom record dictionaries."""
  atoms: list[dict[str, Any]] = []
  for ln in text.splitlines():
    if ln.startswith("ENDMDL"):
      break
    if ln.startswith(("ATOM", "HETATM")) and len(ln) >= 54:
      res = ln[17:20].strip()
      if res in SOLVENT_RESIDUES:
        continue
      name = ln[12:16].strip()
      raw_elem = ln[76:78].strip().upper() if len(ln) >= 78 else ""
      if raw_elem == "CA" or (not raw_elem and name == "CA"):
        element = "C"
      elif raw_elem:
        element = raw_elem
      else:
        element = _infer_element_symbol(name)

      is_h = element == "H" or name.startswith("H")
      if is_h or (len(name) >= 2 and name[0].isdigit() and name[1] == "H"):
        continue
      chain = ln[21]
      ins_code = ln[26].strip() if len(ln) > 26 else ""
      try:
        num = int(ln[22:26])
        x = float(ln[30:38])
        y = float(ln[38:46])
        z = float(ln[46:54])
      except ValueError:
        continue
      atoms.append({
        "name": name,
        "res": res,
        "chain": chain,
        "num": num,
        "ins_code": ins_code,
        "res_id": (num, ins_code) if ins_code else num,
        "xyz": (x, y, z),
        "element": element,
      })
  return atoms


def _normalize_atoms(
  structure: str | Path | list[dict[str, Any]],
) -> list[dict[str, Any]]:
  """Normalize input representation (path, PDB string, or atom dicts) to atom records."""
  if isinstance(structure, Path):
    if not structure.exists():
      raise FileNotFoundError(f"Structure file not found: {structure}")
    return _parse_pdb_text(structure.read_text())
  elif isinstance(structure, str):
    if "\n" in structure or structure.startswith(
      ("ATOM", "HETATM", "HEADER", "REMARK", "MODEL")
    ):
      return _parse_pdb_text(structure)
    p = Path(structure)
    if p.exists() and p.is_file():
      return _parse_pdb_text(p.read_text())
    raise FileNotFoundError(f"Structure file or PDB text not found: {structure}")
  elif isinstance(structure, list):
    if not structure:
      return []
    filtered = []
    for i, atom in enumerate(structure):
      if not isinstance(atom, dict):
        raise TypeError(f"Expected dict at index {i}, got {type(atom).__name__}")
      for key in ("chain", "name", "res", "num", "xyz"):
        if key not in atom:
          raise KeyError(f"Atom record missing required key: '{key}'")
      res = atom["res"]
      if res in SOLVENT_RESIDUES:
        continue
      name = atom["name"]
      raw_elem = atom.get("element", "").strip().upper()
      if raw_elem == "CA" or (not raw_elem and name == "CA"):
        element = "C"
      elif raw_elem:
        element = raw_elem
      else:
        element = _infer_element_symbol(name)
      is_h = element == "H" or name.startswith("H")
      if is_h or (len(name) >= 2 and name[0].isdigit() and name[1] == "H"):
        continue
      atom_copy = dict(atom)
      atom_copy["element"] = element
      filtered.append(atom_copy)
    return filtered
  else:
    raise TypeError(
      f"Unsupported structure input type: {type(structure).__name__}"
    )


def compute_interface_contacts(
  structure: str | Path | list[dict[str, Any]],
  chain_a: str = "A",
  chain_b: str = "B",
  cutoff_heavy: float = 5.0,
  cutoff_ion_pair: float = 4.0,
  cutoff_like_charge: float = 4.5,
  cutoff_polar: float = 3.5,
  cutoff_nonpolar: float = 4.5,
  compute_sasa: bool = False,
  probe_radius: float = 1.40,
  n_points: int = 100,
) -> InterfaceContactResult:
  """Compute interface contacts, residue counts, and charge classifications between two chains.

  Experimental: not covered by proxide's stability guarantee; may change in a minor release.

  Args:
      structure: File path, PDB text string, or list of atom dicts.
      chain_a: Binder chain identifier (default 'A').
      chain_b: Target chain identifier (default 'B').
      cutoff_heavy: Heavy-atom distance cutoff for interface residues (default 5.0 Å).
      cutoff_ion_pair: Distance cutoff for cross-chain ion pairs (default 4.0 Å).
      cutoff_like_charge: Distance cutoff for cross-chain repulsive like charges (default 4.5 Å).
      cutoff_polar: Distance cutoff for polar N...O candidate hydrogen bonds (default 3.5 Å).
      cutoff_nonpolar: Distance cutoff for sidechain C...C nonpolar pairs (default 4.5 Å).
      compute_sasa: Whether to compute Shrake-Rupley buried binder surface area (default False).
      probe_radius: Probe radius for SASA calculation in Angstroms (default 1.40 Å).
      n_points: Number of Fibonacci sphere integration points for SASA (default 100).

  Returns:
      InterfaceContactResult dataclass containing contact counts, residue lists, and fractions.
  """
  if chain_a == chain_b:
    raise ValueError(
      f"Chains must be distinct (got chain_a='{chain_a}', chain_b='{chain_b}')"
    )

  atoms = _normalize_atoms(structure)
  if not atoms:
    raise ValueError("Structure contains no atoms")

  chains_present = {x["chain"] for x in atoms}
  if chain_a not in chains_present:
    raise KeyError(f"Chain '{chain_a}' not found in structure")
  if chain_b not in chains_present:
    raise KeyError(f"Chain '{chain_b}' not found in structure")

  atoms_a = [x for x in atoms if x["chain"] == chain_a]
  atoms_b = [x for x in atoms if x["chain"] == chain_b]

  coords_a = np.array([x["xyz"] for x in atoms_a], dtype=float).reshape(-1, 3)
  coords_b = np.array([x["xyz"] for x in atoms_b], dtype=float).reshape(-1, 3)

  # Cross-chain pairwise atom distance matrix
  dist_mat = np.linalg.norm(coords_a[:, None, :] - coords_b[None, :, :], axis=2)

  # Unique residue identifiers: (num, ins_code)
  res_map_a: dict[tuple[int, str], str] = {}
  for x in atoms_a:
    k = (x["num"], x.get("ins_code", ""))
    res_map_a[k] = x["res"]

  res_map_b: dict[tuple[int, str], str] = {}
  for x in atoms_b:
    k = (x["num"], x.get("ins_code", ""))
    res_map_b[k] = x["res"]

  # Interface residues (any heavy atom within cutoff_heavy)
  a_if_keys = sorted(
    {
      (atoms_a[i]["num"], atoms_a[i].get("ins_code", ""))
      for i in range(len(atoms_a))
      if dist_mat[i].min() <= cutoff_heavy
    }
  )
  b_if_keys = sorted(
    {
      (atoms_b[j]["num"], atoms_b[j].get("ins_code", ""))
      for j in range(len(atoms_b))
      if dist_mat[:, j].min() <= cutoff_heavy
    }
  )

  def _format_res_id(k: tuple[int, str]) -> int | tuple[int, str]:
    return (k[0], k[1]) if k[1] else k[0]

  a_if_residues = [_format_res_id(k) for k in a_if_keys]
  b_if_residues = [_format_res_id(k) for k in b_if_keys]

  # Charge classifications
  ion_pairs = 0
  like_close = 0
  polar_pairs = 0
  nonpolar_pairs = 0

  b_info = [
    (
      CHARGE_SIGN.get((b["res"], b["name"]), 0),
      (b.get("element") or _infer_element_symbol(b["name"])).upper(),
      (
        (b.get("element") or _infer_element_symbol(b["name"])).upper() == "C"
        and b["name"] not in BACKBONE_ATOMS
      ),
    )
    for b in atoms_b
  ]

  for i, a in enumerate(atoms_a):
    sx = CHARGE_SIGN.get((a["res"], a["name"]), 0)
    a_name = a["name"]
    a_elem = (a.get("element") or _infer_element_symbol(a_name)).upper()
    a_is_c = a_elem == "C" and a_name not in BACKBONE_ATOMS

    for j, (sy, b_elem, b_is_c) in enumerate(b_info):
      d = dist_mat[i, j]

      if sx != 0 and sy != 0:
        if sx != sy and d <= cutoff_ion_pair:
          ion_pairs += 1
        if sx == sy and d <= cutoff_like_charge:
          like_close += 1

      if d <= cutoff_polar and {a_elem, b_elem} == {"N", "O"}:
        polar_pairs += 1

      if d <= cutoff_nonpolar and a_is_c and b_is_c:
        nonpolar_pairs += 1

  # Interface charges
  binder_if_charge = sum(RESIDUE_CHARGE.get(res_map_a[k], 0) for k in a_if_keys)
  target_if_charge = sum(RESIDUE_CHARGE.get(res_map_b[k], 0) for k in b_if_keys)

  # Fractions
  n_a_if = len(a_if_keys)
  ek_count = sum(res_map_a[k] in ("GLU", "LYS") for k in a_if_keys)
  hyd_count = sum(res_map_a[k] in HYDROPHOBIC_RESIDUES for k in a_if_keys)

  binder_if_frac_ek = round(ek_count / max(1, n_a_if), 2)
  binder_if_frac_hydrophobic = round(hyd_count / max(1, n_a_if), 2)

  details: dict[str, Any] = {
    "binder_len": len(res_map_a),
    "binder_net_charge": sum(
      RESIDUE_CHARGE.get(res_map_a[k], 0) for k in res_map_a
    ),
    "target_len": len(res_map_b),
    "target_net_charge": sum(
      RESIDUE_CHARGE.get(res_map_b[k], 0) for k in res_map_b
    ),
    "target_if_residues": " ".join(
      f"{THREE_TO_ONE.get(res_map_b[k], 'X')}{k[0]}{k[1]}" for k in b_if_keys
    ),
    "interface_residues_a_types": {
      _format_res_id(k): res_map_a[k] for k in a_if_keys
    },
    "interface_residues_b_types": {
      _format_res_id(k): res_map_b[k] for k in b_if_keys
    },
  }

  buried_binder_a2: int | None = None
  if compute_sasa:
    delta_sasa_float, delta_sasa_int, sasa_details = compute_buried_sasa(
      structure,
      chain_a=chain_a,
      chain_b=chain_b,
      probe_radius=probe_radius,
      n_points=n_points,
    )
    buried_binder_a2 = delta_sasa_int
    details["buried_binder_a2_float"] = delta_sasa_float
    details["sasa_details"] = sasa_details

  return InterfaceContactResult(
    n_binder_interface=n_a_if,
    n_target_interface=len(b_if_keys),
    interface_residues_a=a_if_residues,
    interface_residues_b=b_if_residues,
    ion_pairs=ion_pairs,
    like_close=like_close,
    polar_pairs=polar_pairs,
    nonpolar_pairs=nonpolar_pairs,
    binder_if_charge=binder_if_charge,
    target_if_charge=target_if_charge,
    binder_if_frac_ek=binder_if_frac_ek,
    binder_if_frac_hydrophobic=binder_if_frac_hydrophobic,
    details=details,
    buried_binder_a2=buried_binder_a2,
  )


def compute_interface_contacts_batch(
  structures: list[str | Path | list[dict[str, Any]]],
  chain_a: str = "A",
  chain_b: str = "B",
  cutoff_heavy: float = 5.0,
  cutoff_ion_pair: float = 4.0,
  cutoff_like_charge: float = 4.5,
  cutoff_polar: float = 3.5,
  cutoff_nonpolar: float = 4.5,
  compute_sasa: bool = False,
  probe_radius: float = 1.40,
  n_points: int = 100,
) -> list[InterfaceContactResult]:
  """Batch interface contact computation over multiple structures.

  Experimental: not covered by proxide's stability guarantee; may change in a minor release.
  """
  return [
    compute_interface_contacts(
      s,
      chain_a=chain_a,
      chain_b=chain_b,
      cutoff_heavy=cutoff_heavy,
      cutoff_ion_pair=cutoff_ion_pair,
      cutoff_like_charge=cutoff_like_charge,
      cutoff_polar=cutoff_polar,
      cutoff_nonpolar=cutoff_nonpolar,
      compute_sasa=compute_sasa,
      probe_radius=probe_radius,
      n_points=n_points,
    )
    for s in structures
  ]


# ==============================================================================
# Slice 2: Jaccard, Alignment Overlap, and Epitope Docking
# ==============================================================================


def contact_jaccard(
  a: set[Any] | frozenset[Any],
  b: set[Any] | frozenset[Any],
) -> float:
  """Compute set Jaccard similarity between two contact sets.

  len(a & b) / len(a | b) if union is non-empty, else 0.0.

  Experimental: not covered by proxide's stability guarantee; may change in a minor release.
  """
  sa = set(a)
  sb = set(b)
  union = sa | sb
  if not union:
    return 0.0
  return len(sa & sb) / len(union)


def aligned_contact_overlap(
  contacts_a: set[Any] | frozenset[Any],
  contacts_b: set[Any] | frozenset[Any],
  aligned_pairs: Sequence[tuple[int, int]],
) -> tuple[float, int, int]:
  """Project contacts from two orthologs onto aligned positions in ortholog A numbering.

  Drops unaligned residues (e.g. human 212). Returns (jaccard, n_a_aligned, n_b_aligned).

  Experimental: not covered by proxide's stability guarantee; may change in a minor release.
  """

  def _to_int(k: Any) -> int:
    return k[0] if isinstance(k, tuple) else int(k)

  ca_ints = {_to_int(x) for x in contacts_a}
  cb_ints = {_to_int(x) for x in contacts_b}

  h_aligned = {a for a, b in aligned_pairs if a in ca_ints}
  m_aligned = {a for a, b in aligned_pairs if b in cb_ints}

  j = contact_jaccard(h_aligned, m_aligned)
  return j, len(h_aligned), len(m_aligned)


def load_aligned_pairs(
  csv_text_or_path: str | Path,
  col_a: str = "human_resnum",
  col_b: str = "mouse_resnum",
  in_crop_col: str | None = "in_crop",
  aligned_col: str | None = None,
) -> list[tuple[int, int]]:
  """Parse alignment numbering table into list of aligned residue pairs (col_a, col_b).

  Experimental: not covered by proxide's stability guarantee; may change in a minor release.
  """
  if isinstance(csv_text_or_path, str):
    if not csv_text_or_path.strip():
      return []
    if "\n" in csv_text_or_path or not Path(csv_text_or_path).is_file():
      f = io.StringIO(csv_text_or_path)
    else:
      f = Path(csv_text_or_path).open()
  elif isinstance(csv_text_or_path, Path):
    if not csv_text_or_path.exists():
      raise FileNotFoundError(f"File not found: {csv_text_or_path}")
    content = csv_text_or_path.read_text()
    if not content.strip():
      return []
    f = io.StringIO(content)
  else:
    raise TypeError(
      f"Unsupported type for csv_text_or_path: {type(csv_text_or_path).__name__}"
    )

  with f:
    reader = csv.DictReader(f)
    if reader.fieldnames is None:
      return []

    required = [col_a, col_b]
    if in_crop_col is not None:
      required.append(in_crop_col)
    if aligned_col is not None:
      required.append(aligned_col)

    for c in required:
      if c not in reader.fieldnames:
        raise KeyError(f"Required column '{c}' not found in CSV header")

    pairs: list[tuple[int, int]] = []
    for row in reader:
      val_a = (row.get(col_a) or "").strip()
      val_b = (row.get(col_b) or "").strip()
      if not val_a or not val_b:
        continue

      if in_crop_col is not None:
        in_crop_val = (str(row.get(in_crop_col) or "")).strip()
        if in_crop_val not in ("1", "True", "true"):
          continue

      if aligned_col is not None:
        aligned_val = (str(row.get(aligned_col) or "")).strip()
        if aligned_val not in ("1", "True", "true"):
          continue

      try:
        ra = int(val_a)
        rb = int(val_b)
      except ValueError as err:
        raise ValueError(
          f"Could not parse integer residue number in row: {row}"
        ) from err

      pairs.append((ra, rb))

    return pairs


def contacts_and_dock(
  target_coords: np.ndarray,
  target_resids: np.ndarray | Sequence[int],
  binder_coords: np.ndarray,
  *,
  target_offset: int = 0,
  cutoff: float = 5.0,
  epitope_residues: set[int] | frozenset[int] | None = None,
  min_epitope_contacts: int = 4,
  min_interface_residues: int = 10,
) -> dict[str, Any]:
  """Compute contacted target residues and binary docking verdict.

  Experimental: not covered by proxide's stability guarantee; may change in a minor release.
  """
  target_coords_arr = np.asarray(target_coords, dtype=float)
  binder_coords_arr = np.asarray(binder_coords, dtype=float)
  target_resids_list = list(target_resids)

  if target_coords_arr.shape[0] == 0:
    raise ValueError("target_coords cannot be empty")

  if len(target_resids_list) != target_coords_arr.shape[0]:
    n_c = target_coords_arr.shape[0]
    n_r = len(target_resids_list)
    raise ValueError(
      f"Length mismatch: target_coords has {n_c} atoms, but target_resids has {n_r}"
    )

  if binder_coords_arr.shape[0] == 0:
    return {
      "contacted": [],
      "n_contacted": 0,
      "epitope_contacts": 0,
      "docked": False,
    }

  tree = cKDTree(target_coords_arr)
  hits = tree.query_ball_point(binder_coords_arr, r=cutoff)
  target_indices = {idx for sublist in hits for idx in sublist}
  raw_contacted = {target_resids_list[idx] for idx in target_indices}
  contacted = sorted({int(r) + target_offset for r in raw_contacted})
  n_contacted = len(contacted)

  if epitope_residues is not None:
    epitope_contacts = len(set(contacted) & set(epitope_residues))
    docked = bool(
      epitope_contacts >= min_epitope_contacts
      and n_contacted >= min_interface_residues
    )
  else:
    epitope_contacts = 0
    docked = bool(n_contacted >= min_interface_residues)

  return {
    "contacted": contacted,
    "n_contacted": n_contacted,
    "epitope_contacts": epitope_contacts,
    "docked": docked,
  }


def contacts_and_dock_batch(
  target_coords: np.ndarray,
  target_resids: np.ndarray | Sequence[int],
  binder_coords_list: Sequence[np.ndarray],
  **kwargs: Any,
) -> list[dict[str, Any]]:
  """Batch contact and docking computation reusing target spatial index.

  Experimental: not covered by proxide's stability guarantee; may change in a minor release.
  """
  target_coords_arr = np.asarray(target_coords, dtype=float)
  target_resids_list = list(target_resids)

  if target_coords_arr.shape[0] == 0:
    raise ValueError("target_coords cannot be empty")

  if len(target_resids_list) != target_coords_arr.shape[0]:
    n_c = target_coords_arr.shape[0]
    n_r = len(target_resids_list)
    raise ValueError(
      f"Length mismatch: target_coords has {n_c} atoms, but target_resids has {n_r}"
    )

  target_offset: int = kwargs.get("target_offset", 0)
  cutoff: float = kwargs.get("cutoff", 5.0)
  epitope_residues = kwargs.get("epitope_residues", None)
  min_epitope_contacts: int = kwargs.get("min_epitope_contacts", 4)
  min_interface_residues: int = kwargs.get("min_interface_residues", 10)

  tree = cKDTree(target_coords_arr)
  results: list[dict[str, Any]] = []

  for binder_coords in binder_coords_list:
    b_arr = np.asarray(binder_coords, dtype=float)
    if b_arr.shape[0] == 0:
      results.append({
        "contacted": [],
        "n_contacted": 0,
        "epitope_contacts": 0,
        "docked": False,
      })
      continue

    hits = tree.query_ball_point(b_arr, r=cutoff)
    target_indices = {idx for sublist in hits for idx in sublist}
    raw_contacted = {target_resids_list[idx] for idx in target_indices}
    contacted = sorted({int(r) + target_offset for r in raw_contacted})
    n_contacted = len(contacted)

    if epitope_residues is not None:
      epitope_contacts = len(set(contacted) & set(epitope_residues))
      docked = bool(
        epitope_contacts >= min_epitope_contacts
        and n_contacted >= min_interface_residues
      )
    else:
      epitope_contacts = 0
      docked = bool(n_contacted >= min_interface_residues)

    results.append({
      "contacted": contacted,
      "n_contacted": n_contacted,
      "epitope_contacts": epitope_contacts,
      "docked": docked,
    })

  return results


# ==============================================================================
# Slice 2: Self-Association (SA) Homodimer Screen
# ==============================================================================


def interface_counts(
  atoms_or_coords_a: Any,
  atoms_or_coords_b: Any,
  cutoff: float = 5.0,
) -> tuple[int, int]:
  """Count unique interface residues on chain A and chain B within cutoff distance.

  Experimental: not covered by proxide's stability guarantee; may change in a minor release.
  """

  def _extract(obj: Any) -> tuple[np.ndarray, list[Any]]:
    if isinstance(obj, Mapping):
      if not obj:
        return np.zeros((0, 3), dtype=float), []
      if "xyz" in obj and "resid" in obj:
        return np.asarray(obj["xyz"], dtype=float), list(obj["resid"])
      if "coords" in obj and "resids" in obj:
        return np.asarray(obj["coords"], dtype=float), list(obj["resids"])
      if "coords" in obj and "resid" in obj:
        return np.asarray(obj["coords"], dtype=float), list(obj["resid"])
    if isinstance(obj, (list, tuple)):
      if not obj:
        return np.zeros((0, 3), dtype=float), []
      if isinstance(obj[0], Mapping):
        coords = np.array([x["xyz"] for x in obj], dtype=float)
        resids = [x.get("num") for x in obj]
        return coords, resids
    if isinstance(obj, np.ndarray):
      if obj.size == 0:
        return np.zeros((0, 3), dtype=float), []
    raise TypeError(f"Unsupported input type for interface_counts: {type(obj)}")

  ca, ra = _extract(atoms_or_coords_a)
  cb, rb = _extract(atoms_or_coords_b)

  if len(ca) == 0 or len(cb) == 0:
    return 0, 0

  d = np.linalg.norm(ca[:, None, :] - cb[None, :, :], axis=2)
  count_a = len({
    int(r) if isinstance(r, (int, np.integer)) else r
    for r, m in zip(ra, d.min(axis=1) <= cutoff, strict=True)
    if m
  })
  count_b = len({
    int(r) if isinstance(r, (int, np.integer)) else r
    for r, m in zip(rb, d.min(axis=0) <= cutoff, strict=True)
    if m
  })
  return count_a, count_b


def associates(
  n_a: int,
  n_b: int,
  ipsae: float | None,
  min_if_residues: int = 12,
  min_ipsae: float = 0.4,
) -> bool:
  """Run-level association rule for homodimers.

  Experimental: not covered by proxide's stability guarantee; may change in a minor release.
  """
  if ipsae is None:
    return False
  try:
    val = float(ipsae)
    if math.isnan(val):
      return False
  except (ValueError, TypeError):
    return False

  return bool(n_a >= min_if_residues and n_b >= min_if_residues and val >= min_ipsae)


def _selftest_sa() -> dict[str, bool]:
  """Run synthetic dimer and monomer coordinate checks and associates rules."""
  dimer_a = {
    "xyz": np.array([[float(i) * 3.8, 0.0, 0.0] for i in range(20)]),
    "resid": np.arange(1, 21),
  }
  dimer_b = {
    "xyz": np.array([[float(i) * 3.8, 1.0, 0.0] for i in range(20)]),
    "resid": np.arange(1, 21),
  }
  monomer_b = {
    "xyz": np.array([[float(i) * 3.8, 500.0, 0.0] for i in range(20)]),
    "resid": np.arange(1, 21),
  }

  na_dimer, nb_dimer = interface_counts(dimer_a, dimer_b)
  na_mono, nb_mono = interface_counts(dimer_a, monomer_b)

  dimer_pass = bool(
    na_dimer >= 12 and nb_dimer >= 12 and associates(na_dimer, nb_dimer, 0.65)
  )
  monomer_pass = bool(
    na_mono == 0 and nb_mono == 0 and not associates(na_mono, nb_mono, 0.65)
  )
  rule_pass = bool(
    not associates(20, 20, 0.20)
    and not associates(5, 20, 0.65)
    and not associates(20, 20, None)
    and not associates(20, 20, float("nan"))
  )
  all_pass = bool(dimer_pass and monomer_pass and rule_pass)
  return {
    "all_pass": all_pass,
    "dimer_pass": dimer_pass,
    "monomer_pass": monomer_pass,
    "rule_pass": rule_pass,
  }


def screen_self_association(
  runs: Sequence[Mapping[str, Any]],
  designs: Mapping[str, Mapping[str, Any]] | None = None,
  *,
  min_if_residues: int = 12,
  min_ipsae: float = 0.4,
  monomeric_max_rate: float = 0.20,
  self_associates_min_rate: float = 0.50,
  expected_runs_per_design: int | None = 2,
) -> dict[str, Any]:
  """Aggregate self-association run records and evaluate monomeric vs associating outcome.

  Experimental: not covered by proxide's stability guarantee; may change in a minor release.
  """
  problems: list[str] = []
  n_runs = len(runs)
  n_problems = 0

  n_pos_ctl_assoc = 0
  n_pos_ctl_valid = 0
  n_neg_ctl_assoc = 0
  n_neg_ctl_valid = 0
  n_design_assoc = 0
  n_design_valid = 0
  n_shuffle_assoc = 0
  n_shuffle_valid = 0

  for row in runs:
    did = row.get("design_id", "unknown")
    grp = row.get("group") or row.get("role")
    if grp is None and designs is not None and did in designs:
      grp = designs[did].get("group") or designs[did].get("role")
    if grp is None:
      grp = "design"

    # Validity check
    is_valid = True
    raw_valid = row.get("valid", 1)
    try:
      if int(raw_valid) == 0:
        is_valid = False
    except (ValueError, TypeError):
      is_valid = False

    if not is_valid:
      n_problems += 1
      prob_msg = str(row.get("problem", "invalid run"))[:200]
      problems.append(f"{did}: {prob_msg}")
      continue

    # Association determination
    is_assoc = False
    if "assoc" in row and row["assoc"] not in (None, ""):
      try:
        is_assoc = bool(int(row["assoc"]))
      except (ValueError, TypeError):
        is_assoc = False
    else:
      try:
        na = int(row.get("n_if_a", 0))
        nb = int(row.get("n_if_b", 0))
        ipsae_raw = row.get("ipsae")
        ipsae = float(ipsae_raw) if ipsae_raw not in (None, "") else None
        is_assoc = associates(
          na, nb, ipsae, min_if_residues=min_if_residues, min_ipsae=min_ipsae
        )
      except (ValueError, TypeError) as e:
        n_problems += 1
        problems.append(f"{did}: {e}")
        continue

    if grp == "pos_ctl":
      n_pos_ctl_valid += 1
      if is_assoc:
        n_pos_ctl_assoc += 1
    elif grp == "neg_ctl":
      n_neg_ctl_valid += 1
      if is_assoc:
        n_neg_ctl_assoc += 1
    elif grp == "design":
      n_design_valid += 1
      if is_assoc:
        n_design_assoc += 1
    elif grp == "shuffle":
      n_shuffle_valid += 1
      if is_assoc:
        n_shuffle_assoc += 1

  controls_ok = bool(n_pos_ctl_assoc >= 1 and n_neg_ctl_assoc == 0)

  design_rate = (
    round(n_design_assoc / n_design_valid, 3)
    if n_design_valid > 0
    else float("nan")
  )
  shuffle_rate = (
    round(n_shuffle_assoc / n_shuffle_valid, 3)
    if n_shuffle_valid > 0
    else float("nan")
  )

  if designs is not None and expected_runs_per_design is not None:
    rows_complete = bool(
      n_runs == expected_runs_per_design * len(designs)
      and n_problems == 0
      and not problems
    )
  else:
    rows_complete = bool(n_runs > 0 and n_problems == 0 and not problems)

  # Outcome logic
  if not controls_ok or not rows_complete:
    outcome = "inconclusive"
  elif not math.isnan(design_rate) and design_rate <= monomeric_max_rate:
    outcome = "monomeric"
  elif not math.isnan(design_rate) and design_rate >= self_associates_min_rate:
    outcome = "self_associates"
  else:
    outcome = "inconclusive"

  st = _selftest_sa()
  selftest_all_pass = bool(st["all_pass"])

  return {
    "controls_ok": controls_ok,
    "design_rate": design_rate,
    "n_design_assoc": n_design_assoc,
    "n_pos_ctl_assoc": n_pos_ctl_assoc,
    "n_neg_ctl_assoc": n_neg_ctl_assoc,
    "n_gcn4_assoc": n_pos_ctl_assoc,
    "n_ubiquitin_assoc": n_neg_ctl_assoc,
    "n_problems": n_problems,
    "n_runs": n_runs,
    "n_shuffle_assoc": n_shuffle_assoc,
    "problems": problems[:20],
    "rows_complete": rows_complete,
    "selftest_all_pass": selftest_all_pass,
    "shuffle_rate": shuffle_rate,
    "outcome": outcome,
  }


# ==============================================================================
# Slice 2: Shrake-Rupley SASA and Buried Binder Surface Area
# ==============================================================================


def _generate_sphere_points(n_points: int = 100) -> np.ndarray:
  """Generate approximately uniform points on the unit sphere via Fibonacci spiral."""
  d_lambda = np.pi * (3.0 - np.sqrt(5.0))
  dz = 2.0 / n_points
  indices = np.arange(n_points)
  z = 1.0 - dz / 2.0 - indices * dz
  r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
  lam = indices * d_lambda
  x = r * np.cos(lam)
  y = r * np.sin(lam)
  return np.column_stack((x, y, z))


def compute_shrake_rupley(
  coords: np.ndarray,
  elements: Sequence[str],
  probe_radius: float = 1.40,
  n_points: int = 100,
) -> tuple[float, np.ndarray]:
  """Compute atomic solvent accessible surface area (SASA) via Shrake-Rupley.

  Experimental: not covered by proxide's stability guarantee; may change in a minor release.

  Args:
      coords: (N, 3) array of atomic coordinates in Angstroms.
      elements: Sequence of length N containing chemical element symbols.
      probe_radius: Radius of the solvent probe in Angstroms (default 1.40 A).
      n_points: Number of sphere points for surface integration (default 100).

  Returns:
      tuple of (total_sasa_float, per_atom_sasa_array).
  """
  if len(coords) != len(elements):
    raise ValueError(
      f"Length mismatch: coords has {len(coords)} atoms, but elements has {len(elements)}"
    )

  n_atoms = len(coords)
  if n_atoms == 0:
    return 0.0, np.array([], dtype=float)

  coords_arr = np.asarray(coords, dtype=float)
  radii = np.array(
    [
      BONDI_RADII.get("C" if e.upper() == "CA" else e.upper(), DEFAULT_RADIUS)
      for e in elements
    ],
    dtype=float,
  )
  eff_radii = radii + probe_radius
  max_eff = float(np.max(eff_radii))
  pts = _generate_sphere_points(n_points)

  tree = cKDTree(coords_arr)
  sasa = np.zeros(n_atoms, dtype=float)

  for i in range(n_atoms):
    Ri = eff_radii[i]
    ci = coords_arr[i]
    nbr_indices = tree.query_ball_point(ci, r=Ri + max_eff)
    filtered_nbrs = [
      j
      for j in nbr_indices
      if j != i and np.linalg.norm(ci - coords_arr[j]) < Ri + eff_radii[j]
    ]
    if not filtered_nbrs:
      sasa[i] = 4.0 * np.pi * Ri * Ri
      continue

    tp = ci + Ri * pts
    accessible = np.ones(n_points, dtype=bool)
    for j in filtered_nbrs:
      Rj2 = eff_radii[j] ** 2
      cj = coords_arr[j]
      dist2 = np.sum((tp - cj) ** 2, axis=1)
      accessible[dist2 < Rj2] = False
      if not np.any(accessible):
        break

    sasa[i] = 4.0 * np.pi * Ri * Ri * (np.sum(accessible) / n_points)

  return float(np.sum(sasa)), sasa


def compute_buried_sasa(
  structure_or_free_coords: Any = None,
  *args: Any,
  chain_a: str = "A",
  chain_b: str = "B",
  binder_chain: str | None = None,
  probe_radius: float = 1.40,
  n_points: int = 100,
  **kwargs: Any,
) -> tuple[float, int, dict[str, Any]]:
  """Compute buried solvent accessible surface area of binder chain upon binding.

  Delta SASA_A = SASA_free - SASA_complex.
  Enforces invariant S1: complex_sasa <= free_sasa + 1e-5.

  Experimental: not covered by proxide's stability guarantee; may change in a minor release.

  Returns:
      (delta_sasa_float, delta_sasa_int, details_dict)
  """
  b_chain = binder_chain or chain_a
  t_chain = chain_b

  if len(args) >= 5 or (
    isinstance(structure_or_free_coords, np.ndarray) and len(args) >= 5
  ):
    free_coords = np.asarray(structure_or_free_coords, dtype=float)
    free_elements = list(args[0])
    free_resids = list(args[1])
    complex_coords = np.asarray(args[2], dtype=float)
    complex_elements = list(args[3])
    _complex_resids = list(args[4])

    if len(free_coords) != len(free_elements):
      n_c, n_e = len(free_coords), len(free_elements)
      raise ValueError(
        f"Length mismatch: free_coords has {n_c} atoms, but free_elements has {n_e}"
      )
    if len(free_coords) != len(free_resids):
      n_c, n_r = len(free_coords), len(free_resids)
      raise ValueError(
        f"Length mismatch: free_coords has {n_c} atoms, but free_resids has {n_r}"
      )
    if len(complex_coords) != len(complex_elements):
      n_c, n_e = len(complex_coords), len(complex_elements)
      raise ValueError(
        f"Length mismatch: complex_coords has {n_c} atoms, but complex_elements has {n_e}"
      )
    if len(complex_coords) < len(free_coords):
      n_comp, n_free = len(complex_coords), len(free_coords)
      raise ValueError(
        f"complex_coords ({n_comp} atoms) cannot have fewer atoms than free_coords ({n_free} atoms)"
      )
  else:
    atoms = _normalize_atoms(structure_or_free_coords)
    atoms_a = [x for x in atoms if x["chain"] == b_chain]
    atoms_b = [x for x in atoms if x["chain"] == t_chain]
    if not atoms_a:
      raise KeyError(f"Binder chain '{b_chain}' not found in structure")
    if not atoms_b:
      raise KeyError(f"Target chain '{t_chain}' not found in structure")

    free_coords = np.array([x["xyz"] for x in atoms_a], dtype=float)
    free_elements = [
      _infer_element_symbol(x.get("element") or x.get("name", ""))
      for x in atoms_a
    ]
    free_resids = [
      (x["num"], x["ins_code"]) if x.get("ins_code") else x["num"]
      for x in atoms_a
    ]

    atoms_comp = atoms_a + atoms_b
    complex_coords = np.array([x["xyz"] for x in atoms_comp], dtype=float)
    complex_elements = [
      _infer_element_symbol(x.get("element") or x.get("name", ""))
      for x in atoms_comp
    ]

  _, sasa_free_atoms = compute_shrake_rupley(
    free_coords, free_elements, probe_radius=probe_radius, n_points=n_points
  )
  _, sasa_complex_atoms = compute_shrake_rupley(
    complex_coords,
    complex_elements,
    probe_radius=probe_radius,
    n_points=n_points,
  )

  # Complex binder atoms correspond to the first len(free_coords) atoms
  sasa_complex_binder = sasa_complex_atoms[: len(free_coords)]

  res_sasa_free: dict[Any, float] = {}
  res_sasa_complex: dict[Any, float] = {}

  for i in range(len(free_coords)):
    rid = free_resids[i]
    res_sasa_free[rid] = res_sasa_free.get(rid, 0.0) + sasa_free_atoms[i]
    res_sasa_complex[rid] = (
      res_sasa_complex.get(rid, 0.0) + sasa_complex_binder[i]
    )

  # Check invariant S1
  bad = [
    r
    for r in sorted(
      res_sasa_free.keys(),
      key=lambda x: (x[0] if isinstance(x, tuple) else x, str(x)),
    )
    if res_sasa_complex[r] > res_sasa_free[r] + 1e-5
  ]
  if bad:
    raise ValueError(
      f"complex SASA exceeds free SASA for {len(bad)} binder residues {bad[:5]}; "
      "SASA physical control failed"
    )

  total_free = float(sum(res_sasa_free.values()))
  total_complex = float(sum(res_sasa_complex.values()))
  delta_sasa_float = total_free - total_complex
  delta_sasa_int = int(round(delta_sasa_float))

  res_sasa_buried = {r: res_sasa_free[r] - res_sasa_complex[r] for r in res_sasa_free}

  details: dict[str, Any] = {
    "free_sasa": total_free,
    "complex_sasa": total_complex,
    "delta_sasa": delta_sasa_float,
    "residue_sasa_free": res_sasa_free,
    "residue_sasa_complex": res_sasa_complex,
    "residue_sasa_buried": res_sasa_buried,
  }

  return delta_sasa_float, delta_sasa_int, details


def selftest_slice2() -> dict[str, Any]:
  """Self-contained synthetic selftest for Slice 2 utilities (Jaccard, overlap, dock, SA, SASA).

  Experimental: not covered by proxide's stability guarantee; may change in a minor release.

  Executes in-memory synthetic geometry and decision tests without filesystem or network access.
  """
  # 1. Contact Jaccard
  j_empty = contact_jaccard(set(), set()) == 0.0
  j_ident = contact_jaccard({1, 2, 3}, {1, 2, 3}) == 1.0
  j_part = abs(contact_jaccard({1, 2}, {2, 3}) - 1.0 / 3.0) < 1e-6
  jaccard_pass = bool(j_empty and j_ident and j_part)

  # 2. Alignment mapped overlap
  pairs = [(1, 101), (2, 102), (3, 103)]
  ov_j, ov_ha, ov_ma = aligned_contact_overlap({1, 2, 212}, {101, 102}, pairs)
  ov_empty_j, ov_empty_ha, ov_empty_ma = aligned_contact_overlap({1, 2}, {101, 102}, [])
  overlap_pass = bool(
    ov_j == 1.0
    and ov_ha == 2
    and ov_ma == 2
    and ov_empty_j == 0.0
    and ov_empty_ha == 0
    and ov_empty_ma == 0
  )

  # 3. Contacts and dock
  t_coords = np.array([[float(i) * 3.0, 0.0, 0.0] for i in range(20)])
  t_resids = list(range(1, 21))
  b_near = np.array([[float(i) * 3.0, 2.0, 0.0] for i in range(20)])
  b_far = np.array([[float(i) * 3.0, 500.0, 0.0] for i in range(20)])

  dock_near_default = contacts_and_dock(t_coords, t_resids, b_near)
  dock_near_epi = contacts_and_dock(
    t_coords, t_resids, b_near, epitope_residues={1, 2, 3, 4}
  )
  dock_far = contacts_and_dock(t_coords, t_resids, b_far)
  dock_batched = contacts_and_dock_batch(t_coords, t_resids, [b_near, b_far])

  dock_pass = bool(
    dock_near_default["docked"] is True
    and dock_near_default["n_contacted"] == 20
    and dock_near_epi["docked"] is True
    and dock_near_epi["epitope_contacts"] == 4
    and dock_far["docked"] is False
    and dock_far["n_contacted"] == 0
    and dock_batched == [dock_near_default, dock_far]
  )

  # 4. Self-association screen
  sa_geom = _selftest_sa()
  empty_counts = interface_counts([], []) == (0, 0)
  test_runs = [
    {
      "design_id": "pos",
      "group": "pos_ctl",
      "valid": 1,
      "n_if_a": 18,
      "n_if_b": 18,
      "ipsae": 0.5,
    },
    {
      "design_id": "neg",
      "group": "neg_ctl",
      "valid": 1,
      "n_if_a": 16,
      "n_if_b": 16,
      "ipsae": 0.1,
    },
    {
      "design_id": "d1",
      "group": "design",
      "valid": 1,
      "n_if_a": 5,
      "n_if_b": 5,
      "ipsae": 0.1,
    },
  ]
  screen_res = screen_self_association(test_runs)
  sa_pass = bool(
    sa_geom["all_pass"]
    and empty_counts
    and screen_res["controls_ok"] is True
    and screen_res["outcome"] == "monomeric"
    and screen_res["selftest_all_pass"] is True
  )

  # 5. SASA and S1 invariant
  c_atom = np.array([[0.0, 0.0, 0.0]])
  c_tot, _ = compute_shrake_rupley(c_atom, ["C"])
  c_expected = 4.0 * np.pi * (1.70 + 1.40) ** 2
  sasa_single_ok = abs(c_tot - c_expected) < 0.1

  free_c = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
  tgt_near = np.array([[0.0, 2.0, 0.0], [3.0, 2.0, 0.0]])
  tgt_far = np.array([[0.0, 1000.0, 0.0], [3.0, 1000.0, 0.0]])

  comp_near = np.vstack([free_c, tgt_near])
  comp_far = np.vstack([free_c, tgt_far])
  elems_free = ["C", "C"]
  elems_comp = ["C", "C", "C", "C"]
  r_free = [1, 2]
  r_comp = [1, 2, 10, 11]

  d_float_near, _, _ = compute_buried_sasa(
    free_c, elems_free, r_free, comp_near, elems_comp, r_comp
  )
  d_float_far, _, _ = compute_buried_sasa(
    free_c, elems_free, r_free, comp_far, elems_comp, r_comp
  )
  sasa_pass = bool(
    sasa_single_ok and d_float_near > 0.0 and abs(d_float_far - 0.0) < 1e-6
  )

  all_pass = bool(
    jaccard_pass and overlap_pass and dock_pass and sa_pass and sasa_pass
  )
  return {
    "all_pass": all_pass,
    "jaccard_pass": jaccard_pass,
    "overlap_pass": overlap_pass,
    "dock_pass": dock_pass,
    "sa_pass": sa_pass,
    "sasa_pass": sasa_pass,
  }


selftest_contacts = selftest_slice2

__all__ = [
  "InterfaceContactResult",
  "aligned_contact_overlap",
  "associates",
  "compute_buried_sasa",
  "compute_interface_contacts",
  "compute_interface_contacts_batch",
  "compute_shrake_rupley",
  "contact_jaccard",
  "contacts_and_dock",
  "contacts_and_dock_batch",
  "interface_counts",
  "load_aligned_pairs",
  "screen_self_association",
  "selftest_contacts",
  "selftest_slice2",
]
