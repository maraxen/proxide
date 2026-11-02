"""Interface contact analysis and charge classification between chains."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

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


@dataclass(frozen=True)
class InterfaceContactResult:
  """Interface contact metrics between two chains in a structure."""

  n_binder_interface: int
  n_target_interface: int
  interface_residues_a: list[int]
  interface_residues_b: list[int]
  ion_pairs: int
  like_close: int
  polar_pairs: int
  nonpolar_pairs: int
  binder_if_charge: int
  target_if_charge: int
  binder_if_frac_ek: float
  binder_if_frac_hydrophobic: float
  details: dict[str, Any]

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)


def _parse_pdb_text(text: str) -> list[dict[str, Any]]:
  """Parse standard PDB ATOM/HETATM lines into atom record dictionaries."""
  atoms: list[dict[str, Any]] = []
  for ln in text.splitlines():
    if ln.startswith(("ATOM", "HETATM")) and len(ln) >= 54:
      name = ln[12:16].strip()
      res = ln[17:20].strip()
      chain = ln[21]
      try:
        num = int(ln[22:26])
        x = float(ln[30:38])
        y = float(ln[38:46])
        z = float(ln[46:54])
      except ValueError:
        continue
      atoms.append({"name": name, "res": res, "chain": chain, "num": num, "xyz": (x, y, z)})
  return atoms


def _normalize_atoms(structure: str | Path | list[dict[str, Any]]) -> list[dict[str, Any]]:
  """Normalize input representation (path, PDB string, or atom dicts) to atom records."""
  if isinstance(structure, Path):
    if not structure.exists():
      raise FileNotFoundError(f"Structure file not found: {structure}")
    return _parse_pdb_text(structure.read_text())
  elif isinstance(structure, str):
    if "\n" in structure or structure.startswith(("ATOM", "HETATM", "HEADER", "REMARK")):
      return _parse_pdb_text(structure)
    p = Path(structure)
    if p.exists() and p.is_file():
      return _parse_pdb_text(p.read_text())
    raise FileNotFoundError(f"Structure file or PDB text not found: {structure}")
  elif isinstance(structure, list):
    if not structure:
      return []
    for i, atom in enumerate(structure):
      if not isinstance(atom, dict):
        raise TypeError(f"Expected dict at index {i}, got {type(atom).__name__}")
      for key in ("chain", "name", "res", "num", "xyz"):
        if key not in atom:
          raise KeyError(f"Atom record missing required key: '{key}'")
    return structure
  else:
    raise TypeError(f"Unsupported structure input type: {type(structure).__name__}")


def compute_interface_contacts(
  structure: str | Path | list[dict[str, Any]],
  chain_a: str = "A",
  chain_b: str = "B",
  cutoff_heavy: float = 5.0,
  cutoff_ion_pair: float = 4.0,
  cutoff_like_charge: float = 4.5,
  cutoff_polar: float = 3.5,
  cutoff_nonpolar: float = 4.5,
) -> InterfaceContactResult:
  """Compute interface contacts, residue counts, and charge classifications between two chains.

  Args:
      structure: File path, PDB text string, or list of atom dicts.
      chain_a: Binder chain identifier (default 'A').
      chain_b: Target chain identifier (default 'B').
      cutoff_heavy: Heavy-atom distance cutoff for interface residues (default 5.0 Å).
      cutoff_ion_pair: Distance cutoff for cross-chain ion pairs (default 4.0 Å).
      cutoff_like_charge: Distance cutoff for cross-chain repulsive like charges (default 4.5 Å).
      cutoff_polar: Distance cutoff for polar N...O candidate hydrogen bonds (default 3.5 Å).
      cutoff_nonpolar: Distance cutoff for sidechain C...C nonpolar pairs (default 4.5 Å).

  Returns:
      InterfaceContactResult dataclass containing contact counts, residue lists, and fractions.
  """
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

  # Interface residues (any heavy atom within cutoff_heavy)
  a_if_residues = sorted(
    {atoms_a[i]["num"] for i in range(len(atoms_a)) if dist_mat[i].min() <= cutoff_heavy}
  )
  b_if_residues = sorted(
    {atoms_b[j]["num"] for j in range(len(atoms_b)) if dist_mat[:, j].min() <= cutoff_heavy}
  )

  res_map_a = {x["num"]: x["res"] for x in atoms_a}
  res_map_b = {x["num"]: x["res"] for x in atoms_b}

  # Charge classifications
  ion_pairs = 0
  like_close = 0
  polar_pairs = 0
  nonpolar_pairs = 0

  for i, a in enumerate(atoms_a):
    sx = CHARGE_SIGN.get((a["res"], a["name"]), 0)
    a_name = a["name"]
    a_is_c = len(a_name) > 0 and a_name[0] == "C" and a_name not in BACKBONE_ATOMS
    a_first = a_name[0] if a_name else ""

    for j, b in enumerate(atoms_b):
      d = dist_mat[i, j]
      sy = CHARGE_SIGN.get((b["res"], b["name"]), 0)

      if sx != 0 and sy != 0:
        if sx != sy and d <= cutoff_ion_pair:
          ion_pairs += 1
        if sx == sy and d <= cutoff_like_charge:
          like_close += 1

      b_name = b["name"]
      b_first = b_name[0] if b_name else ""
      if d <= cutoff_polar and {a_first, b_first} == {"N", "O"}:
        polar_pairs += 1

      if (
        d <= cutoff_nonpolar
        and a_is_c
        and (len(b_name) > 0 and b_name[0] == "C" and b_name not in BACKBONE_ATOMS)
      ):
        nonpolar_pairs += 1

  # Interface charges
  binder_if_charge = sum(RESIDUE_CHARGE.get(res_map_a[n], 0) for n in a_if_residues)
  target_if_charge = sum(RESIDUE_CHARGE.get(res_map_b[n], 0) for n in b_if_residues)

  # Fractions
  n_a_if = len(a_if_residues)
  ek_count = sum(res_map_a[n] in ("GLU", "LYS") for n in a_if_residues)
  hyd_count = sum(res_map_a[n] in HYDROPHOBIC_RESIDUES for n in a_if_residues)

  binder_if_frac_ek = round(ek_count / max(1, n_a_if), 2)
  binder_if_frac_hydrophobic = round(hyd_count / max(1, n_a_if), 2)

  details: dict[str, Any] = {
    "binder_len": len(res_map_a),
    "binder_net_charge": sum(RESIDUE_CHARGE.get(res_map_a[n], 0) for n in res_map_a),
    "target_len": len(res_map_b),
    "target_net_charge": sum(RESIDUE_CHARGE.get(res_map_b[n], 0) for n in res_map_b),
    "target_if_residues": " ".join(
      f"{THREE_TO_ONE.get(res_map_b[n], 'X')}{n}" for n in b_if_residues
    ),
    "interface_residues_a_types": {n: res_map_a[n] for n in a_if_residues},
    "interface_residues_b_types": {n: res_map_b[n] for n in b_if_residues},
  }

  return InterfaceContactResult(
    n_binder_interface=n_a_if,
    n_target_interface=len(b_if_residues),
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
) -> list[InterfaceContactResult]:
  """Batch interface contact computation over multiple structures."""
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
    )
    for s in structures
  ]


__all__ = [
  "InterfaceContactResult",
  "compute_interface_contacts",
  "compute_interface_contacts_batch",
]
