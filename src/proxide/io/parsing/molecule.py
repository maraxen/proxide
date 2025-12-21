"""Molecule file parsing for small molecules (MOL2, SDF format)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Molecule:
  """Represents a small molecule with topology.

  This class holds all the information needed to parameterize a ligand
  with GAFF/GAFF2 force fields.

  Attributes:
      name: Molecule name/identifier.
      atom_names: List of atom names (e.g., ["C1", "C2", "H1"]).
      atom_types: GAFF atom types if available (e.g., ["ca", "ca", "ha"]).
      elements: Element symbols (e.g., ["C", "C", "H"]).
      positions: Atomic coordinates in Angstroms, shape (n_atoms, 3).
      charges: Partial charges in elementary charge units.
      bonds: List of (atom_idx1, atom_idx2) tuples.
      bond_orders: Bond orders corresponding to bonds list (1=single, 2=double, etc.).
      residue_name: 3-letter residue code for the molecule (default "LIG").

  """

  name: str
  atom_names: list[str]
  atom_types: list[str]
  elements: list[str]
  positions: np.ndarray
  charges: np.ndarray
  bonds: list[tuple[int, int]]
  bond_orders: list[int] = field(default_factory=list)
  residue_name: str = "LIG"

  @property
  def n_atoms(self) -> int:
    """Number of atoms in the molecule."""
    return len(self.atom_names)

  @property
  def n_bonds(self) -> int:
    """Number of bonds in the molecule."""
    return len(self.bonds)

  @classmethod
  def from_mol2(cls, path: str | Path) -> Molecule:
    """Parse a Tripos MOL2 file.

    MOL2 files commonly contain GAFF atom types assigned by antechamber.

    Args:
        path: Path to the MOL2 file.

    Returns:
        Molecule object with parsed data.

    """
    path = Path(path)

    name = path.stem
    atom_names: list[str] = []
    atom_types: list[str] = []
    elements: list[str] = []
    positions: list[list[float]] = []
    charges: list[float] = []
    bonds: list[tuple[int, int]] = []
    bond_orders: list[int] = []
    residue_name = "LIG"

    current_section = None

    with open(path) as f:
      for line in f:
        line = line.strip()

        # Section headers
        if line.startswith("@<TRIPOS>"):
          current_section = line[9:]  # e.g., "MOLECULE", "ATOM", "BOND"
          continue

        if not line or line.startswith("#"):
          continue

        if current_section == "MOLECULE":
          # First non-empty line after @<TRIPOS>MOLECULE is the name
          if not name or name == path.stem:
            name = line
          continue

        if current_section == "ATOM":
          # Format: atom_id atom_name x y z atom_type [subst_id subst_name charge [status_bit]]
          parts = line.split()
          if len(parts) >= 6:
            _atom_id = int(parts[0])  # 1-indexed
            atom_name = parts[1]
            x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
            atom_type = parts[5]  # GAFF type (e.g., ca, c3, os)

            # Extract element from atom type or name
            element = _extract_element(atom_type, atom_name)

            # Charge is optional (column 9)
            charge = float(parts[8]) if len(parts) >= 9 else 0.0

            # Residue name (column 8)
            if len(parts) >= 8:
              residue_name = parts[7]

            atom_names.append(atom_name)
            atom_types.append(atom_type)
            elements.append(element)
            positions.append([x, y, z])
            charges.append(charge)
          continue

        if current_section == "BOND":
          # Format: bond_id origin_atom_id target_atom_id bond_type [status_bits]
          parts = line.split()
          if len(parts) >= 4:
            atom1_id = int(parts[1]) - 1  # Convert to 0-indexed
            atom2_id = int(parts[2]) - 1
            bond_type = parts[3]  # "1", "2", "3", "ar", "am"

            bonds.append((atom1_id, atom2_id))
            bond_orders.append(_parse_mol2_bond_type(bond_type))
          continue

    return cls(
      name=name,
      atom_names=atom_names,
      atom_types=atom_types,
      elements=elements,
      positions=np.array(positions, dtype=np.float32),
      charges=np.array(charges, dtype=np.float32),
      bonds=bonds,
      bond_orders=bond_orders,
      residue_name=residue_name if len(residue_name) <= 4 else "LIG",
    )

  @classmethod
  def from_sdf(cls, path: str | Path, conformer_idx: int = 0) -> Molecule:
    """Parse an SDF/MOL file.

    SDF files typically don't contain GAFF types - those need to be
    assigned separately (e.g., via RDKit + antechamber or type inference).

    Args:
        path: Path to the SDF file.
        conformer_idx: Which conformer to use if file contains multiple (default 0).

    Returns:
        Molecule object with parsed data.

    """
    path = Path(path)

    with open(path) as f:
      content = f.read()

    # Split by $$$$ for multi-molecule SDF
    molecules = content.split("$$$$")
    if conformer_idx >= len(molecules):
      raise ValueError(
        f"Conformer index {conformer_idx} out of range (file has {len(molecules)} molecules)"
      )

    mol_block = molecules[conformer_idx].strip()
    if not mol_block:
      raise ValueError("Empty molecule block in SDF file")

    lines = mol_block.split("\n")

    # Line 1: Molecule name (or empty)
    name = lines[0].strip() if lines[0].strip() else path.stem

    # Lines 2-3: Comments, timestamp (skip)

    # Line 4: Counts line (V2000 format)
    # Format: aaabbblllfffcccsssxxxrrrpppiiimmmvvvvvv
    #         aaa = number of atoms, bbb = number of bonds
    counts_line = lines[3].strip()
    parts = counts_line.split()
    n_atoms = int(parts[0])
    n_bonds = int(parts[1])

    atom_names: list[str] = []
    atom_types: list[str] = []
    elements: list[str] = []
    positions: list[list[float]] = []
    charges: list[float] = []
    bonds: list[tuple[int, int]] = []
    bond_orders: list[int] = []

    # Atom block (lines 5 to 5+n_atoms-1)
    for i in range(n_atoms):
      line = lines[4 + i]
      # Format: xxxxx.xxxxyyyyy.yyyyzzzzz.zzzz aaaddcccssshhhbbbvvvHHHrrriiimmmeee
      # Positions are in first 30 chars (3 x 10 chars each), element is next 3 chars
      x = float(line[0:10].strip())
      y = float(line[10:20].strip())
      z = float(line[20:30].strip())
      element = line[31:34].strip()

      # Generate atom name from element + index
      atom_name = f"{element}{i + 1}"

      atom_names.append(atom_name)
      atom_types.append("")  # SDF doesn't have GAFF types
      elements.append(element)
      positions.append([x, y, z])
      charges.append(0.0)  # SDF charges are in properties, not atom line

    # Bond block (lines 5+n_atoms to 5+n_atoms+n_bonds-1)
    bond_start = 4 + n_atoms
    for i in range(n_bonds):
      line = lines[bond_start + i]
      # Format: 111222tttsssxxxrrrccc
      # 111 = first atom, 222 = second atom, ttt = bond type
      atom1 = int(line[0:3].strip()) - 1  # 0-indexed
      atom2 = int(line[3:6].strip()) - 1
      bond_type = int(line[6:9].strip())

      bonds.append((atom1, atom2))
      bond_orders.append(bond_type if bond_type <= 3 else 1)  # Aromatic (4) -> 1

    # TODO: Parse M  CHG lines for formal charges

    return cls(
      name=name,
      atom_names=atom_names,
      atom_types=atom_types,
      elements=elements,
      positions=np.array(positions, dtype=np.float32),
      charges=np.array(charges, dtype=np.float32),
      bonds=bonds,
      bond_orders=bond_orders,
    )

  @classmethod
  def from_smiles(cls, smiles: str, name: str = "molecule") -> Molecule:
    """Create a Molecule from a SMILES string.

    Requires RDKit for SMILES parsing and 3D coordinate generation.

    Args:
        smiles: SMILES string.
        name: Name for the molecule.

    Returns:
        Molecule object with generated 3D coordinates.

    Raises:
        ImportError: If RDKit is not installed.

    """
    try:
      from rdkit import Chem  # type: ignore[unresolved-import]
      from rdkit.Chem import AllChem  # type: ignore[unresolved-import]
    except ImportError as e:
      raise ImportError(
        "RDKit is required for SMILES parsing. Install with: pip install rdkit",
      ) from e

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
      raise ValueError(f"Failed to parse SMILES: {smiles}")

    # Add hydrogens
    mol = Chem.AddHs(mol)

    # Generate 3D coordinates
    AllChem.EmbedMolecule(mol, randomSeed=42)  # type: ignore[attr-defined]
    AllChem.MMFFOptimizeMolecule(mol)  # type: ignore[attr-defined]

    conf = mol.GetConformer()

    atom_names: list[str] = []
    atom_types: list[str] = []
    elements: list[str] = []
    positions: list[list[float]] = []
    charges: list[float] = []

    element_counts: dict[str, int] = {}

    for atom in mol.GetAtoms():
      elem = atom.GetSymbol()
      element_counts[elem] = element_counts.get(elem, 0) + 1
      atom_name = f"{elem}{element_counts[elem]}"

      pos = conf.GetAtomPosition(atom.GetIdx())

      atom_names.append(atom_name)
      atom_types.append("")  # Need separate GAFF type assignment
      elements.append(elem)
      positions.append([pos.x, pos.y, pos.z])
      charges.append(0.0)  # Need OpenFF for charges

    bonds: list[tuple[int, int]] = []
    bond_orders: list[int] = []

    for bond in mol.GetBonds():
      bonds.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
      bond_orders.append(int(bond.GetBondTypeAsDouble()))

    return cls(
      name=name,
      atom_names=atom_names,
      atom_types=atom_types,
      elements=elements,
      positions=np.array(positions, dtype=np.float32),
      charges=np.array(charges, dtype=np.float32),
      bonds=bonds,
      bond_orders=bond_orders,
    )

  def parameterize(self, bond_tolerance: float = 1.3) -> None:
    """Assign MD parameters using the Rust backend.

    This will assign GAFF atom types and LJ parameters.
    Charges currently default to zero unless set manually.
    """
    from proxide import _oxidize

    params = _oxidize.parameterize_molecule(
      self.positions,
      self.elements,
      bond_tolerance=bond_tolerance,
    )

    # Update our attributes
    self.atom_types = params["atom_types"]
    self.charges = params["charges"]

    # Optionally update topology if missing
    if not self.bonds:
      self.bonds = [tuple(b) for b in params["bonds"]]
      if "bond_params" in params:
        # We could store these too if needed
        pass

  def _to_rdkit(self):
    """Convert to RDKit Mol object.

    Requires RDKit.
    """
    try:
      from rdkit import Chem
    except ImportError as e:
      raise ImportError(
        "RDKit is required for this operation. Install with: pip install rdkit",
      ) from e

    # Build editable molecule
    mol = Chem.RWMol()

    # Add atoms
    for elem in self.elements:
      atom = Chem.Atom(elem)
      mol.AddAtom(atom)

    # Add bonds
    bond_type_map = {
      1: Chem.BondType.SINGLE,
      2: Chem.BondType.DOUBLE,
      3: Chem.BondType.TRIPLE,
    }

    for (i, j), order in zip(self.bonds, self.bond_orders, strict=True):
      bond_type = bond_type_map.get(order, Chem.BondType.SINGLE)
      mol.AddBond(i, j, bond_type)

    # Convert to regular Mol
    mol = mol.GetMol()

    # Add conformer with our coordinates
    conf = Chem.Conformer(self.n_atoms)
    for i, pos in enumerate(self.positions):
      conf.SetAtomPosition(i, pos.tolist())
    mol.AddConformer(conf, assignId=True)

    return mol


def _extract_element(atom_type: str, atom_name: str) -> str:
  """Extract element symbol from GAFF atom type or atom name.

  GAFF types use lowercase letters (ca, c3, os, n, etc.).
  """
  # GAFF element mapping (first letter or two)
  gaff_elements = {
    "c": "C",
    "n": "N",
    "o": "O",
    "s": "S",
    "p": "P",
    "f": "F",
    "cl": "Cl",
    "br": "Br",
    "i": "I",
    "h": "H",
  }

  # Check for 2-letter elements first
  if len(atom_type) >= 2 and atom_type[:2].lower() in gaff_elements:
    return gaff_elements[atom_type[:2].lower()]

  # Single letter
  if atom_type and atom_type[0].lower() in gaff_elements:
    return gaff_elements[atom_type[0].lower()]

  # Fallback: extract from atom name
  # Pattern: element followed by digits (e.g., C1, CA, N2)
  match = re.match(r"([A-Za-z]{1,2})", atom_name)
  if match:
    elem = match.group(1).capitalize()
    if elem in {"C", "N", "O", "S", "P", "H", "F", "Cl", "Br", "I"}:
      return elem

  return "C"  # Default fallback


def _parse_mol2_bond_type(bond_type: str) -> int:
  """Convert MOL2 bond type to integer bond order."""
  bond_map = {
    "1": 1,  # Single
    "2": 2,  # Double
    "3": 3,  # Triple
    "ar": 1,  # Aromatic (treat as single for now)
    "am": 1,  # Amide
    "du": 1,  # Dummy
    "un": 1,  # Unknown
    "nc": 0,  # Not connected
  }
  return bond_map.get(bond_type.lower(), 1)
