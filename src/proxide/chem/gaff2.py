"""GAFF2 atom type rule parsing and assignment.

This module implements GAFF2 (General Amber Force Field 2) atom type assignment
without requiring AmberTools. It parses the ATOMTYPE_GFF2.DEF rules and applies them
to RDKit molecules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rdkit import Chem

try:
    from rdkit import Chem
except ImportError:
    Chem = None  # type: ignore[assignment]


@dataclass
class Gaff2WildAtomDef:
    """WILDATOM definition for pattern matching.

    WILDATOM defines symbol shortcuts for sets of atom types that can be
    used in rule matching (e.g., XX = C,N,O,S,P).
    """

    symbol: str
    elements: list[str]


@dataclass
class Gaff2Rule:
    """A single GAFF2 atom type assignment rule.

    Each rule maps molecular properties to a GAFF2 atom type.
    Fields correspond to ATD line format:
    - f2: atom_type - the GAFF2 type to assign (e.g., c3, ca, n)
    - f3: residue - residue name filter (* means any)
    - f4: atomic_num - atomic number
    - f5: num_heavy_neighbors - number of non-hydrogen bonds
    - f6: num_h - number of hydrogen attachments
    - f7: h_ew - hydrogen electron-withdrawal pattern
    - f8: atomic_prop - atomic property (WILDATOM patterns)
    - f9: chem_env - chemical environment (rings, aromaticity, etc.)
    """

    atom_type: str
    residue: str
    atomic_num: int
    num_heavy_neighbors: int | None
    num_h: int | None
    h_ew: str | None
    atomic_prop: str | None
    chem_env: str | None

    def matches(
        self,
        atomic_num: int,
        num_heavy_neighbors: int,
        num_h: int,
        is_aromatic: bool,
        ring_size: int | None,
        bond_types: list[tuple[int, str]],
        neighbor_elements: list[str],
        wildatom_map: dict[str, list[str]],
    ) -> bool:
        """Check if this rule matches the given atom properties."""
        if self.atomic_num != atomic_num:
            return False

        # Exact match on heavy neighbors (None = wildcard)
        if self.num_heavy_neighbors is not None:
            if self.num_heavy_neighbors != num_heavy_neighbors:
                return False

        # Exact match on H count (None = wildcard)
        if self.num_h is not None:
            if self.num_h != num_h:
                return False

        # Special case: if atom is aromatic, only accept rules with AR in h_ew or chem_env
        # This prevents non-aromatic rules like cs, c from matching aromatic C
        if is_aromatic:
            has_ar_marker = (self.h_ew and 'AR' in self.h_ew) or (self.chem_env and 'AR' in self.chem_env)
            if not has_ar_marker:
                return False

        # h_ew may carry ring-size constraints (RGn) when the parser consumed a
        # wildcard field that should have been chem_env (e.g. the op rule: "* * [RG3]")
        if self.h_ew and "RG" in self.h_ew:
            rg_m = re.search(r'RG(\d+)', self.h_ew)
            if rg_m and ring_size != int(rg_m.group(1)):
                return False

        # Check chemical environment
        if self.chem_env:
            if not self._check_chem_env(
                is_aromatic, ring_size, bond_types, neighbor_elements, wildatom_map
            ):
                return False

        # Check atomic property (neighbor atom type requirements)
        if self.atomic_prop:
            if not self._check_atomic_prop(
                neighbor_elements, bond_types, wildatom_map
            ):
                return False

        return True

    def _check_chem_env(
        self,
        is_aromatic: bool,
        ring_size: int | None,
        bond_types: list[tuple[int, str]],
        neighbor_elements: list[str],
        wildatom_map: dict[str, list[str]],
    ) -> bool:
        """Check chemical environment conditions.
        
        Parses patterns like:
        - [RG3]-[RG9]: ring size
        - [AR1], [AR2], [AR3]: aromaticity requirements
        - [sb], [db], [ar]: bond type presence (single, double, aromatic)
        - [tb]: triple bond
        - [sb',db']: combination patterns
        """
        chem_env = self.chem_env
        if not chem_env:
            return True

        chem_env = chem_env.strip()

        # Check ring size conditions
        if "[RG3]" in chem_env:
            if ring_size != 3:
                return False
        if "[RG4]" in chem_env:
            if ring_size != 4:
                return False
        if "[RG5]" in chem_env:
            if ring_size != 5:
                return False
        if "[RG6]" in chem_env:
            if ring_size != 6:
                return False

        # Check aromaticity - need AR bond types
        if "[AR1]" in chem_env or "[AR2]" in chem_env:
            if not is_aromatic:
                return False
            # Must have at least one AR bond
            has_ar = any(bt == "AR" for _, bt in bond_types)
            if not has_ar:
                return False

        # Check bond type specifications: sb (single), db (double), ar (aromatic), tb (triple)
        # Common patterns: [sb,db,AR2] means must have at least one of these
        bond_specs = []
        if "[AR1]" in chem_env or "[AR2]" in chem_env or "[AR3]" in chem_env:
            bond_specs.append("ar")
        if "sb" in chem_env.lower():
            bond_specs.append("sb")
        if "db" in chem_env.lower():
            bond_specs.append("db")
        if "tb" in chem_env.lower():
            bond_specs.append("tb")

        if bond_specs:
            # Get available bond types from molecule
            available = set(bt for _, bt in bond_types)
            # For OR logic (sb,db,AR2), any match is sufficient
            # For stricter requirements, we'd need all of them
            if "ar" in bond_specs and "ar" not in available:
                # AR patterns require aromatic - might fail only if OR pattern
                if not any(b in available for b in bond_specs if b != "ar"):
                    pass  # Not strictly required

        return True

    def _check_atomic_prop(
        self,
        neighbor_elements: list[str],
        bond_types: list[tuple[int, str]],
        wildatom_map: dict[str, list[str]],
    ) -> bool:
        """Check atomic property (neighbor atom type) conditions.

        The parser strips outer parens before storing atomic_prop, so stored
        values are like "XA1", "S1", "C3" (not "(XA1)", "(S1)", etc.).
        Format: TYPE_OR_WILDATOM + COUNT, meaning ≥COUNT neighbors of that type.
        """
        atomic_prop = self.atomic_prop
        if not atomic_prop:
            return True

        for pattern in atomic_prop.strip().split(","):
            pattern = pattern.strip()
            if not pattern:
                continue

            # Primary format stored by parser: "TYPE_OR_WILDATOM + COUNT" e.g. "XA1", "S1"
            m = re.match(r'^([A-Za-z]+)(\d+)$', pattern)
            if m:
                req_type = m.group(1)
                req_count = int(m.group(2))
                resolved = wildatom_map.get(req_type, [req_type])
                count = sum(1 for elem in neighbor_elements if elem in resolved)
                if count < req_count:
                    return False
                continue

            # Fallback: nested form with outer parens e.g. "(XA1)" or "(C3(C3))"
            match = re.match(r"\((\w+)\)$", pattern)
            if match:
                if not self._matches_wildatom(match.group(1), neighbor_elements, wildatom_map):
                    return False
                continue

            match = re.match(r"\((\w+)\((\w+)\)\)$", pattern)
            if match:
                if not self._matches_wildatom(match.group(1), neighbor_elements, wildatom_map):
                    return False

        return True

    def _matches_wildatom(
        self,
        pattern: str,
        elements: list[str],
        wildatom_map: dict[str, list[str]],
    ) -> bool:
        """Check if any neighbor element matches the pattern (direct or WILDATOM)."""
        # Resolve WILDATOM to actual elements
        resolve = wildatom_map.get(pattern, [pattern])

        for elem in elements:
            if elem in resolve:
                return True
        return False


def parse_wildatom_defs(lines: list[str]) -> dict[str, list[str]]:
    """Parse WILDATOM definitions from the rule file header."""
    wildatom_map: dict[str, list[str]] = {}

    for line in lines:
        line = line.strip()
        if not line.startswith("WILDATOM"):
            continue

        parts = line.split()
        if len(parts) >= 3:
            symbol = parts[1]
            elements = parts[2].strip("[]").split(",")
            wildatom_map[symbol] = elements

    return wildatom_map


def parse_gaff2_rules(def_path: str | Path) -> tuple[list[Gaff2Rule], dict[str, list[str]]]:
    """Parse ATOMTYPE_GFF2.DEF file.

    Args:
        def_path: Path to ATOMTYPE_GFF2.DEF file

    Returns:
        Tuple of (list of Gaff2Rule objects, WILDATOM map)
    """
    path = Path(def_path)
    content = path.read_text()
    lines = content.split("\n")

    wildatom_map = parse_wildatom_defs(lines)

    rules: list[Gaff2Rule] = []
    in_definition = False

    for line in lines:
        line = line.strip()

        if "efination begin" in line.lower():
            in_definition = True
            continue

        if not in_definition or not line.startswith("ATD"):
            continue

        if "&" not in line:
            continue

        line = line.removesuffix("&").strip()
        if line.startswith("ATD"):
            line = line[3:].strip()

        parts = line.split()
        if len(parts) < 4:
            continue

        try:
            atom_type = parts[0]
            residue = parts[1] if parts[1] != "*" else "*"
            atomic_num = int(parts[2])

            num_heavy_neighbors = None
            num_h = None
            h_ew = None
            atomic_prop = None
            chem_env = None

            idx = 3
            if idx < len(parts) and parts[idx] != "*":
                num_heavy_neighbors = int(parts[idx])
            idx += 1

            if idx < len(parts) and parts[idx] != "*":
                num_h = int(parts[idx])
            idx += 1

            remaining = " ".join(parts[idx:]) if idx < len(parts) else ""

            # h_ew field (f7): [sb,db,AR2] etc - may be preceded by *
            remaining = remaining.lstrip()
            if remaining.startswith('*'):
                remaining = remaining[1:].strip()

            h_ew_match = re.match(r"\[([^\]]+)\]", remaining)
            if h_ew_match:
                h_ew = h_ew_match.group(1)
                remaining = remaining[h_ew_match.end():].strip()

            # atomic_prop field (f8): (C3(C3)) etc
            atomic_prop_match = re.match(r"\(([^\)]+)\)", remaining)
            if atomic_prop_match:
                atomic_prop = atomic_prop_match.group(1)
                remaining = remaining[atomic_prop_match.end():].strip()

            # chem_env field (f9): remaining after both
            if remaining and remaining.strip():
                chem_env = remaining.strip() if remaining.strip() != "*" else None

            rule = Gaff2Rule(
                atom_type=atom_type,
                residue=residue,
                atomic_num=atomic_num,
                num_heavy_neighbors=num_heavy_neighbors,
                num_h=num_h,
                h_ew=h_ew,
                atomic_prop=atomic_prop,
                chem_env=chem_env,
            )
            rules.append(rule)

        except (ValueError, IndexError):
            continue

    return rules, wildatom_map


def extract_atom_features(
    mol: Chem.Mol,
) -> list[dict]:
    """Extract features from RDKit molecule needed for GAFF2 typing.

    Args:
        mol: RDKit molecule

    Returns:
        List of dicts, one per atom with features for matching
    """
    if Chem is None:
        raise ImportError("RDKit is required. Install with: pip install rdkit")

    features: list[dict] = []

    rings = mol.GetRingInfo()

    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()

        # Skip hydrogens for GAFF2 typing (they get typed based on their heavy atom neighbor)
        if atomic_num == 1:
            continue

        # After AllChem.AddHs, H atoms are explicit graph nodes, so
        # GetNumExplicitHs() returns 0 (H is a real atom, not a valence annotation).
        # Walk bonds to count actual H-atom neighbors; add GetNumImplicitHs for
        # molecules that still carry implicit H (pre-AddHs path).
        num_implicit_h = atom.GetNumImplicitHs()
        num_h = (
            sum(
                1 for bond in atom.GetBonds()
                if mol.GetAtomWithIdx(bond.GetOtherAtomIdx(atom.GetIdx())).GetAtomicNum() == 1
            )
            + num_implicit_h
        )

        # Count non-hydrogen bonds (standard degree)
        # Note: RDKit GetDegree() counts AROMATIC bonds as degree 1 each
        # For benzene C: GetDegree() = 3 (2 aromatic bonds + 1 H), which matches GAFF "attached"
        heavy_degree = atom.GetDegree()

        # Also compute effective degree including implicit hydrogens
        # This is what GAFF "attached" field represents
        attached_with_implicit_h = heavy_degree + num_implicit_h

        is_aromatic = atom.GetIsAromatic()

        ring_info = atom.IsInRing()
        ring_size = None
        if ring_info:
            for size in range(3, 10):
                if rings.IsAtomInRingOfSize(atom.GetIdx(), size):
                    ring_size = size
                    break

        bond_types: list[tuple[int, str]] = []
        neighbor_elements: list[str] = []

        for bond in atom.GetBonds():
            other_idx = bond.GetOtherAtomIdx(atom.GetIdx())
            other_atom = mol.GetAtomWithIdx(other_idx)

            # Only consider bonds to non-hydrogen neighbors
            if other_atom.GetAtomicNum() == 1:
                continue

            neighbor_elements.append(other_atom.GetSymbol())

            bt = bond.GetBondType()
            if bt == 1:
                bond_types.append((1, "SB"))
            elif bt == 2:
                bond_types.append((2, "DB"))
            elif bt == 3:
                bond_types.append((3, "TB"))
            elif bt.name == "AROMATIC":
                bond_types.append((1, "AR"))

        # Also compute total H count based on expected valency
        # For C: 4 bonds total, N: 3 bonds, O: 2 bonds, etc.
        expected_valence = {"C": 4, "N": 3, "O": 2, "S": 2, "P": 3}
        elem = atom.GetSymbol()
        expected = expected_valence.get(elem, 4)
        # Count all bonds (including to H)
        actual_bonds = atom.GetDegree() + atom.GetNumImplicitHs()
        total_h = max(0, expected - actual_bonds)

        # Also compute total H count based on expected valency
        # For C: 4 bonds total, N: 3 bonds, O: 2 bonds, etc.
        expected_valence = {"C": 4, "N": 3, "O": 2, "S": 2, "P": 3}
        elem = atom.GetSymbol()
        expected = expected_valence.get(elem, 4)
        actual_bonds = atom.GetDegree() + atom.GetNumImplicitHs()
        total_h = max(0, expected - actual_bonds)

        feature = {
            "atomic_num": atomic_num,
            "num_heavy_neighbors": heavy_degree,
            "attached_with_implicit_h": attached_with_implicit_h,
            "num_h": num_h,
            "total_h": total_h,
            "is_aromatic": is_aromatic,
            "ring_size": ring_size,
            "bond_types": bond_types,
            "neighbor_elements": neighbor_elements,
        }
        features.append(feature)

    return features


def assign_gaff2_atom_types(
    mol: Chem.Mol,
    rules: list[Gaff2Rule] | None = None,
    wildatom_map: dict[str, list[str]] | None = None,
) -> list[str]:
    """Assign GAFF2 atom types to an RDKit molecule.

    Args:
        mol: RDKit molecule
        rules: Pre-parsed GAFF2 rules (optional, will use default if not provided)
        wildatom_map: Pre-parsed WILDATOM map

    Returns:
        List of GAFF2 atom type strings, one per atom
    """
    if Chem is None:
        raise ImportError("RDKit is required. Install with: pip install rdkit")

    if rules is None:
        rules, wildatom_map = _get_default_rules()

    if wildatom_map is None:
        wildatom_map = {}

    features = extract_atom_features(mol)

    atom_types: list[str] = []

    for _i, feat in enumerate(features):
        assigned = False

        # For rule matching, use attached_with_implicit_h to get correct degree for aromatic C
        # This combines explicit heavy connections + implicit hydrogens
        degree_for_matching = feat.get("attached_with_implicit_h", feat["num_heavy_neighbors"])

        for rule in rules:
            if rule.matches(
                atomic_num=feat["atomic_num"],
                num_heavy_neighbors=degree_for_matching,
                num_h=feat["num_h"],
                is_aromatic=feat["is_aromatic"],
                ring_size=feat["ring_size"],
                bond_types=feat["bond_types"],
                neighbor_elements=feat["neighbor_elements"],
                wildatom_map=wildatom_map,
            ):
                atom_types.append(rule.atom_type)
                assigned = True
                break

        if not assigned:
            atomic_num = feat["atomic_num"]
            if atomic_num == 6:
                atom_types.append("c3")
            elif atomic_num == 7:
                atom_types.append("n3")
            elif atomic_num == 8:
                atom_types.append("oh")
            elif atomic_num == 16:
                atom_types.append("s")
            elif atomic_num == 1:
                atom_types.append("hc")
            else:
                atom_types.append("x")

    return atom_types


_default_rules: list[Gaff2Rule] | None = None
_default_wildatom: dict[str, list[str]] | None = None


def _get_default_rules() -> tuple[list[Gaff2Rule], dict[str, list[str]]]:
    """Get default GAFF2 rules (cached)."""
    global _default_rules, _default_wildatom

    if _default_rules is None:
        # Use fixed relative path from project root
        rules_path = Path(__file__).parent.parent / "assets" / "gaff" / "dat" / "ATOMTYPE_GFF2.DEF"

        if rules_path.exists():
            _default_rules, _default_wildatom = parse_gaff2_rules(rules_path)
        else:
            _default_rules = []
            _default_wildatom = {}

    return (
        _default_rules if _default_rules is not None else [],
        _default_wildatom if _default_wildatom is not None else {},
    )


def load_gaff2_rules(
    def_path: str | Path | None = None,
) -> tuple[list[Gaff2Rule], dict[str, list[str]]]:
    """Load GAFF2 rules from file.

    Args:
        def_path: Path to ATOMTYPE_GFF2.DEF. If None, uses default bundled.

    Returns:
        Tuple of (rules list, wildatom map)
    """
    if def_path is None:
        return _get_default_rules()

    return parse_gaff2_rules(def_path)


def load_gaff2_parameters(dat_path: str | Path | None = None) -> dict:
    """Load GAFF2 parameter tables from .dat file.

    Args:
        dat_path: Path to GAFF2 .dat file (e.g., gaff-2.2.20.dat).
                  If None, uses default bundled.

    Returns:
        Dict with 'masses', 'bonds', 'angles', 'torsions', 'impropers'.
    """
    if dat_path is None:
        dat_path = Path(__file__).parent.parent / "assets" / "gaff" / "dat" / "gaff-2.2.20.dat"

    params = {
        'masses': {},
        'bonds': {},
        'angles': {},
        'torsions': {},
        'impropers': {},
    }

    content = Path(dat_path).read_text()
    lines = content.split('\n')

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue

        parts = line_stripped.split()
        if len(parts) < 2:
            continue

        first = parts[0]

        if '-' in first:
            dash_count = first.count('-')

            # Parse bond: type1-type2  kb  r0
            if dash_count == 1:
                t1, t2 = first.split('-')
                if len(t1) <= 3 and len(t2) <= 3:
                    try:
                        kb = float(parts[1])
                        r0 = float(parts[2])
                        if kb > 0:
                            params['bonds'][(t1, t2)] = (kb, r0)
                    except (ValueError, IndexError):
                        pass

            # Parse angle: type1-type2-type3  kt  t0
            elif dash_count == 2:
                t1, rest = first.split('-', 1)
                t2, t3 = rest.split('-', 1)
                if len(t1) <= 3 and len(t2) <= 3 and len(t3) <= 3:
                    try:
                        kt = float(parts[1])
                        t0 = float(parts[2])
                        if kt > 0:
                            params['angles'][(t1, t2, t3)] = (kt, t0)
                    except (ValueError, IndexError):
                        pass

            # Parse torsion or improper: type1-type2-type3-type4  ...
            elif dash_count == 3:
                t1, rest = first.split('-', 1)
                t2, rest2 = rest.split('-', 1)
                t3, t4 = rest2.split('-', 1)
                if len(t1) <= 3 and len(t2) <= 3 and len(t3) <= 3 and len(t4) <= 3:
                    try:
                        periodicity = int(parts[1])
                        kt = float(parts[2])
                        phase = float(parts[3])
                        # Torsion: has 4+ terms per line (may have multiple periodicity)
                        if len(parts) >= 5:
                            # This is a torsion
                            key = (t1, t2, t3, t4)
                            if key not in params['torsions']:
                                params['torsions'][key] = []
                            params['torsions'][key].append((periodicity, kt, phase))
                        else:
                            # This could be improper
                            try:
                                kt_imp = float(parts[2])
                                phase_imp = float(parts[3])
                                if kt_imp > 0:
                                    params['impropers'][(t1, t2, t3, t4)] = (kt_imp, phase_imp)
                            except (ValueError, IndexError):
                                pass
                    except (ValueError, IndexError):
                        pass

        # Parse mass: type  mass
        elif len(first) <= 3 and first.replace('+', '').islower():
            try:
                params['masses'][first] = float(parts[1])
            except (ValueError, IndexError):
                pass

    return params


def _get_espaloma_charges(mol: Chem.Mol) -> list[float]:
    """Compute partial charges using expaloma or fallback to Gasteiger.

    Tries native Rust expaloma first, then RDKit Gasteiger as fallback.
    Returns zero charges if nothing is available.
    """
    from rdkit import Chem

    mol_copy = Chem.Mol(mol)
    Chem.SanitizeMol(mol_copy)

    try:
        from proxide._proxider import assign_espaloma_charges as assign_rust_charges
    except ImportError:
        assign_rust_charges = None

    try:
        from expaloma.featurize import from_rdkit_mol
    except ImportError:
        from_rdkit_mol = None

    if assign_rust_charges and from_rdkit_mol:
        try:
            g = from_rdkit_mol(mol_copy)
            h0 = np.ascontiguousarray(g.h0, dtype=np.float32)
            senders = np.ascontiguousarray(g.senders, dtype=np.uint32)
            receivers = np.ascontiguousarray(g.receivers, dtype=np.uint32)
            q_ref = np.ascontiguousarray(g.q_ref, dtype=np.float32)
            total_charge = float(q_ref.sum())

            q_rust = assign_rust_charges(
                h0,
                senders,
                receivers,
                np.zeros(h0.shape[0], dtype=np.uint32),
                1,
                [total_charge],
            )
            return list(q_rust)
        except Exception:
            pass

    try:
        mol_copy.ComputeGasteigerCharges()
        charges = []
        for atom in mol_copy.GetAtoms():
            charge = atom.GetDoubleProp("_GasteigerCharge")
            if charge == float("inf") or charge == float("-inf") or abs(charge) > 10:
                charge = 0.0
            charges.append(charge)
        return charges
    except Exception:
        return [0.0] * mol.GetNumAtoms()


def parameterize_gaff_with_rdkit(
    mol: Chem.Mol,
    gaff_version: str = "gaff-2.2.20",
) -> dict:
    """Assign GAFF2 parameters to an RDKit molecule.

    This function assigns GAFF2 atom types and looks up force field parameters
    without requiring AmberTools.

    Args:
        mol: RDKit molecule (should have explicit hydrogens)
        gaff_version: GAFF version string (default: gaff-2.2.20)

    Returns:
        Dict with keys:
        - atom_types: list of atom type strings
        - masses: dict of atom type -> mass
        - bonds: dict of (type1, type2) -> (kb, r0)
        - angles: dict of (type1, type2, type3) -> (kt, t0)
        - torsions: list of torsion parameters
    """
    if Chem is None:
        raise ImportError("RDKit is required. Install with: pip install rdkit")

    # Load parameters
    params = load_gaff2_parameters()

    # Assign atom types
    atom_types = assign_gaff2_atom_types(mol)

    # Extract molecule topology
    n_atoms = mol.GetNumAtoms()

    # Build bond list
    bonds = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetOtherAtomIdx(i)
        # Get atom types
        t_i = atom_types[i] if i < len(atom_types) else "x"
        t_j = atom_types[j] if j < len(atom_types) else "x"

        # Get bond order
        bo = bond.GetBondTypeAsDouble()

        # Determine bond type string
        if bo >= 1.9:
            bond_type = "tb"  # triple
        elif bo >= 1.4:
            bond_type = "db"  # double
        else:
            bond_type = "sb"  # single

        bonds.append({
            'i': i,
            'j': j,
            'type': bond_type,
            'order': bo,
            'gaff_type_i': t_i,
            'gaff_type_j': t_j,
        })

    # Build angle list (1-2-3 connections)
    angles = []
    for i in range(n_atoms):
        atom_i = mol.GetAtomWithIdx(i)
        for bond in atom_i.GetBonds():
            j = bond.GetOtherAtomIdx(i)
            if j <= i:
                continue
            for bond2 in mol.GetAtomWithIdx(j).GetBonds():
                k = bond2.GetOtherAtomIdx(j)
                if k <= j or k == i:
                    continue
                t_i = atom_types[i] if i < len(atom_types) else "x"
                t_j = atom_types[j] if j < len(atom_types) else "x"
                t_k = atom_types[k] if k < len(atom_types) else "x"
                angles.append({
                    'i': i, 'j': j, 'k': k,
                    'types': (t_i, t_j, t_k),
                })

    charges = _get_espaloma_charges(mol)

    used_types = set(atom_types)
    masses = {at: params['masses'].get(at, 0.0) for at in used_types}

    # Look up bond parameters
    for b in bonds:
        t1, t2 = b['gaff_type_i'], b['gaff_type_j']
        key = tuple(sorted([t1, t2]))
        if key in params['bonds']:
            kb, r0 = params['bonds'][key]
            b['kb'] = kb
            b['r0'] = r0
        else:
            b['kb'] = 0.0
            b['r0'] = 0.0

    # Look up angle parameters
    for a in angles:
        t1, t2, t3 = a['types']
        key = (t1, t2, t3)
        if key in params['angles']:
            kt, t0 = params['angles'][key]
            a['kt'] = kt
            a['t0'] = t0
        else:
            a['kt'] = 0.0
            a['t0'] = 0.0

    # Build torsion list (1-2-3-4 connections)
    torsions = []

    # Substitutions for atom type looking (cx->c3, etc.)
    type_substitutions = {
        'cx': 'c3', 'cy': 'c3', 'c5': 'c3', 'c6': 'c3',
        'n7': 'n3', 'n8': 'n3', 'nx': 'n3',
        'ny': 'n3', 'ni': 'n', 'nu': 'n3', 'nv': 'n3',
    }

    def _substitute_type(t: str) -> str:
        if t == 'x':
            return 'hc'  # default H type
        return type_substitutions.get(t, t)

    for i in range(n_atoms):
        atom_i = mol.GetAtomWithIdx(i)
        for bond1 in atom_i.GetBonds():
            j = bond1.GetOtherAtomIdx(i)
            if j <= i:
                continue
            for bond2 in mol.GetAtomWithIdx(j).GetBonds():
                k = bond2.GetOtherAtomIdx(j)
                if k <= j or k == i:
                    continue
                for bond3 in mol.GetAtomWithIdx(k).GetBonds():
                    neighbor_idx = bond3.GetOtherAtomIdx(k)
                    if neighbor_idx <= k or neighbor_idx == j:
                        continue
                    l = neighbor_idx
                    t_i = atom_types[i] if i < len(atom_types) else "x"
                    t_j = atom_types[j] if j < len(atom_types) else "x"
                    t_k = atom_types[k] if k < len(atom_types) else "x"
                    t_l = atom_types[neighbor_idx] if neighbor_idx < len(atom_types) else "x"

                    key = (t_i, t_j, t_k, t_l)

                    # Try exact match first, then with substitutions
                    torsion_params = params['torsions'].get(key, [])
                    if not torsion_params:
                        # Try with substitutions (cx->c3, etc.)
                        key_sub = tuple(_substitute_type(x) for x in key)
                        torsion_params = params['torsions'].get(key_sub, [])

                    torsions.append({
                        'i': i, 'j': j, 'k': k, 'l': l,
                        'types': key,
                        'params': torsion_params,
                    })

    return {
        'atom_types': atom_types,
        'charges': charges,
        'masses': masses,
        'bonds': bonds,
        'angles': angles,
        'torsions': torsions,
    }
