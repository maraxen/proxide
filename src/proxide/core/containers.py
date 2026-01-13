"""Dataclasses for protein structures.

This module defines:
- Protein: Residue-level protein representation (Atom37 format)
- ProteinStream, ProteinBatch: Type aliases for sequences
"""

from __future__ import annotations

from collections.abc import Generator, Sequence
from typing import TYPE_CHECKING, Any, Literal

import jax.numpy as jnp
import numpy as np
from flax.struct import dataclass

from proxide.chem.residues import atom_order

if TYPE_CHECKING:
  from proxide.core.types import (
    AlphaCarbonMask,
    BackboneDihedrals,
    ChainIndex,
    OneHotProteinSequence,
    ProteinSequence,
    ResidueIndex,
    StructureAtomicCoordinates,
  )


def include_feature(feature_name: str, include_features: Sequence[str] | None) -> bool:
  """Determine if a feature should be included.

  Args:
      feature_name (str): The name of the feature to check.
      include_features (Sequence[str] | None): The list of features to include.
          If None, no features are included.

  Returns:
      bool: True if the feature should be included, False otherwise.

  """
  if include_features is None:
    return False
  return feature_name in include_features or "all" in include_features


def none_or_jnp(array: np.ndarray | None) -> jnp.ndarray | None:
  """Convert a numpy array to jnp array, or return None if input is None."""
  if array is None:
    return None
  return jnp.asarray(array)


def none_or_numpy(array: np.ndarray | None) -> np.ndarray | None:
  """Convert to numpy array, or return None if input is None."""
  if array is None:
    return None
  return np.asarray(array)


@dataclass(frozen=True, kw_only=True)
class Protein:
  """Residue-level protein structure representation.

  This class represents a protein structure in the 'Atom37' format (N_res, 37, 3),
  commonly used in AlphaFold and protein design models. It also serves as a container
  for derived features like physics parameters and geometric graphs.

  Attributes:
    coordinates: Atom positions in Atom37 format. Shape (N_res, 37, 3).
    aatype: Amino acid type per residue (0-20). Shape (N_res,).
    residue_index: PDB residue numbering. Shape (N_res,).
    chain_index: Chain identifier per residue (integer encoded). Shape (N_res,).
    one_hot_sequence: One-hot encoded sequence. Shape (N_res, 21).
    mask: Alpha carbon presence mask (1.0 for valid, 0.0 for padding). Shape (N_res,).
    atom_mask: Per-atom validity mask. Shape (N_res, 37).
    dihedrals: Backbone dihedral angles (phi, psi, omega). Shape (N_res, 3).
    rbf_features: Pre-computed Radial Basis Function features. Shape (N_res, K, F).
    physics_features: Pre-computed physics/chemistry features. Shape (N_res, F).
    neighbor_indices: Indices of K-nearest neighbors. Shape (N_res, K).
    charges: Partial charges for MD (if parameterized). Shape (N_atoms,) or (N_res, 37).
    sigmas: Lennard-Jones sigma parameters. Shape (N_atoms,) or (N_res, 37).
    epsilons: Lennard-Jones epsilon parameters. Shape (N_atoms,) or (N_res, 37).

  """

  coordinates: StructureAtomicCoordinates
  aatype: ProteinSequence
  residue_index: ResidueIndex
  chain_index: ChainIndex
  one_hot_sequence: OneHotProteinSequence | None = None
  mask: AlphaCarbonMask | None = None
  atom_mask: Any | None = None
  dihedrals: BackboneDihedrals | None = None
  mapping: Any | None = None
  full_coordinates: Any | None = None
  full_atom_mask: Any | None = None
  source: str | None = None

  # Derived features
  physics_features: Any | None = None
  backbone_indices: Any | None = None
  vdw_features: Any | None = None
  rbf_features: Any | None = None
  neighbor_indices: Any | None = None
  electrostatic_features: Any | None = None
  format: Literal["Atom37", "Atom14", "Full", "BackboneOnly"] | None = None

  # Legacy AtomicSystem fields for backward compat during transition
  elements: Any | None = None
  atom_names: Any | None = None
  chain_ids: Any | None = None
  res_names: Any | None = None
  molecule_type: Any | None = None
  atom_types: Any | None = None
  bonds: Any | None = None
  bond_params: Any | None = None
  angles: Any | None = None
  angle_params: Any | None = None
  proper_dihedrals: Any | None = None
  dihedral_params: Any | None = None
  impropers: Any | None = None
  improper_params: Any | None = None
  exclusion_mask: Any | None = None
  charges: Any | None = None
  sigmas: Any | None = None
  epsilons: Any | None = None
  radii: Any | None = None
  masses: Any | None = None
  atom_res_index: Any | None = None

  @classmethod
  def from_rust_dict(
    cls,
    rust_dict: dict,
    source: str | None = None,
    use_jax: bool = True,
  ) -> Protein:
    """Create a Protein instance directly from Rust parser output dictionary.

    This method converts the dictionary output from `oxidize.parse_structure` into
    a `Protein` dataclass, handling type conversion (JAX/NumPy), reshaping for Atom37,
    and unit scaling for physics parameters.

    Process:
        1.  **Detection**: Determine format (Atom37 vs Flat) based on coordinate shape.
        2.  **Conversion**: Convert all arrays to JAX or NumPy based on `use_jax`.
        3.  **Scaling**: Convert units from MD standard (nm, kJ/mol) to Angstroms/kcal/mol.
            -   Length: nm -> Angstrom (x10)
            -   Energy: kJ/mol -> kcal/mol (x0.239)
        4.  **Reshaping**: If Atom37, reshape coordinates (N, 37, 3) and masks.
        5.  **Construction**: Populate the dataclass fields.

    Args:
        rust_dict: Dictionary returned by `oxidize.parse_structure()`.
        source: Optional source identifier (e.g., filename) for metadata.
        use_jax: If True, convert arrays to `jax.numpy` arrays.
                 If False, use `numpy` arrays.

    Returns:
        A formatted `Protein` dataclass instance.

    """
    num_residues = len(rust_dict["aatype"])

    raw_coords = rust_dict["coordinates"]
    raw_mask = rust_dict["atom_mask"]

    is_atom37 = (
      (raw_coords.ndim == 3 and raw_coords.shape[1] == 37)
      or (raw_coords.ndim == 2 and raw_coords.size == num_residues * 37 * 3)
      or (raw_coords.ndim == 1 and raw_coords.size == num_residues * 37 * 3)
    )

    def convert(x: Any, dtype: Any = None) -> Any:
      if use_jax:
        return jnp.asarray(x, dtype=dtype)
      return np.asarray(x, dtype=dtype)

    # Unit conversion constants
    NM_TO_ANGSTROM = 10.0
    KJMOL_TO_KCALMOL = 0.2390057
    BOND_K_FACTOR = KJMOL_TO_KCALMOL / (NM_TO_ANGSTROM**2)

    def convert_params(arr: Any, type_: str) -> Any:
      if arr is None:
        return None
      arr = np.array(arr, dtype=np.float32)
      if type_ == "bond":
        arr[:, 0] *= NM_TO_ANGSTROM
        arr[:, 1] *= BOND_K_FACTOR
      elif type_ == "angle":
        arr[:, 1] *= KJMOL_TO_KCALMOL
      elif type_ == "dihedral" or type_ == "improper":
        arr[:, 2] *= KJMOL_TO_KCALMOL
      elif type_ == "length":
        arr *= NM_TO_ANGSTROM
      elif type_ == "energy":
        arr *= KJMOL_TO_KCALMOL
      if use_jax:
        return jnp.asarray(arr)
      return arr

    if is_atom37:
      coordinates = raw_coords.reshape(num_residues, 37, 3)
      atom_mask_2d = raw_mask.reshape(num_residues, 37)
      mask_ca = atom_mask_2d[:, atom_order["CA"]]

      return cls(
        coordinates=convert(coordinates, dtype=np.float32),
        aatype=convert(rust_dict["aatype"], dtype=np.int8),
        one_hot_sequence=convert(np.eye(21)[rust_dict["aatype"]]),
        mask=convert(mask_ca, dtype=np.float32),
        residue_index=convert(rust_dict["residue_index"], dtype=np.int32),
        chain_index=convert(rust_dict["chain_index"], dtype=np.int32),
        atom_mask=convert(atom_mask_2d, dtype=np.float32),
        # Optional fields
        charges=convert(rust_dict["charges"]) if rust_dict.get("charges") is not None else None,
        radii=convert_params(rust_dict.get("radii") or rust_dict.get("gbsa_radii"), "length"),
        sigmas=convert_params(rust_dict.get("sigmas"), "length"),
        epsilons=convert_params(rust_dict.get("epsilons"), "energy"),
        molecule_type=convert(rust_dict["molecule_type"])
        if rust_dict.get("molecule_type") is not None
        else None,
        atom_types=rust_dict.get("atom_types"),
        bonds=convert(rust_dict["bonds"]) if rust_dict.get("bonds") is not None else None,
        bond_params=convert_params(rust_dict.get("bond_params"), "bond"),
        angles=convert(rust_dict["angles"]) if rust_dict.get("angles") is not None else None,
        angle_params=convert_params(rust_dict.get("angle_params"), "angle"),
        proper_dihedrals=convert(rust_dict["dihedrals"])
        if rust_dict.get("dihedrals") is not None
        else None,
        dihedral_params=convert_params(rust_dict.get("dihedral_params"), "dihedral"),
        impropers=convert(rust_dict["impropers"])
        if rust_dict.get("impropers") is not None
        else None,
        improper_params=convert_params(rust_dict.get("improper_params"), "improper"),
        vdw_features=convert(rust_dict["vdw_features"])
        if rust_dict.get("vdw_features") is not None
        else None,
        rbf_features=convert(rust_dict["rbf_features"])
        if rust_dict.get("rbf_features") is not None
        else None,
        neighbor_indices=convert(rust_dict["neighbor_indices"])
        if rust_dict.get("neighbor_indices") is not None
        else None,
        electrostatic_features=convert(rust_dict["electrostatic_features"])
        if rust_dict.get("electrostatic_features") is not None
        else None,
        format="Atom37",
        source=source,
        chain_ids=rust_dict.get("unique_chain_ids") or (["A"] * len(rust_dict["chain_index"])),
        full_coordinates=convert(raw_coords, dtype=np.float32).reshape(-1, 3),
        full_atom_mask=convert(raw_mask, dtype=np.float32).flatten(),
        masses=convert(rust_dict["masses"]) if rust_dict.get("masses") is not None else None,
      )

    # Flat format (Full)
    atom_names = rust_dict.get("atom_names")
    elements = rust_dict.get("elements")
    if elements is None and atom_names is not None:
      elements = [name[0].upper() if name else "C" for name in atom_names]

    return cls(
      coordinates=convert(raw_coords, dtype=np.float32).reshape(-1, 3),
      full_coordinates=convert(raw_coords, dtype=np.float32).reshape(-1, 3),
      aatype=convert(rust_dict["aatype"], dtype=np.int8),
      one_hot_sequence=convert(np.eye(21)[rust_dict["aatype"]]),
      mask=convert(np.ones(num_residues), dtype=np.float32),
      residue_index=convert(rust_dict["residue_index"], dtype=np.int32),
      chain_index=convert(rust_dict["chain_index"], dtype=np.int32),
      atom_mask=convert(raw_mask, dtype=np.float32),
      elements=elements,
      atom_names=atom_names,
      charges=convert(rust_dict["charges"]) if rust_dict.get("charges") is not None else None,
      radii=convert_params(rust_dict.get("radii") or rust_dict.get("gbsa_radii"), "length"),
      sigmas=convert_params(rust_dict.get("sigmas"), "length"),
      epsilons=convert_params(rust_dict.get("epsilons"), "energy"),
      molecule_type=convert(rust_dict["molecule_type"])
      if rust_dict.get("molecule_type") is not None
      else None,
      atom_types=rust_dict.get("atom_types"),
      bonds=convert(rust_dict["bonds"]) if rust_dict.get("bonds") is not None else None,
      bond_params=convert_params(rust_dict.get("bond_params"), "bond"),
      angles=convert(rust_dict["angles"]) if rust_dict.get("angles") is not None else None,
      angle_params=convert_params(rust_dict.get("angle_params"), "angle"),
      proper_dihedrals=convert(rust_dict["dihedrals"])
      if rust_dict.get("dihedrals") is not None
      else None,
      dihedral_params=convert_params(rust_dict.get("dihedral_params"), "dihedral"),
      impropers=convert(rust_dict["impropers"]) if rust_dict.get("impropers") is not None else None,
      improper_params=convert_params(rust_dict.get("improper_params"), "improper"),
      vdw_features=convert(rust_dict["vdw_features"])
      if rust_dict.get("vdw_features") is not None
      else None,
      rbf_features=convert(rust_dict["rbf_features"])
      if rust_dict.get("rbf_features") is not None
      else None,
      electrostatic_features=convert(rust_dict["electrostatic_features"])
      if rust_dict.get("electrostatic_features") is not None
      else None,
      format="Full",
      source=source,
      masses=convert(rust_dict["masses"]) if rust_dict.get("masses") is not None else None,
    )


# Keep AtomicSystem import for backward compat type hints

ProteinStream = Generator[Protein, None, None]
"""Generator yielding Protein instances."""

ProteinBatch = Sequence[Protein]
"""Sequence of Protein instances."""

OligomerType = Literal["monomer", "heteromer", "homooligomer", "tied_homooligomer"]
