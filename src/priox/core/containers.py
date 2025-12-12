"""Dataclasses for the PrxteinMPNN project.

prxteinmpnn.utils.data_structures
"""

from __future__ import annotations

from collections.abc import Generator, Sequence
from typing import TYPE_CHECKING, Literal

import jax.numpy as jnp
import numpy as np
from flax.struct import dataclass

from priox.chem.residues import atom_order

if TYPE_CHECKING:
  from jaxtyping import Int

  from priox.core.types import (
    AlphaCarbonMask,
    AtomMask,
    BackboneDihedrals,
    ChainIndex,
    OneHotProteinSequence,
    ProteinSequence,
    ResidueIndex,
    StructureAtomicCoordinates,
  )

from priox.core.atomic_system import AtomicSystem


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
  """Convert a numpy array to jnp array, or return None if input is None.

  Args:
      array (np.ndarray | None): Input numpy array or None.

  Returns:
      jnp.ndarray | None: Converted jnp array or None.

  """
  if array is None:
    return None
  return jnp.asarray(array)


def none_or_numpy(array: np.ndarray | None) -> np.ndarray | None:
  """Convert to numpy array, or return None if input is None.

  Args:
      array (np.ndarray | None): Input array or None.

  Returns:
      np.ndarray | None: Converted numpy array or None.

  """
  if array is None:
    return None
  return np.asarray(array)


@dataclass(frozen=True, kw_only=True)
class Protein(AtomicSystem):
  """Protein structure or ensemble representation.

  Attributes:
    coordinates (StructureAtomicCoordinates): Atom positions in the structure, represented as a
      3D array. Cartesian coordinates of atoms in angstroms. The atom types correspond to
      residue_constants.atom_types, i.e. the first three are N, CA, CB. Shape is
      (num_res, num_atom_type, 3), where num_res is the number of residues, num_atom_type is the
      number of atom types (e.g., N, CA, CB, C, O), and 3 is the spatial dimension (x, y, z).
    aatype (Sequence): Amino-acid type for each residue represented as an integer between 0 and 20,
      where 20 is 'X'. Shape is [num_res].
    mask (AlphaCarbonMask): Binary float mask to indicate presence of alpha carbon atom.
      1.0 if an atom is present and 0.0 if not. This should be used for loss masking.
      Shape is [num_res, num_atom_type].
    residue_index (AtomResidueIndex): Residue index as used in PDB. It is not necessarily
      continuous or 0-indexed. Shape is [num_res].
    chain_index (ChainIndex): Chain index for each residue. Shape is [num_res].
    dihedrals (BackboneDihedrals | None): Dihedral angles for backbone atoms (phi, psi, omega).
      Shape is [num_res, 3]. If not provided, defaults to None.
    mapping (jnp.Array | None): Optional array mapping residues in the ensemble to original
      structure indices. Shape is [num_res, num_frames]. If not provided, defaults to None.
    full_coordinates (StructureAtomicCoordinates | None): Full atomic coordinates
      including all heavy atoms. Shape is (num_res, num_full_atom_type, 3), where num_full_atom_type
      is the number of all heavy atom types (e.g., N, CA, CB, C, O, CG, etc.), and 3 is the spatial
      dimension (x, y, z). If not provided, defaults to None.
    full_atom_mask (AtomMask | None): Binary float mask to indicate presence of a particular
      heavy atom. 1.0 if an atom is present and 0.0 if not. This should be used for loss masking.
      Shape is [num_res, num_full_atom_type]. If not provided, defaults to None.

  """

  coordinates: StructureAtomicCoordinates
  aatype: ProteinSequence
  one_hot_sequence: OneHotProteinSequence
  mask: AlphaCarbonMask
  residue_index: ResidueIndex
  chain_index: ChainIndex
  dihedrals: BackboneDihedrals | None = None
  mapping: Int | None = None
  full_coordinates: StructureAtomicCoordinates | None = None
  full_atom_mask: AtomMask | None = None

  # Derived features
  physics_features: jnp.ndarray | None = None
  backbone_indices: jnp.ndarray | None = None

  @classmethod
  def from_rust_dict(
    cls,
    rust_dict: dict,
    source: str | None = None,
    use_jax: bool = True,
  ) -> Protein:
    """Create a Protein instance directly from Rust parser output dictionary.

    This is the preferred method when using the Rust parser, as it avoids
    creating an intermediate ProteinTuple.

    Args:
        rust_dict: Dictionary returned by priox_rs.parse_structure()
        source: Optional source file path for metadata
        use_jax: If True, convert arrays to JAX arrays. If False, use NumPy.

    Returns:
        Protein: The protein dataclass instance.

    """
    num_residues = len(rust_dict["aatype"])
    
    raw_coords = rust_dict["coordinates"]
    raw_mask = rust_dict["atom_mask"]
    
    # Check if we can/should reshape to Atom37 (N, 37, 3)
    # or if we have a flat structure (N_atoms, 3)
    is_atom37 = (raw_coords.ndim == 3 and raw_coords.shape[1] == 37) or \
                (raw_coords.ndim == 2 and raw_coords.size == num_residues * 37 * 3)
                
    if use_jax:
      convert = jnp.asarray
    else:
      convert = np.asarray

    if is_atom37:
      # Reshape coordinates and atom_mask to (N, 37, ...)
      coordinates = raw_coords.reshape(num_residues, 37, 3)
      atom_mask_2d = raw_mask.reshape(num_residues, 37)
      
      # For Protein dataclass (AF2 style), mask is CA mask from the 37 grid
      # But we also need full atom_mask for AtomicSystem
      mask_ca = atom_mask_2d[:, atom_order["CA"]]
      
      return cls(
        coordinates=convert(coordinates, dtype=np.float32),
        aatype=convert(rust_dict["aatype"], dtype=np.int8),
        one_hot_sequence=convert(np.eye(21)[rust_dict["aatype"]]),
        mask=convert(mask_ca, dtype=np.float32),
        residue_index=convert(rust_dict["residue_index"], dtype=np.int32),
        chain_index=convert(rust_dict["chain_index"], dtype=np.int32),
        # AtomicSystem required fields
        atom_mask=convert(atom_mask_2d, dtype=np.float32),
        elements=None,
        atom_names=None,
        # Optional fields from Rust
        charges=convert(rust_dict["charges"]) if rust_dict.get("charges") is not None else None,
        radii=convert(rust_dict["radii"]) if rust_dict.get("radii") is not None else None,
        sigmas=convert(rust_dict["sigmas"]) if rust_dict.get("sigmas") is not None else None,
        epsilons=convert(rust_dict["epsilons"]) if rust_dict.get("epsilons") is not None else None,
        molecule_type=convert(rust_dict["molecule_type"])
        if rust_dict.get("molecule_type") is not None
        else None,
        # Topology / GAFF
        atom_types=rust_dict.get("atom_types"),
        bonds=convert(rust_dict["bonds"]) if rust_dict.get("bonds") is not None else None,
        angles=convert(rust_dict["angles"]) if rust_dict.get("angles") is not None else None,
        proper_dihedrals=convert(rust_dict["dihedrals"])
        if rust_dict.get("dihedrals") is not None
        else None,
        impropers=convert(rust_dict["impropers"]) if rust_dict.get("impropers") is not None else None,
      )
    else:
      # Flat format (Full)
      # In this case, Protein fields like 'mask' (CA mask) need to be derived differently 
      # or populated with dummies if we are treating this as a general AtomicSystem
      
      # We attempt to treat it as "Full Coordinates" stored in the main coordinates field
      # This technically violates the (N, 37, 3) type hint of Protein, but matches AtomicSystem
      
      # For CA mask: we don't have CA info easily unless we assume input is CA-only (unlikely)
      # or check atom names if provided.
      # For now, we set mask to ones (all residues present)
      mask_dummy = np.ones((num_residues, 37), dtype=np.float32) # Dummy
      
      # Full format elements/atom_names handling
      atom_names = rust_dict.get("atom_names")
      elements = rust_dict.get("elements")
      
      if elements is None and atom_names is not None:
          # Infer elements from atom names if missing
          elements = [name[0].upper() if name else "C" for name in atom_names]
      
      return cls(
        coordinates=convert(raw_coords, dtype=np.float32).reshape(-1, 3), # (N_padded, 3)
        aatype=convert(rust_dict["aatype"], dtype=np.int8),
        one_hot_sequence=convert(np.eye(21)[rust_dict["aatype"]]),
        mask=convert(np.ones(num_residues), dtype=np.float32), # Residue mask
        residue_index=convert(rust_dict["residue_index"], dtype=np.int32),
        chain_index=convert(rust_dict["chain_index"], dtype=np.int32),
        # AtomicSystem
        atom_mask=convert(raw_mask, dtype=np.float32), # Flat mask
        
        elements=elements,
        atom_names=atom_names,
        
        charges=convert(rust_dict["charges"]) if rust_dict.get("charges") is not None else None,
        radii=convert(rust_dict["radii"]) if rust_dict.get("radii") is not None else None,
        sigmas=convert(rust_dict["sigmas"]) if rust_dict.get("sigmas") is not None else None,
        epsilons=convert(rust_dict["epsilons"]) if rust_dict.get("epsilons") is not None else None,
        molecule_type=convert(rust_dict["molecule_type"])
        if rust_dict.get("molecule_type") is not None
        else None,
        
        atom_types=rust_dict.get("atom_types"),
        bonds=convert(rust_dict["bonds"]) if rust_dict.get("bonds") is not None else None,
        angles=convert(rust_dict["angles"]) if rust_dict.get("angles") is not None else None,
        proper_dihedrals=convert(rust_dict["dihedrals"])
        if rust_dict.get("dihedrals") is not None
        else None,
        impropers=convert(rust_dict["impropers"]) if rust_dict.get("impropers") is not None else None,
      )


ProteinStream = Generator[Protein, None]
"""Generator yielding Protein instances."""

ProteinBatch = Sequence[Protein]
"""Sequence of Protein instances."""

OligomerType = Literal["monomer", "heteromer", "homooligomer", "tied_homooligomer"]
