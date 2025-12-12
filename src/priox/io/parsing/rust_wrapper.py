"""Python wrapper for Rust parsing extension.

This module provides a high-level interface to the priox_rs Rust extension,
handling data conversion and maintaining API compatibility with existing parsers.
"""

import numpy as np
from pathlib import Path
from typing import Any
from dataclasses import dataclass

try:
  import priox_rs

  RUST_AVAILABLE = True
except ImportError:
  RUST_AVAILABLE = False
  priox_rs = None

from priox.core.containers import Protein


# =============================================================================
# Data Classes for Rust Results
# =============================================================================


@dataclass
class RawAtomData:
  """Raw atom data from low-level parsers (PDB/mmCIF).

  This matches the dictionary returned by parse_pdb and parse_mmcif.
  """

  num_atoms: int
  atom_names: list[str]
  res_names: list[str]
  res_ids: np.ndarray
  chain_ids: list[str]
  coords: np.ndarray  # (N, 3)
  elements: list[str]
  occupancies: np.ndarray
  b_factors: np.ndarray


@dataclass
class ForceFieldData:
  """Force field data loaded from OpenMM-style XML files.

  Contains atom types, residue templates, bond/angle/dihedral parameters,
  and optional CMAP and GBSA data.
  """

  name: str
  num_atom_types: int
  num_residue_templates: int
  num_harmonic_bonds: int
  num_harmonic_angles: int
  num_proper_torsions: int
  num_improper_torsions: int
  num_nonbonded_params: int
  num_gbsa_obc_params: int
  has_cmap: bool
  atom_types: list[dict]
  residue_templates: list[dict]
  harmonic_bonds: list[dict]
  harmonic_angles: list[dict]
  proper_torsions: list[dict]
  improper_torsions: list[dict]
  nonbonded_params: list[dict]
  gbsa_obc_params: list[dict]
  cmap_maps: list[dict] | None = None
  cmap_torsions: list[dict] | None = None

  def get_residue(self, name: str) -> dict | None:
    """Get residue template by name."""
    for template in self.residue_templates:
      if template.get("name") == name:
        return template
    return None

  def get_atom_type(self, name: str) -> dict | None:
    """Get atom type by name."""
    for at in self.atom_types:
      if at.get("name") == name:
        return at
    return None


@dataclass
class MdtrajH5Data:
  """MDTraj HDF5 file metadata.

  Contains trajectory metadata from MDTraj-format HDF5 files.
  """

  num_frames: int
  num_atoms: int
  atom_names: list[str]
  res_names: list[str]
  res_ids: np.ndarray
  chain_ids: list[str]
  elements: list[str]


@dataclass
class MdcathData:
  """MDCATH HDF5 file metadata.

  Contains domain metadata from mdCATH-format HDF5 files.
  """

  domain_id: str
  num_residues: int
  resnames: list[str]
  chain_ids: list[str]
  temperatures: list[str]


# =============================================================================
# Parser Functions
# =============================================================================


def parse_pdb_to_protein(file_path: str | Path, spec=None, use_jax: bool = True) -> Protein:
  """Parse a PDB file and return a Protein directly.

  This is the preferred method for parsing PDB files, as it returns a Protein
  object directly without creating an intermediate ProteinTuple.

  Args:
      file_path: Path to PDB file
      spec: Optional OutputSpec for controlling formatting (default: Atom37)
      use_jax: If True, convert arrays to JAX arrays. If False, use NumPy.

  Returns:
      Protein with parsed structure data

  Raises:
      ImportError: If Rust extension not available
      ValueError: If parsing fails
  """
  if not RUST_AVAILABLE:
    raise ImportError("priox_rs Rust extension not available. Install with maturin.")

  result = priox_rs.parse_structure(str(file_path), spec)
  return Protein.from_rust_dict(result, source=str(file_path), use_jax=use_jax)


def parse_structure(file_path: str | Path, spec=None, use_jax: bool = True) -> Protein:
  """Generic entry point for parsing structures using the Rust backend.

  This function uses the high-performance Rust extension to parse PDB/mmCIF files
  and optionally compute topology, force field parameters, and atom types.

  Args:
      file_path: Path to the structure file.
      spec: Optional OutputSpec configuration object. If None, defaults are used.
            The OutputSpec controls various processing steps:

            - **force_field** (str): Path to force field XML or "gaff" for GAFF atom typing.
            - **parameterize_md** (bool): If True, assign charges/radii from force field.
            - **infer_bonds** (bool): If True, infer connectivity (needed for GAFF).
            - **add_hydrogens** (bool): If True, add missing hydrogens.
            - **compute_electrostatics** (bool): Compute electrostatic features.
            - **include_hetatm** (bool): Include non-protein atoms.
            - **remove_solvent** (bool): Remove water molecules.

      use_jax: If True, returns JAX arrays. If False, returns NumPy arrays.

  Returns:
      Protein: A dataclass containing coordinates, topology, and optional features.

  Examples:
      **Basic Loading:**

      >>> from priox.io import rust_wrapper
      >>> protein = rust_wrapper.parse_structure("1crn.pdb")

      **Getting GAFF Atom Types:**

      >>> from priox.io import rust_wrapper
      >>> # Create spec requesting GAFF types and bond inference
      >>> spec = rust_wrapper.OutputSpec()
      >>> spec.force_field = "gaff"
      >>> spec.infer_bonds = True  # Required for aromaticity detection
      >>>
      >>> system = rust_wrapper.parse_structure("ligand.pdb", spec=spec)
      >>> print(system.atom_types)
      ['c3', 'hc', 'n', ...]

      **Full MD Parameterization:**

      >>> spec.parameterize_md = True
      >>> spec.force_field = "openff-2.0.0.xml"
      >>> system = rust_wrapper.parse_structure("complex.pdb", spec=spec)
      >>> print(system.charges)
  """
  return parse_pdb_to_protein(file_path, spec, use_jax)


# Export Rust types if available
if RUST_AVAILABLE:
  from priox_rs import OutputSpec, CoordFormat, ErrorMode, MissingResidueMode

  # AtomicSystem is now in Python (priox.core.atomic_system)
  # Molecule and Complex classes have been removed

  # Optional XTC support
  parse_xtc = getattr(priox_rs, "parse_xtc", None)
else:
  OutputSpec = None
  CoordFormat = None
  ErrorMode = None
  MissingResidueMode = None
  parse_xtc = None


def parse_pdb_raw_rust(file_path: str | Path) -> RawAtomData:
  """Parse a PDB file and return raw atom data (low-level).

  This is useful for custom processing pipelines that need access
  to the raw atom data before formatting.

  Args:
      file_path: Path to PDB file

  Returns:
      RawAtomData with parsed atom information

  Raises:
      ImportError: If Rust extension not available
      ValueError: If parsing fails
  """
  if not RUST_AVAILABLE:
    raise ImportError("priox_rs Rust extension not available. Install with maturin.")

  result = priox_rs.parse_pdb(str(file_path))

  return RawAtomData(
    num_atoms=result["num_atoms"],
    atom_names=result["atom_names"],
    res_names=result["res_names"],
    res_ids=result["res_ids"],
    chain_ids=result["chain_ids"],
    coords=result["coords"].reshape(-1, 3),
    elements=result["elements"],
    occupancies=result["occupancy"],
    b_factors=result["b_factors"],
  )


def parse_mmcif_rust(file_path: str | Path) -> RawAtomData:
  """Parse an mmCIF file and return raw atom data.

  Args:
      file_path: Path to mmCIF (.cif) file

  Returns:
      RawAtomData with parsed atom information

  Raises:
      ImportError: If Rust extension not available
      ValueError: If parsing fails
  """
  if not RUST_AVAILABLE:
    raise ImportError("priox_rs Rust extension not available. Install with maturin.")

  result = priox_rs.parse_mmcif(str(file_path))

  return RawAtomData(
    num_atoms=result["num_atoms"],
    atom_names=result["atom_names"],
    res_names=result["res_names"],
    res_ids=result["res_ids"],
    chain_ids=result["chain_ids"],
    coords=result["coords"].reshape(-1, 3),
    elements=result["elements"],
    occupancies=result["occupancy"],
    b_factors=result["b_factors"],
  )


def load_forcefield_rust(file_path: str | Path) -> ForceFieldData:
  """Load a force field from an OpenMM-style XML file.

  Args:
      file_path: Path to force field XML file

  Returns:
      ForceFieldData with parsed force field parameters

  Raises:
      ImportError: If Rust extension not available
      ValueError: If parsing fails

  Example:
      >>> ff = load_forcefield_rust("protein.ff19SB.xml")
      >>> print(f"Loaded {ff.num_atom_types} atom types")
      >>> ala = ff.get_residue("ALA")
      >>> print(f"ALA has {len(ala['atoms'])} atoms")
  """
  if not RUST_AVAILABLE:
    raise ImportError("priox_rs Rust extension not available. Install with maturin.")

  result = priox_rs.load_forcefield(str(file_path))

  return ForceFieldData(
    name=result.get("name", ""),
    num_atom_types=result["num_atom_types"],
    num_residue_templates=result["num_residue_templates"],
    num_harmonic_bonds=result["num_harmonic_bonds"],
    num_harmonic_angles=result["num_harmonic_angles"],
    num_proper_torsions=result["num_proper_torsions"],
    num_improper_torsions=result["num_improper_torsions"],
    num_nonbonded_params=result["num_nonbonded_params"],
    num_gbsa_obc_params=result["num_gbsa_obc_params"],
    has_cmap=result["has_cmap"],
    atom_types=result["atom_types"],
    residue_templates=result["residue_templates"],
    harmonic_bonds=result["harmonic_bonds"],
    harmonic_angles=result["harmonic_angles"],
    proper_torsions=result["proper_torsions"],
    improper_torsions=result["improper_torsions"],
    nonbonded_params=result["nonbonded_params"],
    gbsa_obc_params=result.get("gbsa_obc_params", []),
    cmap_maps=result.get("cmap_maps"),
    cmap_torsions=result.get("cmap_torsions"),
  )


def parse_xtc_rust(file_path: str | Path) -> dict[str, Any]:
  """Parse an XTC trajectory file using the Rust extension.

  Args:
      file_path: Path to XTC file

  Returns:
      Dictionary with 'coordinates', 'times', 'num_frames', 'num_atoms'

  Raises:
      ImportError: If Rust extension or trajectory feature not available
      ValueError: If parsing fails
  """
  if not RUST_AVAILABLE:
    raise ImportError("priox_rs Rust extension not available.")

  if parse_xtc is None:
    raise ImportError("parse_xtc not found in priox_rs. Ensure 'trajectories' feature is enabled.")

  return parse_xtc(str(file_path))


# =============================================================================
# HDF5 Parser Functions
# =============================================================================


def parse_mdtraj_h5_metadata(file_path: str | Path) -> MdtrajH5Data:
  """Parse MDTraj HDF5 file and return metadata.

  Args:
      file_path: Path to MDTraj HDF5 file

  Returns:
      MdtrajH5Data with trajectory metadata

  Raises:
      ImportError: If Rust extension or mdcath feature not available
      ValueError: If parsing fails
  """
  if not RUST_AVAILABLE:
    raise ImportError("priox_rs Rust extension not available.")

  if not hasattr(priox_rs, "parse_mdtraj_h5_metadata"):
    raise ImportError("HDF5 support not available. Rebuild with: maturin develop --features mdcath")

  result = priox_rs.parse_mdtraj_h5_metadata(str(file_path))

  return MdtrajH5Data(
    num_frames=result["num_frames"],
    num_atoms=result["num_atoms"],
    atom_names=result["atom_names"],
    res_names=result["res_names"],
    res_ids=np.array(result["res_ids"]),
    chain_ids=result["chain_ids"],
    elements=result["elements"],
  )


def parse_mdtraj_h5_frame(file_path: str | Path, frame_idx: int = 0) -> RawAtomData:
  """Parse a single frame from MDTraj HDF5 file.

  Args:
      file_path: Path to MDTraj HDF5 file
      frame_idx: Frame index to parse (default: 0)

  Returns:
      RawAtomData with frame coordinates and metadata

  Raises:
      ImportError: If Rust extension or mdcath feature not available
      ValueError: If parsing fails
  """
  if not RUST_AVAILABLE:
    raise ImportError("priox_rs Rust extension not available.")

  # Get metadata for atom info
  metadata = parse_mdtraj_h5_metadata(file_path)

  # Get frame coordinates
  frame_result = priox_rs.parse_mdtraj_h5_frame(str(file_path), frame_idx)

  return RawAtomData(
    num_atoms=metadata.num_atoms,
    atom_names=metadata.atom_names,
    res_names=metadata.res_names,
    res_ids=metadata.res_ids,
    chain_ids=metadata.chain_ids,
    coords=frame_result["coords"],
    elements=metadata.elements,
    occupancies=np.ones(metadata.num_atoms, dtype=np.float32),
    b_factors=np.zeros(metadata.num_atoms, dtype=np.float32),
  )


def parse_mdcath_metadata(file_path: str | Path) -> MdcathData:
  """Parse MDCATH HDF5 file and return domain metadata.

  Args:
      file_path: Path to MDCATH HDF5 file

  Returns:
      MdcathData with domain metadata

  Raises:
      ImportError: If Rust extension or mdcath feature not available
      ValueError: If parsing fails
  """
  if not RUST_AVAILABLE:
    raise ImportError("priox_rs Rust extension not available.")

  if not hasattr(priox_rs, "parse_mdcath_metadata"):
    raise ImportError("HDF5 support not available. Rebuild with: maturin develop --features mdcath")

  result = priox_rs.parse_mdcath_metadata(str(file_path))

  return MdcathData(
    domain_id=result["domain_id"],
    num_residues=result["num_residues"],
    resnames=result["resnames"],
    chain_ids=result["chain_ids"],
    temperatures=result["temperatures"],
  )


def get_mdcath_replicas(file_path: str | Path, domain_id: str, temperature: str) -> list[str]:
  """Get list of replicas for a temperature in MDCATH file.

  Args:
      file_path: Path to MDCATH HDF5 file
      domain_id: Domain identifier
      temperature: Temperature key (e.g., "320")

  Returns:
      List of replica identifiers
  """
  if not RUST_AVAILABLE:
    raise ImportError("priox_rs Rust extension not available.")

  return priox_rs.get_mdcath_replicas(str(file_path), domain_id, temperature)


def parse_mdcath_frame(
  file_path: str | Path,
  domain_id: str,
  temperature: str,
  replica: str,
  frame_idx: int = 0,
) -> dict[str, Any]:
  """Parse a single frame from MDCATH HDF5 file.

  Args:
      file_path: Path to MDCATH HDF5 file
      domain_id: Domain identifier
      temperature: Temperature key (e.g., "320")
      replica: Replica identifier
      frame_idx: Frame index to parse (default: 0)

  Returns:
      Dictionary with 'temperature', 'replica', 'frame_idx', 'coords'

  Raises:
      ImportError: If Rust extension or mdcath feature not available
      ValueError: If parsing fails
  """
  if not RUST_AVAILABLE:
    raise ImportError("priox_rs Rust extension not available.")

  return priox_rs.parse_mdcath_frame(str(file_path), domain_id, temperature, replica, frame_idx)


def is_hdf5_support_available() -> bool:
  """Check if HDF5 parsing support is available.

  Returns:
      True if mdcath feature was compiled, False otherwise.
  """
  if not RUST_AVAILABLE:
    return False

  # Check if the function exists
  if not hasattr(priox_rs, "parse_mdtraj_h5_metadata"):
    return False

  # Try to call the function - it raises ImportError if feature not enabled
  try:
    priox_rs.parse_mdtraj_h5_metadata("/nonexistent")
  except ImportError:
    return False
  except ValueError:
    # ValueError means the function is available but file doesn't exist
    return True
  return True


# =============================================================================
# Utility Functions
# =============================================================================


def is_rust_parser_available() -> bool:
  """Check if Rust parser is available."""
  return RUST_AVAILABLE


def get_rust_capabilities() -> dict[str, bool]:
  """Get dictionary of available Rust capabilities.

  Returns:
      Dictionary mapping capability names to availability status.
  """
  if not RUST_AVAILABLE:
    return {
      "parse_pdb": False,
      "parse_mmcif": False,
      "parse_structure": False,
      "load_forcefield": False,
      "parse_xtc": False,
      "parse_mdtraj_h5": False,
      "parse_mdcath": False,
    }

  return {
    "parse_pdb": hasattr(priox_rs, "parse_pdb"),
    "parse_mmcif": hasattr(priox_rs, "parse_mmcif"),
    "parse_structure": hasattr(priox_rs, "parse_structure"),
    "load_forcefield": hasattr(priox_rs, "load_forcefield"),
    "parse_xtc": hasattr(priox_rs, "parse_xtc"),
    "parse_mdtraj_h5": hasattr(priox_rs, "parse_mdtraj_h5_metadata"),
    "parse_mdcath": hasattr(priox_rs, "parse_mdcath_metadata"),
    "atomic_system_types": hasattr(priox_rs, "AtomicSystem"),
  }
