"""Python wrapper for Rust parsing extension.

This module provides a high-level interface to the oxidize Rust extension,
handling data conversion and maintaining API compatibility with existing parsers.
"""

import numpy as np
from pathlib import Path
from typing import Any
from dataclasses import dataclass

import oxidize

from proxide.core.containers import Protein, ProteinStream
from proxide.io.parsing.registry import register_parser, ParsingError


@register_parser(["pdb", "cif", "mmcif", "rust"])
def load_rust(
  file_path: str | Path,
  chain_id: str | list[str] | None = None,
  *,
  extract_dihedrals: bool = False,
  populate_physics: bool = False,
  force_field_name: str | None = None,
  add_hydrogens: bool = True,
  infer_bonds: bool = False,
  **kwargs: Any,
) -> ProteinStream:
  """Load a protein structure using the Rust extension.
  
  Args:
      file_path: Path to the structure file.
      chain_id: Unused for now in Rust parser (filtering happens post-parse if needed).
      extract_dihedrals: Whether to compute dihedrals (happens automatically in Protein).
      populate_physics: Whether to parameterize MD (requires force_field_name).
      force_field_name: Name/path of force field if populate_physics is True.
      add_hydrogens: Whether to add hydrogens.
      infer_bonds: Whether to infer connectivity.
      **kwargs: Additional args passed to OutputSpec.
      
  Yields:
      Protein instances.
  """
  # Construct OutputSpec
  spec = OutputSpec()
  spec.add_hydrogens = add_hydrogens
  spec.infer_bonds = infer_bonds
  # If populate_physics is requested, we need a force field
  if populate_physics and force_field_name:
      spec.parameterize_md = True
      spec.force_field = force_field_name
  elif populate_physics:
      # If physics requested but no FF, we assume we might error or just skip?
      pass
      
  # Handle other kwargs
  if "remove_solvent" in kwargs:
      spec.remove_solvent = kwargs["remove_solvent"]
      
  try:
    if hasattr(file_path, "read"):
        # Handle file-like objects (StringIO, etc)
        import tempfile
        import os
        
        # Read content
        content = file_path.read()
        if hasattr(content, "encode"):
             # String content - assume PDB (default) unless detectable?
             # For StringIO, we don't know suffix. Default to .pdb
             suffix = ".pdb"
             mode = "w"
        else:
             # Bytes content
             suffix = ".pdb" # Fallback
             mode = "wb"
             
        with tempfile.NamedTemporaryFile(mode=mode, suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
            
        try:
            result_dict = oxidize.parse_structure(tmp_path, spec)
            protein = Protein.from_rust_dict(result_dict, source="<stream>")
            yield protein
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    else:
        path_str = str(file_path)
        result_dict = oxidize.parse_structure(path_str, spec)
        protein = Protein.from_rust_dict(result_dict, source=path_str)
        yield protein
    
  except Exception as e:
    raise ParsingError(f"Rust parsing failed for {file_path}: {e}") from e



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
      ValueError: If parsing fails
  """
  result = oxidize.parse_structure(str(file_path), spec)
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

      >>> from proxide.io import rust_wrapper
      >>> protein = rust_wrapper.parse_structure("1crn.pdb")

      **Getting GAFF Atom Types:**

      >>> from proxide.io import rust_wrapper
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


# Export Rust types
from oxidize import OutputSpec, CoordFormat, ErrorMode, MissingResidueMode

# Optional XTC support
parse_xtc = getattr(oxidize, "parse_xtc", None)


def parse_pdb_raw_rust(file_path: str | Path) -> RawAtomData:
  """Parse a PDB file and return raw atom data (low-level).

  This is useful for custom processing pipelines that need access
  to the raw atom data before formatting.

  Args:
      file_path: Path to PDB file

  Returns:
      RawAtomData with parsed atom information

  Raises:
      ValueError: If parsing fails
  """
  result = oxidize.parse_pdb(str(file_path))

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
      ValueError: If parsing fails
  """
  result = oxidize.parse_mmcif(str(file_path))

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
      ValueError: If parsing fails

  Example:
      >>> ff = load_forcefield_rust("protein.ff19SB.xml")
      >>> print(f"Loaded {ff.num_atom_types} atom types")
      >>> ala = ff.get_residue("ALA")
      >>> print(f"ALA has {len(ala['atoms'])} atoms")
  """
  result = oxidize.load_forcefield(str(file_path))

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
      ImportError: If trajectory feature not available
      ValueError: If parsing fails
  """
  if parse_xtc is None:
    raise ImportError("parse_xtc not found in oxidize. Ensure 'trajectories' feature is enabled.")

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
      ImportError: If mdcath feature not available
      ValueError: If parsing fails
  """
  if not hasattr(oxidize, "parse_mdtraj_h5_metadata"):
    raise ImportError("HDF5 support not available. Rebuild with: maturin develop --features mdcath")

  result = oxidize.parse_mdtraj_h5_metadata(str(file_path))

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
      ImportError: If mdcath feature not available
      ValueError: If parsing fails
  """
  # Get metadata for atom info
  metadata = parse_mdtraj_h5_metadata(file_path)

  # Get frame coordinates
  frame_result = oxidize.parse_mdtraj_h5_frame(str(file_path), frame_idx)

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
      ImportError: If mdcath feature not available
      ValueError: If parsing fails
  """
  if not hasattr(oxidize, "parse_mdcath_metadata"):
    raise ImportError("HDF5 support not available. Rebuild with: maturin develop --features mdcath")

  result = oxidize.parse_mdcath_metadata(str(file_path))

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
  return oxidize.get_mdcath_replicas(str(file_path), domain_id, temperature)


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
      ImportError: If mdcath feature not available
      ValueError: If parsing fails
  """
  return oxidize.parse_mdcath_frame(str(file_path), domain_id, temperature, replica, frame_idx)


def is_hdf5_support_available() -> bool:
  """Check if HDF5 parsing support is available.

  Returns:
      True if mdcath feature was compiled, False otherwise.
  """
  # Check if the function exists
  if not hasattr(oxidize, "parse_mdtraj_h5_metadata"):
    return False

  # Try to call the function - it raises ImportError if feature not enabled
  try:
    oxidize.parse_mdtraj_h5_metadata("/nonexistent")
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
  """Check if Rust parser is available.
  
  Always returns True since oxidize is now a hard dependency.
  """
  return True


def get_rust_capabilities() -> dict[str, bool]:
  """Get dictionary of available Rust capabilities.

  Returns:
      Dictionary mapping capability names to availability status.
  """
  return {
    "parse_pdb": hasattr(oxidize, "parse_pdb"),
    "parse_mmcif": hasattr(oxidize, "parse_mmcif"),
    "parse_structure": hasattr(oxidize, "parse_structure"),
    "load_forcefield": hasattr(oxidize, "load_forcefield"),
    "parse_xtc": hasattr(oxidize, "parse_xtc"),
    "parse_mdtraj_h5": hasattr(oxidize, "parse_mdtraj_h5_metadata"),
    "parse_mdcath": hasattr(oxidize, "parse_mdcath_metadata"),
    "atomic_system_types": hasattr(oxidize, "AtomicSystem"),
  }
