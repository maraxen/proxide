"""Python wrapper for Rust parsing extension.

This module provides a high-level interface to the _proxider Rust extension,
handling data conversion and maintaining API compatibility with existing parsers.
"""

import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, cast

import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike

from proxide import _proxider  # type: ignore[unresolved-import]
from proxide._proxider import OutputSpec  # type: ignore[unresolved-import]
from proxide.core.atomic_system import (
  AtomicConstants,
  AtomicState,
  AtomicSystem,
  MolecularTopology,
)
from proxide.core.containers import Protein
from proxide.io.parsing.registry import ParsingError, register_parser

FLAT_COORD_NDIM = 2
ATOM37_DIM = 3


def _convert_rust_dict_to_system(data: dict) -> AtomicSystem:
  """Convert rust dict to AtomicSystem (flat)."""
  # Similar to Protein.from_rust_dict but creating flat AtomicSystem

  coords = data["coordinates"]
  if coords.ndim == ATOM37_DIM:
    coords = coords.reshape(-1, 3)

  mask = data["atom_mask"]
  if mask.ndim > 1:
    mask = mask.flatten()

  atom_names = data.get("atom_names")

  return AtomicSystem(
    topology=MolecularTopology(
      elements=data.get("elements"),
      atom_names=atom_names,
      molecule_type=jnp.array(data["molecule_type"])
      if data.get("molecule_type") is not None
      else None,
      residue_index=jnp.array(data["residue_index"])
      if data.get("residue_index") is not None
      else None,
      chain_index=jnp.array(data["chain_index"]) if data.get("chain_index") is not None else None,
      bonds=jnp.array(data["bonds"]) if data.get("bonds") is not None else None,
    ),
    state=AtomicState(
      coordinates=jnp.array(coords),
      box_vectors=jnp.array(data["box_vectors"]) if data.get("box_vectors") is not None else None,
    ),
    constants=AtomicConstants(
      charges=jnp.array(data["charges"]) if data.get("charges") is not None else None,
      radii=jnp.array(data["radii"]) if data.get("radii") is not None else None,
      sigmas=jnp.array(data["sigmas"]) if data.get("sigmas") is not None else None,
      epsilons=jnp.array(data["epsilons"]) if data.get("epsilons") is not None else None,
      masses=jnp.array(data["masses"]) if data.get("masses") is not None else None,
    )
    if any(k in data for k in ["charges", "radii", "sigmas", "epsilons", "masses"])
    else None,
    atom_mask=jnp.array(mask),
  )


@register_parser(["pdb", "cif", "mmcif", "rust"])
def load_rust(
  file_path: str | Path | IO[str],
  chain_id: str | list[str] | None = None,
  *,
  extract_dihedrals: bool = False,
  populate_physics: bool = False,
  force_field_name: str | None = None,
  add_hydrogens: bool = True,
  infer_bonds: bool = False,
  return_type: str = "Protein",
  output_format_target: str | None = None,
  **kwargs: Any,
) -> Any:
  """Load a protein structure using the Rust extension.

  This function serves as the primary entry point for the global `load_structure` dispatch,
  routing file loading to the appropriate Rust parser.

  Process:
      1.  **Preparation**: Resolve file path (handling file-like objects if needed).
      2.  **Configuration**: Build `OutputSpec` from arguments (hydrogens, MD params).
      3.  **Parsing**: call `_proxider.parse_structure` (Rust).
      4.  **Conversion**: Convert Rust output dict to `Protein` dataclass.
      5.  **Filtering**: Apply chain filtering if `chain_id` is specified.

  Args:
      file_path: Path to the structure file or file-like object.
      chain_id: Filter results to specific chain ID(s).
      extract_dihedrals: Whether to compute backbone dihedrals (default: False).
      populate_physics: Whether to parameterize for MD (requires `force_field_name`).
      force_field_name: Name/path of force field XML (e.g. "protein.ff14SB.xml").
      add_hydrogens: Whether to add missing geometric hydrogens (default: True).
      infer_bonds: Whether to infer bond connectivity (default: False).
      return_type: target class ("Protein" or "AtomicSystem").
      output_format_target: Target format hint ("mpnn" or "general").
      **kwargs: Additional args passed to `OutputSpec`.

  Yields:
      Protein or AtomicSystem instances (as a generator for compatibility).

  """

  spec = OutputSpec(
    add_hydrogens=add_hydrogens,
    infer_bonds=infer_bonds,
    parameterize_md=populate_physics and bool(force_field_name),
    force_field=force_field_name if populate_physics else None,
    output_format_target=output_format_target or "general",  # Rust requires 'general' or 'mpnn'
    remove_solvent=kwargs.get("remove_solvent", True),
  )

  try:
    # Create a usage context for the file path
    # If it's a file-like object, we write to temp file
    # If it's a path, we use it directly

    tmp_path: str | None = None

    try:
      if hasattr(file_path, "read"):
        content = cast(Any, file_path).read()
        if hasattr(content, "encode"):
          suffix = ".pdb"
          mode = "w"
        else:
          suffix = ".pdb"
          mode = "wb"

        with tempfile.NamedTemporaryFile(mode=mode, suffix=suffix, delete=False) as tmp:
          tmp.write(content)
          tmp_path = tmp.name

        path_str = tmp_path
      else:
        path_str = str(file_path)

      result_dict = _proxider.parse_structure(path_str, spec)
      obj = Protein.from_rust_dict(result_dict, source=path_str if tmp_path is None else "<stream>")

      if chain_id:
        target_chains = {chain_id} if isinstance(chain_id, str) else set(chain_id)

        if getattr(obj, "chain_ids", None) is not None:
          unique_ids = cast(Sequence[str], obj.chain_ids)
          allowed_indices = {i for i, cid in enumerate(unique_ids) if cid in target_chains}

          if allowed_indices:
            c_idx = np.array(obj.chain_index)
            mask = np.isin(c_idx, list(allowed_indices))

            if mask.sum() > 0:
              new_coords = obj.coordinates[mask]
              new_aatype = obj.aatype[mask]
              new_res_idx = obj.residue_index[mask] if obj.residue_index is not None else None
              new_chain_idx = obj.chain_index[mask] if obj.chain_index is not None else None
              new_mask = obj.mask[mask] if obj.mask is not None else None
              new_seq = obj.one_hot_sequence[mask] if obj.one_hot_sequence is not None else None

              # Handle full_coordinates / atom_mask slicing
              if new_coords.ndim == ATOM37_DIM:
                # Atom37 mode: coordinates are (N_residues, 37, 3)
                # full_coordinates are (N_residues * 37, 3)
                # We can assume full_coordinates corresponds to flattened coordinates
                new_full_coords = new_coords.reshape(-1, 3)

                # atom_mask (AtomicSystem field)
                if obj.atom_mask is not None and obj.atom_mask.ndim == 2:
                  # If stored as (N_residues, 37), we can slice by residue
                  new_atom_mask = obj.atom_mask[mask]
                else:
                  # If flat, we try to expand mask
                  if (
                    obj.atom_mask is not None
                    and obj.atom_mask.shape[0] == obj.coordinates.shape[0] * 37
                  ):
                    expanded_mask = np.repeat(mask, 37)
                    new_atom_mask = obj.atom_mask[expanded_mask]
                  else:
                    new_atom_mask = None
              else:
                # Flat mode (coords are N_atoms, 3)
                raw_full = (
                  obj.full_coordinates if obj.full_coordinates is not None else obj.coordinates
                )
                new_full_coords = raw_full[mask]
                new_atom_mask = obj.atom_mask[mask] if obj.atom_mask is not None else None

              # Physics params are tricky because they might be (N_atoms,) or (N_res,)
              # depending on storage AtomicSystem stores them as (N_atoms,) currently.
              # Protein doesn't explicitly slice them here (TODO).
              # For now we just filter the main residue-based fields.

              if new_atom_mask is None:
                # Fallback if mask slicing failed
                if new_coords.ndim == ATOM37_DIM:
                  new_atom_mask = jnp.ones((new_coords.shape[0], 37), dtype=jnp.float32)
                else:
                  new_atom_mask = jnp.ones(new_coords.shape[0], dtype=jnp.float32)

              obj = Protein(
                coordinates=new_coords,
                aatype=new_aatype,
                one_hot_sequence=new_seq,
                mask=new_mask,
                residue_index=new_res_idx,
                chain_index=new_chain_idx,
                # AtomicSystem required field
                atom_mask=new_atom_mask,
                full_coordinates=new_full_coords,
                # full_atom_mask should satisfy AtomicSystem's expectation (often flat)
                full_atom_mask=(
                  new_atom_mask.flatten()
                  if (new_atom_mask is not None and new_atom_mask.ndim > 1)
                  else new_atom_mask
                ),
                chain_ids=list(target_chains),  # Update chain list
                source=obj.source,
                coulomb14scale=getattr(obj, "coulomb14scale", None),
                lj14scale=getattr(obj, "lj14scale", None),
                # TODO: Propagate/slice physics
              )

      # Return proper type
      # Since Protein inherits AtomicSystem, it satisfies the type.
      # If specific AtomicSystem conversion is desired (e.g. flat only), we could do it here.
      yield obj

    finally:
      if tmp_path and os.path.exists(tmp_path):
        os.unlink(tmp_path)

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


def _resolve_altlocs(raw: "RawAtomData", mode: str = "occupancy") -> "RawAtomData":
  """Resolve alternate-location (altloc) atoms in raw atom data.

  Atoms in PDB/mmCIF files may appear multiple times with different coordinates
  when they have alternate conformations (altlocs). This function deduplicates
  them deterministically.

  The key is (chain_id, res_id, atom_name). Insertion codes are not tracked in
  RawAtomData, so atoms are assumed to differ only in occupancy/B-factor within
  the same (chain, res_id, atom_name) tuple. This is correct for standard PDB
  altloc usage; structures with both altlocs AND insertion codes on the same
  residue are assumed-away (rare, and the Rust atom37 assembly handles those).

  Args:
      raw: Input RawAtomData (possibly containing altloc duplicates).
      mode: Deduplication strategy.
          - ``"occupancy"`` (default): Keep the atom with the highest occupancy.
            Tiebreak: lower B-factor wins; further tiebreak: lower index wins.
          - ``"all"``: Return ``raw`` unchanged (back-compat; preserves duplicates).

  Returns:
      A new RawAtomData with duplicate (chain_id, res_id, atom_name) rows
      removed according to ``mode``.  When ``mode="all"`` the same object is
      returned.

  """
  if mode == "all":
    return raw

  from collections import defaultdict

  keys = list(zip(raw.chain_ids, raw.res_ids, raw.atom_names, strict=False))
  key_to_indices: dict = defaultdict(list)
  for i, k in enumerate(keys):
    key_to_indices[k].append(i)

  keep: list[int] = []
  for _k in keys:
    # Only add the representative for this key once (use a sentinel)
    pass
  # Build ordered list of kept indices (preserve original order of first winner)
  seen: set = set()
  keep = []
  for _i, k in enumerate(keys):
    if k not in seen:
      idxs = key_to_indices[k]
      if len(idxs) == 1:
        best = idxs[0]
      else:
        # max occupancy, tiebreak lower b_factor, then lower index (stable)
        best = max(
          idxs,
          key=lambda j: (raw.occupancies[j], -raw.b_factors[j], -j),
        )
      keep.append(best)
      seen.add(k)

  if len(keep) == raw.num_atoms:
    # Nothing to deduplicate
    return raw

  coords_arr = np.asarray(raw.coords)
  occ_arr = np.asarray(raw.occupancies)
  bfac_arr = np.asarray(raw.b_factors)
  np.asarray(raw.res_ids)

  return RawAtomData(
    num_atoms=len(keep),
    atom_names=[raw.atom_names[i] for i in keep],
    res_names=[raw.res_names[i] for i in keep],
    res_ids=[raw.res_ids[i] for i in keep],  # type: ignore[arg-type]
    chain_ids=[raw.chain_ids[i] for i in keep],
    coords=coords_arr[keep],
    elements=[raw.elements[i] for i in keep],
    occupancies=occ_arr[keep],
    b_factors=bfac_arr[keep],
  )


def _raw_atom_data_to_pdb(raw: "RawAtomData") -> str:
  """Write a RawAtomData as minimal PDB ATOM/HETATM records.

  Used internally to produce a deduplicated PDB for re-parsing by the Rust
  ``parse_structure`` backend after altloc resolution.

  All atoms are written as ``ATOM`` records (HETATM info is not carried in
  RawAtomData).  Occupancy and B-factor are preserved.  No TER/END records
  are written beyond a trailing ``END``.

  Args:
      raw: Atom data to serialise.

  Returns:
      Multi-line string in PDB ATOM record format.

  """
  lines: list[str] = []
  coords_arr = np.asarray(raw.coords)
  for i in range(raw.num_atoms):
    aname = raw.atom_names[i]
    resname = raw.res_names[i]
    chain = raw.chain_ids[i]
    resseq = int(raw.res_ids[i])
    x, y, z = float(coords_arr[i, 0]), float(coords_arr[i, 1]), float(coords_arr[i, 2])
    occ = float(raw.occupancies[i])
    bfac = float(raw.b_factors[i])
    elem = raw.elements[i] if raw.elements[i] else " "

    # PDB fixed-width: columns 13-16 = atom name (left-pad 1-char names starting at col 14)
    if len(aname) < 4:
      atom_field = f" {aname:<3s}"
    else:
      atom_field = f"{aname:<4s}"

    line = (
      f"ATOM  {i + 1:5d} {atom_field} {resname:<3s} {chain}{resseq:4d}    "
      f"{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{bfac:6.2f}          {elem:>2s}  "
    )
    lines.append(line)
  lines.append("END")
  return "\n".join(lines)


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


def _detect_mmcif(file_path: str | Path) -> bool:
  """Return True if *file_path* should be read as mmCIF rather than PDB.

  Detection order:
  1. Extension: ``.cif`` / ``.mmcif`` → True; ``.pdb`` / ``.ent`` → False.
  2. Content sniff (first 4 KB): presence of ``data_`` or ``_atom_site.``
     anywhere in the header → True.  This handles extension-less paths and
     files with unconventional suffixes.

  Args:
      file_path: Path to the structure file.

  Returns:
      True if the file should be parsed with ``_proxider.parse_mmcif``.

  """
  path = Path(file_path)
  ext = path.suffix.lower()
  if ext in {".cif", ".mmcif"}:
    return True
  if ext in {".pdb", ".ent", ".brk"}:
    return False

  # Extension ambiguous — sniff content
  try:
    with open(path, encoding="utf-8", errors="replace") as fh:
      header = fh.read(4096)
    if "data_" in header or "_atom_site." in header:
      return True
  except OSError:
    pass
  return False


def parse_pdb_to_protein(
  file_path: str | Path,
  spec=None,
  use_jax: bool = True,
  output_format_target: str | None = None,
  altloc: str = "occupancy",
) -> Protein:
  """Parse a PDB or mmCIF file and return a Protein directly.

  Supports both PDB and mmCIF formats.  The format is detected automatically
  from the file extension (``.cif``/``.mmcif`` → mmCIF; ``.pdb``/``.ent`` → PDB)
  with a content-sniff fallback for extension-less or unconventional paths.

  Args:
      file_path: Path to PDB or mmCIF file.
      spec: Optional `OutputSpec` configuration.
      use_jax: If True, return JAX arrays.
      output_format_target: "mpnn" or "general".
      altloc: Alternate-location resolution mode.

          - ``"occupancy"`` (default): keep the highest-occupancy conformer for
            every duplicate (chain, res_id, atom_name) key before Rust atom37
            assembly.
          - ``"all"``: call ``_proxider.parse_structure`` directly — byte-identical
            to pre-altloc-commit behaviour; preserves all altloc atoms.

  Returns:
      A `Protein` dataclass containing the parsed structure.

  Note:
      When altloc duplicates are present, a temporary PDB file is written after
      deduplication and then re-parsed by the Rust ``parse_structure`` backend.
      This round-trip is lossy for structures with more than 99 999 atoms or
      multi-character chain IDs, because the PDB format cannot represent them.
      For typical protein crystal structures these limits are never hit.  A future
      improvement could write the temporary file in mmCIF format to avoid this
      limitation.

  """
  if spec is None:
    spec = OutputSpec()

  if altloc == "all":
    # Back-compat: delegate entirely to the Rust format-auto-detecting backend.
    # This path is byte-identical to the pre-altloc-commit behaviour.
    result = _proxider.parse_structure(str(file_path), spec)
    return Protein.from_rust_dict(result, source=str(file_path), use_jax=use_jax)

  # --- Format-aware raw read ---
  # Use the correct low-level parser so that mmCIF files are not misrouted to
  # _proxider.parse_pdb (PDB-only), which raised "No atoms found in PDB file".
  is_mmcif = _detect_mmcif(file_path)
  if is_mmcif:
    raw_dict = _proxider.parse_mmcif(str(file_path))
  else:
    raw_dict = _proxider.parse_pdb(str(file_path))

  raw = RawAtomData(
    num_atoms=raw_dict["num_atoms"],
    atom_names=raw_dict["atom_names"],
    res_names=raw_dict["res_names"],
    res_ids=raw_dict["res_ids"],
    chain_ids=raw_dict["chain_ids"],
    coords=np.asarray(raw_dict["coords"]).reshape(-1, 3),
    elements=raw_dict["elements"],
    occupancies=raw_dict["occupancy"],
    b_factors=raw_dict["b_factors"],
  )
  resolved = _resolve_altlocs(raw, altloc)

  if resolved is raw:
    # No duplicates found; skip the temp-file round-trip.
    result = _proxider.parse_structure(str(file_path), spec)
    return Protein.from_rust_dict(result, source=str(file_path), use_jax=use_jax)

  # Duplicates found: write a deduplicated temp PDB and re-parse.
  # NOTE: temp-PDB write is lossy for >99 999 atoms or multi-char chain IDs
  # (PDB format constraint).  See docstring for details.
  pdb_text = _raw_atom_data_to_pdb(resolved)
  tmp_path: str | None = None
  try:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as tmp:
      tmp.write(pdb_text)
      tmp_path = tmp.name
    result = _proxider.parse_structure(tmp_path, spec)
    return Protein.from_rust_dict(result, source=str(file_path), use_jax=use_jax)
  finally:
    if tmp_path and os.path.exists(tmp_path):
      os.unlink(tmp_path)


def parse_structure(
  file_path: str | Path,
  spec=None,
  use_jax: bool = True,
  output_format_target: str | None = None,
  altloc: str = "occupancy",
) -> Protein:
  """Parse a protein structure using the Rust extension.

  This is the main high-level API for loading structures in Proxide. It automatically
  detects the file format (PDB/mmCIF) and uses the `_proxider` backend.

  Process:
      1.  **Format Detection**: Rust parser detects PDB/mmCIF/PQR/binary magic bytes.
      2.  **Altloc Resolution**: Duplicate (chain, res_id, atom_name) rows are resolved
          by highest occupancy before Rust atom37 assembly (unless ``altloc="all"``).
      3.  **Parsing**: Reads structure into Rust logic representation.
      4.  **Correction**: Adds hydrogens, infers bonds, or parameterizes MD if requested
          in ``spec``.
      5.  **Conversion**: Returns a Python ``Protein`` object with JAX arrays.

  Args:
      file_path: Path to the structure file.
      spec: Optional ``OutputSpec`` configuration. Controls force fields, hydrogen addition, etc.
      use_jax: If True, return arrays as ``jax.numpy.Array``. If False, use ``numpy.ndarray``.
      output_format_target: Formatting hint ("mpnn", "general").
      altloc: Alternate-location resolution mode. ``"occupancy"`` (default) keeps
          the highest-occupancy conformer for every duplicate (chain, res_id,
          atom_name) key before Rust atom37 assembly. ``"all"`` preserves all
          altloc atoms (legacy behaviour, may mix conformers in χ-angle computation).

  Returns:
      A ``Protein`` dataclass.

  Examples:
      **Basic Loading:**
      >>> protein = parse_structure("1crn.pdb")

      **MD Parameterization:**
      >>> from proxide import OutputSpec
      >>> spec = OutputSpec(parameterize_md=True, force_field="protein.ff14SB.xml")
      >>> protein = parse_structure("1crn.pdb", spec)
      >>> protein.charges.shape
      (327,)

      **Keep all altloc conformers (legacy):**
      >>> protein = parse_structure("1ejg.pdb", altloc="all")

  """
  return parse_pdb_to_protein(file_path, spec, use_jax, output_format_target, altloc=altloc)


parse_xtc = getattr(_proxider, "parse_xtc", None)
parse_mdc = getattr(_proxider, "parse_mdc", None)


class TrajectoryStream:
  """Stream a trajectory file in chunks.

  This class acts as a high-performance wrapper around the native Rust `PyTrajectoryIterator`.
  It reads coordinate frames efficiently without bringing the entire array into memory at once.

  Attributes:
      file_path (str | Path): Path to the trajectory file.
      chunk_size (int): Number of frames loaded per chunk.
  """

  def __init__(self, file_path: str | Path, chunk_size: int = 100):
    """Initialize a streaming reader for trajectory files.

    Args:
        file_path: Path to the trajectory file (.xtc, .dcd).
        chunk_size: Number of frames per chunk.
    """
    from proxide._proxider import PyTrajectoryIterator  # type: ignore

    self.file_path = file_path
    self.chunk_size = chunk_size
    self._iterator = PyTrajectoryIterator(str(file_path), chunk_size)

  def __iter__(self):
    """Yield chunks containing 'coordinates' and tracking metadata."""
    yield from self._iterator

def iterload(file_path: str | Path, chunk_size: int = 100):
  """Stream a trajectory file in chunks (Legacy alias).

  .. warning::
      This function is an alias for :class:`TrajectoryStream`. Use the class
      for type hinting and a cleaner API.

  Args:
      file_path: Path to the trajectory file (.xtc, .dcd)
      chunk_size: Number of frames per chunk

  Yields:
      Dictionaries containing 'coordinates' (chunk_size, N, 3) and metadata.
  """
  yield from TrajectoryStream(file_path, chunk_size)


def write_dcd(file_path: str | Path, n_atoms: int, delta: float = 1.0, has_unit_cell: bool = False):
  """Establish a streaming DCD writer.

  Args:
      file_path: Path to output DCD file
      n_atoms: Number of atoms
      delta: Time step
      has_unit_cell: Whether to include unit cell data
  """
  from proxide._proxider import PyDcdWriter  # type: ignore

  return PyDcdWriter(str(file_path), n_atoms, delta, has_unit_cell)


def parse_pdb_raw_rust(file_path: str | Path, altloc: str = "occupancy") -> RawAtomData:
  """Parse a PDB file and return raw atom data (low-level).

  This is useful for custom processing pipelines that need access
  to the raw atom data before formatting.

  Args:
      file_path: Path to PDB file
      altloc: Alternate-location resolution mode. ``"occupancy"`` (default) keeps
          the highest-occupancy conformer for each (chain, res_id, atom_name).
          ``"all"`` preserves all atoms including altloc duplicates.

  Returns:
      RawAtomData with parsed atom information

  Raises:
      ValueError: If parsing fails

  """
  result = _proxider.parse_pdb(str(file_path))

  raw = RawAtomData(
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
  return _resolve_altlocs(raw, altloc)


def parse_mmcif_rust(file_path: str | Path, altloc: str = "occupancy") -> RawAtomData:
  """Parse an mmCIF file and return raw atom data.

  Args:
      file_path: Path to mmCIF (.cif) file
      altloc: Alternate-location resolution mode. ``"occupancy"`` (default) keeps
          the highest-occupancy conformer for each (chain, res_id, atom_name).
          ``"all"`` preserves all atoms including altloc duplicates.

  Returns:
      RawAtomData with parsed atom information

  Raises:
      ValueError: If parsing fails

  """
  result = _proxider.parse_mmcif(str(file_path))

  raw = RawAtomData(
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
  return _resolve_altlocs(raw, altloc)


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
  result = _proxider.load_forcefield(str(file_path))

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


def parse_xtc_rust(file_path: str | Path) -> dict[str, ArrayLike]:
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
    raise ImportError("parse_xtc not found in _proxider. Ensure 'trajectories' feature is enabled.")

  return parse_xtc(str(file_path))


@register_parser(["mdc"])
def parse_mdc_rust(file_path: str | Path) -> dict[str, ArrayLike]:
  """Parse an MDC trajectory file using the Rust extension.

  Args:
      file_path: Path to MDC file

  Returns:
      Dictionary with 'coordinates', 'times', 'num_frames', 'num_atoms'

  Raises:
      ImportError: If trajectory feature not available
      ValueError: If parsing fails

  """
  if parse_mdc is None:
    raise ImportError("parse_mdc not found in _proxider. Ensure 'mdc' feature is enabled.")

  return parse_mdc(str(file_path))


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
  if not hasattr(_proxider, "parse_mdtraj_h5_metadata"):
    raise ImportError(
      "HDF5/MDTRAJ support not available in this build of Proxide. "
      "This often happens on Windows/macOS where HDF5 dependencies are complex. "
      "Please use the Linux 'full' build or convert your trajectory to XTC/DCD format."
    )

  result = _proxider.parse_mdtraj_h5_metadata(str(file_path))

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
  frame_result = _proxider.parse_mdtraj_h5_frame(str(file_path), frame_idx)

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
  if not hasattr(_proxider, "parse_mdcath_metadata"):
    raise ImportError(
      "MDCATH support not available in this build of Proxide. "
      "This is currently a Linux-only feature (requires HDF5). "
      "Please use the Linux 'full' build or contact support for help with your platform."
    )

  result = _proxider.parse_mdcath_metadata(str(file_path))

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
  return _proxider.get_mdcath_replicas(str(file_path), domain_id, temperature)


def parse_mdcath_frame(
  file_path: str | Path,
  domain_id: str,
  temperature: str,
  replica: str,
  frame_idx: int = 0,
) -> dict[str, ArrayLike]:
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
  return _proxider.parse_mdcath_frame(str(file_path), domain_id, temperature, replica, frame_idx)


def is_hdf5_support_available() -> bool:
  """Check if HDF5 parsing support is available.

  Returns:
      True if mdcath feature was compiled, False otherwise.

  """
  # Check if the function exists
  if not hasattr(_proxider, "parse_mdtraj_h5_metadata"):
    return False

  # Try to call the function - it raises ImportError if feature not enabled
  try:
    _proxider.parse_mdtraj_h5_metadata("/nonexistent")
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

  Always returns True since _proxider is now a hard dependency.
  """
  return True


def get_rust_capabilities() -> dict[str, bool]:
  """Get dictionary of available Rust capabilities.

  Returns:
      Dictionary mapping capability names to availability status.

  """
  return {
    "parse_pdb": hasattr(_proxider, "parse_pdb"),
    "parse_mmcif": hasattr(_proxider, "parse_mmcif"),
    "parse_structure": hasattr(_proxider, "parse_structure"),
    "load_forcefield": hasattr(_proxider, "load_forcefield"),
    "parse_xtc": hasattr(_proxider, "parse_xtc"),
    "parse_mdc": hasattr(_proxider, "parse_mdc"),
    "parse_mdtraj_h5": hasattr(_proxider, "parse_mdtraj_h5_metadata"),
    "parse_mdcath": hasattr(_proxider, "parse_mdcath_metadata"),
    "atomic_system_types": hasattr(_proxider, "AtomicSystem"),
  }
