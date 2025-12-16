"""Force field parameter storage and loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from flax.struct import dataclass, field

from proxide.io.parsing import rust as rw
from proxide.physics.constants import DEFAULT_EPSILON, DEFAULT_SIGMA
from proxide.physics.force_fields.components import (
  AnglePotentialParams,
  AtomTypeParams,
  BondPotentialParams,
  CMAPParams,
  DihedralPotentialParams,
  GAFFNonbondedParams,
  NonbondedGlobalParams,
  UreyBradleyParams,
  VirtualSiteParams,
)


@dataclass(frozen=True)
class FullForceField:
  """Force field parameters stored as a Flax dataclass.

  This is a data container (not a computational class). JAX arrays are
  dynamic leaves that can be updated, while metadata is static.
  """

  # Modular Components
  atom_params: AtomTypeParams
  bond_params: BondPotentialParams
  angle_params: AnglePotentialParams
  dihedral_params: DihedralPotentialParams
  cmap_params: CMAPParams
  urey_bradley_params: UreyBradleyParams
  virtual_site_params: VirtualSiteParams
  global_params: NonbondedGlobalParams
  gaff_nonbonded_params: GAFFNonbondedParams | None = None

  # Top-level Metadata (must have defaults since they follow gaff_nonbonded_params)
  residue_templates: dict[str, list[tuple[str, str]]] = field(
    default_factory=dict,
    pytree_node=False,
  )
  source_files: list[str] = field(default_factory=list, pytree_node=False)

  # --- Backward Compatibility Properties ---

  @property
  def charges_by_id(self) -> jnp.ndarray:
    return self.atom_params.charges

  @property
  def sigmas_by_id(self) -> jnp.ndarray:
    return self.atom_params.sigmas

  @property
  def epsilons_by_id(self) -> jnp.ndarray:
    return self.atom_params.epsilons

  @property
  def radii_by_id(self) -> jnp.ndarray:
    return self.atom_params.radii

  @property
  def scales_by_id(self) -> jnp.ndarray:
    return self.atom_params.scales

  @property
  def cmap_energy_grids(self) -> jnp.ndarray:
    return self.cmap_params.energy_grids

  @property
  def atom_key_to_id(self) -> dict[tuple[str, str], int]:
    return self.atom_params.atom_key_to_id

  @property
  def id_to_atom_key(self) -> list[tuple[str, str]]:
    return self.atom_params.id_to_atom_key

  @property
  def atom_class_map(self) -> dict[str, str]:
    return self.atom_params.atom_class_map

  @property
  def atom_type_map(self) -> dict[str, str]:
    return self.atom_params.atom_type_map

  @property
  def bonds(self) -> list[tuple[str, str, float, float]]:
    return self.bond_params.params

  @property
  def angles(self) -> list[tuple[str, str, str, float, float]]:
    return self.angle_params.params

  @property
  def propers(self) -> list[dict[str, Any]]:
    return self.dihedral_params.propers

  @property
  def impropers(self) -> list[dict[str, Any]]:
    return self.dihedral_params.impropers

  @property
  def cmap_torsions(self) -> list[dict[str, Any]]:
    return self.cmap_params.torsions

  @property
  def urey_bradley_bonds(self) -> list[tuple[str, str, float, float]]:
    return self.urey_bradley_params.params

  @property
  def virtual_sites(self) -> dict[str, list[dict[str, Any]]]:
    return self.virtual_site_params.definitions

  def get_charge(self, residue: str, atom: str) -> float:
    """Get charge for a specific atom."""
    atom_id = self.atom_params.atom_key_to_id.get((residue, atom))
    if atom_id is None:
      return 0.0
    return float(self.atom_params.charges[atom_id])

  def get_lj_params(self, residue: str, atom: str) -> tuple[float, float]:
    """Get Lennard-Jones parameters for a specific atom."""
    atom_id = self.atom_params.atom_key_to_id.get((residue, atom))
    if atom_id is None:
      return DEFAULT_SIGMA, DEFAULT_EPSILON
    return float(self.atom_params.sigmas[atom_id]), float(self.atom_params.epsilons[atom_id])

  def get_gbsa_params(self, residue: str, atom: str) -> tuple[float, float]:
    """Get GBSA parameters for a specific atom."""
    atom_id = self.atom_params.atom_key_to_id.get((residue, atom))
    if atom_id is None:
      return 0.0, 0.0
    return float(self.atom_params.radii[atom_id]), float(self.atom_params.scales[atom_id])


def _convert_rust_ff_to_full(ff_data: rw.ForceFieldData, source_file: str) -> FullForceField:
  """Convert parsed ForceFieldData to FullForceField."""
  # 1. Expand atoms from residue templates to build ID maps
  atom_key_to_id = {}
  id_to_atom_key = []
  atom_class_map = {}
  atom_type_map = {}

  # Build type -> class map
  type_to_class = {}
  for at in ff_data.atom_types:
    t_name = at.get("name")
    t_class = at.get("class")
    if t_name:
      type_to_class[t_name] = t_class if t_class else t_name

  # Pre-process nonbonded params for fast lookup by type/class
  # Note: Using class if available, else type. In XML, often type or class is used.
  # We map type -> (sigma, epsilon)
  # And we map charge: (residue, atom) -> charge (priority) or type -> charge

  nb_by_type = {}
  for nb in ff_data.nonbonded_params:
    if "atom_type" in nb:
      nb_by_type[nb["atom_type"]] = nb

  gbsa_by_type = {}
  for g in ff_data.gbsa_obc_params:
    if "atom_type" in g:
      gbsa_by_type[g["atom_type"]] = g

  # Lists for array construction
  charges_list = []
  sigmas_list = []
  epsilons_list = []
  radii_list = []
  scales_list = []

  idx = 0
  # Sort templates for deterministic order
  sorted_templates = sorted(ff_data.residue_templates, key=lambda x: x["name"])

  for res in sorted_templates:
    res_name = res["name"]
    for atom in res["atoms"]:
      atom_name = atom["name"]
      atom_type = atom["type"]  # This is the type/class used for lookup

      key = (res_name, atom_name)
      if key in atom_key_to_id:
        continue  # Skip duplicates if any

      atom_key_to_id[key] = idx
      id_to_atom_key.append(key)
      atom_type_map[f"{res_name}|{atom_name}"] = atom_type

      # Populate atom_class_map for core.py dihedral matching
      # Key format: "{res_name}_{atom_name}" (e.g. "ALA_CA", "NALA_N")
      class_name = type_to_class.get(atom_type, atom_type)
      atom_class_map[f"{res_name}_{atom_name}"] = class_name

      # Look up params
      # Charge: atom.charge (from residue def) usually takes precedence
      charge = atom.get("charge", 0.0)

      # VdW
      nb = nb_by_type.get(atom_type)
      if nb:
        sigma = nb.get("sigma", DEFAULT_SIGMA)
        epsilon = nb.get("epsilon", DEFAULT_EPSILON)
        # If charge not in atom, check nb? usually nb charge is 0 or global
        if "charge" not in atom and "charge" in nb:
          charge = nb["charge"]
      else:
        sigma = DEFAULT_SIGMA
        epsilon = DEFAULT_EPSILON

      # GBSA
      gbsa = gbsa_by_type.get(atom_type)
      if gbsa:
        radius = gbsa.get("radius", 0.0)
        scale = gbsa.get("scale", 0.0)
      else:
        radius = 0.0
        scale = 0.0

      charges_list.append(charge)
      sigmas_list.append(sigma)
      epsilons_list.append(epsilon)
      radii_list.append(radius)
      scales_list.append(scale)

      idx += 1

  num_atoms = len(charges_list)

  atom_params = AtomTypeParams(
    charges=jnp.array(charges_list, dtype=jnp.float32),
    sigmas=jnp.array(sigmas_list, dtype=jnp.float32),
    epsilons=jnp.array(epsilons_list, dtype=jnp.float32),
    radii=jnp.array(radii_list, dtype=jnp.float32),
    scales=jnp.array(scales_list, dtype=jnp.float32),
    atom_key_to_id=atom_key_to_id,
    id_to_atom_key=id_to_atom_key,
    atom_class_map=atom_class_map,  # TODO: Populate if needed
    atom_type_map=atom_type_map,
  )

  # Bonds
  bonds = []
  for b in ff_data.harmonic_bonds:
    bonds.append((b["class1"], b["class2"], b["length"], b["k"]))
  bond_params = BondPotentialParams(params=bonds)

  # Angles
  angles = []
  for a in ff_data.harmonic_angles:
    angles.append((a["class1"], a["class2"], a["class3"], a.get("angle", 0.0), a["k"]))
  angle_params = AnglePotentialParams(params=angles)

  # Dihedrals
  dihedral_params = DihedralPotentialParams(
    propers=ff_data.proper_torsions,
    impropers=ff_data.improper_torsions,
  )

  # CMAP
  cmap_maps = ff_data.cmap_maps
  if cmap_maps:
    # Assuming all maps same size? Rust returns size per map.
    # Find max size
    max_size = max(m["size"] for m in cmap_maps) if cmap_maps else 24
    num_maps = len(cmap_maps)
    # Flattened energies in Rust. Reshape to (size, size)
    grids = []
    for m in cmap_maps:
      size = m["size"]
      energies = np.array(m["energies"], dtype=np.float32).reshape(size, size)
      # Pad if needed? Assuming usually all 24.
      if size != max_size:
        # Padding logic omitted for brevity, assuming consistency
        pass
      grids.append(energies)
    energy_grids = jnp.array(grids)
  else:
    energy_grids = jnp.zeros((0, 24, 24), dtype=jnp.float32)

  cmap_params = CMAPParams(
    energy_grids=energy_grids,
    torsions=ff_data.cmap_torsions if ff_data.cmap_torsions else [],
  )

  # Defaults for others
  urey_bradley_params = UreyBradleyParams(params=[])
  virtual_site_params = VirtualSiteParams(definitions={})

  # Global
  global_params = NonbondedGlobalParams()  # Use defaults or parse if available in XML? (XML usually doesn't have 1-4 scales in standard place, typically defined by FF type)

  # Helper for residue templates meta
  res_templates_meta = {}
  for res in sorted_templates:
    res_name = res["name"]
    # core.py expects residue_templates to be a list of bond pairs (atom1, atom2)
    # Rust extraction returns "bonds" as list of strings (a1, a2)
    res_templates_meta[res_name] = res.get("bonds", [])

  return FullForceField(
    atom_params=atom_params,
    bond_params=bond_params,
    angle_params=angle_params,
    dihedral_params=dihedral_params,
    cmap_params=cmap_params,
    urey_bradley_params=urey_bradley_params,
    virtual_site_params=virtual_site_params,
    global_params=global_params,
    residue_templates=res_templates_meta,
    source_files=[source_file],
  )


def load_force_field(
  name_or_path: str,
) -> FullForceField:
  """Load force field from disk or assets."""
  path = Path(name_or_path)

  # If not absolute/exists, check assets
  if not path.exists():
    # Try known assets location
    # Assuming module path structure: src/priox/assets
    # We function is in src/priox/physics/force_fields/loader.py
    # Assets in src/priox/assets
    # Relative: ../../assets
    asset_dir = Path(__file__).parent.parent.parent / "assets"

    # Try direct match in root
    potential_path = asset_dir / path.name
    if potential_path.exists():
      path = potential_path
    elif (asset_dir / f"{path.name}.xml").exists():
      path = asset_dir / f"{path.name}.xml"
    else:
      # Search recursively in subdirectories
      found = list(asset_dir.rglob(f"{path.name}"))
      if not found:
        found = list(asset_dir.rglob(f"{path.name}.xml"))
      if found:
        path = found[0]
      else:
        raise ValueError(f"Force field file not found: {name_or_path} (checked {asset_dir})")

  ff_data = rw.load_forcefield_rust(path)
  return _convert_rust_ff_to_full(ff_data, str(path))


def list_available_force_fields() -> list[str]:
  """List available force fields in assets (recursively)."""
  asset_dir = Path(__file__).parent.parent.parent / "assets"
  if not asset_dir.exists():
    return []
  # Get all XML files recursively, return relative paths without .xml extension
  result = []
  for xml_file in asset_dir.rglob("*.xml"):
    # Get path relative to assets directory
    rel_path = xml_file.relative_to(asset_dir)
    # Return stem (filename without .xml) with parent path if in subdir
    if rel_path.parent != Path():
      result.append(str(rel_path.parent / rel_path.stem))
    else:
      result.append(rel_path.stem)
  return sorted(result)
