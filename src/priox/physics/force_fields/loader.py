"""Force field parameter storage and loading utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from huggingface_hub import hf_hub_download, list_repo_files

from priox.physics.constants import DEFAULT_EPSILON, DEFAULT_SIGMA
from priox.physics.force_fields.components import (
    AtomTypeParams,
    BondPotentialParams,
    AnglePotentialParams,
    DihedralPotentialParams,
    CMAPParams,
    UreyBradleyParams,
    VirtualSiteParams,
    NonbondedGlobalParams,
    GAFFNonbondedParams,
)


class FullForceField(eqx.Module):
  """Force field parameters stored as a PyTree.

  This is a data container (not a computational class). JAX arrays are
  dynamic leaves that can be updated, while metadata is static.
  
  Now refactored to use modular components.
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
  
  # Top-level Metadata
  # Top-level Metadata
  residue_templates: dict[str, list[tuple[str, str]]] = eqx.field(static=True, default_factory=dict)
  source_files: list[str] = eqx.field(static=True, default_factory=list)

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


def _make_ff_skeleton(hyperparams: dict[str, Any]) -> FullForceField:
  """Create an empty FullForceField from hyperparameters."""
  num_atoms = len(hyperparams["id_to_atom_key"])
  
  # Infer CMAP shape
  num_maps = hyperparams.pop("num_cmap_maps", 0)
  grid_size = hyperparams.pop("cmap_grid_size", 24)
  
  # Helper to get default or popped value
  def get(key, default):
      return hyperparams.get(key, default)

  # Check compatibility
  cmap_torsions = get("cmap_torsions", [])
  if num_maps == 0 and cmap_torsions:
      max_idx = -1
      for t in cmap_torsions:
          if "map_index" in t:
              max_idx = max(max_idx, t["map_index"])
      if max_idx >= 0:
          num_maps = max_idx + 1

  # Construct Components
  atom_params = AtomTypeParams(
      charges=jnp.zeros(num_atoms, dtype=jnp.float32),
      sigmas=jnp.zeros(num_atoms, dtype=jnp.float32),
      epsilons=jnp.zeros(num_atoms, dtype=jnp.float32),
      radii=jnp.zeros(num_atoms, dtype=jnp.float32),
      scales=jnp.zeros(num_atoms, dtype=jnp.float32),
      atom_key_to_id=hyperparams.get("atom_key_to_id", {}),
      id_to_atom_key=hyperparams.get("id_to_atom_key", []),
      atom_class_map=hyperparams.get("atom_class_map", {}),
      atom_type_map=hyperparams.get("atom_type_map", {}),
  )

  bond_params = BondPotentialParams(params=hyperparams.get("bonds", []))
  angle_params = AnglePotentialParams(params=hyperparams.get("angles", []))
  dihedral_params = DihedralPotentialParams(
      propers=hyperparams.get("propers", []),
      impropers=hyperparams.get("impropers", [])
  )
  cmap_params = CMAPParams(
      energy_grids=jnp.zeros((num_maps, grid_size, grid_size), dtype=jnp.float32),
      torsions=hyperparams.get("cmap_torsions", [])
  )
  urey_bradley_params = UreyBradleyParams(params=hyperparams.get("urey_bradley_bonds", []))
  virtual_site_params = VirtualSiteParams(definitions=hyperparams.get("virtual_sites", {}))
  
  # Global params
  global_params = NonbondedGlobalParams(
      coulomb14scale=hyperparams.get("coulomb14scale", 0.833333),
      lj14scale=hyperparams.get("lj14scale", 0.5),
      cutoff_distance=hyperparams.get("cutoff_distance", 10.0),
      switch_distance=hyperparams.get("switch_distance", 9.0),
      use_dispersion_correction=hyperparams.get("use_dispersion_correction", True),
      use_pme=hyperparams.get("use_pme", False),
      ewald_error_tolerance=hyperparams.get("ewald_error_tolerance", 0.0005),
      dielectric_constant=hyperparams.get("dielectric_constant", 78.5),
  )
  
  # GAFF Params
  gaff_params = None
  if "gaff_type_to_index" in hyperparams:
      num_gaff_types = len(hyperparams["gaff_type_to_index"])
      gaff_params = GAFFNonbondedParams(
          type_to_index=hyperparams["gaff_type_to_index"],
          sigmas=jnp.zeros(num_gaff_types, dtype=jnp.float32),
          epsilons=jnp.zeros(num_gaff_types, dtype=jnp.float32),
      )

  return FullForceField(
      atom_params=atom_params,
      bond_params=bond_params,
      angle_params=angle_params,
      dihedral_params=dihedral_params,
      cmap_params=cmap_params,
      urey_bradley_params=urey_bradley_params,
      virtual_site_params=virtual_site_params,
      global_params=global_params,
      gaff_nonbonded_params=gaff_params,
      residue_templates=hyperparams.get("residue_templates", {}),
      source_files=hyperparams.get("source_files", []),
  )


def save_force_field(
  filepath: Path | str,
  force_field: FullForceField,
) -> None:
  """Save force field to disk using Equinox serialization."""
  filepath = Path(filepath)

  hyperparams = {
    # Atoms
    "atom_key_to_id": force_field.atom_params.atom_key_to_id,
    "id_to_atom_key": force_field.atom_params.id_to_atom_key,
    "atom_class_map": force_field.atom_params.atom_class_map,
    "atom_type_map": force_field.atom_params.atom_type_map,
    # Potentials
    "bonds": force_field.bond_params.params,
    "angles": force_field.angle_params.params,
    "propers": force_field.dihedral_params.propers,
    "impropers": force_field.dihedral_params.impropers,
    "cmap_torsions": force_field.cmap_params.torsions,
    "urey_bradley_bonds": force_field.urey_bradley_params.params,
    "virtual_sites": force_field.virtual_site_params.definitions,
    # Metadata
    "residue_templates": force_field.residue_templates,
    "source_files": force_field.source_files,
    "num_cmap_maps": force_field.cmap_params.energy_grids.shape[0],
    "cmap_grid_size": force_field.cmap_params.energy_grids.shape[1],
    # Global
    "coulomb14scale": force_field.global_params.coulomb14scale,
    "lj14scale": force_field.global_params.lj14scale,
    "cutoff_distance": force_field.global_params.cutoff_distance,
    "switch_distance": force_field.global_params.switch_distance,
    "use_dispersion_correction": force_field.global_params.use_dispersion_correction,
    "use_pme": force_field.global_params.use_pme,
    "ewald_error_tolerance": force_field.global_params.ewald_error_tolerance,
    "dielectric_constant": force_field.global_params.dielectric_constant,
  }

  sanitized_hyperparams = hyperparams.copy()
  if force_field.gaff_nonbonded_params is not None:
      print(f"DEBUG: Saving GAFF Params with {len(force_field.gaff_nonbonded_params.type_to_index)} types")
      hyperparams["gaff_type_to_index"] = force_field.gaff_nonbonded_params.type_to_index
  else:
      print("DEBUG: Force Field has NO GAFF Params to save")

  # JSON keys must be strings, so convert tuple keys in atom_key_to_id
  sanitized_atom_key_to_id = {
      f"{key[0]}|{key[1]}": value
      for key, value in hyperparams["atom_key_to_id"].items()  # type: ignore
  }
  hyperparams["atom_key_to_id"] = sanitized_atom_key_to_id

  with filepath.open("wb") as f:
    # Save metadata structure first
    hyperparams_str = json.dumps(hyperparams)
    f.write(len(hyperparams_str).to_bytes(8, "big"))
    f.write(hyperparams_str.encode("utf-8"))
    
    # Save leaves
    eqx.tree_serialise_leaves(f, force_field)


def load_force_field(
  filepath: Path | str,
) -> FullForceField:
  """Load force field from disk."""
  filepath = Path(filepath)

  with filepath.open("rb") as f:
    # Read metadata length
    length_bytes = f.read(8)
    if not length_bytes:
        raise ValueError("Empty file")
    length = int.from_bytes(length_bytes, "big")
    
    # Read metadata
    hyperparams_bytes = f.read(length)
    hyperparams = json.loads(hyperparams_bytes.decode("utf-8"))

    if "atom_key_to_id" in hyperparams:
      hyperparams["atom_key_to_id"] = {
        tuple(key.split("|", 1)): value for key, value in hyperparams["atom_key_to_id"].items()
      }
    if "id_to_atom_key" in hyperparams:
      hyperparams["id_to_atom_key"] = [
        tuple(item) if isinstance(item, list) else item for item in hyperparams["id_to_atom_key"]
      ]
    skeleton = _make_ff_skeleton(hyperparams)
    return eqx.tree_deserialise_leaves(f, skeleton)


def load_force_field_from_hub(
  force_field_name: str,
  repo_id: str = "maraxen/eqx-ff",
  cache_dir: Path | str | None = None,
) -> FullForceField:
  """Load force field from HuggingFace Hub."""
  filename = f"{force_field_name}.eqx"
  local_path = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    repo_type="dataset",
    cache_dir=cache_dir,
  )
  return load_force_field(local_path)


def list_available_force_fields(
  repo_id: str = "maraxen/eqx-ff",
) -> list[str]:
  """List available force fields on HuggingFace Hub."""
  files = list_repo_files(repo_id, repo_type="dataset")
  force_fields = [f.replace(".eqx", "") for f in files if f.endswith(".eqx")]
  return sorted(force_fields)
