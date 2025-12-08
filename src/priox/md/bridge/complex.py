"""Protein-ligand complex parameterization and merging utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from priox.io.parsing.molecule import Molecule
    from priox.physics.force_fields import FullForceField

from priox.md.bridge.types import SystemParams
from priox.md.bridge.core import parameterize_system
from priox.md.bridge.ligand import parameterize_ligand


def merge_system_params(
    protein_params: SystemParams,
    ligand_params: SystemParams,
    n_protein_atoms: int,
) -> SystemParams:
    """Merge protein and ligand SystemParams into a combined system.
    
    This function handles:
    - Offsetting ligand atom indices by the number of protein atoms
    - Concatenating all parameter arrays
    - Expanding and combining exclusion/scaling matrices
    - Setting up cross-system non-bonded interactions (full strength)
    
    Args:
        protein_params: SystemParams from parameterize_system() for the protein.
        ligand_params: SystemParams from parameterize_ligand() for the ligand.
        n_protein_atoms: Number of atoms in the protein system.
        
    Returns:
        Combined SystemParams for the protein-ligand complex.
    """
    n_lig = len(ligand_params["charges"])
    n_total = n_protein_atoms + n_lig
    
    # --- Simple concatenation for 1D arrays ---
    merged = {}
    for key in ["charges", "masses", "sigmas", "epsilons", "gb_radii", "scaled_radii"]:
        merged[key] = jnp.concatenate([
            protein_params[key],
            ligand_params[key],
        ])
    
    # --- Offset and concatenate bonded terms ---
    # Bonds
    lig_bonds = ligand_params["bonds"] + n_protein_atoms
    merged["bonds"] = jnp.concatenate([protein_params["bonds"], lig_bonds])
    merged["bond_params"] = jnp.concatenate([
        protein_params["bond_params"],
        ligand_params["bond_params"],
    ])
    
    # Constrained bonds
    if len(ligand_params["constrained_bonds"]) > 0:
        lig_cbonds = ligand_params["constrained_bonds"] + n_protein_atoms
    else:
        lig_cbonds = ligand_params["constrained_bonds"]
    merged["constrained_bonds"] = jnp.concatenate([
        protein_params["constrained_bonds"],
        lig_cbonds,
    ])
    merged["constrained_bond_lengths"] = jnp.concatenate([
        protein_params["constrained_bond_lengths"],
        ligand_params["constrained_bond_lengths"],
    ])
    
    # Angles
    lig_angles = ligand_params["angles"] + n_protein_atoms
    merged["angles"] = jnp.concatenate([protein_params["angles"], lig_angles])
    merged["angle_params"] = jnp.concatenate([
        protein_params["angle_params"],
        ligand_params["angle_params"],
    ])
    
    # Dihedrals
    lig_dihedrals = ligand_params["dihedrals"] + n_protein_atoms
    merged["dihedrals"] = jnp.concatenate([protein_params["dihedrals"], lig_dihedrals])
    merged["dihedral_params"] = jnp.concatenate([
        protein_params["dihedral_params"],
        ligand_params["dihedral_params"],
    ])
    
    # Impropers
    lig_impropers = ligand_params["impropers"] + n_protein_atoms
    merged["impropers"] = jnp.concatenate([protein_params["impropers"], lig_impropers])
    merged["improper_params"] = jnp.concatenate([
        protein_params["improper_params"],
        ligand_params["improper_params"],
    ])
    
    # --- Scaling Matrices ---
    # Create expanded matrices (n_total x n_total)
    # Cross-system interactions should be full strength (1.0)
    scale_vdw = np.ones((n_total, n_total), dtype=np.float32)
    scale_elec = np.ones((n_total, n_total), dtype=np.float32)
    
    # Copy protein block
    prot_vdw = np.array(protein_params["scale_matrix_vdw"])
    prot_elec = np.array(protein_params["scale_matrix_elec"])
    scale_vdw[:n_protein_atoms, :n_protein_atoms] = prot_vdw
    scale_elec[:n_protein_atoms, :n_protein_atoms] = prot_elec
    
    # Copy ligand block
    lig_vdw = np.array(ligand_params["scale_matrix_vdw"])
    lig_elec = np.array(ligand_params["scale_matrix_elec"])
    scale_vdw[n_protein_atoms:, n_protein_atoms:] = lig_vdw
    scale_elec[n_protein_atoms:, n_protein_atoms:] = lig_elec
    
    # Self-interactions = 0
    np.fill_diagonal(scale_vdw, 0.0)
    np.fill_diagonal(scale_elec, 0.0)
    
    merged["scale_matrix_vdw"] = jnp.array(scale_vdw)
    merged["scale_matrix_elec"] = jnp.array(scale_elec)
    merged["exclusion_mask"] = merged["scale_matrix_vdw"] > 0.0
    
    # --- Backbone indices ---
    # Ligand has no backbone; keep protein backbone indices
    merged["backbone_indices"] = protein_params["backbone_indices"]
    
    # --- CMAP ---
    # Keep protein CMAP only (ligands don't have CMAP)
    merged["cmap_energy_grids"] = protein_params["cmap_energy_grids"]
    merged["cmap_indices"] = protein_params["cmap_indices"]
    merged["cmap_torsions"] = protein_params["cmap_torsions"]
    merged["cmap_coeffs"] = protein_params["cmap_coeffs"]
    
    # --- Urey-Bradley ---
    # Typically protein-only, but concatenate to be safe
    if len(ligand_params["urey_bradley_bonds"]) > 0:
        lig_ub = ligand_params["urey_bradley_bonds"] + n_protein_atoms
    else:
        lig_ub = ligand_params["urey_bradley_bonds"]
    merged["urey_bradley_bonds"] = jnp.concatenate([
        protein_params["urey_bradley_bonds"],
        lig_ub,
    ])
    merged["urey_bradley_params"] = jnp.concatenate([
        protein_params["urey_bradley_params"],
        ligand_params["urey_bradley_params"],
    ])
    
    # --- Virtual Sites ---
    # Typically protein-only (TIP4P water), but handle both
    if len(ligand_params["virtual_site_def"]) > 0:
        lig_vs = ligand_params["virtual_site_def"] + n_protein_atoms
    else:
        lig_vs = ligand_params["virtual_site_def"]
    merged["virtual_site_def"] = jnp.concatenate([
        protein_params["virtual_site_def"],
        lig_vs,
    ])
    merged["virtual_site_params"] = jnp.concatenate([
        protein_params["virtual_site_params"],
        ligand_params["virtual_site_params"],
    ])
    
    return merged


def parameterize_complex(
    protein_ff: "FullForceField",
    ligand_ff: "FullForceField",
    residues: list[str],
    atom_names: list[str],
    atom_counts: list[int],
    ligand: "Molecule",
    protein_positions: np.ndarray | None = None,
) -> tuple[SystemParams, np.ndarray | None]:
    """Parameterize a protein-ligand complex.
    
    Convenience function that combines:
    1. parameterize_system() for the protein
    2. parameterize_ligand() for the ligand
    3. merge_system_params() to combine them
    
    Args:
        protein_ff: Force field for the protein (e.g., ff19SB).
        ligand_ff: Force field for the ligand (e.g., gaff-2.2.20).
                  Can be the same as protein_ff if it includes GAFF types.
        residues: List of residue names for the protein.
        atom_names: List of atom names for the protein.
        atom_counts: List of atom counts per residue.
        ligand: Parsed Molecule object with GAFF types.
        protein_positions: Optional protein coordinates. If provided along with
                          ligand.positions, returns combined coordinates.
        
    Returns:
        Tuple of (merged_params, combined_positions).
        combined_positions is None if protein_positions is not provided.
    """
    # Parameterize protein
    protein_params = parameterize_system(
        force_field=protein_ff,
        residues=residues,
        atom_names=atom_names,
        atom_counts=atom_counts,
    )
    n_protein = sum(atom_counts)
    
    # Parameterize ligand
    ligand_params = parameterize_ligand(
        molecule=ligand,
        force_field=ligand_ff,
    )
    
    # Merge
    merged = merge_system_params(protein_params, ligand_params, n_protein)
    
    # Combine positions if provided
    combined_positions = None
    if protein_positions is not None:
        combined_positions = np.concatenate([
            protein_positions,
            ligand.positions,
        ], axis=0)
    
    return merged, combined_positions


def add_ligand_to_params(
    protein_params: SystemParams,
    ligand: "Molecule",
    ligand_ff: "FullForceField",
    n_protein_atoms: int,
) -> SystemParams:
    """Add a ligand to existing protein SystemParams.
    
    Useful when you already have parameterized protein and want to add a ligand.
    
    Args:
        protein_params: Pre-computed protein SystemParams.
        ligand: Molecule object for the ligand.
        ligand_ff: GAFF force field for the ligand.
        n_protein_atoms: Number of atoms in the protein.
        
    Returns:
        Merged SystemParams for the complex.
    """
    ligand_params = parameterize_ligand(ligand, ligand_ff)
    return merge_system_params(protein_params, ligand_params, n_protein_atoms)
