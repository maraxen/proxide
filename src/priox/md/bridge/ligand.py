"""Ligand parameterization for GAFF/GAFF2 force fields."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from priox.io.parsing.molecule import Molecule
    from priox.physics.force_fields import FullForceField

from priox.md.bridge.types import SystemParams


def parameterize_ligand(
    molecule: "Molecule",
    force_field: "FullForceField",
    residue_name: str | None = None,
) -> SystemParams:
    """Convert a small molecule to SystemParams using GAFF force field.
    
    This function takes a Molecule object (with GAFF atom types from MOL2)
    and a loaded GAFF force field, and produces a SystemParams dictionary
    compatible with the prolix physics engine.
    
    Args:
        molecule: Parsed Molecule object with GAFF atom types assigned.
        force_field: Loaded GAFF force field (e.g., gaff-2.2.20.eqx).
        residue_name: 3-letter code for the ligand residue. If None, uses
                     molecule.residue_name (default "LIG").
        
    Returns:
        SystemParams dictionary with all bonded and non-bonded parameters.
        
    Raises:
        ValueError: If molecule has no GAFF atom types assigned.
    """
    if not molecule.atom_types or all(t == "" for t in molecule.atom_types):
        raise ValueError(
            "Molecule has no GAFF atom types. "
            "Use a MOL2 file from antechamber or assign types via gaff_typing."
        )
    
    res_name = residue_name or molecule.residue_name
    n_atoms = molecule.n_atoms
    
    # --- Non-bonded Parameters ---
    # For GAFF, we look up by atom type, not residue+atom
    charges = list(molecule.charges)  # Use charges from molecule (MOL2 or OpenFF)
    sigmas = []
    epsilons = []
    radii = []
    scales = []
    
    # Build lookup maps for GAFF force field
    # GAFF stores LJ params in atom_type -> (sigma, epsilon) format
    # We need to construct this from the force field structure
    lj_by_type = _build_gaff_lj_lookup(force_field)
    
    for atom_type in molecule.atom_types:
        if atom_type in lj_by_type:
            sig, eps = lj_by_type[atom_type]
        else:
            # Try lowercase
            if atom_type.lower() in lj_by_type:
                sig, eps = lj_by_type[atom_type.lower()]
            else:
                print(f"Warning: No LJ params for atom type '{atom_type}', using defaults")
                sig, eps = 1.7, 0.1  # Default: carbon-like
        
        sigmas.append(sig)
        epsilons.append(eps)
        radii.append(sig / 2.0)  # Approximate GBSA radius
        scales.append(0.8)  # Default OBC2 scaling factor
    
    # --- Bonded Parameters ---
    # Build bond lookup from force field
    bond_lookup = {}
    for b in force_field.bonds:
        c1, c2, length, k = b
        key = tuple(sorted((c1.lower(), c2.lower())))
        bond_lookup[key] = (length, k)
    
    bonds_list = []
    bond_params_list = []
    
    for idx1, idx2 in molecule.bonds:
        t1 = molecule.atom_types[idx1].lower()
        t2 = molecule.atom_types[idx2].lower()
        key = tuple(sorted((t1, t2)))
        
        bonds_list.append([idx1, idx2])
        
        if key in bond_lookup:
            length, k = bond_lookup[key]
            bond_params_list.append([length, k])
        else:
            # Try wildcard matching or use defaults
            print(f"Warning: No bond params for {t1}-{t2}, using defaults")
            bond_params_list.append([1.4, 300.0])
    
    # --- Angles ---
    # Build adjacency from bonds
    adj = {i: [] for i in range(n_atoms)}
    for i, j in molecule.bonds:
        adj[i].append(j)
        adj[j].append(i)
    
    angle_lookup = {}
    for a in force_field.angles:
        c1, c2, c3, theta, k = a
        angle_lookup[(c1.lower(), c2.lower(), c3.lower())] = (theta, k)
        angle_lookup[(c3.lower(), c2.lower(), c1.lower())] = (theta, k)
    
    angles_list = []
    angle_params_list = []
    seen_angles = set()
    
    for j in range(n_atoms):
        neighbors = adj[j]
        if len(neighbors) < 2:
            continue
        
        for i, k in itertools.combinations(neighbors, 2):
            # Canonicalize
            if i > k:
                i, k = k, i
            
            if (i, j, k) in seen_angles:
                continue
            seen_angles.add((i, j, k))
            
            angles_list.append([i, j, k])
            
            t1 = molecule.atom_types[i].lower()
            t2 = molecule.atom_types[j].lower()
            t3 = molecule.atom_types[k].lower()
            
            params = None
            if (t1, t2, t3) in angle_lookup:
                params = angle_lookup[(t1, t2, t3)]
            elif (t3, t2, t1) in angle_lookup:
                params = angle_lookup[(t3, t2, t1)]
            
            if params:
                theta, k_force = params
                angle_params_list.append([theta, k_force])
            else:
                # Default: 109.5 degrees
                angle_params_list.append([1.91, 50.0])
    
    # --- Scaling Matrices ---
    scale_matrix_vdw = np.ones((n_atoms, n_atoms), dtype=np.float32)
    scale_matrix_elec = np.ones((n_atoms, n_atoms), dtype=np.float32)
    
    # Self
    np.fill_diagonal(scale_matrix_vdw, 0.0)
    np.fill_diagonal(scale_matrix_elec, 0.0)
    
    # 1-2 bonds -> 0.0
    for i, j in bonds_list:
        scale_matrix_vdw[i, j] = 0.0
        scale_matrix_vdw[j, i] = 0.0
        scale_matrix_elec[i, j] = 0.0
        scale_matrix_elec[j, i] = 0.0
    
    # 1-3 angles -> 0.0
    for i, j, k in angles_list:
        scale_matrix_vdw[i, k] = 0.0
        scale_matrix_vdw[k, i] = 0.0
        scale_matrix_elec[i, k] = 0.0
        scale_matrix_elec[k, i] = 0.0
    
    # --- Dihedrals ---
    dihedrals_list = []
    dihedral_params_list = []
    pairs_14 = set()
    seen_dihedrals = set()
    
    for bond_i, (b_j, b_k) in enumerate(molecule.bonds):
        neighbors_j = [n for n in adj[b_j] if n != b_k]
        neighbors_k = [n for n in adj[b_k] if n != b_j]
        
        for i in neighbors_j:
            for l in neighbors_k:
                if i == l:
                    continue
                
                # 1-4 pair
                if i < l:
                    pairs_14.add((i, l))
                else:
                    pairs_14.add((l, i))
                
                # Dihedral
                if (l, b_k, b_j, i) in seen_dihedrals:
                    continue
                seen_dihedrals.add((i, b_j, b_k, l))
                
                t_i = molecule.atom_types[i].lower()
                t_j = molecule.atom_types[b_j].lower()
                t_k = molecule.atom_types[b_k].lower()
                t_l = molecule.atom_types[l].lower()
                
                # Match against force field propers
                matched_terms = _match_gaff_dihedral(
                    force_field.propers, t_i, t_j, t_k, t_l
                )
                
                for term in matched_terms:
                    if abs(term[2]) > 1e-6:  # Filter k=0
                        dihedrals_list.append([i, b_j, b_k, l])
                        dihedral_params_list.append(term)
    
    # 1-4 scaling
    for i, l in pairs_14:
        if scale_matrix_vdw[i, l] > 0.0:  # Not already 0 from 1-2/1-3
            scale_matrix_vdw[i, l] = 0.5
            scale_matrix_vdw[l, i] = 0.5
            scale_matrix_elec[i, l] = 1.0 / 1.2
            scale_matrix_elec[l, i] = 1.0 / 1.2
    
    # --- Impropers ---
    impropers_list = []
    improper_params_list = []
    
    for k in range(n_atoms):
        neighbors = sorted(adj[k])
        if len(neighbors) != 3:
            continue
        
        t_k = molecule.atom_types[k].lower()
        
        for perm in itertools.permutations(neighbors, 3):
            i, j, l = perm
            t_i = molecule.atom_types[i].lower()
            t_j = molecule.atom_types[j].lower()
            t_l = molecule.atom_types[l].lower()
            
            matched_terms = _match_gaff_improper(
                force_field.impropers, t_k, t_i, t_j, t_l
            )
            
            if matched_terms:
                for term in matched_terms:
                    if abs(term[2]) > 1e-6:
                        impropers_list.append([i, j, k, l])
                        improper_params_list.append(term)
                break  # Only need one permutation match
    
    # --- Assemble SystemParams ---
    exclusion_mask = scale_matrix_vdw > 0.0
    
    # Masses from elements
    element_masses = {
        "H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999,
        "S": 32.065, "P": 30.974, "F": 18.998, "Cl": 35.453,
        "Br": 79.904, "I": 126.904,
    }
    masses = [element_masses.get(e, 12.0) for e in molecule.elements]
    
    return {
        "charges": jnp.array(charges, dtype=jnp.float32),
        "masses": jnp.array(masses, dtype=jnp.float32),
        "sigmas": jnp.array(sigmas, dtype=jnp.float32),
        "epsilons": jnp.array(epsilons, dtype=jnp.float32),
        "gb_radii": jnp.array(radii, dtype=jnp.float32),
        "scaled_radii": jnp.array(radii, dtype=jnp.float32) * 0.8,  # Simple scaling
        "bonds": (
            jnp.array(bonds_list, dtype=jnp.int32)
            if bonds_list else jnp.zeros((0, 2), dtype=jnp.int32)
        ),
        "bond_params": (
            jnp.array(bond_params_list, dtype=jnp.float32)
            if bond_params_list else jnp.zeros((0, 2), dtype=jnp.float32)
        ),
        "constrained_bonds": jnp.zeros((0, 2), dtype=jnp.int32),  # No H constraints by default
        "constrained_bond_lengths": jnp.zeros((0,), dtype=jnp.float32),
        "angles": (
            jnp.array(angles_list, dtype=jnp.int32)
            if angles_list else jnp.zeros((0, 3), dtype=jnp.int32)
        ),
        "angle_params": (
            jnp.array(angle_params_list, dtype=jnp.float32)
            if angle_params_list else jnp.zeros((0, 2), dtype=jnp.float32)
        ),
        "backbone_indices": jnp.zeros((1, 4), dtype=jnp.int32),  # No backbone for ligand
        "exclusion_mask": jnp.array(exclusion_mask),
        "scale_matrix_vdw": jnp.array(scale_matrix_vdw),
        "scale_matrix_elec": jnp.array(scale_matrix_elec),
        "dihedrals": (
            jnp.array(dihedrals_list, dtype=jnp.int32)
            if dihedrals_list else jnp.zeros((0, 4), dtype=jnp.int32)
        ),
        "dihedral_params": (
            jnp.array(dihedral_params_list, dtype=jnp.float32)
            if dihedral_params_list else jnp.zeros((0, 3), dtype=jnp.float32)
        ),
        "impropers": (
            jnp.array(impropers_list, dtype=jnp.int32)
            if impropers_list else jnp.zeros((0, 4), dtype=jnp.int32)
        ),
        "improper_params": (
            jnp.array(improper_params_list, dtype=jnp.float32)
            if improper_params_list else jnp.zeros((0, 3), dtype=jnp.float32)
        ),
        # Empty CMAP (ligands don't have CMAP)
        "cmap_energy_grids": jnp.zeros((0, 24, 24), dtype=jnp.float32),
        "cmap_indices": jnp.zeros((0,), dtype=jnp.int32),
        "cmap_torsions": jnp.zeros((0, 5), dtype=jnp.int32),
        "cmap_coeffs": jnp.zeros((0, 24, 24, 4), dtype=jnp.float32),
        # Empty Urey-Bradley / Virtual Sites (not common in GAFF)
        "urey_bradley_bonds": jnp.zeros((0, 2), dtype=jnp.int32),
        "urey_bradley_params": jnp.zeros((0, 2), dtype=jnp.float32),
        "virtual_site_def": jnp.zeros((0, 4), dtype=jnp.int32),
        "virtual_site_params": jnp.zeros((0, 12), dtype=jnp.float32),
    }


def _build_gaff_lj_lookup(force_field: "FullForceField") -> dict[str, tuple[float, float]]:
    """Build atom_type -> (sigma, epsilon) lookup for GAFF force fields.
    
    GAFF force fields store LJ params differently than protein force fields.
    Since the current converter doesn't preserve GAFF type->LJ mapping,
    we use standard GAFF2 values as fallback.
    
    Values are in Angstroms (sigma) and kcal/mol (epsilon).
    """
    # Prefer force field's GAFF params if available
    if hasattr(force_field, 'gaff_nonbonded_params') and force_field.gaff_nonbonded_params is not None:
        params = force_field.gaff_nonbonded_params
        result = {}
        for atom_type, idx in params.type_to_index.items():
            result[atom_type] = (float(params.sigmas[idx]), float(params.epsilons[idx]))
        return result

    # Standard GAFF2 LJ parameters (sigma in Å, epsilon in kcal/mol)
    # These are converted from the OpenMM GAFF XML (nm->Å, kJ/mol->kcal/mol)
    NM_TO_A = 10.0
    KJ_TO_KCAL = 0.239006
    
    gaff2_lj = {
        # Carbon types
        "c": (0.3315 * NM_TO_A, 0.4134 * KJ_TO_KCAL),
        "cs": (0.3315 * NM_TO_A, 0.4134 * KJ_TO_KCAL),
        "c1": (0.3479 * NM_TO_A, 0.6678 * KJ_TO_KCAL),
        "c2": (0.3315 * NM_TO_A, 0.4134 * KJ_TO_KCAL),
        "c3": (0.3398 * NM_TO_A, 0.4510 * KJ_TO_KCAL),
        "ca": (0.3315 * NM_TO_A, 0.4134 * KJ_TO_KCAL),
        "cp": (0.3315 * NM_TO_A, 0.4134 * KJ_TO_KCAL),
        "cq": (0.3315 * NM_TO_A, 0.4134 * KJ_TO_KCAL),
        "cc": (0.3315 * NM_TO_A, 0.4134 * KJ_TO_KCAL),
        "cd": (0.3315 * NM_TO_A, 0.4134 * KJ_TO_KCAL),
        "ce": (0.3315 * NM_TO_A, 0.4134 * KJ_TO_KCAL),
        "cf": (0.3315 * NM_TO_A, 0.4134 * KJ_TO_KCAL),
        "cg": (0.3479 * NM_TO_A, 0.6678 * KJ_TO_KCAL),
        "ch": (0.3479 * NM_TO_A, 0.6678 * KJ_TO_KCAL),
        "cx": (0.3398 * NM_TO_A, 0.4510 * KJ_TO_KCAL),
        "cy": (0.3398 * NM_TO_A, 0.4510 * KJ_TO_KCAL),
        "cu": (0.3315 * NM_TO_A, 0.4134 * KJ_TO_KCAL),
        "cv": (0.3315 * NM_TO_A, 0.4134 * KJ_TO_KCAL),
        "cz": (0.3315 * NM_TO_A, 0.4134 * KJ_TO_KCAL),
        # Hydrogen types
        "h1": (0.2422 * NM_TO_A, 0.0870 * KJ_TO_KCAL),
        "h2": (0.2244 * NM_TO_A, 0.0870 * KJ_TO_KCAL),
        "h3": (0.2066 * NM_TO_A, 0.0870 * KJ_TO_KCAL),
        "h4": (0.2536 * NM_TO_A, 0.0674 * KJ_TO_KCAL),
        "h5": (0.2447 * NM_TO_A, 0.0674 * KJ_TO_KCAL),
        "ha": (0.2625 * NM_TO_A, 0.0674 * KJ_TO_KCAL),
        "hc": (0.2600 * NM_TO_A, 0.0870 * KJ_TO_KCAL),
        "hn": (0.1106 * NM_TO_A, 0.0418 * KJ_TO_KCAL),
        "ho": (0.0538 * NM_TO_A, 0.0197 * KJ_TO_KCAL),
        "hp": (0.1075 * NM_TO_A, 0.0602 * KJ_TO_KCAL),
        "hs": (0.1089 * NM_TO_A, 0.0519 * KJ_TO_KCAL),
        "hw": (0.0000, 0.0),  # TIP3P water hydrogen
        "hx": (0.1887 * NM_TO_A, 0.0870 * KJ_TO_KCAL),
        # Nitrogen types  
        "n": (0.3181 * NM_TO_A, 0.6845 * KJ_TO_KCAL),
        "n1": (0.3274 * NM_TO_A, 0.4594 * KJ_TO_KCAL),
        "n2": (0.3384 * NM_TO_A, 0.3937 * KJ_TO_KCAL),
        "n3": (0.3365 * NM_TO_A, 0.3590 * KJ_TO_KCAL),
        "n4": (0.2500 * NM_TO_A, 16.212 * KJ_TO_KCAL),
        "na": (0.3206 * NM_TO_A, 0.8544 * KJ_TO_KCAL),
        "nb": (0.3384 * NM_TO_A, 0.3937 * KJ_TO_KCAL),
        "nc": (0.3384 * NM_TO_A, 0.3937 * KJ_TO_KCAL),
        "nd": (0.3384 * NM_TO_A, 0.3937 * KJ_TO_KCAL),
        "ne": (0.3384 * NM_TO_A, 0.3937 * KJ_TO_KCAL),
        "nf": (0.3384 * NM_TO_A, 0.3937 * KJ_TO_KCAL),
        "nh": (0.3190 * NM_TO_A, 0.8996 * KJ_TO_KCAL),
        "no": (0.3159 * NM_TO_A, 0.5942 * KJ_TO_KCAL),
        # Oxygen types
        "o": (0.2960 * NM_TO_A, 0.8786 * KJ_TO_KCAL),
        "oh": (0.3066 * NM_TO_A, 0.8803 * KJ_TO_KCAL),
        "os": (0.3000 * NM_TO_A, 0.7113 * KJ_TO_KCAL),
        "ow": (0.3151 * NM_TO_A, 0.6364 * KJ_TO_KCAL),  # TIP3P water oxygen
        # Sulfur types
        "s": (0.3564 * NM_TO_A, 1.0460 * KJ_TO_KCAL),
        "s2": (0.3564 * NM_TO_A, 1.0460 * KJ_TO_KCAL),
        "s4": (0.3564 * NM_TO_A, 1.0460 * KJ_TO_KCAL),
        "s6": (0.3564 * NM_TO_A, 1.0460 * KJ_TO_KCAL),
        "sh": (0.3564 * NM_TO_A, 1.0460 * KJ_TO_KCAL),
        "ss": (0.3564 * NM_TO_A, 1.0460 * KJ_TO_KCAL),
        # Phosphorus types
        "p2": (0.3742 * NM_TO_A, 0.8368 * KJ_TO_KCAL),
        "p3": (0.3742 * NM_TO_A, 0.8368 * KJ_TO_KCAL),
        "p4": (0.3742 * NM_TO_A, 0.8368 * KJ_TO_KCAL),
        "p5": (0.3742 * NM_TO_A, 0.8368 * KJ_TO_KCAL),
        # Halogen types
        "f": (0.3034 * NM_TO_A, 0.3481 * KJ_TO_KCAL),
        "cl": (0.3466 * NM_TO_A, 1.1037 * KJ_TO_KCAL),
        "br": (0.3613 * NM_TO_A, 1.6451 * KJ_TO_KCAL),
        "i": (0.3841 * NM_TO_A, 2.0732 * KJ_TO_KCAL),
    }
    
    lj_lookup = dict(gaff2_lj)  # Start with hardcoded fallback
    
    # Try to extract from force field if available
    for (res, atom), atom_id in force_field.atom_key_to_id.items():
        atom_type = atom.lower() if atom else res.lower()
        
        if atom_id < len(force_field.sigmas_by_id):
            sig = float(force_field.sigmas_by_id[atom_id])
            eps = float(force_field.epsilons_by_id[atom_id])
            # Only override if we got non-zero values
            if sig > 0.1 and eps > 0.001:
                lj_lookup[atom_type] = (sig, eps)
    
    return lj_lookup


def _match_gaff_dihedral(
    propers: list,
    t_i: str, t_j: str, t_k: str, t_l: str
) -> list:
    """Match dihedral types against GAFF proper torsion definitions.
    
    GAFF uses wildcards ("X" or "") for generic matches.
    More specific matches (fewer wildcards) take precedence.
    """
    best_score = -1
    best_terms = []
    
    for proper in propers:
        pc = proper["classes"]
        
        # Forward match
        if _match_classes([t_i, t_j, t_k, t_l], pc):
            score = sum(1 for x in pc if x and x.lower() != "x")
            if score > best_score:
                best_score = score
                best_terms = list(proper["terms"])
            elif score == best_score:
                best_terms.extend(proper["terms"])
            continue
        
        # Reverse match
        if _match_classes([t_l, t_k, t_j, t_i], pc):
            score = sum(1 for x in pc if x and x.lower() != "x")
            if score > best_score:
                best_score = score
                best_terms = list(proper["terms"])
            elif score == best_score:
                best_terms.extend(proper["terms"])
    
    return best_terms


def _match_gaff_improper(
    impropers: list,
    t_center: str, t_1: str, t_2: str, t_3: str
) -> list:
    """Match improper types against GAFF improper definitions.
    
    GAFF impropers have the center atom in position 1 (class1).
    """
    best_score = -1
    best_terms = []
    
    for improper in impropers:
        pc = improper["classes"]
        
        # Center is class1
        if _match_classes([t_center, t_1, t_2, t_3], pc):
            score = sum(1 for x in pc if x and x.lower() != "x")
            if score > best_score:
                best_score = score
                best_terms = improper["terms"]
    
    return best_terms


def _match_classes(types: list[str], pattern: list[str]) -> bool:
    """Check if atom types match a pattern (with wildcard support)."""
    if len(types) != len(pattern):
        return False
    
    for t, p in zip(types, pattern):
        if p and p.lower() != "x" and p.lower() != t.lower():
            return False
    
    return True
