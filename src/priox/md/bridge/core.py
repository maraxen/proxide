import itertools
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from priox.chem import residues as residue_constants
from priox.md.bridge.types import SystemParams
from priox.md.bridge.cmap import compute_bicubic_params
from priox.md.bridge.gbsa import assign_mbondi2_radii, assign_obc2_scaling_factors
from priox.md.bridge.utils import assign_masses

if TYPE_CHECKING:
  from priox.physics.force_fields import FullForceField


def parameterize_system(  # noqa: C901, PLR0912, PLR0915
  force_field: "FullForceField",
  residues: list[str],
  atom_names: list[str],
  atom_counts: list[int] | None = None,
) -> SystemParams:
  """Convert a protein structure and force field into JAX MD compatible arrays.

  Uses template matching based on residue names to define topology (bonds/angles).
  Assumes `atom_names` follows the standard ordering for each residue as defined
  in `residue_constants`.

  Args:
      force_field: The loaded FullForceField object containing parameters.
      residues: List of 3-letter residue codes (e.g., ['ALA', 'GLY']).
      atom_names: List of atom names corresponding to the flat position array.
      atom_counts: Optional list of atom counts per residue. If provided, allows
                   handling missing atoms by slicing `atom_names` explicitly.

  Returns:
      SystemParams dictionary.

  """
  n_atoms = len(atom_names)

  # Lists to collect parameters
  charges_list = []
  sigmas_list = []
  epsilons_list = []
  radii_list = []
  scales_list = []

  bonds_list = []
  bond_params_list = []
  angles_list = []
  angle_params_list = []

  urey_bradley_list = []
  urey_bradley_params_list = []

  virtual_site_def_list = []
  virtual_site_params_list = []

  backbone_indices_list = []

  # Load standard topology templates (fallback)
  std_bonds, _, _std_angles = residue_constants.load_stereo_chemical_props()

  current_atom_idx = 0
  prev_c_idx = -1  # For peptide bond

  # Map: global_idx -> (res_name, atom_name)
  atom_info_map = {}

  # Helper: Get atom class
  def get_class(idx: int) -> str:
      if idx not in atom_info_map:
          return ""
      r_name, a_name = atom_info_map[idx]
      key = f"{r_name}_{a_name}"
      return force_field.atom_class_map.get(key, "")

  # Helper: Get atom type
  def get_type(idx: int) -> str:
      if idx not in atom_info_map:
          return ""
      r_name, a_name = atom_info_map[idx]
      key = f"{r_name}_{a_name}"
      return force_field.atom_type_map.get(key, "")

  # Pre-process FF lookups
  bond_lookup = {}
  for b in force_field.bonds:
      # b is (class1, class2, length, k)  # noqa: ERA001
      c1, c2, length, k = b
      key = tuple(sorted((c1, c2)))
      bond_lookup[key] = (length, k)

  angle_lookup = {}
  for a in force_field.angles:
      # a is (class1, class2, class3, theta, k)  # noqa: ERA001
      c1, c2, c3, theta, k = a
      angle_lookup[(c1, c2, c3)] = (theta, k)
      angle_lookup[(c3, c2, c1)] = (theta, k)

  # Pre-process UB lookup
  ub_lookup = {}
  if hasattr(force_field, "urey_bradley_bonds"):
      for ub in force_field.urey_bradley_bonds:
          # ub is (class1, class2, length, k) (1-3 pair)
          c1, c2, length, k = ub
          ub_lookup[tuple(sorted((c1, c2)))] = (length, k)

  # Pre-calculate available residues in FF
  ff_residues = {r for r, a in force_field.atom_key_to_id}

  for r_i, res_name in enumerate(residues):
    if atom_counts is not None:
        count = atom_counts[r_i]
        res_atom_names = atom_names[current_atom_idx : current_atom_idx + count]
    else:
        # Legacy/Strict mode: Assume full residue
        expected_atoms = residue_constants.residue_atoms.get(res_name, [])
        res_atom_names = expected_atoms

    # --- Residue Mapping Logic ---
    # 1. Terminal Mapping
    mapped_res_name = res_name
    is_n_term = (r_i == 0)
    is_c_term = (r_i == len(residues) - 1)

    # Check for specific terminal atoms to confirm
    has_oxt = "OXT" in res_atom_names

    if is_c_term or has_oxt:
        # Generic C-terminal mapping (e.g. ALA -> CALA)
        # We assume standard Amber naming: C + ResName
        # Check if C+ResName exists in force field
        c_term_name = f"C{res_name}"
        if c_term_name in ff_residues:
             mapped_res_name = c_term_name

    elif is_n_term:
        # Generic N-terminal mapping (e.g. ALA -> NALA)
        # We assume standard Amber naming: N + ResName
        # Check if N+ResName exists in force field
        n_term_name = f"N{res_name}"
        if n_term_name in ff_residues:
            mapped_res_name = n_term_name

    # 2. Histidine Mapping (HIS -> HIE/HID/HIP)
    if res_name == "HIS":
        has_hd1 = "HD1" in res_atom_names
        has_he2 = "HE2" in res_atom_names

        if has_hd1 and has_he2:
            mapped_res_name = "HIP" # Both protonated
        elif has_hd1:
            mapped_res_name = "HID" # Delta protonated
        elif has_he2:
            mapped_res_name = "HIE" # Epsilon protonated (default neutral)
        else:
            mapped_res_name = "HIE" # Default fallback

    # Apply mapping
    mapped_res_name = res_name if res_name == mapped_res_name else mapped_res_name
    # Compute base_res_name by stripping terminal N/C prefix for topology fallback
    # e.g. NGLY -> GLY, CALA -> ALA
    if res_name.startswith('N') and len(res_name) > 1 and res_name[1:] in residue_constants.residue_atoms:
        base_res_name = res_name[1:]
    elif res_name.startswith('C') and len(res_name) > 1 and res_name[1:] in residue_constants.residue_atoms:
        base_res_name = res_name[1:]
    else:
        base_res_name = res_name
    res_name = mapped_res_name  # noqa: PLW2901
    # -----------------------------

    # Map: local_atom_name -> global_index
    local_map = {}

    # Backbone indices for this residue
    bb_indices = [-1, -1, -1, -1] # N, CA, C, O

    for i, name in enumerate(res_atom_names):
      global_idx = current_atom_idx + i
      local_map[name] = global_idx
      atom_info_map[global_idx] = (res_name, name)

      # Non-bonded params from FF
      q = force_field.get_charge(res_name, name)
      sig, eps = force_field.get_lj_params(res_name, name)
      rad, scale = force_field.get_gbsa_params(res_name, name)

      charges_list.append(q)
      sigmas_list.append(sig)
      epsilons_list.append(eps)
      radii_list.append(rad)
      scales_list.append(scale)

      if name == "N":
          bb_indices[0] = global_idx
      elif name == "CA":
          bb_indices[1] = global_idx
      elif name == "C":
          bb_indices[2] = global_idx
      elif name == "O":
          bb_indices[3] = global_idx

    backbone_indices_list.append(bb_indices)

    # -----------------------------
    # Virtual Sites
    # -----------------------------
    if hasattr(force_field, "virtual_sites") and res_name in force_field.virtual_sites:
        for vs_def in force_field.virtual_sites[res_name]:
            # vs_def: type, siteName, atoms=[n1,n2,n3], p=[..], wo=[..], wx=[..], wy=[..]
            s_name = vs_def["siteName"]
            if s_name in local_map:
                vs_idx = local_map[s_name]
                
                # Parents
                p_atoms = vs_def["atoms"]
                # Verify all parents exist in local_map (assumption: VS purely local)
                if all(pa in local_map for pa in p_atoms):
                    p_indices = [local_map[pa] for pa in p_atoms]
                    
                    # Store Def: [vs_idx, p1, p2, p3]
                    virtual_site_def_list.append([vs_idx, p_indices[0], p_indices[1], p_indices[2]])
                    
                    # Store Params: [p1,p2,p3, wo.., wx.., wy..]
                    # Flatten the arrays
                    params = vs_def["p"] + vs_def["wo"] + vs_def["wx"] + vs_def["wy"]
                    virtual_site_params_list.append(params)
                else:
                    print(f"Warning: Virtual Site parents missed for {s_name} in {res_name}")

    # Internal Bonds
    # Priority: FF Templates > residue_constants

    bonds_found = False

    # 1. Try Force Field Templates (includes Hydrogens!)
    if hasattr(force_field, "residue_templates") and res_name in force_field.residue_templates:
        template = force_field.residue_templates[res_name]
        bonds_found = True
        
        # Add Bonds from Template
        for atom1, atom2 in template:
            a1_name = atom1
            a2_name = atom2
            
            if a1_name in local_map and a2_name in local_map:
                idx1 = local_map[a1_name]
                idx2 = local_map[a2_name]
                bonds_list.append([idx1, idx2])

                # Lookup params
                c1 = get_class(idx1)
                c2 = get_class(idx2)
                key = tuple(sorted((c1, c2)))
                
                if key in bond_lookup:
                    l_a, k_kcal_a2 = bond_lookup[key]
                    length = l_a
                    k = k_kcal_a2
                    # k = (k_nm / 418.4) # REMOVED: Converted in convert_all_xmls.py
                    bond_params_list.append([length, k])
                else:
                    print(f"Warning: No bond params for {key} in {res_name}")
                    bond_params_list.append([1.33, 300.0])
    
    # 2. Fallback to residue_constants (only if no template found)
    # Use base_res_name (e.g. GLY) instead of res_name (e.g. NGLY) for lookup
    if not bonds_found and base_res_name in std_bonds:
      for bond in std_bonds[base_res_name]:
        if bond.atom1_name in local_map and bond.atom2_name in local_map:
          idx1 = local_map[bond.atom1_name]
          idx2 = local_map[bond.atom2_name]
          bonds_list.append([idx1, idx2])

          # Lookup params
          c1 = get_class(idx1)
          c2 = get_class(idx2)
          key = tuple(sorted((c1, c2)))

          if key in bond_lookup:
              l_a, k_kcal_a2 = bond_lookup[key]
              length = l_a
              k = k_kcal_a2
              # k = (k_nm / 418.4) # REMOVED: Converted in convert_all_xmls.py
              bond_params_list.append([length, k])
          else:
              bond_params_list.append([bond.length, 300.0])
      
      # 3. Infer hydrogen bonds from naming conventions (if no template)
      # Hydrogens are named with pattern: H{parent}N (e.g., HA, HB, HG, HD)
      # or numbered: H1, H2, H3 (terminal amino), HA2, HA3, HB2, HB3, etc.
      hydrogen_map = {
          # H attached to N (backbone/terminal amino)
          'H': 'N', 'HN': 'N', 'H1': 'N', 'H2': 'N', 'H3': 'N',
          # HA attached to CA
          'HA': 'CA', 'HA2': 'CA', 'HA3': 'CA', '1HA': 'CA', '2HA': 'CA',
          # HB attached to CB
          'HB': 'CB', 'HB2': 'CB', 'HB3': 'CB', '1HB': 'CB', '2HB': 'CB', '3HB': 'CB',
          # HG attached to CG
          'HG': 'CG', 'HG2': 'CG', 'HG3': 'CG', 'HG1': 'OG1', # THR special case
          'HG21': 'CG2', 'HG22': 'CG2', 'HG23': 'CG2',
          '1HG': 'CG', '2HG': 'CG', '1HG1': 'CG1', '2HG1': 'CG1',
          '1HG2': 'CG2', '2HG2': 'CG2', '3HG2': 'CG2',
          # HD attached to CD
          'HD': 'CD', 'HD2': 'CD2', 'HD3': 'CD', 'HD1': 'CD1', 'HD11': 'CD1', 'HD12': 'CD1', 'HD13': 'CD1',
          'HD21': 'CD2', 'HD22': 'CD2', 'HD23': 'ND2',
          '1HD': 'CD', '2HD': 'CD', '1HD1': 'CD1', '2HD1': 'CD1', '3HD1': 'CD1',
          '1HD2': 'CD2', '2HD2': 'CD2', '3HD2': 'CD2',
          # HE attached to CE/NE
          'HE': 'NE', 'HE1': 'NE1', 'HE2': 'NE2', 'HE3': 'CE3',
          'HE21': 'NE2', 'HE22': 'NE2',
          '1HE': 'CE', '2HE': 'CE', '3HE': 'CE',
          # HZ attached to CZ/NZ
          'HZ': 'CZ', 'HZ2': 'CZ2', 'HZ3': 'CZ3', 'HZ1': 'NZ',
          '1HZ': 'NZ', '2HZ': 'NZ', '3HZ': 'NZ',
          # HH attached to NH/OH
          'HH': 'OH', 'HH2': 'NH2', 'HH11': 'NH1', 'HH12': 'NH1', 'HH21': 'NH2', 'HH22': 'NH2',
          # Terminal OXT
          'HXT': 'OXT',
      }
      
      for h_name in res_atom_names:
          if h_name.startswith('H') and h_name in local_map:
              parent = hydrogen_map.get(h_name)
              if parent and parent in local_map:
                  h_idx = local_map[h_name]
                  parent_idx = local_map[parent]
                  bonds_list.append([h_idx, parent_idx])
                  
                  # Lookup bond params
                  c1 = get_class(h_idx)
                  c2 = get_class(parent_idx)
                  key = tuple(sorted((c1, c2)))
                  
                  if key in bond_lookup:
                      l_a, k_kcal_a2 = bond_lookup[key]
                      bond_params_list.append([l_a, k_kcal_a2])
                  else:
                      # Default H bond: 1.0 Å, k=340 kcal/mol/Å^2
                      bond_params_list.append([1.0, 340.0])

    # Internal Angles - Generate from Bonds
    # Build local adjacency for this residue's atoms
    # We need to consider that bonds might connect to previous residue (Peptide Bond)
    # But here we are iterating residues.
    # Actually, it's better to generate ALL angles after ALL bonds are found for the whole system.
    # But parameterize_system iterates residues.
    # If we do it here, we only find intra-residue angles (and maybe peptide bond
    # angles if we handle prev_c).

    # BETTER APPROACH:
    # Collect ALL bonds first (including peptide bonds), then generate ALL
    # angles/dihedrals at the end.
    # Currently, bonds are collected per residue + peptide bond.
    # So `bonds_list` grows.
    # We can move Angle generation to AFTER the residue loop.
    # -----------------------------------------------------------------------
    # Peptide Bond (Prev C -> Curr N)
    if prev_c_idx != -1 and "N" in local_map:
      curr_n_idx = local_map["N"]
      bonds_list.append([prev_c_idx, curr_n_idx])

      c1 = get_class(prev_c_idx)
      c2 = get_class(curr_n_idx)
      key = tuple(sorted((c1, c2)))

      if key in bond_lookup:
          l_a, k_kcal_a2 = bond_lookup[key]
          length = l_a
          k = k_kcal_a2
          # k = (k_nm / 418.4) # REMOVED: Converted in convert_all_xmls.py
          bond_params_list.append([length, k])
      else:
          bond_params_list.append([1.33, 300.0])

    # Update prev_c
    prev_c_idx = local_map.get("C", -1)

    current_atom_idx += len(res_atom_names)

  # -----------------------------------------------------------------------
  # Generate Angles from Bonds (After all bonds are collected)
  # -----------------------------------------------------------------------
  # Build adjacency for angles
  adj_angles = {i: [] for i in range(n_atoms)}
  for b in bonds_list:
      adj_angles[b[0]].append(b[1])
      adj_angles[b[1]].append(b[0])

  angles_list = []
  angle_params_list = []
  seen_angles = set()

  for j in range(n_atoms):
      neighbors = adj_angles[j]
      if len(neighbors) < 2:  # noqa: PLR2004
          continue

      for i, k in itertools.combinations(neighbors, 2):
          # Angle i-j-k
          # Sort i, k to avoid duplicates (i-j-k vs k-j-i)
          # Sort i, k to avoid duplicates (i-j-k vs k-j-i)
          if i > k:
              i_idx, k_idx = k, i
          else:
              i_idx, k_idx = i, k

          if (i_idx, j, k_idx) in seen_angles:
              continue
          seen_angles.add((i_idx, j, k_idx))

          angles_list.append([i_idx, j, k_idx])

          # Lookup params
          c1 = get_class(i_idx)
          c2 = get_class(j)
          c3 = get_class(k_idx)

          params = None
          if (c1, c2, c3) in angle_lookup:
              params = angle_lookup[(c1, c2, c3)]
          elif (c3, c2, c1) in angle_lookup:
              params = angle_lookup[(c3, c2, c1)]

          if params:
              theta, k_kcal_rad2 = params
              k_force = k_kcal_rad2
              # k_force = (k_kj / 4.184) # REMOVED: Converted in convert_all_xmls.py
              angle_params_list.append([theta, k_force])
          else:
              # Fallback: 109.5 degrees (1.91 rad), k=100.0 (50 kcal/mol/rad^2)
              angle_params_list.append([1.91, 100.0])

          # Validation: Urey-Bradley (1-3 interaction)
          # Check if (c1, c3) has UB term
          ub_key = tuple(sorted((c1, c3)))
          if ub_key in ub_lookup:
              ub_d, ub_k = ub_lookup[ub_key]
              urey_bradley_list.append([i_idx, k_idx]) # 1-3 pair
              urey_bradley_params_list.append([ub_d, ub_k])

  # -----------------------------------------------------------------------
  # Scaling Matrices (1-2, 1-3, 1-4)
  # -----------------------------------------------------------------------
  # Initialize with 1.0
  scale_matrix_vdw = jnp.ones((n_atoms, n_atoms), dtype=jnp.float32)
  scale_matrix_elec = jnp.ones((n_atoms, n_atoms), dtype=jnp.float32)

  # Mask self
  diag_indices = jnp.diag_indices(n_atoms)
  scale_matrix_vdw = scale_matrix_vdw.at[diag_indices].set(0.0)
  scale_matrix_elec = scale_matrix_elec.at[diag_indices].set(0.0)

  # 1-2 (Bonds) -> 0.0
  if len(bonds_list) > 0:
      b_idx = jnp.array(bonds_list, dtype=jnp.int32)
      scale_matrix_vdw = scale_matrix_vdw.at[b_idx[:, 0], b_idx[:, 1]].set(0.0)
      scale_matrix_vdw = scale_matrix_vdw.at[b_idx[:, 1], b_idx[:, 0]].set(0.0)
      scale_matrix_elec = scale_matrix_elec.at[b_idx[:, 0], b_idx[:, 1]].set(0.0)
      scale_matrix_elec = scale_matrix_elec.at[b_idx[:, 1], b_idx[:, 0]].set(0.0)

  # 1-3 (Angles) -> 0.0
  if len(angles_list) > 0:
      a_idx = jnp.array(angles_list, dtype=jnp.int32)
      scale_matrix_vdw = scale_matrix_vdw.at[a_idx[:, 0], a_idx[:, 2]].set(0.0)
      scale_matrix_vdw = scale_matrix_vdw.at[a_idx[:, 2], a_idx[:, 0]].set(0.0)
      scale_matrix_elec = scale_matrix_elec.at[a_idx[:, 0], a_idx[:, 2]].set(0.0)
      scale_matrix_elec = scale_matrix_elec.at[a_idx[:, 2], a_idx[:, 0]].set(0.0)

  # Legacy exclusion mask
  exclusion_mask = (scale_matrix_vdw > 0.0)

  # -----------------------------------------------------------------------
  # Torsions (Dihedrals) & 1-4 Scaling
  # -----------------------------------------------------------------------
  adj = {i: [] for i in range(n_atoms)}
  for b in bonds_list:
      adj[b[0]].append(b[1])
      adj[b[1]].append(b[0])

  dihedrals_list = []
  dihedral_params_list = []

  # Find Dihedrals
  seen_dihedrals = set()

  # 1-4 pairs set to avoid duplicates
  pairs_14 = set()

  for b in bonds_list:
      j, k = b[0], b[1]
      neighbors_j = [n for n in adj[j] if n != k]
      neighbors_k = [n for n in adj[k] if n != j]

      for i in neighbors_j:
          for l_idx in neighbors_k:
              # 1-4 Pair: i and l_idx
              if i < l_idx:
                  pairs_14.add((i, l_idx))
              else:
                  pairs_14.add((l_idx, i))

              # Dihedral i-j-k-l_idx
              if (l_idx, k, j, i) in seen_dihedrals:
                  continue
              seen_dihedrals.add((i, j, k, l_idx))

              c_i, c_j, c_k, c_l = get_class(i), get_class(j), get_class(k), get_class(l_idx)
              t_i, t_j, t_k, t_l = get_type(i), get_type(j), get_type(k), get_type(l_idx)

              best_match_score = -1
              best_terms = []

              for proper in force_field.propers:
                  pc = proper["classes"]
                  score = sum(1 for x in pc if x != "")

                  # Forward
                  match_fwd = True
                  for idx, (c, t) in enumerate([(c_i, t_i), (c_j, t_j), (c_k, t_k), (c_l, t_l)]):
                      if pc[idx] != "" and pc[idx] != c and pc[idx] != t:
                          match_fwd = False
                          break

                  if match_fwd:
                      if score > best_match_score:
                          best_match_score = score
                          best_terms = list(proper["terms"]) # Copy
                      elif score == best_match_score:
                          best_terms.extend(proper["terms"]) # Accumulate
                      continue

                  # Reverse
                  match_rev = True
                  for idx, (c, t) in enumerate([(c_l, t_l), (c_k, t_k), (c_j, t_j), (c_i, t_i)]):
                      if pc[idx] != "" and pc[idx] != c and pc[idx] != t:
                          match_rev = False
                          break

                  if match_rev:
                      if score > best_match_score:
                          best_match_score = score
                          best_terms = list(proper["terms"]) # Copy
                      elif score == best_match_score:
                          best_terms.extend(proper["terms"]) # Accumulate

              if best_terms:
                  if {i, j, k, l_idx} == {11, 10, 13, 14}:
                      print(f"DEBUG CORE: Torsion {i}-{j}-{k}-{l_idx} Matched {len(best_terms)} terms.")
                      print(f"  Classes: {c_i}-{c_j}-{c_k}-{c_l}")
                      print(f"  Types: {t_i}-{t_j}-{t_k}-{t_l}")
                      print(f"  Best Terms: {best_terms}")

                  for term in best_terms:
                      # Filter out k=0 terms to avoid phantom topology
                      if abs(term[2]) > 1e-6:  # noqa: PLR2004
                          dihedrals_list.append([i, j, k, l_idx])
                          dihedral_params_list.append(term)


  # Apply 1-4 Scaling
  if pairs_14:
      p14 = jnp.array(list(pairs_14), dtype=jnp.int32)
      # Check if currently 0.0 (meaning 1-2 or 1-3)
      current_vals = scale_matrix_vdw[p14[:, 0], p14[:, 1]]

      # Only update if not already 0.0
      mask_update = current_vals > 0.0

      p14_update = p14[mask_update]
      if len(p14_update) > 0:
          scale_matrix_vdw = scale_matrix_vdw.at[p14_update[:, 0], p14_update[:, 1]].set(0.5)
          scale_matrix_vdw = scale_matrix_vdw.at[p14_update[:, 1], p14_update[:, 0]].set(0.5)

          scale_matrix_elec = scale_matrix_elec.at[
              p14_update[:, 0], p14_update[:, 1],
          ].set(1.0 / 1.2)
          scale_matrix_elec = scale_matrix_elec.at[
              p14_update[:, 1], p14_update[:, 0],
          ].set(1.0 / 1.2)

  # -----------------------------------------------------------------------
  # CMAP
  # -----------------------------------------------------------------------
  cmap_torsions_list = [] # [i, j, k, l, m]
  cmap_indices_list = [] # map_index

  for k in range(n_atoms):
      neighbors_k = adj[k]
      for j in neighbors_k:
          for l_idx in neighbors_k:
              if j == l_idx:
                  continue
              if j > l_idx:
                  continue

              # Neighbors of j
              for i in adj[j]:
                  if i == k:
                      continue

                  # Neighbors of l_idx
                  for m in adj[l_idx]:
                      if m == k:
                          continue

                      # Path i-j-k-l_idx-m
                      c_i, c_j, c_k, c_l, c_m = (
                          get_class(i),
                          get_class(j),
                          get_class(k),
                          get_class(l_idx),
                          get_class(m),
                      )
                      t_i, t_j, t_k, t_l, t_m = (
                          get_type(i),
                          get_type(j),
                          get_type(k),
                          get_type(l_idx),
                          get_type(m),
                      )

                      for cmap_def in force_field.cmap_torsions:
                          def_classes = cmap_def["classes"]
                          match = True
                          # Check each atom against class OR type
                          if (
                              (def_classes[0] != c_i and def_classes[0] != t_i)
                              or (def_classes[1] != c_j and def_classes[1] != t_j)
                              or (def_classes[2] != c_k and def_classes[2] != t_k)
                              or (def_classes[3] != c_l and def_classes[3] != t_l)
                              or (def_classes[4] != c_m and def_classes[4] != t_m)
                          ):
                              match = False

                          if match:
                              cmap_torsions_list.append([i, j, k, l_idx, m])
                              cmap_indices_list.append(cmap_def["map_index"])
                              break

  # -----------------------------------------------------------------------
  # Impropers
  # -----------------------------------------------------------------------
  impropers_list = []
  improper_params_list = []

  for k in range(n_atoms):
      neighbors = sorted(adj[k])
      if len(neighbors) != 3:  # noqa: PLR2004
          continue


      c_k = get_class(k)
      t_k = get_type(k)

      best_match_score = -1
      best_terms = []
      best_perm = None

      for perm in itertools.permutations(neighbors, 3):
          i, j, l_idx = perm
          c_i, c_j, c_l = get_class(i), get_class(j), get_class(l_idx)
          t_i, t_j, t_l = get_type(i), get_type(j), get_type(l_idx)

          # Check against all impropers
          # OpenMM/Amber XML usually defines Improper as Center-N1-N2-N3
          # (class1-class2-class3-class4)  # noqa: ERA001
          # where class1 is Center.
          # Or sometimes class3 is Center (Amber standard i-j-k-l).
          # We need to support both or infer from XML.
          # Based on protein.ff19SB.xml analysis, class1 seems to be Center.
          # So we match: class1=k, class2=i, class3=j, class4=l

          for improper in force_field.impropers:
              pc = improper["classes"]

              # Match assuming class1 is Center (k)
              # And class2,3,4 are neighbors i,j,l_idx
              match = True
              if (
                  (pc[0] != "" and pc[0] != c_k and pc[0] != t_k)
                  or (pc[1] != "" and pc[1] != c_i and pc[1] != t_i)
                  or (pc[2] != "" and pc[2] != c_j and pc[2] != t_j)
                  or (pc[3] != "" and pc[3] != c_l and pc[3] != t_l)
              ):
                  match = False

              if match:
                  score = sum(1 for x in pc if x != "")
                  if score > best_match_score:
                      best_match_score = score
                      best_match_score = score
                      best_terms = improper["terms"]
                      best_perm = (i, j, k, l_idx)
                  # If scores equal, we keep the first one found? Or last?
                  # Amber precedence usually says specific overrides general.
                  # If same specificity, order in file matters.
                  # We iterate propers in order. So we should update if score >= best?
                  # But permutations loop complicates this.
                  # Let's stick to strict > for now, or >= if we want last in file.
                  # But we are inside permutation loop.
                  elif score == best_match_score:
                       # If same score, maybe this permutation is better?
                       # Or maybe it's the same definition matched by different permutation?
                       # If terms are same, doesn't matter.
                       pass

      if best_terms and best_perm:
          i_final, j_final, k_final, l_final = best_perm
          # Amber improper geometry is usually i-j-k-l (k is center).
          # So we add [i, j, k, l]
          for term in best_terms:
              # Filter out k=0 terms
              if abs(term[2]) > 1e-6:  # noqa: PLR2004
                  impropers_list.append([i_final, j_final, k_final, l_final])
                  improper_params_list.append(term)

  # Compute CMAP spline coefficients
  cmap_energy_grids = jnp.array(force_field.cmap_energy_grids)

  cmap_coeffs_list = []
  for grid in force_field.cmap_energy_grids:
      coeffs = compute_bicubic_params(np.array(grid))
      cmap_coeffs_list.append(coeffs)

  if cmap_coeffs_list:
      cmap_coeffs = jnp.array(np.stack(cmap_coeffs_list))
  else:
      # Empty case
      cmap_coeffs = jnp.zeros((0, 24, 24, 4))

  # Calculate GBSA parameters
  # Calculate GBSA parameters
  # Check if we have valid radii from FF (sum > 0)
  radii_arr = jnp.array(radii_list)
  scales_arr = jnp.array(scales_list)
  
  if jnp.sum(radii_arr) > 0.1:
      gb_radii_val = radii_arr
      # OpenMM obc2.xml defines scaled radii as: (radius - offset) * scale
      # But usually 'radius' in XML is the intrinsic radius.
      # And 'scale' is the scaling factor.
      # The offset is applied during calculation or pre-subtracted?
      # In OpenMM GBSAOBCForce:
      # B_i depends on scaled_radius = (r_i - offset) * scale_i
      # So we store intrinsic radius as gb_radii, and pre-calculate scaled_radii
      # Heuristic to detect if scales_arr contains Scale Factors (~0.8) or Scaled Radii in nm (~0.12)
      mean_scale = jnp.mean(scales_arr)
      # If mean is small (< 0.3), assume it's Scaled Radius in nm (OpenMM default output)
      if mean_scale < 0.3:
          # Convert nm -> Angstroms
          scaled_radii_val = scales_arr * 10.0
      else:
          # Assume Scale Factors (dimensionless)
          offset = 0.09
          scaled_radii_val = (gb_radii_val - offset) * scales_arr
  else:
      # Fallback to legacy mbondi2
      gb_radii_val = jnp.array(assign_mbondi2_radii(atom_names, residues, bonds_list))
      offset = 0.09 # Reverted to 0.09
      scaled_radii_val = (gb_radii_val - offset) * jnp.array(assign_obc2_scaling_factors(atom_names))

  # Convert to JAX arrays
  return {
      "charges": jnp.array(charges_list),
      "masses": jnp.array(assign_masses(atom_names)),
      "sigmas": jnp.array(sigmas_list),
      "epsilons": jnp.array(epsilons_list),
      "gb_radii": gb_radii_val,
      "scaled_radii": scaled_radii_val,
      "bonds": (
          jnp.array(bonds_list, dtype=jnp.int32)
          if bonds_list
          else jnp.zeros((0, 2), dtype=jnp.int32)
      ),
      "bond_params": (
          jnp.array(bond_params_list) if bond_params_list else jnp.zeros((0, 2))
      ),
      "constrained_bonds": jnp.array(
          [
              b
              for i, b in enumerate(bonds_list)
              if atom_names[b[0]].startswith("H") or atom_names[b[1]].startswith("H")
          ],
          dtype=jnp.int32,
      ).reshape(-1, 2),
      "constrained_bond_lengths": jnp.array(
          [
              bond_params_list[i][0]
              for i, b in enumerate(bonds_list)
              if atom_names[b[0]].startswith("H") or atom_names[b[1]].startswith("H")
          ],
          dtype=jnp.float32,
      ),
      "angles": (
          jnp.array(angles_list, dtype=jnp.int32)
          if angles_list
          else jnp.zeros((0, 3), dtype=jnp.int32)
      ),
      "angle_params": (
          jnp.array(angle_params_list) if angle_params_list else jnp.zeros((0, 2))
      ),
      "backbone_indices": jnp.array(backbone_indices_list, dtype=jnp.int32),
      "exclusion_mask": jnp.array(exclusion_mask),
      "scale_matrix_vdw": jnp.array(scale_matrix_vdw),
      "scale_matrix_elec": jnp.array(scale_matrix_elec),
      "dihedrals": (
          jnp.array(dihedrals_list, dtype=jnp.int32)
          if dihedrals_list
          else jnp.zeros((0, 4), dtype=jnp.int32)
      ),
      "dihedral_params": (
          jnp.array(dihedral_params_list) if dihedral_params_list else jnp.zeros((0, 3))
      ),
      "impropers": (
          jnp.array(impropers_list, dtype=jnp.int32)
          if impropers_list
          else jnp.zeros((0, 4), dtype=jnp.int32)
      ),
      "improper_params": (
          jnp.array(improper_params_list) if improper_params_list else jnp.zeros((0, 3))
      ),
      "cmap_energy_grids": cmap_energy_grids,
      "cmap_indices": (
          jnp.array(cmap_indices_list, dtype=jnp.int32)
          if cmap_indices_list
          else jnp.zeros((0,), dtype=jnp.int32)
      ),
      "cmap_torsions": (
          jnp.array(cmap_torsions_list, dtype=jnp.int32)
          if cmap_torsions_list
          else jnp.zeros((0, 5), dtype=jnp.int32)
      ),
      "cmap_coeffs": cmap_coeffs,
      "urey_bradley_bonds": (
          jnp.array(urey_bradley_list, dtype=jnp.int32)
          if urey_bradley_list
          else jnp.zeros((0, 2), dtype=jnp.int32)
      ),
      "urey_bradley_params": (
          jnp.array(urey_bradley_params_list, dtype=jnp.float32)
          if urey_bradley_params_list
          else jnp.zeros((0, 2), dtype=jnp.float32)
      ),
      "virtual_site_def": (
          jnp.array(virtual_site_def_list, dtype=jnp.int32)
          if virtual_site_def_list
          else jnp.zeros((0, 4), dtype=jnp.int32)
      ),
      "virtual_site_params": (
          jnp.array(virtual_site_params_list, dtype=jnp.float32)
          if virtual_site_params_list
          else jnp.zeros((0, 12), dtype=jnp.float32)
      ),
  }
