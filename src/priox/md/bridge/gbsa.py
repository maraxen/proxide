def assign_mbondi2_radii(  # noqa: C901, PLR0912
  atom_names: list[str],
  _residue_names: list[str],
  bonds: list[list[int]],
) -> list[float]:
  """Assign intrinsic radii for Generalized Born calculations using mbondi2 scheme.

  Reference:
      Onufriev, Bashford, Case, "Exploring native states and large-scale dynamics
      with the generalized born model", Proteins 55, 383-394 (2004).

  Rules (MBondi2):
      C: 1.70 Å
      N: 1.55 Å
      O: 1.50 Å
      S: 1.80 Å
      H (generic): 1.20 Å
      H (bound to N): 1.30 Å
      C (C1/C2/C3 > 13.0 mass): 2.20 Å (Not fully implemented, using 1.70 default for C)

  Args:
      atom_names: List of atom names.
      residue_names: List of residue names (aligned with atom blocks).
      bonds: List of [i, j] bond indices.

  Returns:
      List of radii.

  """
  n_atoms = len(atom_names)
  radii = [0.0] * n_atoms

  # Build adjacency for H-bonding check
  adj = {i: [] for i in range(n_atoms)}
  for i, j in bonds:
    adj[i].append(j)
    adj[j].append(i)

  # Heuristic to map flat atom list to residues for context if needed
  # For mbondi2, we mostly need element type and neighbors.

  for i, name in enumerate(atom_names):
    element = name[0]  # Simple element inference

    if element == "H":
      # Check if bonded to Nitrogen
      is_bound_to_nitrogen = False
      for neighbor in adj[i]:
        if atom_names[neighbor].startswith("N"):
          is_bound_to_nitrogen = True
          break

      if is_bound_to_nitrogen:
        radii[i] = 1.30
      else:
        radii[i] = 1.20

    elif element == "C":
      # Simplified C radius (1.70).
      # Full mbondi2 distinguishes C types by mass/hybridization which is hard to
      # infer here without mass.
      # Most protein C are 1.70 except maybe some sidechain terminals?
      # The reference implementation uses 1.70 as default for C.
      radii[i] = 1.70

    elif element == "N":
      radii[i] = 1.55

    elif element == "O":
      radii[i] = 1.50

    elif element == "S":
      radii[i] = 1.80

    elif element == "P":
      radii[i] = 1.85

    elif element == "F":
      radii[i] = 1.50

    elif element == "Cl":
      radii[i] = 1.70

    else:
      # Default fallback
      radii[i] = 1.50

  return radii


def assign_obc2_scaling_factors(atom_names: list[str]) -> list[float]:
  """Assign scaling factors for OBC2 GBSA calculation.

  Reference:
      Onufriev, Bashford, Case, Proteins 55, 383-394 (2004).

  Factors:
      H: 0.85
      C: 0.72
      N: 0.79
      O: 0.85
      F: 0.88
      P: 0.86
      S: 0.96
      Other: 0.80
  """
  factors = []
  for name in atom_names:
      element = name[0]
      if element == "H":
          factors.append(0.85)
      elif element == "C":
          factors.append(0.72)
      elif element == "N":
          factors.append(0.79)
      elif element == "O":
          factors.append(0.85)
      elif element == "F":
          factors.append(0.88)
      elif element == "P":
          factors.append(0.86)
      elif element == "S":
          factors.append(0.96)
      else:
          factors.append(0.80)
  return factors
