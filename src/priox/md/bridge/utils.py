def assign_masses(atom_names: list[str]) -> list[float]:
  """Assign atomic masses based on element type.

  Args:
      atom_names: List of atom names.

  Returns:
      List of masses in amu.

  """
  masses = []
  for name in atom_names:
      element = name[0]
      if element == "H":
          masses.append(1.008)
      elif element == "C":
          masses.append(12.011)
      elif element == "N":
          masses.append(14.007)
      elif element == "O":
          masses.append(15.999)
      elif element == "S":
          masses.append(32.06)
      elif element == "P":
          masses.append(30.97)
      elif element == "F":
          masses.append(18.998)
      else:
          masses.append(12.0) # Default
  return masses
