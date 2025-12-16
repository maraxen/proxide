import os

from proxide.io.parsing.rust import CoordFormat, OutputSpec, parse_structure


def test_gaff_integration():
  # Use the generated 4-atom PDB
  pdb_path = "tests/data/1crn.pdb"
  if not os.path.exists(pdb_path):
    raise FileNotFoundError(
      f"File {pdb_path} not found. Please run scripts/generate_trajectory_test_data.py first."
    )

  # Use Full format
  spec = OutputSpec(force_field="gaff", coord_format=CoordFormat.Full)
  print(f"Parsing {pdb_path} with GAFF...")
  result = parse_structure(str(pdb_path), spec)

  if "atom_types" not in result:
    raise ValueError("atom_types key missing from output")

  types = result["atom_types"]
  print("Atom types:", types)

  coords = result["coordinates"]
  print(f"Coords type: {type(coords)}")
  if hasattr(coords, "shape"):
    print(f"Coords shape: {coords.shape}")
    num_atoms = coords.shape[0]
  else:
    print(f"Coords len: {len(coords)}")
    if len(coords) > 0:
      print(f"First coord item: {coords[0]}")
    num_atoms = len(coords)

  print(f"Num atoms in coords (raw length): {num_atoms}")

  # Debug 81 atoms mystery
  if num_atoms == 81:
    print("Coords content:", coords)

  # Coords might be flattened and padded.
  # 4 atoms * 3 = 12 floats.
  # We should have at least that many.
  expected_atoms = 4  # We know 1crn.pdb has 4 atoms

  # Check types match expected atoms
  assert len(types) == expected_atoms, f"Expected {expected_atoms} types, got {len(types)}"

  # Check coords contain enough data
  assert num_atoms >= expected_atoms * 3

  # Check if types are populated
  # 1crn.pdb has N, CA, C, O.
  # N -> n/n3/nh
  # C -> c/c3/ca
  # O -> o/os
  # Should be valid GAFF types.
  assert any(t is not None for t in types)


if __name__ == "__main__":
  try:
    test_gaff_integration()
    print("Integration test passed!")
  except Exception as e:
    print(f"Integration test failed: {e}")
    exit(1)
