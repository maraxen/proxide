import os
import sys

# Add src to path
sys.path.append(os.path.abspath("src"))

from proxide.core.containers import Protein
from proxide.io.parsing.rust import OutputSpec, parse_structure


def verify_structural_data():
  pdb_path = "tests/data/1crn.pdb"
  if not os.path.exists(pdb_path):
    print(f"Error: {pdb_path} not found.")
    sys.exit(1)

  print(f"Loading {pdb_path} with GAFF typing...")

  # Enable GAFF typing and bond inference
  spec = OutputSpec(
    force_field="gaff",
    infer_bonds=True,
  )

  protein: Protein = parse_structure(pdb_path, spec=spec)

  print("Structure loaded successfully.")

  # 1. Verify Atom Types
  if protein.atom_types is not None:
    print(f"PASS: atom_types present (first 5: {protein.atom_types[:5]})")
    assert len(protein.atom_types) == len(protein.coordinates), "atom_types length mismatch"
  else:
    print("FAIL: atom_types is None")
    sys.exit(1)

  # 2. Verify Bonds
  if protein.bonds is not None:
    print(f"PASS: bonds present (shape: {protein.bonds.shape})")
    assert protein.bonds.shape[1] == 2, "bonds shape incorrect"
  else:
    print("FAIL: bonds is None")
    sys.exit(1)

  # 3. Verify Proper Dihedrals (Renamed field)
  if protein.proper_dihedrals is not None:
    print(f"PASS: proper_dihedrals present (shape: {protein.proper_dihedrals.shape})")
    assert protein.proper_dihedrals.shape[1] == 4, "proper_dihedrals shape incorrect"
  else:
    # Note: 1crn is small, might not have many dihedrals if logic is strict, but usually should have some.
    # But wait, does 1crn (crambin) have proper dihedrals? Yes.
    print(
      "WARNING: proper_dihedrals is None (might be expected for simple inference if no ForceField XML providing params, but GAFF typing implies inference?)"
    )
    # Actually proper_dihedrals in output usually comes from `topology.proper_dihedrals`.
    # The Rust code populates it if `infer_bonds` is True.
    # Let's check logic:
    # Rust: if spec.infer_bonds { let topology = ...; if !topology.proper_dihedrals.is_empty() { dict.set_item("dihedrals", ...) } }
    # So if infer_bonds is True, we should get dihedrals.
    print("FAIL: proper_dihedrals is None but infer_bonds=True")
    sys.exit(1)

  print("\nAll verification checks passed!")


if __name__ == "__main__":
  verify_structural_data()
