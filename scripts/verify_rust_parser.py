#!/usr/bin/env python3
"""Standalone verification script for Rust parser.

Tests the Atom37 formatter without requiring full test environment.
"""

import sys
import tempfile

# Import from package
from proxide.io.parsing.rust import parse_structure


def verify_rust_parser():
  print("Verifying Rust parser...")

  # Create a simple PDB
  pdb_content = """
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N
ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00 20.00           C
ATOM      3  C   ALA A   1       2.000   0.000   0.000  1.00 20.00           C
ATOM      4  O   ALA A   1       3.000   0.000   0.000  1.00 20.00           O
ATOM      5  CB  ALA A   1       1.000   1.000   0.000  1.00 20.00           C
END
"""
  with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as tmp:
    tmp.write(pdb_content.strip())
    tmp_path = tmp.name

  try:
    # Parse
    print(f"Parsing {tmp_path}...")
    protein = parse_structure(tmp_path)

    # Verify
    print("Success! Protein object created.")
    print(f"Coordinates shape: {protein.coordinates.shape}")

    assert protein.coordinates.shape == (1, 37, 3)
    print("Verification passed.")

  finally:
    import os

    if os.path.exists(tmp_path):
      os.unlink(tmp_path)


def main():
  verify_rust_parser()


if __name__ == "__main__":
  sys.exit(main())
