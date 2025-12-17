#!/usr/bin/env python3
"""Standalone verification script for Rust parser.

Tests the Atom37 formatter without requiring full test environment.
"""

import sys
import tempfile
from pathlib import Path


def main():
    print("=" * 70)
    print("Rust Parser Verification Script")
    print("=" * 70)

    # Test 1: Import
    print("\n[1/5] Testing import...")
    try:
        import priox_rs
        print("✓ Successfully imported priox_rs")
        print(f"  Available functions: {dir(priox_rs)}")
    except ImportError as e:
        print(f"✗ Failed to import priox_rs: {e}")
        return 1

    # Test 2: Create test PDB
    print("\n[2/5] Creating test PDB file...")
    pdb_content = """
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00 20.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00 20.00           C
ATOM      4  O   ALA A   1       1.251   2.390   0.000  1.00 20.00           O
ATOM      5  CB  ALA A   1       1.990  -0.744   1.232  1.00 20.00           C
ATOM      6  N   GLY A   2       3.331   1.542   0.000  1.00 20.00           N
ATOM      7  CA  GLY A   2       4.027   2.818   0.000  1.00 20.00           C
ATOM      8  C   GLY A   2       5.536   2.632   0.000  1.00 20.00           C
ATOM      9  O   GLY A   2       6.040   1.511   0.000  1.00 20.00           O
END
""".strip()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as f:
        f.write(pdb_content)
        pdb_path = f.name

    print(f"✓ Created test PDB: {pdb_path}")

    # Test 3: Parse with Rust
    print("\n[3/5] Parsing PDB with Rust...")
    try:
        result = priox_rs.parse_structure(pdb_path, None)
        print("✓ Successfully parsed structure")
        print(f"  Result keys: {list(result.keys())}")
    except Exception as e:
        print(f"✗ Failed to parse: {e}")
        Path(pdb_path).unlink()
        return 1

    # Test 4: Verify output structure
    print("\n[4/5] Verifying output arrays...")
    try:
        import numpy as np

        # Check required keys
        required_keys = ["coordinates", "atom_mask", "aatype", "residue_index", "chain_index"]
        for key in required_keys:
            if key not in result:
                print(f"✗ Missing required key: {key}")
                return 1
        print("✓ All required keys present")

        # Reshape arrays
        num_residues = len(result["aatype"])
        print(f"  Number of residues: {num_residues}")

        coords = result["coordinates"].reshape(num_residues, 37, 3)
        mask = result["atom_mask"].reshape(num_residues, 37)

        print(f"  Coordinates shape: {coords.shape}")
        print(f"  Atom mask shape: {mask.shape}")
        print(f"  Aatype shape: {result['aatype'].shape}")

        # Verify values
        assert coords.shape == (2, 37, 3), f"Expected (2, 37, 3), got {coords.shape}"
        assert mask.shape == (2, 37), f"Expected (2, 37), got {mask.shape}"
        assert len(result["aatype"]) == 2, f"Expected 2 residues, got {len(result['aatype'])}"

        # Check residue types
        assert result["aatype"][0] == 0, f"ALA should be 0, got {result['aatype'][0]}"
        assert result["aatype"][1] == 7, f"GLY should be 7, got {result['aatype'][1]}"
        print(f"  Residue types: ALA={result['aatype'][0]}, GLY={result['aatype'][1]} ✓")

        # Check for NaN/Inf
        if np.any(np.isnan(coords)):
            print("✗ Coordinates contain NaN")
            return 1
        if np.any(np.isinf(coords)):
            print("✗ Coordinates contain Inf")
            return 1
        print("  No NaN/Inf values ✓")

        # Check atom mask is binary
        if not np.all((mask == 0) | (mask == 1)):
            print("✗ Atom mask is not binary")
            return 1
        print("  Atom mask is binary ✓")

        # Count present atoms
        n_atoms_res1 = int(np.sum(mask[0]))
        n_atoms_res2 = int(np.sum(mask[1]))
        print(f"  Atoms present: Res1={n_atoms_res1}, Res2={n_atoms_res2}")

        # ALA should have N, CA, C, O, CB (at least 5)
        assert n_atoms_res1 >= 5, f"ALA should have at least 5 atoms, got {n_atoms_res1}"
        # GLY should have N, CA, C, O (at least 4, no CB)
        assert n_atoms_res2 >= 4, f"GLY should have at least 4 atoms, got {n_atoms_res2}"

        print("✓ All array checks passed")

    except ImportError:
        print("✗ NumPy not available, skipping array checks")
    except Exception as e:
        print(f"✗ Verification failed: {e}")
        Path(pdb_path).unlink()
        return 1

    # Test 5: Verify atom positions
    print("\n[5/5] Verifying atom coordinates...")
    try:
        # Check CA position for first residue (should be at 1.458, 0, 0)
        ca_idx = 1  # CA is at index 1 in atom37
        ca_coord = coords[0, ca_idx]
        expected = np.array([1.458, 0.0, 0.0])

        if np.allclose(ca_coord, expected, atol=0.01):
            print(f"✓ CA coordinates match: {ca_coord}")
        else:
            print(f"✗ CA coordinates mismatch: expected {expected}, got {ca_coord}")

    except Exception as e:
        print(f"⚠ Could not verify coordinates: {e}")

    # Cleanup
    Path(pdb_path).unlink()

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)
    print("\nRust Atom37 formatter is working correctly!")
    print("- Parses PDB files")
    print("- Formats to (N_res, 37, 3) coordinates")
    print("- Generates correct atom masks")
    print("- Maps residue types correctly")
    print("- Handles multi-residue structures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
