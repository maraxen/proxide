
import sys
from pathlib import Path
import numpy as np
from priox.io.parsing.dispatch import load_structure
from priox.core.containers import Protein

def validate_rust_parser():
    print("Testing Rust parser...")
    
    # Use a known PDB file, e.g. 1crn if available, or create a dummy one
    # I'll check if 1crn.pdb exists in current dir or assets
    pdb_path = Path("1crn.pdb")
    if not pdb_path.exists():
        # Try to find one or fail
        print(f"Warning: {pdb_path} not found. Checking src/priox/assets or similar.")
        # Actually, I can just write a dummy PDB for testing
        with open("test.pdb", "w") as f:
            f.write("ATOM      1  N   THR A   1      17.047  14.099   3.625  1.00 13.79           N  \n")
            f.write("ATOM      2  CA  THR A   1      16.967  12.784   4.338  1.00 10.80           C  \n")
            f.write("TER\n")
        pdb_path = Path("test.pdb")

    # specific format="rust"
    try:
        stream = load_structure(str(pdb_path), file_format="rust")
        proteins = list(stream)
        
        if len(proteins) == 0:
            print("FAILED: No proteins returned")
            sys.exit(1)
            
        p = proteins[0]
        if not isinstance(p, Protein):
            print(f"FAILED: Expected Protein, got {type(p)}")
            sys.exit(1)
            
        print("Success: Loaded Protein via Rust parser")
        print(f"  Residues: {len(p.residue_index)}")
        print(f"  Coordinates shape: {p.coordinates.shape}")
        
        if p.coordinates.shape[0] != 1: # We have 1 residue
             print(f"Warning: Expected 1 residue, got {p.coordinates.shape[0]}")

    except Exception as e:
        print(f"FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    validate_rust_parser()
