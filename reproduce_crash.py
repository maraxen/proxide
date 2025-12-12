
import sys
import os

# Ensure the local priox package is in the path
sys.path.insert(0, os.path.abspath("."))

try:
    from priox_rs import parse_xtc
    print("Successfully imported parse_xtc")
except ImportError as e:
    print(f"Failed to import parse_xtc: {e}")
    sys.exit(1)

xtc_file = "tests/data/trajectories/test.xtc"
print(f"Attempting to parse {xtc_file}...")

try:
    # Based on test_trajectory_parity.py, it seems parse_xtc takes a filename
    # However, I should check the signature if possible. 
    # Usually it returns a dict or list of frames.
    frames = parse_xtc(xtc_file)
    print(f"Successfully parsed {len(frames)} frames/items.")
except Exception as e:
    print(f"Caught Python exception: {e}")

print("Done.")
