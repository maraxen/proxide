
import glob
import os

import priox_rs

print(f"DEBUG: priox_rs file: {priox_rs.__file__}")

assets_dir = "/home/marielle/workspace/priox/src/priox/assets"
xml_files = glob.glob(f"{assets_dir}/**/*.xml", recursive=True)

print(f"Checking {len(xml_files)} files...")

for f in sorted(xml_files):
    # Skip known irrelevant files
    if os.path.basename(f) in ["pdbNames.xml", "residues.xml", "hydrogens.xml", "glycam-hydrogens.xml"]:
        continue

    try:
        priox_rs.load_forcefield(f)
        # print(f"PASS: {f}")
    except ValueError as e:
        print(f"FAIL: {f} -> {e}")
