import glob
import os

import oxidize

print(f"DEBUG: oxidize file: {oxidize.__file__}")

assets_dir = "/home/marielle/workspace/priox/src/priox/assets"
xml_files = glob.glob(f"{assets_dir}/**/*.xml", recursive=True)

print(f"Checking {len(xml_files)} files...")

for f in sorted(xml_files):
  # Skip known irrelevant files
  if os.path.basename(f) in [
    "pdbNames.xml",
    "residues.xml",
    "hydrogens.xml",
    "glycam-hydrogens.xml",
  ]:
    continue

  try:
    oxidize.load_forcefield(f)
    # print(f"PASS: {f}")
  except ValueError as e:
    print(f"FAIL: {f} -> {e}")
