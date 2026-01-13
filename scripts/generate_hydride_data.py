#!/usr/bin/env python3
"""Generate fragment library for hydrogen addition.

This script creates the fragments.pickle file used by the Rust hydrogen
addition pipeline. It uses biotite to fetch residue data from CCD.

Usage:
    uv run python scripts/generate_hydride_data.py        # Standard residues only (fast)
    uv run python scripts/generate_hydride_data.py --full # Full CCD (slow)
"""

import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np

try:
  from tqdm import tqdm
except ImportError:

  def tqdm(iterable, **kwargs):
    desc = kwargs.get("desc", "")
    items = list(iterable)
    print(f"{desc}: Processing {len(items)} items...")
    for i, item in enumerate(items):
      if i % 10 == 0:
        print(f"  {i + 1}/{len(items)}...", end="\r")
      yield item
    print(f"  Done ({len(items)} items).")


# Imports from installed packages
from biotite.structure import info
from biotite.structure.info.ccd import get_ccd
from hydride import FragmentLibrary

# Output path for the fragment library
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_PICKLE = PROJECT_ROOT / "hydride" / "src" / "hydride" / "fragments.pickle"
OUTPUT_BINARY = PROJECT_ROOT / "rust_ext" / "data" / "fragments.bin"

PROMINENT_MOLECULES = [
  "ALA",
  "ARG",
  "ASN",
  "ASP",
  "CYS",
  "GLN",
  "GLU",
  "GLY",
  "HIS",
  "ILE",
  "LEU",
  "LYS",
  "MET",
  "PHE",
  "PRO",
  "SER",
  "THR",
  "TRP",
  "TYR",
  "VAL",
  "A",
  "C",
  "G",
  "T",
  "U",
  "DA",
  "DC",
  "DG",
  "DT",
  "DU",
  "HOH",
]


def get_mol_names_in_ccd():
  """Get all molecule names from the CCD."""
  ccd = get_ccd()
  atom_category = ccd["chem_comp_atom"]
  return np.unique(atom_category["comp_id"].as_array()).tolist()


def generate(full_ccd: bool = False):
  """Generate the fragment library."""
  print(f"Generating {'FULL' if full_ccd else 'STANDARD'} fragment library...")

  std_fragment_library = FragmentLibrary()

  if full_ccd:
    print("Getting ALL CCD molecules (this will take a while)...")
    mol_names = list(PROMINENT_MOLECULES)  # Start with known good ones
    all_ccd = get_mol_names_in_ccd()
    mol_names.extend([m for m in all_ccd if m not in PROMINENT_MOLECULES])
  else:
    print("Getting STANDARD molecules only...")
    mol_names = list(PROMINENT_MOLECULES)

  count = 0
  errors = 0
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    for mol_name in tqdm(mol_names, desc="Molecules"):
      try:
        mol = info.residue(mol_name)
        std_fragment_library.add_molecule(mol)
        count += 1
      except Exception:
        errors += 1
        continue

  print(f"\nGeneration complete. Added {count} molecules ({errors} failed).")

  # Save pickle format
  OUTPUT_PICKLE.parent.mkdir(parents=True, exist_ok=True)
  with open(OUTPUT_PICKLE, "wb") as f:
    pickle.dump(std_fragment_library._frag_dict, f)
  print(f"Saved pickle to {OUTPUT_PICKLE}")
  print(f"  Size: {OUTPUT_PICKLE.stat().st_size / 1024:.1f} KB")

  # Also run conversion to binary
  print("\nConverting to binary format...")
  convert_to_binary(std_fragment_library._frag_dict)


def convert_to_binary(frag_dict: dict):
  """Convert fragment dict to binary format for Rust."""
  import struct

  OUTPUT_BINARY.parent.mkdir(parents=True, exist_ok=True)

  with open(OUTPUT_BINARY, "wb") as f:
    # Header
    f.write(b"FRAG")
    f.write(struct.pack("<I", 1))  # Version 1
    f.write(struct.pack("<I", len(frag_dict)))  # Num entries

    for key, value in frag_dict.items():
      element, charge, stereo, bond_types = key
      res_name, atom_name, heavy_coord, hydrogen_coord = value

      # Element (2 bytes, padded with null)
      elem = element.encode("ascii")[:2].ljust(2, b"\x00")
      f.write(elem)

      # Charge and stereo
      f.write(struct.pack("<b", charge))
      f.write(struct.pack("<b", stereo))

      # Bond types
      f.write(struct.pack("<B", len(bond_types)))
      f.writelines(struct.pack("<B", bt) for bt in bond_types)

      # Num hydrogens
      num_h = len(hydrogen_coord) if hydrogen_coord is not None else 0
      f.write(struct.pack("<B", num_h))

      # Heavy coords (always 9 floats)
      heavy = np.array(heavy_coord, dtype=np.float32).flatten()
      f.write(heavy.tobytes())

      # Hydrogen coords
      if num_h > 0:
        hydrogen = np.array(hydrogen_coord, dtype=np.float32).flatten()
        f.write(hydrogen.tobytes())

  print(f"Saved binary to {OUTPUT_BINARY}")
  print(f"  Size: {OUTPUT_BINARY.stat().st_size / 1024:.1f} KB")
  print(f"  Entries: {len(frag_dict)}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Generate hydride fragment library")
  parser.add_argument("--full", action="store_true", help="Generate from full CCD (slow)")
  args = parser.parse_args()

  generate(full_ccd=args.full)
