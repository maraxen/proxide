#!/usr/bin/env python3
"""Convert hydride fragments.pickle to Protocol Buffers format.

This script converts the hydride fragment library from Python pickle format
to Protocol Buffers for safe, cross-language use in the Rust extension.

Usage:
    python convert_fragments.py [--output fragments.pb]

Requirements:
    - hydride (with fragments.pickle generated)
    - protobuf Python package
"""

import argparse
import pickle
from datetime import UTC, datetime
from pathlib import Path

# We'll use protobuf's Python bindings
# First, generate Python code from .proto:
#   protoc --python_out=. fragments.proto


def load_pickle_fragments(pickle_path: Path) -> dict:
  """Load the hydride fragments pickle file."""
  with open(pickle_path, "rb") as f:
    return pickle.load(f)


def convert_to_protobuf_dict(frag_dict: dict) -> dict:
  """Convert the pickle dictionary to a structure matching our protobuf schema.

  The pickle structure is:
      key: (element, charge, stereo, tuple(bond_types))
      value: (res_name, atom_name, heavy_coord (3,3), hydrogen_coord (k,3))
  """
  entries = []

  for key, value in frag_dict.items():
    element, charge, stereo, bond_types = key
    res_name, atom_name, heavy_coord, hydrogen_coord = value

    entry = {
      "key": {
        "element": element,
        "charge": charge,
        "stereo": stereo,
        "bond_types": list(bond_types),
      },
      "fragment": {
        "residue_name": res_name,
        "atom_name": atom_name,
        "heavy_coords": heavy_coord.flatten().tolist(),
        "hydrogen_coords": hydrogen_coord.flatten().tolist(),
        "num_hydrogens": len(hydrogen_coord),
      },
    }
    entries.append(entry)

  # Get unique elements for stats
  unique_elements = set(k[0] for k in frag_dict)

  library = {
    "version": "1.0.0",
    "source": "RCSB Chemical Component Dictionary via hydride",
    "generated_at": datetime.now(UTC).isoformat(),
    "entries": entries,
    "num_entries": len(entries),
    "num_unique_elements": len(unique_elements),
  }

  return library


def write_json_preview(library: dict, output_path: Path):
  """Write a JSON preview of the first few entries for transparency."""
  import json

  preview = {
    "version": library["version"],
    "source": library["source"],
    "generated_at": library["generated_at"],
    "num_entries": library["num_entries"],
    "num_unique_elements": library["num_unique_elements"],
    "sample_entries": library["entries"][:5],  # First 5 entries as sample
  }

  json_path = output_path.with_suffix(".json")
  with open(json_path, "w") as f:
    json.dump(preview, f, indent=2)
  print(f"JSON preview written to: {json_path}")


def write_binary_format(library: dict, output_path: Path):
  """Write the library in a simple binary format.

  Format (all little-endian):
  - Header: b'FRAG' (4 bytes)
  - Version: u32
  - Num entries: u32
  - For each entry:
      - element: 2 bytes (padded)
      - charge: i8
      - stereo: i8
      - num_bond_types: u8
      - bond_types: [u8; num_bond_types]
      - num_hydrogens: u8
      - heavy_coords: [f32; 9]
      - hydrogen_coords: [f32; num_hydrogens * 3]
  """
  import struct

  import numpy as np

  with open(output_path, "wb") as f:
    # Header
    f.write(b"FRAG")
    f.write(struct.pack("<I", 1))  # Version 1
    f.write(struct.pack("<I", library["num_entries"]))

    for entry in library["entries"]:
      key = entry["key"]
      frag = entry["fragment"]

      # Element (2 bytes, padded with null)
      elem = key["element"].encode("ascii")[:2].ljust(2, b"\x00")
      f.write(elem)

      # Charge and stereo
      f.write(struct.pack("<b", key["charge"]))
      f.write(struct.pack("<b", key["stereo"]))

      # Bond types
      bond_types = key["bond_types"]
      f.write(struct.pack("<B", len(bond_types)))
      for bt in bond_types:
        f.write(struct.pack("<B", bt))

      # Num hydrogens
      f.write(struct.pack("<B", frag["num_hydrogens"]))

      # Heavy coords (always 9 floats)
      heavy = np.array(frag["heavy_coords"], dtype=np.float32)
      f.write(heavy.tobytes())

      # Hydrogen coords
      hydrogen = np.array(frag["hydrogen_coords"], dtype=np.float32)
      f.write(hydrogen.tobytes())

  print(f"Binary fragment library written to: {output_path}")
  print(f"  Size: {output_path.stat().st_size / 1024:.1f} KB")


def main():
  parser = argparse.ArgumentParser(description="Convert hydride fragments to binary format")
  parser.add_argument(
    "--pickle",
    type=Path,
    default=Path("hydride/src/hydride/fragments.pickle"),
    help="Path to fragments.pickle",
  )
  parser.add_argument(
    "--output",
    type=Path,
    default=Path("rust_ext/data/fragments.bin"),
    help="Output path for binary fragment library",
  )
  args = parser.parse_args()

  if not args.pickle.exists():
    print(f"Error: Pickle file not found: {args.pickle}")
    print("Run 'pip install -e hydride/' first to generate fragments.pickle")
    return 1

  print(f"Loading fragments from: {args.pickle}")
  frag_dict = load_pickle_fragments(args.pickle)
  print(f"  Loaded {len(frag_dict)} fragments")

  print("Converting to protobuf-compatible structure...")
  library = convert_to_protobuf_dict(frag_dict)

  # Ensure output directory exists
  args.output.parent.mkdir(parents=True, exist_ok=True)

  # Write JSON preview for transparency
  write_json_preview(library, args.output)

  # Write binary format
  write_binary_format(library, args.output)

  return 0


if __name__ == "__main__":
  exit(main())
