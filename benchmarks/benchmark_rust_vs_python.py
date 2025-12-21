"""Benchmark script for comparing Rust vs Python (Biotite/OpenMM) performance.

This script generates synthetic structures and force fields to measure parsing speed.
"""

import gc
import logging
import tempfile
import time
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("benchmark")

# Imports
try:
  from proxide.io.parsing.rust import (
    is_rust_parser_available,
    load_forcefield_rust,
    parse_mmcif_rust,
    parse_pdb_rust,
  )

  RUST_AVAILABLE = is_rust_parser_available()
except ImportError:
  RUST_AVAILABLE = False

try:
  # from biotite.structure.io.pdb import PDBFile
  # from biotite.structure.io.pdbx import PDBxFile
  from proxide.io.parsing.biotite import load_biotite

  BIOTITE_AVAILABLE = True
except ImportError as e:
  print(f"Biotite import failed: {e}")
  BIOTITE_AVAILABLE = False

try:
  # import openmm
  from openmm import app as openmm_app

  OPENMM_AVAILABLE = True
except ImportError:
  OPENMM_AVAILABLE = False


def generate_pdb_content(num_residues: int) -> str:
  """Generate minimal PDB content."""
  lines = ["HEADER    GENERATED STRUCTURE"]
  atom_serial = 1
  for res_idx in range(1, num_residues + 1):
    # ALA atoms
    atoms = [
      ("N", "N", 0.0, 0.0, 0.0),
      ("CA", "C", 1.5, 0.0, 0.0),
      ("C", "C", 2.5, 1.0, 0.0),
      ("O", "O", 2.5, 2.2, 0.0),
      ("CB", "C", 1.5, -1.5, 0.0),
    ]

    for name, elem, x, y, z in atoms:
      # Shift residue
      bias = res_idx * 3.8
      line = (
        f"ATOM  {atom_serial:5d}  {name:<3s} ALA A{res_idx:4d}    "
        f"{x + bias:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           {elem}"
      )
      lines.append(line)
      atom_serial += 1
  lines.append("END")
  return "\n".join(lines)


def generate_mmcif_content(num_residues: int) -> str:
  """Generate minimal mmCIF content."""
  lines = [
    "data_generated",
    "loop_",
    "_atom_site.group_PDB",
    "_atom_site.id",
    "_atom_site.type_symbol",
    "_atom_site.label_atom_id",
    "_atom_site.label_comp_id",
    "_atom_site.label_asym_id",
    "_atom_site.label_seq_id",
    "_atom_site.Cartn_x",
    "_atom_site.Cartn_y",
    "_atom_site.Cartn_z",
    "_atom_site.occupancy",
    "_atom_site.B_iso_or_equiv",
    "_atom_site.auth_seq_id",
    "_atom_site.auth_asym_id",
  ]

  atom_serial = 1
  for res_idx in range(1, num_residues + 1):
    atoms = [
      ("N", "N", 0.0, 0.0, 0.0),
      ("CA", "C", 1.5, 0.0, 0.0),
      ("C", "C", 2.5, 1.0, 0.0),
      ("O", "O", 2.5, 2.2, 0.0),
      ("CB", "C", 1.5, -1.5, 0.0),
    ]

    for name, elem, x, y, z in atoms:
      bias = res_idx * 3.8
      line = (
        f"ATOM {atom_serial} {elem} {name} ALA A {res_idx} "
        f"{x + bias:.3f} {y:.3f} {z:.3f} 1.00 20.00 {res_idx} A"
      )
      lines.append(line)
      atom_serial += 1
  return "\n".join(lines)


class BenchmarkTimer:
  def __init__(self, name: str):
    self.name = name
    self.start = 0.0
    self.end = 0.0

  def __enter__(self):
    gc.disable()
    self.start = time.perf_counter()
    return self

  def __exit__(self, *args):
    self.end = time.perf_counter()
    gc.enable()
    self.duration = self.end - self.start


def run_benchmarks():
  if not RUST_AVAILABLE:
    print("Rust extension not available. Skipping benchmarks.")
    return

  # Silence Rust logs
  import os

  os.environ["RUST_LOG"] = "error"

  print(f"{'Test Case':<30} | {'Rust (s)':<10} | {'Python (s)':<10} | {'Speedup':<8}")
  print("-" * 65)

  sizes = [100, 1000, 5000]  # Number of residues

  with tempfile.TemporaryDirectory() as tmpdir:
    tmp_path = Path(tmpdir)

    # 1. PDB Parsing
    # --------------
    for size in sizes:
      pdb_content = generate_pdb_content(size)
      pdb_file = tmp_path / f"bench_{size}.pdb"
      pdb_file.write_text(pdb_content)

      # Warmup Rust
      try:
        parse_pdb_rust(str(pdb_file))
      except Exception as e:
        print(f"Rust PDB Error: {e}")
        continue

      # Benchmark Rust
      with BenchmarkTimer("Rust PDB") as t_rust:
        for _ in range(5):
          parse_pdb_rust(str(pdb_file))
      rust_time = t_rust.duration / 5

      # Benchmark Python (Biotite)
      if BIOTITE_AVAILABLE:
        # Warmup
        try:
          list(load_biotite(str(pdb_file)))
        except Exception as e:
          print(f"Python PDB Error during warmup: {e}")

        with BenchmarkTimer("Py PDB") as t_py:
          for _ in range(5):
            list(load_biotite(str(pdb_file)))
        py_time = t_py.duration / 5
      else:
        py_time = 0.0

      # Additional debug
      if py_time == 0.0 and BIOTITE_AVAILABLE:
        print(f"DEBUG: Py PDB time is 0.0. Duration: {t_py.duration}")

      speedup = py_time / rust_time if rust_time > 0 and py_time > 0 else 0.0
      print(
        f"PDB Parsing ({size} res)      | {rust_time:.5f}    | {py_time:.5f}    | {speedup:.1f}x"
      )

    # 2. mmCIF Parsing
    # ----------------
    print("-" * 65)
    for size in sizes:
      cif_content = generate_mmcif_content(size)
      cif_file = tmp_path / f"bench_{size}.cif"
      cif_file.write_text(cif_content)

      # Warmup
      try:
        parse_mmcif_rust(str(cif_file))
      except Exception as e:
        print(f"Rust CIF Error: {e}")
        continue

      # Benchmark Rust
      with BenchmarkTimer("Rust CIF") as t_rust:
        for _ in range(5):
          parse_mmcif_rust(str(cif_file))
      rust_time = t_rust.duration / 5

      # Benchmark Python (Biotite)
      if BIOTITE_AVAILABLE:
        # Warmup
        list(load_biotite(str(cif_file)))

        with BenchmarkTimer("Py CIF") as t_py:
          for _ in range(5):
            list(load_biotite(str(cif_file)))
        py_time = t_py.duration / 5
      else:
        py_time = 0.0

      speedup = py_time / rust_time if rust_time > 0 and py_time > 0 else 0.0
      print(
        f"mmCIF Parsing ({size} res)    | {rust_time:.5f}    | {py_time:.5f}    | {speedup:.1f}x"
      )

    # 3. Force Field Loading
    # ----------------------
    print("-" * 65)
    ff_xml = """<?xml version="1.0" encoding="UTF-8"?>
<ForceField>
  <AtomTypes>
"""
    for i in range(100):
      ff_xml += f'    <Type name="T{i}" class="C{i}" element="C" mass="12.01"/>\n'
    ff_xml += "  </AtomTypes>\n  <Residues>\n"
    for i in range(100):
      ff_xml += f'    <Residue name="R{i}">\n'
      ff_xml += f'      <Atom name="A{i}" type="T{i}" charge="0.0"/>\n'
      ff_xml += "    </Residue>\n"
    ff_xml += "  </Residues>\n</ForceField>"

    ff_file = tmp_path / "bench_ff.xml"
    ff_file.write_text(ff_xml)

    # Benchmark Rust
    with BenchmarkTimer("Rust FF") as t_rust:
      for _ in range(10):
        load_forcefield_rust(str(ff_file))
    rust_time = t_rust.duration / 10

    # Benchmark OpenMM (if available)
    if OPENMM_AVAILABLE:
      with BenchmarkTimer("OpenMM FF") as t_py:
        for _ in range(10):
          openmm_app.ForceField(str(ff_file))
      py_time = t_py.duration / 10
    else:
      py_time = 0.0

    speedup = py_time / rust_time if rust_time > 0 and py_time > 0 else 0.0
    print(
      f"Force Field Loading (100 types)| {rust_time:.5f}    | {py_time:.5f}    | {speedup:.1f}x"
    )


if __name__ == "__main__":
  run_benchmarks()
