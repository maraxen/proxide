import os  # os is used for os.unlink, so it must be kept.
import tempfile

from proxide._oxidize import OutputSpec

from proxide.io.parsing import pqr as pqr_parser
from proxide.io.parsing import rust as rust_parser

PDB_CONTENT = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.500   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.500   1.500   0.000  1.00  0.00           C
ATOM      4  N   GLY B   1      10.000  10.000  10.000  1.00  0.00           N
ATOM      5  CA  GLY B   1      11.500  10.000  10.000  1.00  0.00           C
"""

# PQR with extra columns?
PQR_CONTENT = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  -0.50 1.80
"""


def debug_rust_dict():
  with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as tmp:
    tmp.write(PDB_CONTENT)
    tmp_name = tmp.name

  try:
    spec = OutputSpec()
    data = rust_parser._oxidize.parse_structure(tmp_name, spec)
    print("Rust Dict Keys:", data.keys())
    if "chain_ids" in data:
      print("chain_ids:", data["chain_ids"])
    if "chain_index" in data:
      print("chain_index:", data["chain_index"])
    if "unique_chain_ids" in data:
      print("unique_chain_ids:", data["unique_chain_ids"])
  finally:
    os.unlink(tmp_name)


def debug_pqr_dict():
  with tempfile.NamedTemporaryFile(suffix=".pqr", mode="w", delete=False) as tmp:
    tmp.write(PQR_CONTENT)
    tmp_name = tmp.name

  try:
    data = pqr_parser.parse_pqr_rust(tmp_name)
    print("PQR Dict Keys:", data.keys())
    if "epsilons" in data:
      print("epsilons:", data["epsilons"])
  finally:
    os.unlink(tmp_name)


if __name__ == "__main__":
  print("--- Debugging Rust Dict ---")
  debug_rust_dict()
  print("\n--- Debugging PQR Dict ---")
  debug_pqr_dict()
