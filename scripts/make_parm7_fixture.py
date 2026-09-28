"""Write the synthetic parm7 fixture used by `crates/proxide-io/src/formats/parm7.rs` tests.

Two chains -- A: CYX-GLN, B: CYX -- joined only by the SG5-SG26 disulfide, one water,
one Na+. HE21/HE22 are atoms 19/20 so that the packed pair straddles the first 20a4 line
break. Re-run to regenerate:

    python3 scripts/make_parm7_fixture.py crates/proxide-io/src/formats/tests/parm7_fixture.parm7
"""

import sys

ATOMS = [  # (name, residue index, atomic number, mass)
  *[(n, 0, z, m) for n, z, m in [("N", 7, 14.01), ("CA", 6, 12.01), ("C", 6, 12.01), ("O", 8, 16.0), ("CB", 6, 12.01), ("SG", 16, 32.06)]],
  *[(n, 1, z, m) for n, z, m in [
    ("N", 7, 14.01), ("CA", 6, 12.01), ("C", 6, 12.01), ("O", 8, 16.0), ("CB", 6, 12.01), ("CG", 6, 12.01),
    ("CD", 6, 12.01), ("OE1", 8, 16.0), ("NE2", 7, 14.01), ("H", 1, 1.008), ("HA", 1, 1.008), ("HB2", 1, 1.008),
    ("HB3", 1, 1.008), ("HE21", 1, 1.008), ("HE22", 1, 1.008)]],
  *[(n, 2, z, m) for n, z, m in [("N", 7, 14.01), ("CA", 6, 12.01), ("C", 6, 12.01), ("O", 8, 16.0), ("CB", 6, 12.01), ("SG", 16, 32.06)]],
  ("O", 3, 8, 16.0), ("H1", 3, 1, 1.008), ("H2", 3, 1, 1.008),
  ("Na+", 4, 11, 22.99),
]
RESIDUES = ["CYX", "GLN", "CYX", "WAT", "Na+"]
BONDS_H = [(12, 15), (13, 16), (14, 17), (15, 18), (19, 14), (20, 14), (27, 28), (27, 29)]
BONDS = [(0, 1), (1, 2), (2, 3), (1, 4), (4, 5), (2, 6), (6, 7), (7, 8), (8, 9), (7, 10), (10, 11),
         (11, 12), (12, 13), (12, 14), (21, 22), (22, 23), (23, 24), (22, 25), (25, 26), (5, 26)]


def block(flag: str, fmt: str, values: list, per_line: int, width: int, kind: str) -> str:
  out = [f"%FLAG {flag:<74}", f"%FORMAT({fmt})"]
  cells = []
  for v in values:
    if kind == "a":
      cells.append(f"{v:<{width}}")
    elif kind == "I":
      cells.append(f"{v:>{width}d}")
    else:
      cells.append(f"{v:>{width}.8E}")
  for i in range(0, max(len(cells), 1), per_line):
    out.append("".join(cells[i:i + per_line]))
  return "\n".join(out) + "\n"


def main(path: str) -> None:
  n_atoms, n_res = len(ATOMS), len(RESIDUES)
  first = [next(i for i, a in enumerate(ATOMS) if a[1] == r) for r in range(n_res)]
  charges = [0.0] * n_atoms
  charges[30] = 18.2223  # Na+ carries +1 e
  pointers = [0] * 31
  pointers[0], pointers[11] = n_atoms, n_res
  pointers[2], pointers[3] = len(BONDS_H), len(BONDS)
  text = "%VERSION  VERSION_STAMP = V0001.000  DATE = 09/28/26  00:00:00\n"
  text += block("TITLE", "20a4", ["fixture"], 20, 4, "a")
  text += block("POINTERS", "10I8", pointers, 10, 8, "I")
  text += block("ATOM_NAME", "20a4", [a[0] for a in ATOMS], 20, 4, "a")
  text += block("CHARGE", "5E16.8", charges, 5, 16, "E")
  text += block("ATOMIC_NUMBER", "10I8", [a[2] for a in ATOMS], 10, 8, "I")
  text += block("MASS", "5E16.8", [a[3] for a in ATOMS], 5, 16, "E")
  text += block("RESIDUE_LABEL", "20a4", RESIDUES, 20, 4, "a")
  text += block("RESIDUE_POINTER", "10I8", [f + 1 for f in first], 10, 8, "I")
  text += block("BONDS_INC_HYDROGEN", "10I8", [v for a, b in BONDS_H for v in (3 * a, 3 * b, 1)], 10, 8, "I")
  text += block("BONDS_WITHOUT_HYDROGEN", "10I8", [v for a, b in BONDS for v in (3 * a, 3 * b, 1)], 10, 8, "I")
  with open(path, "w") as fh:
    fh.write(text)


if __name__ == "__main__":
  main(sys.argv[1])
