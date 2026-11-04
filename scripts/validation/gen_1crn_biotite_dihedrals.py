"""Generate the external reference table for proxide-confind's backbone dihedrals on 1CRN.

Debt #3224: the confind parity gate's omega check never ran proxide, so nothing
compared proxide's phi/psi/omega with an independent implementation on a real
structure.  This script computes the reference with biotite (an implementation
proxide does not share code with) and writes a checked-in TSV that
``crates/proxide-confind/tests/test_1crn_dihedral_reference.rs`` reads.

Conventions written to the table (IUPAC, degrees, range (-180, 180]):

- ``phi_i   = torsion(C_{i-1}, N_i, CA_i, C_i)``
- ``psi_i   = torsion(N_i, CA_i, C_i, N_{i+1})``
- ``omega_i = torsion(CA_{i-1}, C_{i-1}, N_i, CA_i)`` -- the peptide bond *into*
  residue i, which is proxide-confind's indexing.  biotite indexes omega on the
  residue *before* the bond; the script detects that from where biotite's NaN
  falls and shifts it, refusing to write if the NaN is not where expected.

Undefined angles (chain termini) are written as ``NA``.

Run (needs biotite, the ``trajectories`` extra)::

    uv run --no-sync python scripts/validation/gen_1crn_biotite_dihedrals.py \
        --pdb crates/proxide-io/tests/data/1CRN.pdb \
        --out crates/proxide-confind/tests/data/1crn_biotite_backbone_dihedrals.tsv
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import math
from pathlib import Path

import biotite
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import numpy as np

LOG = logging.getLogger("gen_1crn_biotite_dihedrals")

# sha256 of the wwPDB download recorded in crates/proxide-io/tests/data/PROVENANCE.md.
EXPECTED_PDB_SHA256 = "42199a30a0701864a2a5cc76cd7f35cc544cd0e65fbcf63e03c166543249b811"


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _fmt(rad: float) -> str:
  if math.isnan(rad):
    return "NA"
  return f"{math.degrees(rad):.6f}"


def build_rows(pdb_path: Path) -> list[tuple[int, str, float, float, float]]:
  """Return (res_seq, res_name, phi, psi, omega) per residue, radians, proxide indexing."""
  atoms = pdb.PDBFile.read(str(pdb_path)).get_structure(model=1)
  atoms = atoms[struc.filter_amino_acids(atoms)]
  if len(set(atoms.chain_id.tolist())) != 1:
    raise SystemExit("expected exactly one chain in 1CRN")
  phi, psi, omega_bio = struc.dihedral_backbone(atoms)
  res_ids, res_names = struc.get_residues(atoms)
  n = len(res_ids)
  if not (len(phi) == len(psi) == len(omega_bio) == n):
    raise SystemExit(f"dihedral length mismatch: n_res={n} phi={len(phi)} omega={len(omega_bio)}")

  # Undefined positions pin biotite's indexing: phi[0], psi[-1] must be NaN.  For omega,
  # NaN at the LAST residue means biotite stores the i -> i+1 bond on residue i.
  if not (math.isnan(phi[0]) and math.isnan(psi[-1])):
    raise SystemExit("biotite phi/psi termini are not NaN where expected")
  nan_idx = [i for i, w in enumerate(omega_bio) if math.isnan(w)]
  if nan_idx != [n - 1]:
    raise SystemExit(f"unexpected biotite omega NaN positions {nan_idx}; refusing to guess")
  omega = np.concatenate([[np.nan], omega_bio[:-1]])
  if any(math.isnan(x) for x in phi[1:]) or any(math.isnan(x) for x in psi[:-1]):
    raise SystemExit("interior phi/psi undefined; the structure is missing backbone atoms")

  return [
    (int(res_ids[i]), str(res_names[i]), float(phi[i]), float(psi[i]), float(omega[i]))
    for i in range(n)
  ]


def main() -> None:
  logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument("--pdb", type=Path, required=True)
  ap.add_argument("--out", type=Path, required=True)
  args = ap.parse_args()

  digest = _sha256(args.pdb)
  if digest != EXPECTED_PDB_SHA256:
    raise SystemExit(f"{args.pdb} sha256 {digest} != PROVENANCE.md {EXPECTED_PDB_SHA256}")

  rows = build_rows(args.pdb)
  lines = [
    "# External reference backbone dihedrals for 1CRN (debt #3224). Do not hand-edit;",
    "# regenerate with scripts/validation/gen_1crn_biotite_dihedrals.py.",
    f"# source_pdb_sha256\t{digest}",
    f"# biotite_version\t{biotite.__version__}",
    "# convention\tIUPAC degrees; omega_i = torsion(CA_{i-1}, C_{i-1}, N_i, CA_i); NA = undefined",
    "res_seq\tres_name\tphi_deg\tpsi_deg\tomega_deg",
  ]
  lines += [f"{s}\t{name}\t{_fmt(p)}\t{_fmt(q)}\t{_fmt(w)}" for s, name, p, q, w in rows]
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text("\n".join(lines) + "\n")
  LOG.info("wrote %d residues to %s (biotite %s)", len(rows), args.out, biotite.__version__)


if __name__ == "__main__":
  main()
