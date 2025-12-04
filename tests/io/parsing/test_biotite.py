
"""Tests for Biotite parsing utilities."""

import tempfile

from priox.io.parsing import biotite
from priox.io.parsing.structures import ProcessedStructure
from priox.chem import residues as rc
from priox.io.parsing.utils import (
    atom_array_dihedrals,
)

# 1UBQ PDB content (first model, residues 1-3 to ensure dihedrals can be calculated)
# MET 1, GLN 2, ILE 3
PDB_1UBQ_FRAG = """
ATOM      1  N   MET A   1      27.340  24.430   2.614  1.00  9.67           N
ATOM      2  CA  MET A   1      26.266  25.413   2.842  1.00 10.38           C
ATOM      3  C   MET A   1      26.913  26.639   3.531  1.00  9.62           C
ATOM      4  O   MET A   1      27.886  26.463   4.263  1.00  9.62           O
ATOM      5  CB  MET A   1      25.112  24.880   3.649  1.00 13.77           C
ATOM      6  CG  MET A   1      25.353  24.860   5.134  1.00 16.29           C
ATOM      7  SD  MET A   1      23.930  23.959   5.904  1.00 17.17           S
ATOM      8  CE  MET A   1      24.447  23.984   7.620  1.00 16.11           C
ATOM      9  N   GLN A   2      26.335  27.770   3.258  1.00  9.27           N
ATOM     10  CA  GLN A   2      26.850  29.021   3.898  1.00  9.07           C
ATOM     11  C   GLN A   2      26.100  29.253   5.202  1.00  8.72           C
ATOM     12  O   GLN A   2      24.865  29.024   5.330  1.00  9.13           O
ATOM     13  CB  GLN A   2      28.317  28.703   4.172  1.00 12.96           C
ATOM     14  CG  GLN A   2      29.537  28.318   3.270  1.00 16.92           C
ATOM     15  CD  GLN A   2      30.826  28.974   3.784  1.00 18.25           C
ATOM     16  OE1 GLN A   2      31.332  28.625   4.857  1.00 17.52           O
ATOM     17  NE2 GLN A   2      31.339  29.932   3.023  1.00 17.70           N
ATOM     18  N   ILE A   3      26.832  29.774   6.179  1.00  9.54           N
ATOM     19  CA  ILE A   3      26.230  30.158   7.451  1.00 10.37           C
ATOM     20  C   ILE A   3      26.963  31.428   7.842  1.00 10.34           C
ATOM     21  O   ILE A   3      27.817  31.896   7.078  1.00 10.70           O
ATOM     22  CB  ILE A   3      26.299  29.043   8.527  1.00 11.23           C
ATOM     23  CG1 ILE A   3      25.127  28.093   8.314  1.00 12.59           C
ATOM     24  CG2 ILE A   3      27.632  28.329   8.487  1.00 12.00           C
ATOM     25  CD1 ILE A   3      23.939  28.691   7.533  1.00 14.99           C
"""

def test_atom_array_dihedrals():
    """Test the atom_array_dihedrals function."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as tmp:
        tmp.write(PDB_1UBQ_FRAG)
        filepath = tmp.name
    # Use biotite.load_structure_with_hydride to get AtomArray directly
    atom_array = biotite.load_structure_with_hydride(filepath)
    dihedrals = atom_array_dihedrals(atom_array)

    # Biotite calculates dihedrals for internal residues.
    # For a 3-residue chain, only the middle residue has both Phi and Psi.
    # The first residue lacks Phi (no previous C).
    # The last residue lacks Psi (no next N).
    # omega is usually calculated for the bond between Res(i) and Res(i+1).

    # atom_array_dihedrals implementation in utils.py filters out NaN values:
    # clean_dihedrals = dihedrals[~np.any(np.isnan(dihedrals), axis=-1)]

    # So we expect only 1 valid set of dihedrals (for residue 2).

    assert dihedrals is not None
    assert len(dihedrals) == 1
