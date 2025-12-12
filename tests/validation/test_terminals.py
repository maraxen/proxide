
import pytest
import numpy as np
import biotite.structure.io.pdb as pdb
import biotite.structure as struc
import hydride
from pathlib import Path
from priox_rs import parse_structure, OutputSpec, CoordFormat

@pytest.fixture
def peptide_pdb(tmp_path):
    """Create a valid ALA-GLY dipeptide PDB file."""
    # Approximate geometry (doesn't need to be perfect, just correct connectivity distance)
    pdb_content = """
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.000   1.500   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.500   2.500   0.000  1.00  0.00           O
ATOM      5  CB  ALA A   1       2.000  -0.800   1.200  1.00  0.00           C
ATOM      6  N   GLY A   2       3.300   1.500   0.000  1.00  0.00           N
ATOM      7  CA  GLY A   2       4.000   2.800   0.000  1.00  0.00           C
ATOM      8  C   GLY A   2       5.500   2.800   0.000  1.00  0.00           C
ATOM      9  O   GLY A   2       6.200   3.800   0.000  1.00  0.00           O
"""
    p = tmp_path / "peptide.pdb"
    p.write_text(pdb_content.strip())
    return str(p)

def test_terminal_hydrogens_counts(peptide_pdb):
    """Verify N-terminal and C-terminal hydrogen counts match hydride."""
    
    # 1. Run Hydride (Reference)
    pdb_file = pdb.PDBFile.read(peptide_pdb)
    biotite_struc = pdb_file.get_structure()[0]
    biotite_struc = biotite_struc[biotite_struc.element != "H"]
    
    # Infer bonds
    biotite_struc.bonds = struc.connect_via_residue_names(biotite_struc)
    # Check if bonds are found (distance might be off in my manual coords)
    # If manual coords are bad, bond inference might fail.
    # Let's trust connect_via_residue_names which uses templates, NOT distance.
    
    if "charge" not in biotite_struc.get_annotation_categories():
        biotite_struc.set_annotation("charge", np.zeros(biotite_struc.array_length(), dtype=int))
    
    hydride_struc, _ = hydride.add_hydrogen(biotite_struc)
    
    # Analyze Reference
    # Res 1 (ALA) -> N-term
    # Res 2 (GLY) -> C-term
    
    mask_res1_h = (hydride_struc.res_id == 1) & (hydride_struc.element == "H")
    mask_res2_h = (hydride_struc.res_id == 2) & (hydride_struc.element == "H")
    
    ref_count_1 = np.sum(mask_res1_h)
    ref_count_2 = np.sum(mask_res2_h)
    
    # Standard counts:
    # ALA 1 (N-term): 
    #   N (NH3+): +3
    #   CA: +1
    #   CB (Methyl): +3
    #   Total = 7
    # GLY 2 (C-term):
    #   N (NH): +1
    #   CA: +2
    #   C (COO-): 0 ? Or COOH (+1)? Hydride default is pH 7 -> COO- -> 0 H on C.
    #   Total = 3
    
    print(f"Ref counts: Res1={ref_count_1}, Res2={ref_count_2}")
    
    # 2. Run Rust
    spec = OutputSpec(
        add_hydrogens=True,
        coord_format=CoordFormat.Full,
        remove_solvent=True
    )
    result = parse_structure(peptide_pdb, spec)
    
    shape = result['coord_shape']
    n_res, max_atoms, _ = shape
    res_indices = np.array(result['residue_index'])
    
    # Get Rust counts
    atom_mask = np.array(result['atom_mask']).reshape(n_res, max_atoms)
    atom_names = np.array(result['atom_names']).reshape(n_res, max_atoms)
    
    # Res 1
    idx_1 = np.where(res_indices == 1)[0][0]
    names_1 = atom_names[idx_1][atom_mask[idx_1] > 0.5]
    rust_count_1 = sum(1 for n in names_1 if n.strip().startswith("H"))
    
    # Res 2
    idx_2 = np.where(res_indices == 2)[0][0]
    names_2 = atom_names[idx_2][atom_mask[idx_2] > 0.5]
    rust_count_2 = sum(1 for n in names_2 if n.strip().startswith("H"))
    
    print(f"Rust counts: Res1={rust_count_1}, Res2={rust_count_2}")
    
    # Debug info if mismatch
    if rust_count_1 != ref_count_1:
         print(f"Res 1 Mismatch. Rust names: {[n for n in names_1 if n.startswith('H')]}")
    if rust_count_2 != ref_count_2:
         print(f"Res 2 Mismatch. Rust names: {[n for n in names_2 if n.startswith('H')]}")

    # Assertions
    # Allow small difference if we decide neutral terminals are okay
    # But for parity, should match.
    assert rust_count_1 == ref_count_1, f"N-term mismatch: Rust={rust_count_1}, Ref={ref_count_1}"
    assert rust_count_2 == ref_count_2, f"C-term mismatch: Rust={rust_count_2}, Ref={ref_count_2}"

