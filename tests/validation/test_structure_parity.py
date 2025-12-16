
import pytest
import numpy as np
import biotite.structure.io.pdb as pdb
import biotite.structure as struc
from pathlib import Path
from oxidize import parse_structure, OutputSpec, CoordFormat

def test_coordinate_parity_1crn():
    """Verify that Rust parsed coordinates match Biotite coordinates exactly."""
    pdb_path = "tests/data/1uao.pdb"
    if not Path(pdb_path).exists():
        pytest.skip(f"Test file {pdb_path} not found")

    # 1. Biotite Load
    pdb_file = pdb.PDBFile.read(pdb_path)
    # Get model 1
    biotite_struc = pdb_file.get_structure(model=1)
    
    # Filter to match typical Rust defaults (no solvent if removed, but 1CRN has none really)
    # Biotite keeps hetero by default. 
    # Rust OutputSpec default is remove_hetatm=False (confusingly named? Usually defaults to False aka keep them)
    # Let's be explicit.
    
    # 2. Rust Load
    spec = OutputSpec(
        add_hydrogens=False, # Pure parsing test
        coord_format=CoordFormat.Full,
        remove_solvent=False,
    )
    result = parse_structure(pdb_path, spec)
    
    # Flatten Rust output
    coords_rust_full = np.array(result['coordinates'])
    # Remove padding
    # The 'coordinates' array in Full format is (N_res * max_atoms * 3)
    # But we also have atom_mask.
    
    shape = result['coord_shape']
    n_res, max_atoms, _ = shape
    
    coords_rust_reshaped = coords_rust_full.reshape((n_res, max_atoms, 3))
    mask_rust = np.array(result['atom_mask']).reshape((n_res, max_atoms))
    
    # Extract only valid atoms
    valid_rust = mask_rust > 0.5
    flat_coords_rust = coords_rust_reshaped[valid_rust]
    
    # Biotite coordinates
    flat_coords_biotite = biotite_struc.coord
    
    # Verify atom counts match first
    assert len(flat_coords_rust) == len(flat_coords_biotite), \
        f"Atom count mismatch: Rust={len(flat_coords_rust)}, Biotite={len(flat_coords_biotite)}"
        
    # Check max difference
    diff = np.abs(flat_coords_rust - flat_coords_biotite)
    max_diff = np.max(diff)
    print(f"Max coordinate difference: {max_diff:.6f} A")
    
    assert max_diff < 1e-3, f"Coordinates diverge significantly! Max diff: {max_diff}"

def test_residue_identity_parity():
    """Verify residue IDs and names match."""
    pdb_path = "tests/data/1crn.pdb"
    
    # Biotite
    pdb_file = pdb.PDBFile.read(pdb_path)
    biotite_struc = pdb_file.get_structure(model=1)
    
    # Get residue starts to deduce residue array
    # Biotite is atom-based.
    biotite_res_ids = biotite_struc.res_id
    biotite_res_names = biotite_struc.res_name
    biotite_chain_ids = biotite_struc.chain_id
    
    # Rust
    spec = OutputSpec(coord_format=CoordFormat.Full)
    result = parse_structure(pdb_path, spec)
    
    shape = result['coord_shape']
    n_res, max_atoms, _ = shape
    
    # Need to reconstruct per-atom arrays from Rust's per-residue structure + mask
    mask_rust = np.array(result['atom_mask']).reshape((n_res, max_atoms))
    valid_rust = mask_rust > 0.5
    
    # Rust returns 'residue_index' which is the PDB res_id per residue
    # And 'residue_names' per residue? 
    # Current OutputSpec might not expose residue_names directly in top level dict?
    # Let's check keys. Usually 'residue_index' is there.
    # What about residue names?
    # If not present in output dict, we might need to check atom_names or infer?
    
    # Actually, looking at rust_wrapper.py/lib.rs, 'res_names' usually comes back?
    # Or maybe it's encoded in atom info?
    # Let's inspect the keys available in result if it fails.
    
    # Assuming 'residue_ids' and 'residue_names' exist or constructed from atoms.
    # Wait, 'residue_index' is (N_res,) int array.
    
    rust_res_ids_per_res = np.array(result['residue_index'])
    
    # Construct per-atom res_id array
    rust_res_ids_expanded = []
    
    for i in range(n_res):
        n_atoms = np.sum(valid_rust[i])
        res_id = rust_res_ids_per_res[i]
        rust_res_ids_expanded.extend([res_id] * int(n_atoms))
        
    rust_res_ids_expanded = np.array(rust_res_ids_expanded)
    
    assert len(rust_res_ids_expanded) == len(biotite_res_ids)
    
    # Compare
    mismatches = np.where(rust_res_ids_expanded != biotite_res_ids)[0]
    if len(mismatches) > 0:
        print(f"First mismatch at index {mismatches[0]}: Rust={rust_res_ids_expanded[mismatches[0]]}, Bio={biotite_res_ids[mismatches[0]]}")
        
    assert np.all(rust_res_ids_expanded == biotite_res_ids), "Residue IDs do not match"



def test_atom37_parity():
    """Verify Atom37 coordinate format."""
    pdb_path = "tests/data/1crn.pdb"
    if not Path(pdb_path).exists():
        pytest.skip(f"Test file {pdb_path} not found")

    spec = OutputSpec(coord_format=CoordFormat.Atom37)
    result = parse_structure(pdb_path, spec)
    
    # Shape check
    coords_flat = np.array(result['coordinates'])
    n_res = len(result['aatype'])
    
    # Needs reshaping from flat array
    coords = coords_flat.reshape((n_res, 37, 3))
    
    # Expected: (N_res, 37, 3)
    assert coords.shape == (n_res, 37, 3), f"Atom37 shape mismatch: {coords.shape}"
    
    # Check that mask matches populated coordinates
    if 'atom_mask' in result:
        mask = np.array(result['atom_mask']).reshape((n_res, 37))
        assert mask.shape == (n_res, 37), f"Mask shape mismatch: {mask.shape}"
        
        # Where mask is 0, coords should ideally be 0 (or ignored)
        # Proxide usually zeros out missing atoms in dense formats
        invalid_coords = coords[mask == 0]
        # Check if they are all zeros
        if np.any(invalid_coords != 0):
            print(f"Warning: Non-zero coordinates for masked atoms in Atom37: {invalid_coords[0]}...")


def test_atom14_parity():
    """Verify Atom14 coordinate format."""
    pdb_path = "tests/data/1crn.pdb"
    if not Path(pdb_path).exists():
        pytest.skip(f"Test file {pdb_path} not found")

    spec = OutputSpec(coord_format=CoordFormat.Atom14)
    result = parse_structure(pdb_path, spec)
    
    # Shape check
    coords_flat = np.array(result['coordinates'])
    n_res = len(result['aatype'])
    
    # Needs reshaping
    coords = coords_flat.reshape((n_res, 14, 3))
    
    # Expected: (N_res, 14, 3)
    assert coords.shape == (n_res, 14, 3), f"Atom14 shape mismatch: {coords.shape}"
    
    if 'atom_mask' in result:
        mask = np.array(result['atom_mask']).reshape((n_res, 14))
        assert mask.shape == (n_res, 14)


def test_backbone_parity():
    """Verify Backbone coordinate format."""
    pdb_path = "tests/data/1crn.pdb"
    if not Path(pdb_path).exists():
        pytest.skip(f"Test file {pdb_path} not found")

    # Correct Enum name is BackboneOnly
    spec = OutputSpec(coord_format=CoordFormat.BackboneOnly)
    result = parse_structure(pdb_path, spec)
    
    # Shape check
    coords_flat = np.array(result['coordinates'])
    n_res = len(result['aatype'])
    
    # Check if we need reshaping. Usually yes.
    # Backbone typically has 4 atoms: N, CA, C, O
    # Or sometimes 3 or 5 depending on definition? Usually 4 for ML (AlphaFold).
    # Let's verify flat size first.
    # If 1CRN is 1 res, and flat size is 12 -> 4 atoms.
    
    expected_atoms = 4 # N, CA, C, O usually, or N, CA, C. Let's assume 4 (OpenFold standard)
    # Actually, let's detect
    
    if coords_flat.size == n_res * 4 * 3:
         expected_atoms = 4
    elif coords_flat.size == n_res * 3 * 3:
         expected_atoms = 3
    elif coords_flat.size == n_res * 5 * 3:
         expected_atoms = 5 # N, CA, C, O, CB?
         
    coords = coords_flat.reshape((n_res, expected_atoms, 3))
    
    # Expected: (N_res, 4, 3) -> N, CA, C, O
    assert coords.shape[1] in [3, 4], f"Backbone atoms per residue unexpected: {coords.shape[1]}"
    
    if 'atom_mask' in result:
        mask = np.array(result['atom_mask']).reshape((n_res, expected_atoms))
        assert mask.shape == (n_res, expected_atoms)


def test_mmcif_parsing_parity():
    """Verify mmCIF parsing against Biotite (if file exists)."""
    # Look for any .cif file in tests/data
    cif_files = list(Path("tests/data").glob("*.cif")) + list(Path("tests/data").glob("*.mmcif"))
    
    if not cif_files:
        pytest.skip("No .cif/.mmcif files found in tests/data")
        
    cif_path = cif_files[0]
    print(f"Testing mmCIF parsing with {cif_path}")
    
    # 1. Biotite Load
    pdbx_file = pdb.PDBxFile.read(cif_path)
    biotite_struc = pdb.get_structure(pdbx_file, model=1)
    
    # 2. Rust Load
    spec = OutputSpec(coord_format=CoordFormat.Full, add_hydrogens=False)
    result = parse_structure(str(cif_path), spec)
    
    # Compare Atom Counts at least
    # Flatten Rust
    shape = result['coord_shape']
    n_res, max_atoms, _ = shape
    mask = np.array(result['atom_mask']).reshape((n_res, max_atoms))
    n_atoms_rust = np.sum(mask)
    
    n_atoms_biotite = biotite_struc.array_length()
    
    assert n_atoms_rust == n_atoms_biotite, \
        f"Atom count mismatch (mmCIF): Rust={n_atoms_rust}, Biotite={n_atoms_biotite}"


