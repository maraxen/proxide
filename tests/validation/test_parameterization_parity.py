
import numpy as np
import pytest
from pathlib import Path
from priox.physics.force_fields.loader import load_force_field
from priox.md.bridge.core import parameterize_system
from priox.io.parsing.rust import parse_structure, OutputSpec, CoordFormat

def test_parameterization_parity():
    # 1. Setup paths
    # Create a synthetic PDB with 3 residues to test N-term, Internal, C-term
    # ALA - GLY - ALA
    import priox
    
    pdb_path = Path("tests/validation/parity_test_3res.pdb")
    with open(pdb_path, "w") as f:
        f.write("""ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.000   1.500   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.500   2.500   0.000  1.00  0.00           O
ATOM      5  N   GLY A   2       3.500   1.500   0.000  1.00  0.00           N
ATOM      6  CA  GLY A   2       4.000   3.000   0.000  1.00  0.00           C
ATOM      7  C   GLY A   2       5.500   3.000   0.000  1.00  0.00           C
ATOM      8  O   GLY A   2       6.000   4.000   0.000  1.00  0.00           O
ATOM      9  N   ALA A   3       6.500   2.000   0.000  1.00  0.00           N
ATOM     10  CA  ALA A   3       7.500   1.500   0.000  1.00  0.00           C
ATOM     11  C   ALA A   3       7.000   0.500  -1.000  1.00  0.00           C
ATOM     12  O   ALA A   3       7.500  -0.500  -1.000  1.00  0.00           O
END
""")


    # 2. Get Raw Structure (for legacy input)
    # We use the Rust parser itself to get correct atom names/residues as a base
    # to avoid parsing differences confounding the parameterization test.
    base_spec = OutputSpec()
    base_spec.coord_format = CoordFormat.Full
    base_data = parse_structure(str(pdb_path), spec=base_spec)
    
    res_ids = base_data.residue_index
    res_names = []
    # aatype is int array of 0-20. We need 3-letter codes.
    # We can rely on rust's "res_names" being populated in the source dict if accessible,
    # but Protein object might not expose it easily directly as attribute unless we check `base_data.to_dict()` if implemented?
    # Wait, base_data is a Protein. It has aatype.
    # Let's inspect what base_data has.
    # Actually, let's just get the raw dict from Rust directly to be safe and avoid Protein wrapper overhead for this test.
    import priox.io.parsing.rust as rw_mod
    import priox_rs
    
    # helper to get raw dict
    def get_raw_rust_data(path, spec=None):
        return priox_rs.parse_structure(path, spec)

    base_raw = get_raw_rust_data(str(pdb_path), spec=base_spec)
    
    res_indices = base_raw["residue_index"]
    aatype = base_raw["aatype"]
    atom_mask = base_raw["atom_mask"]
    atom_names_dense = base_raw["atom_names"]
    
    # 2a. Reconstruct flat lists for legacy input
    # Map aatype to 3-letter codes
    # AlphaFold order: A, R, N, D, C, Q, E, G, H, I, L, K, M, F, P, S, T, W, Y, V
    aa_map = [
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
        "UNK"
    ]
    
    residues = []
    # aatype is per-residue (N_res)
    for idx in aatype:
        if 0 <= idx < 20:
            residues.append(aa_map[idx])
        else:
            residues.append("UNK")
            
    # Flatten atom_names using mask
    atom_names = []
    flat_indices = [] # keep track for verification
    
    # atom_mask and names are flattened from (N_res, MaxAtoms) in the dict?
    # View formatter: dict "atom_names" is flat list of strings involved in the padded array?
    # FormattedFull says: atom_names: Vec<String> (N_res * max_atoms)
    # So we simply iterate mask.
    
    atom_mask_flat = np.array(atom_mask).flatten()
    # base_raw["atom_names"] is a list of strings
    
    for i, present in enumerate(atom_mask_flat):
        if present > 0.5:
            name = atom_names_dense[i]
            if name.strip(): # Should be non-empty if present
                atom_names.append(name)
            else:
                # Fallback or error?
                pass
                
    print(f"DEBUG: Found {len(atom_names)} atoms in base_raw.")
    print(f"DEBUG: Atom names: {atom_names}")

    # Calculate atom_counts per residue
    # residue_index is per-residue? No, FormattedFull "residue_index" is vector of length N_res?
    # FormattedFull doc: "pub residue_index: Vec<i32> // (N_res)"
    # Ah, FormattedFull does NOT provide per-atom residue indices.
    # It provides per-residue info.
    # We need atom counts per residue.
    # FormattedFull has "atom_mask" (N_res * MaxAtoms).
    # We can sum mask per residue.
    
    atom_mask_reshaped = np.array(atom_mask).reshape((len(residues), -1))
    atom_counts = np.sum(atom_mask_reshaped, axis=1).astype(int).tolist()
    
    # Load Force Field
    ff = load_force_field("protein.ff14SB")
    
    # 3. Run Legacy Parameterization
    print("Running Legacy Parameterization...")
    legacy_params = parameterize_system(
        force_field=ff,
        residues=residues,
        atom_names=atom_names,
        atom_counts=atom_counts,
    )
    
    print(f"DEBUG: priox location: {priox.__file__}")
    print(f"DEBUG: Legacy params keys: {legacy_params.keys()}")
    
    # 4. Run Rust Parameterization
    print("Running Rust Parameterization...")
    rust_spec = OutputSpec()
    rust_spec.coord_format = CoordFormat.Full
    rust_spec.parameterize_md = True
    
    # Resolve absolute path to FF because Rust doesn't know about python package assets
    import priox
    ff_path = Path(priox.__file__).parent / "assets" / "protein.ff14SB.xml"
    if not ff_path.exists():
        pytest.fail(f"Force field file not found at {ff_path}")
        
    rust_spec.force_field = str(ff_path)
    
    rust_data = get_raw_rust_data(str(pdb_path), spec=rust_spec)
    
    print(f"DEBUG: Rust data keys: {rust_data.keys()}")
    
    # 5. Compare Results
    
    # A. Charges
    # Legacy returns JAX arrays, Rust returns numpy arrays (via pyo3)
    legacy_charges = np.array(legacy_params["charges"])
    rust_charges = rust_data["charges"]
    
    print(f"Legacy charges shape: {legacy_charges.shape}")
    print(f"Rust charges shape: {rust_charges.shape}")
    print(f"Legacy charges: {legacy_charges}")
    print(f"Rust charges: {rust_charges}")
    
    np.testing.assert_allclose(legacy_charges, rust_charges, atol=1e-4, err_msg="Charges do not match")
    print("Charges Match!")
    
    # B. Bonds
    # Legacy bonds: list of [i, j] (or array)
    legacy_bonds = np.array(legacy_params["bonds"])
    rust_bonds = rust_data["bonds"]
    
    # Sort for comparison (undirected)
    legacy_bonds_sorted = np.sort(np.sort(legacy_bonds, axis=1), axis=0)
    rust_bonds_sorted = np.sort(np.sort(rust_bonds, axis=1), axis=0)
    
    assert legacy_bonds_sorted.shape == rust_bonds_sorted.shape, f"Bond count mismatch: {legacy_bonds_sorted.shape} vs {rust_bonds_sorted.shape}"
    np.testing.assert_array_equal(legacy_bonds_sorted, rust_bonds_sorted, err_msg="Bonds topology mismatch")
    print("Bonds Topology Matches!")
    
    # B.1 Bond Params
    # Finding corresponding params is tricky if sorted differently. 
    # But for parity, usually order is preserved if generated linearly.
    # Let's try direct comparison first, else matching
    # Note: Legacy might be sorted?
    
    # C. Angles
    legacy_angles = np.array(legacy_params["angles"]) # was angles.idx
    rust_angles = rust_data["angles"]
    
    # Sort: [i, j, k] -> j is center. i, k can swap.
    # Normalize angles: if i > k, swap i,k
    def normalize_angles(arr):
        arr = arr.copy()
        mask = arr[:, 0] > arr[:, 2]
        arr[mask, 0], arr[mask, 2] = arr[mask, 2], arr[mask, 0]
        # Then sort by rows
        # lexical sort
        order = np.lexsort((arr[:, 2], arr[:, 1], arr[:, 0]))
        return arr[order]

    legacy_angles_norm = normalize_angles(legacy_angles)
    rust_angles_norm = normalize_angles(rust_angles)
    
    assert legacy_angles_norm.shape == rust_angles_norm.shape, f"Angle count mismatch: {legacy_angles_norm.shape} vs {rust_angles_norm.shape}"
    np.testing.assert_array_equal(legacy_angles_norm, rust_angles_norm, err_msg="Angles topology mismatch")
    print("Angles Topology Matches!")
    
    # D. Dihedrals (Propers)
    # Legacy keys might vary, checking "dihedrals.idx" or "propers.idx"
    # Looking at core.py code in my memory/context: returns 'dihedrals' list in the dict?
    # core.py returns SystemParams dict. Keys: 'bonds.idx', 'angles.idx', 'dihedrals.idx', 'impropers.idx'
    
    if "dihedrals" in legacy_params:
        legacy_dihedrals = np.array(legacy_params["dihedrals"]) # was dihedrals.idx
        rust_dihedrals = rust_data["dihedrals"]

        
        # Normalize: [i, j, k, l]. If i > l, reverse whole thing.
        def normalize_dihedrals(arr):
            arr = arr.copy()
            mask = arr[:, 0] > arr[:, 3]
            arr[mask] = arr[mask][:, ::-1]
            order = np.lexsort((arr[:, 3], arr[:, 2], arr[:, 1], arr[:, 0]))
            return arr[order]

        legacy_dihedrals_norm = normalize_dihedrals(legacy_dihedrals)
        rust_dihedrals_norm = normalize_dihedrals(rust_dihedrals)
        
        if len(legacy_dihedrals) != len(rust_dihedrals):
             print(f"Legacy Count: {len(legacy_dihedrals)}, Rust Count: {len(rust_dihedrals)}")
             # Print diffs only on failure if needed, or just fail
        
        assert len(legacy_dihedrals) == len(rust_dihedrals), f"Dihedral count mismatch: {len(legacy_dihedrals)} vs {len(rust_dihedrals)}"

        assert legacy_dihedrals_norm.shape == rust_dihedrals_norm.shape, f"Dihedral count mismatch: {legacy_dihedrals_norm.shape} vs {rust_dihedrals_norm.shape}"
        np.testing.assert_array_equal(legacy_dihedrals_norm, rust_dihedrals_norm, err_msg="Dihedrals topology mismatch")
        print("Dihedrals Topology Matches!")


if __name__ == "__main__":
    test_parameterization_parity()
